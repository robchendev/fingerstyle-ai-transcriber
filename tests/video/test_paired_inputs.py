from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import wave

import av
import numpy as np

from core import EvidenceError, sha256
from geometry import STATE_CODES
from hand_roles import RoleConfig, assign_hand_roles, load_role_observations
from paired_inputs import (
    FEATURE_LAYOUT, INPUT_REPRESENTATION, prepare_paired_inputs,
    prepare_paired_inputs_v5, structured_observations,
)
from scripts.fretboard_features import FEATURE_LAYOUT as FRETBOARD_FEATURE_LAYOUT
from scripts.fretboard_geometry import FretboardGeometry
from scripts.video_features import VELOCITY_SLICES
from tests.video import test_hand_roles as fixtures


class PairedInputTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RoleInputTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        f.video.unlink()
        with av.open(str(f.video), "w") as container:
            stream = container.add_stream("ffv1", rate=25)
            stream.width, stream.height, stream.pix_fmt = *fixtures.SIZE, "yuv420p"
            for i in range(8):
                image = np.full((fixtures.SIZE[1], fixtures.SIZE[0], 3), i * 25, np.uint8)
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                frame.pts, frame.time_base = i * 40, Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        f.shots["videoSha256"] = sha256(f.video)
        f.write("shots", f.shots)
        for name in ("annotations", "hands", "geometry"):
            document = json.loads(f.paths[name].read_text())
            document.update(videoSha256=sha256(f.video), shotsSha256=sha256(f.paths["shots"]))
            f.write(name, document)
        document = json.loads(f.paths["geometry"].read_text())
        document["annotationsSha256"] = sha256(f.paths["annotations"])
        f.write("geometry", document)
        f.hand["image_landmarks"][:, 0, :, 0] += np.arange(8, dtype=np.float32)[:, None] * .002
        f.geo["confidence"][:, 5] = 0
        f.geo["coordinates"][:, 5] = np.nan
        f.save_arrays()
        self.roles, _ = assign_hand_roles(
            f.video, f.paths["shots"], f.paths["hands"], f.paths["geometry"], f.paths["annotations"],
            f.root / "roles", config=RoleConfig(plucking_screen_side="left"),
        )
        self.audio, self.alignment = f.root / "trimmed.wav", f.root / "alignment.json"
        with wave.open(str(self.audio), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(np.zeros(16000, np.int16).tobytes())
        self.alignment.write_text(json.dumps({
            "schemaVersion": 1, "kind": "video-to-trimmed-audio-alignment",
            "videoSha256": sha256(f.video), "trimmedAudioSha256": sha256(self.audio),
            "videoStartSecondsForTrimmedAudioZero": 0., "status": "supported", "rate": [1, 1], "correlation": .9,
        }))
        self.clips = [{"label": "bounded", "startPts": 40, "endPtsExclusive": 280}]

    def build(self, **options):
        f = self.fixture
        return prepare_paired_inputs(
            f.video, f.paths["shots"], f.paths["hands"], f.paths["geometry"], f.paths["annotations"],
            self.roles, self.audio, self.alignment, f.root / "paired", self.clips,
            pair_id="synthetic", **options,
        )

    def fretboard(self, *, invalid_row=None, detector_row=0):
        f = self.fixture
        count = len(f.hand["pts"])
        keypoints = np.tile(np.array([
            [[.2, .3], [.2, .5]],
            [[.5, .3], [.5, .5]],
            [[.8, .3], [.8, .5]],
        ], np.float32), (count, 1, 1, 1))
        available = np.ones((count, 3, 2), bool)
        if invalid_row is not None:
            available[invalid_row, 1:] = False
            keypoints[invalid_row, 1:] = np.nan
        source = np.full(count, 2, np.int8)
        source[detector_row] = 1
        detector_anchor = source == 1
        arrays = {
            "pts": f.hand["pts"].copy(), "shot_id": f.hand["shot_id"].copy(),
            "keypoints": keypoints, "available": available,
            "confidence": np.linspace(.9, .83, count, dtype=np.float32),
            "source": source,
            "age_seconds": np.arange(count, dtype=np.float32) * .04,
            "flow_error": np.arange(count, dtype=np.float32) * .01,
            "detector_anchor": detector_anchor,
        }
        directory = f.root / f"fretboard-{invalid_row}-{detector_row}"
        directory.mkdir()
        array_path = directory / "fretboard.npz"
        np.savez_compressed(array_path, **arrays)
        report = {
            "schemaVersion": 1, "kind": "six-point-fretboard-observations",
            "videoSha256": sha256(f.video), "shotsSha256": sha256(f.paths["shots"]),
            "timeBase": f.shots["timeBase"], "frameCount": count,
            "shotCount": len(f.shots["shots"]),
            "sourceEncoding": {"unavailable": 0, "detector": 1, "optical_flow": 2},
            "arrays": "fretboard.npz", "arraysSha256": sha256(array_path),
        }
        report_path = directory / "fretboard.json"
        report_path.write_text(json.dumps(report))
        return report_path, arrays

    def build_v5(self, fretboard_path, **options):
        f = self.fixture
        return prepare_paired_inputs_v5(
            f.video, f.paths["shots"], f.paths["hands"], f.paths["geometry"], f.paths["annotations"],
            self.roles, fretboard_path, self.audio, self.alignment, f.root / "paired-v5", self.clips,
            pair_id="synthetic", **options,
        )

    def observations(self):
        inputs = self.fixture.load()
        _, roles = load_role_observations(inputs, self.roles)
        return inputs, roles

    def test_real_video_geometry_velocity_native_pts_without_pixel_decoding_or_targets(self):
        open_media = av.open

        def open_audio_only(path, *args, **kwargs):
            self.assertNotEqual(Path(path).resolve(), self.fixture.video.resolve())
            return open_media(path, *args, **kwargs)

        with patch.object(av, "open", side_effect=open_audio_only):
            path, report = self.build()
        self.assertEqual(report["schemaVersion"], 4)
        self.assertEqual(report["inputRepresentation"], INPUT_REPRESENTATION)
        self.assertNotIn("imageSize", report)
        self.assertEqual(report["featureDimension"], 194)
        self.assertEqual(report["featureLayout"], FEATURE_LAYOUT)
        with np.load(path.with_name("inputs.npz")) as arrays:
            self.assertEqual(set(arrays.files), {"audio_seconds", "pts", "technique_available", "segment_id", "structured", "structured_available"})
            np.testing.assert_array_equal(arrays["pts"], [40, 80, 120, 160, 200, 240])
            np.testing.assert_allclose(arrays["audio_seconds"], np.arange(1, 7) * .04)
            masks, structured = arrays["structured_available"], arrays["structured"]
            self.assertTrue(masks[:, :2, :42].all())
            self.assertFalse(masks[:, 2:].any())
            self.assertFalse(masks[:, :, 94:96].any())
            self.assertFalse(masks[..., 186:].any())
            self.assertTrue(masks[:, :2, 92:94].all())
            self.assertFalse(masks[[0, 3], :, 42:84].any())
            np.testing.assert_allclose(structured[1:3, 0, 42:84:2], 2 / 350 / .04, atol=1e-5)
            np.testing.assert_allclose(structured[:, :2, 96], 350 / np.hypot(1000, 600), atol=1e-6)
            np.testing.assert_allclose(structured[:, :2, 97], 60 / np.hypot(1000, 600), atol=1e-6)
            self.assertTrue(np.isfinite(structured).all())
            self.assertTrue((structured[~masks] == 0).all())
            self.assertNotEqual(arrays["segment_id"][2, 0], arrays["segment_id"][3, 0])
        with self.assertRaisesRegex(EvidenceError, "overwrite"):
            self.build()
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [os.environ.get("TRANSCRIPTION_PYTHON", str(root / ".venv" / "Scripts" / "python.exe") if (root / ".venv" / "Scripts" / "python.exe").is_file() else sys.executable), "-B", "-c",
             "import sys; from scripts.paired_video import load_inference_video; "
             "video=load_inference_video(sys.argv[1],sys.argv[2]); "
             "assert video.window([.04,.08,.12,.16,.20,.24])['structured'].shape == (6,4,194)",
             str(path), sha256(self.audio)],
            cwd=root, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_audio_trim_end_between_frames_keeps_native_observations(self):
        self.clips[0]["endPtsExclusive"] = 259
        path, report = self.build()
        self.assertEqual(report["clips"][0]["endPtsExclusive"], 259)
        with np.load(path.with_name("inputs.npz")) as arrays:
            np.testing.assert_array_equal(arrays["pts"], [40, 80, 120, 160, 200, 240])
        self.assertEqual(report["frameCount"], 6)

    def test_schema_five_appends_projected_fingertips_quality_and_velocity_without_changing_d194(self):
        fretboard_path, fretboard = self.fretboard()
        inputs, roles = self.observations()
        legacy, indices = structured_observations(inputs, roles, self.clips)
        with np.load(fretboard_path.with_name("fretboard.npz")) as source:
            observations = {name: source[name] for name in source.files}
        extended, extended_indices = structured_observations(inputs, roles, self.clips, fretboard=observations)
        np.testing.assert_array_equal(extended_indices, indices)
        np.testing.assert_array_equal(extended["structured"][..., :194], legacy["structured"])
        np.testing.assert_array_equal(extended["structured_available"][..., :194], legacy["structured_available"])
        np.testing.assert_array_equal(extended["segment_id"], legacy["segment_id"])
        self.assertEqual(extended["structured"].shape, (6, 4, 233))

        geometry = FretboardGeometry.from_keypoints(
            fretboard["keypoints"][1] * fixtures.SIZE, fretboard["available"][1], fixtures.SIZE,
        )
        point = inputs["hand"]["image_landmarks"][1, 0, 4, :2] * fixtures.SIZE
        coordinate = geometry.coordinate(point)
        np.testing.assert_allclose(
            extended["structured"][0, 0, 194:196],
            [coordinate.scale_position, coordinate.string_position], atol=1e-6,
        )
        self.assertTrue(extended["structured_available"][0, 0, 194:219].all())
        np.testing.assert_allclose(
            extended["structured"][0, 0, 229:233],
            [fretboard["confidence"][1], .04, .01, 0.], atol=1e-7,
        )
        self.assertTrue(extended["structured_available"][0, 0, 229:233].all())
        expected_scale_velocity = .002 / .6 / .04
        np.testing.assert_allclose(extended["structured"][1:3, 0, 219:229:2], expected_scale_velocity, atol=1e-5)
        np.testing.assert_allclose(extended["structured"][1:3, 0, 220:229:2], 0., atol=1e-5)
        self.assertTrue(extended["structured_available"][1:3, 0, 219:229].all())
        self.assertFalse(extended["structured_available"][0, :, 219:229].any())

        path, report = self.build_v5(fretboard_path)
        self.assertEqual((report["schemaVersion"], report["featureDimension"]), (5, 233))
        self.assertEqual(report["featureLayout"], FRETBOARD_FEATURE_LAYOUT)
        self.assertEqual(set(report["inputPaths"]) - set(inputs["paths"]), {
            "roles", "roleArrays", "trimmedAudio", "alignment", "fretboard", "fretboardArrays",
        })
        self.assertEqual(report["inputSha256"]["fretboard"], sha256(fretboard_path))
        self.assertEqual(report["inputSha256"]["fretboardArrays"], sha256(fretboard_path.with_name("fretboard.npz")))
        self.assertIn("no contact or pressure", report["fretboardEvidencePolicy"])
        with np.load(path.with_name("inputs.npz")) as arrays:
            np.testing.assert_array_equal(arrays["structured"][..., :194], legacy["structured"])
            np.testing.assert_array_equal(arrays["structured_available"][..., :194], legacy["structured_available"])

    def test_schema_five_masks_invalid_geometry_and_resets_velocity_on_source_change(self):
        fretboard_path, _ = self.fretboard(invalid_row=2, detector_row=3)
        path, _ = self.build_v5(fretboard_path)
        with np.load(path.with_name("inputs.npz")) as arrays:
            masks, values = arrays["structured_available"], arrays["structured"]
            self.assertFalse(masks[1, :, 194:].any())
            self.assertTrue((values[1, :, 194:] == 0).all())
            self.assertFalse(masks[2, :, 219:229].any())
            self.assertFalse(masks[3, :, 219:229].any())
            self.assertTrue(masks[4, :2, 219:229].all())

        inputs, roles = self.observations()
        inputs["hand"]["image_landmarks"][1, 0, 4, 0] = .85
        with np.load(fretboard_path.with_name("fretboard.npz")) as source:
            fretboard = {name: source[name] for name in source.files}
        extended, _ = structured_observations(inputs, roles, self.clips, fretboard=fretboard)
        self.assertTrue(extended["structured_available"][0, 0, 194:196].all())
        self.assertFalse(extended["structured_available"][0, 0, 204])
        self.assertEqual(extended["structured"][0, 0, 204], 0.)

    def test_schema_five_rejects_unbound_or_misaligned_fretboard_sources(self):
        fretboard_path, _ = self.fretboard()
        document = json.loads(fretboard_path.read_text())
        document["videoSha256"] = "0" * 64
        fretboard_path.write_text(json.dumps(document))
        with self.assertRaisesRegex(EvidenceError, "do not match"):
            self.build_v5(fretboard_path)

    def test_independent_body_masks_and_reset_boundaries(self):
        for change in ("geometry", "point", "track", "detector", "gap"):
            with self.subTest(change=change):
                inputs, roles = self.observations()
                if change == "geometry":
                    inputs["geometry"]["confidence"][2, 0] = 0
                elif change == "point":
                    inputs["hand"]["image_landmarks"][2, 0, 8] = np.nan
                elif change == "track":
                    roles["track_id"][2:, 0] += 10
                elif change == "detector":
                    inputs["hand"]["detection_source"][2:] = 2
                else:
                    inputs["hand"]["pts"][2:] += 200
                    roles["pts"][2:] += 200
                    inputs["documents"]["shots"]["shots"][-1]["endPtsExclusive"] += 200
                clips = [{"label": "all", "startPts": 0, "endPtsExclusive": inputs["documents"]["shots"]["shots"][-1]["endPtsExclusive"]}]
                arrays, _ = structured_observations(inputs, roles, clips)
                self.assertFalse(arrays["structured_available"][2, 0, 42:84].any())
                self.assertNotEqual(arrays["segment_id"][1, 0], arrays["segment_id"][2, 0])
                if change in ("point", "geometry"):
                    self.assertFalse(arrays["structured_available"][3, 0, 42:84].any())

    def test_all_twenty_one_landmarks_use_joint_zero_nut_one_frame(self):
        inputs, roles = self.observations()
        geometry = inputs["geometry"]["coordinates"]
        geometry[:, 2:4, 0] += .04
        geometry[:, 2:4, 1] += .01
        points = inputs["hand"]["image_landmarks"][..., :2]
        points[:, 0, :, 0] = .48 + np.arange(21) * .01
        points[:, 0, :, 1] = .37 + np.arange(21) * .003
        arrays, indices = structured_observations(inputs, roles, self.clips)
        values = arrays["structured"][:, 0, :42].reshape(-1, 21, 2)
        expected = points[indices, 0].copy()
        expected[..., 0] = (expected[..., 0] - .45) / .35
        expected[..., 1] = (expected[..., 1] - .4) / .1
        np.testing.assert_allclose(values, expected, atol=1e-6)
        self.assertEqual(len(np.unique(values[0], axis=0)), 21)
        np.testing.assert_allclose(arrays["structured"][:, :2, 84:86], np.tile([1., 0.], (len(indices), 2, 1)), atol=1e-6)
        np.testing.assert_allclose(arrays["structured"][:, :, 86:88], 0., atol=1e-6)
        changed = 14
        points[:, 0, changed, 0] += .035
        updated, _ = structured_observations(inputs, roles, self.clips)
        difference = updated["structured"][:, 0, :42] - arrays["structured"][:, 0, :42]
        self.assertEqual(np.flatnonzero(np.abs(difference).max(0) > 1e-6).tolist(), [changed * 2])
        np.testing.assert_allclose(difference[:, changed * 2], .1, atol=1e-6)

    def test_camera_similarity_motion_preserves_guitar_positions_and_velocities(self):
        inputs, roles = self.observations()
        original, _ = structured_observations(inputs, roles, self.clips)
        size = np.array(fixtures.SIZE, np.float64)
        for i in range(len(inputs["hand"]["pts"])):
            angle = .02 + i * .008
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            scale = .85 + .005 * i
            for points in (inputs["hand"]["image_landmarks"][i, ..., :2], inputs["geometry"]["coordinates"][i]):
                pixels = points * size
                points[:] = (((pixels - size / 2) @ rotation.T) * scale + size / 2 + [10 + i, -5]) / size
        moved, indices = structured_observations(inputs, roles, self.clips)
        np.testing.assert_array_equal(moved["structured_available"], original["structured_available"])
        np.testing.assert_allclose(moved["structured"][:, :, :96], original["structured"][:, :, :96], atol=2e-5)
        expected_scale = original["structured"][:, :, 96:98] * (.85 + .005 * indices)[:, None, None]
        np.testing.assert_allclose(moved["structured"][:, :, 96:98], expected_scale, atol=1e-6)

    def test_missing_precise_geometry_is_unavailable_not_replaced_with_pixels(self):
        f = self.fixture
        f.geo["coordinates"][:] = np.nan
        f.geo["confidence"][:] = 0
        f.save_arrays()
        self.roles, _ = assign_hand_roles(
            f.video, f.paths["shots"], f.paths["hands"], f.paths["geometry"], f.paths["annotations"],
            f.root / "missing-geometry-roles", config=RoleConfig(plucking_screen_side="left"),
        )
        path, _ = self.build()
        with np.load(path.with_name("inputs.npz")) as arrays:
            self.assertNotIn("images", arrays.files)
            self.assertFalse(arrays["structured_available"][..., :98].any())
            self.assertTrue(arrays["structured_available"][:, :2, 98:140].all())
            self.assertFalse(arrays["structured_available"][:, 2:].any())
            self.assertTrue((arrays["segment_id"][:, :2] >= 0).all())

    def test_unknown_roles_and_minus_one_tracks_retain_independent_hands(self):
        f = self.fixture
        f.geo["coordinates"][:] = np.nan
        f.geo["confidence"][:] = 0
        f.geo["state"][:] = STATE_CODES["calibration_unstable"]
        f.save_arrays()
        self.roles, _ = assign_hand_roles(
            f.video, f.paths["shots"], f.paths["hands"], f.paths["geometry"], f.paths["annotations"],
            f.root / "unknown-roles",
        )
        inputs, roles = self.observations()
        self.assertFalse(roles["role"].any())
        self.assertTrue((roles["track_id"] == -1).all())
        path, report = self.build()
        with np.load(path.with_name("inputs.npz")) as arrays:
            self.assertFalse(arrays["structured_available"][..., :98].any())
            self.assertFalse(arrays["structured_available"][:, :2].any())
            self.assertTrue(arrays["structured_available"][:, 2:, 98:140].all())
            self.assertTrue(arrays["structured_available"][1:3, 2:, 140:184].all())
            self.assertGreater(abs(arrays["structured"][1, 3, 182]), 0)
        self.assertEqual(report["coverage"]["framesWithGeometry"], 0)
        self.assertEqual(report["coverage"]["independentHandObservations"], 12)
        self.assertEqual(report["coverage"]["unassignedHandObservations"], 12)

    def test_source_mutation_pts_validation_and_cli_contract(self):
        f = self.fixture
        f.geo["pts"][1] += 1
        f.save_arrays()
        with self.assertRaisesRegex(EvidenceError, "match exactly"):
            self.build()
        import cli

        argv = ["prepare-paired-inputs", "--video", "v", "--shots", "s", "--hands", "h",
                "--geometry", "g", "--annotations", "a", "--roles", "r", "--audio", "au",
                "--alignment", "al", "--output-directory", "out", "--pair-id", "p",
                "--clip", "0", "120", "--clip", "160", "280"]
        with patch.object(cli, "prepare_paired_inputs", return_value=(Path("inputs.json"), {"frameCount": 6, "featureDimension": 194})) as build, patch("builtins.print"):
            self.assertEqual(cli.main(argv), 0)
        self.assertNotIn("image_size", build.call_args.kwargs)
        self.assertEqual(len(build.call_args.args[9]), 2)
        with patch("sys.stderr"), self.assertRaises(SystemExit) as raised:
            cli.main([*argv, "--image-size", "96"])
        self.assertEqual(raised.exception.code, 2)

    def test_source_mutation_during_feature_extraction_removes_incomplete_bundle(self):
        import paired_inputs

        extract = paired_inputs.structured_observations

        def changed(*args):
            result = extract(*args)
            self.alignment.write_text(self.alignment.read_text() + " ")
            return result

        with patch.object(paired_inputs, "structured_observations", side_effect=changed):
            with self.assertRaisesRegex(EvidenceError, "changed"):
                self.build()
        self.assertFalse((self.fixture.root / "paired").exists())


class IndependentHandTests(unittest.TestCase):
    def inputs(self, count=8, known=False):
        points = np.tile(fixtures.hands(), (count, 1, 1, 1))
        points[:, 0, 8, 0] += np.arange(count) * .001
        points[:, 1, 8, 1] += np.arange(count) * .001
        pts = np.arange(count, dtype=np.int64) * 40
        shots = {
            "timeBase": [1, 1000], "width": fixtures.SIZE[0], "height": fixtures.SIZE[1],
            "shots": [{"shotId": 0, "startPts": 0, "endPtsExclusive": count * 40}],
        }
        inputs = {
            "hand": {"pts": pts, "shot_id": np.zeros(count, np.int32), "image_landmarks": points,
                     "detection_source": np.ones(count, np.int8)},
            "geometry": {"coordinates": np.tile(fixtures.geometry(), (count, 1, 1)),
                         "confidence": np.ones((count, 6)), "state": np.full(count, STATE_CODES["trackable"] if known else STATE_CODES["unknown"])},
            "documents": {"shots": shots},
        }
        if not known:
            inputs["geometry"]["coordinates"][:] = np.nan
            inputs["geometry"]["confidence"][:] = 0
        roles = {"role": np.tile([1, 2] if known else [0, 0], (count, 1)),
                 "track_id": np.tile([7, 8] if known else [-1, -1], (count, 1))}
        return inputs, roles

    def produce(self, inputs, roles, clips=None):
        if clips is None:
            clips = [{"label": "all", "startPts": 0, "endPtsExclusive": inputs["documents"]["shots"]["shots"][-1]["endPtsExclusive"]}]
        return structured_observations(inputs, roles, clips)[0]

    def test_unknown_hand_views_keep_local_fingers_and_movement_not_geometry_or_role(self):
        inputs, roles = self.inputs()
        result = self.produce(inputs, roles)
        masks, values = result["structured_available"], result["structured"]
        self.assertEqual(values.shape, (8, 4, 194))
        self.assertFalse(masks[:, :2].any())
        self.assertFalse(masks[..., :98].any())
        self.assertTrue(masks[:, 2:, 98:140].all())
        self.assertTrue(masks[1:, 2:, 140:184].all())
        self.assertGreater(abs(values[1, 2, 157]), 0)
        self.assertGreater(abs(values[1, 3, 156]), 0)
        self.assertFalse(roles["role"].any())
        np.testing.assert_array_equal(masks.any(-1).sum(-1), 2)

    def test_source_pixel_local_coordinates_are_translation_scale_invariant(self):
        inputs, roles = self.inputs()
        original = self.produce(inputs, roles)
        points = inputs["hand"]["image_landmarks"][..., :2]
        points[:] = points * .75 + [.08, .1]
        changed = self.produce(inputs, roles)
        np.testing.assert_array_equal(changed["structured_available"], original["structured_available"])
        for start, stop in ((98, 140), (140, 182), (184, 186)):
            np.testing.assert_allclose(changed["structured"][..., start:stop], original["structured"][..., start:stop], atol=1e-4 if start == 140 else 3e-6)
        wrist = points[0, 0, 0] * fixtures.SIZE
        scale = np.median(np.linalg.norm((points[0, 0, [5, 9, 13, 17]] * fixtures.SIZE) - wrist, axis=-1))
        expected = (points[0, 0, 8] * fixtures.SIZE - wrist) / scale
        np.testing.assert_allclose(changed["structured"][0, 3, 114:116], expected, atol=1e-6)

    def test_slot_swaps_follow_palm_identity_and_not_anatomical_handedness(self):
        inputs, roles = self.inputs()
        original = self.produce(inputs, roles)
        inputs["hand"]["image_landmarks"][1::2] = inputs["hand"]["image_landmarks"][1::2, ::-1].copy()
        inputs["hand"]["handedness"] = np.tile([1, 0], (8, 1))
        swapped = self.produce(inputs, roles)
        for name in original:
            np.testing.assert_allclose(swapped[name], original[name], atol=1e-6)

    def test_wrist_velocity_uses_actual_pts_and_previous_observed_palm_scale(self):
        inputs, roles = self.inputs()
        inputs["hand"]["pts"] = np.array([0, 30, 80, 120, 160, 200, 240, 280], np.int64)
        points = inputs["hand"]["image_landmarks"]
        wrist = points[0, 0, 0, :2].copy()
        points[1:, 0, :, :2] = wrist + (points[1:, 0, :, :2] - wrist) * 1.2 + [.003, .002]
        scale = np.linalg.norm((points[0, 0, 5, :2] - wrist) * fixtures.SIZE)
        result = self.produce(inputs, roles)
        observed_delta = (points[1, 0, 0, :2].astype(np.float64) - wrist) * fixtures.SIZE
        np.testing.assert_allclose(result["structured"][1, 3, 182:184], observed_delta / scale / .03, atol=1e-6)
        self.assertTrue(result["structured_available"][1, 3, 182:184].all())

    def test_missing_unknown_hand_resets_only_its_track_and_generic_slot(self):
        inputs, roles = self.inputs()
        inputs["hand"]["image_landmarks"][2, 0] = np.nan
        result = self.produce(inputs, roles)
        self.assertFalse(result["structured_available"][2, 3].any())
        for start, stop in VELOCITY_SLICES:
            self.assertFalse(result["structured_available"][3, 3, start:stop].any())
        self.assertTrue(result["structured_available"][3, 2, 140:184].all())
        self.assertNotEqual(result["segment_id"][1, 3], result["segment_id"][3, 3])
        self.assertEqual(result["segment_id"][1, 2], result["segment_id"][3, 2])

    def test_missing_cut_gap_detector_and_clip_reset_all_motion(self):
        for change in ("missing", "shot", "gap", "detector", "clip", "cut_uncertainty", "transition", "guitar_absent"):
            with self.subTest(change=change):
                inputs, roles = self.inputs(known=True)
                clips = None
                row = 3
                if change == "missing":
                    inputs["hand"]["image_landmarks"][2, 0] = np.nan
                elif change == "shot":
                    inputs["hand"]["shot_id"][3:] = 1
                elif change == "gap":
                    inputs["hand"]["pts"][3:] += 120
                    inputs["documents"]["shots"]["shots"][-1]["endPtsExclusive"] += 120
                elif change == "detector":
                    inputs["hand"]["detection_source"][3:] = 2
                elif change == "clip":
                    clips = [{"label": "a", "startPts": 0, "endPtsExclusive": 120},
                             {"label": "b", "startPts": 120, "endPtsExclusive": 320}]
                elif change == "cut_uncertainty":
                    inputs["documents"]["shots"]["automaticReview"] = {
                        "method": "conservative-cut-boundaries-v1", "uncertainIntervals": [
                            {"startPts": 80, "endPtsExclusive": 120, "reason": "uncertain-cut"}],
                    }
                else:
                    inputs["geometry"]["state"][2] = STATE_CODES[change]
                result = self.produce(inputs, roles, clips)
                for start, stop in VELOCITY_SLICES:
                    self.assertFalse(result["structured_available"][row, 0, start:stop].any())
                self.assertNotEqual(result["segment_id"][1, 0], result["segment_id"][row, 0])
                if change in ("transition", "guitar_absent", "cut_uncertainty"):
                    self.assertFalse(result["structured_available"][2].any())
                if change == "missing":
                    self.assertTrue(result["structured_available"][row, 1, 140:184].all())

    def test_bounds_tiny_palms_missing_wrist_and_partial_fingers(self):
        for change in ("bounds", "tiny", "wrist", "mcps"):
            with self.subTest(change=change):
                inputs, roles = self.inputs()
                points = inputs["hand"]["image_landmarks"][:, 0]
                if change == "bounds":
                    points[:, :, 0] = 1.01
                elif change == "tiny":
                    points[..., :2] = points[:, 0:1, :2] + (points[..., :2] - points[:, 0:1, :2]) * .01
                elif change == "wrist":
                    points[:, 0] = np.nan
                else:
                    points[:, [5, 13]] = np.nan
                result = self.produce(inputs, roles)
                self.assertEqual(int(result["structured_available"][0].any(-1).sum()), 1)
        inputs, roles = self.inputs()
        inputs["hand"]["image_landmarks"][:, 0, [4, 8, 12, 16, 20]] = np.nan
        result = self.produce(inputs, roles)
        self.assertTrue(result["structured_available"][:, 3, 184:186].all())
        self.assertFalse(result["structured_available"][:, 3, 114:116].any())
        self.assertTrue(result["structured_available"][:, 3, 98:100].all())
        np.testing.assert_allclose(np.linalg.norm(result["structured"][:, 3, 184:186], axis=-1), 1., atol=1e-6)

    def test_known_roles_do_not_duplicate_into_generic_views_and_role_changes_reset(self):
        inputs, roles = self.inputs(known=True)
        roles["role"][3:, 0] = 0
        roles["track_id"][3:, 0] = -1
        result = self.produce(inputs, roles)
        np.testing.assert_array_equal(result["structured_available"].any(-1).sum(-1), 2)
        self.assertFalse(result["structured_available"][:3, 2:].any())
        self.assertFalse(result["structured_available"][3:, 0].any())
        for start, stop in VELOCITY_SLICES:
            self.assertFalse(result["structured_available"][3, 2:, start:stop].any())
        self.assertTrue(result["structured_available"][3:, 1, :42].all())

    def test_duplicate_and_ambiguous_palms_do_not_fabricate_motion_identity(self):
        inputs, roles = self.inputs()
        points = inputs["hand"]["image_landmarks"]
        points[2, 1] = points[2, 0]
        result = self.produce(inputs, roles)
        self.assertEqual(int(result["structured_available"][2].any(-1).sum()), 1)
        for start, stop in VELOCITY_SLICES:
            self.assertFalse(result["structured_available"][2, :, start:stop].any())
        from paired_inputs import _PalmTracker, _palm_observation

        tracker = _PalmTracker()
        palm = _palm_observation(fixtures.hands()[0, :, :2], np.array(fixtures.SIZE))
        before = tracker.update({0: palm, 1: palm}, True)
        after = tracker.update({0: palm, 1: palm}, False)
        self.assertFalse(set(before.values()) & set(after.values()))


if __name__ == "__main__":
    unittest.main()
