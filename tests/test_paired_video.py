from collections import Counter
from copy import deepcopy
import json
import os
import stat
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import paired_video
from scripts.dataset_io import ROOT, sha256
from scripts.dataset_release import validate_release
from scripts.paired_video import FEATURE_LAYOUT, INPUT_REPRESENTATION, SCHEMA_VERSION, STRUCTURED_DIM, VIDEO_FIELDS, PairedVideoIndex, build_index, empty_video, index_manifest, load_inference_video
from scripts.transcriber_audio import FeatureConfig, HarnessError
from scripts.transcriber_data import PAIRED_TECHNIQUE_TARGET_POLICY, TrainingDataset, collate_windows
from scripts.transcriber_model import ModelConfig
from scripts.video_features import VELOCITY_SLICES, VIEW_ORDER
from scripts.fretboard_features import (
    FEATURE_LAYOUT as FRETBOARD_FEATURE_LAYOUT,
    INPUT_REPRESENTATION as FRETBOARD_INPUT_REPRESENTATION,
)
from tests.test_dataset_release import synthetic_release


def bundle_fixture(root, audio_path, identifier="piece-0", pts=(1000, 1040, 1080, 1120)):
    directory = root / identifier
    directory.mkdir()
    pts = np.asarray(pts, np.int64)
    paths = {}
    for name in ("video", "shots", "hands", "geometry", "annotations", "handArrays", "geometryArrays", "roles", "roleArrays", "alignment"):
        path = directory / (name + (".npz" if name.endswith("Arrays") else ".json"))
        if name != "handArrays":
            path.write_text(json.dumps({"fixture": name}))
        paths[name] = path
    paths["trimmedAudio"] = audio_path
    np.savez(paths["handArrays"], pts=pts)
    paths["alignment"].write_text(json.dumps({
        "kind": "video-to-trimmed-audio-alignment", "schemaVersion": 1, "status": "supported",
        "videoSha256": sha256(paths["video"]), "trimmedAudioSha256": sha256(audio_path),
        "rate": [1, 1], "videoStartSecondsForTrimmedAudioZero": 0.,
    }))
    n = len(pts)
    arrays = {
        "audio_seconds": pts.astype(np.float64) / 1000, "pts": pts,
        "technique_available": np.ones(n, bool),
        "segment_id": np.tile([0, 1, -1, -1], (n, 1)).astype(np.int64),
        "structured": np.zeros((n, 4, STRUCTURED_DIM), np.float32),
        "structured_available": np.zeros((n, 4, STRUCTURED_DIM), bool),
    }
    arrays["structured_available"][:, :2, :186] = True
    for start, stop in VELOCITY_SLICES:
        arrays["structured_available"][0, :, start:stop] = False
    arrays["structured"][:, :2, 84] = 1.
    arrays["structured"][:, :2, 184] = 1.
    array_path = directory / "inputs.npz"
    np.savez_compressed(array_path, **arrays)
    report = {
        "kind": "paired-video-inputs", "schemaVersion": SCHEMA_VERSION, "id": identifier,
        "inputRepresentation": INPUT_REPRESENTATION,
        "audioSha256": sha256(audio_path), "videoSha256": sha256(paths["video"]),
        "timeBase": [1, 1000], "featureDimension": STRUCTURED_DIM,
        "arraysPath": "inputs.npz", "arraysSha256": sha256(array_path),
        "inputPaths": {name: str(path) for name, path in paths.items()},
        "inputSha256": {name: sha256(path) for name, path in paths.items()},
        "frameCount": n, "viewOrder": VIEW_ORDER, "featureLayout": deepcopy(FEATURE_LAYOUT),
        "clock": {"sampleRate": 8000, "offsetSamples": 0, "rate": [1, 1]},
        "clips": [{"startPts": int(pts[0]), "endPtsExclusive": int(pts[-1]) + 40}],
        "correspondenceIntervals": [], "maximumGapSeconds": .09,
    }
    path = directory / "inputs.json"
    path.write_text(json.dumps(report))
    return path, report, arrays


def schema5_bundle_fixture(root, audio_path, identifier="piece-0", pts=(1000, 1040, 1080, 1120)):
    path, report, arrays = bundle_fixture(root, audio_path, identifier, pts)
    directory = path.parent
    fretboard_arrays = directory / "fretboard.npz"
    count = len(arrays["pts"])
    np.savez_compressed(
        fretboard_arrays,
        pts=arrays["pts"], shot_id=np.zeros(count, np.int32),
        keypoints=np.tile(np.array([
            [[.2, .3], [.2, .5]], [[.5, .3], [.5, .5]], [[.8, .3], [.8, .5]],
        ], np.float32), (count, 1, 1, 1)),
        available=np.ones((count, 3, 2), bool),
        confidence=np.full(count, .9, np.float32),
        source=np.full(count, 2, np.int8),
        age_seconds=np.zeros(count, np.float32),
        flow_error=np.zeros(count, np.float32),
        detector_anchor=np.zeros(count, bool),
    )
    fretboard_report = directory / "fretboard.json"
    fretboard_report.write_text(json.dumps({
        "kind": "six-point-fretboard-observations", "schemaVersion": 1,
        "videoSha256": report["videoSha256"], "shotsSha256": report["inputSha256"]["shots"],
        "timeBase": report["timeBase"], "frameCount": count, "arrays": "fretboard.npz",
        "sourceEncoding": {"unavailable": 0, "detector": 1, "optical_flow": 2},
        "arraysSha256": sha256(fretboard_arrays),
    }))
    report["schemaVersion"] = 5
    report["inputRepresentation"] = FRETBOARD_INPUT_REPRESENTATION
    report["featureDimension"] = 233
    report["featureLayout"] = deepcopy(FRETBOARD_FEATURE_LAYOUT)
    report["inputPaths"].update(
        fretboard=str(fretboard_report), fretboardArrays=str(fretboard_arrays),
    )
    report["inputSha256"].update(
        fretboard=sha256(fretboard_report), fretboardArrays=sha256(fretboard_arrays),
    )
    extended = {
        **arrays,
        "structured": np.zeros((*arrays["structured"].shape[:-1], 233), np.float32),
        "structured_available": np.zeros((*arrays["structured_available"].shape[:-1], 233), bool),
    }
    extended["structured"][..., :194] = arrays["structured"]
    extended["structured_available"][..., :194] = arrays["structured_available"]
    extended["structured"][:, :2, 229] = .9
    extended["structured_available"][:, :2, 229:233] = True
    np.savez_compressed(path.with_name("inputs.npz"), **extended)
    report["arraysSha256"] = sha256(path.with_name("inputs.npz"))
    path.write_text(json.dumps(report))
    return path, report, extended




class PairedVideoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="paired-tests-", dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.manifest = synthetic_release(self.root / "release")
        _, self.records, _ = validate_release(self.manifest)
        self.audio = self.root / "release" / "audio" / "piece-0.flac"
        self.bundle, self.report, self.arrays = bundle_fixture(self.root, self.audio)

    def save(self):
        np.savez_compressed(self.bundle.with_name("inputs.npz"), **self.arrays)
        self.report["arraysSha256"] = sha256(self.bundle.with_name("inputs.npz"))
        self.bundle.write_text(json.dumps(self.report))

    def load(self):
        return load_inference_video(self.bundle, sha256(self.audio))

    def index(self):
        return build_index(self.manifest, [self.bundle], self.root / "index.json", root=self.root)[0]

    def test_index_release_discovery_preserves_schema_four_and_hash_checks_pathless_indexes(self):
        index = self.index()
        document = json.loads(index.read_text())
        self.assertEqual(document["schemaVersion"], 4)
        self.assertEqual(document["manifestPath"], str(self.manifest.relative_to(self.root)))
        self.assertEqual(index_manifest(index, root=self.root), self.manifest)
        self.assertEqual(index_manifest(index, root=self.root, default_manifest=self.bundle), self.manifest)
        with self.assertRaisesRegex(HarnessError, "manifest hash"):
            index_manifest(index, root=self.root, manifest_path=self.bundle)
        del document["manifestPath"]
        older = self.root / "older-index.json"
        older.write_text(json.dumps(document))
        self.assertEqual(index_manifest(older, root=self.root, default_manifest=self.manifest), self.manifest)
        self.assertEqual(index_manifest(older, root=self.root, manifest_path=self.manifest), self.manifest)
        with self.assertRaisesRegex(HarnessError, "supply --manifest"):
            index_manifest(older, root=self.root)
        with self.assertRaisesRegex(HarnessError, "manifest hash"):
            index_manifest(older, root=self.root, default_manifest=self.bundle)
        PairedVideoIndex(older, sha256(self.manifest), self.records, root=self.root)
        document["manifestPath"] = "..\\manifest.json"
        older.write_text(json.dumps(document))
        with self.assertRaisesRegex(HarnessError, "escape"):
            index_manifest(older, root=self.root)

    def test_sparse_native_frames_and_nearest_audio_mapping(self):
        video = self.load()
        result = video.window(np.arange(49, 60) * .02)
        self.assertEqual(set(result), VIDEO_FIELDS)
        self.assertEqual(result["structured"].shape, (4, 4, 194))
        self.assertEqual(result["frame_indices"][0], -1)
        self.assertEqual(result["frame_indices"][-1], -1)
        self.assertEqual(video.identity["featureDimension"], 194)
        self.assertEqual(video.identity["inputRepresentation"], INPUT_REPRESENTATION)
        self.assertNotIn("imageSize", video.identity)
        self.assertLess(len(result["structured"]), len(result["frame_indices"]))
        self.assertFalse(result["structured_available"][..., 186:].any())
        self.assertFalse(result["structured"][..., 186:].any())







    def test_unavailable_frame_and_cut_are_not_bridged(self):
        self.arrays["structured"][1] = 0
        self.arrays["structured_available"][1] = False
        self.arrays["segment_id"][1] = -1
        self.arrays["segment_id"][2:] = [2, 3, -1, -1]
        for start, stop in VELOCITY_SLICES:
            self.arrays["structured_available"][2, :, start:stop] = False
        self.save()
        result = self.load().window([1., 1.02, 1.04, 1.06, 1.08, 1.10])
        self.assertEqual(result["frame_indices"].tolist(), [0, -1, -1, -1, 2, 2])
        self.arrays["segment_id"][3] = [4, 5, -1, -1]
        for start, stop in VELOCITY_SLICES:
            self.arrays["structured_available"][3, :, start:stop] = False
        self.save()
        result = self.load().window([1.08, 1.10, 1.12])
        self.assertEqual(result["frame_indices"].tolist(), [1, -1, 2])

    def test_frame_distance_is_capped_and_structure_is_usable(self):
        self.report["clips"][-1]["endPtsExclusive"] = 1200
        self.save()
        result = self.load().window([1.12, 1.16, 1.166])
        self.assertTrue(result["structured_available"].any())
        self.assertEqual(result["frame_indices"].tolist(), [1, 1, -1])

    def test_missing_record_placeholder_and_unknown_id_rejection(self):
        index = PairedVideoIndex(self.index(), sha256(self.manifest), self.records, root=self.root)
        result = index.window("piece-1", [0., .02])
        expected = empty_video(2)
        for key in expected:
            torch.testing.assert_close(result[key], expected[key], rtol=0, atol=0)
        with self.assertRaisesRegex(HarnessError, "Unknown"):
            index.window("not-in-release", [1.])
        self.assertEqual(index.window("piece-0", [4.])["frame_indices"].tolist(), [-1])

    def test_schema_four_and_five_load_distinctly_and_preserve_index_dimension(self):
        legacy = self.load()
        v5_path, report, arrays = schema5_bundle_fixture(
            self.root, self.audio, "piece-v5", pts=(1000, 1040, 1080, 1120),
        )
        report["id"] = "piece-0"
        v5_path.write_text(json.dumps(report))
        v5 = load_inference_video(v5_path, sha256(self.audio))
        self.assertEqual((legacy.schema_version, legacy.feature_dimension), (4, 194))
        self.assertEqual((v5.schema_version, v5.feature_dimension), (5, 233))
        self.assertEqual(legacy.window([1.])["structured"].shape[-1], 194)
        self.assertEqual(v5.window([1.])["structured"].shape[-1], 233)
        np.testing.assert_array_equal(v5.arrays["structured"][..., :194], arrays["structured"][..., :194])

        v5_index, _ = build_index(self.manifest, [v5_path], self.root / "v5-index.json", root=self.root)
        index = PairedVideoIndex(v5_index, sha256(self.manifest), self.records, root=self.root)
        self.assertEqual(index.window("piece-1", [0., .02])["structured"].shape, (1, 4, 233))
        observed = index.window("piece-0", [1., 1.04])
        missing = index.window("piece-1", [0., .02])
        items = [
            {"features": torch.zeros(2, 2), "conditioning": torch.zeros(2, 12), "targets": {}, "masks": {}, "metadata": {}, "video": video}
            for video in (observed, missing)
        ]
        self.assertEqual(collate_windows(items)["video"]["structured"].shape[-1], 233)

    def test_index_and_collation_reject_mixed_schema_dimensions(self):
        second_audio = self.root / "release" / "audio" / "piece-1.flac"
        second, _, _ = schema5_bundle_fixture(self.root, second_audio, "piece-1")
        with self.assertRaisesRegex(HarnessError, "only one schema"):
            build_index(self.manifest, [self.bundle, second], self.root / "mixed-index.json", root=self.root)
        legacy = empty_video(2)
        extended = empty_video(2, 233)
        items = [
            {"features": torch.zeros(2, 2), "conditioning": torch.zeros(2, 12), "targets": {}, "masks": {}, "metadata": {}, "video": video}
            for video in (legacy, extended)
        ]
        with self.assertRaisesRegex(HarnessError, "mix paired-video schemas"):
            collate_windows(items)

    def test_full_release_split_group_audio_and_manifest_identity_are_required(self):
        path = self.index()
        original = json.loads(path.read_text())
        for key, value in (("split", "validation"), ("groupId", "piece-1"), ("audioSha256", "0" * 64), ("id", "unknown")):
            with self.subTest(key=key):
                document = deepcopy(original)
                document["records"][0][key] = value
                path.write_text(json.dumps(document))
                with self.assertRaises(HarnessError):
                    PairedVideoIndex(path, sha256(self.manifest), self.records, root=self.root)
        path.write_text(json.dumps(original))
        with self.assertRaisesRegex(HarnessError, "different release"):
            PairedVideoIndex(path, "0" * 64, self.records, root=self.root)
        with self.assertRaisesRegex(HarnessError, "unknown"):
            PairedVideoIndex(path, sha256(self.manifest), self.records[1:], root=self.root)
        grouped = deepcopy(self.records)
        grouped[1][0]["groupId"] = grouped[0][0]["groupId"]
        with self.assertRaisesRegex(HarnessError, "grouped"):
            PairedVideoIndex(path, sha256(self.manifest), grouped, root=self.root)

    def test_shape_finite_mask_and_extra_target_arrays_fail_closed(self):
        original = deepcopy(self.arrays)
        mutations = (
            ("structured", lambda value: value[:, :, :-1]),
            ("structured", lambda value: np.full_like(value, np.nan)),
            ("structured", lambda value: value.astype(np.float64)),
            ("audio_seconds", lambda value: value + .001),
            ("pts", lambda value: value[::-1]),
            ("segment_id", lambda value: np.full_like(value, -1)),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                self.arrays = deepcopy(original)
                self.arrays[name] = mutate(self.arrays[name])
                self.save()
                with self.assertRaises(HarnessError):
                    self.load()
        self.arrays = deepcopy(original)
        self.arrays["fret_targets"] = np.zeros(4)
        self.save()
        with self.assertRaisesRegex(HarnessError, "exactly"):
            self.load()

    def test_clock_crop_extent_native_cadence_and_feature_masks_are_validated(self):
        original = deepcopy(self.report)
        for change in ("sampleRate", "extent", "cadence", "layout"):
            with self.subTest(change=change):
                self.report = deepcopy(original)
                if change == "sampleRate":
                    self.report["clock"]["sampleRate"] = 16000
                elif change == "extent":
                    self.report["clips"][-1]["endPtsExclusive"] = 8000
                elif change == "cadence":
                    path = Path(self.report["inputPaths"]["handArrays"])
                    np.savez(path, pts=np.array([1000, 1020, 1040, 1080, 1120], np.int64))
                    self.report["inputSha256"]["handArrays"] = sha256(path)
                else:
                    self.report["featureLayout"]["landmarkXY"]["points"][0] = 99
                self.save()
                with self.assertRaises(HarnessError):
                    self.load()
        self.report = deepcopy(original)
        np.savez(Path(self.report["inputPaths"]["handArrays"]), pts=self.arrays["pts"])
        self.report["inputSha256"]["handArrays"] = sha256(Path(self.report["inputPaths"]["handArrays"]))
        self.arrays["structured_available"][1, 0, 0] = False
        self.save()
        with self.assertRaisesRegex(HarnessError, "both XY"):
            self.load()

    def test_geometry_frame_rejects_edge_midpoint_origin_instead_of_joint(self):
        self.arrays["structured"][:, :2, 84] = .97
        self.arrays["structured"][:, :2, 86] = -.03
        self.save()
        with self.assertRaisesRegex(HarnessError, "joint zero and nut one"):
            self.load()

    def test_hash_mutation_alias_paths_and_overwrite_are_rejected(self):
        with self.assertRaisesRegex(HarnessError, "audio hash"):
            load_inference_video(self.bundle, "0" * 64)
        video = self.load()
        source = Path(self.report["inputPaths"]["video"])
        source.write_text("mutated")
        with self.assertRaisesRegex(HarnessError, "changed"):
            video.check_unchanged()
        with self.assertRaisesRegex(HarnessError, "hash"):
            self.load()
        source.write_text(json.dumps({"fixture": "video"}))
        path = self.index()
        with self.assertRaisesRegex(HarnessError, "new unaliased"):
            build_index(self.manifest, [self.bundle], path, root=self.root)
        index = PairedVideoIndex(path, sha256(self.manifest), self.records, root=self.root)
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(HarnessError, "changed"):
            index.check_unchanged()
        alias = self.root / "alias.json"
        os.link(self.bundle, alias)
        with self.assertRaisesRegex(HarnessError, "regular"):
            load_inference_video(alias, sha256(self.audio))

    def test_index_checks_each_file_and_parent_once_and_still_guards_other_recordings(self):
        second_audio = self.root / "release" / "audio" / "piece-1.flac"
        second, report, _ = bundle_fixture(self.root, second_audio, "piece-1")
        path, _ = build_index(self.manifest, [self.bundle, second], self.root / "index.json", root=self.root)
        index = PairedVideoIndex(path, sha256(self.manifest), self.records, root=self.root)
        calls = Counter()
        lstat = Path.lstat

        def observed(path):
            calls[path] += 1
            return lstat(path)

        with patch.object(Path, "lstat", observed), patch.object(paired_video, "_file", side_effect=AssertionError("Do not repeat path canonicalization")):
            index.check_unchanged()
        self.assertEqual(calls, Counter({path: 1 for path in (*index._guards, *index._parents)}))
        Path(report["inputPaths"]["hands"]).write_text("changed nonselected recording")
        with self.assertRaisesRegex(HarnessError, "changed"):
            index.window("piece-0", [1., 1.04])

    def test_runtime_guards_reject_new_parent_aliases_and_file_hardlinks(self):
        video = self.load()
        parent = self.bundle.parent
        lstat = Path.lstat
        for mode, attributes in ((stat.S_IFLNK, 0), (stat.S_IFDIR, 0x400)):
            with self.subTest(mode=mode, attributes=attributes):
                def changed(path):
                    return SimpleNamespace(st_mode=mode, st_file_attributes=attributes) if path == parent else lstat(path)

                with patch.object(Path, "lstat", changed), self.assertRaisesRegex(HarnessError, "alias"):
                    video.window([1.])
        alias = self.root / "new-hardlink.npz"
        os.link(self.bundle.with_name("inputs.npz"), alias)
        try:
            with self.assertRaisesRegex(HarnessError, "independent regular"):
                video.window([1.])
        finally:
            alias.unlink()

    def test_runtime_guards_detect_replacement_with_same_bytes_and_modified_time(self):
        video = self.load()
        source = self.bundle.with_name("inputs.npz")
        before = source.stat()
        replacement = self.root / "replacement.npz"
        replacement.write_bytes(source.read_bytes())
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        os.replace(replacement, source)
        with self.assertRaisesRegex(HarnessError, "changed"):
            video.window([1.])

    def test_existing_dataset_fallback_targets_and_separate_video_padding(self):
        path = self.index()
        features = FeatureConfig(sample_rate=8000, n_fft=512, hop_length=160, n_mels=16, f_max=3000)
        model = ModelConfig(n_mels=16)
        plain = TrainingDataset(self.manifest, "train", features, model, root=self.root)
        paired = TrainingDataset(self.manifest, "train", features, model, root=self.root, video_index_path=path)
        validation = TrainingDataset(self.manifest, "validation", features, model, root=self.root, video_index_path=path)
        self.assertEqual(paired.video_identity, validation.video_identity)
        self.assertEqual(paired.video_paired_window_indices, [0])
        self.assertEqual(validation.video_paired_window_indices, [])
        old, missing = plain[0], validation[0]
        with patch.object(paired.video_index, "check_unchanged", wraps=paired.video_index.check_unchanged) as check:
            new = paired[0]
        check.assert_called_once_with()
        self.assertNotIn("video", old)
        self.assertIsNone(plain.video_window(plain.records[0], np.array([1., 1.02])))
        for name in ("features", "conditioning"):
            torch.testing.assert_close(old[name], new[name], rtol=0, atol=0)
        for category in ("targets", "masks"):
            for name in old[category]:
                expected = old[category][name].clone()
                if category == "masks" and name == "technique":
                    from scripts.technique_supervision import TECHNIQUE_TYPES

                    axis = TECHNIQUE_TYPES.index("rasgueado")
                    expected[:, axis] &= old["targets"]["technique"][:, axis] > 0
                torch.testing.assert_close(expected, new[category][name], rtol=0, atol=0)
        self.assertFalse(missing["video"]["structured_available"].any())
        batch = collate_windows([new, missing])
        self.assertEqual(batch["video"]["structured"].shape, (2, 4, 4, 194))
        self.assertEqual(batch["video"]["frame_indices"].shape, batch["valid_frames"].shape)
        self.assertTrue((batch["video"]["segment_id"][1] == -1).all())
        self.assertNotIn("video", collate_windows([old]))
        with self.assertRaisesRegex(HarnessError, "mix"):
            collate_windows([old, new])

    def test_paired_rasgueado_absence_is_unknown_and_native_positives_survive(self):
        from scripts.score_alignment import ScoreClock
        from scripts.technique_supervision import TECHNIQUE_TYPES, projected_techniques

        path = self.index()
        before = {asset: sha256(asset) for asset in (self.root / "release").rglob("*") if asset.is_file()}
        features = FeatureConfig(sample_rate=8000, n_fft=512, hop_length=160, n_mels=16, f_max=3000)
        model = ModelConfig(n_mels=16, architecture_version=2)
        plain = TrainingDataset(self.manifest, "train", features, model, root=self.root)
        paired = TrainingDataset(self.manifest, "train", features, model, root=self.root, video_index_path=path)
        axis = TECHNIQUE_TYPES.index("rasgueado")
        old, new = plain[0], paired[0]
        self.assertTrue(old["masks"]["technique"][:, axis].any())
        self.assertFalse(new["masks"]["technique"][:, axis].any())
        self.assertFalse(new["targets"]["technique"][:, axis].any())
        self.assertNotIn("pairedTechniqueTargetPolicy", old["metadata"])
        self.assertEqual(new["metadata"]["pairedTechniqueTargetPolicy"], PAIRED_TECHNIQUE_TARGET_POLICY)
        self.assertEqual(paired.video_identity["targetPolicy"], PAIRED_TECHNIQUE_TARGET_POLICY)
        for dataset in (plain, paired):
            record = dataset.records[0]
            labels = deepcopy(record["data"].labels)
            labels["targets"]["notes"][0]["sourceSegments"][0]["beatTechniques"] = {"rasgueado": "ami_2"}
            record["techniques"] = projected_techniques(labels, record["candidate"], ScoreClock(labels, record["data"].normalization))
        old, new = plain[0], paired[0]
        self.assertEqual(int(new["targets"]["technique"][:, axis].sum()), 1)
        self.assertEqual(int(new["masks"]["technique"][:, axis].sum()), 1)
        positive = new["targets"]["technique"][:, axis] > 0
        self.assertTrue(new["masks"]["technique"][positive, axis].all())
        for name in ("technique_direction", "technique_strings"):
            torch.testing.assert_close(old["targets"][name], new["targets"][name], rtol=0, atol=0)
            torch.testing.assert_close(old["masks"][name], new["masks"][name], rtol=0, atol=0)
        self.assertEqual(before, {asset: sha256(asset) for asset in before})

    def test_collation_pads_audio_indices_with_minus_one(self):
        items = []
        for length in (2, 4):
            items.append({
                "features": torch.zeros(length, 2), "conditioning": torch.zeros(length, 12),
                "targets": {}, "masks": {}, "metadata": {},
                "video": empty_video(length),
            })
        batch = collate_windows(items)
        self.assertEqual(batch["video"]["frame_indices"].tolist(), [[-1] * 4, [-1] * 4])
        self.assertEqual(batch["video"]["structured"].shape[1], 1)

    def test_index_contains_only_structure_and_coverage_needs_no_additional_array_io(self):
        second_audio = self.root / "release" / "audio" / "piece-1.flac"
        second, _, _ = bundle_fixture(self.root, second_audio, "piece-1")
        path, report = build_index(self.manifest, [self.bundle, second], self.root / "index.json", root=self.root)
        index = PairedVideoIndex(path, sha256(self.manifest), self.records, root=self.root)
        self.assertEqual(report["schemaVersion"], 4)
        self.assertEqual(report["inputRepresentation"], INPUT_REPRESENTATION)
        self.assertNotIn("imageSize", report)
        for bundle in index.bundles.values():
            self.assertNotIn("images", bundle.arrays)
            self.assertEqual(set(bundle.arrays), set(self.arrays))
        with patch.object(paired_video.np, "load", side_effect=AssertionError("Coverage must use resident structured data")):
            for identifier in ("piece-0", "piece-1"):
                available, technique = index.availability_at(identifier, [1., 1.04])
                self.assertTrue(available[:, :2].all())
                self.assertFalse(available[:, 2:].any())
                self.assertTrue(technique.all())
                self.assertTrue(index.has_usable(identifier, 1., 1.1))

    def test_inference_availability_and_empty_windows_preserve_source_guards(self):
        video = self.load()
        with patch.object(paired_video.np, "load", side_effect=AssertionError("No window-time source array loading")):
            available, techniques = video.availability_at([1., 1.04, 4.])
            self.assertEqual(available.tolist(), [[True, True, False, False], [True, True, False, False], [False] * 4])
            self.assertEqual(techniques.tolist(), [True, True, False])
            self.assertEqual(video.window([4.])["frame_indices"].tolist(), [-1])
            self.assertEqual(video.window([])["structured"].shape[0], 1)
            video.window([1.])
            video.window([1.04])
            self.bundle.with_name("inputs.npz").write_bytes(self.bundle.with_name("inputs.npz").read_bytes() + b"changed")
            for action in (lambda: video.window([1.]), lambda: video.availability_at([1.]), video.check_unchanged):
                with self.assertRaisesRegex(HarnessError, "changed"):
                    action()

    def test_rgb_schemas_controls_and_payloads_are_rejected_not_ignored(self):
        original = deepcopy(self.report)
        for fields in ({"schemaVersion": 1}, {"schemaVersion": 2}, {"schemaVersion": 3}, {"imageSize": 96}, {"images": "old-inputs.npz"}):
            self.report = {**original, **fields}
            self.save()
            before = self.bundle.read_bytes()
            with self.assertRaisesRegex(HarnessError, "RGB|image fields"):
                self.load()
            self.assertEqual(self.bundle.read_bytes(), before)
        self.report = original
        for name in ("images", "available"):
            self.arrays[name] = np.zeros(0, np.uint8)
            self.save()
            with self.assertRaisesRegex(HarnessError, "RGB/image arrays"):
                self.load()
            del self.arrays[name]
        self.save()
        path = self.index()
        report = json.loads(path.read_text())
        for fields in ({"schemaVersion": 1}, {"schemaVersion": 2}, {"schemaVersion": 3}, {"imageSize": 96}):
            path.write_text(json.dumps({**report, **fields}))
            with self.assertRaisesRegex(HarnessError, "RGB"):
                PairedVideoIndex(path, sha256(self.manifest), self.records, root=self.root)

    def test_collation_rejects_obsolete_rgb_item_fields(self):
        item = {
            "features": torch.zeros(2, 2), "conditioning": torch.zeros(2, 12),
            "targets": {}, "masks": {}, "metadata": {}, "video": empty_video(2),
        }
        item["video"]["images"] = torch.zeros(0)
        with self.assertRaisesRegex(HarnessError, "RGB"):
            collate_windows([item])

    def test_unknown_views_are_usable_without_any_calibration(self):
        self.arrays["structured"][:, 2:] = self.arrays["structured"][:, :2]
        self.arrays["structured_available"][:, 2:] = self.arrays["structured_available"][:, :2]
        self.arrays["structured"][:, :2] = 0
        self.arrays["structured_available"][:, :2] = False
        self.arrays["structured"][..., :98] = 0
        self.arrays["structured_available"][..., :98] = False
        self.arrays["segment_id"][:] = [-1, -1, 0, 1]
        self.save()
        video = self.load()
        available, techniques = video.availability_at([1., 1.04, 4.])
        self.assertEqual(available.tolist(), [[False, False, True, True]] * 2 + [[False] * 4])
        geometry, hands, coarse = video.feature_availability_at([1., 1.04, 4.])
        self.assertFalse(geometry.any())
        self.assertFalse(coarse.any())
        np.testing.assert_array_equal(hands, available)
        self.assertTrue(video.has_usable(1., 1.1))
        features = FeatureConfig(sample_rate=8000, n_fft=512, hop_length=160, n_mels=16, f_max=3000)
        dataset = TrainingDataset(self.manifest, "train", features, ModelConfig(n_mels=16), root=self.root, video_index_path=self.index())
        coverage = dataset.video_paired_coverage
        self.assertEqual(dataset.video_paired_window_indices, [0])
        self.assertGreater(coverage["audioFramesWithIndependentHand"], 0)
        self.assertEqual(coverage["audioFramesWithGeometry"], 0)
        self.assertGreater(coverage["audioFramesWithUsableVideo"], 0)
        self.assertEqual(coverage["audioFramesWithUsablePluckingVideo"], 0)
        self.assertEqual(coverage["audioFramesWithUsableUnassignedVideo"], coverage["audioFramesWithUsableVideo"])
        batch = collate_windows([dataset[0]])
        self.assertEqual(batch["video"]["structured"].shape[-2:], (4, 194))

    def test_all_velocity_groups_require_previous_points_and_matching_segments(self):
        original = deepcopy(self.arrays)
        for start, stop in VELOCITY_SLICES:
            with self.subTest(start=start):
                self.arrays = deepcopy(original)
                self.arrays["structured_available"][0, 0, start:stop] = True
                self.save()
                with self.assertRaisesRegex(HarnessError, "motion"):
                    self.load()
        for source, velocity in ((0, 42), (110, 152), (98, 182)):
            with self.subTest(source=source):
                self.arrays = deepcopy(original)
                self.arrays["structured_available"][0, 0, source:source + 2] = False
                if source == 98:
                    self.arrays["structured_available"][0, 0, 98:] = False
                    self.arrays["structured"][0, 0, 98:] = 0
                self.save()
                with self.assertRaisesRegex(HarnessError, "motion"):
                    self.load()
        self.arrays = deepcopy(original)
        self.save()
        window = self.load().window([1.12])
        for start, stop in VELOCITY_SLICES:
            self.assertFalse(window["structured_available"][0, :, start:stop].any())

    def test_local_wrist_orientation_and_unavailable_values_are_validated(self):
        original = deepcopy(self.arrays)
        for column, value, pattern in ((98, .01, "wrist zero"), (184, .8, "unit vector"), (140, .2, "explicitly zero")):
            with self.subTest(column=column):
                self.arrays = deepcopy(original)
                self.arrays["structured"][0, 0, column] = value
                self.save()
                with self.assertRaisesRegex(HarnessError, pattern):
                    self.load()
        self.arrays = deepcopy(original)
        self.arrays["structured_available"][2, 0, 185] = False
        self.save()
        with self.assertRaisesRegex(HarnessError, "both XY"):
            self.load()

    def test_old_numeric_schemas_and_dimensions_require_regeneration(self):
        self.report["schemaVersion"] = 2
        self.save()
        with self.assertRaisesRegex(HarnessError, "regenerate schema 3/2"):
            self.load()
        self.report["schemaVersion"] = 4
        for dimension in (98, 186, 193):
            self.arrays["structured"] = np.zeros((4, 4, dimension), np.float32)
            self.arrays["structured_available"] = np.zeros((4, 4, dimension), bool)
            self.save()
            with self.assertRaisesRegex(HarnessError, "shape"):
                self.load()


if __name__ == "__main__":
    unittest.main()
