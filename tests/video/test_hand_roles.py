from fractions import Fraction
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import av

import numpy as np

from core import EvidenceError, PRIVATE_OUTPUT_ROOT, sha256
from geometry import GEOMETRY_POINTS, STATE_CODES
from hand_roles import REASON_CODES, RoleConfig, RoleTracker, assign_hand_roles, assign_role_arrays, frame_features, load_role_inputs, load_role_observations


SIZE = (1000, 600)


def geometry():
    return np.array([
        [.8, .4], [.45, .4], [.45, .35], [.45, .45], [.3, .4], [.15, .4],
    ], dtype=np.float32)


def hands():
    result = np.zeros((2, 21, 3), dtype=np.float32)
    result[0, :, :2] = [.65, .46]
    result[0, 0, :2] = [.67, .5]
    result[0, [4, 8, 12, 16, 20], :2] = [.65, .405]
    result[1, :, :2] = [.28, .38]
    result[1, 0, :2] = [.25, .36]
    result[1, [4, 8, 12, 16, 20], :2] = [.31, .42]
    return result


def feature(points=None, coordinates=None, confidence=None, state=0, config=RoleConfig()):
    return frame_features(
        hands() if points is None else points,
        geometry() if coordinates is None else coordinates,
        np.ones(6, np.float32) if confidence is None else confidence,
        state, SIZE, config,
    )


class RoleAlgorithmTests(unittest.TestCase):
    def test_roles_use_geometry_and_keep_one_visible_hand(self):
        tracker = RoleTracker()
        tracker.update(feature(), 0., 0)
        roles, confidence, _, _ = tracker.update(feature(), .1, 0)
        self.assertEqual(roles.tolist(), [1, 2])
        self.assertTrue((confidence > 0).all())
        single = hands()
        single[0] = np.nan
        roles, _, _, _ = tracker.update(feature(single), .14, 0)
        self.assertEqual(roles.tolist(), [0, 2])

    def test_motion_transform_and_mirrored_playing_do_not_change_roles(self):
        original = feature()
        for angle, scale, mirror in ((.3, .6, 1), (-.2, .7, -1)):
            matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            matrix[:, 0] *= mirror
            def transform(points):
                pixels = points * SIZE
                return (((pixels - [500, 300]) @ matrix.T) * scale + [500, 300]) / SIZE
            moved_hands = hands()
            moved_hands[..., :2] = transform(moved_hands[..., :2])
            moved = feature(moved_hands, transform(geometry()))
            np.testing.assert_allclose(moved["scores"], original["scores"], atol=1e-5)
            np.testing.assert_allclose(moved["anchors"], original["anchors"], atol=1e-5)

    def test_array_slot_swap_is_not_a_hand_role_swap(self):
        tracker = RoleTracker()
        tracker.update(feature(), 0., 0)
        roles, _, identities, _ = tracker.update(feature(), .1, 0)
        swapped_roles, _, swapped_ids, _ = tracker.update(feature(hands()[::-1].copy()), .14, 0)
        self.assertEqual(swapped_roles.tolist(), [2, 1])
        self.assertEqual(swapped_ids.tolist(), identities[::-1].tolist())
        self.assertEqual(tracker.swap_count, 0)

    def test_cuts_and_long_gaps_reset_confirmation(self):
        for change_shot in (True, False):
            tracker = RoleTracker()
            tracker.update(feature(), 0., 0)
            _, _, old_ids, _ = tracker.update(feature(), .1, 0)
            roles, _, ids, _ = tracker.update(feature(), .14 if change_shot else 1., 1 if change_shot else 0)
            self.assertEqual(roles.tolist(), [0, 0])
            self.assertFalse(set(ids.tolist()) & set(old_ids.tolist()))

    def test_missing_and_unstable_geometry_never_inherit_a_role(self):
        tracker = RoleTracker()
        tracker.update(feature(), 0., 0)
        tracker.update(feature(), .1, 0)
        for state in (STATE_CODES["transition"], STATE_CODES["guitar_absent"], STATE_CODES["calibration_unstable"]):
            roles, confidence, _, _ = tracker.update(feature(state=state), .14, 0)
            self.assertFalse(roles.any())
            self.assertFalse(confidence.any())
        confidence = np.ones(6, np.float32)
        confidence[0] = .1
        self.assertFalse(feature(confidence=confidence)["available"].any())
        self.assertFalse(feature(points=np.full((2, 21, 3), np.nan, np.float32))["available"].any())

    def test_ambiguity_and_competing_hands_are_unknown_not_forced_pairs(self):
        tracker = RoleTracker(RoleConfig(confirmation_seconds=0))
        ambiguous = feature()
        ambiguous["scores"][:] = [.8, .75]
        roles, _, _, _ = tracker.update(ambiguous, 0., 0)
        self.assertFalse(roles.any())
        competing = feature()
        competing["scores"][:] = [.95, .01]
        roles, _, _, _ = tracker.update(competing, .1, 0)
        self.assertFalse(roles.any())

    def test_plucking_requires_reviewed_body_geometry(self):
        confidence = np.ones(6, np.float32)
        confidence[4:] = 0
        result = feature(confidence=confidence)
        self.assertEqual(result["bodyConfidence"], 0)
        self.assertFalse(result["scores"][:, 1].any())
        roles, _, _, reason = RoleTracker(RoleConfig(confirmation_seconds=0)).update(result, 0., 0)
        self.assertEqual(roles.tolist(), [1, 0])
        self.assertEqual(reason[1], REASON_CODES["body_geometry_unavailable"])

    def test_outside_landmarks_and_degenerate_geometry_produce_unknown(self):
        bad = hands()
        bad[:, :, :2] = 4
        self.assertFalse(feature(points=bad)["available"].any())
        points = geometry()
        points[0] = points[1]
        self.assertFalse(feature(coordinates=points)["available"].any())

    def test_role_scores_do_not_depend_on_anatomical_metadata(self):
        n = 8
        hand = {
            "pts": np.arange(n, dtype=np.int64) * 40, "shot_id": np.zeros(n, np.int32),
            "image_landmarks": np.tile(hands(), (n, 1, 1, 1)), "handedness": np.zeros((n, 2), np.int8),
            "detection_source": np.ones(n, np.int8),
        }
        geo = {"coordinates": np.tile(geometry(), (n, 1, 1)), "confidence": np.ones((n, 6), np.float32), "state": np.zeros(n, np.int8)}
        shots = {"timeBase": [1, 1000], "width": SIZE[0], "height": SIZE[1]}
        a, _ = assign_role_arrays(hand, geo, shots)
        hand["handedness"][:] = 1
        b, _ = assign_role_arrays(hand, geo, shots)
        np.testing.assert_array_equal(a["role"], b["role"])
        np.testing.assert_array_equal(a["role_confidence"], b["role_confidence"])
        self.assertFalse(np.array_equal(a["anatomical_handedness"], b["anatomical_handedness"]))

    def test_reviewed_screen_order_labels_pairs_without_geometry_or_confirmation(self):
        config = RoleConfig(plucking_screen_side="left")
        tracker = RoleTracker(config)
        for i, state in enumerate((STATE_CODES["trackable"], STATE_CODES["guitar_partial"], STATE_CODES["calibration_unstable"], STATE_CODES["unknown"])):
            points = hands()[::-1].copy() if i % 2 else hands()
            result = feature(points, coordinates=np.full((6, 2), np.nan), confidence=np.zeros(6), state=state, config=config)
            roles, confidence, _, _ = tracker.update(result, i * .04, i)
            self.assertEqual(roles.tolist(), [2, 1] if i % 2 else [1, 2])
            self.assertTrue((confidence > 0).all())
            self.assertTrue(np.isnan(result["coordinates"]).all())
            self.assertFalse(result["scores"].any())
        mirrored = hands()
        mirrored[..., 0] = 1 - mirrored[..., 0]
        config = RoleConfig(plucking_screen_side="right")
        result = feature(mirrored, config=config)
        roles, _, _, _ = RoleTracker(config).update(result, 0, 0)
        self.assertEqual(roles.tolist(), [1, 2])

    def test_screen_order_uses_mean_of_all_visible_landmarks(self):
        points = hands()
        points[0, :, :2] = [.2, .4]
        points[0, [0, 5, 9, 13, 17], 0] = .7
        points[1, :, :2] = [.5, .5]
        config = RoleConfig(plucking_screen_side="left")
        result = feature(points, coordinates=np.full((6, 2), np.nan), confidence=np.zeros(6), config=config)
        self.assertLess(result["imageCenters"][0, 0], result["imageCenters"][1, 0])
        np.testing.assert_allclose(result["imageCenters"], points[..., :2].mean(axis=1))
        roles, _, _, _ = RoleTracker(config).update(result, 0., 0)
        self.assertEqual(roles.tolist(), [2, 1])

    def test_screen_order_uses_visible_hand_even_when_wrist_is_out_of_frame(self):
        points = hands()
        points[:, 0, 1] = 1.1
        config = RoleConfig(plucking_screen_side="left")
        result = feature(points, config=config)
        roles, _, _, _ = RoleTracker(config).update(result, 0, 0)
        self.assertEqual(roles.tolist(), [1, 2])
        self.assertTrue(np.isnan(result["coordinates"]).all())

    def test_screen_order_does_not_fill_absence_duplicate_hands_or_broll(self):
        config = RoleConfig(plucking_screen_side="left")
        tracker = RoleTracker(config)
        tracker.update(feature(config=config), 0, 0)
        for state in (STATE_CODES["transition"], STATE_CODES["guitar_absent"]):
            roles, _, _, _ = tracker.update(feature(state=state, config=config), .04, 0)
            self.assertFalse(roles.any())
        for points in (np.full((2, 21, 3), np.nan), np.tile(hands()[0:1], (2, 1, 1))):
            result = feature(points, coordinates=np.full((6, 2), np.nan), confidence=np.zeros(6), config=config)
            roles, _, _, _ = tracker.update(result, .08, 0)
            self.assertFalse(roles.any())
        single = hands()
        single[0] = np.nan
        roles, _, _, _ = tracker.update(feature(single, config=config), .12, 0)
        self.assertEqual(roles.tolist(), [0, 2])
        result = feature(single, coordinates=np.full((6, 2), np.nan), confidence=np.zeros(6), config=config)
        roles, _, _, _ = tracker.update(result, .16, 0)
        self.assertFalse(roles.any())

    def test_single_hand_geometry_cannot_flip_a_screen_identified_track(self):
        config = RoleConfig(plucking_screen_side="left")
        tracker = RoleTracker(config)
        _, _, ids, _ = tracker.update(feature(config=config), 0, 0)
        single = hands()
        single[1] = np.nan
        conflicting = feature(single, config=config)
        conflicting["scores"][0] = [0, 1]
        for time in (.1, .2):
            roles, _, current_ids, reason = tracker.update(conflicting, time, 0)
            self.assertEqual(current_ids[0], ids[0])
            self.assertEqual(roles.tolist(), [0, 0])
            self.assertEqual(reason[0], REASON_CODES["ambiguous_role"])
        roles, _, _, _ = tracker.update(feature(config=config), .24, 0)
        self.assertEqual(roles.tolist(), [1, 2])
        self.assertEqual(tracker.swap_count, 0)

    def test_configuration_rejects_invalid_thresholds(self):
        for values in ({"ambiguity_margin": 0}, {"confirmation_seconds": -1}, {"minimum_role_score": 2}, {"minimum_geometry_confidence": float("nan")}, {"plucking_screen_side": "up"}):
            with self.assertRaises(EvidenceError):
                RoleConfig(**values)


class RoleInputTests(unittest.TestCase):
    def setUp(self):
        PRIVATE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        temporary = TemporaryDirectory(prefix="role-test-", dir=PRIVATE_OUTPUT_ROOT)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.video = self.root / "source.mkv"
        self.video.write_bytes(b"synthetic-video-identity")
        self.paths = {name: self.root / f"{name}.json" for name in ("shots", "hands", "geometry", "annotations")}
        self.pts = np.arange(8, dtype=np.int64) * 40
        self.ids = np.repeat([0, 1], 4).astype(np.int32)
        self.shots = {
            "schemaVersion": 1, "kind": "video-shot-inspection", "videoSha256": sha256(self.video),
            "timeBase": [1, 1000], "width": SIZE[0], "height": SIZE[1], "frameCount": 8, "shotCount": 2,
            "firstPts": 0, "lastPts": 280,
            "shots": [{"shotId": i, "startPts": i * 160, "endPtsExclusive": (i + 1) * 160} for i in range(2)],
        }
        self.write("shots", self.shots)
        self.annotations = {
            "schemaVersion": 1, "kind": "guitar-geometry-annotations", "videoSha256": sha256(self.video),
            "shotsSha256": sha256(self.paths["shots"]), "pointOrder": list(GEOMETRY_POINTS),
            "coordinateSpace": "normalized_full_frame", "reviewComplete": True,
            "shots": [{
                **row, "keyframePts": row["startPts"], "keyframeImage": "synthetic.jpg", "state": "trackable",
                "reviewNote": "synthetic",
                "geometry": {name: {"x": float(point[0]), "y": float(point[1]), "confidence": 1.} for name, point in zip(GEOMETRY_POINTS, geometry())},
            } for row in self.shots["shots"]],
        }
        self.write("annotations", self.annotations)
        common = {
            "schemaVersion": 1, "videoSha256": sha256(self.video), "shotsSha256": sha256(self.paths["shots"]),
            "timeBase": [1, 1000], "frameCount": 8, "shotCount": 2,
        }
        self.write("hands", {**common, "kind": "fingerstyle-hand-observations", "arrays": "hands.npz"})
        self.write("geometry", {
            **common, "kind": "guitar-geometry-observations", "arrays": "geometry.npz",
            "pointOrder": list(GEOMETRY_POINTS), "stateEncoding": STATE_CODES,
            "annotationsSha256": sha256(self.paths["annotations"]),
        })
        self.hand = {
            "pts": self.pts.copy(), "shot_id": self.ids.copy(), "hand_count": np.full(8, 2, np.int8),
            "image_landmarks": np.tile(hands(), (8, 1, 1, 1)),
            "handedness": np.zeros((8, 2), np.int8), "handedness_score": np.ones((8, 2), np.float32),
            "detection_source": np.ones(8, np.int8),
        }
        self.geo = {
            "pts": self.pts.copy(), "shot_id": self.ids.copy(), "coordinates": np.tile(geometry(), (8, 1, 1)),
            "confidence": np.ones((8, 6), np.float32), "source": np.ones((8, 6), np.int8), "state": np.zeros(8, np.int8),
        }
        self.save_arrays()

    def write(self, name, document):
        self.paths[name].write_text(json.dumps(document), encoding="utf-8")

    def save_arrays(self):
        np.savez_compressed(self.root / "hands.npz", **self.hand)
        np.savez_compressed(self.root / "geometry.npz", **self.geo)

    def load(self):
        return load_role_inputs(self.video, self.paths["shots"], self.paths["hands"], self.paths["geometry"], self.paths["annotations"])

    def test_bound_inputs_and_private_output_roundtrip(self):
        loaded = self.load()
        self.assertIn("handArrays", loaded["hashes"])
        output = self.root / "roles"
        path, report = assign_hand_roles(self.video, self.paths["shots"], self.paths["hands"], self.paths["geometry"], self.paths["annotations"], output)
        self.assertTrue(path.is_file())
        self.assertEqual(report["frameCount"], 8)
        self.assertFalse(report["anatomicalHandednessUsedForRole"])
        self.assertFalse(report["trainingPerformed"])
        self.assertEqual(report["framesWithBothRoles"], 4)
        self.assertEqual(report["trackingResetCount"], 2)
        with np.load(output / "roles.npz", allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["role"][0], [0, 0])
            np.testing.assert_array_equal(arrays["role"][4], [0, 0])
            self.assertFalse(set(arrays["track_id"][3]) & set(arrays["track_id"][4]))
        with self.assertRaises(EvidenceError):
            assign_hand_roles(self.video, self.paths["shots"], self.paths["hands"], self.paths["geometry"], self.paths["annotations"], output)

    def test_pts_and_shot_mismatch_are_rejected_before_fusion(self):
        self.geo["pts"][1] += 1
        self.save_arrays()
        with self.assertRaisesRegex(EvidenceError, "match exactly"):
            self.load()
        self.geo["pts"] = self.pts.copy()
        self.geo["shot_id"][1] = 1
        self.save_arrays()
        with self.assertRaisesRegex(EvidenceError, "match exactly"):
            self.load()

    def test_screen_order_provenance_does_not_claim_available_guitar_coordinates(self):
        self.geo["coordinates"][:] = np.nan
        self.geo["confidence"][:] = 0
        self.save_arrays()
        config = RoleConfig(plucking_screen_side="left")
        path, report = assign_hand_roles(self.video, self.paths["shots"], self.paths["hands"], self.paths["geometry"], self.paths["annotations"], self.root / "screen-roles", config=config)
        self.assertEqual(report["config"]["plucking_screen_side"], "left")
        self.assertEqual(report["framesWithBothRoles"], 8)
        with np.load(path.with_name("roles.npz"), allow_pickle=False) as arrays:
            self.assertTrue((arrays["assignment_source"] == 2).all())
            self.assertTrue(arrays["role_available"].all())
            self.assertTrue(np.isnan(arrays["guitar_landmarks"]).all())

    def test_hash_and_review_mismatch_are_not_silently_accepted(self):
        report = json.loads(self.paths["hands"].read_text())
        report["videoSha256"] = "0" * 64
        self.write("hands", report)
        with self.assertRaisesRegex(EvidenceError, "source video"):
            self.load()
        report["videoSha256"] = sha256(self.video)
        self.write("hands", report)
        self.annotations["reviewComplete"] = False
        self.write("annotations", self.annotations)
        with self.assertRaisesRegex(EvidenceError, "completed reviewed"):
            self.load()

    def test_matching_arrays_still_must_agree_with_shot_intervals(self):
        self.hand["shot_id"][1] = self.geo["shot_id"][1] = 1
        self.save_arrays()
        with self.assertRaisesRegex(EvidenceError, "PTS interval"):
            self.load()

    def test_bad_dtypes_and_inconsistent_hand_presence_are_rejected(self):
        self.hand["pts"] = self.pts.astype(np.float64)
        self.save_arrays()
        with self.assertRaisesRegex(EvidenceError, "requires int64"):
            self.load()
        self.hand["pts"] = self.pts.copy()
        self.hand["hand_count"][0] = 0
        self.save_arrays()
        with self.assertRaisesRegex(EvidenceError, "Hand counts disagree"):
            self.load()

    def test_source_bound_role_array_hash_guard(self):
        self.video.unlink()
        with av.open(str(self.video), "w") as container:
            stream = container.add_stream("ffv1", rate=25)
            stream.width, stream.height, stream.pix_fmt = *SIZE, "yuv420p"
            for i in range(8):
                frame = av.VideoFrame.from_ndarray(np.full((SIZE[1], SIZE[0], 3), i * 20, np.uint8), format="rgb24")
                frame.pts, frame.time_base = i * 40, Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        self.shots["videoSha256"] = sha256(self.video)
        self.write("shots", self.shots)
        for name in ("annotations", "hands", "geometry"):
            document = json.loads(self.paths[name].read_text())
            document.update(videoSha256=sha256(self.video), shotsSha256=sha256(self.paths["shots"]))
            self.write(name, document)
        document = json.loads(self.paths["geometry"].read_text())
        document["annotationsSha256"] = sha256(self.paths["annotations"])
        self.write("geometry", document)
        path, _ = assign_hand_roles(self.video, self.paths["shots"], self.paths["hands"], self.paths["geometry"], self.paths["annotations"], self.root / "roles", config=RoleConfig(plucking_screen_side="left"))
        load_role_observations(self.load(), path)
        with (path.parent / "roles.npz").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(EvidenceError, "arrays differ"):
            load_role_observations(self.load(), path)

    def test_cli_dispatches_role_assignment_with_explicit_inputs(self):
        import cli
        common = ["--video", "v", "--shots", "s", "--hands", "h", "--geometry", "g", "--annotations", "a", "--output-directory", "o"]
        with patch.object(cli, "assign_hand_roles", return_value=(Path("roles.json"), {
            "frameCount": 8, "framesWithFretting": 4, "framesWithPlucking": 4, "withinShotRoleSwapCount": 0,
        })) as assign, patch("builtins.print"):
            self.assertEqual(cli.main(["assign-hand-roles", *common]), 0)
            assign.assert_called_once_with("v", "s", "h", "g", "a", "o", config=RoleConfig())
            assign.reset_mock()
            self.assertEqual(cli.main(["assign-hand-roles", *common, "--plucking-screen-side", "left"]), 0)
            assign.assert_called_once_with("v", "s", "h", "g", "a", "o", config=RoleConfig(plucking_screen_side="left"))


if __name__ == "__main__":
    unittest.main()
