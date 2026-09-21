from fractions import Fraction
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import av
import numpy as np

from core import EvidenceError, PRIVATE_OUTPUT_ROOT, sha256
from geometry import validate_geometry_annotations
from hand_tracking import track_hands
from shot_inspector import select_shots
from tests.video import test_hand_roles as role_fixtures


class BoundedVideoTests(unittest.TestCase):
    def setUp(self):
        PRIVATE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        temporary = TemporaryDirectory(prefix="bounded-video-test-", dir=PRIVATE_OUTPUT_ROOT)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.video = self.root / "source.mkv"
        with av.open(str(self.video), "w") as container:
            stream = container.add_stream("ffv1", rate=25)
            stream.width, stream.height, stream.pix_fmt = 320, 180, "yuv420p"
            for i in range(12):
                frame = av.VideoFrame.from_ndarray(np.full((180, 320, 3), i * 10, np.uint8), format="rgb24")
                frame.pts, frame.time_base = i * 40, Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        self.shots_path = self.root / "full-shots.json"
        self.shots = {
            "schemaVersion": 1, "kind": "video-shot-inspection", "videoSha256": sha256(self.video),
            "videoStreamIndex": 0, "timeBase": [1, 1000], "width": 320, "height": 180,
            "firstPts": 0, "lastPts": 440, "frameCount": 12, "shotCount": 3,
            "config": {}, "shots": [{"shotId": i, "startPts": 160 * i, "endPtsExclusive": 160 * (i + 1)} for i in range(3)],
        }
        self.shots_path.write_text(json.dumps(self.shots))

    def test_same_pipeline_processes_only_selected_shot_and_preserves_absolute_pts(self):
        selected_path, report = select_shots(self.video, self.shots_path, 2, 2, self.root / "selected")
        self.assertEqual((report["firstPts"], report["lastPts"], report["frameCount"]), (160, 280, 4))
        self.assertEqual(report["shots"][0]["sourceShotId"], 1)
        self.assertEqual(report["sourceInspectionSha256"], sha256(self.shots_path))
        timestamps = []
        tracker = SimpleNamespace(detect_for_video=lambda image, time: (timestamps.append(time) or SimpleNamespace(hand_landmarks=[])), close=lambda: None)
        model = self.root / "hand.task"
        model.write_bytes(b"synthetic-model")
        with patch("hand_tracking._hand_landmarker", return_value=tracker):
            _, hands = track_hands(self.video, selected_path, model, self.root / "hands")
        self.assertEqual(timestamps, [0, 40, 80, 120])
        self.assertEqual(hands["frameCount"], 4)
        with np.load(self.root / "hands" / "hands.npz") as arrays:
            np.testing.assert_array_equal(arrays["pts"], [160, 200, 240, 280])
            np.testing.assert_array_equal(arrays["shot_id"], [0, 0, 0, 0])
        self.assertEqual(sha256(self.video), self.shots["videoSha256"])
        with self.assertRaises(EvidenceError):
            select_shots(self.video, self.shots_path, 2, 2, self.root / "selected")
        with self.assertRaises(EvidenceError):
            select_shots(self.video, self.shots_path, 0, 2, self.root / "invalid")

class AdditionalSeedValidationTests(unittest.TestCase):
    def test_additional_seeds_cannot_cross_shots_or_repeat_pts(self):
        fixture = role_fixtures.RoleInputTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        annotations, shots = fixture.annotations, fixture.shots
        seed = {"keyframePts": 40, "keyframeImage": "seed.jpg", "geometry": {"nut": {"x": .8, "y": .4, "confidence": 1}}, "reviewNote": "synthetic"}
        annotations["shots"][0]["additionalKeyframes"] = [seed]
        rows = validate_geometry_annotations(annotations, shots, shots["videoSha256"], sha256(fixture.paths["shots"]))
        self.assertEqual(rows[0]["additionalKeyframes"][0]["geometry"]["nut"], [.8, .4, 1.])
        for pts in (0, 160):
            seed["keyframePts"] = pts
            with self.assertRaises(EvidenceError):
                validate_geometry_annotations(annotations, shots, shots["videoSha256"], sha256(fixture.paths["shots"]))


if __name__ == "__main__":
    unittest.main()
