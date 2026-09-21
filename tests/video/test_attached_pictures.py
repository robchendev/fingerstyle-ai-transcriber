from importlib.util import find_spec
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

if any(find_spec(name) is None for name in ("av", "cv2", "mediapipe")):
    raise unittest.SkipTest("Run video tests with the isolated vision environment.")

import av
import numpy as np
from PIL import Image

from core import EvidenceError, PRIVATE_OUTPUT_ROOT, sha256, video_stream
from batch_video import _select_interval, _timeline
from hand_tracking import track_hands
from shot_inspector import ShotConfig, _video_stream, inspect_shots, select_shots
from scripts.audio_tools import executable


class AttachedPictureTests(unittest.TestCase):
    def test_stream_selection_ignores_artwork_without_guessing_between_video_tracks(self):
        def stream(index, artwork=False):
            disposition = av.stream.Disposition.default
            if artwork:
                disposition |= av.stream.Disposition.attached_pic
            return SimpleNamespace(index=index, disposition=disposition)
        motion, other, cover = stream(3), stream(7), stream(0, True)
        for values in ([motion], [cover, motion], [motion, cover]):
            self.assertIs(video_stream(SimpleNamespace(streams=SimpleNamespace(video=values))), motion)
        for values in ([], [cover], [motion, other], [cover, motion, other]):
            with self.subTest(count=len(values)), self.assertRaisesRegex(EvidenceError, "Exactly one video stream"):
                video_stream(SimpleNamespace(streams=SimpleNamespace(video=values)))

    def test_mp4_cover_does_not_enter_shot_timeline_or_hand_tracking(self):
        PRIVATE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="cover-stream-test-", dir=PRIVATE_OUTPUT_ROOT) as directory:
            root = Path(directory)
            cover, media = root / "cover.png", root / "performance.mp4"
            Image.new("RGB", (64, 48), "red").save(cover)
            subprocess.run([
                executable("ffmpeg"), "-nostdin", "-v", "error", "-n",
                "-f", "lavfi", "-i", "color=c=blue:s=160x120:r=10:d=1",
                "-i", str(cover), "-map", "0:v:0", "-map", "1:v:0",
                "-c:v:0", "libx264", "-threads:v:0", "1", "-c:v:1", "copy",
                "-disposition:v:1", "attached_pic", str(media),
            ], check=True, capture_output=True)
            original_hash = sha256(media)
            with av.open(str(media)) as container:
                self.assertEqual(len(container.streams.video), 2)
                expected = video_stream(container)
                expected_index = expected.index
                expected_pts = [frame.pts for frame in container.decode(expected)]
            path, report = inspect_shots(media, root / "shots", ShotConfig(analysis_width=160))
            self.assertEqual(report["videoStreamIndex"], expected_index)
            self.assertEqual(report["framePts"], expected_pts)
            self.assertEqual(report["frameCount"], 10)
            with av.open(str(media)) as container:
                self.assertEqual(_video_stream(container, report).index, expected_index)
                with self.assertRaisesRegex(EvidenceError, "differ"):
                    _video_stream(container, {**report, "videoStreamIndex": 1})
            timeline = _timeline(media, path, root / "timeline")
            document = json.loads(timeline.read_text())
            self.assertEqual(document["pts"], expected_pts)
            clips = [{"startPts": expected_pts[0], "endPtsExclusive": document["endPtsExclusive"]}]
            interval_path = _select_interval(media, path, timeline, clips, root / "interval")
            self.assertEqual(json.loads(interval_path.read_text())["framePts"], expected_pts)
            fallback = dict(report)
            fallback.pop("framePts")
            fallback.pop("frameEndPtsExclusive")
            fallback_path = root / "uncached-shots.json"
            fallback_path.write_text(json.dumps(fallback), encoding="utf-8")
            fallback_timeline = _timeline(media, fallback_path, root / "uncached-timeline")
            self.assertEqual(json.loads(fallback_timeline.read_text())["pts"], expected_pts)
            selected_path, _ = select_shots(media, path, 1, 1, root / "selection")
            model = root / "synthetic.task"
            model.write_bytes(b"tracker is mocked, video decoding is real")
            observed = []
            tracker = SimpleNamespace(
                close=lambda: None,
                detect_for_video=lambda image, timestamp: observed.append((image.width, image.height, timestamp)),
            )
            observations = (0, np.full((2, 21, 3), np.nan, np.float32),
                            np.full((2, 21, 3), np.nan, np.float32),
                            np.full(2, -1, np.int8), np.zeros(2, np.float32))
            with patch("hand_tracking._hand_landmarker", return_value=tracker), patch("hand_tracking._observations", return_value=observations):
                _, tracked = track_hands(media, selected_path, model, root / "hands")
            self.assertEqual(tracked["frameCount"], 10)
            self.assertEqual([value[2] for value in observed], list(range(0, 1000, 100)))
            self.assertTrue(all(value[:2] == (160, 120) for value in observed))
            self.assertEqual(sha256(media), original_hash)
