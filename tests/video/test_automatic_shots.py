from copy import deepcopy
from fractions import Fraction
import json
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import av
import cv2
import numpy as np

from core import EvidenceError, PRIVATE_OUTPUT_ROOT, sha256
from shot_inspector import (
    ShotConfig, _frame_end, frame_timeline_sha256, inspect_shots, prepare_automatic_shots,
    _merge_intervals, select_shots, transition_evidence, validate_frame_timeline,
)


class AutomaticShotTests(unittest.TestCase):
    def setUp(self):
        self.root = PRIVATE_OUTPUT_ROOT / f"automatic-shot-test-{uuid4().hex}"
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.video = self.root / "source.mkv"
        self.native_pts = [0, 40, 100, 180, 220, 300, 340, 440, 480, 520, 640, 700]
        with av.open(str(self.video), "w") as container:
            stream = container.add_stream("ffv1", rate=1000)
            stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
            stream.width, stream.height, stream.pix_fmt = 80, 64, "yuv420p"
            for pts in self.native_pts:
                frame = av.VideoFrame.from_ndarray(np.zeros((64, 80, 3), np.uint8), format="rgb24")
                frame.pts, frame.time_base = pts, Fraction(1, 1000)
                for packet in stream.encode(frame):
                    packet.duration = 23
                    container.mux(packet)
            for packet in stream.encode():
                packet.duration = 23
                container.mux(packet)
        self.path, self.report = inspect_shots(self.video, self.root / "inspection", ShotConfig(analysis_width=80))

    def save(self, report):
        self.path.write_text(json.dumps(report), encoding="utf-8")

    def add_candidates(self):
        self.report["lowContrastCutCandidates"] = [{"pts": 180, "seconds": .18, "score": .23, "accepted": False}]
        self.report["dissolveCandidates"] = [{
            "pts": 480, "seconds": .48, "startSeconds": .44, "endSeconds": .52,
            "endpointScore": .15, "reviewRequired": True,
        }]
        self.save(self.report)

    def accepted_candidates(self):
        return patch("shot_inspector.transition_evidence", side_effect=lambda a, b, c, kind: {
            "disposition": "accepted-hard-cut" if kind == "low-contrast-cut" else "accepted-dissolve",
            "reason": "synthetic-confirmation", "evidence": {},
        })

    def decode_counter(self):
        counter = {"frames": 0, "opens": 0, "seeks": 0}
        original_open = av.open

        class CountedContainer:
            def __init__(self, container):
                self.container = container

            def __getattr__(self, name):
                return getattr(self.container, name)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.container.close()

            def seek(self, *args, **kwargs):
                counter["seeks"] += 1
                return self.container.seek(*args, **kwargs)

            def decode(self, *args, **kwargs):
                for frame in self.container.decode(*args, **kwargs):
                    counter["frames"] += 1
                    yield frame

        def open_counted(*args, **kwargs):
            counter["opens"] += 1
            return CountedContainer(original_open(*args, **kwargs))

        return counter, patch("shot_inspector.av.open", side_effect=open_counted)

    def test_inspection_caches_native_variable_cadence_and_thumbnails_in_one_pass(self):
        counter, counted = self.decode_counter()
        with counted:
            path, report = inspect_shots(self.video, self.root / "single-pass", ShotConfig(analysis_width=80))
        self.assertEqual(counter, {"frames": 12, "opens": 1, "seeks": 0})
        self.assertEqual(report["framePts"], self.native_pts)
        with av.open(str(self.video)) as container:
            stream = container.streams.video[0]
            final = list(container.decode(stream))[-1]
            expected_end, _ = _frame_end(final, stream)
        self.assertEqual(report["frameEndPtsExclusive"], expected_end)
        self.assertGreater(expected_end, report["lastPts"] + 1)
        self.assertEqual(validate_frame_timeline(report), (self.native_pts, expected_end))
        shot = report["shots"][0]
        self.assertEqual(shot["reviewFrameSha256"], sha256(path.parent / shot["reviewFrame"]))

    def test_automatic_splits_candidates_and_masks_transition_with_native_boundaries(self):
        self.add_candidates()
        before = self.path.read_bytes()
        video_hash = sha256(self.video)
        counter, counted = self.decode_counter()
        with counted, self.accepted_candidates():
            path, report = prepare_automatic_shots(self.video, self.path, self.root / "automatic")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(sha256(self.video), video_hash)
        self.assertEqual(report["sourceInspectionSha256"], sha256(self.path))
        self.assertEqual(report["framePts"], self.native_pts)
        self.assertEqual([shot["startPts"] for shot in report["shots"]], [0, 180, 440, 640])
        metadata = report["automaticReview"]
        self.assertEqual(metadata["method"], "conservative-cut-boundaries-v1")
        self.assertFalse(metadata["reviewRequired"])
        self.assertFalse(metadata["manualReviewPerformed"])
        self.assertEqual(metadata["timelineDecodedFrameCount"], 0)
        self.assertEqual(metadata["addedBoundaryCount"], 3)
        self.assertEqual(metadata["uncertainIntervals"], [
            {"startPts": 440, "endPtsExclusive": 640, "reason": "dissolve"},
        ])
        self.assertEqual(counter["seeks"], 1)
        self.assertEqual(counter["frames"], metadata["candidateEvidenceDecodedFrameCount"])
        self.assertEqual(metadata["thumbnailDecodedFrameCount"], 0)
        self.assertLess(counter["frames"], len(self.native_pts))
        for shot in report["shots"]:
            self.assertEqual(shot["reviewFrameSha256"], sha256(path.parent / shot["reviewFrame"]))
            self.assertEqual(shot["reviewState"], "unknown")
        validate_frame_timeline(report)

    def test_cached_unchanged_inspection_needs_no_video_decode(self):
        with patch("shot_inspector.av.open", side_effect=AssertionError("unexpected decode")):
            path, report = prepare_automatic_shots(self.video, self.path, self.root / "automatic")
            repeated_path, repeated = prepare_automatic_shots(self.video, path, self.root / "repeat")
        self.assertEqual(report["automaticReview"]["thumbnailDecodedFrameCount"], 0)
        self.assertEqual((repeated_path, repeated), (path, report))
        self.assertFalse((self.root / "repeat").exists())
        with self.assertRaises(EvidenceError):
            prepare_automatic_shots(self.video, self.path, self.root / "automatic")

    def test_legacy_sentinel_decodes_once_and_recovers_actual_final_duration(self):
        expected_end = self.report["frameEndPtsExclusive"]
        for key in ("framePts", "frameEndPtsExclusive", "frameTimelineSha256", "frameTimelineProvenance"):
            self.report.pop(key)
        self.report["shots"][-1]["endPtsExclusive"] = self.report["lastPts"] + 1
        self.save(self.report)
        self.assertIsNone(validate_frame_timeline(self.report))
        counter, counted = self.decode_counter()
        with counted:
            _, report = prepare_automatic_shots(self.video, self.path, self.root / "automatic")
        self.assertEqual(counter["frames"], 12)
        self.assertEqual(report["framePts"], self.native_pts)
        self.assertEqual(report["frameEndPtsExclusive"], expected_end)
        self.assertEqual(report["automaticReview"]["timelineDecodedFrameCount"], 12)
        self.assertEqual(report["automaticReview"]["thumbnailDecodedFrameCount"], 0)

    def test_legacy_bounded_inspection_preserves_partial_source_end(self):
        self.report.update(firstPts=180, lastPts=300, frameCount=3)
        self.report["shots"] = [{"shotId": 0, "startPts": 180, "endPtsExclusive": 315}]
        for key in ("framePts", "frameEndPtsExclusive", "frameTimelineSha256", "frameTimelineProvenance"):
            self.report.pop(key)
        self.save(self.report)
        _, report = prepare_automatic_shots(self.video, self.path, self.root / "automatic")
        self.assertEqual(validate_frame_timeline(report), ([180, 220, 300], 315))
        self.assertLess(report["automaticReview"]["timelineDecodedFrameCount"], 12)

    def test_cached_partial_end_stays_an_interval_end_not_an_invented_frame(self):
        self.report.update(lastPts=300, frameCount=6, framePts=self.native_pts[:6], frameEndPtsExclusive=315)
        self.report["shots"][0]["endPtsExclusive"] = 315
        self.report["frameTimelineSha256"] = frame_timeline_sha256(self.report)
        self.report["dissolveCandidates"] = [{
            "pts": 220, "startPts": 180, "endPts": 300, "endpointScore": .15,
        }]
        self.save(self.report)
        with self.accepted_candidates():
            _, report = prepare_automatic_shots(self.video, self.path, self.root / "automatic")
        self.assertEqual(validate_frame_timeline(report), (self.native_pts[:6], 315))
        self.assertNotIn(315, [row["startPts"] for row in report["shots"]])
        self.assertEqual(report["automaticReview"]["uncertainIntervals"], [
            {"startPts": 180, "endPtsExclusive": 315, "reason": "dissolve"},
        ])

    def test_manual_reviewed_cuts_remain_unchanged_and_preferred(self):
        self.add_candidates()
        self.report["shots"][0]["boundaryProvenance"] = "reviewed"
        self.save(self.report)
        before = self.path.read_bytes()
        with patch("shot_inspector.av.open", side_effect=AssertionError("unexpected decode")):
            path, report = prepare_automatic_shots(self.video, self.path, self.root / "automatic")
        self.assertEqual((path, report), (self.path, self.report))
        self.assertNotIn("automaticReview", report)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse((self.root / "automatic").exists())

    def test_selection_scopes_timeline_counts_masks_and_copies_thumbnails(self):
        self.add_candidates()
        with self.accepted_candidates():
            automatic_path, automatic = prepare_automatic_shots(self.video, self.path, self.root / "automatic")
        with patch("shot_inspector.av.open", side_effect=AssertionError("unexpected decode")):
            path, selected = select_shots(self.video, automatic_path, 2, 3, self.root / "selected")
        self.assertEqual(validate_frame_timeline(selected), ([180, 220, 300, 340, 440, 480, 520], 640))
        self.assertEqual(selected["sourceFrameCount"], 12)
        self.assertEqual(selected["frameCount"], 7)
        self.assertEqual(selected["sourceShotCount"], automatic["shotCount"])
        self.assertEqual([shot["sourceShotId"] for shot in selected["shots"]], [1, 2])
        self.assertEqual(selected["automaticReview"]["uncertainIntervals"], [
            {"startPts": 440, "endPtsExclusive": 640, "reason": "dissolve"},
        ])
        for shot in selected["shots"]:
            self.assertEqual(sha256(path.parent / shot["reviewFrame"]), shot["reviewFrameSha256"])

    def test_invalid_cached_timeline_is_rejected_instead_of_silently_decoded(self):
        changes = [
            {"framePts": [0, 0, *self.native_pts[2:]]},
            {"framePts": [False, *self.native_pts[1:]]},
            {"framePts": [0, 100, 40, *self.native_pts[3:]]},
            {"frameEndPtsExclusive": 700},
            {"frameEndPtsExclusive": 999},
            {"frameCount": 11}, {"firstPts": 1}, {"lastPts": 699},
            {"timeBase": [0, 1000]}, {"frameTimelineSha256": "f" * 64},
        ]
        for change in changes:
            with self.subTest(change=change):
                report = {**deepcopy(self.report), **change}
                with self.assertRaises(EvidenceError):
                    validate_frame_timeline(report)
        report = deepcopy(self.report)
        del report["framePts"]
        with self.assertRaises(EvidenceError):
            validate_frame_timeline(report)
        report = deepcopy(self.report)
        report["shots"][0]["startPts"] = 1
        with self.assertRaises(EvidenceError):
            validate_frame_timeline(report)

    def test_nonnative_candidate_and_changed_thumbnail_are_rejected(self):
        self.add_candidates()
        self.report["lowContrastCutCandidates"][0]["pts"] = 181
        self.save(self.report)
        with self.assertRaisesRegex(EvidenceError, "native"):
            prepare_automatic_shots(self.video, self.path, self.root / "invalid")
        self.report["lowContrastCutCandidates"] = []
        self.report["dissolveCandidates"] = []
        self.save(self.report)
        (self.path.parent / self.report["shots"][0]["reviewFrame"]).write_bytes(b"changed")
        with self.assertRaisesRegex(EvidenceError, "thumbnail hash"):
            prepare_automatic_shots(self.video, self.path, self.root / "invalid-thumbnail")
        self.assertFalse((self.root / "invalid-thumbnail").exists())

    def test_frame_end_uses_duration_or_source_stream_never_nominal_rate(self):
        stream = SimpleNamespace(time_base=Fraction(1, 1000), duration=910, start_time=10)
        frame = SimpleNamespace(pts=800, duration=73, time_base=Fraction(1, 1000))
        self.assertEqual(_frame_end(frame, stream), (873, "decoded-final-frame-duration"))
        frame.duration = 0
        self.assertEqual(_frame_end(frame, stream), (920, "video-stream-duration"))
        stream.duration = None
        with self.assertRaisesRegex(EvidenceError, "trustworthy"):
            _frame_end(frame, stream)

    def test_strong_cut_decisions_stay_the_same_without_a_thumbnail_second_pass(self):
        scores = [(value, value, value) for value in (0, 0, 0, .5, 0, 0, 0, .6, 0, 0, 0)]
        counter, counted = self.decode_counter()
        with counted, patch("shot_inspector.cut_components", side_effect=scores):
            _, report = inspect_shots(self.video, self.root / "strong-cuts", ShotConfig(analysis_width=80))
        self.assertEqual([row["startPts"] for row in report["shots"]], [0, 220, 480])
        self.assertEqual([row["boundaryProvenance"] for row in report["shots"]], ["stream", "automatic", "automatic"])
        self.assertEqual(counter["frames"], 12)
        self.assertEqual(counter["opens"], 1)
        validate_frame_timeline(report)

    def test_inspection_rejects_duplicate_or_decreasing_decoded_pts(self):
        with av.open(str(self.video)) as container:
            frames = list(container.decode(container.streams.video[0]))
        stream = SimpleNamespace(
            type="video", index=0, time_base=Fraction(1, 1000), average_rate=Fraction(25),
            disposition=av.stream.Disposition(0),
            codec_context=SimpleNamespace(name="test", width=80, height=64),
        )
        for index, invalid in enumerate(([frames[0], frames[0]], [frames[1], frames[0]])):
            fake = SimpleNamespace(streams=SimpleNamespace(video=[stream]), decode=lambda _: iter(invalid), close=lambda: None)
            output = self.root / f"invalid-pts-{index}"
            with patch("shot_inspector.av.open", return_value=fake), self.assertRaisesRegex(EvidenceError, "strictly increasing"):
                inspect_shots(self.video, output, ShotConfig(analysis_width=80))
            self.assertFalse(output.exists())

    def test_reviewed_addition_seeks_only_its_missing_native_preview(self):
        review = self.root / "review.json"
        review.write_text(json.dumps({
            "schemaVersion": 1, "kind": "video-shot-boundary-review",
            "videoSha256": sha256(self.video), "addBoundaries": [{"seconds": .31, "type": "hard_cut"}],
        }), encoding="utf-8")
        counter, counted = self.decode_counter()
        with counted:
            path, report = inspect_shots(self.video, self.root / "reviewed", ShotConfig(analysis_width=80), review)
        self.assertEqual([row["startPts"] for row in report["shots"]], [0, 300])
        self.assertEqual(report["shots"][1]["boundaryProvenance"], "reviewed")
        self.assertEqual(counter["seeks"], 1)
        self.assertLess(counter["frames"], 24)
        self.assertEqual(sha256(path.parent / report["shots"][1]["reviewFrame"]), report["shots"][1]["reviewFrameSha256"])

    def test_spatial_motion_is_not_a_dissolve_but_independent_crossfade_is(self):
        rng = np.random.default_rng(41)
        def scene():
            return cv2.resize(rng.integers(0, 256, (45, 80), dtype=np.uint8), (320, 180), interpolation=cv2.INTER_CUBIC)
        start, end = scene(), scene()
        moved = lambda distance: cv2.warpAffine(start, np.float32([[1, 0, distance], [0, 1, 0]]),
                                               (320, 180), borderMode=cv2.BORDER_REFLECT)
        motion = transition_evidence(start, moved(4), moved(8), "dissolve")
        self.assertEqual(motion["disposition"], "coherent-motion")
        blend = ((start.astype(np.float32) + end) / 2).astype(np.uint8)
        dissolve = transition_evidence(start, blend, end, "dissolve")
        self.assertEqual(dissolve["disposition"], "accepted-dissolve")
        self.assertGreater(dissolve["evidence"]["blendImprovement"], .95)
        self.assertEqual(transition_evidence(start, end, end, "low-contrast-cut")["disposition"], "accepted-hard-cut")

    def test_textureless_weak_candidates_remain_unmasked_diagnostics(self):
        self.add_candidates()
        _, report = prepare_automatic_shots(self.video, self.path, self.root / "automatic")
        review = report["automaticReview"]
        self.assertEqual(report["shotCount"], 1)
        self.assertEqual(review["uncertainIntervals"], [])
        self.assertEqual(review["unresolvedCandidateCount"], 2)
        self.assertEqual(len(review["diagnosticIntervals"]), 2)
        self.assertTrue(all(row["disposition"] == "uncertain" for row in review["candidateDispositions"]))

    def test_supported_overlapping_dissolves_merge_without_midpoint_resets(self):
        self.add_candidates()
        self.report["lowContrastCutCandidates"] = []
        self.report["dissolveCandidates"].append({
            "pts": 520, "startPts": 480, "endPts": 640, "endpointScore": .16,
        })
        self.save(self.report)
        with self.accepted_candidates():
            _, report = prepare_automatic_shots(self.video, self.path, self.root / "automatic")
        self.assertEqual([row["startPts"] for row in report["shots"]], [0, 440, 700])
        self.assertEqual(report["automaticReview"]["uncertainIntervals"], [
            {"startPts": 440, "endPtsExclusive": 700, "reason": "dissolve"},
        ])
        self.assertEqual(report["automaticReview"]["acceptedDissolveCount"], 2)
        self.assertEqual(_merge_intervals([
            {"startPts": 0, "endPtsExclusive": 20, "reason": "a"},
            {"startPts": 10, "endPtsExclusive": 30, "reason": "b"},
        ]), [{"startPts": 0, "endPtsExclusive": 30, "reason": "a;b"}])

    def test_legacy_candidate_evidence_and_previews_share_the_timeline_decode(self):
        self.add_candidates()
        for key in ("framePts", "frameEndPtsExclusive", "frameTimelineSha256", "frameTimelineProvenance"):
            self.report.pop(key)
        self.report["shots"][-1]["endPtsExclusive"] = 701
        self.save(self.report)
        counter, counted = self.decode_counter()
        with counted, self.accepted_candidates():
            _, report = prepare_automatic_shots(self.video, self.path, self.root / "automatic")
        self.assertEqual(counter["frames"], 12)
        self.assertEqual(report["automaticReview"]["candidateEvidenceDecodedFrameCount"], 0)
        self.assertEqual(report["automaticReview"]["thumbnailDecodedFrameCount"], 0)


if __name__ == "__main__":
    unittest.main()
