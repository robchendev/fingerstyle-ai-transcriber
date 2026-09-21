import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from scripts.transcriber_audio import HarnessError
from scripts.transcriber_events import OutputTimeline, checkpoint_event_score, evaluate_events, event_counts, match_events, summarize_counts
from tests.test_score_alignment import clock_fixture


class TimeModel(torch.nn.Module):
    def forward(self, features, conditioning, lengths):
        batch, count, _ = features.shape
        outputs = {
            "note_onset_logits": torch.full((batch, count, 6), -12.),
            "fret_logits": torch.zeros(batch, count, 6, 37),
            "pitch_logits": torch.zeros(batch, count, 6, 128),
            "voice_logits": torch.zeros(batch, count, 6, 4),
            "duration_log": torch.full((batch, count, 6), math.log(2)),
            "harmonic_logits": torch.full((batch, count, 6), -12.),
            "harmonic_kind_logits": torch.zeros(batch, count, 6, 4),
            "harmonic_node_logits": torch.zeros(batch, count, 6, 6),
            "percussion_logits": torch.full((batch, count, 3), -12.),
        }
        outputs["note_onset_logits"][:, :, 0] = 12 - 100 * (features[:, :, 0] - 3).abs()
        outputs["pitch_logits"][:, :, :, 40] = 5
        outputs["percussion_logits"][:, :, 0] = 12 - 100 * (features[:, :, 0] - 3).abs()
        outputs["percussion_logits"][:, :, 1] = 12 - 100 * (features[:, :, 0] - 4).abs()
        return outputs


class WindowFixture:
    def __init__(self, complete):
        labels, normalization = clock_fixture()
        labels["conditioning"]["instrument"] = {"openStringMidi": [40, 45, 50, 55, 59, 64], "capoFret": 0}
        labels["conditioning"]["providedTiming"]["sourceTimeSignatureChanges"] = []
        labels["targets"] = {
            "notes": [{
                "id": "n", "voiceIndex": 0, "string": 6, "fret": 0, "soundingPitchMidi": 40,
                "onsetQuarter": [3, 1], "notatedDurationQuarter": [1, 1], "isAttack": True,
                "labelMask": {"attack": True, "pitch": True, "fingering": True, "notatedDuration": True},
                "sourceSegments": [{"graceMode": None}],
            }],
            "gestures": [{"id": "g", "voiceIndex": 0, "onsetQuarter": [3, 1], "technique": "wrist_thump", "scoreOnsetKnown": True, "graceMode": None}],
        }
        self.feature_config = SimpleNamespace(sample_rate=100, hop_length=2, hop_seconds=.02)
        self.records = [{
            "data": SimpleNamespace(labels=labels, normalization=normalization),
            "row": {"id": "synthetic", "sampleRate": 100},
            "candidate": {"denseMapping": [{"clipSeconds": float(time), "referenceSeconds": float(time), "scoreQuarter": float(time)} for time in (0, 3, 6)]},
            "negativeAllowed": [True] * 6, "percussionAnnotationsComplete": complete,
        }]
        self.windows = []
        for start, stop in ((0, 400), (200, 600)):
            targets = {"negativePercussionSupervision": complete}
            if complete:
                targets["percussionAnnotationCoverage"] = [[0., 4.]]
            self.windows.append((self.records[0], {"startSample": start, "stopSampleExclusive": stop, "targets": targets}))

    def check_unchanged(self):
        pass

    def _features(self, record, window):
        times = np.arange((window["stopSampleExclusive"] - window["startSample"]) // 2) * .02
        return torch.tensor((times + window["startSample"] / 100)[:, None], dtype=torch.float32), times


class EventEvaluationTests(unittest.TestCase):
    def test_joint_event_scoring_is_limited_to_prepared_intervals_not_tracking_availability(self):
        dataset = WindowFixture(True)
        dataset.video_training_intervals = lambda record: [[2.5, 3.5]]
        report = evaluate_events(TimeModel(), dataset, "cpu")
        self.assertEqual(report["recordings"][0]["scoredClipRanges"], [[2.5, 3.5]])
        metrics = report["metricsByToleranceSeconds"]["0.1"]
        self.assertEqual(metrics["wrist_thump"]["true_positive"], 1)
        self.assertEqual(metrics["thumb_slap"]["false_positive"], 0)
        self.assertEqual(metrics["thumb_slap"]["unscorable_predictions"], 1)

    def test_event_evaluation_forwards_coarse_contract_through_missing_context(self):
        from scripts.paired_video import empty_video, validate_video_tensors

        dataset = WindowFixture(True)
        dataset.video_training_intervals = lambda record: [[2.5, 3.5]]
        observed = []

        def video_window(record, times):
            video = empty_video(len(times))
            if times[0] == 0:
                video["structured_available"][0, 0, 98:140] = True
                video["structured_available"][0, 0, 186:188] = True
                video["structured"][0, 0, 186:188] = torch.tensor([-5., -1.])
                video["segment_id"][0, 0] = 0
                video["frame_indices"].zero_()
            validate_video_tensors(video, len(times))
            return video

        class VideoTimeModel(TimeModel):
            def forward(self, features, conditioning, lengths, *, video):
                observed.append(video)
                return super().forward(features, conditioning, lengths)

        dataset.video_window = video_window
        report = evaluate_events(VideoTimeModel(), dataset, "cpu")
        self.assertEqual(len(observed), 2)
        self.assertEqual(observed[0]["structured"].shape, (1, 1, 4, 194))
        self.assertTrue(observed[0]["structured_available"][..., 186:188].any())
        self.assertFalse(observed[1]["structured_available"].any())
        self.assertEqual(report["recordings"][0]["scoredClipRanges"], [[2.5, 3.5]])
        self.assertEqual(report["metricsByToleranceSeconds"]["0.1"]["wrist_thump"]["true_positive"], 1)

    def test_matching_is_one_to_one_maximal_not_nearest_first(self):
        truth = [{"onsetSeconds": time, "string": 6} for time in (0., .1)]
        predictions = [{"onsetSeconds": time, "string": 6} for time in (.06, .16)]
        self.assertEqual(match_events(truth, predictions, .06, ("string",)), [(0, 0), (1, 1)])
        duplicate = event_counts(truth[:1], predictions[:1] * 2, .1, ("string",), lambda _: True, has_negative_coverage=True)
        self.assertEqual((duplicate["true_positive"], duplicate["false_positive"]), (1, 1))
        predictions[0]["string"] = 1
        self.assertEqual(len(match_events(truth, predictions, .06, ("string",))), 1)

    def test_unknown_negatives_do_not_create_precision_or_f1(self):
        counts = event_counts([], [{"onsetSeconds": 1., "technique": "thumb_slap"}], .1, ("technique",), lambda _: False, has_negative_coverage=False)
        metrics = summarize_counts(counts)
        self.assertIsNone(metrics["precision"])
        self.assertIsNone(metrics["f1"])
        self.assertEqual(metrics["unscorable_predictions"], 1)

    def test_v2_checkpoint_score_requires_complete_technique_string_set(self):
        counts = {
            "true_positive": 2, "false_positive": 1, "false_negative": 1,
            "reference_events": 3, "predicted_events": 3, "unscorable_predictions": 0,
            "absolute_onset_error_sum": 0., "has_negative_coverage": True,
        }
        empty = {**counts, "true_positive": 0, "false_positive": 0, "false_negative": 0, "reference_events": 0, "predicted_events": 0}
        metrics = {
            "string_fret_pitch_onset": summarize_counts(counts),
            "wrist_thump": summarize_counts(empty),
            "thumb_slap": summarize_counts(empty),
            "percussive_hit": summarize_counts(empty),
            "technique_string_set": summarize_counts(counts),
            "technique_onset": summarize_counts({**counts, "true_positive": 3, "false_positive": 0, "false_negative": 0}),
        }
        result = checkpoint_event_score({"metricsByToleranceSeconds": {"0.1": metrics}, "windowVisits": 4})
        self.assertEqual(result["score"], 2 / 3)
        self.assertIn("technique-string-set", result["metric"])

    def test_stitching_interpolates_shifted_windows_and_refuses_gaps(self):
        times = np.arange(10) * .1
        combined = OutputTimeline(times)
        for start, stop in ((0., .65), (.35, 1.)):
            local = start + np.arange(7) * .1
            local = local[local < stop]
            combined.add({"linear": torch.tensor(local[:, None], dtype=torch.float32)}, local, stop)
        self.assertTrue(torch.isfinite(combined.finish()["linear"]).all())
        torch.testing.assert_close(combined.finish()["linear"][1:9, 0], torch.tensor(times[1:9], dtype=torch.float32))
        gap = OutputTimeline(times)
        gap.add({"linear": torch.ones(2, 1)}, times[:2], .2)
        with self.assertRaisesRegex(HarnessError, "uncovered"):
            gap.finish()

    def test_overlapping_windows_produce_one_event_and_conditional_percussion_metrics(self):
        for complete in (False, True):
            model = TimeModel()
            report = evaluate_events(model, WindowFixture(complete), "cpu")
            self.assertTrue(model.training)
            metrics = report["metricsByToleranceSeconds"]["0.1"]
            self.assertEqual(metrics["string_pitch_onset"]["reference_events"], 1)
            self.assertEqual(metrics["string_pitch_onset"]["true_positive"], 1)
            self.assertEqual(metrics["string_pitch_onset"]["predicted_events"], 1)
            self.assertEqual(metrics["string_pitch_onset"]["f1"], 1.)
            self.assertEqual(metrics["wrist_thump"]["recall"], 1.)
            self.assertEqual(metrics["thumb_slap"]["false_positive"], int(complete))
            self.assertEqual(len(report["recordings"][0]["predictions"]["percussion"]), 2)
            if not complete:
                self.assertIsNone(metrics["wrist_thump"]["f1"])
            else:
                self.assertEqual(metrics["wrist_thump"]["f1"], 1.)
            selection = checkpoint_event_score(report)
            self.assertEqual(selection["windows"], 2)
            self.assertAlmostEqual(selection["score"], .8 if complete else 1.)

    def test_declared_coverage_outside_scoring_interior_is_not_precision_coverage(self):
        dataset = WindowFixture(True)
        for _, window in dataset.windows:
            window["targets"]["percussionAnnotationCoverage"] = [[0., .1]]
        dataset.windows = dataset.windows[:1]
        report = evaluate_events(TimeModel(), dataset, "cpu")
        self.assertIsNone(report["metricsByToleranceSeconds"]["0.1"]["wrist_thump"]["precision"])


if __name__ == "__main__":
    unittest.main()
