from copy import deepcopy
from types import SimpleNamespace
import unittest

from scripts.training_windows import projected_targets, sample_windows, targets_in_window
from scripts.score_alignment import AlignmentInputError


class TrainingWindowTests(unittest.TestCase):
    def test_window_geometry_is_bounded_with_overlap_and_no_tiny_tail(self):
        self.assertEqual(sample_windows([(0, 1700)], 100), [(0, 800), (600, 1400), (1200, 1700)])
        self.assertEqual(sample_windows([(0, 150)], 100), [])
        with self.assertRaises(AlignmentInputError):
            sample_windows([(0, 100)], 100, {"windowSeconds": 8, "strideSeconds": 9, "minimumWindowSeconds": 2})

    def test_projection_preserves_unknown_grace_and_cross_window_duration_masks(self):
        def note(identifier, onset, duration, grace=None):
            return {"id": identifier, "voiceIndex": 1, "string": 6, "fret": 0, "soundingPitchMidi": 40, "onsetQuarter": [onset, 1], "notatedDurationQuarter": [duration, 1] if duration else None, "isAttack": True, "sourceSegments": [{"graceMode": grace}], "labelMask": {"attack": True, "pitch": True, "fingering": True, "notatedDuration": duration is not None}}
        labels = {"targets": {
            "notes": [note("held", 0, 6), note("attack", 3, 3), note("grace", 4, None, "BeforeBeat")],
            "gestures": [
                {"id": "known", "technique": "percussive_hit", "voiceIndex": 0, "onsetQuarter": [2, 1], "scoreOnsetKnown": True, "graceMode": None, "labelMask": {"gesture": True}},
                {"id": "unknown", "technique": "percussive_hit", "voiceIndex": 0, "onsetQuarter": [4, 1], "scoreOnsetKnown": False, "graceMode": None, "labelMask": {"gesture": True}},
            ],
        }}
        before = deepcopy(labels)
        candidate = {"denseMapping": [{"referenceSeconds": float(i), "clipSeconds": float(i)} for i in range(9)]}
        notes, gestures = projected_targets(labels, candidate, SimpleNamespace(seconds=float))
        result = targets_in_window(notes, gestures, 200, 500, 100)
        held, attack, grace = result["notes"]
        self.assertTrue(held["carryIn"])
        self.assertFalse(held["supervisionMask"]["onset"])
        self.assertTrue(attack["supervisionMask"]["onset"])
        self.assertTrue(attack["supervisionMask"]["pitch"])
        self.assertFalse(attack["supervisionMask"]["notatedDuration"])
        self.assertEqual(attack["notatedDurationQuarter"], [3, 1])
        self.assertFalse(grace["supervisionMask"]["onset"])
        self.assertFalse(grace["supervisionMask"]["pitch"])
        self.assertTrue(result["gestures"][0]["supervisionMask"]["gesture"])
        self.assertFalse(result["gestures"][1]["supervisionMask"]["gesture"])
        self.assertFalse(result["negativePercussionSupervision"])
        self.assertTrue(all(not n["supervisionMask"]["acousticRelease"] for n in result["notes"]))
        self.assertEqual(labels, before)

    def test_projection_does_not_extrapolate_unmapped_notes(self):
        labels = {"targets": {"notes": [], "gestures": [{"id": "before", "technique": "percussive_hit", "voiceIndex": 0, "onsetQuarter": [0, 1], "scoreOnsetKnown": True, "labelMask": {"gesture": True}}]}}
        candidate = {"denseMapping": [{"referenceSeconds": 1., "clipSeconds": 0.}, {"referenceSeconds": 2., "clipSeconds": 1.}]}
        notes, gestures = projected_targets(labels, candidate, SimpleNamespace(seconds=float))
        self.assertIsNone(gestures[0]["proposedOnsetClipSeconds"])
        self.assertEqual(targets_in_window(notes, gestures, 0, 100, 100)["gestures"], [])

    def test_resolved_gesture_membership_is_positive_without_optional_mask(self):
        gesture = {"sourceGestureId": "wrist", "technique": "wrist_thump", "voiceIndex": 0, "onsetQuarter": [1, 1], "proposedOnsetClipSeconds": 1., "onsetTimingKnownInScore": True, "sourceLabelMask": {}}
        self.assertTrue(targets_in_window([], [gesture], 0, 200, 100)["gestures"][0]["supervisionMask"]["gesture"])
        gesture["sourceLabelMask"] = {"gesture": False}
        self.assertFalse(targets_in_window([], [gesture], 0, 200, 100)["gestures"][0]["supervisionMask"]["gesture"])


if __name__ == "__main__":
    unittest.main()
