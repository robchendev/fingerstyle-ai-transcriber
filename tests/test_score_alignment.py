from copy import deepcopy
from fractions import Fraction
import math
import unittest

from scripts.score_alignment import AlignmentInputError, ScoreClock, matching_events


def clock_fixture():
    labels = {
        "measureVisits": [
            {"visitIndex": 0, "measureIndex": 0, "onsetQuarter": [0, 1]},
            {"visitIndex": 1, "measureIndex": 1, "onsetQuarter": [3, 1]},
        ],
        "conditioning": {"providedTiming": {"timeSignature": [6, 8]}},
    }
    report = {
        "durationQuarter": [6, 1],
        "normalizedTempoEvents": [
            {"measureIndex": 0, "positionRatio": [0, 1], "offsetQuarter": [0, 1], "bpm": 40, "beatUnit": [3, 8], "quarterBpm": [60, 1], "linear": False},
        ],
    }
    return labels, report


class ScoreAlignmentTests(unittest.TestCase):
    def test_dotted_unit_and_short_measures_use_score_coordinates(self):
        labels, report = clock_fixture()
        clock = ScoreClock(labels, report)
        self.assertEqual(clock.duration_seconds, 6)
        self.assertEqual(clock.seconds(Fraction(3, 2)), 1.5)
        labels["measureVisits"][1]["onsetQuarter"] = [1, 1]
        report["durationQuarter"] = [4, 1]
        self.assertEqual(ScoreClock(labels, report).duration_seconds, 4)

    def test_tempo_change_and_reference_clock_inverse(self):
        labels, report = clock_fixture()
        report["normalizedTempoEvents"].append({"measureIndex": 1, "positionRatio": [0, 1], "offsetQuarter": [0, 1], "bpm": 120, "beatUnit": [1, 4], "quarterBpm": [120, 1], "linear": False})
        clock = ScoreClock(labels, report)
        self.assertEqual(clock.duration_seconds, 4.5)
        self.assertEqual(clock.seconds(3), 3)
        self.assertEqual(clock.seconds(4), 3.5)
        for quarter in (0, .5, 2.9, 3, 4.5, 6):
            self.assertAlmostEqual(clock.quarter_at(clock.seconds(quarter)), quarter)

    def test_exact_score_end_uses_the_same_boundary_as_reference_duration(self):
        labels, report = clock_fixture()
        report["normalizedTempoEvents"].append({"measureIndex": 1, "positionRatio": [3, 5], "offsetQuarter": [9, 5], "bpm": 90, "beatUnit": [1, 4], "quarterBpm": [90, 1], "linear": False})
        clock = ScoreClock(labels, report)
        self.assertEqual(clock.seconds(clock.total_quarter), clock.duration_seconds)
        self.assertLess(clock.seconds(Fraction(599, 100)), clock.duration_seconds)

    def test_nominal_ramp_integrates_and_inverts_in_score_position(self):
        labels, report = clock_fixture()
        report["normalizedTempoEvents"][0]["linear"] = True
        report["normalizedTempoEvents"].append({"measureIndex": 1, "positionRatio": [1, 1], "offsetQuarter": [3, 1], "bpm": 120, "beatUnit": [1, 4], "quarterBpm": [120, 1], "linear": False})
        clock = ScoreClock(labels, report)
        self.assertAlmostEqual(clock.duration_seconds, 6 * math.log(2))
        for quarter in (0, 1, 3, 5, 6):
            self.assertAlmostEqual(clock.quarter_at(clock.seconds(quarter)), quarter)
        report["normalizedTempoEvents"][-1].update(bpm=30, quarterBpm=[30, 1])
        clock = ScoreClock(labels, report)
        self.assertAlmostEqual(clock.duration_seconds, 12 * math.log(2))
        self.assertAlmostEqual(clock.quarter_at(clock.seconds(5)), 5)

    def test_bad_clock_coordinates_and_missing_ramp_endpoint_fail(self):
        for field, value in (("offsetQuarter", [1, 1]), ("quarterBpm", [40, 1]), ("linear", True)):
            labels, report = clock_fixture()
            report["normalizedTempoEvents"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                ScoreClock(labels, report)
        labels, report = clock_fixture()
        clock = ScoreClock(labels, report)
        for value in (-1, 7, float("nan")):
            with self.subTest(value=value), self.assertRaises(AlignmentInputError):
                clock.seconds(value)
            with self.subTest(inverse=value), self.assertRaises(AlignmentInputError):
                clock.quarter_at(value)

    def test_matching_cues_do_not_turn_unknowns_into_timed_labels(self):
        labels, report = clock_fixture()
        note = {"id": "n", "onsetQuarter": [0, 1], "isAttack": True, "soundingPitchMidi": 36, "notatedDurationQuarter": None, "labelMask": {"pitch": True, "notatedDuration": False}, "sourceSegments": [{"graceMode": None}]}
        grace = {**deepcopy(note), "id": "grace", "sourceSegments": [{"graceMode": "BeforeBeat"}]}
        labels["targets"] = {
            "notes": [note, grace],
            "gestures": [
                {"id": "g", "technique": "percussive_hit", "onsetQuarter": [1, 1], "scoreOnsetKnown": True, "graceMode": None},
                {"id": "unknown", "technique": "percussive_hit", "onsetQuarter": [2, 1], "scoreOnsetKnown": False, "graceMode": None},
            ],
        }
        before = deepcopy(labels)
        events, omissions = matching_events(labels, ScoreClock(labels, report))
        self.assertEqual(events, [{"onset": 0.0, "end": None, "pitch": 36, "percussive": False}, {"onset": 1.0, "end": None, "pitch": None, "percussive": True}])
        self.assertEqual(omissions, {"duration_uses_transient_matching_kernel": 1, "unknown_gesture_onset": 1, "unknown_or_grace_attack": 1})
        self.assertEqual(labels, before)
        labels["targets"] = {"notes": [grace], "gestures": []}
        with self.assertRaisesRegex(AlignmentInputError, "No resolved"):
            matching_events(labels, ScoreClock(labels, report))


if __name__ == "__main__":
    unittest.main()
