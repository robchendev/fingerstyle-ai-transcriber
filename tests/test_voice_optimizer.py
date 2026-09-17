from copy import deepcopy
import unittest

from scripts.transcriber_audio import HarnessError
from scripts.voice_optimizer import optimize_voices


def note(onset, duration, pitch, voice):
    return {
        "scoreOnsetQuarter": onset,
        "scoreDurationQuarter": duration,
        "soundingPitchMidi": pitch,
        "voiceIndex": voice,
    }


class VoiceOptimizerTests(unittest.TestCase):
    def test_overlapping_new_attack_moves_to_other_voice_and_reduces_fragments(self):
        document = {
            "notes": [
                note([0, 1], [1, 1], 60, 0),
                note([1, 2], [1, 2], 64, 0),
            ],
        }
        before = deepcopy(document)
        result, report = optimize_voices(document)
        self.assertEqual(document, before)
        self.assertEqual({value["voiceIndex"] for value in result["notes"]}, {0, 1})
        self.assertLess(report["estimatedSegmentsAfter"], report["estimatedSegmentsBefore"])
        self.assertFalse(report["rawHypothesesModified"])

    def test_same_chord_duration_prefers_one_voice(self):
        document = {
            "notes": [
                note([0, 1], [1, 1], 60, 0),
                note([0, 1], [1, 1], 64, 1),
            ],
        }
        result, _ = optimize_voices(document)
        self.assertEqual(len({value["voiceIndex"] for value in result["notes"]}), 1)

    def test_invalid_intervals_fail(self):
        with self.assertRaises(HarnessError):
            optimize_voices({"notes": [note([0, 1], [0, 1], 60, 0)]})


if __name__ == "__main__":
    unittest.main()
