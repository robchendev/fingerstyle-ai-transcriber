from copy import deepcopy
import unittest

from scripts.draft_cleanup import DraftProfile, clean_hypotheses
from scripts.transcriber_audio import HarnessError


def note(time, string, confidence, *, harmonic=None, uncertainty=()):
    return {
        "onsetSeconds": time,
        "string": string,
        "fret": 0,
        "soundingPitchMidi": 40,
        "voiceIndex": 0,
        "notatedDurationQuarter": 1,
        "harmonic": harmonic,
        "confidence": confidence,
        "uncertainty": list(uncertainty),
    }


class DraftCleanupTests(unittest.TestCase):
    def test_calibrated_profile_filters_clusters_and_preserves_raw_document(self):
        document = {
            "notes": [
                note(1.0, 6, 0.95),
                note(1.02, 5, 0.94),
                note(1.03, 6, 0.91),
                note(1.06, 6, 0.99),
                note(2.0, 4, 0.89),
            ],
            "percussion": [
                {"onsetSeconds": 1.01, "technique": "thumb_slap", "confidence": 0.7},
                {"onsetSeconds": 1.03, "technique": "thumb_slap", "confidence": 0.65},
                {"onsetSeconds": 3, "technique": "wrist_thump", "confidence": 0.59},
            ],
        }
        before = deepcopy(document)
        cleaned, report = clean_hypotheses(document)
        self.assertEqual(document, before)
        self.assertEqual([(value["onsetSeconds"], value["string"]) for value in cleaned["notes"]], [(1.02, 5), (1.06, 6)])
        self.assertEqual(cleaned["percussion"], [{"onsetSeconds": 1.02, "technique": "thumb_slap", "confidence": 0.7}])
        self.assertEqual(report["sourceCounts"], {"notes": 5, "percussion": 3})
        self.assertEqual(report["retainedCounts"], {"notes": 2, "percussion": 1})
        self.assertAlmostEqual(report["maximumOnsetClusterDisplacementSeconds"], 0.02)
        self.assertFalse(report["rawHypothesesModified"])

    def test_uncalibrated_harmonics_are_removed_but_note_is_retained(self):
        harmonic = {"type": "Natural", "fret": 12, "confidence": 0.999}
        document = {"notes": [note(1, 1, 0.99, harmonic=harmonic)], "percussion": []}
        cleaned, report = clean_hypotheses(document)
        self.assertIsNone(cleaned["notes"][0]["harmonic"])
        self.assertEqual(report["removedHarmonics"][0]["reason"], "disabled_uncalibrated_harmonic_head")
        enabled, _ = clean_hypotheses(document, DraftProfile(include_harmonics=True, harmonic_threshold=0.9))
        self.assertEqual(enabled["notes"][0]["harmonic"], harmonic)
        mismatched = {"notes": [note(1, 1, 0.99, harmonic=harmonic, uncertainty=("sounding_pitch_harmonic_mismatch",))], "percussion": []}
        cleaned, report = clean_hypotheses(mismatched, DraftProfile(include_harmonics=True, harmonic_threshold=0.9))
        self.assertIsNone(cleaned["notes"][0]["harmonic"])
        self.assertEqual(report["removedHarmonics"][0]["reason"], "pitch_fret_harmonic_mismatch")

    def test_invalid_profile_and_events_fail_explicitly(self):
        with self.assertRaises(ValueError):
            DraftProfile(note_threshold=2)
        with self.assertRaises(TypeError):
            DraftProfile(include_harmonics=1)
        with self.assertRaises(HarnessError):
            clean_hypotheses({"notes": [note(float("nan"), 1, 0.9)], "percussion": []})
        with self.assertRaises(HarnessError):
            clean_hypotheses({"notes": [], "percussion": [{"onsetSeconds": 0, "technique": "", "confidence": 0.9}]})


if __name__ == "__main__":
    unittest.main()
