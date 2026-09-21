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
    def test_thumb_slap_override_does_not_lower_other_percussion_or_note_thresholds(self):
        document = {
            "notes": [note(0., 1, .75)], "percussion": [
                {"onsetSeconds": time, "technique": kind, "confidence": .4}
                for time, kind in ((0., "thumb_slap"), (1., "wrist_thump"), (2., "percussive_hit"))
            ],
        }
        result, report = clean_hypotheses(document, DraftProfile(note_threshold=.8, percussion_threshold=.8, thumb_slap_threshold=.3))
        self.assertEqual(result["notes"], [])
        self.assertEqual([event["technique"] for event in result["percussion"]], ["thumb_slap"])
        self.assertEqual(report["percussionThresholds"], {"thumb_slap": .3, "wrist_thump": .8, "percussive_hit": .8})
        self.assertFalse(clean_hypotheses(document, DraftProfile(percussion_threshold=.8))[0]["percussion"])
        for value in (-.1, 1.1, True, float("nan")):
            with self.assertRaises(ValueError):
                DraftProfile(thumb_slap_threshold=value)

    def test_rasgueado_candidates_are_not_discarded_by_technique_name(self):
        ordinary = [note(time, 1, .99) for time in (1.875, 1.9375, 2.)]
        generated = {**note(1.90, 2, 1., uncertainty=("technique_membership_completed_attack",)),
                     "completionParent": {"technique": "rasgueado", "onsetSeconds": 1.90}}
        document = {
            "notes": [*ordinary, generated], "percussion": [],
            "techniques": [
                {"onsetSeconds": 1.90, "technique": "rasgueado", "confidence": 1., "strings": [1, 2],
                 "stringMembershipConfidence": {str(i): float(i in (1, 2)) for i in range(1, 7)}},
                *[{"onsetSeconds": time, "technique": "brush", "direction": "Down", "confidence": 1.,
                   "strings": [1], "stringMembershipConfidence": {str(i): float(i == 1) for i in range(1, 7)}}
                  for time in (1.875, 1.9375, 2.)],
            ],
        }
        before = deepcopy(document)
        cleaned, report = clean_hypotheses(document)
        self.assertEqual(document, before)
        self.assertEqual(len(cleaned["notes"]), 4)
        self.assertTrue(any(n.get("completionParent") == generated["completionParent"] for n in cleaned["notes"]))
        self.assertEqual([t["technique"] for t in cleaned["techniques"]], ["rasgueado", "brush", "brush", "brush"])
        self.assertFalse(report["removedNotes"])
        self.assertFalse(report["removedTechniques"])

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
        self.assertEqual([(value["onsetSeconds"], value["string"]) for value in cleaned["notes"]], [(1., 6), (1., 5), (1.03, 6), (1.06, 6)])
        self.assertEqual(cleaned["percussion"], [{"onsetSeconds": 1., "technique": "thumb_slap", "confidence": 0.7}])
        self.assertEqual(report["sourceCounts"], {"notes": 5, "percussion": 3})
        self.assertEqual(report["retainedCounts"], {"notes": 4, "percussion": 1})
        self.assertAlmostEqual(report["maximumOnsetClusterDisplacementSeconds"], 0.02)
        self.assertFalse(report["rawHypothesesModified"])

    def test_repeated_32nds_survive_at_multiple_tempos_and_exact_duplicates_do_not(self):
        for bpm in (60, 122, 240, 320):
            step = 60 / bpm / 8
            document = {"notes": [note(index * step, string, .99) for index in range(4) for string in (6, 4)], "percussion": []}
            document["notes"].append(note(0, 6, .95))
            result, report = clean_hypotheses(document)
            self.assertEqual(len(result["notes"]), 8)
            self.assertEqual(sorted({n["onsetSeconds"] for n in result["notes"]}), [index * step for index in range(4)])
            self.assertEqual(report["removedNoteCount"], 1)

    def test_fast_alternating_strings_are_not_misread_as_one_chord(self):
        document = {
            "metadata": {"tempo": {"bpm": 240, "beatUnit": [1, 4]}, "timeSignature": [4, 4]},
            "notes": [note(0, 6, .99), note(.03125, 5, .99)], "percussion": [],
        }
        result, report = clean_hypotheses(document)
        self.assertEqual([n["onsetSeconds"] for n in result["notes"]], [0, .03125])
        self.assertEqual(report["effectiveChordGroupingSeconds"], .015625)

    def test_simultaneous_different_pitches_on_provisional_string_are_retained(self):
        document = {
            "notes": [note(0, 6, .99), {**note(0, 6, .98), "soundingPitchMidi": 45}],
            "percussion": [],
        }
        result, report = clean_hypotheses(document)
        self.assertEqual([n["soundingPitchMidi"] for n in result["notes"]], [40, 45])
        self.assertEqual(report["removedNoteCount"], 0)

    def test_weak_member_uses_calibrated_parent_not_independent_attack_threshold(self):
        document = {
            "notes": [note(0, 6, .65, uncertainty=("technique_membership_completed_attack",))],
            "percussion": [],
            "techniques": [{
                "onsetSeconds": 0, "technique": "brush", "confidence": .95,
                "strings": [6], "stringMembershipConfidence": {str(i): .65 if i == 6 else .1 for i in range(1, 7)},
            }],
        }
        cleaned, _ = clean_hypotheses(document)
        self.assertEqual(len(cleaned["notes"]), 1)
        strict, report = clean_hypotheses(document, DraftProfile(note_threshold=.98, strict_note_confidence=True))
        self.assertEqual(strict["notes"], [])
        self.assertEqual(report["removedNotes"][0]["reason"], "below_note_threshold")
        document["notes"][0]["confidence"] = .99
        strict, _ = clean_hypotheses(document, DraftProfile(note_threshold=.98, strict_note_confidence=True))
        self.assertEqual(len(strict["notes"]), 1)
        document["notes"][0]["confidence"] = .65
        document["techniques"][0]["confidence"] = .85
        cleaned, _ = clean_hypotheses(document)
        self.assertEqual(cleaned["notes"], [])
        document["techniques"][0]["confidence"] = .95
        document["notes"][0]["completionParent"] = {"technique": "arpeggio", "onsetSeconds": 0}
        cleaned, _ = clean_hypotheses(document)
        self.assertEqual(cleaned["notes"], [])

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

    def test_technique_thresholds_are_per_class_and_rebuild_string_membership(self):
        document = {
            "notes": [],
            "percussion": [],
            "techniques": [
                {
                    "onsetSeconds": 0, "technique": "brush", "confidence": .89,
                    "strings": [1], "stringMembershipConfidence": {str(i): .95 for i in range(1, 7)},
                },
                {
                    "onsetSeconds": 1, "technique": "arpeggio", "confidence": .95,
                    "strings": [1], "stringMembershipConfidence": {
                        "1": .9, "2": .6, "3": .49, "4": .2, "5": .1, "6": .8,
                    },
                },
            ],
        }
        cleaned, report = clean_hypotheses(document)
        self.assertEqual([(event["technique"], event["strings"]) for event in cleaned["techniques"]], [("arpeggio", [1, 2, 6])])
        self.assertEqual(report["sourceTechniqueCount"], 2)
        self.assertEqual(report["removedTechniqueCount"], 1)

    def test_membership_completed_note_requires_retained_parent_and_member_string(self):
        document = {
            "notes": [{
                "onsetSeconds": 1, "string": 3, "confidence": .95,
                "uncertainty": ["technique_membership_completed_attack"],
            }],
            "percussion": [],
            "techniques": [{
                "onsetSeconds": 1, "technique": "brush", "confidence": .95,
                "strings": [3], "stringMembershipConfidence": {
                    "1": .1, "2": .1, "3": .59, "4": .1, "5": .1, "6": .1,
                },
            }],
        }
        cleaned, report = clean_hypotheses(document)
        self.assertEqual(cleaned["notes"], [])
        self.assertEqual(len(report["removedOrphanTechniqueMembers"]), 1)


if __name__ == "__main__":
    unittest.main()
