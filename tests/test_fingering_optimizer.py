from copy import deepcopy
import unittest

from scripts.fingering_optimizer import MAX_FRETTED_SPAN, MAX_RAPID_POSITION_SHIFT, optimize_fingerings
from scripts.transcriber_audio import HarnessError


def note(pitch, string, fret, confidence=.95):
    return {
        "scoreOnsetQuarter": [0, 1],
        "onsetSeconds": 0,
        "soundingPitchMidi": pitch,
        "string": string,
        "fret": fret,
        "confidence": confidence,
    }


class FingeringOptimizerTests(unittest.TestCase):
    def document(self, notes):
        return {
            "metadata": {"openStringMidi": [35, 42, 47, 54, 59, 64], "capoFret": 1},
            "notes": notes,
        }

    def test_unisons_are_deduplicated_and_stretched_chord_is_revoiced(self):
        document = self.document([
            note(36, 6, 7),
            note(48, 5, 2),
            note(60, 4, 7),
            note(60, 3, 0, .96),
            note(60, 2, 0),
            note(67, 1, 2),
        ])
        before = deepcopy(document)
        result, report = optimize_fingerings(document)
        self.assertEqual(document, before)
        self.assertEqual(len({value["soundingPitchMidi"] for value in result["notes"]}), len(result["notes"]))
        self.assertEqual(len({value["string"] for value in result["notes"]}), len(result["notes"]))
        fretted = [value["fret"] for value in result["notes"] if value["fret"]]
        self.assertLessEqual(max(fretted) - min(fretted), MAX_FRETTED_SPAN)
        self.assertEqual({value["soundingPitchMidi"] for value in result["notes"]}, {36, 48, 60, 67})
        self.assertEqual(report["removedCount"], 2)
        self.assertTrue(all(value["reason"] == "simultaneous_unison_duplicate" for value in report["removed"]))
        self.assertFalse(report["rawHypothesesModified"])

    def test_more_than_six_unique_pitches_drops_low_confidence_voice(self):
        pitches = [40, 45, 50, 55, 59, 64, 67]
        document = self.document([note(pitch, max(1, 6 - index), 0, .99 - index * .05) for index, pitch in enumerate(pitches)])
        result, report = optimize_fingerings(document)
        self.assertLessEqual(len(result["notes"]), 6)
        self.assertTrue(any(value["reason"] == "unplayable_chord_voice" for value in report["removed"]))

    def test_phrase_context_prefers_open_position_and_repeated_string(self):
        opening = [
            note(44, 5, 1),
            note(51, 4, 3),
            note(56, 3, 1),
            note(67, 1, 2),
        ]
        first_c = {**note(60, 3, 5), "scoreOnsetQuarter": [1, 1], "onsetSeconds": .5}
        second_c = {**note(60, 3, 5), "scoreOnsetQuarter": [2, 1], "onsetSeconds": 1}
        result, _ = optimize_fingerings(self.document([*opening, first_c, second_c]))
        c_notes = [value for value in result["notes"] if value["soundingPitchMidi"] == 60]
        self.assertEqual([(value["string"], value["fret"]) for value in c_notes], [(2, 0), (2, 0)])

    def test_pitch_set_forcing_extended_barre_is_reported_not_hidden(self):
        document = self.document([
            note(39, 6, 3),
            note(46, 5, 3),
            note(51, 4, 3),
            note(58, 3, 3),
            note(63, 2, 3),
            note(65, 1, 0),
        ])
        result, report = optimize_fingerings(document)
        self.assertEqual(len(result["notes"]), 6)
        self.assertEqual(report["difficultChordCount"], 1)
        self.assertEqual(report["difficultChords"][0]["reason"], "extended_barre_required_by_retained_pitch_set")

    def test_pitch_without_guitar_position_and_malformed_input_fail(self):
        result, report = optimize_fingerings(self.document([note(10, 6, 0)]))
        self.assertEqual(result["notes"], [])
        self.assertEqual(report["removed"][0]["reason"], "pitch_has_no_playable_string")
        with self.assertRaises(HarnessError):
            optimize_fingerings({"metadata": {}, "notes": []})

    def test_impossible_string_arranger_score_cannot_drop_only_playable_position(self):
        value = note(40, 6, 0, .99)
        value["arrangerStringLogits"] = [0, 0, 0, 0, 0, 25]
        result, report = optimize_fingerings(self.document([value]))
        self.assertEqual([(item["string"], item["fret"]) for item in result["notes"]], [(6, 4)])
        # This synthetic document's tuning has string 6 at MIDI35 plus capo1.
        self.assertEqual(report["removedCount"], 0)

    def test_rapid_position_shift_revoices_nearby_pitch_and_drops_unreachable_outlier(self):
        opening = note(56, 3, 1)
        nearby = {**note(63, 3, 8), "scoreOnsetQuarter": [1, 1], "onsetSeconds": .5}
        outlier = {**note(76, 1, 11), "scoreOnsetQuarter": [1, 1], "onsetSeconds": .5}
        result, report = optimize_fingerings(self.document([opening, nearby, outlier]))
        selected = {item["soundingPitchMidi"]: (item["string"], item["fret"]) for item in result["notes"]}
        self.assertEqual(selected[63], (2, 3))
        self.assertNotIn(76, selected)
        self.assertTrue(any(item["reason"] == "rapid_position_shift_unplayable" for item in report["removed"]))
        self.assertEqual(report["maximumRapidPositionShift"], MAX_RAPID_POSITION_SHIFT)


if __name__ == "__main__":
    unittest.main()
