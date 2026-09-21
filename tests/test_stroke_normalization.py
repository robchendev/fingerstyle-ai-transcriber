from copy import deepcopy
from fractions import Fraction
import unittest

from scripts.stroke_normalization import normalize_downstroke_bursts
from tests.test_rhythm_inference import metadata


def burst(positions=(Fraction(29, 8), Fraction(15, 4), Fraction(31, 8), Fraction(4))):
    notes, techniques = [], []
    for index, position in enumerate(positions):
        for string, pitch in ((6, 40), (5, 47)):
            notes.append({
                "noteId": f"{index}-{string}", "onsetSeconds": float(position) / 2,
                "scoreOnsetQuarter": [position.numerator, position.denominator],
                "scoreDurationQuarter": [1, 8], "notatedDurationQuarter": .125,
                "string": string, "fret": 0 if string == 6 else 2, "voiceIndex": 0,
                "soundingPitchMidi": pitch, "harmonic": None, "confidence": .99, "uncertainty": [],
            })
        techniques.append({
            "onsetSeconds": float(position) / 2, "scoreOnsetQuarter": [position.numerator, position.denominator],
            "technique": "brush", "direction": "Down", "strings": [5, 6], "confidence": .99,
            "stringMembershipConfidence": {str(i): .99 if i in (5, 6) else .01 for i in range(1, 7)},
        })
    return {"metadata": metadata(), "notes": notes, "techniques": techniques, "percussion": []}


class StrokeNormalizationTests(unittest.TestCase):
    def test_overwritten_burst_becomes_three_downstrokes_landing_on_beat(self):
        original = burst()
        snapshot = deepcopy(original)
        result, report = normalize_downstroke_bursts(original)
        self.assertEqual(original, snapshot)
        self.assertEqual(report["changedGroupCount"], 1)
        self.assertEqual(len(result["notes"]), 6)
        self.assertEqual([t["scoreOnsetQuarter"] for t in result["techniques"]], [[15, 4], [31, 8], [4, 1]])
        self.assertEqual([t["strokeFinger"] for t in result["techniques"]], ["a", "m", "i"])
        self.assertEqual([t["direction"] for t in result["techniques"]], ["Down"] * 3)
        self.assertEqual({n["soundingPitchMidi"] for n in result["notes"]}, {40, 47})
        self.assertEqual(report["changes"][0]["mergedRepeatedPitchAttacks"], 2)
        self.assertEqual(normalize_downstroke_bursts(result)[0], result)

    def test_valid_triple_and_ordinary_eighth_strokes_remain_unchanged(self):
        for positions in ((Fraction(15, 4), Fraction(31, 8), Fraction(4)),
                          (Fraction(5, 2), Fraction(3), Fraction(7, 2), Fraction(4))):
            document = burst(positions)
            result, report = normalize_downstroke_bursts(document)
            self.assertEqual(result, document)
            self.assertEqual(report["changedGroupCount"], 0)

    def test_pitch_changes_upstrokes_and_anchored_relationships_are_not_merged(self):
        for change in ("pitch", "up", "grace", "origin", "intervening"):
            document = burst()
            if change == "pitch":
                document["notes"][0]["soundingPitchMidi"] = 42
            elif change == "up":
                document["techniques"][1]["direction"] = "Up"
            elif change == "grace":
                document["notes"][0]["grace"] = {"sourceFret": 2}
            elif change == "origin":
                document["notes"][1]["connectionOriginNoteId"] = document["notes"][0]["noteId"]
            else:
                document["notes"].append({**document["notes"][0], "scoreOnsetQuarter": [59, 16]})
            result, report = normalize_downstroke_bursts(document)
            self.assertEqual(report["changedGroupCount"], 0, change)
            self.assertCountEqual(result["notes"], document["notes"])

    def test_compound_evidence_keeps_notes_and_can_normalize_supported_dense_strokes(self):
        document = burst()
        for event in document["techniques"]:
            event["technique"] = "rasgueado"
        result, report = normalize_downstroke_bursts(document)
        self.assertEqual(report["changedGroupCount"], 1)
        self.assertEqual([t["technique"] for t in result["techniques"]], ["brush"] * 3)
        isolated = burst((Fraction(4),))
        isolated["techniques"][0]["technique"] = "rasgueado"
        self.assertEqual(normalize_downstroke_bursts(isolated)[0], isolated)

    def test_landing_respects_supplied_dotted_quarter_beat_unit(self):
        document = burst((Fraction(41, 8), Fraction(21, 4), Fraction(43, 8), Fraction(11, 2)))
        document["metadata"]["tempo"]["beatUnit"] = [3, 8]
        self.assertEqual(normalize_downstroke_bursts(document)[1]["changedGroupCount"], 0)
        document = burst((Fraction(41, 8), Fraction(21, 4), Fraction(43, 8), Fraction(11, 2)))
        document["metadata"]["tempo"]["beatUnit"] = [1, 8]
        result, report = normalize_downstroke_bursts(document)
        self.assertEqual(report["changedGroupCount"], 1)
        self.assertEqual(result["techniques"][-1]["scoreOnsetQuarter"], [11, 2])

    def test_extra_landing_repetition_merges_without_erasing_a_chord_change(self):
        document = burst((Fraction(15, 4), Fraction(31, 8), Fraction(4), Fraction(33, 8)))
        for note in document["notes"][4:]:
            note["soundingPitchMidi"] += 2
        result, report = normalize_downstroke_bursts(document)
        self.assertEqual(report["changedGroupCount"], 1)
        landing = [n for n in result["notes"] if n["scoreOnsetQuarter"] == [4, 1]]
        self.assertEqual({n["soundingPitchMidi"] for n in landing}, {42, 49})
        self.assertEqual({n["soundingPitchMidi"] for n in result["notes"][:4]}, {40, 47})


if __name__ == "__main__":
    unittest.main()
