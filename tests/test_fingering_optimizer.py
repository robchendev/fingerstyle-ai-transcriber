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


def at(value, quarter, seconds=None):
    value = deepcopy(value)
    value["scoreOnsetQuarter"] = [quarter, 1] if type(quarter) is int else quarter
    value["onsetSeconds"] = float(quarter) / 2 if seconds is None else seconds
    return value


def hand_range(low, high, onset=0, confidence=1):
    return {
        "scoreOnsetQuarter": [onset, 1], "minimumFret": low, "maximumFret": high,
        "confidence": confidence, "source": "synthetic-hand-range",
    }


def with_grace(value, interval, transition="none", source_fret=0):
    value = deepcopy(value)
    value.setdefault("noteId", "anchor")
    value["grace"] = {
        "confidence": .99, "sourceFret": source_fret,
        "sourcePitchMidi": value["soundingPitchMidi"] - interval,
        "intervalSemitones": interval, "mode": "BeforeBeat", "transition": transition,
        "fretConfidence": .95, "modeConfidence": .95, "transitionConfidence": .95,
        "anchorNoteId": value["noteId"], "onsetSeconds": None, "timingKnown": False,
    }
    return value


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

    def test_distinct_simultaneous_pitches_can_share_provisional_string(self):
        document = self.document([
            {**note(56, 3, 1), "_connectionToken": "low"},
            {**note(63, 3, 8), "_connectionToken": "middle"},
            {**note(68, 3, 13), "_connectionToken": "high"},
        ])
        before = deepcopy(document)
        result, report = optimize_fingerings(document)
        self.assertEqual(document, before)
        self.assertEqual([value["soundingPitchMidi"] for value in result["notes"]], [56, 63, 68])
        self.assertEqual(len({value["string"] for value in result["notes"]}), 3)
        self.assertEqual(report["removedCount"], 0)
        self.assertEqual(report["retainedNoteCount"], 3)
        actual_changes = []
        for index, (source, selected) in enumerate(zip(document["notes"], result["notes"])):
            self.assertEqual(selected["scoreOnsetQuarter"], source["scoreOnsetQuarter"])
            self.assertEqual(selected["onsetSeconds"], source["onsetSeconds"])
            self.assertEqual(selected["_connectionToken"], source["_connectionToken"])
            expected_pitch = document["metadata"]["openStringMidi"][6 - selected["string"]] + 1 + selected["fret"]
            self.assertEqual(selected["soundingPitchMidi"], expected_pitch)
            if (source["string"], source["fret"]) != (selected["string"], selected["fret"]):
                actual_changes.append(index)
        self.assertEqual(report["changedCount"], len(actual_changes))
        self.assertEqual([value["index"] for value in report["changes"]], actual_changes)

    def test_mixed_same_string_chord_drops_only_true_unison(self):
        document = self.document([
            {**note(56, 3, 1, .99), "_connectionToken": "survivor"},
            {**note(56, 4, 8, .8), "_connectionToken": "unison"},
            {**note(63, 3, 8), "_connectionToken": "middle"},
            {**note(68, 3, 13), "_connectionToken": "high"},
        ])
        result, report = optimize_fingerings(document)
        self.assertEqual([value["soundingPitchMidi"] for value in result["notes"]], [56, 63, 68])
        self.assertEqual(report["sourceNoteCount"], 4)
        self.assertEqual(report["retainedNoteCount"], 3)
        self.assertEqual(report["removedCount"], 1)
        self.assertEqual(report["removed"][0]["reason"], "simultaneous_unison_duplicate")
        self.assertEqual(report["removed"][0]["_connectionToken"], "unison")
        self.assertEqual(report["removed"][0]["keptIndex"], 0)

    def test_phrase_context_can_prefer_open_position_and_repeated_string(self):
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
        self.assertEqual(report["difficultChords"][0]["reason"], "extended_barre_in_selected_fingering")

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

    def test_lookahead_revoices_opening_and_keeps_formerly_dropped_outlier(self):
        opening = note(56, 3, 1)
        nearby = {**note(63, 3, 8), "scoreOnsetQuarter": [1, 1], "onsetSeconds": .5}
        outlier = {**note(76, 1, 11), "scoreOnsetQuarter": [1, 1], "onsetSeconds": .5}
        result, report = optimize_fingerings(self.document([opening, nearby, outlier]))
        selected = {item["soundingPitchMidi"]: (item["string"], item["fret"]) for item in result["notes"]}
        self.assertEqual(set(selected), {56, 63, 76})
        self.assertLessEqual(abs(selected[56][1] - min(selected[63][1], selected[76][1])), MAX_RAPID_POSITION_SHIFT)
        self.assertEqual(report["removedCount"], 0)
        self.assertEqual(report["maximumRapidPositionShift"], MAX_RAPID_POSITION_SHIFT)
        self.assertEqual(report["rapidShiftPolicy"], "report_only_never_drop")

    def test_three_onset_lookahead_avoids_low_high_low_excursion(self):
        document = self.document([
            at(note(56, 3, 1), 0), at(note(77, 1, 12), 1), at(note(56, 3, 1), 2),
        ])
        result, report = optimize_fingerings(document)
        self.assertEqual([value["soundingPitchMidi"] for value in result["notes"]], [56, 77, 56])
        frets = [value["fret"] for value in result["notes"]]
        self.assertEqual(frets[0], frets[2])
        self.assertTrue(all(abs(left - right) <= 5 for left, right in zip(frets, frets[1:])))
        self.assertEqual(report["removedCount"], 0)
        self.assertEqual(report["difficultMovementCount"], 0)
        self.assertGreater(report["changedCount"], 0)

    def test_legitimate_high_position_phrase_is_not_penalized_for_height(self):
        chord = [note(pitch, string, 8) for pitch, string in [(44, 6), (51, 5), (56, 4), (63, 3), (68, 2)]]
        document = self.document([at(value, onset) for onset in range(3) for value in chord])
        result, report = optimize_fingerings(document)
        self.assertEqual([(value["string"], value["fret"]) for value in result["notes"]], [
            (value["string"], value["fret"]) for value in document["notes"]
        ])
        self.assertEqual(report["changedCount"], 0)
        self.assertEqual(report["removedCount"], 0)

    def test_high_barre_shift_is_not_hidden_by_one_low_outlier(self):
        chord = [note(pitch, string, 8) for pitch, string in [(44, 6), (51, 5), (56, 4), (63, 3), (68, 2)]]
        document = self.document([note(37, 6, 1), *[at(value, 1) for value in chord]])
        result, report = optimize_fingerings(document)
        self.assertEqual([value["fret"] for value in result["notes"]], [1, 8, 8, 8, 8, 8])
        self.assertEqual(report["removedCount"], 0)
        self.assertEqual(report["difficultMovementCount"], 1)

    def test_long_rest_allows_position_change_and_seconds_control_motion_rate(self):
        fast = self.document([at(note(56, 3, 1), 0), at(note(76, 1, 11), 1, .25)])
        rested = deepcopy(fast)
        rested["notes"][0]["scoreDurationQuarter"] = [1, 4]
        rested["notes"][1]["onsetSeconds"] = 4
        fast_result, _ = optimize_fingerings(fast)
        rest_result, rest_report = optimize_fingerings(rested)
        self.assertGreater(fast_result["notes"][0]["fret"], 1)
        self.assertEqual([(value["string"], value["fret"]) for value in rest_result["notes"]], [(3, 1), (1, 11)])
        self.assertEqual(rest_report["difficultMovementCount"], 0)
        self.assertEqual(rest_report["removedCount"], 0)

    def test_unavoidable_fast_motion_is_reported_without_deleting_low_confidence_note(self):
        document = self.document([at(note(39, 6, 3), 0), at(note(88, 1, 23, .001), 1, .25)])
        result, report = optimize_fingerings(document)
        self.assertEqual(len(result["notes"]), 2)
        self.assertEqual(report["removedCount"], 0)
        self.assertEqual(report["difficultMovementCount"], 1)
        movement = report["difficultMovements"][0]
        self.assertEqual((movement["shiftFrets"], movement["elapsedSeconds"], movement["fretsPerSecond"]), (20, .25, 80))

    def test_harmonic_node_position_is_not_revoiced_as_ordinary_sounding_pitch(self):
        harmonic = {**note(84, 1, 7), "harmonic": {"type": "Natural", "fret": 7}, "fretBasePitchMidi": 72}
        result, report = optimize_fingerings(self.document([harmonic]))
        self.assertEqual(result["notes"], [harmonic])
        self.assertEqual(report["changedCount"], 0)
        self.assertEqual(report["harmonicPositions"][0]["reason"], "harmonic_position_preserved")
        unsupported = {**note(90, 1, 6), "harmonic": {"type": "Unknown", "fret": 6}}
        result, report = optimize_fingerings(self.document([unsupported]))
        self.assertEqual(result["notes"], [unsupported])
        self.assertEqual(report["harmonicPositions"][0]["reason"], "unsupported_harmonic_position_preserved")

    def test_artificial_harmonic_keeps_fretted_base_and_natural_uses_touch_node_for_span(self):
        artificial = {**note(82, 1, 5), "harmonic": {"type": "Artificial", "fret": 12}, "fretBasePitchMidi": 70}
        result, report = optimize_fingerings(self.document([artificial]))
        self.assertEqual(result["notes"], [artificial])
        self.assertTrue(report["harmonicPositions"][0]["pitchConsistent"])
        natural = {**note(77, 1, 0), "harmonic": {"type": "Natural", "fret": [12, 1]}}
        result, report = optimize_fingerings(self.document([natural, note(56, 3, 1)]))
        self.assertEqual(len(result["notes"]), 2)
        ordinary = next(value for value in result["notes"] if value.get("harmonic") is None)
        self.assertLessEqual(abs(ordinary["fret"] - 12), 5)
        self.assertEqual(report["removedCount"], 0)

    def test_bound_connection_uses_exact_origin_not_intervening_attack(self):
        origin = {**at(note(63, 2, 3), 0), "_connectionToken": "origin", "voiceIndex": 0}
        intervening = {**at(note(64, 2, 4), [1, 2], .25), "_connectionToken": "middle", "voiceIndex": 0}
        destination = {
            **at(note(65, 1, 0), 1), "_connectionToken": "destination",
            "_connectionOriginToken": "origin", "voiceIndex": 0, "legato": {"type": "hammerOn"},
        }
        document = self.document([origin, intervening, destination])
        before = deepcopy(document)
        result, report = optimize_fingerings(document)
        self.assertEqual(document, before)
        first, middle, last = result["notes"]
        self.assertEqual(first["string"], last["string"])
        self.assertNotEqual(first["string"], middle["string"])
        self.assertEqual(last["_connectionOriginToken"], "origin")
        self.assertEqual(report["connectionConflictCount"], 0)
        self.assertTrue(all("_connectionToken" in change for change in report["changes"]))

    def test_impossible_connection_preserves_both_pitches_and_reports_conflict(self):
        origin = {**at(note(39, 6, 3), 0), "_connectionToken": "origin"}
        destination = {**at(note(89, 1, 24), 1), "_connectionToken": "destination", "_connectionOriginToken": "origin"}
        result, report = optimize_fingerings(self.document([origin, destination]))
        self.assertEqual(len(result["notes"]), 2)
        self.assertEqual(report["connectionConflicts"][0]["reason"], "connection_requires_different_strings")

    def test_connection_across_window_commit_boundary_keeps_pair_identity(self):
        notes = [at(note(56, 3, 1), onset) for onset in range(5)]
        notes.extend([
            {**at(note(63, 2, 3), 5), "_connectionToken": "origin"},
            {**at(note(65, 1, 0), 6), "_connectionToken": "destination", "_connectionOriginToken": "origin"},
            at(note(56, 3, 1), 7),
        ])
        result, report = optimize_fingerings(self.document(notes))
        self.assertEqual(result["notes"][5]["string"], result["notes"][6]["string"])
        self.assertEqual(report["connectionConflictCount"], 0)

    def test_removed_unison_origin_is_reported_not_rebound_to_survivor(self):
        origin = {**at(note(63, 2, 3, .5), 0), "_connectionToken": "origin"}
        duplicate = {**at(note(63, 2, 3, .99), 0), "_connectionToken": "survivor"}
        destination = {**at(note(65, 2, 5), 1), "_connectionToken": "destination", "_connectionOriginToken": "origin"}
        result, report = optimize_fingerings(self.document([origin, duplicate, destination]))
        self.assertEqual(len(result["notes"]), 2)
        self.assertEqual(result["notes"][1]["_connectionOriginToken"], "origin")
        self.assertEqual(report["connectionConflicts"][0]["reason"], "connection_endpoint_not_retained")
        self.assertEqual(report["removed"][0]["_connectionToken"], "origin")

    def test_evidence_masks_are_validated_and_never_reduce_pitch_retention(self):
        first = {**note(67, 1, 2), "allowedFingeringCandidates": [{"string": 1, "fret": 2}]}
        second = {**note(68, 1, 3), "allowedFingeringCandidates": [{"string": 1, "fret": 3}]}
        result, report = optimize_fingerings(self.document([first, second]))
        self.assertEqual(len(result["notes"]), 2)
        self.assertEqual(report["evidenceConflicts"][0]["reason"], "evidence_reduces_chord_pitch_retention")
        only_high = {**note(56, 3, 1), "allowedFingeringCandidates": [{"string": 4, "fret": 8}]}
        result, report = optimize_fingerings(self.document([only_high]))
        self.assertEqual((result["notes"][0]["string"], result["notes"][0]["fret"]), (4, 8))
        self.assertEqual(report["evidenceConflictCount"], 0)
        empty = {**note(56, 3, 1), "allowedFingeringCandidates": []}
        result, report = optimize_fingerings(self.document([empty]))
        self.assertEqual(len(result["notes"]), 1)
        self.assertEqual(report["evidenceConflicts"][0]["reason"], "evidence_has_no_compatible_position")
        with self.assertRaises(HarnessError):
            optimize_fingerings(self.document([{**first, "allowedFingeringCandidates": [{"string": 1, "fret": 3}]}]))

    def test_learned_candidate_likelihood_can_select_high_position_without_changing_pitch(self):
        value = {**note(56, 3, 1), "arrangerStringLogits": [0, 0, 10, 0, 0, 0]}
        result, report = optimize_fingerings(self.document([value]))
        self.assertEqual((result["notes"][0]["string"], result["notes"][0]["fret"]), (4, 8))
        self.assertEqual(result["notes"][0]["soundingPitchMidi"], 56)
        self.assertEqual(report["removedCount"], 0)

    def test_cardinality_precedes_all_confidence_and_likelihood_costs(self):
        document = {
            "metadata": {"openStringMidi": [40, 45, 50, 55, 59, 64], "capoFret": 0},
            "notes": [note(pitch, 6 - index, 0, .00001) for index, pitch in enumerate([40, 45, 50, 55, 59, 64])],
        }
        for value in document["notes"]:
            value["arrangerStringLogits"] = [100, -100, -100, -100, -100, -100]
        result, report = optimize_fingerings(document)
        self.assertEqual(len(result["notes"]), 6)
        self.assertEqual(report["removedCount"], 0)

    def test_deterministic_output_and_final_unison_winner_provenance(self):
        document = self.document([note(56, 3, 1, .5), note(56, 4, 8, .8), note(56, 3, 1, .99), at(note(77, 1, 12), 1)])
        first = optimize_fingerings(document)
        self.assertEqual(first, optimize_fingerings(document))
        self.assertTrue(all(value["keptIndex"] == 2 for value in first[1]["removed"]))

    def test_invalid_logits_are_rejected_even_when_pitch_has_no_candidate(self):
        with self.assertRaises(HarnessError):
            optimize_fingerings(self.document([{**note(10, 6, 0), "arrangerStringLogits": [float("nan")] * 6}]))

    def test_soft_hand_range_guides_position_and_keeps_provenance(self):
        document = self.document([note(56, 3, 1)])
        baseline, _ = optimize_fingerings(document)
        document["handPositionEvidence"] = [hand_range(7, 9)]
        before = deepcopy(document)
        result, report = optimize_fingerings(document)
        self.assertEqual(document, before)
        self.assertEqual(baseline["notes"][0]["fret"], 1)
        self.assertEqual(result["notes"][0]["fret"], 8)
        self.assertEqual(result["notes"][0]["soundingPitchMidi"], 56)
        evidence = report["handPositionEvidence"][0]
        self.assertEqual((evidence["selectedPosition"], evidence["distanceOutsideRange"], evidence["status"]), (8, 0, "satisfied"))
        self.assertEqual(evidence["source"], "synthetic-hand-range")
        self.assertEqual(report["handPositionEvidenceCount"], 1)
        self.assertEqual(report["handPositionEvidenceConflictCount"], 0)
        result["handPositionEvidence"][0]["source"] = "changed-copy"
        self.assertEqual(document["handPositionEvidence"][0]["source"], "synthetic-hand-range")

    def test_conflicting_soft_hand_range_never_drops_pitch_or_revoices_harmonic(self):
        ordinary = note(39, 6, 3)
        harmonic = {**at(note(84, 1, 7), 1), "harmonic": {"type": "Natural", "fret": 7}, "fretBasePitchMidi": 72}
        document = self.document([ordinary, harmonic])
        document["handPositionEvidence"] = [hand_range(18, 22), hand_range(19, 20, onset=1)]
        result, report = optimize_fingerings(document)
        self.assertEqual(len(result["notes"]), 2)
        self.assertEqual(result["notes"][1], harmonic)
        self.assertEqual(report["removedCount"], 0)
        self.assertEqual(report["handPositionEvidenceConflictCount"], 2)
        self.assertEqual([value["distanceOutsideRange"] for value in report["handPositionEvidence"]], [15, 12])

    def test_bound_connection_has_priority_over_soft_hand_range(self):
        origin = {**at(note(63, 2, 3), 0), "_connectionToken": "origin"}
        destination = {**at(note(65, 1, 0), 1), "_connectionToken": "destination", "_connectionOriginToken": "origin"}
        document = self.document([origin, destination])
        document["handPositionEvidence"] = [hand_range(0, 0, onset=1)]
        result, report = optimize_fingerings(document)
        self.assertEqual(result["notes"][0]["string"], result["notes"][1]["string"])
        self.assertEqual(result["notes"][1]["_connectionOriginToken"], "origin")
        self.assertEqual(report["connectionConflictCount"], 0)
        self.assertEqual(report["handPositionEvidenceConflictCount"], 1)

    def test_unused_or_zero_weight_hand_evidence_is_reported(self):
        document = self.document([note(36, 6, 0), at(note(56, 3, 1), 1)])
        document["handPositionEvidence"] = [
            hand_range(8, 10), hand_range(8, 10, onset=1, confidence=0), hand_range(8, 10, onset=2),
        ]
        result, report = optimize_fingerings(document)
        self.assertEqual([value["fret"] for value in result["notes"]], [0, 1])
        self.assertEqual([value["status"] for value in report["handPositionEvidence"]], [
            "no_non_open_contacts", "zero_weight", "no_chord_at_evidence_onset",
        ])
        self.assertEqual(report["unusedHandPositionEvidenceCount"], 3)
        self.assertEqual(report["handPositionEvidenceConflictCount"], 0)

    def test_hand_evidence_validates_range_time_confidence_and_source(self):
        for invalid in [
            {**hand_range(8, 10), "minimumFret": 11},
            {**hand_range(8, 10), "maximumFret": 25},
            {**hand_range(8, 10), "minimumFret": True},
            {**hand_range(8, 10), "scoreOnsetQuarter": [0, 0]},
            {**hand_range(8, 10), "scoreOnsetQuarter": [.5, 1]},
            {**hand_range(8, 10), "confidence": float("nan")},
            {**hand_range(8, 10), "confidence": 2},
            {**hand_range(8, 10), "source": " "},
            {**hand_range(8, 10), "unknownField": 1},
        ]:
            with self.subTest(invalid=invalid), self.assertRaises(HarnessError):
                optimize_fingerings({**self.document([note(56, 3, 1)]), "handPositionEvidence": [invalid]})
        with self.assertRaises(HarnessError):
            optimize_fingerings({**self.document([]), "handPositionEvidence": [
                hand_range(8, 10), {**hand_range(8, 10), "scoreOnsetQuarter": [0, 2]},
            ]})
        without_rhythm = note(56, 3, 1)
        without_rhythm.pop("scoreOnsetQuarter")
        with self.assertRaises(HarnessError):
            optimize_fingerings({**self.document([without_rhythm]), "handPositionEvidence": [hand_range(8, 10)]})

    def test_two_pass_pitch_repair_preserves_tokens_harmonics_and_inference_extras(self):
        origin = {**note(56, 3, 1), "_connectionToken": "origin"}
        destination = {
            **at(note(63, 1, 0), 1), "fretBasePitchMidi": 65,
            "_connectionToken": "destination", "_connectionOriginToken": "origin",
            "durationConfidence": .7, "modelExtras": {"frameIndices": [10, 11], "scores": [.1, .9]},
        }
        harmonic = {
            **at(note(84, 1, 7), 2), "harmonic": {"type": "Natural", "fret": 7},
            "fretBasePitchMidi": 72, "_connectionToken": "harmonic",
        }
        document = self.document([origin, destination, harmonic])
        document["inferenceExtras"] = {"calibration": {"threshold": .25}}
        before = deepcopy(document)
        first, _ = optimize_fingerings(document)
        self.assertEqual(document, before)
        first["notes"].append({**at(note(65, 1, 0), 3), "_connectionToken": "completed", "symbolicCompletion": {"confidence": .9}})
        before_second = deepcopy(first)
        second, _ = optimize_fingerings(first)
        self.assertEqual(first, before_second)
        self.assertEqual(second["inferenceExtras"], document["inferenceExtras"])
        by_token = {value["_connectionToken"]: value for value in second["notes"]}
        self.assertEqual(by_token["destination"]["_connectionOriginToken"], "origin")
        self.assertEqual(by_token["destination"]["modelExtras"], destination["modelExtras"])
        self.assertEqual(by_token["destination"]["durationConfidence"], .7)
        self.assertEqual(by_token["completed"]["symbolicCompletion"], {"confidence": .9})
        self.assertEqual(by_token["harmonic"], harmonic)
        for value in second["notes"]:
            if value.get("harmonic") is None:
                expected = document["metadata"]["openStringMidi"][6 - value["string"]] + 1 + value["fret"]
                self.assertEqual(value["soundingPitchMidi"], expected)
                self.assertEqual(value["fretBasePitchMidi"], expected)

    def test_grace_requires_playable_source_pitch_on_the_selected_main_string(self):
        value = with_grace(note(60, 2, 0), 5, "hammer_on")
        value["arrangerStringLogits"] = [0, 0, 0, 0, 15, 0]
        document = self.document([value])
        before = deepcopy(document)
        result, report = optimize_fingerings(document)
        self.assertEqual(document, before)
        selected = result["notes"][0]
        self.assertEqual((selected["string"], selected["fret"]), (3, 5))
        self.assertEqual((selected["grace"]["sourceFret"], selected["grace"]["sourcePitchMidi"]), (0, 55))
        self.assertIsNone(selected["grace"]["onsetSeconds"])
        self.assertFalse(selected["grace"]["timingKnown"])
        self.assertEqual(selected["scoreOnsetQuarter"], value["scoreOnsetQuarter"])
        self.assertEqual(report["removedCount"], 0)
        self.assertEqual(report["graceConflictCount"], 0)
        self.assertEqual(len(result["notes"]), 1)

    def test_grace_source_fret_is_recomputed_after_revoicing_with_identities_intact(self):
        value = with_grace({
            **note(63, 2, 3), "noteId": "destination", "connectionOriginNoteId": "filtered-origin",
        }, 2, "hammer_on", source_fret=1)
        document = self.document([value])
        document["handPositionEvidence"] = [hand_range(8, 8)]
        result, report = optimize_fingerings(document)
        selected = result["notes"][0]
        self.assertEqual((selected["string"], selected["fret"]), (3, 8))
        self.assertEqual(selected["grace"]["sourceFret"], 6)
        self.assertEqual(selected["grace"]["sourcePitchMidi"], 61)
        self.assertEqual(selected["noteId"], "destination")
        self.assertEqual(selected["connectionOriginNoteId"], "filtered-origin")
        self.assertEqual(selected["grace"]["anchorNoteId"], "destination")
        self.assertEqual(report["graceChangedCount"], 1)
        self.assertEqual(report["graceFingerings"][0]["grace"]["sourceFret"], 1)
        self.assertEqual(report["graceFingerings"][0]["selectedSourceFret"], 6)
        self.assertEqual(report["graceFingerings"][0]["requiredSourceFret"], 6)
        self.assertEqual(report["graceFingerings"][0]["selectedFret"], 8)
        self.assertEqual(report["changes"][0]["noteId"], "destination")

    def test_grace_source_uses_sounding_pitch_not_harmonic_node_minus_interval(self):
        value = with_grace({
            **note(84, 1, 7), "harmonic": {"type": "Natural", "fret": 7}, "fretBasePitchMidi": 72,
        }, 2, "hammer_on", source_fret=5)
        result, report = optimize_fingerings(self.document([value]))
        selected = result["notes"][0]
        self.assertEqual((selected["string"], selected["fret"], selected["fretBasePitchMidi"]), (1, 7, 72))
        self.assertEqual(selected["harmonic"], value["harmonic"])
        self.assertEqual((selected["grace"]["sourcePitchMidi"], selected["grace"]["sourceFret"]), (82, 17))
        self.assertEqual(report["changedCount"], 0)
        self.assertEqual(report["graceChangedCount"], 1)

    def test_grace_conflict_cannot_delete_an_otherwise_feasible_main_pitch(self):
        value = with_grace(note(44, 5, 1), 7, "hammer_on", source_fret=1)
        document = self.document([note(39, 6, 3), value])
        result, report = optimize_fingerings(document)
        self.assertEqual([selected["soundingPitchMidi"] for selected in result["notes"]], [39, 44])
        self.assertIsNone(result["notes"][1]["grace"])
        self.assertEqual(report["removedCount"], 0)
        self.assertEqual(report["graceConflictCount"], 1)
        self.assertEqual(report["graceFingerings"][0]["reason"], "grace_source_below_selected_open_string")
        self.assertEqual(report["graceFingerings"][0]["selectedString"], 5)
        self.assertEqual(report["graceFingerings"][0]["selectedFret"], 1)
        self.assertEqual(report["graceFingerings"][0]["requiredSourceFret"], -6)
        self.assertEqual(report["graceFingerings"][0]["sourcePitchMidi"], 37)
        self.assertEqual(report["graceFingerings"][0]["intervalSemitones"], 7)
        self.assertEqual(report["graceFingerings"][0]["grace"], value["grace"])

    def test_bound_connection_precedes_conflicting_grace_gesture(self):
        origin = {**at(note(39, 6, 3), 0), "_connectionToken": "origin"}
        destination = with_grace({
            **at(note(44, 5, 1), 1), "_connectionToken": "destination", "_connectionOriginToken": "origin",
        }, -23, "pull_off", source_fret=24)
        result, report = optimize_fingerings(self.document([origin, destination]))
        self.assertEqual(result["notes"][0]["string"], result["notes"][1]["string"])
        self.assertEqual(result["notes"][1]["_connectionOriginToken"], "origin")
        self.assertIsNone(result["notes"][1]["grace"])
        self.assertEqual(report["connectionConflictCount"], 0)
        self.assertEqual(report["graceConflictCount"], 1)
        self.assertEqual(report["removedCount"], 0)
        self.assertEqual(report["graceFingerings"][0]["reason"], "grace_source_above_selected_fret_limit")
        self.assertEqual(report["graceFingerings"][0]["requiredSourceFret"], 31)

    def test_impossible_grace_source_does_not_remove_its_anchor(self):
        value = with_grace(note(36, 6, 0), 1, "hammer_on")
        result, report = optimize_fingerings(self.document([value]))
        self.assertEqual(len(result["notes"]), 1)
        self.assertEqual(result["notes"][0]["soundingPitchMidi"], 36)
        self.assertIsNone(result["notes"][0]["grace"])
        self.assertEqual(report["graceConflictCount"], 1)

    def test_invalid_grace_transition_direction_is_suppressed_explicitly(self):
        for transition, interval in [("hammer_on", -2), ("hammer_on", 0), ("pull_off", 2), ("pull_off", 0), ("slide_1", 0), ("slide_2", 0)]:
            with self.subTest(transition=transition, interval=interval):
                result, report = optimize_fingerings(self.document([with_grace(note(63, 2, 3), interval, transition)]))
                self.assertIsNone(result["notes"][0]["grace"])
                self.assertEqual(report["removedCount"], 0)
                self.assertEqual(report["graceFingerings"][0]["reason"], "grace_transition_interval_mismatch")
        result, report = optimize_fingerings(self.document([with_grace(note(63, 2, 3), 0)]))
        self.assertIsNotNone(result["notes"][0]["grace"])
        self.assertEqual(report["graceConflictCount"], 0)

    def test_grace_source_pitch_and_anchor_conflicts_are_not_silently_rebound(self):
        for field, value, reason in [
            ("sourcePitchMidi", 60, "grace_source_pitch_interval_mismatch"),
            ("anchorNoteId", "other-anchor", "grace_anchor_note_id_mismatch"),
        ]:
            with self.subTest(field=field):
                source = with_grace(note(63, 2, 3), 2, "hammer_on", source_fret=1)
                source["grace"][field] = value
                result, report = optimize_fingerings(self.document([source]))
                self.assertIsNone(result["notes"][0]["grace"])
                self.assertEqual(report["graceFingerings"][0]["reason"], reason)
                self.assertEqual(report["graceFingerings"][0]["grace"][field], value)
        malformed = with_grace(note(63, 2, 3), 2)
        malformed["grace"]["intervalSemitones"] = 2.5
        with self.assertRaises(HarnessError):
            optimize_fingerings(self.document([malformed]))

    def test_grace_on_removed_unison_stays_with_its_original_anchor(self):
        original = with_grace({
            **note(63, 2, 3, .5), "noteId": "removed", "_connectionToken": "original",
        }, 2, "hammer_on", source_fret=1)
        survivor = {**note(63, 2, 3, .99), "noteId": "surviving", "_connectionToken": "surviving"}
        destination = {
            **at(note(65, 2, 5), 1), "noteId": "destination", "_connectionToken": "destination",
            "connectionOriginNoteId": "removed", "_connectionOriginToken": "original",
        }
        result, report = optimize_fingerings(self.document([original, survivor, destination]))
        self.assertEqual([value["noteId"] for value in result["notes"]], ["surviving", "destination"])
        self.assertNotIn("grace", result["notes"][0])
        self.assertEqual(result["notes"][1]["connectionOriginNoteId"], "removed")
        self.assertEqual(result["notes"][1]["_connectionOriginToken"], "original")
        self.assertEqual(report["graceFingerings"][0]["status"], "anchor_not_retained")
        self.assertEqual(report["graceFingerings"][0]["grace"]["anchorNoteId"], "removed")
        self.assertEqual(report["connectionConflicts"][0]["reason"], "connection_endpoint_not_retained")


if __name__ == "__main__":
    unittest.main()
