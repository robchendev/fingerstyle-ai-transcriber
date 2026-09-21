from copy import deepcopy
from fractions import Fraction
import unittest

from scripts.rhythm_inference import PerformanceMap, infer_notated_timing
from scripts.transcriber_audio import HarnessError


def metadata(bpm=120, meter=(4, 4)):
    return {
        "openStringMidi": [40, 45, 50, 55, 59, 64],
        "capoFret": 0,
        "tempo": {"bpm": bpm, "beatUnit": [1, 4]},
        "timeSignature": list(meter),
        "tempoChanges": [],
        "timeSignatureChanges": [],
    }


def evidence(beats, downbeats):
    return {
        "kind": "audio-beat-evidence",
        "audioSha256": "audio",
        "beatSeconds": beats,
        "downbeatSeconds": downbeats,
    }


def note(time, duration=1):
    return {
        "onsetSeconds": time,
        "string": 6,
        "fret": 0,
        "soundingPitchMidi": 40,
        "voiceIndex": 0,
        "notatedDurationQuarter": duration,
        "harmonic": None,
        "confidence": .95,
        "uncertainty": [],
    }


class RhythmInferenceTests(unittest.TestCase):
    def test_fingerstyle_policy_keeps_ordinary_onsets_and_ends_on_sixteenths(self):
        document = {"metadata": metadata(), "audioDurationSeconds": 3, "percussion": [], "techniques": [],
                    "notes": [note(time, .37) for time in (0., .31, .57, .81, 1.31)]}
        result, report = infer_notated_timing(document, evidence([i / 2 for i in range(7)], [0., 2.]), rhythm_policy="fingerstyle")
        self.assertFalse(result["stroke32Windows"])
        self.assertEqual(report["onsetOptimization"]["tripletBeatCount"], 0)
        for event in result["notes"]:
            start = Fraction(*event["scoreOnsetQuarter"])
            stop = start + Fraction(*event["scoreDurationQuarter"])
            self.assertEqual((start * 4).denominator, 1)
            self.assertEqual((stop * 4).denominator, 1)

    def test_fingerstyle_policy_reserves_32nds_for_supported_three_downstrokes(self):
        document = {
            "metadata": metadata(), "audioDurationSeconds": 3, "percussion": [],
            "notes": [note(time, .125) for time in (1.875, 1.9375, 2.)],
            "techniques": [{"technique": "brush", "direction": "Down", "onsetSeconds": time} for time in (1.875, 1.9375, 2.)],
        }
        result, _ = infer_notated_timing(document, evidence([i / 2 for i in range(7)], [0., 2.]), rhythm_policy="fingerstyle")
        self.assertEqual(result["stroke32Windows"], [{"startQuarter": [15, 4], "endQuarter": [4, 1]}])
        self.assertEqual([Fraction(*n["scoreOnsetQuarter"]) for n in result["notes"]],
                         [Fraction(15, 4), Fraction(31, 8), Fraction(4)])

    def test_clean_beat_sequence_corrects_phase_and_rubato(self):
        beats = [0.1, 0.6, 1.12, 1.61, 2.1, 2.62, 3.1, 3.6, 4.1]
        mapper = PerformanceMap(evidence(beats, [0.1, 2.1, 4.1]), metadata(), 4.2, event_seconds=[0.1])
        self.assertAlmostEqual(float(mapper.quarter_at(.1)), 0)
        self.assertAlmostEqual(float(mapper.quarter_at(1.12)), 2)
        self.assertAlmostEqual(mapper.seconds_at(Fraction(3)), 1.61)
        self.assertEqual(mapper.report()["mappingStrategy"], "sequential-beat-count")

    def test_intro_silence_and_pickup_have_distinct_origins(self):
        intro = PerformanceMap(evidence([.8, 1.3, 1.8, 2.3, 2.8], [.8, 2.8]), metadata(), 3)
        self.assertEqual(float(intro.quarter_at(.8)), 0)
        pickup = PerformanceMap(evidence([0, .5, 1, 1.5, 2], [.5]), metadata(), 2.1, event_seconds=[0])
        self.assertEqual(float(pickup.quarter_at(0)), 0)
        self.assertEqual(float(pickup.quarter_at(.5)), 1)
        self.assertGreater(pickup.report()["pickupEvidenceCount"], 0)

    def test_unreliable_beat_count_falls_back_to_nominal_tempo(self):
        beats = [0, .5, 1, 1.5, 2, 2.5, 3.5, 4, 4.5, 5]
        mapper = PerformanceMap(evidence(beats, [0, 2, 4]), metadata(), 5)
        if mapper.report()["mappingStrategy"] == "nominal-tempo-fallback":
            self.assertAlmostEqual(float(mapper.quarter_at(5)), 10, places=5)
        else:
            self.assertLess(mapper.report()["sequentialEndpointDriftRatio"], .0351)

    def test_constraint_solver_outputs_shared_chord_time_and_conventional_duration(self):
        document = {
            "metadata": metadata(),
            "audioDurationSeconds": 3,
            "notes": [note(.11, .92), {**note(.13, 1.8), "string": 5, "soundingPitchMidi": 45}],
            "percussion": [{"onsetSeconds": .12, "technique": "thumb_slap", "confidence": .9}],
        }
        before = deepcopy(document)
        result, report = infer_notated_timing(document, evidence([.1, .6, 1.1, 1.6, 2.1, 2.6], [.1, 2.1]))
        self.assertEqual(document, before)
        self.assertEqual({tuple(value["scoreOnsetQuarter"]) for value in result["notes"] + result["percussion"]}, {(0, 1)})
        self.assertEqual(result["notes"][0]["scoreDurationQuarter"], [1, 1])
        self.assertEqual(result["notes"][1]["scoreDurationQuarter"], [2, 1])
        self.assertEqual(report["onsetOptimization"]["status"], "optimal")
        self.assertEqual(report["eventsAfterBeatSupport"], {"notes": 0, "percussion": 0})
        self.assertFalse(report["rawHypothesesModified"])

    def test_events_after_last_detected_beat_are_reported_and_optionally_retained(self):
        document = {
            "metadata": metadata(),
            "audioDurationSeconds": 4,
            "notes": [note(.1), note(3.5)],
            "percussion": [],
        }
        result, report = infer_notated_timing(document, evidence([.1, .6, 1.1, 1.6, 2.1], [.1, 2.1]))
        self.assertEqual(len(result["notes"]), 1)
        self.assertEqual(report["eventsAfterBeatSupport"], {"notes": 1, "percussion": 0})
        self.assertFalse(report["beatUnsupportedTailIncluded"])
        result, report = infer_notated_timing(
            document, evidence([.1, .6, 1.1, 1.6, 2.1], [.1, 2.1]),
            include_unsupported_tail=True,
        )
        self.assertEqual(len(result["notes"]), 2)
        self.assertTrue(report["beatUnsupportedTailIncluded"])

    def test_isolated_triplet_is_rejected_but_complete_triplet_group_is_allowed(self):
        beat_evidence = evidence([0, .5, 1, 1.5, 2], [0, 2])
        isolated = {
            "metadata": metadata(),
            "audioDurationSeconds": 2,
            "notes": [note(1 / 3)],
            "percussion": [],
        }
        result, report = infer_notated_timing(isolated, beat_evidence)
        self.assertNotEqual(Fraction(*result["notes"][0]["scoreOnsetQuarter"]).denominator, 3)
        self.assertEqual(report["onsetOptimization"]["tripletBeatCount"], 0)
        grouped = {
            "metadata": metadata(),
            "audioDurationSeconds": 2,
            "notes": [note(1 / 6), {**note(1 / 3), "string": 5, "soundingPitchMidi": 45}],
            "percussion": [],
        }
        result, report = infer_notated_timing(grouped, beat_evidence)
        self.assertEqual(
            {Fraction(*value["scoreOnsetQuarter"]) for value in result["notes"]},
            {Fraction(1, 3), Fraction(2, 3)},
        )
        self.assertEqual(report["onsetOptimization"]["tripletBeatCount"], 1)

    def test_malformed_evidence_fails(self):
        with self.assertRaises(HarnessError):
            PerformanceMap(evidence([0, 0], [0]), metadata(), 1)
        with self.assertRaises(HarnessError):
            PerformanceMap(evidence([0, .5], [.25]), metadata(), 1)

    def test_perfect_32nd_pair_and_downbeat_survive_end_to_end(self):
        from scripts.draft_cleanup import clean_hypotheses

        for bpm in (60, 122, 240):
            seconds_per_quarter = 60 / bpm
            beat_evidence = evidence([i * seconds_per_quarter for i in range(9)], [i * seconds_per_quarter for i in (0, 4, 8)])
            onsets = [Fraction(15, 4), Fraction(31, 8), Fraction(4)]
            document = {
                "metadata": metadata(bpm), "audioDurationSeconds": 8 * seconds_per_quarter,
                "notes": [note(float(q) * seconds_per_quarter, 1 / 8) for q in onsets],
                "percussion": [], "techniques": [],
            }
            cleaned, _ = clean_hypotheses(document)
            result, _ = infer_notated_timing(cleaned, beat_evidence)
            self.assertEqual([Fraction(*n["scoreOnsetQuarter"]) for n in result["notes"]], onsets)
            self.assertEqual([n["scoreDurationQuarter"] for n in result["notes"]], [[1, 8]] * 3)

    def test_two_nearby_observations_cannot_fake_two_distinct_triplet_positions(self):
        document = {
            "metadata": metadata(), "audioDurationSeconds": 2,
            "notes": [note(1 / 3), {**note(1 / 3 + .001), "string": 5}],
            "percussion": [],
        }
        _, report = infer_notated_timing(document, evidence([0, .5, 1, 1.5, 2], [0, 2]))
        self.assertEqual(report["onsetOptimization"]["tripletBeatCount"], 0)

    def test_triplet_durations_do_not_end_between_the_selected_rhythm_grids(self):
        document = {
            "metadata": metadata(), "audioDurationSeconds": 2,
            "notes": [note(1 / 6, .5), {**note(1 / 3, .5), "string": 5}],
            "percussion": [],
        }
        result, _ = infer_notated_timing(document, evidence([0, .5, 1, 1.5, 2], [0, 2]))
        for value in result["notes"]:
            onset = Fraction(*value["scoreOnsetQuarter"])
            end = onset + Fraction(*value["scoreDurationQuarter"])
            self.assertEqual((end * (6 if end < 1 else 8)).denominator, 1)

    def test_compound_hypotheses_cannot_move_or_split_independent_eighths(self):
        document = {"metadata": metadata(), "audioDurationSeconds": 2,
                    "notes": [note(t, .5) for t in (0., .25, .5, .75, 1.)], "percussion": [], "techniques": []}
        beats = evidence([0., .5, 1., 1.5, 2.], [0., 2.])
        baseline, _ = infer_notated_timing(document, beats)
        extra = deepcopy(document)
        for time in (.21, .46, .71, .96):
            extra["notes"].append({**note(time, .5), "uncertainty": ["technique_membership_completed_attack"],
                                   "completionParent": {"technique": "rasgueado", "onsetSeconds": time}})
            extra["techniques"].append({"technique": "rasgueado", "onsetSeconds": time})
        result, report = infer_notated_timing(extra, beats)
        self.assertEqual([n["scoreOnsetQuarter"] for n in result["notes"]], [n["scoreOnsetQuarter"] for n in baseline["notes"]])
        self.assertEqual(len(report["onsetOptimization"]["mergedCompoundDuplicates"]), 4)
        self.assertEqual([n["scoreDurationQuarter"] for n in result["notes"]], [[1, 2]] * 5)

    def test_compound_chord_tones_survive_alignment_to_an_independent_attack(self):
        document = {"metadata": metadata(), "audioDurationSeconds": 2, "percussion": [], "techniques": [],
                    "notes": [note(.5, .5)]}
        for string, pitch in ((6, 40), (5, 45)):
            document["notes"].append({**note(.46, .5), "string": string, "soundingPitchMidi": pitch,
                                     "uncertainty": ["technique_membership_completed_attack"],
                                     "completionParent": {"technique": "rasgueado", "onsetSeconds": .46}})
        result, _ = infer_notated_timing(document, evidence([0., .5, 1., 1.5, 2.], [0., 2.]))
        self.assertEqual(len(result["notes"]), 2)
        self.assertEqual({n["soundingPitchMidi"] for n in result["notes"]}, {40, 45})
        self.assertEqual({tuple(n["scoreOnsetQuarter"]) for n in result["notes"]}, {(1, 1)})

    def test_eighth_note_jitter_does_not_activate_32nds_for_one_beat(self):
        document = {"metadata": metadata(), "audioDurationSeconds": 2,
                    "notes": [note(t, .5) for t in (0., .25, .5, .75, 1., 1.25, 1.45, 1.75)],
                    "percussion": [], "techniques": []}
        result, _ = infer_notated_timing(document, evidence([0., .5, 1., 1.5, 2.], [0., 2.]))
        self.assertEqual([Fraction(*n["scoreOnsetQuarter"]) for n in result["notes"]],
                         [Fraction(i, 2) for i in range(8)])

    def test_separately_detected_downstroke_protects_a_compound_completed_32nd(self):
        document = {
            "metadata": metadata(), "audioDurationSeconds": 3, "percussion": [],
            "notes": [note(1.875, .125), note(2., .5), {
                **note(1.9375, .125), "uncertainty": ["technique_membership_completed_attack"],
                "completionParent": {"technique": "rasgueado", "onsetSeconds": 1.9375},
            }],
            "techniques": [{"onsetSeconds": t, "technique": "brush", "direction": "Down"} for t in (1.875, 1.9375, 2.)],
        }
        result, report = infer_notated_timing(document, evidence([i / 2 for i in range(7)], [0., 2.]))
        self.assertEqual(sorted(Fraction(*n["scoreOnsetQuarter"]) for n in result["notes"]),
                         [Fraction(15, 4), Fraction(31, 8), Fraction(4)])
        self.assertFalse(report["onsetOptimization"]["mergedCompoundDuplicates"])

    def test_delayed_brush_completion_fills_a_chord_without_a_second_attack(self):
        document = {
            "metadata": metadata(), "audioDurationSeconds": 2, "percussion": [],
            "notes": [note(.5, 1.), {**note(.55, 1.), "string": 5, "soundingPitchMidi": 45,
                                   "uncertainty": ["technique_membership_completed_attack"],
                                   "completionParent": {"technique": "brush", "onsetSeconds": .55}}],
            "techniques": [{"onsetSeconds": .55, "technique": "brush", "direction": "Down"}],
        }
        before = deepcopy(document)
        result, report = infer_notated_timing(document, evidence([0., .5, 1., 1.5, 2.], [0., 2.]))
        self.assertEqual(document, before)
        self.assertEqual(len(result["notes"]), 2)
        self.assertEqual({tuple(n["scoreOnsetQuarter"]) for n in result["notes"]}, {(1, 1)})
        self.assertEqual(result["techniques"][0]["scoreOnsetQuarter"], [1, 1])
        self.assertEqual(len(report["onsetOptimization"]["alignedCompoundCandidates"]), 1)

    def test_similar_chord_durations_share_an_end_without_shortening_the_bass(self):
        document = {
            "metadata": metadata(), "audioDurationSeconds": 3, "percussion": [], "techniques": [],
            "notes": [
                {**note(0., 2.3), "voiceIndex": 1},
                *[{**note(0., duration), "string": string, "soundingPitchMidi": pitch}
                  for string, pitch, duration in ((3, 55, 1.1), (2, 59, .9), (1, 64, .8))],
            ],
        }
        result, report = infer_notated_timing(document, evidence([i / 2 for i in range(7)], [0., 2.]))
        self.assertEqual(result["notes"][0]["scoreDurationQuarter"], [2, 1])
        self.assertEqual([n["scoreDurationQuarter"] for n in result["notes"][1:]], [[1, 1]] * 3)
        self.assertTrue(report["durationOptimization"]["chordDurationNormalization"])

    def test_unsupported_completion_is_reported_but_weak_acoustic_attack_can_support_a_chord(self):
        from scripts.draft_cleanup import clean_hypotheses

        for has_weak_attack in (False, True):
            with self.subTest(has_weak_attack=has_weak_attack):
                document = {
                    "metadata": metadata(), "audioDurationSeconds": 2, "percussion": [],
                    "notes": [note(0., .5), {**note(.75, .5), "string": 5, "soundingPitchMidi": 45,
                                           "uncertainty": ["technique_membership_completed_attack"],
                                           "completionParent": {"technique": "brush", "onsetSeconds": .75}}],
                    "techniques": [{"onsetSeconds": .75, "technique": "brush", "direction": "Down", "strings": [5], "confidence": .99,
                                    "stringMembershipConfidence": {str(i): float(i == 5) for i in range(1, 7)}}],
                }
                if has_weak_attack:
                    document["notes"].append({**note(.75, .5), "confidence": .7})
                cleaned, _ = clean_hypotheses(document)
                result, report = infer_notated_timing(cleaned, evidence([0., .5, 1., 1.5, 2.], [0., 2.]))
                self.assertEqual(len(result["notes"]), 2 if has_weak_attack else 1)
                self.assertEqual(len(report["onsetOptimization"]["deferredUnsupportedCandidates"]), 0 if has_weak_attack else 1)

    def test_three_downstroke_components_survive_when_only_the_landing_has_a_strong_note(self):
        document = {
            "metadata": metadata(), "audioDurationSeconds": 3, "percussion": [],
            "notes": [note(2., .5), *[
                {**note(t, .125), "uncertainty": ["technique_membership_completed_attack"],
                 "completionParent": {"technique": "brush", "onsetSeconds": t}} for t in (1.875, 1.9375)
            ]],
            "techniques": [{"onsetSeconds": t, "technique": "brush", "direction": "Down"} for t in (1.875, 1.9375, 2.)],
            "acousticAttackEvidence": [{"onsetSeconds": 2., "confidence": .99}],
        }
        result, report = infer_notated_timing(document, evidence([i / 2 for i in range(7)], [0., 2.]))
        self.assertEqual(sorted(Fraction(*n["scoreOnsetQuarter"]) for n in result["notes"]),
                         [Fraction(15, 4), Fraction(31, 8), Fraction(4)])
        self.assertFalse(report["onsetOptimization"]["deferredUnsupportedCandidates"])

    def test_duration_normalization_bridges_only_a_small_predicted_gap(self):
        for following, expected in ((.5, [1, 1]), (1., [3, 4])):
            document = {"metadata": metadata(), "audioDurationSeconds": 3, "percussion": [], "techniques": [],
                        "notes": [note(0., .8), note(following, 1.)]}
            result, _ = infer_notated_timing(document, evidence([i / 2 for i in range(7)], [0., 2.]))
            self.assertEqual(result["notes"][0]["scoreDurationQuarter"], expected)


if __name__ == "__main__":
    unittest.main()
