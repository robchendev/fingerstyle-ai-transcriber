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

    def test_malformed_evidence_fails(self):
        with self.assertRaises(HarnessError):
            PerformanceMap(evidence([0, 0], [0]), metadata(), 1)
        with self.assertRaises(HarnessError):
            PerformanceMap(evidence([0, .5], [.25]), metadata(), 1)


if __name__ == "__main__":
    unittest.main()
