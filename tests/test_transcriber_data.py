from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.transcriber_data import EpochShuffleSampler, collate_windows, encode_targets, training_conditioning


VOCABULARY = SimpleNamespace(PERCUSSION_TYPES=("wrist_thump", "thumb_slap", "percussive_hit"), HARMONIC_TYPES=("Natural", "Artificial", "Tap", "Pinch"), HARMONIC_FRETS=(5, 7, 9, 12, 19, 24))
MODEL = SimpleNamespace(max_fret=36, max_voices=4)


class TargetEncodingTests(unittest.TestCase):
    def test_sample_clock_roundoff_is_not_real_timing_extrapolation(self):
        from scripts.transcriber_audio import HarnessError
        from tests.test_score_alignment import clock_fixture

        labels, normalization = clock_fixture()
        labels["conditioning"]["instrument"] = {"openStringMidi": [40, 45, 50, 55, 59, 64], "capoFret": 0}
        labels["conditioning"]["providedTiming"]["sourceTimeSignatureChanges"] = []
        record = {
            "data": SimpleNamespace(labels=labels, normalization=normalization),
            "candidate": {"denseMapping": [{"clipSeconds": .35000000000000003, "scoreQuarter": 0.}, {"clipSeconds": 6., "scoreQuarter": 6.}]},
        }
        self.assertEqual(training_conditioning(record, np.array([.35, .37])).shape, (2, 12))
        with self.assertRaisesRegex(HarnessError, "outside"):
            training_conditioning(record, np.array([.35 - 1 / 48000, .37]))

    def test_confirmed_percussion_negatives_keep_gaps_boundaries_and_close_positives(self):
        gestures = [
            {"sourceGestureId": name, "technique": "wrist_thump", "onsetWindowSeconds": time, "supervisionMask": {"gesture": True, "onset": True}}
            for name, time in (("a", .8), ("b", .84))
        ]
        source = {"targets": {"notes": [], "gestures": [{"id": name, "technique": "wrist_thump"} for name in ("a", "b")]}}
        window = {"targets": {"notes": [], "gestures": gestures, "negativePercussionSupervision": True, "percussionAnnotationCoverage": [[.5, 1.], [1.2, 1.8]]}}
        with patch.dict("sys.modules", {"scripts.transcriber_model": VOCABULARY}):
            targets, masks, _ = encode_targets(window, source, np.arange(100) * .02, MODEL, negative_onsets_allowed=[True] * 6)
        self.assertTrue(masks["percussion"][35].all())
        self.assertFalse(targets["percussion"][35].any())
        self.assertFalse(masks["percussion"][55].any())
        self.assertFalse(masks["percussion"][5].any())
        self.assertFalse(masks["percussion"][90].any())
        self.assertFalse(masks["percussion"][39, 0])
        self.assertTrue(masks["percussion"][40, 0])
        self.assertTrue(masks["percussion"][42, 0])
        self.assertEqual(int(targets["percussion"].sum()), 2)

    def source_note(self, identifier="n", **changes):
        return {"id": identifier, "labelMask": {"attack": True, "pitch": True, "fingering": True, "notatedDuration": True}, "sourceSegments": [{"harmonic": {"type": "Natural", "fret": [12, 1]}}], **changes}

    def projected(self, identifier="n", **changes):
        return {
            "sourceNoteId": identifier, "string": 6, "fret": 12, "soundingPitchMidi": 52, "voiceIndex": 2,
            "onsetWindowSeconds": .8, "notatedDurationQuarter": [2, 1],
            "sourceLabelMask": {"attack": True, "pitch": True, "fingering": True, "notatedDuration": True},
            "supervisionMask": {"onset": True, "pitch": True, "fingering": True, "notatedDuration": True}, **changes,
        }

    def encode(self, notes, source_notes, gestures=(), source_gestures=()):
        with patch.dict("sys.modules", {"scripts.transcriber_model": VOCABULARY}):
            return encode_targets({"targets": {"notes": notes, "gestures": list(gestures)}}, {"targets": {"notes": source_notes, "gestures": list(source_gestures)}}, np.arange(100) * .02, MODEL, negative_onsets_allowed=[False, True, True, True, True, True])

    def test_voice_harmonic_and_positive_percussion_can_coexist(self):
        gesture = {"sourceGestureId": "g", "technique": "wrist_thump", "onsetWindowSeconds": .8, "supervisionMask": {"gesture": True, "onset": True}}
        targets, masks, collisions = self.encode([self.projected()], [self.source_note()], [gesture], [{"id": "g", "technique": "wrist_thump"}])
        self.assertEqual(targets["voice"][40, 0], 2)
        self.assertEqual(targets["harmonic_kind"][40, 0], 0)
        self.assertEqual(targets["harmonic_node"][40, 0], 3)
        self.assertTrue(masks["harmonic"][40, 0])
        self.assertEqual(targets["percussion"][40, 0], 1)
        self.assertEqual(int(masks["percussion"].sum()), 1)
        self.assertFalse(torch.any(masks["note_onset"][:40, 0]))
        self.assertTrue(masks["note_onset"][40, 0])
        self.assertTrue(masks["note_onset"][60, 1])
        self.assertEqual(collisions, 0)

    def test_collision_masks_categories_but_preserves_attack_presence(self):
        notes = [self.projected("a"), self.projected("b", fret=7)]
        targets, masks, collisions = self.encode(notes, [self.source_note("a"), self.source_note("b")])
        self.assertEqual(collisions, 1)
        self.assertEqual(targets["note_onset"][40, 0], 1)
        self.assertTrue(masks["note_onset"][40, 0])
        for name in ("fret", "pitch", "voice", "duration_log", "harmonic"):
            self.assertFalse(masks[name][40, 0])

    def test_carry_in_and_unknown_onsets_are_not_new_attacks(self):
        note = self.projected(supervisionMask={"onset": False, "pitch": False, "fingering": False, "notatedDuration": False})
        targets, masks, _ = self.encode([note], [self.source_note()])
        self.assertEqual(int(targets["note_onset"].sum()), 0)
        self.assertEqual(int(masks["harmonic"].sum()), 0)
        self.assertEqual(int(masks["pitch"].sum()), 0)

    def test_sampler_is_reproducible_by_epoch_without_consuming_global_rng(self):
        sampler = EpochShuffleSampler(range(12), seed=9)
        state = torch.random.get_rng_state().clone()
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))
        torch.testing.assert_close(state, torch.random.get_rng_state())

    def test_collation_masks_padding_and_keeps_metadata_out_of_features(self):
        items = []
        for length in (3, 5):
            items.append({"features": torch.ones(length, 16), "conditioning": torch.ones(length, 12), "targets": {"note_onset": torch.ones(length, 6)}, "masks": {"note_onset": torch.ones(length, 6, dtype=torch.bool)}, "metadata": {"private_id": str(length)}})
        batch = collate_windows(items)
        self.assertEqual(batch["features"].shape, (2, 5, 16))
        self.assertFalse(batch["valid_frames"][0, 3])
        self.assertFalse(batch["masks"]["note_onset"][0, 3].any())
        self.assertEqual(batch["metadata"], [{"private_id": "3"}, {"private_id": "5"}])


if __name__ == "__main__":
    unittest.main()
