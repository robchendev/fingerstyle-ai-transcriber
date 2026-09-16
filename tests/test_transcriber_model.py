from dataclasses import FrozenInstanceError, asdict
import json
import math
import unittest

import torch

from scripts.transcriber_model import (
    FingerstyleTranscriber, HARMONIC_FRETS, HARMONIC_TYPES, LOSS_STAT_KEYS,
    LOSS_WEIGHTS, ModelConfig, PERCUSSION_TYPES, PRESENCE_CALIBRATION,
    decode_events, masked_loss,
)


def synthetic_outputs(batch=2, frames=5, *, requires_grad=False):
    shapes = {
        "note_onset_logits": (6,),
        "fret_logits": (6, 37),
        "pitch_logits": (6, 128),
        "voice_logits": (6, 4),
        "duration_log": (6,),
        "harmonic_logits": (6,),
        "harmonic_kind_logits": (6, 4),
        "harmonic_node_logits": (6, 6),
        "percussion_logits": (3,),
    }
    return {name: torch.zeros(batch, frames, *shape, requires_grad=requires_grad) for name, shape in shapes.items()}


def synthetic_targets(batch=2, frames=5):
    names = ("note_onset", "fret", "pitch", "voice", "duration_log", "harmonic",
             "harmonic_kind", "harmonic_node", "percussion")
    categorical = ("fret", "pitch", "voice", "harmonic_kind", "harmonic_node")
    targets = {
        name: torch.full((batch, frames, 3 if name == "percussion" else 6),
                         -999 if name in categorical else float("nan"),
                         dtype=torch.long if name in categorical else torch.float32)
        for name in names
    }
    masks = {name: torch.zeros_like(value, dtype=torch.bool) for name, value in targets.items()}
    return targets, masks, torch.ones(batch, frames, dtype=torch.bool)


def event_outputs(frames):
    outputs = {name: value[0] for name, value in synthetic_outputs(1, frames).items()}
    for name in ("note_onset_logits", "harmonic_logits", "percussion_logits"):
        outputs[name].fill_(-12)
    outputs["duration_log"].fill_(math.log1p(1))
    return outputs


def choose_category(outputs, name, frame, axis, category):
    outputs[name][frame, axis].fill_(-12)
    outputs[name][frame, axis, category] = 12


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_config_is_validated_frozen_and_serializable(self):
        config = ModelConfig()
        self.assertEqual(asdict(config), {
            "n_mels": 96, "conditioning_dim": 12, "hidden_size": 128,
            "recurrent_layers": 2, "max_fret": 36, "max_voices": 4, "dropout": .1,
        })
        self.assertEqual(ModelConfig(**json.loads(json.dumps(asdict(config)))), config)
        with self.assertRaises(FrozenInstanceError):
            config.max_fret = 10
        for options in ({"n_mels": 0}, {"conditioning_dim": 11}, {"hidden_size": 0},
                        {"recurrent_layers": -1}, {"max_fret": -1}, {"max_fret": 128},
                        {"max_voices": 0}, {"dropout": 1}, {"dropout": -1}, {"dropout": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ModelConfig(**options)
        for options in ({"n_mels": True}, {"max_fret": 2.5}, {"dropout": "0.1"}):
            with self.subTest(options=options), self.assertRaises(TypeError):
                ModelConfig(**options)
        with self.assertRaises(TypeError):
            FingerstyleTranscriber({})

    def test_variable_shapes_and_finite_backward(self):
        for batch, frames, mels in ((1, 1, 1), (3, 9, 11), (2, 4, 96)):
            with self.subTest(batch=batch, frames=frames, mels=mels):
                config = ModelConfig(n_mels=mels, hidden_size=8, recurrent_layers=2,
                                     max_fret=8, max_voices=3, dropout=0)
                model = FingerstyleTranscriber(config)
                features = torch.randn(batch, frames, mels, requires_grad=True)
                conditioning = torch.randn(batch, frames, 12, requires_grad=True)
                lengths = torch.arange(batch) % frames + 1
                outputs = model(features, conditioning, lengths)
                expected = {
                    "note_onset_logits": (6,), "fret_logits": (6, 9),
                    "pitch_logits": (6, 128), "voice_logits": (6, 3),
                    "duration_log": (6,), "harmonic_logits": (6,),
                    "harmonic_kind_logits": (6, 4), "harmonic_node_logits": (6, 6),
                    "percussion_logits": (3,),
                }
                self.assertEqual(set(outputs), set(expected))
                for name, tail in expected.items():
                    self.assertEqual(outputs[name].shape, (batch, frames, *tail))
                    self.assertTrue(torch.isfinite(outputs[name]).all())
                self.assertTrue((outputs["duration_log"] >= 0).all())
                sum(value.square().mean() for value in outputs.values()).backward()
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertTrue(torch.isfinite(features.grad).all())
                self.assertTrue(torch.isfinite(conditioning.grad).all())
                for row, length in enumerate(lengths):
                    self.assertEqual(torch.count_nonzero(features.grad[row, length:]).item(), 0)
                    self.assertEqual(torch.count_nonzero(conditioning.grad[row, length:]).item(), 0)

    def test_padding_does_not_change_real_frames(self):
        model = FingerstyleTranscriber(ModelConfig(n_mels=9, hidden_size=8, recurrent_layers=1, dropout=0)).eval()
        features = torch.randn(2, 8, 9)
        conditioning = torch.randn(2, 8, 12)
        padded = model(features, conditioning, torch.tensor([3, 8]))
        short = model(features[:1, :3], conditioning[:1, :3])
        changed_features, changed_conditioning = features.clone(), conditioning.clone()
        changed_features[0, 3:] = 1000
        changed_conditioning[0, 3:] = -1000
        changed = model(changed_features, changed_conditioning, torch.tensor([3, 8]))
        for name in padded:
            torch.testing.assert_close(padded[name][0, :3], short[name][0], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(padded[name], changed[name])
            self.assertEqual(torch.count_nonzero(padded[name][0, 3:]).item(), 0)

    def test_conditioning_and_frequency_order_affect_outputs(self):
        torch.manual_seed(7)
        model = FingerstyleTranscriber(ModelConfig(n_mels=12, hidden_size=8, recurrent_layers=1, dropout=0)).eval()
        features = torch.zeros(1, 5, 12)
        features[:, :, 1] = 4
        conditioning = torch.zeros(1, 5, 12)
        baseline = model(features, conditioning)["pitch_logits"]
        conditioned = model(features, conditioning + 1)["pitch_logits"]
        permuted = model(features.flip(-1), conditioning)["pitch_logits"]
        self.assertGreater((baseline - conditioned).abs().max().item(), 1e-5)
        self.assertGreater((baseline - permuted).abs().max().item(), 1e-5)

    def test_invalid_forward_inputs_fail_explicitly(self):
        model = FingerstyleTranscriber(ModelConfig(n_mels=8, hidden_size=4, recurrent_layers=1))
        features, conditioning = torch.zeros(2, 3, 8), torch.zeros(2, 3, 12)
        for feature, condition, length, error in (
            ([], conditioning, None, TypeError),
            (features[:, 0], conditioning, None, ValueError),
            (features[:, :0], conditioning[:, :0], None, ValueError),
            (features[:, :, :7], conditioning, None, ValueError),
            (features, conditioning[:, :, :11], None, ValueError),
            (features.long(), conditioning, None, TypeError),
            (features, conditioning.double(), None, TypeError),
            (features, conditioning, torch.tensor([1., 2.]), TypeError),
            (features, conditioning, torch.tensor([1]), ValueError),
            (features, conditioning, torch.tensor([0, 2]), ValueError),
            (features, conditioning, torch.tensor([4, 2]), ValueError),
            (features + float("nan"), conditioning, None, ValueError),
            (features, conditioning + float("inf"), None, ValueError),
        ):
            with self.subTest(error=error, length=length), self.assertRaises(error):
                model(feature, condition, length)


class MaskedLossTests(unittest.TestCase):
    def test_all_masked_is_differentiable_zero_with_no_priors(self):
        outputs = synthetic_outputs(requires_grad=True)
        targets, masks, valid = synthetic_targets()
        loss, stats = masked_loss(outputs, targets, masks, valid)
        self.assertEqual(loss.item(), 0)
        self.assertEqual(set(stats), set(LOSS_STAT_KEYS))
        self.assertTrue(all(value == {"sum": 0.0, "count": 0} for value in stats.values()))
        loss.backward()
        for value in outputs.values():
            self.assertIsNotNone(value.grad)
            self.assertEqual(torch.count_nonzero(value.grad).item(), 0)

    def test_unknown_and_padded_targets_have_zero_gradient(self):
        outputs = synthetic_outputs(requires_grad=True)
        targets, masks, valid = synthetic_targets()
        valid[1, 2:] = False
        for name in targets:
            targets[name][0, 1, 0] = 1
            masks[name][0, 1, 0] = True
            masks[name][1, 2:] = True
        loss, stats = masked_loss(outputs, targets, masks, valid, sparsity_weight=0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, value in outputs.items():
            expected = torch.zeros_like(value, dtype=torch.bool)
            expected[0, 1, 0] = True
            self.assertEqual(torch.count_nonzero(value.grad[~expected]).item(), 0, name)
            self.assertGreater(torch.count_nonzero(value.grad[expected]).item(), 0, name)
        self.assertEqual(stats["note_onset_positive"]["count"], 1)
        self.assertEqual(stats["note_onset_negative"]["count"], 0)

    def test_note_onset_balances_known_sides_not_label_counts(self):
        outputs = synthetic_outputs(1, 3)
        targets, masks, valid = synthetic_targets(1, 3)
        targets["note_onset"].fill_(0)
        masks["note_onset"].fill_(True)
        targets["note_onset"][0, 0, 0] = 1
        outputs["note_onset_logits"].fill_(1)
        outputs["note_onset_logits"][0, 0, 0] = 2
        loss, stats = masked_loss(outputs, targets, masks, valid)
        positive = torch.nn.functional.softplus(torch.tensor(-2.)).item()
        negative = torch.nn.functional.softplus(torch.tensor(1.)).item()
        self.assertAlmostEqual(loss.item(), (positive + negative) / 2, places=6)
        self.assertAlmostEqual(stats["note_onset_positive"]["sum"], positive, places=6)
        self.assertEqual(stats["note_onset_positive"]["count"], 1)
        self.assertEqual(stats["note_onset_negative"]["count"], 17)
        masks["note_onset"] = targets["note_onset"].bool()
        positive_only, stats = masked_loss(outputs, targets, masks, valid)
        self.assertAlmostEqual(positive_only.item(), positive, places=6)
        self.assertEqual(stats["note_onset_negative"]["count"], 0)
        masks["note_onset"] = ~targets["note_onset"].bool()
        negative_only, stats = masked_loss(outputs, targets, masks, valid)
        self.assertAlmostEqual(negative_only.item(), negative, places=6)
        self.assertEqual(stats["note_onset_positive"]["count"], 0)

    def test_positive_only_heads_reject_masked_negatives(self):
        for name in ("harmonic", "percussion"):
            outputs = synthetic_outputs()
            targets, masks, valid = synthetic_targets()
            targets[name][0, 0, 0] = 0
            masks[name][0, 0, 0] = True
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "positive-only"):
                masked_loss(outputs, targets, masks, valid)

    def test_priors_are_separate_and_only_use_evidenced_classes_and_known_attacks(self):
        outputs = synthetic_outputs(1, 4, requires_grad=True)
        targets, masks, valid = synthetic_targets(1, 4)
        valid[0, 3] = False
        for frame in (0, 2):
            targets["note_onset"][0, frame, 0] = 1
            masks["note_onset"][0, frame, 0] = True
        targets["harmonic"][0, 0, 0] = 1
        masks["harmonic"][0, 0, 0] = True
        targets["percussion"][0, 1, 1] = 1
        masks["percussion"][0, 1, 1] = True
        original_masks = {name: value.clone() for name, value in masks.items()}
        with_prior, stats = masked_loss(outputs, targets, masks, valid, sparsity_weight=.2)
        without_prior, _ = masked_loss(outputs, targets, masks, valid, sparsity_weight=0)
        self.assertAlmostEqual((with_prior - without_prior).item(), .2, places=6)
        self.assertEqual(stats["harmonic_sparsity"], {"sum": 1.0, "count": 2})
        self.assertEqual(stats["percussion_sparsity"], {"sum": 1.5, "count": 3})
        self.assertEqual(stats["harmonic_positive"]["count"], 1)
        self.assertEqual(stats["percussion_positive"]["count"], 1)
        self.assertEqual(stats["note_onset_negative"]["count"], 0)
        with_prior.backward()
        self.assertGreater(outputs["harmonic_logits"].grad[0, 2, 0].item(), 0)
        self.assertEqual(outputs["harmonic_logits"].grad[0, 1, 0].item(), 0)
        self.assertGreater(outputs["percussion_logits"].grad[0, 0, 1].item(), 0)
        self.assertEqual(torch.count_nonzero(outputs["percussion_logits"].grad[:, :, [0, 2]]).item(), 0)
        self.assertEqual(torch.count_nonzero(outputs["percussion_logits"].grad[:, 3:]).item(), 0)
        for name in masks:
            torch.testing.assert_close(masks[name], original_masks[name])

    def test_no_positive_evidence_disables_presence_priors(self):
        outputs = synthetic_outputs(requires_grad=True)
        targets, masks, valid = synthetic_targets()
        targets["note_onset"][0, 0, 0] = 1
        masks["note_onset"][0, 0, 0] = True
        loss, stats = masked_loss(outputs, targets, masks, valid)
        loss.backward()
        for name in ("harmonic", "percussion"):
            self.assertEqual(stats[f"{name}_positive"]["count"], 0)
            self.assertEqual(stats[f"{name}_sparsity"]["count"], 0)
            self.assertEqual(torch.count_nonzero(outputs[f"{name}_logits"].grad).item(), 0)
        targets["harmonic"][0, 0, 0] = 1
        masks["harmonic"][0, 0, 0] = True
        all_padded, stats = masked_loss(outputs, targets, masks, torch.zeros_like(valid))
        self.assertEqual(all_padded.item(), 0)
        self.assertTrue(all(value["count"] == 0 for value in stats.values()))

    def test_category_collision_masks_remain_independent(self):
        outputs = synthetic_outputs(1, 1, requires_grad=True)
        targets, masks, valid = synthetic_targets(1, 1)
        for name in ("note_onset", "harmonic"):
            targets[name][0, 0, 0] = 1
            masks[name][0, 0, 0] = True
        targets["pitch"][0, 0, 0] = 60
        masks["pitch"][0, 0, 0] = True
        # A collision can leave pitch known while fret/voice/kind/node remain
        # unknown. Their sentinel labels must never reach cross entropy.
        loss, stats = masked_loss(outputs, targets, masks, valid, sparsity_weight=0)
        loss.backward()
        self.assertEqual(stats["pitch"]["count"], 1)
        for name in ("fret", "voice", "harmonic_kind", "harmonic_node"):
            self.assertEqual(stats[name]["count"], 0)
            self.assertEqual(torch.count_nonzero(outputs[f"{name}_logits"].grad).item(), 0)

    def test_sums_and_counts_reconstruct_component_objective(self):
        outputs = synthetic_outputs(1, 2)
        targets, masks, valid = synthetic_targets(1, 2)
        for name in targets:
            targets[name][0, 0, 0] = 1
            masks[name][0, 0, 0] = True
        targets["note_onset"][0, 1, 0] = 0
        masks["note_onset"][0, 1, 0] = True
        loss, stats = masked_loss(outputs, targets, masks, valid)
        means = {name: stat["sum"] / stat["count"] if stat["count"] else 0 for name, stat in stats.items()}
        reconstructed = LOSS_WEIGHTS["note_onset"] * (means["note_onset_positive"] + means["note_onset_negative"]) / 2
        reconstructed += sum(weight * means[name] for name, weight in LOSS_WEIGHTS.items() if name != "note_onset")
        reconstructed += .02 * (means["harmonic_sparsity"] + means["percussion_sparsity"])
        self.assertAlmostEqual(loss.item(), reconstructed, places=5)
        for stat in stats.values():
            self.assertIsInstance(stat["sum"], float)
            self.assertIsInstance(stat["count"], int)

    def test_invalid_loss_inputs_are_explicit(self):
        outputs = synthetic_outputs()
        targets, masks, valid = synthetic_targets()
        for weight in (-1, float("nan"), float("inf")):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                masked_loss(outputs, targets, masks, valid, sparsity_weight=weight)
        with self.assertRaises(TypeError):
            masked_loss(outputs, targets, masks, valid.float())
        with self.assertRaises(ValueError):
            masked_loss(outputs, targets, masks, valid[:, :1])
        with self.assertRaises(ValueError):
            masked_loss({}, targets, masks, valid)
        with self.assertRaises(ValueError):
            masked_loss(outputs, {}, masks, valid)
        with self.assertRaises(ValueError):
            masked_loss(outputs, targets, {}, valid)
        for name, invalid, error in (("fret", -1, ValueError), ("pitch", 128, ValueError),
                                     ("voice", 4, ValueError), ("harmonic_kind", 4, ValueError),
                                     ("harmonic_node", 6, ValueError), ("duration_log", -1, ValueError),
                                     ("note_onset", .5, ValueError), ("harmonic", float("inf"), ValueError)):
            local_targets, local_masks, _ = synthetic_targets()
            local_targets[name][0, 0, 0] = invalid
            local_masks[name][0, 0, 0] = True
            with self.subTest(name=name), self.assertRaises(error):
                masked_loss(outputs, local_targets, local_masks, valid)
        wrong_masks = dict(masks, fret=masks["fret"].float())
        wrong_targets = dict(targets, pitch=targets["pitch"].float())
        with self.assertRaises(TypeError):
            masked_loss(outputs, targets, wrong_masks, valid)
        with self.assertRaises(TypeError):
            masked_loss(outputs, wrong_targets, masks, valid)


class EventDecoderTests(unittest.TestCase):
    tuning = (40, 45, 50, 55, 59, 64)

    def decode(self, outputs, seconds, **kwargs):
        return decode_events(outputs, seconds, tuning=self.tuning, capo=0, **kwargs)

    def test_voiced_notes_and_percussion_coexist_without_carrier_fingering(self):
        outputs = event_outputs(4)
        outputs["note_onset_logits"][1, [0, 5]] = 8
        outputs["percussion_logits"][1, :2] = 8
        choose_category(outputs, "fret_logits", 1, 0, 3)
        choose_category(outputs, "pitch_logits", 1, 0, 43)
        choose_category(outputs, "voice_logits", 1, 0, 1)
        choose_category(outputs, "pitch_logits", 1, 5, 64)
        decoded = self.decode(outputs, [2, 2.02, 2.04, 2.06])
        self.assertEqual(len(decoded["notes"]), 2)
        self.assertEqual([note["string"] for note in decoded["notes"]], [6, 1])
        self.assertEqual([note["voiceIndex"] for note in decoded["notes"]], [1, 0])
        self.assertAlmostEqual(decoded["notes"][0]["notatedDurationQuarter"], 1, places=6)
        self.assertIsNone(decoded["notes"][0]["harmonic"])
        self.assertEqual(decoded["notes"][0]["uncertainty"], [])
        self.assertEqual([item["technique"] for item in decoded["percussion"]], list(PERCUSSION_TYPES[:2]))
        for item in decoded["percussion"]:
            self.assertEqual(item["onsetSeconds"], 2.02)
            self.assertEqual(item["presenceCalibration"], PRESENCE_CALIBRATION)
            self.assertNotIn("string", item)
            self.assertNotIn("fret", item)
        json.dumps(decoded, allow_nan=False)

    def test_plateaus_repeated_attacks_and_late_audio_are_not_truncated(self):
        outputs = event_outputs(14)
        scores = torch.tensor([8, 8, -8, 9, -8, 9, -8, -8, 8, 8, 8, -8, -8, 10.])
        outputs["note_onset_logits"][:, 0] = scores
        outputs["percussion_logits"][:, 2] = scores
        seconds = [0., .01, .02, .03, .04, .05, .06, .07, 10., 10.01, 10.02, 10.03, 100., 101.]
        decoded = self.decode(outputs, seconds, min_gap_seconds=.04)
        expected = [.03, 10., 101.]
        self.assertEqual([note["onsetSeconds"] for note in decoded["notes"]], expected)
        self.assertEqual([item["onsetSeconds"] for item in decoded["percussion"]], expected)
        self.assertEqual(decoded, self.decode(outputs, seconds, min_gap_seconds=.04))

    def test_entire_plateau_is_one_attack_and_nms_is_per_string(self):
        outputs = event_outputs(6)
        outputs["note_onset_logits"][:, 0] = 5
        outputs["note_onset_logits"][[1, 4], 1] = 5
        decoded = self.decode(outputs, [4, 4.01, 4.02, 4.03, 4.2, 4.21])
        self.assertEqual([(note["onsetSeconds"], note["string"]) for note in decoded["notes"]],
                         [(4., 6), (4.01, 5), (4.2, 5)])

    def test_harmonic_sounding_pitch_uses_kind_node_tuning_and_capo(self):
        self.assertEqual(HARMONIC_FRETS, (5, 7, 9, 12, 19, 24))
        offsets = (24, 19, 28, 12, 19, 24)
        for kind_index, kind in enumerate(HARMONIC_TYPES):
            for node_index, offset in enumerate(offsets):
                with self.subTest(kind=kind, node=HARMONIC_FRETS[node_index]):
                    outputs = event_outputs(1)
                    outputs["note_onset_logits"][0, 0] = 8
                    outputs["harmonic_logits"][0, 0] = 8
                    choose_category(outputs, "fret_logits", 0, 0, 7)
                    choose_category(outputs, "harmonic_kind_logits", 0, 0, kind_index)
                    choose_category(outputs, "harmonic_node_logits", 0, 0, node_index)
                    expected = 40 + 2 + offset + (0 if kind == "Natural" else 7)
                    choose_category(outputs, "pitch_logits", 0, 0, expected)
                    note = decode_events(outputs, [0.], tuning=self.tuning, capo=2)["notes"][0]
                    self.assertEqual(note["soundingPitchMidi"], expected)
                    self.assertEqual(note["fretBasePitchMidi"], 49)
                    self.assertEqual(note["expectedSoundingPitchMidi"], expected)
                    self.assertEqual(note["harmonic"]["type"], kind)
                    self.assertEqual(note["harmonic"]["fret"], HARMONIC_FRETS[node_index])
                    self.assertNotIn("sounding_pitch_fret_mismatch", note["uncertainty"])
                    self.assertNotIn("sounding_pitch_harmonic_mismatch", note["uncertainty"])
                    choose_category(outputs, "pitch_logits", 0, 0, expected - 1)
                    inconsistent = decode_events(outputs, [0.], tuning=self.tuning, capo=2)["notes"][0]
                    self.assertEqual(inconsistent["soundingPitchMidi"], expected - 1)
                    self.assertIn("sounding_pitch_harmonic_mismatch", inconsistent["uncertainty"])

    def test_pitch_mismatch_is_reported_not_repaired_and_duration_is_bounded(self):
        outputs = event_outputs(1)
        outputs["note_onset_logits"][0, 0] = 8
        outputs["duration_log"][0, 0] = 1000
        choose_category(outputs, "fret_logits", 0, 0, 3)
        choose_category(outputs, "pitch_logits", 0, 0, 99)
        note = self.decode(outputs, [20.])["notes"][0]
        self.assertEqual(note["fret"], 3)
        self.assertEqual(note["soundingPitchMidi"], 99)
        self.assertEqual(note["notatedDurationQuarter"], 64)
        self.assertEqual(note["uncertainty"], ["sounding_pitch_fret_mismatch", "duration_clipped"])
        json.dumps(self.decode(outputs, [20.]), allow_nan=False)

    def test_empty_timeline_and_below_threshold_are_not_true_negatives(self):
        decoded = self.decode(event_outputs(0), [])
        self.assertEqual(decoded["notes"], [])
        self.assertEqual(decoded["percussion"], [])
        self.assertIn("not a confirmed negative", decoded["policy"]["presenceAbsence"])
        self.assertEqual(self.decode(event_outputs(3), torch.arange(3.))["notes"], [])

    def test_decoder_invalid_inputs_fail_explicitly(self):
        outputs = event_outputs(3)
        for options in ({"onset_threshold": -1}, {"onset_threshold": 1.01},
                        {"percussion_threshold": float("nan")}, {"harmonic_threshold": 2},
                        {"min_gap_seconds": -1}, {"max_duration_quarter": 0},
                        {"max_duration_quarter": float("inf")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.decode(outputs, [0, 1, 2], **options)
        with self.assertRaises(TypeError):
            self.decode(outputs, [0, 1, 2], onset_threshold=True)
        for seconds, error in (([0, 0, 1], ValueError), ([0, 2, 1], ValueError),
                               ([-1, 0, 1], ValueError), ([0, 1], ValueError),
                               ([0, float("nan"), 2], ValueError), ("abc", TypeError),
                               ([False, 1, 2], TypeError), (torch.ones(3, dtype=torch.bool), TypeError)):
            with self.subTest(seconds=seconds), self.assertRaises(error):
                self.decode(outputs, seconds)
        for tuning, capo, error in ((self.tuning[:5], 0, ValueError), (self.tuning, -1, ValueError),
                                    ([40.] * 6, 0, TypeError), ([127] * 6, 1, ValueError),
                                    (self.tuning, True, TypeError), ("abcdef", 0, TypeError)):
            with self.subTest(tuning=tuning, capo=capo), self.assertRaises(error):
                decode_events(outputs, [0, 1, 2], tuning=tuning, capo=capo)
        for name in outputs:
            malformed = dict(outputs)
            malformed[name] = outputs[name][:2]
            with self.subTest(head=name), self.assertRaises(ValueError):
                self.decode(malformed, [0, 1, 2])
        for replacement, error in ((torch.zeros(3, 6, dtype=torch.long), TypeError),
                                   (torch.full((3, 6), float("nan")), ValueError)):
            malformed = dict(outputs, note_onset_logits=replacement)
            with self.assertRaises(error):
                self.decode(malformed, [0, 1, 2])
        outputs["duration_log"][0, 0] = -1
        with self.assertRaises(ValueError):
            self.decode(outputs, [0, 1, 2])
        with self.assertRaises(ValueError):
            self.decode(synthetic_outputs(1, 3), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
