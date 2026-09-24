"""Synthetic structure/audio fixtures only; no private data or downloaded weights."""

import copy
from dataclasses import FrozenInstanceError, asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from scripts import transcriber_runtime as runtime
from scripts.transcriber_events import evaluate_events
from scripts.transcriber_model import FingerstyleTranscriber, ModelConfig, masked_loss
from scripts.transcriber_video import (
    ROLE_FEATURE_GROUPS, ROLE_FEATURE_GROUP_INDICES,
    AudioVideoTranscriber, VideoConfig,
)
from scripts.video_features import STRUCTURED_DIM, VELOCITY_SLICES, VIEW_ORDER
from tests.test_transcriber_events import WindowFixture
from tests.test_transcriber_model import synthetic_targets


ROOT = Path(__file__).resolve().parents[1]


def visual_inputs(batch=1, frames=5, count=5):
    generator = torch.Generator().manual_seed(81)
    result = {
        "technique_available": torch.ones(batch, count, dtype=torch.bool),
        "segment_id": torch.zeros(batch, count, len(VIEW_ORDER), dtype=torch.long),
        "structured": torch.randn(batch, count, len(VIEW_ORDER), STRUCTURED_DIM, generator=generator),
        "structured_available": torch.ones(batch, count, len(VIEW_ORDER), STRUCTURED_DIM, dtype=torch.bool),
        "frame_indices": torch.arange(frames).clamp_max(count - 1).expand(batch, -1).clone(),
    }
    result["structured_available"][:, :, 2:, 186:] = False
    result["structured"][:, :, 2:, 186:] = 0
    return result


def without_evidence(video):
    video["structured_available"].zero_()
    video["segment_id"].fill_(-1)
    return video


def small_model(version=2, seed=17, n_mels=8, *, modality_dropout=.2, audio_dropout=.3):
    torch.manual_seed(seed)
    audio = FingerstyleTranscriber(ModelConfig(
        architecture_version=version, n_mels=n_mels, hidden_size=8,
        recurrent_layers=1, max_fret=3, max_voices=2, dropout=audio_dropout,
    ))
    return AudioVideoTranscriber(audio, VideoConfig(hidden_size=8, modality_dropout=modality_dropout))


def supervised_batch(*, evidence=True, index=0):
    batch, frames = 1, 5
    targets, masks, valid = synthetic_targets(batch, frames, architecture_version=2)
    targets["note_onset"].zero_()
    targets["note_onset"][:, 1, 0] = 1
    masks["note_onset"].fill_(True)
    for name, value in (("fret", 3), ("pitch", 43)):
        targets[name][:, 1, 0] = value
        masks[name][:, 1, 0] = True
    for name in ("percussion", "technique"):
        targets[name].zero_()
        targets[name][:, 2, 0] = 1
        masks[name].fill_(True)
    targets["technique_direction"][:, 2, 0] = 0
    masks["technique_direction"][:, 2, 0] = True
    targets["technique_strings"][:, 2, 0] = torch.tensor([1., 0., 1., 0., 0., 0.])
    masks["technique_strings"][:, 2, 0] = True
    video = visual_inputs()
    if not evidence:
        without_evidence(video)
    return {
        "features": torch.linspace(-.5, .5, frames * 8).reshape(batch, frames, 8) + index / 10,
        "conditioning": torch.zeros(batch, frames, 12),
        "lengths": torch.tensor([frames]),
        "targets": targets, "masks": masks, "valid_frames": valid, "video": video,
    }


class PairedSyntheticDataset(Dataset):
    """Genuine paired windows, including locally unavailable tracking evidence."""

    def __init__(self, available=(False, True, False, True)):
        self.available = available

    def __len__(self):
        return len(self.available)

    def __getitem__(self, index):
        batch = supervised_batch(evidence=self.available[index], index=index)
        return {
            name: {key: value[0] for key, value in item.items()} if isinstance(item, dict) else item[0]
            for name, item in batch.items()
        }


def paired_loaders(dataset):
    return tuple(DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    ) for seed in (101, 202))


class VideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        rng = runtime._capture_rng()
        self.addCleanup(runtime._restore_rng, rng)
        self.model = small_model().eval()
        self.features = torch.linspace(-1, 1, 40).reshape(1, 5, 8)
        self.conditioning = torch.zeros(1, 5, 12)
        self.video = visual_inputs()

    def forward(self, video=None, *, model=None, lengths=None):
        return (self.model if model is None else model)(
            self.features, self.conditioning, lengths, video=video,
        )

    def assert_outputs_equal(self, left, right):
        self.assertEqual(set(left), set(right))
        for name in left:
            self.assertTrue(torch.equal(left[name], right[name]), name)

    def test_config_is_explicit_joint_and_json_serializable(self):
        config = VideoConfig()
        self.assertEqual(asdict(config), {
            "hidden_size": 64, "temporal_layers": 1, "structured_dim": 194,
            "input_schema_version": 4, "architecture_version": 5, "modality_dropout": .2,
            "feature_group_version": "anatomy-representation-groups-v1",
        })
        self.assertEqual(VideoConfig(**json.loads(json.dumps(asdict(config)))), config)
        with self.assertRaises(FrozenInstanceError):
            config.hidden_size = 12
        for options in (
            {"hidden_size": 0}, {"temporal_layers": 0},
            {"structured_dim": 98}, {"structured_dim": 186}, {"structured_dim": 193}, {"structured_dim": 195},
            {"input_schema_version": 1}, {"input_schema_version": 2}, {"input_schema_version": 3},
            {"architecture_version": 1}, {"architecture_version": 2}, {"architecture_version": 3}, {"architecture_version": 4},
            {"feature_group_version": "unknown"},
            {"modality_dropout": -.1}, {"modality_dropout": 1.1},
            {"modality_dropout": float("nan")}, {"modality_dropout": float("inf")},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                VideoConfig(**options)
        with self.assertRaises(TypeError):
            VideoConfig(hidden_size=True)
        for options in ({"freeze_audio": True}, {"modality_dropout": True}, {"modality_dropout": ".2"}):
            with self.subTest(options=options), self.assertRaises(TypeError):
                VideoConfig(**options)
        with self.assertRaises(TypeError):
            AudioVideoTranscriber(torch.nn.Linear(2, 2), config)
        with self.assertRaises(TypeError):
            AudioVideoTranscriber(self.model.audio, {})

    def test_contract_and_joint_modules_are_structure_only_and_reject_old_rgb(self):
        self.assertEqual(set(self.video), {
            "structured", "structured_available", "technique_available", "segment_id", "frame_indices",
        })
        self.assertFalse(hasattr(self.model.video_config, "image_size"))
        for branch in (self.model.position_branch, self.model.technique_branch, self.model.anonymous_branch):
            self.assertFalse(hasattr(branch, "image_encoder"))
            self.assertFalse(any(isinstance(module, torch.nn.Conv2d) for module in branch.modules()))
        with self.assertRaisesRegex(TypeError, "image_size"):
            VideoConfig(image_size=96)
        for name, value in (
            ("images", torch.zeros(1, 5, 2, 3, 8, 8)),
            ("rgb", torch.zeros(1, 5, 2, 3, 8, 8)),
            ("available", torch.ones(1, 5, 2, dtype=torch.bool)),
        ):
            with self.subTest(obsolete=name), self.assertRaisesRegex(ValueError, "RGB fields are unsupported"):
                self.forward({**self.video, name: value})
        incompatible = copy.deepcopy(self.model.state_dict())
        incompatible["position_branch.image_encoder.0.weight"] = torch.zeros(16, 3, 3, 3)
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            self.model.load_state_dict(incompatible, strict=True)

    def test_role_feature_groups_partition_d194_and_use_role_specific_nonzero_biases(self):
        flattened = [index for group in ROLE_FEATURE_GROUP_INDICES for index in group]
        self.assertEqual(len(ROLE_FEATURE_GROUPS), 8)
        self.assertEqual(sorted(flattened), list(range(194)))
        self.assertEqual(len(flattened), len(set(flattened)))
        fretting = self.model.position_branch.feature_gates.log_scales.exp()
        plucking = self.model.technique_branch.feature_gates.log_scales.exp()
        anonymous = self.model.anonymous_branch.feature_gates.log_scales.exp()
        fingertip_position = ROLE_FEATURE_GROUPS.index("fingertip_position")
        thumb_motion = ROLE_FEATURE_GROUPS.index("thumb_motion")
        fingertip_motion = ROLE_FEATURE_GROUPS.index("fingertip_motion")
        self.assertGreater(fretting[fingertip_position], fretting[thumb_motion])
        self.assertGreater(plucking[thumb_motion], 1)
        self.assertGreater(plucking[fingertip_motion], 1)
        torch.testing.assert_close(anonymous, torch.ones_like(anonymous))
        for branch in (
            self.model.position_branch, self.model.technique_branch,
            self.model.anonymous_branch,
        ):
            self.assertTrue(branch.feature_gates.log_scales.requires_grad)

    def test_role_gates_learn_without_turning_masked_values_into_evidence(self):
        model = small_model(version=4, modality_dropout=0, audio_dropout=0).train()
        video = copy.deepcopy(self.video)
        video["structured"].requires_grad_(True)
        outputs = model(self.features, self.conditioning, video=video)
        outputs["fret_logits"].square().mean().backward()
        for branch in (model.position_branch, model.technique_branch, model.anonymous_branch):
            self.assertGreater(branch.feature_gates.log_scales.grad.abs().sum().item(), 0)
        masked = without_evidence(copy.deepcopy(self.video))
        masked["structured"].fill_(1e20)
        self.assert_outputs_equal(
            model(self.features, self.conditioning),
            model(self.features, self.conditioning, video=masked),
        )

    def test_no_visual_evidence_exactly_uses_audio_core_outputs(self):
        model = small_model(version=4).eval()
        with torch.no_grad():
            hidden, valid = model.audio.encode(self.features, self.conditioning)
            expected = model.audio.decode_hidden(hidden, valid)
        for video in (None, without_evidence(copy.deepcopy(self.video))):
            self.assert_outputs_equal(
                expected,
                model(self.features, self.conditioning, video=video),
            )

    def test_all_architectures_fuse_before_all_original_heads_once(self):
        for version in (1, 2, 3, 4):
            with self.subTest(version=version):
                model = small_model(version).eval()
                base = self.forward(model=model)
                with mock.patch.object(model.audio, "decode_hidden", wraps=model.audio.decode_hidden) as decode:
                    actual = self.forward(self.video, model=model)
                self.assertEqual(decode.call_count, 1)
                self.assertIs(model.config, model.audio.config)
                self.assertEqual(model.head_shapes, model.audio.head_shapes)
                self.assertTrue(any(name.startswith("audio.") for name in model.state_dict()))
                self.assertFalse(hasattr(model, "residual_heads"))
                for name, shape in model.head_shapes.items():
                    self.assertEqual(actual[name].shape, (1, 5, *shape))
                    self.assertTrue(torch.isfinite(actual[name]).all(), name)
                    self.assertFalse(torch.equal(base[name], actual[name]), name)
                self.assertTrue((actual["duration_log"] >= 0).all())

    def test_absent_all_masked_and_unmapped_frames_exactly_fall_back_after_learning(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.01)
        batch = supervised_batch()
        loss, _ = masked_loss(self.forward(self.video), batch["targets"], batch["masks"], batch["valid_frames"])
        loss.backward()
        optimizer.step()
        base = self.forward()
        for video in (None, without_evidence(copy.deepcopy(self.video))):
            self.assert_outputs_equal(base, self.forward(video))
        self.video["frame_indices"].fill_(-1)
        self.assert_outputs_equal(base, self.forward(self.video))
        self.video["frame_indices"][:] = torch.tensor([0, -1, 2, -1, 4])
        adjusted = self.forward(self.video)
        for name in adjusted:
            self.assertTrue(torch.equal(base[name][:, [1, 3]], adjusted[name][:, [1, 3]]), name)

    def test_known_and_anonymous_video_views_can_inform_every_v4_head(self):
        model = small_model(version=4).eval()
        original = self.forward(self.video, model=model)
        for view in range(4):
            changed = copy.deepcopy(self.video)
            changed["structured"][:, :, view] = 0
            altered = self.forward(changed, model=model)
            for name in original:
                with self.subTest(view=view, head=name):
                    self.assertFalse(torch.equal(original[name], altered[name]), name)
        video = copy.deepcopy(self.video)
        video["structured"].requires_grad_(True)
        outputs = self.forward(video, model=model)
        for name, output in outputs.items():
            gradient, = torch.autograd.grad(output.sum(), video["structured"], retain_graph=True)
            for view in range(4):
                with self.subTest(view=view, head=name):
                    self.assertGreater(gradient[:, :, view].abs().sum().item(), 0)

    def test_geometry_absent_anonymous_hands_and_audio_train_every_head(self):
        model = small_model(version=4, modality_dropout=0, audio_dropout=0).train()
        video = without_evidence(copy.deepcopy(self.video))
        video["structured_available"][:, :, 2:, 98:186] = True
        video["segment_id"][:, :, 2] = 11
        video["segment_id"][:, :, 3] = 22
        video["structured"].requires_grad_(True)
        features = self.features.clone().requires_grad_(True)
        outputs = model(features, self.conditioning, video=video)
        absent = model(features, self.conditioning)
        for name, value in outputs.items():
            with self.subTest(head=name):
                self.assertFalse(torch.equal(value, absent[name]))
                acoustic_gradient, hand_gradient = torch.autograd.grad(
                    value.sum(), (features, video["structured"]), retain_graph=True,
                )
                self.assertGreater(acoustic_gradient.abs().sum().item(), 0)
                self.assertEqual(torch.count_nonzero(hand_gradient[:, :, :, :98]).item(), 0)
                self.assertEqual(torch.count_nonzero(hand_gradient[:, :, :2]).item(), 0)
                for view in (2, 3):
                    self.assertGreater(hand_gradient[:, :, view, 98:].abs().sum().item(), 0)
        before = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0)
        sum(value.square().mean() for value in outputs.values()).backward()
        optimizer.step()
        for name in ("audio.conv1.weight", "anonymous_branch.structure_encoder.0.weight",
                     "anonymous_branch.recurrent.weight_ih_l0", "fusion.0.weight"):
            self.assertFalse(torch.equal(before[name], model.state_dict()[name]), name)
        for name in model.head_shapes:
            key = f"audio.heads.{name}.weight"
            self.assertFalse(torch.equal(before[key], model.state_dict()[key]), key)

    def test_coarse_context_informs_all_heads_without_exact_geometry_and_masked_values_are_inert(self):
        model = small_model(version=4).eval()
        video = without_evidence(copy.deepcopy(self.video))
        video["structured_available"][:, :, :2, 98:194] = True
        video["segment_id"][:, :, 0] = 0
        video["segment_id"][:, :, 1] = 1
        video["structured"][:, :, :2, 186:188] -= 12
        original = self.forward(video, model=model)
        for view in (0, 1):
            for start, stop in ((186, 188), (188, 190), (190, 192), (192, 193), (193, 194)):
                with self.subTest(view=view, feature=start):
                    changed = copy.deepcopy(video)
                    changed["structured"][:, :, view, start:stop] += 3
                    actual = self.forward(changed, model=model)
                    self.assertTrue(any(not torch.equal(original[name], actual[name]) for name in original))
        without_coarse = copy.deepcopy(video)
        without_coarse["structured_available"][..., 186:] = False
        base = self.forward(without_coarse, model=model)
        without_coarse["structured"][..., 186:] = 1e20
        self.assert_outputs_equal(base, self.forward(without_coarse, model=model))
        for name in original:
            self.assertFalse(torch.equal(original[name], base[name]), name)
        self.assertFalse(torch.equal(base["technique_logits"], self.forward(model=model)["technique_logits"]))
        video["structured_available"][:, :, 2, 186:188] = True
        video["segment_id"][:, :, 2] = 2
        with self.assertRaisesRegex(ValueError, "anonymous"):
            self.forward(video, model=model)

    def test_anonymous_slot_swap_is_exactly_invariant_with_independent_track_boundaries(self):
        video = copy.deepcopy(self.video)
        video["segment_id"][:, 3:, 2] = 12
        video["structured_available"][:, 1, 3] = False
        video["segment_id"][:, 1, 3] = -1
        video["segment_id"][:, 2:, 3] = 23
        original = self.forward(video)
        swapped = copy.deepcopy(video)
        for name in ("structured", "structured_available", "segment_id"):
            swapped[name] = video[name][:, :, [0, 1, 3, 2]]
        self.assert_outputs_equal(original, self.forward(swapped))
        video["structured"].requires_grad_(True)
        hidden = []
        handle = self.model.anonymous_branch.register_forward_hook(
            lambda module, args, result: hidden.append(result[0])
        )
        try:
            self.forward(video)
        finally:
            handle.remove()
        self.assertEqual(len(hidden), 2)
        for slot, output in enumerate(hidden, 2):
            gradient, = torch.autograd.grad(output[:, -1].sum(), video["structured"], retain_graph=True)
            other = 3 if slot == 2 else 2
            self.assertEqual(torch.count_nonzero(gradient[:, :, other]).item(), 0)
            self.assertEqual(torch.count_nonzero(gradient[:, :2, slot]).item(), 0)
            self.assertGreater(gradient[:, -1, slot].abs().sum().item(), 0)

    def test_anonymous_pool_means_available_tracks_and_reports_count_not_slot_identity(self):
        video = without_evidence(copy.deepcopy(self.video))
        video["structured_available"][:, :, 2, 98:186] = True
        video["segment_id"][:, :, 2] = 0
        video["structured"][:, :, 3] = video["structured"][:, :, 2]
        inputs = []
        handle = self.model.fusion.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach()))
        try:
            self.forward(video)
            video["structured_available"][:, :, 3] = video["structured_available"][:, :, 2]
            video["segment_id"][:, :, 3] = 7
            self.forward(video)
        finally:
            handle.remove()
        self.assertTrue(torch.equal(inputs[0][:, :, :-3], inputs[1][:, :, :-3]))
        self.assertTrue((inputs[0][:, :, -1] == 1).all())
        self.assertTrue((inputs[1][:, :, -1] == 2).all())
        self.assertEqual(torch.count_nonzero(inputs[1][:, :, -3:-1]).item(), 0)

    def test_masked_contents_cannot_change_neighbors_or_receive_gradients(self):
        self.video["structured_available"][:, 2, :, :84] = False
        self.video["structured_available"][:, 3] = False
        self.video["segment_id"][:, 3] = -1
        original = self.forward(self.video)
        altered = copy.deepcopy(self.video)
        altered["structured"][~altered["structured_available"]] = 1e20
        self.assert_outputs_equal(original, self.forward(altered))
        altered["structured"].requires_grad_(True)
        outputs = self.forward(altered)
        sum(value.square().mean() for value in outputs.values()).backward()
        self.assertEqual(torch.count_nonzero(altered["structured"].grad[~altered["structured_available"]]).item(), 0)
        self.assertFalse(any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in self.model.modules()))

    def test_structured_only_joint_motion_and_guitar_geometry_receive_training_gradients(self):
        self.video["structured"].requires_grad_(True)
        outputs = self.forward(self.video)
        (outputs["fret_logits"].square().mean() + outputs["technique_logits"].square().mean()).backward()
        for view, branch in enumerate((
            self.model.position_branch, self.model.technique_branch,
            self.model.anonymous_branch, self.model.anonymous_branch,
        )):
            for name, start, stop in (
                ("21 landmark xy pairs", 0, 42),
                ("21 landmark xy velocities", 42, 84),
                ("six guitar-anchor xy pairs", 84, 96),
                ("neck length and board width", 96, 98),
                ("21 independent hand xy pairs", 98, 140),
                ("21 independent hand xy velocities", 140, 182),
                ("wrist velocity", 182, 184),
                ("hand orientation", 184, 186),
            ):
                with self.subTest(view=view, evidence=name):
                    gradient = self.video["structured"].grad[:, :, view, start:stop]
                    pair_gradients = gradient.reshape(*gradient.shape[:2], -1, 2).abs().sum(dim=(0, 1, 3))
                    self.assertTrue((pair_gradients > 0).all())
            for module in (branch.structure_encoder, branch.recurrent):
                self.assertGreater(sum(
                    parameter.grad.abs().sum().item()
                    for parameter in module.parameters() if parameter.grad is not None
                ), 0)
        for name in ("fret_logits", "technique_logits"):
            self.assertGreater(self.model.audio.heads[name].weight.grad.abs().sum().item(), 0)

    def test_shot_track_cuts_and_gaps_reset_both_temporal_directions(self):
        for gap in (False, True):
            with self.subTest(gap=gap):
                video = copy.deepcopy(self.video)
                if gap:
                    video["structured_available"][:, 2] = False
                    video["segment_id"][:, 2] = -1
                else:
                    video["segment_id"][:, 2:] = 17
                original = self.forward(video)
                for edited, unaffected in ((slice(None, 2), slice(3, None)), (slice(3, None), slice(None, 2))):
                    altered = copy.deepcopy(video)
                    altered["structured"][:, edited] *= -3
                    changed = self.forward(altered)
                    for name in original:
                        self.assertTrue(torch.equal(original[name][:, unaffected], changed[name][:, unaffected]), name)

    def test_technique_mismatch_is_a_structure_boundary_not_a_fretting_boundary(self):
        self.video["technique_available"][:, 2] = False
        original = self.forward(self.video)
        without_plucking = copy.deepcopy(self.video)
        without_plucking["structured_available"][:, :, 1:] = False
        without_plucking["segment_id"][:, :, 1:] = -1
        fretting_only = self.forward(without_plucking)
        for name in original:
            self.assertTrue(torch.equal(original[name][:, 2], fretting_only[name][:, 2]), name)
        for edited, unaffected in ((slice(None, 2), slice(3, None)), (slice(3, None), slice(None, 2))):
            altered = copy.deepcopy(self.video)
            altered["structured"][:, edited, 1:] *= -5
            changed = self.forward(altered)
            for name in original:
                self.assertTrue(torch.equal(original[name][:, unaffected], changed[name][:, unaffected]), name)
        altered = copy.deepcopy(self.video)
        altered["structured"][:, 2, 1:] *= -10
        self.assert_outputs_equal(original, self.forward(altered))
        altered["structured"][:, 2, 0] *= -10
        changed = self.forward(altered)
        self.assertFalse(torch.equal(original["fret_logits"][:, 3:], changed["fret_logits"][:, 3:]))

    def test_recomputed_velocities_cannot_carry_mismatched_positions_into_gated_tracks(self):
        self.video["technique_available"][:, 2] = False
        for start, stop, source in ((42, 84, 0), (140, 182, 98), (182, 184, 98), (188, 190, 186)):
            self.video["structured"][:, 1:, :, start:stop] = (
                self.video["structured"][:, 1:, :, source:source + stop - start]
                - self.video["structured"][:, :-1, :, source:source + stop - start]
            ) / .04
        original = self.forward(self.video)
        altered = copy.deepcopy(self.video)
        altered["structured"][:, 2, 1:, :42] += 10
        altered["structured"][:, 2, 1:, 98:140] += 10
        altered["structured"][:, 2, 1, 186:188] += 10
        for start, stop, source in ((42, 84, 0), (140, 182, 98), (182, 184, 98), (188, 190, 186)):
            altered["structured"][:, 1:, 1:, start:stop] = (
                altered["structured"][:, 1:, 1:, source:source + stop - start]
                - altered["structured"][:, :-1, 1:, source:source + stop - start]
            ) / .04
        changed = self.forward(altered)
        for name in original:
            self.assertTrue(torch.equal(original[name], changed[name]), name)
        altered["structured"].requires_grad_(True)
        inputs = []
        handles = [branch.structure_encoder.register_forward_pre_hook(
            lambda module, args: inputs.append(args[0].detach().clone())
        ) for branch in (self.model.technique_branch, self.model.anonymous_branch)]
        try:
            outputs = self.forward(altered)
        finally:
            for handle in handles:
                handle.remove()
        sum(value.square().mean() for value in outputs.values()).backward()
        self.assertEqual(len(inputs), 3)
        for start, stop in VELOCITY_SLICES:
            for frame in (0, 3):
                for encoded in inputs:
                    self.assertEqual(torch.count_nonzero(encoded[:, frame, start:stop]).item(), 0)
                    self.assertEqual(torch.count_nonzero(encoded[:, frame, STRUCTURED_DIM + start:STRUCTURED_DIM + stop]).item(), 0)
                self.assertEqual(torch.count_nonzero(altered["structured"].grad[:, frame, 1:, start:stop]).item(), 0)
            for view in ((1,) if start == 188 else (1, 2, 3)):
                self.assertGreater(torch.count_nonzero(altered["structured"].grad[:, 4, view, start:stop]).item(), 0)
            self.assertGreater(torch.count_nonzero(altered["structured"].grad[:, 3, 0, start:stop]).item(), 0)
        fretting_changed = copy.deepcopy(self.video)
        fretting_changed["structured"][:, 3, 0, 42:84] += 10
        self.assertFalse(torch.equal(original["fret_logits"], self.forward(fretting_changed)["fret_logits"]))

    def test_technique_velocity_reset_covers_cuts_gaps_and_structure_only_segments(self):
        for boundary in ("cut", "gap"):
            with self.subTest(boundary=boundary):
                video = copy.deepcopy(self.video)
                if boundary == "cut":
                    video["segment_id"][:, 3:, 1:] = 1
                else:
                    video["structured_available"][:, 2, 1:] = False
                    video["segment_id"][:, 2, 1:] = -1
                original = self.forward(video)
                altered = copy.deepcopy(video)
                for start, stop in VELOCITY_SLICES:
                    altered["structured"][:, [0, 3], 1:, start:stop] += 100
                self.assert_outputs_equal(original, self.forward(altered))
                altered["structured"][:, 4, 1:, 140:184] += 10
                self.assertFalse(torch.equal(original["technique_logits"], self.forward(altered)["technique_logits"]))

    def test_velocity_only_evidence_cannot_bootstrap_a_technique_segment(self):
        video = without_evidence(copy.deepcopy(self.video))
        for start, stop in VELOCITY_SLICES:
            video["structured_available"][:, :, 1:2 if start == 188 else 4, start:stop] = True
        video["segment_id"][:, :, 1:] = 0
        base = self.forward()
        self.assert_outputs_equal(base, self.forward(video))
        video["structured_available"][:, 2, 1:, 98:140] = True
        outputs = self.forward(video)
        for name in outputs:
            self.assertTrue(torch.equal(outputs[name][:, :2], base[name][:, :2]), name)
        self.assertFalse(torch.equal(outputs["technique_logits"][:, 3:], base["technique_logits"][:, 3:]))
        video["frame_indices"].fill_(1)
        self.assert_outputs_equal(base, self.forward(video))

    def test_note_loss_learns_from_plucking_except_local_correspondence_exclusions(self):
        self.video["structured"].requires_grad_(True)
        self.video["technique_available"][:, 2] = False
        outputs = self.forward(self.video)
        outputs["note_onset_logits"].sum().backward()
        self.assertGreater(torch.count_nonzero(self.video["structured"].grad[:, :, 0]).item(), 0)
        self.assertGreater(torch.count_nonzero(self.video["structured"].grad[:, :, 1]).item(), 0)
        self.assertGreater(torch.count_nonzero(self.video["structured"].grad[:, :, 2:]).item(), 0)
        self.assertEqual(torch.count_nonzero(self.video["structured"].grad[:, 2, 1:]).item(), 0)
        self.assertTrue(all(parameter.grad is not None for parameter in self.model.technique_branch.parameters()))
        self.assertTrue(all(parameter.grad is not None for parameter in self.model.anonymous_branch.parameters()))

    def test_padding_audio_and_video_is_inert(self):
        lengths = torch.tensor([3])
        self.video["frame_indices"][:, 3:] = -1
        self.video["structured_available"][:, 3:] = False
        self.video["segment_id"][:, 3:] = -1
        original = self.forward(self.video, lengths=lengths)
        altered = copy.deepcopy(self.video)
        altered["structured"][:, 3:] = 1e20
        self.assert_outputs_equal(original, self.forward(altered, lengths=lengths))
        for value in original.values():
            self.assertEqual(torch.count_nonzero(value[:, 3:]).item(), 0)

    def test_variable_audio_lengths_and_native_video_counts_preserve_masking_and_gradients(self):
        model = AudioVideoTranscriber(
            FingerstyleTranscriber(ModelConfig(
                architecture_version=4, n_mels=8, hidden_size=7, recurrent_layers=1, dropout=0,
            )), VideoConfig(hidden_size=5, temporal_layers=2),
        ).eval()
        video = visual_inputs(batch=2, frames=7, count=3)
        video["frame_indices"][0] = torch.tensor([0, -1, 2, 2, 2, 2, 2])
        features = torch.randn(2, 7, 8, requires_grad=True)
        conditioning = torch.randn(2, 7, 12, requires_grad=True)
        lengths = torch.tensor([3, 6])
        inputs = []
        handle = model.fusion.register_forward_pre_hook(lambda module, args: inputs.append(args[0].clone()))
        try:
            original = model(features, conditioning, lengths, video=video)
        finally:
            handle.remove()
        valid = torch.arange(7)[None, :] < lengths[:, None]
        changed_features = features.masked_fill(~valid[:, :, None], 1000)
        changed_conditioning = conditioning.masked_fill(~valid[:, :, None], -1000)
        self.assert_outputs_equal(
            original, model(changed_features, changed_conditioning, lengths, video=video),
        )
        visual = inputs[0][:, :, 2 * model.config.hidden_size:]
        self.assertEqual(torch.count_nonzero(visual[~valid]).item(), 0)
        self.assertEqual(torch.count_nonzero(visual[0, 1]).item(), 0)
        self.assertTrue(torch.equal(visual[1, 2], visual[1, 5]))
        for value in original.values():
            self.assertEqual(torch.count_nonzero(value[~valid]).item(), 0)
        sum(value.square().mean() for value in original.values()).backward()
        self.assertEqual(torch.count_nonzero(features.grad[~valid]).item(), 0)
        self.assertEqual(torch.count_nonzero(conditioning.grad[~valid]).item(), 0)
        self.assertGreater(features.grad[valid].abs().sum().item(), 0)
        self.assertGreater(conditioning.grad[valid].abs().sum().item(), 0)

    def test_joint_loss_updates_acoustic_backbone_numeric_branches_and_fusion_immediately(self):
        self.model = small_model(modality_dropout=0)
        self.model.train()
        self.assertTrue(self.model.training)
        self.assertTrue(self.model.position_branch.training)
        self.assertTrue(all(module.training for module in self.model.audio.modules()))
        self.assertTrue(all(parameter.requires_grad for parameter in self.model.parameters()))
        initial = copy.deepcopy(self.model.state_dict())
        batch = supervised_batch()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.01, weight_decay=0)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            outputs = runtime._outputs(self.model, batch)
            loss, _ = masked_loss(outputs, batch["targets"], batch["masks"], batch["valid_frames"])
            loss.backward()
            for module in (
                self.model.audio.conv1, self.model.audio.conv2, self.model.audio.projection,
                self.model.audio.recurrent, self.model.position_branch.structure_encoder,
                self.model.position_branch.recurrent, self.model.technique_branch.structure_encoder,
                self.model.technique_branch.recurrent, self.model.anonymous_branch.structure_encoder,
                self.model.anonymous_branch.recurrent, self.model.fusion,
            ):
                self.assertGreater(sum(
                    parameter.grad.abs().sum().item() for parameter in module.parameters()
                    if parameter.grad is not None
                ), 0)
            optimizer.step()
        for name in (
            "audio.conv1.weight", "audio.conv2.weight", "audio.projection.0.weight",
            "audio.recurrent.weight_ih_l0", "position_branch.structure_encoder.0.weight",
            "position_branch.recurrent.weight_ih_l0", "technique_branch.structure_encoder.0.weight",
            "technique_branch.recurrent.weight_ih_l0", "anonymous_branch.structure_encoder.0.weight",
            "anonymous_branch.recurrent.weight_ih_l0", "fusion.0.weight",
        ):
            self.assertFalse(torch.equal(initial[name], self.model.state_dict()[name]), name)
        self.model.eval().train()
        self.assertTrue(self.model.audio.training)

    def test_nonlinear_fusion_has_audio_video_interaction_not_additive_outputs(self):
        base = self.forward()
        original = self.forward(self.video)
        changed_conditioning = self.conditioning.clone()
        changed_conditioning[:, :, 0] = 1
        other_base = self.model(self.features, changed_conditioning)
        other = self.model(self.features, changed_conditioning, video=self.video)
        self.assertFalse(torch.equal(
            original["fret_logits"] - base["fret_logits"],
            other["fret_logits"] - other_base["fret_logits"],
        ))
        self.assertGreater((
            original["fret_logits"] - base["fret_logits"] -
            other["fret_logits"] + other_base["fret_logits"]
        ).abs().max().item(), 1e-5)
        original["fret_logits"].sum().backward()
        self.assertGreater(self.model.fusion[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.audio.recurrent.weight_ih_l0.grad.abs().sum().item(), 0)

    def test_masked_rasgueado_has_no_hidden_sparsity_penalty_or_gradient(self):
        from scripts.technique_supervision import TECHNIQUE_TYPES

        batch = supervised_batch()
        axis = TECHNIQUE_TYPES.index("rasgueado")
        batch["masks"]["technique"][:, :, axis] = False
        batch["masks"]["technique"][:, 2, axis] = True
        batch["targets"]["technique"][:, 2, axis] = 1
        outputs = runtime._outputs(self.model, batch)
        loss, stats = masked_loss(
            outputs, batch["targets"], batch["masks"], batch["valid_frames"], sparsity_weight=.9,
        )
        unknown = torch.zeros_like(outputs["technique_logits"], dtype=torch.bool)
        unknown[:, :, axis] = ~batch["masks"]["technique"][:, :, axis]
        changed = {**outputs, "technique_logits": outputs["technique_logits"] + unknown.float() * 100}
        changed_loss, changed_stats = masked_loss(
            changed, batch["targets"], batch["masks"], batch["valid_frames"], sparsity_weight=.9,
        )
        self.assertTrue(torch.equal(loss, changed_loss))
        self.assertEqual(stats, changed_stats)
        gradient, = torch.autograd.grad(loss, outputs["technique_logits"])
        self.assertEqual(torch.count_nonzero(gradient[unknown]).item(), 0)
        self.assertNotEqual(gradient[0, 2, axis].item(), 0)

    def test_whole_video_dropout_keeps_audio_learning_without_mutating_inputs(self):
        model = small_model(modality_dropout=1, audio_dropout=0).train()
        before = copy.deepcopy(self.video)
        inputs = []
        handle = model.fusion.register_forward_pre_hook(lambda module, args: inputs.append(args[0].clone()))
        try:
            dropped = self.forward(self.video, model=model)
            absent = self.forward(model=model)
        finally:
            handle.remove()
        self.assert_outputs_equal(dropped, absent)
        self.assertEqual(torch.count_nonzero(inputs[0][:, :, 2 * model.config.hidden_size:]).item(), 0)
        self.assert_outputs_equal(before, self.video)
        batch = supervised_batch()
        audio_before = model.audio.conv1.weight.clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0)
        loss, _ = masked_loss(dropped, batch["targets"], batch["masks"], batch["valid_frames"])
        loss.backward()
        self.assertGreater(model.audio.conv1.weight.grad.abs().sum().item(), 0)
        self.assertEqual(model.fusion[0].weight.grad.abs().sum().item(), 0)
        for branch in (model.position_branch, model.technique_branch, model.anonymous_branch):
            self.assertTrue(all(
                parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0
                for parameter in branch.parameters()
            ))
        optimizer.step()
        self.assertFalse(torch.equal(audio_before, model.audio.conv1.weight))
        self.assert_outputs_equal(before, self.video)
        model.eval()
        with mock.patch("torch.rand", side_effect=AssertionError("Evaluation must not drop modalities")):
            present = self.forward(self.video, model=model)
            missing = self.forward(model=model)
        self.assertFalse(torch.equal(present["fret_logits"], missing["fret_logits"]))
        self.assert_outputs_equal(missing, self.forward(without_evidence(copy.deepcopy(self.video)), model=model))

    def test_dropout_is_per_example_for_all_four_views_and_uses_restorable_torch_rng(self):
        model = small_model(modality_dropout=.5, audio_dropout=0).train()
        batch = 8
        features = self.features.expand(batch, -1, -1).clone()
        conditioning = self.conditioning.expand(batch, -1, -1).clone()
        video = visual_inputs(batch=batch)
        video["structured_available"][-1] = False
        video["segment_id"][-1] = -1
        original = copy.deepcopy(video)
        torch.manual_seed(41)
        rng = torch.get_rng_state()
        dropped = torch.rand(batch) < .5
        torch.set_rng_state(rng)
        inputs = []
        handle = model.fusion.register_forward_pre_hook(lambda module, args: inputs.append(args[0].clone()))
        try:
            outputs = model(features, conditioning, video=video)
            torch.set_rng_state(rng)
            replay = model(features, conditioning, video=video)
        finally:
            handle.remove()
        self.assert_outputs_equal(outputs, replay)
        self.assert_outputs_equal(original, video)
        presence = inputs[0][:, :, -3:]
        expected = (~dropped)[:, None, None].expand(batch, 5, 3).float().clone()
        expected[:, :, 2] *= 2
        expected[-1] = 0
        self.assertTrue(torch.equal(presence, expected))
        self.assertTrue(dropped[:-1].any())
        self.assertTrue((~dropped[:-1]).any())
        visual = inputs[0][:, :, 2 * model.config.hidden_size:-3]
        self.assertEqual(torch.count_nonzero(visual[dropped]).item(), 0)

    def test_no_video_still_trains_acoustic_backbone_and_missing_modality_fusion(self):
        model = small_model(audio_dropout=0).train()
        for video in (None, without_evidence(copy.deepcopy(self.video))):
            model.zero_grad(set_to_none=True)
            with mock.patch("torch.rand", side_effect=AssertionError("No evidence to drop")):
                outputs = self.forward(video, model=model)
            outputs["note_onset_logits"].square().mean().backward()
            self.assertGreater(model.audio.conv1.weight.grad.abs().sum().item(), 0)
            self.assertEqual(model.fusion[0].weight.grad.abs().sum().item(), 0)

    def test_video_dtype_shape_finite_index_and_mask_validation_is_strict(self):
        malformed = []
        video = copy.deepcopy(self.video)
        video["structured"] = video["structured"].double()
        malformed.append(video)
        for value in (float("nan"), float("inf")):
            video = without_evidence(copy.deepcopy(self.video))
            video["structured"].flatten()[0] = value
            malformed.append(video)
        for name in ("technique_available", "structured_available"):
            video = copy.deepcopy(self.video)
            video[name] = video[name].float()
            malformed.append(video)
        for name in ("segment_id", "frame_indices"):
            video = copy.deepcopy(self.video)
            video[name] = video[name].int()
            malformed.append(video)
        for name, value in (
            ("frame_indices", -2), ("frame_indices", 5), ("segment_id", -2), ("segment_id", -1),
        ):
            video = copy.deepcopy(self.video)
            video[name].flatten()[0] = value
            malformed.append(video)
        video = without_evidence(copy.deepcopy(self.video))
        video["segment_id"][:, 0] = 0
        malformed.append(video)
        video = copy.deepcopy(self.video)
        video["structured"] = video["structured"][:, :, :1]
        malformed.append(video)
        video = copy.deepcopy(self.video)
        video["structured"] = video["structured"][..., :97]
        malformed.append(video)
        video = copy.deepcopy(self.video)
        video["structured"] = video["structured"][:, :0]
        malformed.append(video)
        video = copy.deepcopy(self.video)
        video["target_fret"] = torch.zeros(1)
        malformed.append(video)
        video = copy.deepcopy(self.video)
        del video["structured_available"]
        malformed.append(video)
        for index, video in enumerate(malformed):
            with self.subTest(index=index), self.assertRaises((ValueError, TypeError)):
                self.forward(video)

    def test_batched_examples_do_not_normalize_or_recur_into_each_other(self):
        video = visual_inputs(batch=2)
        features = self.features.expand(2, -1, -1).clone()
        conditioning = self.conditioning.expand(2, -1, -1).clone()
        original = self.model(features, conditioning, video=video)
        video["structured"][1] *= -10
        changed = self.model(features, conditioning, video=video)
        self.assert_outputs_equal(
            {name: value[0] for name, value in original.items()},
            {name: value[0] for name, value in changed.items()},
        )

    def test_runtime_evaluation_carries_real_nested_inputs_and_preserves_parameters(self):
        batch = supervised_batch()
        moved = runtime._batch_to_device(batch, torch.device("cpu"))
        self.assertEqual(set(moved["video"]), set(batch["video"]))
        outputs = runtime._outputs(self.model, moved)
        expected, stats = masked_loss(outputs, moved["targets"], moved["masks"], moved["valid_frames"])
        state = copy.deepcopy(self.model.state_dict())
        metrics = runtime.evaluate_model(self.model, [batch], "cpu")
        self.assertAlmostEqual(metrics["loss"], expected.item(), places=5)
        self.assertEqual(metrics["loss_statistics"], stats)
        self.assert_outputs_equal(state, self.model.state_dict())
        self.assertFalse(self.model.audio.training)
        bad = copy.deepcopy(batch)
        bad["video"] = None
        with self.assertRaisesRegex(ValueError, "video must be a mapping"):
            runtime._batch_to_device(bad, torch.device("cpu"))

    def test_event_evaluation_requests_absolute_times_and_uses_structure_only_model(self):
        dataset = WindowFixture(True)
        model = small_model(version=1, n_mels=1)
        visits = []

        def video_window(record, audio_times):
            self.assertIs(record, dataset.records[0])
            visits.append(audio_times.copy())
            video = visual_inputs(frames=len(audio_times), count=3)
            return {name: value[0] for name, value in video.items()}

        dataset.video_window = video_window
        with mock.patch.object(model, "forward", wraps=model.forward) as forward:
            report = evaluate_events(model, dataset, "cpu")
        self.assertEqual(report["windowVisits"], 2)
        self.assertEqual([float(times[0]) for times in visits], [0., 2.])
        self.assertTrue(np.allclose(visits[1], visits[0] + 2))
        self.assertEqual(forward.call_count, 2)
        self.assertTrue(all("video" in call.kwargs for call in forward.call_args_list))
        self.assertTrue(model.training)
        self.assertTrue(model.audio.training)
        dataset.video_window = lambda record, audio_times: None
        with mock.patch.object(model.audio, "forward", wraps=model.audio.forward) as forward:
            evaluate_events(model.audio, dataset, "cpu")
        self.assertTrue(all("video" not in call.kwargs for call in forward.call_args_list))

    def test_paired_rasgueado_event_absence_is_unknown_but_native_attributes_remain_scorable(self):
        from scripts.transcriber_data import PAIRED_TECHNIQUE_TARGET_POLICY

        for paired, native_positive in ((False, False), (True, False), (True, True)):
            with self.subTest(paired=paired, native_positive=native_positive):
                dataset = WindowFixture(True)
                dataset.records[0]["techniqueAnnotationsComplete"] = True
                if paired:
                    dataset.video_identity = {"targetPolicy": PAIRED_TECHNIQUE_TARGET_POLICY}
                if native_positive:
                    dataset.records[0]["techniques"] = [{
                        "proposedOnsetClipSeconds": 3., "techniques": ["rasgueado"],
                        "directions": {"rasgueado": "Down"}, "directionMasks": {"rasgueado": True},
                        "stringsByTechnique": {"rasgueado": [6]},
                    }]
                decoded = {
                    "notes": [], "percussion": [], "policy": {},
                    "techniques": [
                        {"onsetSeconds": 3., "technique": name, "direction": "Up", "strings": [5]}
                        for name in ("rasgueado", "brush")
                    ],
                }
                with mock.patch("scripts.transcriber_model.decode_events", return_value=decoded):
                    report = evaluate_events(small_model(version=1, n_mels=1), dataset, "cpu")
                metrics = report["metricsByToleranceSeconds"]["0.1"]
                self.assertEqual(metrics["technique_brush"]["false_positive"], 1)
                rasgueado = metrics["technique_rasgueado"]
                self.assertEqual(rasgueado["true_positive"], int(native_positive))
                self.assertEqual(rasgueado["false_positive"], int(not paired))
                self.assertEqual(rasgueado["unscorable_predictions"], int(paired and not native_positive))
                if paired:
                    self.assertIsNone(rasgueado["precision"])
                    self.assertIsNone(rasgueado["f1"])
                    self.assertEqual(report["pairedTechniqueTargetPolicy"], PAIRED_TECHNIQUE_TARGET_POLICY)
                else:
                    self.assertNotIn("pairedTechniqueTargetPolicy", report)
                if native_positive:
                    self.assertEqual(metrics["technique_direction"]["false_positive"], 2)
                    self.assertEqual(metrics["technique_string_set"]["false_positive"], 2)


class VideoRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        rng = runtime._capture_rng()
        self.addCleanup(runtime._restore_rng, rng)
        directory = tempfile.TemporaryDirectory(prefix=".synthetic-video-tests-", dir=ROOT)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def train(self, name, steps, *, model=None, dataset=None, resume=None):
        model = small_model() if model is None else model
        dataset = PairedSyntheticDataset() if dataset is None else dataset
        train, validation = paired_loaders(dataset)
        summary = runtime.run_training(
            model, train, validation,
            runtime.TrainingConfig(epochs=2, max_steps=steps, device="cpu", weight_decay=.5),
            self.root / name, {
                "model": asdict(model.config), "video": {"config": asdict(model.video_config)},
                "dataset": "synthetic-only",
            }, resume=resume,
        )
        return summary, model

    def assert_tree_equal(self, left, right):
        self.assertEqual(type(left), type(right))
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for name in left:
                self.assert_tree_equal(left[name], right[name])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_all_missing_paired_batch_updates_audio_and_can_checkpoint_and_resume(self):
        model = small_model()
        initial = copy.deepcopy(model.state_dict())
        dataset = PairedSyntheticDataset((False, False))
        summary, model = self.train("missing", 1, model=model, dataset=dataset)
        resumed, model = self.train("missing", 2, model=model, dataset=dataset, resume=summary["latest_checkpoint"])
        self.assertFalse(torch.equal(initial["audio.conv1.weight"], model.audio.conv1.weight))
        self.assertEqual(summary["optimizer_updates"], 1)
        self.assertEqual(summary["optimizer_skipped_batches"], 0)
        self.assertEqual(resumed["optimizer_updates"], 1)
        self.assertEqual(resumed["optimizer_skipped_batches"], 0)
        checkpoint = runtime.load_checkpoint(resumed["latest_checkpoint"])
        self.assertTrue(checkpoint["optimizer_state"]["state"])
        self.assertEqual(checkpoint["global_step"], 2)
        self.assertEqual(checkpoint["history"][-1]["optimizer_updates"], 2)
        self.assertEqual(checkpoint["history"][-1]["optimizer_skipped_batches"], 0)
        corrupted = copy.deepcopy(checkpoint)
        corrupted["history"][-1]["optimizer_updates"] = 0
        corrupted["history"][-1]["optimizer_skipped_batches"] = 2
        with self.assertRaisesRegex(ValueError, "Optimizer"):
            runtime._validate_checkpoint(corrupted)

    def test_joint_resume_through_missing_evidence_and_dropout_is_bitwise_identical(self):
        initial_audio = copy.deepcopy(small_model().audio.state_dict())
        full, full_model = self.train("full", 4)
        partial, _ = self.train("partial", 1)
        resumed, restored = self.train(
            "partial", 4, model=small_model(seed=999), resume=partial["latest_checkpoint"],
        )
        self.assertEqual((full["optimizer_updates"], full["optimizer_skipped_batches"]), (4, 0))
        self.assertEqual((resumed["optimizer_updates"], resumed["optimizer_skipped_batches"]), (3, 0))
        final = runtime.load_checkpoint(full["latest_checkpoint"])
        loaded = runtime.load_checkpoint(resumed["latest_checkpoint"])
        for name in ("model_state", "optimizer_state", "rng", "loader_state", "cursor"):
            self.assert_tree_equal(final[name], loaded[name])
        self.assert_tree_equal(full_model.state_dict(), restored.state_dict())
        self.assertFalse(torch.equal(initial_audio["conv1.weight"], restored.audio.conv1.weight))
        self.assertEqual(len(loaded["optimizer_state"]["param_groups"][0]["params"]), sum(
            parameter.requires_grad for parameter in restored.parameters()
        ))
        clone = small_model(seed=71)
        clone.load_state_dict(loaded["model_state"], strict=True)
        batch = supervised_batch()
        clone.eval()
        restored.eval()
        self.assert_tree_equal(runtime._outputs(clone, batch), runtime._outputs(restored, batch))

    def test_voice_only_objective_updates_audio_and_both_video_branches(self):
        class VoiceOnlyDataset(PairedSyntheticDataset):
            def __getitem__(self, index):
                batch = super().__getitem__(index)
                for mask in batch["masks"].values():
                    mask.zero_()
                batch["targets"]["voice"][1, 0] = 1
                batch["masks"]["voice"][1, 0] = True
                return batch

        model = small_model(modality_dropout=0)
        state = copy.deepcopy(model.state_dict())
        summary, model = self.train("voice-only", 1, model=model, dataset=VoiceOnlyDataset((True,)))
        self.assertEqual(summary["optimizer_updates"], 1)
        self.assertEqual(summary["optimizer_skipped_batches"], 0)
        for name in ("audio.conv1.weight", "position_branch.structure_encoder.0.weight",
                     "technique_branch.structure_encoder.0.weight", "anonymous_branch.structure_encoder.0.weight",
                     "audio.heads.voice_logits.weight"):
            self.assertFalse(torch.equal(state[name], model.state_dict()[name]), name)

    def test_missing_batch_after_an_update_keeps_acoustic_optimizer_learning(self):
        dataset = PairedSyntheticDataset((True, False))
        first, model = self.train("momentum", 1, dataset=dataset)
        before = runtime.load_checkpoint(first["latest_checkpoint"])
        after, model = self.train("momentum", 2, model=model, dataset=dataset, resume=first["latest_checkpoint"])
        checkpoint = runtime.load_checkpoint(after["latest_checkpoint"])
        self.assertFalse(torch.equal(before["model_state"]["audio.conv1.weight"], model.audio.conv1.weight))
        before_step = before["optimizer_state"]["state"][0]["step"]
        self.assertEqual(checkpoint["optimizer_state"]["state"][0]["step"], before_step + 1)
        self.assertEqual(after["optimizer_updates"], 1)
        self.assertEqual(after["optimizer_skipped_batches"], 0)

    def test_rgb_frozen_or_unversioned_paired_checkpoints_are_explicitly_unsupported(self):
        summary, _ = self.train("schema", 1)
        original = runtime.load_checkpoint(summary["latest_checkpoint"])
        for change in ("rgb", "unversioned", "old-version", "schema-three", "frozen", "unversioned-architecture", "old-architecture"):
            with self.subTest(change=change):
                checkpoint = copy.deepcopy(original)
                config = checkpoint["identity"]["video"]["config"]
                if change == "rgb":
                    config["image_size"] = 96
                elif change == "unversioned":
                    del config["input_schema_version"]
                elif change == "old-version":
                    config["input_schema_version"] = 2
                elif change == "schema-three":
                    config.update(input_schema_version=3, architecture_version=3, structured_dim=186)
                elif change == "frozen":
                    config["freeze_audio"] = True
                elif change == "unversioned-architecture":
                    del config["architecture_version"]
                else:
                    config["architecture_version"] = 2
                path = self.root / f"{change}.pt"
                torch.save(checkpoint, path)
                with self.assertRaisesRegex(ValueError, "paired checkpoints are unsupported"):
                    runtime.load_checkpoint(path)

    def test_unlabelled_visual_training_batch_is_skipped_without_inventing_labels(self):
        class PartlyUnlabelledDataset(PairedSyntheticDataset):
            def __getitem__(self, index):
                batch = super().__getitem__(index)
                if index == 0:
                    for mask in batch["masks"].values():
                        mask.zero_()
                return batch

        model = small_model()
        initial = copy.deepcopy(model.state_dict())
        dataset = PartlyUnlabelledDataset((True, True))
        with mock.patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("Unknown labels must not update")):
            summary, model = self.train("unlabelled", 1, model=model, dataset=dataset)
        self.assert_tree_equal(initial, model.state_dict())
        self.assertEqual((summary["optimizer_updates"], summary["optimizer_skipped_batches"]), (0, 1))
        resumed, _ = self.train("unlabelled", 2, model=model, dataset=dataset, resume=summary["latest_checkpoint"])
        self.assertEqual((resumed["optimizer_updates"], resumed["optimizer_skipped_batches"]), (1, 0))

    def test_training_dataset_bundle_collation_and_runtime_use_the_same_visual_contract(self):
        from scripts.paired_video import build_index
        from scripts.transcriber_audio import FeatureConfig
        from scripts.transcriber_data import TrainingDataset, collate_windows
        from tests.test_dataset_release import synthetic_release
        from tests.test_paired_video import bundle_fixture

        manifest = synthetic_release(self.root / "release")
        bundle, _, _ = bundle_fixture(self.root, self.root / "release" / "audio" / "piece-0.flac")
        index, _ = build_index(manifest, [bundle], self.root / "index.json", root=self.root)
        feature_config = FeatureConfig(sample_rate=8000, n_fft=512, hop_length=160, n_mels=16, f_max=3000)
        model_config = ModelConfig(architecture_version=2, n_mels=16, hidden_size=8, recurrent_layers=1)
        datasets = [TrainingDataset(
            manifest, split, feature_config, model_config, root=self.root, video_index_path=index,
        ) for split in ("train", "validation")]
        batch = collate_windows([dataset[0] for dataset in datasets])
        model = AudioVideoTranscriber(
            FingerstyleTranscriber(model_config), VideoConfig(hidden_size=8),
        ).eval()
        moved = runtime._batch_to_device(batch, torch.device("cpu"))
        outputs = runtime._outputs(model, moved)
        base = model(moved["features"], moved["conditioning"], moved["lengths"])
        for name in base:
            self.assertTrue(torch.equal(base[name][1], outputs[name][1]), name)
        mapped = moved["video"]["frame_indices"][0] >= 0
        self.assertTrue(mapped.any())
        self.assertFalse(torch.equal(base["fret_logits"][0, mapped], outputs["fret_logits"][0, mapped]))
        metrics = runtime.evaluate_model(model, [batch], "cpu")
        self.assertTrue(metrics["available"])
        self.assertEqual(metrics["windows"], 2)
        report = evaluate_events(model, datasets[0], "cpu")
        self.assertEqual(report["windowVisits"], len(datasets[0]))


if __name__ == "__main__":
    unittest.main()
