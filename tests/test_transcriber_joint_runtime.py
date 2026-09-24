"""Joint optimizer/checkpoint regressions using synthetic numeric pairs only."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from torch.utils.data import DataLoader

from scripts import transcriber_runtime as runtime
from scripts.transcriber_data import collate_windows
from scripts.transcriber_model import FingerstyleTranscriber, ModelConfig
from scripts.transcriber_video import AudioVideoTranscriber, VideoConfig
from scripts.video_features import STRUCTURED_DIM, VELOCITY_SLICES
from tests.test_transcriber_model import synthetic_targets


ROOT = Path(__file__).resolve().parents[1]


class JointRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(prefix=".joint-runtime-", dir=ROOT)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        rng, threads = runtime._capture_rng(), torch.get_num_threads()
        self.addCleanup(runtime._restore_rng, rng)
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)

    def model(self, dropout=.4, seed=71):
        torch.manual_seed(seed)
        return AudioVideoTranscriber(
            FingerstyleTranscriber(ModelConfig(architecture_version=2, n_mels=4, hidden_size=4, recurrent_layers=1, dropout=.3)),
            VideoConfig(hidden_size=4, modality_dropout=dropout),
        )

    def items(self):
        items = []
        for index in range(4):
            targets, masks, _ = synthetic_targets(1, 5, architecture_version=2)
            for name in ("note_onset", "percussion", "technique"):
                targets[name].zero_()
                targets[name][:, 1, 0] = 1
                masks[name].fill_(True)
            items.append({
                "features": torch.arange(20).float().reshape(5, 4) / 20 + index / 10,
                "conditioning": torch.zeros(5, 12),
                "targets": {name: value[0] for name, value in targets.items()},
                "masks": {name: value[0] for name, value in masks.items()},
                "metadata": {},
                "video": {
                    "structured": torch.full((5, 4, STRUCTURED_DIM), .1 + index / 100),
                    "structured_available": torch.ones(5, 4, STRUCTURED_DIM, dtype=torch.bool),
                    "technique_available": torch.ones(5, dtype=torch.bool),
                    "segment_id": torch.arange(4).expand(5, -1).clone(),
                    "frame_indices": torch.arange(5),
                },
            })
            video = items[-1]["video"]
            video["structured_available"][:, 2:, 186:] = False
            for start, stop in VELOCITY_SLICES:
                video["structured_available"][0, :, start:stop] = False
            video["structured"][:, :, 84:88] = torch.tensor([1., 0., 0., 0.])
            video["structured"][:, :, 98:100] = 0
            video["structured"][:, :, 184:186] = torch.tensor([1., 0.])
            video["structured"][:, :, 190:192] = torch.tensor([1., 0.])
            video["structured"][:, :, 193] = 1
            video["structured"].masked_fill_(~video["structured_available"], 0)
        return items

    def train(self, name, steps, *, dropout=.4, seed=71, resume=None):
        model = self.model(dropout, seed)
        loaders = [DataLoader(
            self.items(), batch_size=1, collate_fn=collate_windows,
            generator=torch.Generator().manual_seed(seed), num_workers=0,
        ) for seed in (101, 202)]
        result = runtime.run_training(
            model, *loaders, runtime.TrainingConfig(epochs=2, max_steps=steps, device="cpu"),
            self.root / name, {"model": asdict(model.config), "video": {"config": asdict(model.video_config)}},
            resume=resume,
        )
        return result, model, runtime.load_checkpoint(result["latest_checkpoint"])

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_modality_dropout_exact_resume_restores_all_states_and_both_branches_learn(self):
        initial = deepcopy(self.model().state_dict())
        _, model, full = self.train("full", 6)
        partial, _, _ = self.train("resumed", 2)
        _, _, resumed = self.train("resumed", 6, seed=991, resume=partial["latest_checkpoint"])
        for name in ("model_state", "optimizer_state", "rng", "loader_state", "cursor"):
            self.assert_tree_equal(full[name], resumed[name])
        state = model.state_dict()
        for prefix in ("audio.", "position_branch.", "technique_branch.", "anonymous_branch.", "fusion."):
            self.assertTrue(any(not torch.equal(initial[name], value) for name, value in state.items() if name.startswith(prefix)), prefix)
        self.assertEqual(full["history"][-1]["optimizer_updates"], 6)
        self.assertEqual(full["history"][-1]["optimizer_skipped_batches"], 0)
        self.assertEqual(len(full["optimizer_state"]["param_groups"][0]["params"]), len(list(model.parameters())))

    def test_all_modality_dropped_batches_update_audio_without_visual_correction(self):
        initial = deepcopy(self.model(dropout=1.).state_dict())
        result, model, _ = self.train("dropped", 2, dropout=1.)
        self.assertEqual((result["optimizer_updates"], result["optimizer_skipped_batches"]), (2, 0))
        state = model.state_dict()
        self.assertTrue(any(not torch.equal(initial[name], value) for name, value in state.items() if name.startswith("audio.")))
        self.assertTrue(all(torch.equal(initial[name], value) for name, value in state.items() if name.startswith(("position_branch.", "technique_branch.", "anonymous_branch.", "fusion."))))

    def test_joint_checkpoint_rejects_frozen_semantics_and_incomplete_optimizer(self):
        _, _, checkpoint = self.train("schema", 1)
        for kind in ("missing-version", "old-version", "old-schema", "old-schema3", "old-dimension", "frozen-field", "staged-initialization", "missing-parameter", "wrong-moments", "missing-counts", "anonymous-shape"):
            payload = deepcopy(checkpoint)
            config = payload["identity"]["video"]["config"]
            if kind == "missing-version":
                del config["architecture_version"]
            elif kind == "old-version":
                config["architecture_version"] = 4
            elif kind == "old-schema":
                config["input_schema_version"] = 2
            elif kind == "old-schema3":
                config.update(input_schema_version=3, architecture_version=3, structured_dim=186)
            elif kind == "old-dimension":
                config["structured_dim"] = 98
            elif kind == "frozen-field":
                config["freeze_audio"] = True
            elif kind == "staged-initialization":
                payload["identity"]["initialization"] = {"kind": "frozen-audio-with-new-visual-residual"}
            elif kind == "missing-parameter":
                payload["optimizer_state"]["param_groups"][0]["params"].pop()
            elif kind == "wrong-moments":
                next(iter(payload["optimizer_state"]["state"].values()))["exp_avg"] = torch.zeros(1)
            elif kind == "anonymous-shape":
                payload["model_state"]["anonymous_branch.structure_encoder.0.weight"] = torch.zeros(4, 196)
            else:
                del payload["history"][0]["optimizer_updates"]
                del payload["history"][0]["optimizer_skipped_batches"]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                runtime._validate_checkpoint(payload)


if __name__ == "__main__":
    unittest.main()
