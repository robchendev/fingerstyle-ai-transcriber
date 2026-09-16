"""Only synthetic fixtures and tiny toy numerical updates; never pilot data."""

from __future__ import annotations

import copy
import json
import math
import os
import pickle
import random
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from scripts import transcriber_runtime as runtime


IDENTITY = {
    "manifest_sha256": "synthetic-manifest",
    "feature": {"dimensions": 1},
    "model": {"kind": "synthetic-toy", "inputs": 2},
    "encoding": {"kind": "synthetic"},
    "config": {"batch_size": 2},
}
WEIGHTS = {
    "note_onset": 1.0, "fret": 1.0, "pitch": 1.0, "voice": 0.25,
    "duration_log": 0.25, "harmonic_positive": 0.5, "harmonic_kind": 0.25,
    "harmonic_node": 0.25, "percussion_positive": 1.0,
}
STAT_KEYS = (
    "note_onset_positive", "note_onset_negative", "fret", "pitch", "voice", "duration_log",
    "harmonic_positive", "harmonic_kind", "harmonic_node", "percussion_positive",
    "harmonic_sparsity", "percussion_sparsity",
)


def toy_loss(outputs, targets, masks, valid_frames, *, sparsity_weight=0.02):
    logits = outputs["note_onset_logits"]
    target = targets["note_onset"]
    mask = masks["note_onset"] & valid_frames
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    total = logits.sum() * 0
    stats = {name: {"sum": 0.0, "count": 0} for name in STAT_KEYS}
    onset_means = []
    for name, selected in (
        ("note_onset_positive", mask & (target > 0.5)),
        ("note_onset_negative", mask & (target <= 0.5)),
    ):
        values = raw[selected]
        count = int(values.numel())
        numerator = values.sum()
        stats[name] = {"sum": float(numerator.detach()), "count": count}
        if count:
            onset_means.append(numerator / count)
    if onset_means:
        total = total + WEIGHTS["note_onset"] * torch.stack(onset_means).mean()
    for name in ("harmonic", "percussion"):
        if f"{name}_logits" in outputs:
            prediction = outputs[f"{name}_logits"]
            valid = valid_frames.reshape(*valid_frames.shape, *((1,) * (prediction.ndim - 2))).expand_as(prediction)
            values = prediction.sigmoid()[valid]
            stats[f"{name}_sparsity"] = {"sum": values.sum().detach().item(), "count": values.numel()}
            if values.numel():
                total = total + sparsity_weight * values.mean()
    return total, stats


class ToyDataset(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, index):
        length = 2 + index % 2
        return {
            "features": torch.tensor([[(index + frame + 1) / 10] for frame in range(3)]),
            "conditioning": torch.full((3, 1), 0.25),
            "lengths": torch.tensor(length),
            "targets": {"note_onset": torch.tensor([1.0, 0.0, 1.0])},
            "masks": {"note_onset": torch.ones(3, dtype=torch.bool)},
            "valid_frames": torch.arange(3) < length,
            "metadata": "synthetic-not-model-conditioning",
        }


class EpochSampler(Sampler):
    def __init__(self, dataset, seed=29):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.35)
        self.linear = nn.Linear(2, 1)

    def forward(self, features, conditioning, lengths):
        logits = self.linear(self.dropout(torch.cat((features, conditioning), dim=-1))).squeeze(-1)
        return {
            "note_onset_logits": logits, "harmonic_logits": logits * 0.5, "percussion_logits": logits * 0.25,
        }


class ConstantOutputs(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs
        self.dropout = nn.Dropout()
        self.grad_enabled = None

    def forward(self, features, conditioning, lengths):
        self.grad_enabled = torch.is_grad_enabled()
        return self.outputs


class UnsupportedCheckpointObject:
    pass


def loaders(dataset=None):
    dataset = ToyDataset() if dataset is None else dataset
    train = DataLoader(
        dataset, batch_size=2, sampler=EpochSampler(dataset),
        generator=torch.Generator().manual_seed(101), num_workers=0,
    )
    validation = DataLoader(
        dataset, batch_size=2, shuffle=False,
        generator=torch.Generator().manual_seed(202), num_workers=0,
    )
    return train, validation


def toy_model(seed=43):
    torch.manual_seed(seed)
    return ToyModel()


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(
            prefix=".synthetic-runtime-tests-", dir=Path(__file__).resolve().parents[1],
        )
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.loss_patch = mock.patch.object(runtime, "_loss_api", return_value=(toy_loss, WEIGHTS))
        self.loss_patch.start()
        self.addCleanup(self.loss_patch.stop)
        self.original_rng = runtime._capture_rng()
        self.addCleanup(runtime._restore_rng, self.original_rng)
        self.config = runtime.TrainingConfig(epochs=2, max_steps=2, device="cpu")

    def train(self, name="run", *, config=None, model=None, resume=None, pair=None):
        train, validation = loaders() if pair is None else pair
        model = toy_model() if model is None else model
        summary = runtime.run_training(
            model, train, validation, self.config if config is None else config,
            self.root / name, IDENTITY, resume=resume,
        )
        return summary, model

    def assert_tree_equal(self, left, right):
        self.assertEqual(type(left), type(right))
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_tree_equal(first, second)
        else:
            self.assertEqual(left, right)

    def test_training_config_and_device_validation(self):
        for change in (
            {"epochs": 0}, {"epochs": True}, {"learning_rate": 0}, {"learning_rate": math.nan},
            {"weight_decay": -1}, {"gradient_clip": 0}, {"seed": -1}, {"seed": 2**32},
            {"max_steps": 0}, {"max_steps": True}, {"sparsity_weight": math.inf}, {"device": "mps"},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                runtime.TrainingConfig(**change)
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(runtime.resolve_device("auto"), torch.device("cpu"))
            for device in ("cuda", "cuda:0", "cuda:2"):
                with self.assertRaisesRegex(RuntimeError, "unavailable"):
                    runtime.resolve_device(device)
        for device in ("cuda:-1", "CPU", "", None):
            with self.assertRaises(ValueError):
                runtime.resolve_device(device)
        with mock.patch.object(torch.cuda, "is_available", return_value=True), mock.patch.object(
            torch.cuda, "device_count", return_value=1,
        ):
            with self.assertRaisesRegex(ValueError, "index"):
                runtime.resolve_device("cuda:1")

    def test_checkpoint_round_trip_exact_forward_and_json_artifacts(self):
        summary, model = self.train()
        checkpoint = runtime.load_checkpoint(summary["latest_checkpoint"], expected_identity=IDENTITY)
        self.assertEqual(checkpoint["cursor"], {"epoch": 0, "next_batch_index": 2, "global_step": 2})
        self.assertEqual(checkpoint["global_step"], 2)
        self.assertEqual(checkpoint["training_config"], asdict(self.config))
        self.assertEqual(runtime.TrainingConfig(**checkpoint["training_config"]), self.config)
        self.assertEqual(checkpoint["identity"]["model"], IDENTITY["model"])
        self.assertEqual(checkpoint["schema_version"], runtime.SCHEMA_VERSION)
        clone = toy_model(seed=71)
        clone.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        clone.eval()
        batch = next(iter(loaders()[1]))
        with torch.no_grad():
            self.assertTrue(torch.equal(
                model(batch["features"], batch["conditioning"], batch["lengths"])["note_onset_logits"],
                clone(batch["features"], batch["conditioning"], batch["lengths"])["note_onset_logits"],
            ))
        self.assertIsInstance(checkpoint["rng"]["numpy"]["keys"], list)
        for filename in ("run.json", "metrics.json"):
            with (self.root / "run" / filename).open(encoding="utf-8") as stream:
                json.load(stream)
        self.assertTrue((self.root / "run" / "best.pt").is_file())
        self.assertEqual(list((self.root / "run").glob("*.pending")), [])
        json.dumps(summary, allow_nan=False)

    def test_mid_epoch_resume_matches_uninterrupted_dropout_and_optimizer(self):
        full, _ = self.train("full", config=replace(self.config, max_steps=6))
        partial, _ = self.train("resumed")
        random.random()
        np.random.random(4)
        torch.rand(8)
        resumed, _ = self.train(
            "resumed", config=replace(self.config, max_steps=6),
            model=toy_model(seed=999), resume=partial["latest_checkpoint"],
        )
        left = runtime.load_checkpoint(full["latest_checkpoint"])
        right = runtime.load_checkpoint(resumed["latest_checkpoint"])
        for key in ("model_state", "optimizer_state", "cursor", "rng", "loader_state"):
            self.assert_tree_equal(left[key], right[key])
        self.assertEqual(resumed["global_step"], 6)
        self.assertEqual(resumed["epoch"], 2)
        self.assertEqual(resumed["next_batch_index"], 0)

    def test_epoch_boundary_resume_and_total_step_ceiling(self):
        partial, _ = self.train(config=replace(self.config, epochs=1, max_steps=None))
        self.assertEqual(partial["global_step"], 3)
        self.assertEqual(partial["next_batch_index"], 0)
        resumed, _ = self.train(
            config=replace(self.config, epochs=2, max_steps=4), resume=partial["latest_checkpoint"],
        )
        self.assertEqual(resumed["global_step"], 4)
        self.assertEqual(resumed["next_batch_index"], 1)
        before = (self.root / "run" / "latest.pt").read_bytes()
        with mock.patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("Unexpected update")):
            no_op, _ = self.train(
                config=replace(self.config, epochs=2, max_steps=4), resume=resumed["latest_checkpoint"],
            )
        self.assertEqual(no_op["global_step"], 4)
        self.assertEqual(before, (self.root / "run" / "latest.pt").read_bytes())
        for config in (replace(self.config, max_steps=3), replace(self.config, epochs=1, max_steps=5)):
            with self.assertRaises(ValueError):
                self.train(config=config, resume=resumed["latest_checkpoint"])

    def test_wrong_manifest_model_and_config_identity_rejected(self):
        summary, _ = self.train()
        for key in ("manifest_sha256", "feature", "model", "encoding", "config"):
            identity = copy.deepcopy(IDENTITY)
            identity[key] = {"different": True} if key != "manifest_sha256" else "different-synthetic-manifest"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identity"):
                runtime.load_checkpoint(summary["latest_checkpoint"], expected_identity=identity)
        for config in (
            replace(self.config, learning_rate=0.5), replace(self.config, seed=18),
            replace(self.config, gradient_clip=2), replace(self.config, sparsity_weight=0.1),
        ):
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "configuration"):
                self.train(config=config, resume=summary["latest_checkpoint"])
        with self.assertRaisesRegex(ValueError, "exact run"):
            self.train("other", resume=summary["latest_checkpoint"])
        with self.assertRaisesRegex(ValueError, "latest.pt"):
            self.train(resume=self.root / "run" / "best.pt")

    def test_unsafe_and_malformed_checkpoints_are_rejected(self):
        summary, _ = self.train()
        good = runtime.load_checkpoint(summary["latest_checkpoint"])
        unsafe = self.root / "unsafe.pt"
        torch.save(UnsupportedCheckpointObject(), unsafe)
        with self.assertRaises(pickle.UnpicklingError):
            runtime.load_checkpoint(unsafe)
        mutations = (
            lambda value: value.update(schema_version=99),
            lambda value: value.update(unknown=True),
            lambda value: value.pop("model_state"),
            lambda value: value["cursor"].update(global_step=-1),
            lambda value: value["cursor"].update(next_batch_index=99),
            lambda value: value["cursor"].update(global_step=1),
            lambda value: value.update(global_step=1),
            lambda value: value.update(best_score=math.nan),
            lambda value: value["training_config"].update(learning_rate=math.inf),
            lambda value: value["rng"]["numpy"].update(keys=[1]),
            lambda value: value["rng"].update(torch_cpu=torch.zeros(1)),
            lambda value: value["model_state"].update(bad=torch.tensor(math.inf)),
            lambda value: value["optimizer_state"].update(param_groups=[]),
            lambda value: value["optimizer_state"]["param_groups"][0].update(lr=0.9),
            lambda value: value.update(history=[]),
        )
        for index, mutate in enumerate(mutations):
            payload = copy.deepcopy(good)
            mutate(payload)
            filename = self.root / f"malformed-{index}.pt"
            torch.save(payload, filename)
            with self.subTest(index=index), self.assertRaises((ValueError, RuntimeError)):
                runtime.load_checkpoint(filename)
        with mock.patch.object(torch, "load", wraps=torch.load) as load:
            runtime.load_checkpoint(summary["latest_checkpoint"])
            self.assertIs(load.call_args.kwargs["weights_only"], True)
            self.assertEqual(load.call_args.kwargs["map_location"], "cpu")

    def test_fresh_directory_and_aliased_file_guards_preserve_existing_files(self):
        protected = self.root / "run"
        protected.mkdir()
        sentinel = protected / "source.txt"
        sentinel.write_text("synthetic source must survive", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "absent or empty"):
            self.train()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "synthetic source must survive")
        file_path = self.root / "not-a-directory"
        file_path.write_text("synthetic file", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.train("not-a-directory")
        summary, _ = self.train("valid")
        alias = self.root / "hardlink.pt"
        os.link(summary["latest_checkpoint"], alias)
        with self.assertRaisesRegex(ValueError, "non-aliased"):
            runtime.load_checkpoint(alias)
        with self.assertRaisesRegex(ValueError, "non-aliased"):
            self.train("valid", resume=summary["latest_checkpoint"])

    def test_atomic_publication_failure_preserves_previous_file(self):
        checkpoint = self.root / "checkpoint.pt"
        checkpoint.write_bytes(b"synthetic previous checkpoint")

        def failed_write(stream):
            stream.write(b"incomplete replacement")
            raise OSError("synthetic interrupted write")

        with self.assertRaisesRegex(OSError, "interrupted"):
            runtime._atomic_write(checkpoint, failed_write)
        self.assertEqual(checkpoint.read_bytes(), b"synthetic previous checkpoint")
        self.assertEqual(list(self.root.glob("*.pending")), [])
        with self.assertRaises(FileExistsError):
            runtime._atomic_write(checkpoint, lambda stream: stream.write(b"new"), replace=False)
        self.assertEqual(checkpoint.read_bytes(), b"synthetic previous checkpoint")

    def test_foreign_best_checkpoint_and_concurrent_run_lock_are_rejected(self):
        summary, _ = self.train()
        lock = self.root / "run" / ".training.lock"
        lock.write_bytes(b"synthetic concurrent claim")
        before = (self.root / "run" / "latest.pt").read_bytes()
        with self.assertRaises(FileExistsError):
            self.train(resume=summary["latest_checkpoint"], config=replace(self.config, max_steps=3))
        self.assertEqual(lock.read_bytes(), b"synthetic concurrent claim")
        self.assertEqual(before, (self.root / "run" / "latest.pt").read_bytes())
        lock.unlink()
        best = runtime.load_checkpoint(self.root / "run" / "best.pt")
        best["run_id"] = "0" * 32
        torch.save(best, self.root / "run" / "best.pt")
        with self.assertRaisesRegex(ValueError, "consistent run"):
            self.train(resume=summary["latest_checkpoint"])

    def test_resume_rejects_checkpoint_replaced_during_preflight(self):
        summary, _ = self.train()
        checkpoint = runtime.load_checkpoint(summary["latest_checkpoint"])
        evaluate = runtime.evaluate_model

        def replace_checkpoint(*args, **kwargs):
            result = evaluate(*args, **kwargs)
            replacement = self.root / "replacement.pt"
            torch.save(checkpoint, replacement)
            os.replace(replacement, summary["latest_checkpoint"])
            return result

        with mock.patch.object(runtime, "evaluate_model", side_effect=replace_checkpoint):
            with self.assertRaisesRegex(ValueError, "lock was acquired"):
                self.train(resume=summary["latest_checkpoint"], config=replace(self.config, max_steps=3))
        self.assertFalse((self.root / "run" / ".training.lock").exists())
        self.assertEqual(runtime.load_checkpoint(summary["latest_checkpoint"])["cursor"]["global_step"], 2)

    def test_missing_or_unreproducible_loaders_fail_before_training(self):
        train, validation = loaders()
        train.generator = None
        with self.assertRaisesRegex(ValueError, "own CPU"):
            self.train(pair=(train, validation))
        dataset = ToyDataset()
        random_train = DataLoader(
            dataset, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(1),
        )
        with self.assertRaisesRegex(ValueError, "set_epoch"):
            self.train(pair=(random_train, validation))
        train, validation = loaders()
        validation.generator = train.generator
        with self.assertRaisesRegex(ValueError, "independent"):
            self.train(pair=(train, validation))
        self.assertFalse((self.root / "run").exists())

    def test_stochastic_data_and_changed_sampler_seed_reject_resume(self):
        class StochasticDataset(ToyDataset):
            def __getitem__(self, index):
                torch.rand(1)
                return super().__getitem__(index)

        with self.assertRaisesRegex(ValueError, "global RNG"):
            self.train("stochastic", pair=loaders(StochasticDataset()))
        self.assertFalse((self.root / "stochastic" / "latest.pt").exists())
        summary, _ = self.train()
        pair = loaders()
        pair[0].sampler.seed += 1
        with self.assertRaisesRegex(ValueError, "loader configuration changed"):
            self.train(resume=summary["latest_checkpoint"], pair=pair)

    def test_nonfinite_loss_and_gradient_never_publish_checkpoint(self):
        def nonfinite_loss(*args, **kwargs):
            loss, stats = toy_loss(*args, **kwargs)
            return loss * math.nan, stats

        with mock.patch.object(runtime, "_loss_api", return_value=(nonfinite_loss, WEIGHTS)):
            with self.assertRaisesRegex(ValueError, "finite scalar"):
                self.train("bad-loss")
        model = toy_model()
        model.linear.weight.register_hook(lambda gradient: torch.full_like(gradient, math.nan))
        with self.assertRaisesRegex(ValueError, "Nonfinite model gradient"):
            self.train("bad-gradient", model=model)
        self.assertFalse((self.root / "bad-loss" / "latest.pt").exists())
        self.assertFalse((self.root / "bad-gradient" / "latest.pt").exists())

    def test_empty_supervision_is_unavailable_and_training_rejects_it(self):
        class UnlabelledDataset(ToyDataset):
            def __getitem__(self, index):
                value = super().__getitem__(index)
                value["masks"]["note_onset"].zero_()
                return value

        train, validation = loaders(UnlabelledDataset())
        model = toy_model()
        metrics = runtime.evaluate_model(model, validation, "cpu")
        self.assertIsNone(metrics["loss"])
        self.assertFalse(metrics["available"])
        self.assertFalse(metrics["note_onset_frame"]["available"])
        self.assertIsNone(metrics["note_onset_frame"]["f1"])
        with self.assertRaisesRegex(ValueError, "meaningful supervised"):
            self.train(pair=(train, validation))
        self.assertFalse((self.root / "run").exists())
        empty = runtime.evaluate_model(model, [], "cpu")
        self.assertIsNone(empty["loss"])
        self.assertEqual(empty["batches"], 0)

    def test_evaluation_preserves_modes_rng_parameters_and_never_updates(self):
        model = toy_model()
        model.train()
        model.dropout.eval()
        state = copy.deepcopy(model.state_dict())
        validation = loaders()[1]
        rng = runtime._capture_rng()
        loader_rng = validation.generator.get_state()
        with mock.patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("Unexpected update")):
            metrics = runtime.evaluate_model(model, validation, "cpu")
        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)
        self.assert_tree_equal(state, model.state_dict())
        self.assert_tree_equal(rng, runtime._capture_rng())
        self.assertTrue(torch.equal(loader_rng, validation.generator.get_state()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        stats = metrics["loss_statistics"]
        onset_means = [
            stats[name]["sum"] / stats[name]["count"]
            for name in ("note_onset_positive", "note_onset_negative") if stats[name]["count"]
        ]
        expected = WEIGHTS["note_onset"] * sum(onset_means) / len(onset_means)
        expected += self.config.sparsity_weight * sum(
            stats[name]["sum"] / stats[name]["count"] for name in ("harmonic_sparsity", "percussion_sparsity")
        )
        self.assertAlmostEqual(metrics["loss"], expected)

    def test_loss_aggregation_weights_counts_not_batch_means(self):
        model = toy_model()
        dataset = ToyDataset()
        split = DataLoader(dataset, batch_size=4, generator=torch.Generator().manual_seed(1))
        whole = DataLoader(dataset, batch_size=6, generator=torch.Generator().manual_seed(1))
        split_metrics = runtime.evaluate_model(model, split, "cpu")
        whole_metrics = runtime.evaluate_model(model, whole, "cpu")
        self.assertAlmostEqual(split_metrics["loss"], whole_metrics["loss"], places=6)
        self.assertEqual(split_metrics["loss_statistics"]["harmonic_sparsity"]["count"], 15)
        self.assertEqual(split_metrics["loss_statistics"]["percussion_sparsity"]["count"], 15)
        zero_prior = runtime.evaluate_model(model, split, "cpu", sparsity_weight=0.0)
        for name in ("harmonic_sparsity", "percussion_sparsity"):
            self.assertEqual(zero_prior["loss_statistics"][name], split_metrics["loss_statistics"][name])
        model.eval()
        batch_losses = []
        with torch.no_grad():
            for batch in split:
                outputs = model(batch["features"], batch["conditioning"], batch["lengths"])
                loss, _ = toy_loss(outputs, batch["targets"], batch["masks"], batch["valid_frames"])
                batch_losses.append(loss.item())
        self.assertNotAlmostEqual(split_metrics["loss"], sum(batch_losses) / len(batch_losses), places=5)

    def test_metrics_respect_masks_padding_and_positive_only_annotations(self):
        batch = {
            "features": torch.zeros(1, 4, 1),
            "conditioning": torch.zeros(1, 4, 1),
            "lengths": torch.tensor([3]),
            "valid_frames": torch.tensor([[True, True, True, False]]),
            "targets": {
                "note_onset": torch.tensor([[1.0, 0.0, 1.0, 1.0]]),
                "fret": torch.tensor([[0, 1, 1, 0]]),
                "pitch": torch.tensor([[0, 1, 1, 0]]),
                "voice": torch.tensor([[0, 1, 1, 0]]),
                "duration_log": torch.tensor([[1.0, 4.0, 2.0, 3.0]]).log1p(),
                "percussion": torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
                "harmonic": torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
            },
            "masks": {
                "note_onset": torch.tensor([[True, True, False, True]]),
                "fret": torch.tensor([[True, True, False, True]]),
                "pitch": torch.zeros(1, 4, dtype=torch.bool),
                "voice": torch.zeros(1, 4, dtype=torch.bool),
                "duration_log": torch.tensor([[True, False, True, True]]),
                "percussion": torch.tensor([[True, False, False, True]]),
                "harmonic": torch.tensor([[True, False, False, True]]),
            },
            "metadata": ["synthetic-unused-provenance"],
        }
        outputs = {
            "note_onset_logits": torch.tensor([[1.0, 1.0, -1.0, 1.0]]),
            "fret_logits": torch.tensor([[[2.0, 0.0]] * 4]),
            "pitch_logits": torch.tensor([[[2.0, 0.0]] * 4]),
            "voice_logits": torch.tensor([[[2.0, 0.0]] * 4]),
            "duration_log": torch.tensor([[2.0, 7.0, 2.0, 3.0]]).log1p(),
            "percussion_logits": torch.tensor([[-1.0, 1.0, 1.0, 1.0]]),
            "harmonic_logits": torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
        }
        model = ConstantOutputs(outputs)
        metrics = runtime.evaluate_model(model, [batch], "cpu")
        self.assertFalse(model.grad_enabled)
        note = metrics["note_onset_frame"]
        self.assertEqual(note["count"], 2)
        self.assertEqual(note["precision"], 0.5)
        self.assertEqual(note["recall"], 1)
        self.assertAlmostEqual(note["f1"], 2 / 3)
        self.assertEqual(metrics["fret_accuracy"]["accuracy"], 0.5)
        self.assertFalse(metrics["pitch_accuracy"]["available"])
        self.assertIsNone(metrics["voice_accuracy"]["accuracy"])
        self.assertAlmostEqual(metrics["duration_quarter_mae"]["mae"], 0.5)
        percussion = metrics["percussion_positive"]
        self.assertEqual(percussion["labelled_positive_count"], 1)
        self.assertEqual(percussion["recall_at_labelled_positives"], 0)
        self.assertAlmostEqual(percussion["emitted_positive_rate"], 2 / 3)
        for name in ("percussion_positive", "harmonic_presence_positive"):
            self.assertNotIn("precision", metrics[name])
            self.assertNotIn("f1", metrics[name])
        self.assertEqual(metrics["harmonic_presence_positive"]["recall_at_labelled_positives"], 1)
        batch["masks"]["percussion"].zero_()
        unlabelled = runtime.evaluate_model(model, [batch], "cpu")["percussion_positive"]
        self.assertFalse(unlabelled["available"])
        self.assertIsNone(unlabelled["recall_at_labelled_positives"])
        self.assertAlmostEqual(unlabelled["emitted_positive_rate"], 2 / 3)

    def test_bad_loss_statistics_and_evaluation_failure_preserve_model_mode(self):
        def bad_stats(*args, **kwargs):
            loss, stats = toy_loss(*args, **kwargs)
            stats["note_onset_positive"]["sum"] = math.inf
            return loss, stats

        model = toy_model()
        with mock.patch.object(runtime, "_loss_api", return_value=(bad_stats, WEIGHTS)):
            with self.assertRaisesRegex(ValueError, "finite"):
                runtime.evaluate_model(model, loaders()[1], "cpu")
        self.assertTrue(model.training)
        with mock.patch.object(runtime, "_loss_api", return_value=(toy_loss, {})):
            with self.assertRaisesRegex(ValueError, "LOSS_WEIGHTS"):
                runtime.evaluate_model(model, loaders()[1], "cpu")

    def test_harmonic_kind_and_node_are_categorical_with_positive_only_presence(self):
        targets = {
            "harmonic": torch.zeros(1, 3, 6),
            "harmonic_kind": torch.zeros(1, 3, 6, dtype=torch.long),
            "harmonic_node": torch.zeros(1, 3, 6, dtype=torch.long),
            "percussion": torch.zeros(1, 3, 3),
        }
        masks = {name: torch.zeros_like(value, dtype=torch.bool) for name, value in targets.items()}
        for frame in (0, 2):
            targets["harmonic"][0, frame, 0] = 1
            targets["harmonic_kind"][0, frame, 0] = 3
            targets["harmonic_node"][0, frame, 0] = 5
            targets["percussion"][0, frame, 0] = 1
            for mask in masks.values():
                mask[0, frame, 0] = True
        batch = {
            "features": torch.zeros(1, 3, 1), "conditioning": torch.zeros(1, 3, 1),
            "lengths": torch.tensor([2]), "valid_frames": torch.tensor([[True, True, False]]),
            "targets": targets, "masks": masks,
        }
        outputs = {
            "harmonic_logits": torch.ones(1, 3, 6),
            "harmonic_kind_logits": torch.zeros(1, 3, 6, 4),
            "harmonic_node_logits": torch.zeros(1, 3, 6, 6),
            "percussion_logits": torch.ones(1, 3, 3),
        }
        outputs["harmonic_logits"][0, 0, 0] = -1
        outputs["harmonic_kind_logits"][0, 0, 0, 3] = 2
        outputs["harmonic_node_logits"][0, 0, 0, 5] = 3
        def categorical_loss(outputs, targets, masks, valid_frames, *, sparsity_weight):
            stats = {name: {"sum": 0.0, "count": 0} for name in STAT_KEYS}
            total = outputs["harmonic_logits"].sum() * 0
            for name in ("harmonic", "harmonic_kind", "harmonic_node", "percussion"):
                valid = valid_frames[..., None].expand_as(targets[name])
                selected = masks[name] & valid
                logits = outputs[f"{name}_logits"]
                if name in ("harmonic_kind", "harmonic_node"):
                    values = F.cross_entropy(logits[selected], targets[name][selected], reduction="none")
                    stat_name = name
                else:
                    self.assertTrue(torch.all(targets[name][selected] == 1).item())
                    values = F.binary_cross_entropy_with_logits(
                        logits[selected], targets[name][selected], reduction="none",
                    )
                    prior = logits.sigmoid()[valid]
                    stats[f"{name}_sparsity"] = {"sum": prior.sum().item(), "count": prior.numel()}
                    total = total + sparsity_weight * prior.mean()
                    stat_name = f"{name}_positive"
                stats[stat_name] = {"sum": values.sum().item(), "count": values.numel()}
                total = total + WEIGHTS[stat_name] * values.mean()
            return total, stats

        with mock.patch.object(runtime, "_loss_api", return_value=(categorical_loss, WEIGHTS)):
            metrics = runtime.evaluate_model(ConstantOutputs(outputs), [batch], "cpu")
        stats = metrics["loss_statistics"]
        self.assertEqual(stats["harmonic_kind"]["count"], 1)
        self.assertEqual(stats["harmonic_node"]["count"], 1)
        self.assertAlmostEqual(stats["harmonic_node"]["sum"], math.log(math.exp(3) + 5) - 3, places=6)
        self.assertEqual(stats["harmonic_positive"]["count"], 1)
        self.assertEqual(stats["percussion_positive"]["count"], 1)
        self.assertEqual(stats["harmonic_sparsity"]["count"], 12)
        self.assertEqual(stats["percussion_sparsity"]["count"], 6)
        expected = sum(
            WEIGHTS[name] * stats[name]["sum"]
            for name in ("harmonic_positive", "harmonic_kind", "harmonic_node", "percussion_positive")
        ) + 0.02 * (stats["harmonic_sparsity"]["sum"] / 12 + stats["percussion_sparsity"]["sum"] / 6)
        self.assertAlmostEqual(metrics["loss"], expected)
        harmonic = metrics["harmonic_presence_positive"]
        self.assertEqual(harmonic["recall_at_labelled_positives"], 0)
        self.assertAlmostEqual(harmonic["emitted_positive_rate"], 11 / 12)
        for name in ("harmonic_presence_positive", "percussion_positive"):
            self.assertNotIn("precision", metrics[name])
            self.assertNotIn("f1", metrics[name])
        self.assertNotIn("harmonic_fret_mae", metrics)

    def test_frozen_objective_balances_onsets_and_keeps_priors_separate(self):
        weights = {
            "note_onset": 1.0, "fret": 1.0, "pitch": 1.0, "voice": 0.25,
            "duration_log": 0.25, "harmonic_positive": 0.5, "harmonic_kind": 0.25,
            "harmonic_node": 0.25, "percussion_positive": 1.0,
        }
        stats = {
            "note_onset_positive": {"sum": 4.0, "count": 2},
            "note_onset_negative": {"sum": 9.0, "count": 9},
            "fret": {"sum": 0.0, "count": 0}, "pitch": {"sum": 0.0, "count": 0},
            "voice": {"sum": 0.0, "count": 0}, "duration_log": {"sum": 0.0, "count": 0},
            "harmonic_positive": {"sum": 3.0, "count": 3},
            "harmonic_kind": {"sum": 0.0, "count": 0}, "harmonic_node": {"sum": 0.0, "count": 0},
            "percussion_positive": {"sum": 2.0, "count": 1},
            "harmonic_sparsity": {"sum": 4.0, "count": 8},
            "percussion_sparsity": {"sum": 6.0, "count": 30},
        }
        parsed = runtime._stats(stats, weights)
        self.assertAlmostEqual(runtime._objective(parsed, weights, 0.02), 1.5 + 0.5 + 2 + 0.02 * (0.5 + 0.2))
        stats["note_onset_negative"] = {"sum": 0.0, "count": 0}
        self.assertAlmostEqual(runtime._objective(stats, weights, 0.02), 2 + 0.5 + 2 + 0.02 * (0.5 + 0.2))
        for name in stats:
            if name not in ("harmonic_sparsity", "percussion_sparsity"):
                stats[name] = {"sum": 0.0, "count": 0}
        self.assertIsNone(runtime._objective(stats, weights, 0.02))

    def test_disagreement_between_loss_and_statistics_is_rejected(self):
        def inconsistent_loss(*args, **kwargs):
            loss, stats = toy_loss(*args, **kwargs)
            return loss + 1, stats

        with mock.patch.object(runtime, "_loss_api", return_value=(inconsistent_loss, WEIGHTS)):
            with self.assertRaisesRegex(ValueError, "statistics disagree"):
                runtime.evaluate_model(toy_model(), loaders()[1], "cpu")

    def test_unknown_target_nan_is_allowed_only_outside_effective_masks(self):
        batch = {
            "features": torch.zeros(1, 3, 1), "conditioning": torch.zeros(1, 3, 1),
            "lengths": torch.tensor([2]), "valid_frames": torch.tensor([[True, True, False]]),
            "targets": {"note_onset": torch.tensor([[1.0, math.nan, math.nan]])},
            "masks": {"note_onset": torch.tensor([[True, False, True]])},
        }
        model = ConstantOutputs({"note_onset_logits": torch.zeros(1, 3)})
        metrics = runtime.evaluate_model(model, [batch], "cpu")
        self.assertTrue(metrics["available"])
        self.assertEqual(metrics["note_onset_frame"]["count"], 1)
        self.assertEqual(metrics["loss_statistics"]["note_onset_positive"]["count"], 1)
        batch["masks"]["note_onset"][0, 1] = True
        with self.assertRaisesRegex(ValueError, "supervised targets.note_onset"):
            runtime.evaluate_model(model, [batch], "cpu")

    def test_actual_model_evaluation_matches_frozen_loss_without_training(self):
        from scripts.transcriber_model import LOSS_STAT_KEYS, LOSS_WEIGHTS, FingerstyleTranscriber, ModelConfig, masked_loss

        self.assertEqual(WEIGHTS, LOSS_WEIGHTS)
        self.assertEqual(STAT_KEYS, LOSS_STAT_KEYS)
        config = ModelConfig(hidden_size=8, recurrent_layers=1, max_fret=3, max_voices=2)
        model = FingerstyleTranscriber(config)
        lengths = torch.tensor([4, 2])
        valid = torch.arange(4)[None, :] < lengths[:, None]
        targets = {}
        masks = {}
        categorical = {"fret", "pitch", "voice", "harmonic_kind", "harmonic_node"}
        for name in (
            "note_onset", "fret", "pitch", "voice", "duration_log",
            "harmonic", "harmonic_kind", "harmonic_node", "percussion",
        ):
            shape = (2, 4, 3 if name == "percussion" else 6)
            targets[name] = torch.full(shape, -100, dtype=torch.long) if name in categorical else torch.full(shape, math.nan)
            masks[name] = torch.zeros(shape, dtype=torch.bool)
        targets["note_onset"][valid] = 0
        masks["note_onset"].fill_(True)
        for name, value in (
            ("note_onset", 1), ("fret", 2), ("pitch", 60), ("voice", 1),
            ("duration_log", math.log1p(0.5)), ("harmonic", 1),
            ("harmonic_kind", 2), ("harmonic_node", 4),
        ):
            targets[name][0, 1, 0] = value
            masks[name][0, 1, 0] = True
        targets["percussion"][1, 1, 1] = 1
        masks["percussion"][1, 1, 1] = True
        batch = {
            "features": torch.linspace(-0.3, 0.7, 2 * 4 * config.n_mels).reshape(2, 4, config.n_mels),
            "conditioning": torch.zeros(2, 4, config.conditioning_dim),
            "lengths": lengths, "valid_frames": valid, "targets": targets, "masks": masks,
        }
        model.eval()
        with torch.no_grad():
            outputs = model(batch["features"], batch["conditioning"], lengths)
            expected_loss, expected_stats = masked_loss(outputs, targets, masks, valid, sparsity_weight=0.02)
        self.assertEqual(set(expected_stats), set(LOSS_STAT_KEYS))
        model.train()
        state = copy.deepcopy(model.state_dict())
        self.loss_patch.stop()
        with mock.patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("Actual model must not train")):
            metrics = runtime.evaluate_model(model, [batch], "cpu")
        self.assertAlmostEqual(metrics["loss"], expected_loss.item(), places=5)
        self.assertEqual(metrics["loss_statistics"], expected_stats)
        self.assertEqual(metrics["loss_statistics"]["note_onset_positive"]["count"], 1)
        self.assertEqual(metrics["loss_statistics"]["note_onset_negative"]["count"], 35)
        self.assertEqual(metrics["harmonic_presence_positive"]["labelled_positive_count"], 1)
        self.assertEqual(metrics["percussion_positive"]["labelled_positive_count"], 1)
        self.assertTrue(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assert_tree_equal(state, model.state_dict())

    def test_loss_stat_schema_rejects_removed_aliases_and_missing_keys(self):
        stats = {name: {"sum": 0.0, "count": 0} for name in STAT_KEYS}
        for original, alias in (
            ("harmonic_sparsity", "sparse_prior"), ("harmonic_sparsity", "sparsity_prior"),
            ("harmonic_sparsity", "sparsity"), ("harmonic_positive", "harmonic"),
            ("percussion_positive", "percussion"), ("note_onset_positive", "note_onset"),
            ("harmonic_node", "harmonic_fret"),
        ):
            invalid = copy.deepcopy(stats)
            invalid[alias] = invalid.pop(original)
            with self.subTest(alias=alias), self.assertRaisesRegex(ValueError, "loss statistics schema"):
                runtime._stats(invalid, WEIGHTS)
        stats.pop("percussion_sparsity")
        with self.assertRaisesRegex(ValueError, "loss statistics schema"):
            runtime._stats(stats, WEIGHTS)


if __name__ == "__main__":
    unittest.main()
