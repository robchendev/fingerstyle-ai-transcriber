"""Fake-clock wall-clock pauses; every optimizer input is synthetic."""

from copy import deepcopy
from dataclasses import asdict, replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts import transcriber, transcriber_runtime as runtime
from scripts.dataset_io import ROOT, read_json
from scripts.transcriber_video import VideoConfig
from tests.test_transcriber_runtime import IDENTITY, WEIGHTS, loaders, toy_loss, toy_model
from tests import test_transcriber_runtime as runtime_fixtures
from tests import test_transcriber_joint_runtime as joint_fixtures


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class BudgetTests(unittest.TestCase):
    assert_tree_equal = runtime_fixtures.RuntimeTests.assert_tree_equal

    def setUp(self):
        directory = TemporaryDirectory(prefix=".synthetic-budget-", dir=ROOT)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.clock = Clock()
        self.enterContext(patch.object(runtime.time, "perf_counter", self.clock))
        self.enterContext(patch.object(runtime, "_loss_api", return_value=(toy_loss, WEIGHTS)))
        rng = runtime._capture_rng()
        self.addCleanup(runtime._restore_rng, rng)
        self.config = runtime.TrainingConfig(epochs=2, max_steps=6, device="cpu", max_seconds=100.)

    def run_toy(self, name, *, config=None, progress=None, resume=None, evaluator=None, **kwargs):
        return runtime.run_training(
            toy_model(), *loaders(), config or self.config, self.root / name, IDENTITY,
            progress=progress, resume=resume, event_evaluator=evaluator, **kwargs,
        )

    def checkpoint(self, summary):
        return runtime.load_checkpoint(summary["latest_checkpoint"])

    def assert_exact_resume(self, full, resumed):
        full, resumed = self.checkpoint(full), self.checkpoint(resumed)
        for key in ("model_state", "optimizer_state", "rng", "loader_state", "cursor", "history"):
            self.assert_tree_equal(full[key], resumed[key])

    def test_mid_epoch_pause_preserves_cursor_without_fabricating_best(self):
        full = self.run_toy("full")

        def pause(message):
            if "Epoch 1/2 | batch 1/" in message:
                self.clock.now = 95.

        partial = self.run_toy("resumed", progress=pause)
        self.assertEqual(partial["stopped_by"], "max_seconds")
        self.assertEqual(partial["status"], "paused")
        self.assertEqual(partial["global_step"], 1)
        self.assertEqual(partial["total_optimizer_updates"], 1)
        self.assertTrue(partial["validation_pending"])
        checkpoint = self.checkpoint(partial)
        self.assertEqual(checkpoint["resume_state"]["phase"], "training")
        self.assertEqual(checkpoint["resume_state"]["optimizer_updates"], 1)
        self.assertEqual(checkpoint["history"], [])
        self.assertIsNone(partial["best_checkpoint"])
        self.assertIsNone(partial["best_event_checkpoint"])
        self.assertFalse((self.root / "resumed" / "best.pt").exists())
        self.assertFalse((self.root / "resumed" / "best-events.pt").exists())
        self.clock.now = 500.
        resumed = self.run_toy("resumed", config=replace(self.config, max_seconds=200.), resume=partial["latest_checkpoint"])
        self.assertEqual(resumed["optimizer_updates"], 5)
        self.assertFalse(resumed["validation_pending"])
        self.assert_exact_resume(full, resumed)

    def test_expired_before_initial_validation_saves_zero_update_resume_state(self):
        self.clock.now = 100.
        with patch.object(runtime, "evaluate_model", side_effect=AssertionError("No validation budget")), patch.object(
            torch.optim.AdamW, "step", side_effect=AssertionError("No update budget"),
        ):
            partial = self.run_toy("initial", started_at=0., deadline=100.)
        self.assertFalse(partial["training_started"])
        self.assertEqual(partial["global_step"], 0)
        self.assertTrue(partial["validation_pending"])
        self.assertIsNone(partial["validation"])
        self.assertEqual(partial["elapsed_seconds"], 100.)
        state = self.checkpoint(partial)
        self.assertEqual(state["resume_state"]["phase"], "initial_validation")
        self.assertEqual(state["optimizer_state"]["state"], {})
        resumed = self.run_toy("initial", resume=partial["latest_checkpoint"])
        full = self.run_toy("full")
        self.assert_exact_resume(full, resumed)

    def test_pause_during_initial_validation_restores_readonly_rng(self):
        def pause(message):
            if "Initial validation: batch 1/" in message:
                self.clock.now = 95.

        partial = self.run_toy("initial", progress=pause)
        self.assertEqual(partial["global_step"], 0)
        self.assertIsNone(partial["validation"])
        resumed = self.run_toy("initial", resume=partial["latest_checkpoint"])
        full = self.run_toy("full")
        self.assert_exact_resume(full, resumed)

    def test_event_timeout_leaves_validation_pending_and_replays_no_updates(self):
        event_result = {"score": .5, "metric": "synthetic-event-f1", "windows": 6}
        full = self.run_toy("full", evaluator=lambda _: event_result)

        def interrupted(model):
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            torch.rand(4)
            self.clock.now = 95.
            batch = next(iter(loaders()[1]))
            model(batch["features"], batch["conditioning"], batch["lengths"])
            self.fail("The event-forward boundary should stop the callback")

        partial = self.run_toy("events", evaluator=interrupted)
        state = self.checkpoint(partial)
        self.assertEqual(state["global_step"], 3)
        self.assertEqual(state["resume_state"]["phase"], "validation")
        self.assertEqual(state["history"], [])
        self.assertIsNone(partial["best_score"])
        self.assertIsNone(partial["best_event_checkpoint"])
        with self.assertRaisesRegex(ValueError, "event-evaluation policy"):
            self.run_toy("events", resume=partial["latest_checkpoint"])
        resumed = self.run_toy("events", resume=partial["latest_checkpoint"], evaluator=lambda _: event_result)
        self.assertEqual(resumed["optimizer_updates"], 3)
        self.assert_exact_resume(full, resumed)
        self.assertEqual([row["global_step"] for row in self.checkpoint(resumed)["history"]], [3, 6])

    def test_later_pause_keeps_only_real_best_metrics(self):
        def pause(message):
            if "Epoch 2/2 | batch 1/" in message:
                self.clock.now = 95.

        partial = self.run_toy("later", progress=pause, evaluator=lambda _: {"score": .7, "metric": "synthetic-f1", "windows": 6})
        latest = self.checkpoint(partial)
        best = runtime.load_checkpoint(partial["best_checkpoint"])
        events = runtime.load_checkpoint(partial["best_event_checkpoint"])
        self.assertEqual(latest["global_step"], 4)
        self.assertEqual([row["global_step"] for row in latest["history"]], [3])
        self.assertEqual(best["global_step"], 3)
        self.assertEqual(events["global_step"], 3)
        self.assertTrue(latest["resume_state"]["validation_pending"])

    def test_checkpoint_overrun_is_reported_and_reserve_scales(self):
        save = torch.save

        def slow_save(*args, **kwargs):
            save(*args, **kwargs)
            self.clock.now += 20.

        def pause(message):
            if "Initial validation: starting" in message:
                self.clock.now = 95.

        with patch.object(torch, "save", side_effect=slow_save):
            result = self.run_toy("slow-checkpoint", progress=pause)
        self.assertEqual(result["checkpoint_reserve_seconds"], 10.)
        self.assertEqual(result["elapsed_seconds"], 115.)
        self.assertEqual(result["budget_overrun_seconds"], 15.)
        self.assertEqual(result["status"], "paused")
        self.assertEqual(runtime.TrainingBudget(61200.).reserve_seconds, 120.)
        self.assertAlmostEqual(runtime.TrainingBudget(.1).reserve_seconds, .01)

    def test_pending_checkpoint_schema_counts_and_identity_fail_closed(self):
        self.clock.now = 95.
        partial = self.run_toy("schema", started_at=0., deadline=100.)
        original = self.checkpoint(partial)
        for change in (
            {"phase": "unknown"}, {"validation_pending": False}, {"validation_pending": 1},
            {"optimizer_updates": 1}, {"optimizer_skipped_batches": -1},
            {"stopped_by": "done"}, {"unknown": 1},
            {"stopped_by": None}, {"event_evaluation_required": 1},
        ):
            payload = deepcopy(original)
            payload["resume_state"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                runtime._validate_checkpoint(payload)
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.run_toy("schema", config=replace(self.config, learning_rate=.2), resume=partial["latest_checkpoint"])
        for value in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(self.config, max_seconds=value)

    def test_pause_before_first_batch_does_not_double_advance_loader_generator(self):
        def pause(message):
            if "Epoch 1/2: starting" in message:
                self.clock.now = 95.

        partial = self.run_toy("loader", progress=pause)
        self.assertEqual(partial["global_step"], 0)
        self.assertFalse(partial["validation_pending"])
        resumed = self.run_toy("loader", resume=partial["latest_checkpoint"])
        full = self.run_toy("full")
        self.assert_exact_resume(full, resumed)

    def test_standalone_readonly_evaluation_does_not_change_resume_state(self):
        def pause(message):
            if "Epoch 1/2 | batch 1/" in message:
                self.clock.now = 95.

        partial = self.run_toy("readonly", progress=pause)
        before = Path(partial["latest_checkpoint"]).read_bytes()
        model = toy_model()
        model.load_state_dict(self.checkpoint(partial)["model_state"])
        self.clock.now = 10000.
        runtime.evaluate_model(model, loaders()[1], "cpu")
        self.assertEqual(Path(partial["latest_checkpoint"]).read_bytes(), before)
        resumed = self.run_toy("readonly", resume=partial["latest_checkpoint"])
        full = self.run_toy("full")
        self.assert_exact_resume(full, resumed)


class JointBudgetTests(unittest.TestCase):
    model = joint_fixtures.JointRuntimeTests.model
    items = joint_fixtures.JointRuntimeTests.items
    assert_tree_equal = joint_fixtures.JointRuntimeTests.assert_tree_equal

    def test_joint_dropout_budget_resume_matches_every_training_state(self):
        from torch.utils.data import DataLoader
        from scripts.transcriber_data import collate_windows

        previous_threads, rng = torch.get_num_threads(), runtime._capture_rng()
        self.addCleanup(torch.set_num_threads, previous_threads)
        self.addCleanup(runtime._restore_rng, rng)
        torch.set_num_threads(1)
        clock = Clock()

        with TemporaryDirectory(prefix=".joint-budget-", dir=ROOT) as directory, patch.object(runtime.time, "perf_counter", clock):
            def train(name, *, resume=None, progress=None):
                model = self.model()
                pair = [DataLoader(self.items(), batch_size=1, collate_fn=collate_windows,
                                   generator=torch.Generator().manual_seed(seed)) for seed in (101, 202)]
                return runtime.run_training(model, *pair, runtime.TrainingConfig(epochs=2, max_seconds=100., device="cpu"),
                                            Path(directory) / name, {"model": asdict(model.config), "video": {"config": asdict(model.video_config)}},
                                            resume=resume, progress=progress)

            def pause(message):
                if "Epoch 1/2 | batch 1/" in message:
                    clock.now = 95.

            full = train("full")
            partial = train("resumed", progress=pause)
            self.assertEqual(partial["total_optimizer_updates"], 1)
            resumed = train("resumed", resume=partial["latest_checkpoint"])
            left, right = runtime.load_checkpoint(full["latest_checkpoint"]), runtime.load_checkpoint(resumed["latest_checkpoint"])
            for key in ("model_state", "optimizer_state", "rng", "loader_state", "cursor", "history", "resume_state"):
                self.assert_tree_equal(left[key], right[key])

    def test_joint_budget_checkpoint_counts_skipped_batches_without_validation_history(self):
        from torch.utils.data import DataLoader
        from scripts.transcriber_data import collate_windows

        previous_threads, rng = torch.get_num_threads(), runtime._capture_rng()
        self.addCleanup(torch.set_num_threads, previous_threads)
        self.addCleanup(runtime._restore_rng, rng)
        torch.set_num_threads(1)
        clock = Clock()
        with TemporaryDirectory(prefix=".joint-budget-skips-", dir=ROOT) as directory, patch.object(runtime.time, "perf_counter", clock):
            def train(name, *, resume=None, progress=None):
                model = self.model()
                train_items, validation_items = self.items(), self.items()
                for mask in train_items[0]["masks"].values():
                    mask.zero_()
                pair = [DataLoader(items, batch_size=1, collate_fn=collate_windows,
                                   generator=torch.Generator().manual_seed(seed))
                        for items, seed in ((train_items, 101), (validation_items, 202))]
                return runtime.run_training(model, *pair, runtime.TrainingConfig(epochs=1, max_seconds=100., device="cpu"),
                                            Path(directory) / name, {"model": asdict(model.config), "video": {"config": asdict(model.video_config)}},
                                            resume=resume, progress=progress)

            def pause(message):
                if "Epoch 1/1 | batch 1/" in message:
                    clock.now = 95.

            full = train("full")
            partial = train("resumed", progress=pause)
            checkpoint = runtime.load_checkpoint(partial["latest_checkpoint"])
            self.assertEqual(partial["global_step"], 1)
            self.assertEqual(partial["total_optimizer_updates"], 0)
            self.assertEqual(partial["total_optimizer_skipped_batches"], 1)
            self.assertEqual(checkpoint["history"], [])
            self.assertEqual(checkpoint["optimizer_state"]["state"], {})
            with self.assertRaisesRegex(ValueError, "actual optimizer updates"):
                transcriber.checkpoint_model(partial["latest_checkpoint"], "cpu")
            resumed = train("resumed", resume=partial["latest_checkpoint"])
            left, right = runtime.load_checkpoint(full["latest_checkpoint"]), runtime.load_checkpoint(resumed["latest_checkpoint"])
            for key in ("model_state", "optimizer_state", "rng", "loader_state", "cursor", "history", "resume_state"):
                self.assert_tree_equal(left[key], right[key])


class BudgetCommandTests(unittest.TestCase):
    def test_setup_expiration_returns_summary_without_creating_model_or_optimizer(self):
        clock = Clock()
        values = transcriber.load_config()
        config = {**values[0], "video": {"index": "synthetic-index.json", "model": asdict(VideoConfig())}}
        dataset = SimpleNamespace()

        def slow_dataset(*_):
            clock.now = 110.
            return dataset

        with TemporaryDirectory(prefix=".budget-cli-", dir=ROOT) as directory, patch.object(transcriber.time, "perf_counter", clock), patch.object(
            transcriber, "load_config", return_value=(config, *values[1:]),
        ), patch.object(transcriber, "make_dataset", side_effect=slow_dataset) as make_dataset, patch(
            "scripts.transcriber_model.FingerstyleTranscriber", side_effect=AssertionError("No model budget"),
        ), patch("torch.optim.AdamW", side_effect=AssertionError("No optimizer budget")), patch("sys.stdout", new=StringIO()):
            command = ["train", "--data-root", directory, "--run-dir", "runs\\paused", "--max-hours", str(100 / 3600)]
            self.assertEqual(transcriber.main(command), 0)
            result = read_json(Path(directory) / "runs" / "paused" / "summary.json")
            self.assertEqual(make_dataset.call_count, 1)
            self.assertEqual(result["stopped_by"], "max_seconds")
            self.assertFalse(result["training_started"])
            self.assertIsNone(result["latest_checkpoint"])
            self.assertEqual(result["budget_overrun_seconds"], 10.)
            self.assertEqual(result["setup_seconds"], 110.)
            self.assertEqual(result["global_step"], 0)
            self.assertIn("new empty run directory", result["resume_action"])
            with patch("sys.stderr", new=StringIO()):
                self.assertEqual(transcriber.main(command), 1)

    def test_budget_is_not_run_identity_and_cli_overrides_config(self):
        values = transcriber.load_config()
        config = {**values[0], "video": {"index": "synthetic-index.json", "model": asdict(VideoConfig())}}
        dataset = SimpleNamespace(manifest_sha256="synthetic", video_identity={"synthetic": True})
        left = transcriber.run_identity(dataset, config, values[1], values[2], values[3], "cpu")
        right = transcriber.run_identity(dataset, config, values[1], values[2], replace(values[3], max_seconds=61200.), "cpu")
        self.assertEqual(left, right)
        clock = Clock()
        with TemporaryDirectory(prefix=".budget-override-", dir=ROOT) as directory, patch.object(transcriber.time, "perf_counter", clock), patch.object(
            transcriber, "load_config", return_value=(config, *values[1:]),
        ), patch.object(transcriber, "_train_with_budget", side_effect=runtime.TrainingBudgetExpired) as train, patch("sys.stdout", new=StringIO()):
            result = transcriber.train(transcriber.argument_parser().parse_args(
                ["train", "--data-root", directory, "--run-dir", "runs\\override", "--max-hours", "17"],
            ))
            self.assertEqual(result["max_seconds"], 61200.)
            self.assertEqual(train.call_args.args[4].max_seconds, 61200.)
            self.assertEqual(train.call_args.args[6].deadline, 61200.)


if __name__ == "__main__":
    unittest.main()
