"""Explicitly invoked training, masked evaluation, and safe local checkpoints."""

from __future__ import annotations

import json
import math
import os
import random
import re
import stat
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler


SCHEMA_VERSION = 1
_PRIOR_NAMES = frozenset({"harmonic_sparsity", "percussion_sparsity"})
_NOTE_ONSET_STATS = ("note_onset_positive", "note_onset_negative")
_LOSS_STAT_KEYS = (
    *_NOTE_ONSET_STATS, "fret", "pitch", "voice", "duration_log", "harmonic_positive",
    "harmonic_kind", "harmonic_node", "percussion_positive", "harmonic_sparsity", "percussion_sparsity",
)
_CHECKPOINT_KEYS = {
    "schema_version", "run_id", "identity", "training_config", "runtime", "model_state",
    "optimizer_state", "cursor", "global_step", "best_score", "rng", "loader_state", "history",
}
_OPTIMIZER_OPTIONS = {
    "betas": (0.9, 0.999), "eps": 1e-8, "amsgrad": False, "maximize": False,
    "foreach": False, "capturable": False, "differentiable": False, "fused": False,
}


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value, name, minimum=0.0, *, positive=False):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < minimum or (positive and value == minimum):
        raise ValueError(f"{name} must be {'>' if positive else '>='} {minimum}")
    return float(value)


def _device_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", value):
        raise ValueError("device must be auto, cpu, cuda, or cuda:N")
    return value


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    seed: int = 17
    device: str = "auto"
    max_steps: int | None = None
    sparsity_weight: float = 0.02

    def __post_init__(self):
        _integer(self.epochs, "epochs", 1)
        _finite(self.learning_rate, "learning_rate", positive=True)
        _finite(self.weight_decay, "weight_decay")
        _finite(self.gradient_clip, "gradient_clip", positive=True)
        _integer(self.seed, "seed")
        if self.seed > 2**32 - 1:
            raise ValueError("seed must fit an unsigned 32-bit integer")
        _device_name(self.device)
        if self.max_steps is not None:
            _integer(self.max_steps, "max_steps", 1)
        _finite(self.sparsity_weight, "sparsity_weight")


def resolve_device(device="auto"):
    """Resolve a requested device without silently downgrading explicit CUDA."""
    _device_name(device)
    if device == "cpu" or (device == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({device}) but is unavailable")
    index = int(device.split(":")[1]) if ":" in device else torch.cuda.current_device()
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {index} is unavailable")
    return torch.device("cuda", index)


def _loss_api():
    from scripts.transcriber_model import LOSS_WEIGHTS, masked_loss

    return masked_loss, LOSS_WEIGHTS


def _json_copy(value):
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _json_copy(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_json_copy(item) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("Expected finite JSON primitives, not custom objects")


def _json_equal(left, right):
    return json.dumps(_json_copy(left), sort_keys=True) == json.dumps(_json_copy(right), sort_keys=True)


def _identity(value):
    result = _json_copy(value)
    if not isinstance(result, dict) or not isinstance(result.get("model"), dict):
        raise ValueError("identity must be an object containing a model configuration object")
    return result


def _plain_path(path):
    result = Path(os.path.abspath(path))
    for component in (result, *result.parents):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"Refusing an aliased or reparse-point path: {component}")
    return result


def _regular_file(path):
    path = _plain_path(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"Expected a regular, non-aliased file: {path}")
    return path


def _file_stamp(path):
    info = _regular_file(path).stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _atomic_write(path, write, *, replace=True):
    path = _plain_path(path)
    if path.exists():
        _regular_file(path)
    pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        with pending.open("xb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            _regular_file(path)
        if replace:
            os.replace(pending, path)
        else:
            # Publish a complete new file without replacing a concurrent claim.
            os.link(pending, path)
    finally:
        if pending.exists():
            pending.unlink()


def _write_json(path, value, *, replace=True):
    data = (json.dumps(_json_copy(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _atomic_write(path, lambda stream: stream.write(data), replace=replace)


def _read_json(path):
    with _regular_file(path).open("r", encoding="utf-8") as stream:
        return _json_copy(json.load(stream))


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"Invalid {name} schema")


def _tensor(value, name, *, byte=False, finite=True):
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise ValueError(f"{name} must be a dense tensor")
    if byte and (value.dtype != torch.uint8 or value.ndim != 1):
        raise ValueError(f"{name} must be a one-dimensional byte tensor")
    if finite and (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
        raise ValueError(f"{name} contains nonfinite values")


def _safe_tree(value, name):
    if isinstance(value, torch.Tensor):
        _tensor(value, name)
    elif type(value) in (dict, list, tuple):
        entries = value.items() if isinstance(value, dict) else enumerate(value)
        for key, item in entries:
            if type(key) not in (str, int):
                raise ValueError(f"Invalid key in {name}")
            _safe_tree(item, name)
    elif value is not None and type(value) not in (str, bool, int, float):
        raise ValueError(f"Unsafe object in {name}")
    elif type(value) is float and not math.isfinite(value):
        raise ValueError(f"Nonfinite value in {name}")


def _capture_rng():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _validate_rng(rng):
    _keys(rng, {"python", "numpy", "torch_cpu", "torch_cuda"}, "RNG")
    python_state = rng["python"]
    if not isinstance(python_state, tuple) or len(python_state) != 3:
        raise ValueError("Invalid Python RNG state")
    if python_state[0] != 3 or not isinstance(python_state[1], tuple) or len(python_state[1]) != 625:
        raise ValueError("Invalid Python RNG state")
    for word in python_state[1][:-1]:
        if _integer(word, "Python RNG word") > 2**32 - 1:
            raise ValueError("Invalid Python RNG word")
    if _integer(python_state[1][-1], "Python RNG position") > 624:
        raise ValueError("Invalid Python RNG position")
    if python_state[2] is not None and (
        type(python_state[2]) not in (int, float) or not math.isfinite(python_state[2])
    ):
        raise ValueError("Invalid Python Gaussian cache")
    numpy_state = rng["numpy"]
    _keys(numpy_state, {"bit_generator", "keys", "position", "has_gauss", "cached_gaussian"}, "NumPy RNG")
    if numpy_state["bit_generator"] != "MT19937" or type(numpy_state["keys"]) is not list:
        raise ValueError("Invalid NumPy RNG algorithm or keys")
    if len(numpy_state["keys"]) != 624:
        raise ValueError("Invalid NumPy RNG keys")
    for word in numpy_state["keys"]:
        if _integer(word, "NumPy RNG word") > 2**32 - 1:
            raise ValueError("Invalid NumPy RNG word")
    if _integer(numpy_state["position"], "NumPy RNG position") > 624:
        raise ValueError("Invalid NumPy RNG position")
    if type(numpy_state["has_gauss"]) is not int or numpy_state["has_gauss"] not in (0, 1):
        raise ValueError("Invalid NumPy Gaussian flag")
    if type(numpy_state["cached_gaussian"]) not in (int, float) or not math.isfinite(numpy_state["cached_gaussian"]):
        raise ValueError("Invalid NumPy Gaussian cache")
    _tensor(rng["torch_cpu"], "CPU RNG", byte=True)
    torch.Generator().set_state(rng["torch_cpu"])
    if type(rng["torch_cuda"]) is not list:
        raise ValueError("Invalid CUDA RNG states")
    for value in rng["torch_cuda"]:
        _tensor(value, "CUDA RNG", byte=True)
        if value.numel() == 0:
            raise ValueError("Empty CUDA RNG state")


def _restore_rng(rng):
    random.setstate(rng["python"])
    state = rng["numpy"]
    np.random.set_state((
        state["bit_generator"], np.asarray(state["keys"], dtype=np.uint32),
        state["position"], state["has_gauss"], state["cached_gaussian"],
    ))
    torch.set_rng_state(rng["torch_cpu"])
    if rng["torch_cuda"]:
        if not torch.cuda.is_available() or len(rng["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG topology differs from this runtime")
        torch.cuda.set_rng_state_all(rng["torch_cuda"])


def _loader_signature(loader, *, training):
    if not isinstance(loader, DataLoader) or loader.num_workers != 0:
        raise ValueError("Reproducible training requires a DataLoader with num_workers=0")
    if (
        not isinstance(loader.generator, torch.Generator) or loader.generator.device.type != "cpu"
        or loader.generator is torch.default_generator
    ):
        raise ValueError("DataLoader must use its own CPU torch.Generator")
    if type(loader.batch_sampler) is not BatchSampler:
        raise ValueError("Exact resume requires the standard DataLoader batch sampler")
    if training:
        if not isinstance(loader.sampler, SequentialSampler) and not callable(getattr(loader.sampler, "set_epoch", None)):
            raise ValueError("Training sampler must be sequential or support deterministic set_epoch(epoch)")
    elif not isinstance(loader.sampler, SequentialSampler):
        raise ValueError("Validation DataLoader must have fixed sequential order")
    batches = _integer(len(loader), "loader batch count", 1)
    size = _integer(len(loader.dataset), "loader dataset size", 1)
    return {
        "batches": batches, "dataset_size": size, "batch_size": loader.batch_size,
        "drop_last": loader.drop_last, "num_workers": loader.num_workers,
        "sampler": f"{type(loader.sampler).__module__}.{type(loader.sampler).__qualname__}",
        "sampler_seed": _json_copy(getattr(loader.sampler, "seed", None)),
    }


def _validate_loader_signature(signature):
    _keys(signature, {
        "batches", "dataset_size", "batch_size", "drop_last", "num_workers", "sampler", "sampler_seed",
    }, "loader signature")
    _integer(signature["batches"], "loader batches", 1)
    _integer(signature["dataset_size"], "dataset size", 1)
    if signature["batch_size"] is not None:
        _integer(signature["batch_size"], "batch size", 1)
    if type(signature["drop_last"]) is not bool or signature["num_workers"] != 0:
        raise ValueError("Invalid loader options")
    if type(signature["sampler"]) is not str or not signature["sampler"]:
        raise ValueError("Invalid sampler identity")
    _json_copy(signature["sampler_seed"])


def _validate_checkpoint(payload):
    _keys(payload, _CHECKPOINT_KEYS, "checkpoint")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema version")
    if not isinstance(payload["run_id"], str) or not re.fullmatch("[0-9a-f]{32}", payload["run_id"]):
        raise ValueError("Invalid run identity")
    _identity(payload["identity"])
    _keys(payload["training_config"], asdict(TrainingConfig()), "training config")
    config = TrainingConfig(**payload["training_config"])
    runtime = payload["runtime"]
    _keys(runtime, {
        "torch_version", "numpy_version", "device", "train_loader", "validation_loader",
        "num_threads", "num_interop_threads", "float32_matmul_precision",
    }, "runtime")
    for key in ("torch_version", "numpy_version", "device"):
        if type(runtime[key]) is not str or not runtime[key]:
            raise ValueError(f"Invalid runtime {key}")
    _device_name(runtime["device"])
    if runtime["device"] == "auto":
        raise ValueError("Checkpoint device must be resolved")
    _integer(runtime["num_threads"], "runtime thread count", 1)
    _integer(runtime["num_interop_threads"], "runtime interop thread count", 1)
    if runtime["float32_matmul_precision"] not in ("highest", "high", "medium"):
        raise ValueError("Invalid runtime float32 matrix multiplication precision")
    _validate_loader_signature(runtime["train_loader"])
    _validate_loader_signature(runtime["validation_loader"])
    state = payload["model_state"]
    if type(state) is not dict or not state or any(type(key) is not str for key in state):
        raise ValueError("Invalid model state")
    for key, value in state.items():
        _tensor(value, f"model_state.{key}")
    optimizer = payload["optimizer_state"]
    _keys(optimizer, {"state", "param_groups"}, "optimizer")
    if type(optimizer["state"]) is not dict or type(optimizer["param_groups"]) is not list or not optimizer["param_groups"]:
        raise ValueError("Invalid optimizer state")
    _safe_tree(optimizer, "optimizer state")
    parameters = set()
    for group in optimizer["param_groups"]:
        if not isinstance(group, dict) or type(group.get("params")) is not list or not group["params"]:
            raise ValueError("Invalid optimizer parameter group")
        for parameter in group["params"]:
            _integer(parameter, "optimizer parameter index")
            if parameter in parameters:
                raise ValueError("Duplicate optimizer parameter index")
            parameters.add(parameter)
        if group.get("lr") != config.learning_rate or group.get("weight_decay") != config.weight_decay:
            raise ValueError("Optimizer hyperparameters differ from checkpoint config")
        allowed = set(_OPTIMIZER_OPTIONS) | {"params", "lr", "weight_decay", "decoupled_weight_decay"}
        if set(group) - allowed or not set(_OPTIMIZER_OPTIONS).issubset(group):
            raise ValueError("Invalid AdamW parameter group schema")
        for key, expected in _OPTIMIZER_OPTIONS.items():
            if not _json_equal(group[key], expected):
                raise ValueError(f"AdamW option {key} differs from this runtime")
        if "decoupled_weight_decay" in group and group["decoupled_weight_decay"] is not True:
            raise ValueError("AdamW must use decoupled weight decay")
    cursor = payload["cursor"]
    _keys(cursor, {"epoch", "next_batch_index", "global_step"}, "cursor")
    for key, value in cursor.items():
        _integer(value, f"cursor.{key}")
    _integer(payload["global_step"], "global_step")
    if payload["global_step"] != cursor["global_step"]:
        raise ValueError("Checkpoint global_step differs from its resume cursor")
    if cursor["epoch"] > config.epochs or cursor["next_batch_index"] >= runtime["train_loader"]["batches"]:
        raise ValueError("Checkpoint cursor exceeds its training bounds")
    if cursor["epoch"] == config.epochs and cursor["next_batch_index"] != 0:
        raise ValueError("Completed training cannot have a partial epoch")
    if config.max_steps is not None and cursor["global_step"] > config.max_steps:
        raise ValueError("Checkpoint exceeds its global-step ceiling")
    expected_steps = cursor["epoch"] * runtime["train_loader"]["batches"] + cursor["next_batch_index"]
    if cursor["global_step"] != expected_steps:
        raise ValueError("Checkpoint global step and batch cursor disagree")
    for parameter, state in optimizer["state"].items():
        if type(parameter) is not int or parameter not in parameters:
            raise ValueError("Unknown optimizer state parameter")
        _keys(state, {"step", "exp_avg", "exp_avg_sq"}, "AdamW parameter state")
        step = state["step"]
        if not isinstance(step, torch.Tensor) or step.numel() != 1:
            raise ValueError("Invalid AdamW step tensor")
        step = float(step.item())
        if not step.is_integer() or step <= 0 or step > cursor["global_step"]:
            raise ValueError("Invalid AdamW step cursor")
        first, second = state["exp_avg"], state["exp_avg_sq"]
        _tensor(first, "AdamW first moment")
        _tensor(second, "AdamW second moment")
        if first.shape != second.shape or first.dtype != second.dtype or torch.any(second < 0):
            raise ValueError("Invalid AdamW moment tensors")
    if payload["best_score"] is not None:
        _finite(payload["best_score"], "best score")
    _validate_rng(payload["rng"])
    _keys(payload["loader_state"], {"train", "validation", "epoch_start"}, "loader state")
    for key, value in payload["loader_state"].items():
        _tensor(value, f"loader RNG {key}", byte=True)
        torch.Generator().set_state(value)
    if type(payload["history"]) is not list:
        raise ValueError("Invalid checkpoint history")
    _json_copy(payload["history"])
    previous_step = 0
    for entry in payload["history"]:
        _keys(entry, {"epoch", "global_step", "epoch_complete", "validation"}, "history entry")
        _integer(entry["epoch"], "history epoch")
        step = _integer(entry["global_step"], "history global step", 1)
        if step <= previous_step or step > cursor["global_step"] or type(entry["epoch_complete"]) is not bool:
            raise ValueError("Invalid history cursor")
        if not isinstance(entry["validation"], dict):
            raise ValueError("Invalid validation history")
        _finite(entry["validation"].get("loss"), "validation loss")
        previous_step = step
    if cursor["global_step"] == 0:
        if payload["history"] or payload["best_score"] is not None or optimizer["state"]:
            raise ValueError("Untrained checkpoint cannot claim training history")
    elif (
        not payload["history"] or previous_step != cursor["global_step"]
        or payload["best_score"] is None or not optimizer["state"]
    ):
        raise ValueError("Trained checkpoint is missing its validation history")
    if payload["history"] and payload["best_score"] != min(entry["validation"]["loss"] for entry in payload["history"]):
        raise ValueError("Best score differs from the recorded validation history")
    return payload


def load_checkpoint(path, *, expected_identity=None):
    """Read only the versioned tensor/primitive format; never enable pickle."""
    with _regular_file(path).open("rb") as stream:
        payload = torch.load(stream, weights_only=True, map_location="cpu")
    _validate_checkpoint(payload)
    if expected_identity is not None and not _json_equal(payload["identity"], _identity(expected_identity)):
        raise ValueError("Checkpoint identity differs from the requested data/model/config identity")
    return payload


def _batch_to_device(batch, device):
    if not isinstance(batch, Mapping):
        raise ValueError("A batch must be a mapping")
    required = {"features", "conditioning", "lengths", "targets", "masks", "valid_frames"}
    if not required.issubset(batch):
        raise ValueError("Batch is missing required model inputs, targets, or masks")
    result = {}
    for key in ("features", "conditioning", "lengths", "valid_frames"):
        _tensor(batch[key], key)
        result[key] = batch[key].to(device)
    for key in ("targets", "masks"):
        if not isinstance(batch[key], Mapping):
            raise ValueError(f"Batch {key} must be a mapping")
        result[key] = {}
        for name, value in batch[key].items():
            _tensor(value, f"{key}.{name}", finite=key != "targets")
            result[key][name] = value.to(device)
    features, conditioning = result["features"], result["conditioning"]
    valid, lengths = result["valid_frames"], result["lengths"]
    if features.ndim != 3 or conditioning.ndim != 3 or features.shape[:2] != conditioning.shape[:2]:
        raise ValueError("Features and conditioning must have matching B,T dimensions")
    if valid.dtype != torch.bool or valid.shape != features.shape[:2]:
        raise ValueError("valid_frames must be a boolean B,T tensor")
    if lengths.ndim != 1 or lengths.shape[0] != features.shape[0] or lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("lengths must be an integer B tensor")
    if torch.any(lengths <= 0) or torch.any(lengths > features.shape[1]):
        raise ValueError("Batch lengths must be within its padded frame bounds")
    expected_valid = torch.arange(features.shape[1], device=device)[None, :] < lengths[:, None]
    if not torch.equal(valid, expected_valid):
        raise ValueError("valid_frames must exactly identify the frames before each length")
    if set(result["targets"]) != set(result["masks"]):
        raise ValueError("Target and mask task keys must match")
    for name, target in result["targets"].items():
        effective_mask = _mask(target, result["masks"][name], valid)
        _tensor(target[effective_mask], f"supervised targets.{name}")
    return result


def _outputs(model, batch):
    # Metadata and target voices never enter the forward call.
    outputs = model(batch["features"], batch["conditioning"], batch["lengths"])
    if not isinstance(outputs, dict):
        raise ValueError("Model forward must return an output dictionary")
    for name, value in outputs.items():
        _tensor(value, f"outputs.{name}")
    return outputs


def _stat_weight(name, weights):
    key = "note_onset" if name in _NOTE_ONSET_STATS else name
    if key not in weights:
        raise ValueError(f"Missing LOSS_WEIGHTS entry for loss statistic {name}")
    return _finite(weights[key], f"LOSS_WEIGHTS.{key}")


def _stats(stats, weights):
    if not isinstance(stats, Mapping) or not isinstance(weights, Mapping):
        raise ValueError("Loss statistics and weights must be mappings")
    _keys(stats, _LOSS_STAT_KEYS, "loss statistics")
    result = {}
    for name, item in stats.items():
        if name not in _PRIOR_NAMES:
            _stat_weight(name, weights)
        _keys(item, {"sum", "count"}, f"loss statistic {name}")
        count = _integer(item["count"], f"{name}.count")
        numerator = _finite(item["sum"], f"{name}.sum")
        if count == 0 and numerator != 0:
            raise ValueError("An empty loss statistic must have a zero numerator")
        result[name] = {"sum": numerator, "count": count}
    return result


def _objective(stats, weights, sparsity_weight):
    supervised = sum(
        value["count"] for name, value in stats.items()
        if name not in _PRIOR_NAMES and _stat_weight(name, weights) > 0
    )
    if supervised == 0:
        return None
    onset_means = [
        stats[name]["sum"] / stats[name]["count"]
        for name in _NOTE_ONSET_STATS if name in stats and stats[name]["count"]
    ]
    loss = weights["note_onset"] * sum(onset_means) / len(onset_means) if onset_means else 0.0
    for name, value in stats.items():
        if value["count"] and name not in _NOTE_ONSET_STATS:
            weight = sparsity_weight if name in _PRIOR_NAMES else _stat_weight(name, weights)
            loss += weight * value["sum"] / value["count"]
    return _finite(loss, "aggregated loss")


def _loss(outputs, batch, loss_function, weights, sparsity_weight):
    loss, stats = loss_function(
        outputs, batch["targets"], batch["masks"], batch["valid_frames"],
        sparsity_weight=sparsity_weight,
    )
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss).item():
        raise ValueError("Loss must be a finite scalar tensor")
    stats = _stats(stats, weights)
    expected = _objective(stats, weights, sparsity_weight)
    if expected is not None and not math.isclose(loss.detach().item(), expected, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("Loss tensor and reported weighted statistics disagree")
    return loss, stats


def _mask(target, mask, valid):
    if mask.dtype != torch.bool or mask.shape != target.shape or target.shape[:2] != valid.shape:
        raise ValueError("Target masks must be boolean tensors matching their targets")
    return mask & valid.reshape(*valid.shape, *((1,) * (target.ndim - 2)))


def _prediction(outputs, name):
    key = f"{name}_logits"
    if key in outputs:
        return outputs[key]
    if name in outputs:
        return outputs[name]
    raise ValueError(f"Missing model output for labelled task {name}")


def _metric_counts(outputs, batch, counts):
    targets, masks, valid = batch["targets"], batch["masks"], batch["valid_frames"]
    for name in ("note_onset", "fret", "pitch", "voice", "duration", "harmonic_presence", "percussion"):
        target_name = name
        if name == "duration" and ("duration_log" in targets or "duration_log" in masks):
            target_name = "duration_log"
        elif name == "harmonic_presence" and ("harmonic" in targets or "harmonic" in masks):
            target_name = "harmonic"
        if target_name not in targets and target_name not in masks:
            continue
        if target_name not in targets or target_name not in masks:
            raise ValueError(f"Task {target_name} is missing targets or masks")
        target = targets[target_name]
        mask = _mask(target, masks[target_name], valid)
        prediction = _prediction(outputs, target_name)
        count = counts[name]
        if name in ("fret", "pitch", "voice"):
            if prediction.shape[:-1] != target.shape:
                raise ValueError(f"Invalid categorical output shape for {name}")
            count["correct"] += int(((prediction.argmax(-1) == target) & mask).sum().item())
            count["count"] += int(mask.sum().item())
        elif name == "duration":
            if prediction.shape != target.shape:
                raise ValueError("Invalid duration output shape")
            selected_prediction, selected_target = prediction[mask].double(), target[mask].double()
            if target_name == "duration_log":
                selected_prediction, selected_target = selected_prediction.expm1(), selected_target.expm1()
            error = (selected_prediction - selected_target).abs().sum().item()
            count["sum"] += _finite(error, "duration error in quarter-note units")
            count["count"] += int(mask.sum().item())
        else:
            if prediction.shape != target.shape:
                raise ValueError(f"Invalid binary output shape for {name}")
            emitted = prediction >= 0
            positive = target > 0.5
            if name == "note_onset":
                count["count"] += int(mask.sum().item())
                count["tp"] += int((emitted & positive & mask).sum().item())
                count["fp"] += int((emitted & ~positive & mask).sum().item())
                count["fn"] += int((~emitted & positive & mask).sum().item())
            else:
                labelled = positive & mask
                emission_mask = valid.reshape(*valid.shape, *((1,) * (target.ndim - 2))).expand_as(target)
                count["count"] += int(labelled.sum().item())
                count["tp"] += int((emitted & labelled).sum().item())
                count["emitted"] += int((emitted & emission_mask).sum().item())
                count["positions"] += int(emission_mask.sum().item())


def _divide(numerator, denominator):
    return numerator / denominator if denominator else None


def _metrics(counts):
    note = counts["note_onset"]
    tp, fp, fn = note["tp"], note["fp"], note["fn"]
    result = {
        "note_onset_frame": {
            "available": bool(note["count"]), "count": note["count"],
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": _divide(tp, tp + fp), "recall": _divide(tp, tp + fn),
            "f1": _divide(2 * tp, 2 * tp + fp + fn),
        },
    }
    for name in ("fret", "pitch", "voice"):
        value = counts[name]
        result[f"{name}_accuracy"] = {
            "available": bool(value["count"]), "count": value["count"],
            "correct": value["correct"], "accuracy": _divide(value["correct"], value["count"]),
        }
    duration = counts["duration"]
    result["duration_quarter_mae"] = {
        "available": bool(duration["count"]), "count": duration["count"],
        "absolute_error_sum": duration["sum"], "mae": _divide(duration["sum"], duration["count"]),
    }
    for name in ("harmonic_presence", "percussion"):
        value = counts[name]
        result[f"{name}_positive"] = {
            "available": bool(value["count"]), "labelled_positive_count": value["count"],
            "recalled_positive_count": value["tp"], "recall_at_labelled_positives": _divide(value["tp"], value["count"]),
            "emitted_positive_count": value["emitted"], "emission_position_count": value["positions"],
            "emitted_positive_rate": _divide(value["emitted"], value["positions"]),
        }
    return result


def evaluate_model(model, loader, device, *, sparsity_weight=0.02):
    """Frame-level, mask-aware metrics; not event or complete-score accuracy."""
    _finite(sparsity_weight, "sparsity_weight")
    device = resolve_device(str(device))
    loss_function, weights = _loss_api()
    modes = [(module, module.training) for module in model.modules()]
    rng = _capture_rng()
    generator = getattr(loader, "generator", None)
    generator_state = generator.get_state() if isinstance(generator, torch.Generator) else None
    totals = {name: {"sum": 0.0, "count": 0} for name in _LOSS_STAT_KEYS}
    counts = {
        name: {"count": 0, "sum": 0.0, "correct": 0, "tp": 0, "fp": 0, "fn": 0, "emitted": 0, "positions": 0}
        for name in ("note_onset", "fret", "pitch", "voice", "duration", "harmonic_presence", "percussion")
    }
    batches = frames = 0
    try:
        model.eval()
        with torch.no_grad():
            for source_batch in loader:
                batch = _batch_to_device(source_batch, device)
                outputs = _outputs(model, batch)
                _, stats = _loss(outputs, batch, loss_function, weights, sparsity_weight)
                for name, value in stats.items():
                    total = totals[name]
                    total["sum"] += value["sum"]
                    total["count"] += value["count"]
                _metric_counts(outputs, batch, counts)
                batches += 1
                frames += int(batch["valid_frames"].sum().item())
    finally:
        for module, mode in modes:
            module.training = mode
        _restore_rng(rng)
        if generator_state is not None:
            generator.set_state(generator_state)
    totals = _stats(totals, weights)
    loss = _objective(totals, weights, sparsity_weight)
    return _json_copy({
        "loss": loss, "available": loss is not None, "batches": batches, "valid_frames": frames,
        "loss_statistics": totals, **_metrics(counts),
    })


def _fixed_config(config):
    return {key: value for key, value in config.items() if key not in ("epochs", "max_steps")}


def _require_same_rng(before, after):
    if not (
        before["python"] == after["python"] and before["numpy"] == after["numpy"]
        and torch.equal(before["torch_cpu"], after["torch_cpu"])
        and len(before["torch_cuda"]) == len(after["torch_cuda"])
        and all(torch.equal(left, right) for left, right in zip(before["torch_cuda"], after["torch_cuda"]))
    ):
        raise ValueError("Data loading consumed global RNG; exact training resume is unsupported")


def _checked_next(iterator):
    # Deterministic features/samplers must not advance the model's global RNG.
    before = _capture_rng()
    try:
        batch = next(iterator)
    except StopIteration as error:
        raise ValueError("Training loader ended before its declared batch count") from error
    _require_same_rng(before, _capture_rng())
    return batch


def _iterator(loader):
    before = _capture_rng()
    iterator = iter(loader)
    _require_same_rng(before, _capture_rng())
    return iterator


def _require_validation(metrics):
    if not metrics["batches"] or not metrics["valid_frames"] or metrics["loss"] is None:
        raise ValueError("Validation needs data and a meaningful supervised objective")


def run_training(model, train_loader, validation_loader, config, run_dir, identity, *, resume=None):
    """Perform bounded AdamW updates only when explicitly called by the owner.

    Epochs and max_steps are total ceilings. Exact resume requires unchanged,
    deterministic map-style loaders with private generators and zero workers.
    The caller must seed model construction separately for reproducible new runs.
    """
    if not isinstance(config, TrainingConfig):
        raise ValueError("config must be a TrainingConfig")
    identity = _identity(identity)
    model_config = getattr(model, "config", None)
    if model_config is not None and is_dataclass(model_config) and not _json_equal(asdict(model_config), identity["model"]):
        raise ValueError("Model configuration differs from identity.model")
    device = resolve_device(config.device)
    runtime = {
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
        "device": str(device), "train_loader": _loader_signature(train_loader, training=True),
        "validation_loader": _loader_signature(validation_loader, training=False),
        "num_threads": torch.get_num_threads(), "num_interop_threads": torch.get_num_interop_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if train_loader.generator is validation_loader.generator:
        raise ValueError("Train and validation loaders must have independent generators")
    run_dir = _plain_path(run_dir)
    if run_dir == run_dir.parent:
        raise ValueError("A filesystem root cannot be a training run directory")
    checkpoint = None
    source_stamps = {}
    if resume is None:
        if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
            raise ValueError("A fresh run directory must be absent or empty")
        run_id = uuid.uuid4().hex
    else:
        if _plain_path(resume) != run_dir / "latest.pt":
            raise ValueError("Resume must use latest.pt from this exact run directory")
        source_stamps = {
            name: _file_stamp(run_dir / name) for name in ("run.json", "latest.pt", "best.pt")
        }
        checkpoint = load_checkpoint(resume, expected_identity=identity)
        run_id = checkpoint["run_id"]
        manifest = _read_json(run_dir / "run.json")
        _keys(manifest, {"schema_version", "run_id", "identity", "config"}, "run manifest")
        _keys(manifest["config"], asdict(TrainingConfig()), "run training config")
        TrainingConfig(**manifest["config"])
        if (
            manifest["schema_version"] != SCHEMA_VERSION or manifest["run_id"] != run_id
            or not _json_equal(manifest["identity"], identity)
            or not _json_equal(_fixed_config(manifest["config"]), _fixed_config(asdict(config)))
            or not _json_equal(_fixed_config(checkpoint["training_config"]), _fixed_config(asdict(config)))
        ):
            raise ValueError("Existing run configuration or identity differs from the requested resume")
        if not _json_equal(checkpoint["runtime"], runtime):
            raise ValueError("Runtime, device, or loader configuration changed; exact resume is unsupported")
        cursor = checkpoint["cursor"]
        if config.epochs < cursor["epoch"] or (config.epochs == cursor["epoch"] and cursor["next_batch_index"]):
            raise ValueError("epochs is below the checkpoint cursor")
        if config.max_steps is not None and config.max_steps < cursor["global_step"]:
            raise ValueError("max_steps is below the checkpoint global step")
        for filename in ("run.json", "metrics.json", "latest.pt", "best.pt"):
            if (run_dir / filename).exists():
                _regular_file(run_dir / filename)
        best = load_checkpoint(run_dir / "best.pt", expected_identity=identity)
        if (
            best["run_id"] != run_id or best["cursor"]["global_step"] > cursor["global_step"]
            or not best["history"] or best["history"][-1]["validation"]["loss"] != checkpoint["best_score"]
        ):
            raise ValueError("best.pt and latest.pt do not describe a consistent run")
        if any(_file_stamp(run_dir / name) != stamp for name, stamp in source_stamps.items()):
            raise ValueError("Run changed while its checkpoint was being loaded")
    loss_function, weights = _loss_api()
    modes = [(module, module.training) for module in model.modules()]
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    lock = None
    lock_path = run_dir / ".training.lock"
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
            **_OPTIMIZER_OPTIONS,
        )
        if checkpoint is None:
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)
            epoch = next_batch = global_step = 0
            best_score = None
            history = []
            epoch_start = train_loader.generator.get_state()
        else:
            model.load_state_dict(checkpoint["model_state"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            for parameter, state in optimizer.state.items():
                if state["exp_avg"].shape != parameter.shape or state["exp_avg_sq"].shape != parameter.shape:
                    raise ValueError("Optimizer moment shapes differ from the model parameters")
            epoch, next_batch, global_step = (checkpoint["cursor"][key] for key in ("epoch", "next_batch_index", "global_step"))
            best_score, history = checkpoint["best_score"], checkpoint["history"]
            train_loader.generator.set_state(checkpoint["loader_state"]["train"])
            validation_loader.generator.set_state(checkpoint["loader_state"]["validation"])
            epoch_start = checkpoint["loader_state"]["epoch_start"]
            _restore_rng(checkpoint["rng"])
        validation = evaluate_model(model, validation_loader, device, sparsity_weight=config.sparsity_weight)
        _require_validation(validation)
        if checkpoint is None:
            if run_dir.exists():
                if not run_dir.is_dir() or any(run_dir.iterdir()):
                    raise ValueError("Fresh run directory became nonempty before publication")
            else:
                run_dir.mkdir(parents=True, exist_ok=False)
        lock = lock_path.open("xb")
        if any(_file_stamp(run_dir / name) != stamp for name, stamp in source_stamps.items()):
            raise ValueError("Run changed before the training lock was acquired; reload latest.pt")
        _write_json(run_dir / "run.json", {
            "schema_version": SCHEMA_VERSION, "run_id": run_id, "identity": identity, "config": asdict(config),
        }, replace=checkpoint is not None)
        while epoch < config.epochs and (config.max_steps is None or global_step < config.max_steps):
            if callable(getattr(train_loader.sampler, "set_epoch", None)):
                before = _capture_rng()
                train_loader.sampler.set_epoch(epoch)
                _require_same_rng(before, _capture_rng())
            resuming_partial = next_batch > 0
            saved_generator = train_loader.generator.get_state()
            if resuming_partial:
                train_loader.generator.set_state(epoch_start)
            else:
                epoch_start = train_loader.generator.get_state()
            iterator = _iterator(train_loader)
            for _ in range(next_batch):
                _checked_next(iterator)
            if resuming_partial and not torch.equal(train_loader.generator.get_state(), saved_generator):
                raise ValueError("Cannot reproduce the checkpoint's mid-epoch loader RNG state")
            model.train()
            epoch_index = epoch
            while next_batch < runtime["train_loader"]["batches"]:
                batch = _batch_to_device(_checked_next(iterator), device)
                optimizer.zero_grad(set_to_none=True)
                outputs = _outputs(model, batch)
                loss, stats = _loss(outputs, batch, loss_function, weights, config.sparsity_weight)
                if _objective(stats, weights, config.sparsity_weight) is None:
                    raise ValueError("Training batch has no meaningful supervised objective")
                loss.backward()
                gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                if not gradients:
                    raise ValueError("Training loss produced no model gradients")
                if any(not torch.isfinite(gradient).all().item() for gradient in gradients):
                    raise ValueError("Nonfinite model gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                for name, value in model.state_dict().items():
                    _tensor(value, f"updated model.{name}")
                global_step += 1
                next_batch += 1
                if config.max_steps is not None and global_step >= config.max_steps:
                    break
            epoch_complete = next_batch == runtime["train_loader"]["batches"]
            if epoch_complete:
                before = _capture_rng()
                if next(iterator, None) is not None:
                    raise ValueError("Training loader exceeded its declared batch count")
                _require_same_rng(before, _capture_rng())
                epoch += 1
                next_batch = 0
            validation = evaluate_model(model, validation_loader, device, sparsity_weight=config.sparsity_weight)
            _require_validation(validation)
            history.append({
                "epoch": epoch_index, "global_step": global_step,
                "epoch_complete": epoch_complete, "validation": validation,
            })
            improved = best_score is None or validation["loss"] < best_score
            if improved:
                best_score = validation["loss"]
            payload = {
                "schema_version": SCHEMA_VERSION, "run_id": run_id, "identity": identity,
                "training_config": asdict(config), "runtime": runtime, "global_step": global_step,
                "model_state": dict(model.state_dict()), "optimizer_state": optimizer.state_dict(),
                "cursor": {"epoch": epoch, "next_batch_index": next_batch, "global_step": global_step},
                "best_score": best_score, "rng": _capture_rng(),
                "loader_state": {
                    "train": train_loader.generator.get_state(),
                    "validation": validation_loader.generator.get_state(), "epoch_start": epoch_start,
                },
                "history": history,
            }
            _validate_checkpoint(payload)
            if improved:
                _atomic_write(run_dir / "best.pt", lambda stream: torch.save(payload, stream))
            _atomic_write(run_dir / "latest.pt", lambda stream: torch.save(payload, stream))
            _write_json(run_dir / "metrics.json", history)
        return _json_copy({
            "run_dir": str(run_dir), "global_step": global_step, "epoch": epoch,
            "next_batch_index": next_batch, "best_score": best_score, "validation": validation,
            "stopped_by": "epochs" if epoch >= config.epochs else "max_steps",
            "latest_checkpoint": str(run_dir / "latest.pt"), "best_checkpoint": str(run_dir / "best.pt"),
        })
    finally:
        if lock is not None:
            lock.close()
            _regular_file(lock_path).unlink()
        for module, mode in modes:
            module.training = mode
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
