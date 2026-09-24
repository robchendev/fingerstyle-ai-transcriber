"""Explicitly invoked training, masked evaluation, and safe local checkpoints."""

from __future__ import annotations

import json
import math
import os
import random
import re
import stat
import time
import uuid
from collections.abc import Mapping, Sized
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path

import numpy as np
import torch

from .connection_supervision import CONNECTION_TYPES, V4_NOTE_TECHNIQUE_TYPES, vocabularies
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler


SCHEMA_VERSION = 3
INFERENCE_FORMAT = "fingerstyle-inference"
INFERENCE_SCHEMA_VERSION = 1
_PRIOR_NAMES = frozenset({"harmonic_sparsity", "percussion_sparsity"})
_NOTE_ONSET_STATS = ("note_onset_positive", "note_onset_negative")
_PERCUSSION_STATS = ("percussion_positive", "percussion_negative")
_TECHNIQUE_STATS = ("technique_positive", "technique_negative")
_TECHNIQUE_STRING_STATS = ("technique_strings_positive", "technique_strings_negative")
_CONNECTION_STATS = ("connection_positive", "connection_negative")
_NOTE_TECHNIQUE_STATS = ("note_technique_positive", "note_technique_negative")
_GRACE_STATS = ("grace_positive", "grace_negative")
_CALIBRATION_THRESHOLDS = (.25, .5, .75, .9)
_CALIBRATION_TASKS = (
    *(f"connection:{name}" for name in CONNECTION_TYPES[1:]),
    *(f"note_technique:{name}" for name in V4_NOTE_TECHNIQUE_TYPES),
    "grace",
)
_LOSS_STAT_KEYS = (
    *_NOTE_ONSET_STATS, "fret", "pitch", "voice", "duration_log", "harmonic_positive",
    "harmonic_kind", "harmonic_node", *_PERCUSSION_STATS, "harmonic_sparsity", "percussion_sparsity",
    *_TECHNIQUE_STATS, "technique_direction", *_TECHNIQUE_STRING_STATS,
    *_CONNECTION_STATS, *_NOTE_TECHNIQUE_STATS,
    "bend_curve",
    *_GRACE_STATS, "grace_fret", "grace_mode", "grace_transition",
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
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    seed: int = 17
    device: str = "auto"
    max_steps: int | None = None
    sparsity_weight: float = 0.02
    max_seconds: float | None = None
    event_patience: int = 12
    learning_rate_patience: int = 4
    learning_rate_factor: float = 0.5
    minimum_learning_rate: float = 0.00001

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
        if self.max_seconds is not None:
            _finite(self.max_seconds, "max_seconds", positive=True)
        _integer(self.event_patience, "event_patience", 1)
        _integer(self.learning_rate_patience, "learning_rate_patience", 1)
        if self.learning_rate_patience >= self.event_patience:
            raise ValueError("learning_rate_patience must be less than event_patience")
        _finite(self.learning_rate_factor, "learning_rate_factor", positive=True)
        if self.learning_rate_factor >= 1:
            raise ValueError("learning_rate_factor must be below one")
        _finite(self.minimum_learning_rate, "minimum_learning_rate", positive=True)
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError("minimum_learning_rate cannot exceed learning_rate")


class TrainingBudgetExpired(Exception):
    """Cooperative stop at a safe boundary, never an asynchronous interruption."""


class TrainingBudget:
    """Per-invocation monotonic budget, including caller setup and checkpoint I/O.

    Reserve up to two minutes (10% of short budgets) for checkpoint publication.
    Native operations are not interrupted; overruns are reported, not concealed.
    """

    def __init__(self, max_seconds, *, started_at=None, deadline=None, reserve_seconds=120.):
        self.started_at = time.perf_counter() if started_at is None else _finite(started_at, "budget start")
        self.max_seconds = None if max_seconds is None else _finite(max_seconds, "max_seconds", positive=True)
        self.reserve_seconds = min(_finite(reserve_seconds, "checkpoint reserve"), self.max_seconds * .1) if self.max_seconds is not None else 0.
        self.deadline = self.started_at + self.max_seconds if self.max_seconds is not None else None
        if deadline is not None:
            deadline = _finite(deadline, "budget deadline")
            if self.deadline is None or not math.isclose(deadline, self.deadline, abs_tol=1e-6, rel_tol=0):
                raise ValueError("Budget deadline must equal invocation start plus max_seconds")
            self.deadline = deadline

    def check(self):
        if self.deadline is not None and time.perf_counter() >= self.deadline - self.reserve_seconds:
            raise TrainingBudgetExpired()

    def report(self):
        elapsed = time.perf_counter() - self.started_at
        return {
            "max_seconds": self.max_seconds, "elapsed_seconds": elapsed,
            "checkpoint_reserve_seconds": self.reserve_seconds,
            "budget_overrun_seconds": max(0., elapsed - self.max_seconds) if self.max_seconds is not None else 0.,
            "budget_policy": "per-invocation cooperative wall clock including setup, validation and checkpoint I/O; native operations can overrun",
        }


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


def _joint_video_config(identity):
    from .transcriber_video import VideoConfig
    from .fretboard_features import SCHEMA_VERSION as VIDEO_SCHEMA_VERSION, STRUCTURED_DIM
    from .video_features import SCHEMA_VERSION as LEGACY_VIDEO_SCHEMA_VERSION, STRUCTURED_DIM as LEGACY_STRUCTURED_DIM

    video = identity.get("video")
    if not isinstance(video, dict) or not isinstance(video.get("config"), dict):
        raise ValueError("Invalid joint checkpoint video configuration")
    values = video["config"]
    architecture = values.get("architecture_version")
    contract = (
        (VIDEO_SCHEMA_VERSION, STRUCTURED_DIM, "anatomy-fretboard-groups-v2")
        if architecture == 6
        else (LEGACY_VIDEO_SCHEMA_VERSION, LEGACY_STRUCTURED_DIM, "anatomy-representation-groups-v1")
        if architecture == 5
        else None
    )
    if "image_size" in values or "freeze_audio" in values or contract is None:
        raise ValueError("Historical, frozen, RGB, or unversioned paired checkpoints are unsupported")
    if (
        values.get("input_schema_version") != contract[0]
        or values.get("structured_dim") != contract[1]
        or values.get("feature_group_version") != contract[2]
    ):
        raise ValueError("Mismatched paired checkpoints are unsupported; architecture and numeric input contract disagree")
    _keys(values, asdict(VideoConfig()), "joint video model configuration")
    initialization = identity.get("initialization")
    if initialization is not None and initialization != {
        "kind": "joint-audio-numeric-video-from-scratch",
        "audioParametersFrozen": False, "optimizerStateImported": False,
    }:
        raise ValueError("Joint checkpoints cannot use staged or imported acoustic initialization")
    return VideoConfig(**values)


def _validate_checkpoint(payload):
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] not in (1, 2, SCHEMA_VERSION):
        raise ValueError("Unsupported checkpoint schema version")
    version = payload["schema_version"]
    _keys(payload, _CHECKPOINT_KEYS | ({"resume_state"} if version >= 2 else set()), "checkpoint")
    if not isinstance(payload["run_id"], str) or not re.fullmatch("[0-9a-f]{32}", payload["run_id"]):
        raise ValueError("Invalid run identity")
    _identity(payload["identity"])
    if "video" in payload["identity"]:
        _joint_video_config(payload["identity"])
    config_keys = set(asdict(TrainingConfig()))
    if version == 1:
        config_keys.discard("max_seconds")
    if version < 3:
        config_keys -= {"event_patience", "learning_rate_patience", "learning_rate_factor", "minimum_learning_rate"}
    _keys(payload["training_config"], config_keys, "training config")
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
        lr = group.get("lr")
        if (
            type(lr) not in (int, float) or not math.isfinite(lr)
            or not config.minimum_learning_rate <= lr <= config.learning_rate
            or group.get("weight_decay") != config.weight_decay
        ):
            raise ValueError("Optimizer hyperparameters differ from checkpoint config")
        allowed = set(_OPTIMIZER_OPTIONS) | {"params", "lr", "weight_decay", "decoupled_weight_decay"}
        if set(group) - allowed or not set(_OPTIMIZER_OPTIONS).issubset(group):
            raise ValueError("Invalid AdamW parameter group schema")
        for key, expected in _OPTIMIZER_OPTIONS.items():
            if not _json_equal(group[key], expected):
                raise ValueError(f"AdamW option {key} differs from this runtime")
        if "decoupled_weight_decay" in group and group["decoupled_weight_decay"] is not True:
            raise ValueError("AdamW must use decoupled weight decay")
    if "video" in payload["identity"]:
        from .transcriber_model import FingerstyleTranscriber, ModelConfig
        from .transcriber_video import AudioVideoTranscriber

        # Validate every joint parameter without allocating weights or consuming RNG.
        with torch.device("meta"):
            expected = AudioVideoTranscriber(
                FingerstyleTranscriber(ModelConfig(**payload["identity"]["model"])),
                _joint_video_config(payload["identity"]),
            )
        parameter_shapes = [parameter.shape for parameter in expected.parameters()]
        if len(optimizer["param_groups"]) != 1 or optimizer["param_groups"][0]["params"] != list(range(len(parameter_shapes))):
            raise ValueError("Joint optimizer must contain all acoustic, numeric-video and fusion parameters in model order")
        if set(state) != set(expected.state_dict()) or any(state[name].shape != value.shape for name, value in expected.state_dict().items()):
            raise ValueError("Joint checkpoint model tensors differ from its declared architecture")
        for index, moments in optimizer["state"].items():
            if type(index) is not int or not 0 <= index < len(parameter_shapes):
                raise ValueError("Unknown joint optimizer parameter")
            if not isinstance(moments, dict) or any(not isinstance(moments.get(name), torch.Tensor) or moments[name].shape != parameter_shapes[index] for name in ("exp_avg", "exp_avg_sq")):
                raise ValueError("Joint optimizer moment shapes differ from the model parameters")
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
    previous_updates = previous_skips = 0
    optimizer_counts = None
    for entry in payload["history"]:
        fields = {"epoch", "global_step", "epoch_complete", "validation"}
        if "optimizer_updates" in entry or "optimizer_skipped_batches" in entry:
            fields |= {"optimizer_updates", "optimizer_skipped_batches"}
        _keys(entry, fields, "history entry")
        _integer(entry["epoch"], "history epoch")
        step = _integer(entry["global_step"], "history global step", 1)
        if step <= previous_step or step > cursor["global_step"] or type(entry["epoch_complete"]) is not bool:
            raise ValueError("Invalid history cursor")
        if not isinstance(entry["validation"], dict):
            raise ValueError("Invalid validation history")
        _finite(entry["validation"].get("loss"), "validation loss")
        if "optimizer_updates" in entry:
            if not isinstance(payload["identity"].get("video"), dict):
                raise ValueError("Optimizer skip accounting requires a joint audio/video identity")
            updates = _integer(entry["optimizer_updates"], "optimizer updates")
            skips = _integer(entry["optimizer_skipped_batches"], "optimizer skipped batches")
            if updates < previous_updates or skips < previous_skips or updates + skips != step:
                raise ValueError("Optimizer activity and processed batch cursor disagree")
            previous_updates, previous_skips = optimizer_counts = updates, skips
        elif optimizer_counts is not None:
            raise ValueError("Checkpoint history lost optimizer activity counts")
        elif "video" in payload["identity"]:
            raise ValueError("Joint checkpoint history requires actual optimizer update and skipped-batch counts")
        previous_step = step
    if version >= 2:
        resume_state = payload["resume_state"]
        _keys(resume_state, {"phase", "validation_pending", "event_evaluation_required", "optimizer_updates", "optimizer_skipped_batches", "stopped_by"}, "resume state")
        phase = resume_state["phase"]
        if phase not in ("initial_validation", "training", "validation"):
            raise ValueError("Invalid resume phase")
        if resume_state["stopped_by"] not in (None, "max_seconds"):
            raise ValueError("Invalid resume stop reason")
        if type(resume_state["event_evaluation_required"]) is not bool:
            raise ValueError("Invalid resume event-evaluation policy")
        if type(resume_state["validation_pending"]) is not bool or resume_state["validation_pending"] != (phase == "initial_validation" or previous_step < cursor["global_step"]):
            raise ValueError("Resume validation status differs from its history")
        if (phase == "initial_validation" and cursor["global_step"]) or (phase == "validation" and not resume_state["validation_pending"]):
            raise ValueError("Resume phase differs from its validation cursor")
        if resume_state["validation_pending"] and resume_state["stopped_by"] != "max_seconds":
            raise ValueError("Incomplete validation requires an explicit budget pause")
        updates = _integer(resume_state["optimizer_updates"], "resume optimizer updates")
        skips = _integer(resume_state["optimizer_skipped_batches"], "resume optimizer skipped batches")
        if updates + skips != cursor["global_step"] or updates < previous_updates or skips < previous_skips:
            raise ValueError("Resume optimizer counts differ from the cursor/history")
        if previous_step == cursor["global_step"] and optimizer_counts is not None and optimizer_counts != (updates, skips):
            raise ValueError("Resume optimizer counts differ from validated counts")
        optimizer_counts = updates, skips
    if optimizer_counts is not None:
        if bool(optimizer["state"]) != bool(optimizer_counts[0]):
            raise ValueError("Optimizer state and actual update count disagree")
        if any(float(state["step"].item()) > optimizer_counts[0] for state in optimizer["state"].values()):
            raise ValueError("AdamW step exceeds actual optimizer updates")
    if cursor["global_step"] == 0:
        if payload["history"] or payload["best_score"] is not None or optimizer["state"]:
            raise ValueError("Untrained checkpoint cannot claim training history")
    elif version == 1 and (
        not payload["history"] or previous_step != cursor["global_step"]
        or payload["best_score"] is None or (not optimizer["state"] and optimizer_counts is None)
    ):
        raise ValueError("Trained checkpoint is missing its validation history")
    expected_best = min((entry["validation"]["loss"] for entry in payload["history"]), default=None)
    if payload["best_score"] != expected_best:
        raise ValueError("Best score differs from the recorded validation history")
    return payload


def checkpoint_identity(payload):
    """Return model inputs without inventing training identity for an export."""
    if payload.get("format") != INFERENCE_FORMAT:
        return payload["identity"]
    identity = {"model": payload["model_config"], "features": payload["feature_config"]}
    if payload["video_config"] is not None:
        identity["video"] = {"config": payload["video_config"]}
    return identity


def _validate_inference_checkpoint(payload):
    from .transcriber_audio import FeatureConfig
    from .transcriber_model import FingerstyleTranscriber, ModelConfig
    from .transcriber_video import AudioVideoTranscriber, VideoConfig

    _keys(payload, {"format", "schema_version", "model_config", "video_config", "feature_config", "model_state"}, "inference checkpoint")
    if payload["format"] != INFERENCE_FORMAT or type(payload["schema_version"]) is not int or payload["schema_version"] != INFERENCE_SCHEMA_VERSION:
        raise ValueError("Unsupported inference checkpoint format or version")
    configs = {}
    for key, cls in (("model_config", ModelConfig), ("feature_config", FeatureConfig), ("video_config", VideoConfig)):
        values = payload[key]
        if key == "video_config" and values is None:
            continue
        if key == "video_config" and values.get("architecture_version") == 4 and "feature_group_version" not in values:
            values["feature_group_version"] = None
        if key == "video_config" and values.get("architecture_version") == 5 and "feature_group_version" not in values:
            values["feature_group_version"] = "anatomy-representation-groups-v1"
        if key == "video_config" and "experiment_mode" not in values:
            values["experiment_mode"] = "geometry"
        _keys(values, {field.name for field in fields(cls)}, key)
        try:
            configs[key] = cls(**values)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid {key}: {error}") from error
    if configs["model_config"].n_mels != configs["feature_config"].n_mels:
        raise ValueError("Inference feature and model dimensions disagree")
    # Shape validation must neither allocate another model nor consume RNG state.
    with torch.device("meta"):
        model = FingerstyleTranscriber(configs["model_config"])
        if "video_config" in configs:
            model = AudioVideoTranscriber(model, configs["video_config"])
    expected = model.state_dict()
    state = payload["model_state"]
    _keys(state, expected, "inference model state")
    for name, value in state.items():
        if type(value) is not torch.Tensor or value.requires_grad:
            raise ValueError(f"Inference parameter {name} must be a detached tensor")
        _tensor(value, f"model_state.{name}")
        if value.device.type != "cpu" or value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f"Inference parameter {name} has incompatible shape, dtype or device")
    return payload




def load_checkpoint(path, *, expected_identity=None, allow_inference=False):
    """Read only the versioned tensor/primitive format; never enable pickle."""
    with _regular_file(path).open("rb") as stream:
        payload = torch.load(stream, weights_only=True, map_location="cpu")
    if isinstance(payload, dict) and "format" in payload:
        _validate_inference_checkpoint(payload)
        if not allow_inference or expected_identity is not None:
            raise ValueError("Inference-only checkpoints cannot resume training or provide training identity")
        return payload
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
    if "video" in batch:
        if not isinstance(batch["video"], Mapping):
            raise ValueError("Batch video must be a mapping")
        result["video"] = {}
        for name, value in batch["video"].items():
            _tensor(value, f"video.{name}")
            result["video"][name] = value.to(device)
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
    optional = {"video": batch["video"]} if "video" in batch else {}
    outputs = model(batch["features"], batch["conditioning"], batch["lengths"], **optional)
    if not isinstance(outputs, dict):
        raise ValueError("Model forward must return an output dictionary")
    for name, value in outputs.items():
        _tensor(value, f"outputs.{name}")
    return outputs


def _stat_weight(name, weights):
    # Percussion weights scale both observed counts and BCE sums when pooled.
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
        if name == "percussion_positive":
            percussion = [key for key in _PERCUSSION_STATS if stats[key]["count"]]
            if len(percussion) == 1:
                observed = stats[percussion[0]]
                loss += observed["sum"] / observed["count"]
            elif percussion:
                loss += sum(stats[key]["sum"] * _stat_weight(key, weights) for key in percussion) / sum(
                    stats[key]["count"] * _stat_weight(key, weights) for key in percussion
                )
        elif name in ("technique_positive", "technique_strings_positive"):
            group = (
                _TECHNIQUE_STATS if name == "technique_positive"
                else _TECHNIQUE_STRING_STATS
            )
            observed = [key for key in group if stats[key]["count"]]
            if observed:
                loss += sum(stats[key]["sum"] * _stat_weight(key, weights) for key in observed) / sum(
                    stats[key]["count"] * _stat_weight(key, weights) for key in observed
                )
        elif name in ("connection_positive", "note_technique_positive", "grace_positive"):
            group = {"connection_positive": _CONNECTION_STATS, "note_technique_positive": _NOTE_TECHNIQUE_STATS, "grace_positive": _GRACE_STATS}[name]
            observed = [key for key in group if stats[key]["count"]]
            if observed:
                loss += sum(
                    stats[key]["sum"] / stats[key]["count"] * _stat_weight(key, weights)
                    for key in observed
                ) / sum(_stat_weight(key, weights) for key in observed)
        elif value["count"] and name not in (*_NOTE_ONSET_STATS, *_PERCUSSION_STATS, *_TECHNIQUE_STATS, *_TECHNIQUE_STRING_STATS, *_CONNECTION_STATS, *_NOTE_TECHNIQUE_STATS, *_GRACE_STATS):
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


def _binary_counts(count, emitted, positive, mask):
    count["count"] += int(mask.sum())
    for name, selected in (
        ("tp", emitted & positive), ("fp", emitted & ~positive),
        ("fn", ~emitted & positive), ("tn", ~emitted & ~positive),
    ):
        count[name] += int((selected & mask).sum())


def _calibration_counts(counts, name, scores, positive, mask):
    for threshold in _CALIBRATION_THRESHOLDS:
        _binary_counts(counts[f"calibration:{name}:{threshold}"], scores >= threshold, positive, mask)


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
            if name in ("note_onset", "percussion"):
                frame_count = counts["percussion_frame"] if name == "percussion" else count
                frame_count["count"] += int(mask.sum().item())
                frame_count["tp"] += int((emitted & positive & mask).sum().item())
                frame_count["fp"] += int((emitted & ~positive & mask).sum().item())
                frame_count["fn"] += int((~emitted & positive & mask).sum().item())
                frame_count["tn"] += int((~emitted & ~positive & mask).sum().item())
            if name != "note_onset":
                labelled = positive & mask
                emission_mask = valid.reshape(*valid.shape, *((1,) * (target.ndim - 2))).expand_as(target)
                count["count"] += int(labelled.sum().item())
                count["tp"] += int((emitted & labelled).sum().item())
                count["emitted"] += int((emitted & emission_mask).sum().item())
                count["positions"] += int(emission_mask.sum().item())
    if "technique" in targets:
        target = targets["technique"]
        mask = _mask(target, masks["technique"], valid)
        prediction = _prediction(outputs, "technique")
        if prediction.shape != target.shape:
            raise ValueError("Invalid binary output shape for technique")
        emitted, positive = prediction >= 0, target > .5
        count = counts["technique_frame"]
        count["count"] += int(mask.sum().item())
        count["tp"] += int((emitted & positive & mask).sum().item())
        count["fp"] += int((emitted & ~positive & mask).sum().item())
        count["fn"] += int((~emitted & positive & mask).sum().item())
        count["tn"] += int((~emitted & ~positive & mask).sum().item())
        direction_mask = _mask(targets["technique_direction"], masks["technique_direction"], valid)
        direction = _prediction(outputs, "technique_direction")
        if direction.shape[:-1] != targets["technique_direction"].shape:
            raise ValueError("Invalid categorical output shape for technique_direction")
        counts["technique_direction"]["correct"] += int(((direction.argmax(-1) == targets["technique_direction"]) & direction_mask).sum().item())
        counts["technique_direction"]["count"] += int(direction_mask.sum().item())
        string_target = targets["technique_strings"]
        string_mask = _mask(string_target, masks["technique_strings"], valid)
        string_prediction = _prediction(outputs, "technique_strings")
        if string_prediction.shape != string_target.shape:
            raise ValueError("Invalid binary output shape for technique_strings")
        emitted, positive = string_prediction >= 0, string_target > .5
        count = counts["technique_strings"]
        count["count"] += int(string_mask.sum().item())
        count["tp"] += int((emitted & positive & string_mask).sum().item())
        count["fp"] += int((emitted & ~positive & string_mask).sum().item())
        count["fn"] += int((~emitted & positive & string_mask).sum().item())
        count["tn"] += int((~emitted & ~positive & string_mask).sum().item())
    if "connection" in targets:
        connection_types, note_technique_types = vocabularies(4 if "grace_logits" in outputs else 3)
        target = targets["connection"]
        mask = _mask(target, masks["connection"], valid)
        prediction = _prediction(outputs, "connection")
        if prediction.shape[:-1] != target.shape:
            raise ValueError("Invalid categorical output shape for connection")
        selected = prediction.argmax(-1)
        counts["connection"]["correct"] += int(((selected == target) & mask).sum())
        counts["connection"]["count"] += int(mask.sum())
        probabilities = prediction.softmax(-1)
        for axis, name in enumerate(connection_types):
            emitted, positive = selected == axis, target == axis
            count = counts[f"connection:{name}"]
            count["count"] += int(mask.sum())
            count["tp"] += int((emitted & positive & mask).sum())
            count["fp"] += int((emitted & ~positive & mask).sum())
            count["fn"] += int((~emitted & positive & mask).sum())
            count["tn"] += int((~emitted & ~positive & mask).sum())
            if axis:
                _calibration_counts(counts, f"connection:{name}", probabilities[..., axis].masked_fill(selected != axis, 0), positive, mask)
        target = targets["note_technique"]
        mask = _mask(target, masks["note_technique"], valid)
        prediction = _prediction(outputs, "note_technique")
        if prediction.shape != target.shape:
            raise ValueError("Invalid binary output shape for note_technique")
        emitted, positive = prediction >= 0, target > .5
        count = counts["note_technique"]
        count["count"] += int(mask.sum())
        count["tp"] += int((emitted & positive & mask).sum())
        count["fp"] += int((emitted & ~positive & mask).sum())
        count["fn"] += int((~emitted & positive & mask).sum())
        count["tn"] += int((~emitted & ~positive & mask).sum())
        probabilities = prediction.sigmoid()
        for axis, name in enumerate(note_technique_types):
            selected_mask = mask[..., axis]
            selected_emitted, selected_positive = emitted[..., axis], positive[..., axis]
            count = counts[f"note_technique:{name}"]
            count["count"] += int(selected_mask.sum())
            count["tp"] += int((selected_emitted & selected_positive & selected_mask).sum())
            count["fp"] += int((selected_emitted & ~selected_positive & selected_mask).sum())
            count["fn"] += int((~selected_emitted & selected_positive & selected_mask).sum())
            count["tn"] += int((~selected_emitted & ~selected_positive & selected_mask).sum())
            _calibration_counts(counts, f"note_technique:{name}", probabilities[..., axis], selected_positive, selected_mask)
    if "grace" in targets:
        mask = _mask(targets["grace"], masks["grace"], valid)
        scores = outputs["grace_logits"].sigmoid()
        _binary_counts(counts["grace"], scores >= .5, targets["grace"] > .5, mask)
        _calibration_counts(counts, "grace", scores, targets["grace"] > .5, mask)
        for name in ("grace_fret", "grace_mode", "grace_transition"):
            mask = _mask(targets[name], masks[name], valid)
            count = counts[name]
            count["count"] += int(mask.sum())
            count["correct"] += int(((outputs[f"{name}_logits"].argmax(-1) == targets[name]) & mask).sum())


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
    percussion = counts["percussion_frame"]
    tp, fp, fn, tn = (percussion[key] for key in ("tp", "fp", "fn", "tn"))
    available = fp + tn > 0
    result["percussion_frame"] = {
        "available": available, "count": percussion["count"],
        "labelled_positive_count": tp + fn, "labelled_negative_count": fp + tn,
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "precision": _divide(tp, tp + fp) if available else None,
        "recall": _divide(tp, tp + fn) if available else None,
        "f1": _divide(2 * tp, 2 * tp + fp + fn) if available else None,
    }
    for name in ("technique_frame", "technique_strings"):
        value = counts[name]
        tp, fp, fn, tn = (value[key] for key in ("tp", "fp", "fn", "tn"))
        result[name] = {
            "available": bool(value["count"]), "count": value["count"],
            "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
            "precision": _divide(tp, tp + fp), "recall": _divide(tp, tp + fn),
            "f1": _divide(2 * tp, 2 * tp + fp + fn),
        }
    direction = counts["technique_direction"]
    result["technique_direction_accuracy"] = {
        "available": bool(direction["count"]), "count": direction["count"],
        "correct": direction["correct"], "accuracy": _divide(direction["correct"], direction["count"]),
    }
    connection = counts["connection"]
    result["connection_accuracy"] = {
        "available": bool(connection["count"]), "count": connection["count"],
        "correct": connection["correct"], "accuracy": _divide(connection["correct"], connection["count"]),
    }
    result["connection_classes"] = {}
    for name in CONNECTION_TYPES:
        value = counts[f"connection:{name}"]
        tp, fp, fn, tn = (value[key] for key in ("tp", "fp", "fn", "tn"))
        result["connection_classes"][name] = {
            "available": bool(value["count"]), "count": value["count"],
            "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
            "precision": _divide(tp, tp + fp), "recall": _divide(tp, tp + fn),
            "f1": _divide(2 * tp, 2 * tp + fp + fn),
        }
    value = counts["note_technique"]
    tp, fp, fn, tn = (value[key] for key in ("tp", "fp", "fn", "tn"))
    result["note_technique_frame"] = {
        "available": bool(value["count"]), "count": value["count"],
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "precision": _divide(tp, tp + fp), "recall": _divide(tp, tp + fn),
        "f1": _divide(2 * tp, 2 * tp + fp + fn),
    }
    result["note_technique_classes"] = {}
    for name in V4_NOTE_TECHNIQUE_TYPES:
        value = counts[f"note_technique:{name}"]
        tp, fp, fn, tn = (value[key] for key in ("tp", "fp", "fn", "tn"))
        result["note_technique_classes"][name] = {
            "available": bool(value["count"]), "count": value["count"],
            "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
            "precision": _divide(tp, tp + fp), "recall": _divide(tp, tp + fn),
            "f1": _divide(2 * tp, 2 * tp + fp + fn),
        }
    value = counts["grace"]
    tp, fp, fn, tn = (value[key] for key in ("tp", "fp", "fn", "tn"))
    result["grace_frame"] = {
        "available": bool(value["count"]), "count": value["count"],
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "precision": _divide(tp, tp + fp) if fp + tn else None,
        "recall": _divide(tp, tp + fn),
        "f1": _divide(2 * tp, 2 * tp + fp + fn) if fp + tn else None,
    }
    for name in ("grace_fret", "grace_mode", "grace_transition"):
        value = counts[name]
        result[f"{name}_accuracy"] = {
            "available": bool(value["count"]), "count": value["count"],
            "correct": value["correct"], "accuracy": _divide(value["correct"], value["count"]),
        }
    result["technique_calibration"] = {}
    for name in _CALIBRATION_TASKS:
        curve = []
        for threshold in _CALIBRATION_THRESHOLDS:
            value = counts[f"calibration:{name}:{threshold}"]
            tp, fp, fn, tn = (value[key] for key in ("tp", "fp", "fn", "tn"))
            curve.append({
                "threshold": threshold, "true_positive": tp, "false_positive": fp,
                "false_negative": fn, "true_negative": tn,
                "precision": _divide(tp, tp + fp) if fp + tn else None,
                "recall": _divide(tp, tp + fn),
                "f1": _divide(2 * tp, 2 * tp + fp + fn) if fp + tn and tp + fn else None,
            })
        if any(item["true_positive"] + item["false_positive"] + item["false_negative"] + item["true_negative"] for item in curve):
            eligible = [item for item in curve if item["f1"] is not None]
            result["technique_calibration"][name] = {
                "thresholds": curve,
                "bestThreshold": max(eligible, key=lambda item: (item["f1"], item["threshold"]))["threshold"] if eligible else None,
                "scope": "masked validation frames at known note anchors; not calibrated probabilities or decoded-event thresholds",
            }
    return result


def format_duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _emit(progress, message):
    if progress is not None:
        progress(message)


def _plucking_thumb_voice_counts(outputs, batch, counts):
    video = batch.get("video")
    if video is None or video["structured"].shape[-1] < 233 or "voice" not in batch["masks"]:
        return
    indices = video["frame_indices"]
    mapped = indices >= 0
    safe = indices.clamp_min(0)
    structured = video["structured"][:, :, 1]
    available = video["structured_available"][:, :, 1]
    gather_values = safe[:, :, None].expand(-1, -1, structured.shape[-1])
    values = structured.gather(1, gather_values)
    masks = available.gather(1, gather_values)
    thumb_string = values[:, :, 195]
    thumb_motion = values[:, :, 219:221]
    reliable = (
        mapped
        & masks[:, :, 194:196].all(-1)
        & masks[:, :, 219:221].all(-1)
        & masks[:, :, 229:231].all(-1)
        & (values[:, :, 229] >= .5)
        & (values[:, :, 230] <= .5)
        & (thumb_motion.norm(dim=-1) >= .25)
    )
    predictions = outputs["voice_logits"].argmax(-1)
    targets = batch["targets"]["voice"]
    voice_masks = batch["masks"]["voice"]
    for string in range(6):
        selected = reliable & voice_masks[:, :, string] & ((thumb_string - string).abs() <= .75)
        for name, target_voice in (("plucking_thumb_voice", None), ("plucking_thumb_voice:0", 0), ("plucking_thumb_voice:1", 1)):
            subset = selected if target_voice is None else selected & (targets[:, :, string] == target_voice)
            count = int(subset.sum().item())
            counts[name]["count"] += count
            counts[name]["correct"] += int(((predictions[:, :, string] == targets[:, :, string]) & subset).sum().item())


def _progress_due(completed, total, now, last_log):
    return completed == 1 or completed == total or completed % 10 == 0 or now - last_log >= 5


def evaluate_model(model, loader, device, *, sparsity_weight=0.02, progress=None, phase="Validation", budget=None):
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
        name: {"count": 0, "sum": 0.0, "correct": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "emitted": 0, "positions": 0}
        for name in ("note_onset", "fret", "pitch", "voice", "duration", "harmonic_presence", "percussion", "percussion_frame")
        + ("technique_frame", "technique_direction", "technique_strings")
        + ("connection", "note_technique")
        + tuple(f"connection:{name}" for name in CONNECTION_TYPES)
        + tuple(f"note_technique:{name}" for name in V4_NOTE_TECHNIQUE_TYPES)
        + ("grace", "grace_fret", "grace_mode", "grace_transition")
        + ("plucking_thumb_voice", "plucking_thumb_voice:0", "plucking_thumb_voice:1")
        + tuple(f"calibration:{name}:{threshold}" for name in _CALIBRATION_TASKS for threshold in _CALIBRATION_THRESHOLDS)
    }
    batches = frames = windows = 0
    started = last_log = time.perf_counter()
    total_batches = len(loader) if progress is not None and isinstance(loader, Sized) else None
    _emit(progress, f"{phase}: starting.")
    try:
        model.eval()
        with torch.no_grad():
            for source_batch in _budget_batches(loader, budget):
                batch = _batch_to_device(source_batch, device)
                outputs = _outputs(model, batch)
                _, stats = _loss(outputs, batch, loss_function, weights, sparsity_weight)
                for name, value in stats.items():
                    total = totals[name]
                    total["sum"] += value["sum"]
                    total["count"] += value["count"]
                _metric_counts(outputs, batch, counts)
                _plucking_thumb_voice_counts(outputs, batch, counts)
                batches += 1
                frames += int(batch["valid_frames"].sum().item())
                windows += batch["features"].shape[0]
                now = time.perf_counter()
                if progress is not None and _progress_due(batches, total_batches, now, last_log):
                    total = f"/{total_batches}" if total_batches is not None else ""
                    _emit(progress, f"{phase}: batch {batches}{total} | windows {windows} | elapsed {format_duration(now - started)}")
                    last_log = now
    finally:
        for module, mode in modes:
            module.training = mode
        _restore_rng(rng)
        if generator_state is not None:
            generator.set_state(generator_state)
    totals = _stats(totals, weights)
    loss = _objective(totals, weights, sparsity_weight)
    loss_text = f"{loss:.6f}" if loss is not None else "unavailable (no supervised labels)"
    _emit(progress, f"{phase}: finished | loss {loss_text} | {windows} windows | elapsed {format_duration(time.perf_counter() - started)}")
    metrics = _metrics(counts)
    metrics["plucking_thumb_voice_accuracy"] = {
        name: {
            "available": bool(counts[key]["count"]),
            "count": counts[key]["count"],
            "correct": counts[key]["correct"],
            "accuracy": _divide(counts[key]["correct"], counts[key]["count"]),
        }
        for name, key in (
            ("all", "plucking_thumb_voice"),
            ("codeVoice0DisplayedVoice1", "plucking_thumb_voice:0"),
            ("codeVoice1DisplayedVoice2", "plucking_thumb_voice:1"),
        )
    }
    return _json_copy({
        "loss": loss, "available": loss is not None, "batches": batches, "valid_frames": frames, "windows": windows,
        "loss_statistics": totals, **metrics,
    })


def _fixed_config(config):
    return {key: value for key, value in config.items() if key not in ("epochs", "max_steps", "max_seconds")}


def _budget_batches(loader, budget):
    if budget is not None:
        budget.check()
    iterator = iter(loader)
    while True:
        if budget is not None:
            budget.check()
        try:
            batch = next(iterator)
        except StopIteration:
            return
        if budget is not None:
            budget.check()
        yield batch


def _require_same_rng(before, after, *, source="Data loading"):
    if not (
        before["python"] == after["python"] and before["numpy"] == after["numpy"]
        and torch.equal(before["torch_cpu"], after["torch_cpu"])
        and len(before["torch_cuda"]) == len(after["torch_cuda"])
        and all(torch.equal(left, right) for left, right in zip(before["torch_cuda"], after["torch_cuda"]))
    ):
        raise ValueError(f"{source} consumed global RNG; exact training resume is unsupported")


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


def _event_result(value, *, metric=None):
    result = _json_copy(value)
    fields = {"score", "metric", "windows"}
    v4 = isinstance(result, dict) and result.get("metric") == "harmonic-mean-base-event-micro-f1-and-non-none-technique-macro-f1@100ms-v4"
    if v4:
        fields |= {"baseEventScore", "nonNoneTechniqueScore", "techniqueClasses"}
    _keys(result, fields, "decoded-event evaluation")
    result["score"] = _finite(result["score"], "decoded-event score")
    if result["score"] > 1:
        raise ValueError("Decoded-event score must be within [0, 1]")
    if type(result["metric"]) is not str or not result["metric"].strip():
        raise ValueError("Decoded-event metric must be a nonempty string")
    if metric is not None and result["metric"] != metric:
        raise ValueError("Decoded-event metric must stay stable within a run")
    _integer(result["windows"], "decoded-event windows")
    if v4:
        for key in ("baseEventScore", "nonNoneTechniqueScore"):
            _finite(result[key], key)
            if result[key] > 1:
                raise ValueError(f"{key} must be within [0, 1]")
        _integer(result["techniqueClasses"], "supported technique classes", minimum=1)
        base, technique = result["baseEventScore"], result["nonNoneTechniqueScore"]
        expected = 2 * base * technique / (base + technique) if base + technique else 0.
        if not math.isclose(result["score"], expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("V4 checkpoint score differs from its base/technique harmonic mean.")
    return result


def _best_event_selection(history, run_id):
    best = None
    for entry in history:
        if "decoded_events" not in entry["validation"]:
            continue
        result = _event_result(
            entry["validation"]["decoded_events"], metric=best["metric"] if best else None,
        )
        if best is None or result["score"] > best["score"]:
            best = {
                "run_id": run_id, "metric": result["metric"], "score": result["score"],
                "global_step": entry["global_step"], "checkpoint": "best-events.pt",
            }
    return best


def _event_plateau(history):
    scores = [
        entry["validation"]["decoded_events"]["score"]
        for entry in history if "decoded_events" in entry["validation"]
    ]
    if not scores:
        return 0
    best = max(range(len(scores)), key=lambda index: scores[index])
    return len(scores) - 1 - best


def _evaluate_events(model, evaluator, *, metric=None, budget=None):
    modes = [(module, module.training) for module in model.modules()]
    rng = _capture_rng()
    hook = None
    try:
        if budget is not None:
            budget.check()
            hook = model.register_forward_pre_hook(lambda *_: budget.check())
        model.eval()
        with torch.no_grad():
            result = evaluator(model)
        if budget is not None:
            budget.check()
        _require_same_rng(rng, _capture_rng(), source="Event evaluation")
        return _event_result(result, metric=metric)
    finally:
        if hook is not None:
            hook.remove()
        for module, mode in modes:
            module.training = mode
        _restore_rng(rng)


def run_training(model, train_loader, validation_loader, config, run_dir, identity, *, resume=None, progress=None,
                 event_evaluator=None, started_at=None, deadline=None, checkpoint_reserve_seconds=120.):
    """Perform AdamW updates only when explicitly requested.

    Epochs and max_steps are total ceilings. Exact resume requires unchanged,
    deterministic map-style loaders with private generators and zero workers.
    The caller must seed model construction separately for reproducible new runs.
    Optional decoded-event evaluation runs after each training validation, not
    the initial preflight. Its highest stable-metric score selects best-events.pt;
    best.pt remains the lowest-loss checkpoint and latest.pt the resume cursor.
    For joint audio/video models, global_step counts consumed batches;
    history separately records cumulative optimizer updates and zero-gradient
    skips. Skipped batches change neither weights, weight decay nor momentum.
    max_seconds is a fresh per-invocation cooperative budget. Pass the caller's
    monotonic started_at/deadline to include setup; native I/O can overrun it.
    """
    started = time.perf_counter()
    training_windows = validation_windows = event_validation_windows = 0
    if not isinstance(config, TrainingConfig):
        raise ValueError("config must be a TrainingConfig")
    budget = TrainingBudget(config.max_seconds, started_at=started if started_at is None else started_at,
                            deadline=deadline, reserve_seconds=checkpoint_reserve_seconds)
    if event_evaluator is not None and not callable(event_evaluator):
        raise ValueError("event_evaluator must be callable or None")
    identity = _identity(identity)
    model_config = getattr(model, "config", None)
    if model_config is not None and is_dataclass(model_config) and not _json_equal(asdict(model_config), identity["model"]):
        raise ValueError("Model configuration differs from identity.model")
    video_config = getattr(model, "video_config", None)
    joint_model = video_config is not None
    if joint_model and (
        not is_dataclass(video_config)
        or not _json_equal(asdict(video_config), identity.get("video", {}).get("config"))
    ):
        raise ValueError("Joint model configuration differs from identity.video.config")
    if joint_model:
        _joint_video_config(identity)
        if any(not parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Joint training requires all acoustic, numeric-video and fusion parameters to be trainable")
    elif "video" in identity:
        raise ValueError("Joint identity requires a joint audio/video model")
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
    best_event = None
    event_files = ("best-events.pt", "event-selection.json")
    source_stamps = {}
    if resume is None:
        if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
            raise ValueError("A fresh run directory must be absent or empty")
        run_id = uuid.uuid4().hex
        _emit(progress, f"Starting new run: {run_dir}")
    else:
        if _plain_path(resume) != run_dir / "latest.pt":
            raise ValueError("Resume must use latest.pt from this exact run directory")
        source_stamps = {
            name: _file_stamp(run_dir / name) for name in ("run.json", "latest.pt")
        }
        _emit(progress, f"Loading resume checkpoint: {resume}")
        checkpoint = load_checkpoint(resume, expected_identity=identity)
        pending = checkpoint.get("resume_state", {})
        if pending.get("stopped_by") == "max_seconds" and pending["event_evaluation_required"] != (event_evaluator is not None):
            raise ValueError("A budget-paused run must retain its event-evaluation policy on resume")
        run_id = checkpoint["run_id"]
        manifest = _read_json(run_dir / "run.json")
        _keys(manifest, {"schema_version", "run_id", "identity", "config"}, "run manifest")
        manifest_config_keys = set(asdict(TrainingConfig()))
        if manifest["schema_version"] == 1:
            manifest_config_keys.discard("max_seconds")
        if manifest["schema_version"] < 3:
            manifest_config_keys -= {"event_patience", "learning_rate_patience", "learning_rate_factor", "minimum_learning_rate"}
        _keys(manifest["config"], manifest_config_keys, "run training config")
        TrainingConfig(**manifest["config"])
        if (
            manifest["schema_version"] not in (1, 2, SCHEMA_VERSION) or manifest["run_id"] != run_id
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
        if (run_dir / "best.pt").exists() != (checkpoint["best_score"] is not None):
            raise ValueError("best.pt and latest.pt do not describe a consistent run")
        if checkpoint["best_score"] is not None:
            source_stamps["best.pt"] = _file_stamp(run_dir / "best.pt")
            best = load_checkpoint(run_dir / "best.pt", expected_identity=identity)
            if (
                best["run_id"] != run_id or best["cursor"]["global_step"] > cursor["global_step"]
                or not best["history"] or best["history"][-1]["validation"]["loss"] != checkpoint["best_score"]
            ):
                raise ValueError("best.pt and latest.pt do not describe a consistent run")
        best_event = _best_event_selection(checkpoint["history"], run_id)
        for filename in event_files:
            present = (run_dir / filename).exists()
            if present != (best_event is not None):
                raise ValueError("Decoded-event selection files and checkpoint history are inconsistent")
            if present:
                source_stamps[filename] = _file_stamp(run_dir / filename)
        if best_event is not None:
            selection = _read_json(run_dir / "event-selection.json")
            selected = load_checkpoint(run_dir / "best-events.pt", expected_identity=identity)
            if (
                not _json_equal(selection, best_event) or selected["run_id"] != run_id
                or selected["global_step"] != best_event["global_step"]
                or not _json_equal(_best_event_selection(selected["history"], run_id), best_event)
            ):
                raise ValueError("best-events.pt and event selection do not describe a consistent run")
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
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_parameters, lr=config.learning_rate, weight_decay=config.weight_decay,
            **_OPTIMIZER_OPTIONS,
        )
        if checkpoint is None:
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)
            epoch = next_batch = global_step = 0
            optimizer_updates = optimizer_skips = 0
            best_score = None
            history = []
            epoch_start = train_loader.generator.get_state()
            phase = "initial_validation"
        else:
            model.load_state_dict(checkpoint["model_state"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            for parameter, state in optimizer.state.items():
                if state["exp_avg"].shape != parameter.shape or state["exp_avg_sq"].shape != parameter.shape:
                    raise ValueError("Optimizer moment shapes differ from the model parameters")
            epoch, next_batch, global_step = (checkpoint["cursor"][key] for key in ("epoch", "next_batch_index", "global_step"))
            best_score, history = checkpoint["best_score"], checkpoint["history"]
            counts = checkpoint.get("resume_state", history[-1] if history else {})
            optimizer_updates = counts.get("optimizer_updates", global_step)
            optimizer_skips = counts.get("optimizer_skipped_batches", 0)
            phase = checkpoint.get("resume_state", {}).get("phase", "training")
            train_loader.generator.set_state(checkpoint["loader_state"]["train"])
            validation_loader.generator.set_state(checkpoint["loader_state"]["validation"])
            epoch_start = checkpoint["loader_state"]["epoch_start"]
            _restore_rng(checkpoint["rng"])
        initial_epoch, initial_step = epoch, global_step
        initial_updates, initial_skips = optimizer_updates, optimizer_skips
        validation = history[-1]["validation"] if history else None
        expired = False
        early_stopped = False
        try:
            budget.check()
            budget_resume = checkpoint is not None and checkpoint.get("resume_state", {}).get("stopped_by") == "max_seconds"
            if not budget_resume or phase == "initial_validation":
                initial_validation = evaluate_model(model, validation_loader, device, sparsity_weight=config.sparsity_weight,
                                                    progress=progress, phase="Initial validation", budget=budget)
                _require_validation(initial_validation)
                validation_windows += initial_validation["windows"]
                if phase == "initial_validation":
                    phase = "training"
                validation = initial_validation
            budget.check()
        except TrainingBudgetExpired:
            expired = True
        if checkpoint is None:
            if run_dir.exists():
                if not run_dir.is_dir() or any(run_dir.iterdir()):
                    raise ValueError("Fresh run directory became nonempty before publication")
            else:
                run_dir.mkdir(parents=True, exist_ok=False)
        lock = lock_path.open("xb")
        if any(_file_stamp(run_dir / name) != stamp for name, stamp in source_stamps.items()):
            raise ValueError("Run changed before the training lock was acquired; reload latest.pt")
        if best_event is None and any((run_dir / name).exists() for name in event_files):
            raise ValueError("Decoded-event selection files appeared before the training lock was acquired")
        _write_json(run_dir / "run.json", {
            "schema_version": SCHEMA_VERSION, "run_id": run_id, "identity": identity, "config": asdict(config),
        }, replace=checkpoint is not None)

        def save_checkpoint(*, improved=False, event_improved=False, stopped_by=None):
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
                "resume_state": {
                    "phase": phase,
                    "validation_pending": phase == "initial_validation" or global_step > (history[-1]["global_step"] if history else 0),
                    "event_evaluation_required": event_evaluator is not None,
                    "optimizer_updates": optimizer_updates, "optimizer_skipped_batches": optimizer_skips,
                    "stopped_by": stopped_by,
                },
            }
            _validate_checkpoint(payload)
            _emit(progress, "Saving checkpoint and metrics...")
            if improved:
                _atomic_write(run_dir / "best.pt", lambda stream: torch.save(payload, stream))
            if event_improved:
                _emit(progress, "Saving best-events.pt and event-selection.json...")
                _atomic_write(run_dir / "best-events.pt", lambda stream: torch.save(payload, stream))
                _write_json(run_dir / "event-selection.json", best_event)
                _emit(progress, f"Saved best-events.pt | best decoded-event {best_event['metric']} {best_event['score']:.6f}")
            _atomic_write(run_dir / "latest.pt", lambda stream: torch.save(payload, stream))
            _write_json(run_dir / "metrics.json", history)
            saved = "latest.pt, best.pt and metrics.json" if improved else "latest.pt and metrics.json"
            _emit(progress, f"Saved {saved} | best validation loss {best_score}")

        try:
            if expired:
                raise TrainingBudgetExpired()
            if global_step > (history[-1]["global_step"] if history else 0) and config.max_steps is not None and global_step >= config.max_steps:
                phase = "validation"
            while phase == "validation" or epoch < config.epochs and (config.max_steps is None or global_step < config.max_steps):
                budget.check()
                epoch_started = last_log = time.perf_counter()
                if phase != "validation":
                    first_batch = next_batch
                    batch_count = runtime["train_loader"]["batches"]
                    batch_limit = batch_count if config.max_steps is None else min(batch_count, next_batch + config.max_steps - global_step)
                    _emit(progress, f"Epoch {epoch + 1}/{config.epochs}: starting at batch {next_batch + 1}/{batch_count} | step {global_step}")
                    if callable(getattr(train_loader.sampler, "set_epoch", None)):
                        before = _capture_rng()
                        train_loader.sampler.set_epoch(epoch)
                        _require_same_rng(before, _capture_rng())
                    resuming_partial = next_batch > 0
                    saved_generator = train_loader.generator.get_state()
                    if resuming_partial:
                        train_loader.generator.set_state(epoch_start)
                    else:
                        epoch_start = saved_generator
                    try:
                        iterator = _iterator(train_loader)
                        for _ in range(next_batch):
                            budget.check()
                            _checked_next(iterator)
                        if resuming_partial and not torch.equal(train_loader.generator.get_state(), saved_generator):
                            raise ValueError("Cannot reproduce the checkpoint's mid-epoch loader RNG state")
                        model.train()
                        while next_batch < batch_limit:
                            budget.check()
                            batch = _batch_to_device(_checked_next(iterator), device)
                            budget.check()
                            optimizer.zero_grad(set_to_none=True)
                            outputs = _outputs(model, batch)
                            loss, stats = _loss(outputs, batch, loss_function, weights, config.sparsity_weight)
                            supervised = _objective(stats, weights, config.sparsity_weight) is not None
                            if not supervised and not joint_model:
                                raise ValueError("Training batch has no meaningful supervised objective")
                            if supervised and loss.requires_grad:
                                loss.backward()
                            elif supervised:
                                raise ValueError("Training loss produced no model gradients")
                            gradients = [parameter.grad for parameter in trainable_parameters if parameter.grad is not None]
                            if supervised and not gradients:
                                raise ValueError("Training loss produced no model gradients")
                            if any(not torch.isfinite(gradient).all().item() for gradient in gradients):
                                raise ValueError("Nonfinite model gradient")
                            if joint_model:
                                for parameter in trainable_parameters:
                                    if parameter.grad is not None and not torch.count_nonzero(parameter.grad).item():
                                        parameter.grad = None
                            if any(parameter.grad is not None for parameter in trainable_parameters):
                                torch.nn.utils.clip_grad_norm_(trainable_parameters, config.gradient_clip, error_if_nonfinite=True)
                                optimizer.step()
                                optimizer_updates += 1
                            else:
                                optimizer_skips += 1
                            for name, value in model.state_dict().items():
                                _tensor(value, f"updated model.{name}")
                            global_step += 1
                            next_batch += 1
                            training_windows += batch["features"].shape[0]
                            now = time.perf_counter()
                            processed = next_batch - first_batch
                            if progress is not None and _progress_due(processed, batch_limit - first_batch, now, last_log):
                                eta = (now - epoch_started) / processed * (batch_limit - next_batch)
                                activity = f" | optimizer updates {optimizer_updates}, skipped {optimizer_skips}" if joint_model else ""
                                _emit(progress, f"Epoch {epoch + 1}/{config.epochs} | batch {next_batch}/{batch_count} | step {global_step}{activity} | batch loss {loss.detach().item():.6f} | elapsed {format_duration(now - started)} | epoch training ETA {format_duration(eta)}")
                                last_log = now
                    except TrainingBudgetExpired:
                        # Reconstructing/skipping loader batches must not advance its saved RNG.
                        if next_batch == first_batch:
                            train_loader.generator.set_state(saved_generator)
                        raise
                    if next_batch == batch_count:
                        before = _capture_rng()
                        if next(iterator, None) is not None:
                            raise ValueError("Training loader exceeded its declared batch count")
                        _require_same_rng(before, _capture_rng())
                        epoch += 1
                        next_batch = 0
                    phase = "validation"
                epoch_complete = next_batch == 0
                epoch_index = epoch - 1 if epoch_complete else epoch
                budget.check()
                current_validation = evaluate_model(model, validation_loader, device, sparsity_weight=config.sparsity_weight,
                                                    progress=progress, phase=f"Epoch {epoch_index + 1} validation", budget=budget)
                _require_validation(current_validation)
                validation_windows += current_validation["windows"]
                event_improved = False
                if event_evaluator is not None:
                    budget.check()
                    _emit(progress, f"Epoch {epoch_index + 1} decoded-event validation: starting.")
                    events = _evaluate_events(model, event_evaluator, metric=best_event["metric"] if best_event else None, budget=budget)
                    current_validation["decoded_events"] = events
                    event_validation_windows += events["windows"]
                    event_improved = best_event is None or events["score"] > best_event["score"]
                    _emit(progress, f"Decoded-event {events['metric']}: {events['score']:.6f} | {events['windows']} windows")
                budget.check()
                validation = current_validation
                history.append({
                    "epoch": epoch_index, "global_step": global_step,
                    "epoch_complete": epoch_complete, "validation": validation,
                })
                if joint_model:
                    history[-1].update(optimizer_updates=optimizer_updates, optimizer_skipped_batches=optimizer_skips)
                improved = best_score is None or validation["loss"] < best_score
                if improved:
                    best_score = validation["loss"]
                if event_improved:
                    best_event = _best_event_selection(history, run_id)
                plateau = _event_plateau(history) if event_evaluator is not None else 0
                if (
                    event_evaluator is not None and not event_improved and plateau > 0
                    and plateau % config.learning_rate_patience == 0
                ):
                    before = optimizer.param_groups[0]["lr"]
                    after = max(config.minimum_learning_rate, before * config.learning_rate_factor)
                    for group in optimizer.param_groups:
                        group["lr"] = after
                    if after < before:
                        _emit(progress, f"Reduced learning rate: {before:.8g} -> {after:.8g}")
                phase = "training"
                save_checkpoint(improved=improved, event_improved=event_improved)
                state = "complete" if epoch_complete else "paused at step ceiling"
                _emit(progress, f"Epoch {epoch_index + 1}/{config.epochs}: {state} | elapsed {format_duration(time.perf_counter() - epoch_started)}")
                if event_evaluator is not None and plateau >= config.event_patience:
                    early_stopped = True
                    _emit(progress, f"Early stopping: decoded-event score did not improve for {plateau} validations.")
                    break
                budget.check()
        except TrainingBudgetExpired:
            expired = True
            _emit(progress, "Wall-clock budget reached; saving exact resume state without incomplete validation scores.")
            save_checkpoint(stopped_by="max_seconds")
        budget_report = budget.report()
        return _json_copy({
            "run_dir": str(run_dir), "global_step": global_step, "epoch": epoch,
            "next_batch_index": next_batch, "best_score": best_score, "validation": validation,
            "status": "paused" if expired else "complete",
            "training_started": optimizer_updates > initial_updates,
            "stopped_by": "max_seconds" if expired else "decoded_event_patience" if early_stopped else "epochs" if epoch >= config.epochs else "max_steps",
            "validation_pending": phase == "initial_validation" or global_step > (history[-1]["global_step"] if history else 0),
            "resume_phase": phase, "epochs_remaining": max(0, config.epochs - epoch),
            "latest_checkpoint": str(run_dir / "latest.pt"), "best_checkpoint": str(run_dir / "best.pt") if best_score is not None else None,
            "best_event_checkpoint": str(run_dir / best_event["checkpoint"]) if best_event else None,
            "best_event_score": best_event["score"] if best_event else None,
            **budget_report, "training_elapsed_seconds": time.perf_counter() - started,
            "epochs_requested": config.epochs, "epochs_completed_this_run": epoch - initial_epoch,
            "training_steps_processed": global_step - initial_step,
            "optimizer_updates": optimizer_updates - initial_updates,
            "optimizer_skipped_batches": optimizer_skips - initial_skips,
            "total_optimizer_updates": optimizer_updates, "total_optimizer_skipped_batches": optimizer_skips,
            "training_windows_processed": training_windows, "validation_windows_processed": validation_windows,
            "event_validation_windows_processed": event_validation_windows,
            "dataset_windows": {"train": runtime["train_loader"]["dataset_size"], "validation": runtime["validation_loader"]["dataset_size"]},
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
