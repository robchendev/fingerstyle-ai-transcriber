"""Local PyTorch commands. Training is an explicit human-run operation."""

import argparse
from collections import Counter
from dataclasses import asdict, fields
import math
from pathlib import Path
import random
import sys

import numpy as np
import scipy
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from .align_pilot import publish_json
from .catalogs import ROOT, read_json, sha256
from .transcriber_audio import FeatureConfig, HarnessError, audio_features, conditioning_features, read_audio_window
from .transcriber_data import EpochShuffleSampler, PilotDataset, collate_windows


def private_output(path, root=ROOT):
    root = Path(root).resolve()
    path = Path(path)
    path = path if path.is_absolute() else root / path
    absolute = path.absolute()
    if path.resolve() != absolute or not any(absolute.is_relative_to(root / name) for name in ("runs", "cache", "checkpoints")):
        raise HarnessError("Output must be an unaliased path under the private runs, cache or checkpoints directory.")
    return absolute


def default_config():
    from .transcriber_model import ModelConfig
    from .transcriber_runtime import TrainingConfig
    return {
        "schemaVersion": 1, "features": asdict(FeatureConfig()), "model": asdict(ModelConfig()),
        "training": asdict(TrainingConfig()),
        "data": {"manifest": "data\\pilot-training-manifest.json", "batch_size": 4, "num_workers": 0, "num_threads": 4, "cache": "cache\\transcriber"},
    }


def load_config(path=None):
    from .transcriber_model import ModelConfig
    from .transcriber_runtime import TrainingConfig
    config = read_json(Path(path)) if path else default_config()
    if not isinstance(config, dict) or type(config.get("schemaVersion")) is not int or config["schemaVersion"] != 1 or set(config) != {"schemaVersion", "features", "model", "training", "data"}:
        raise HarnessError("Expected version-1 harness configuration.")
    instances = []
    for key, cls in (("features", FeatureConfig), ("model", ModelConfig), ("training", TrainingConfig)):
        values = config[key]
        if not isinstance(values, dict) or set(values) - {field.name for field in fields(cls)}:
            raise HarnessError(f"Unknown {key} configuration fields.")
        instances.append(cls(**values))
    data = config["data"]
    if not isinstance(data, dict) or set(data) != {"manifest", "batch_size", "num_workers", "num_threads", "cache"}:
        raise HarnessError("Unknown or missing data-loader configuration.")
    for key, lower, upper in (("batch_size", 1, 64), ("num_workers", 0, 0), ("num_threads", 1, 32)):
        if type(data[key]) is not int or not lower <= data[key] <= upper:
            raise HarnessError(f"Invalid data-loader {key}.")
    if not isinstance(data["manifest"], str) or not isinstance(data["cache"], str):
        raise HarnessError("Manifest and cache paths must be explicit strings.")
    if instances[0].n_mels != instances[1].n_mels or instances[1].conditioning_dim != 12:
        raise HarnessError("Feature and model dimensions disagree.")
    return config, *instances


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(config, feature_config, model_config, split, root, manifest_override=None):
    manifest = Path(manifest_override or config["data"]["manifest"])
    if not manifest.is_absolute():
        manifest = Path(root) / manifest
    return PilotDataset(manifest, split, feature_config, model_config, root=root, cache_dir=private_output(config["data"]["cache"], root))


def make_loader(dataset, config, training, *, shuffle):
    sampler = EpochShuffleSampler(dataset, training.seed) if shuffle else None
    return DataLoader(
        dataset, batch_size=config["data"]["batch_size"], sampler=sampler, shuffle=False,
        num_workers=config["data"]["num_workers"], collate_fn=collate_windows,
        generator=torch.Generator().manual_seed(training.seed),
    )


def run_identity(dataset, config, features, model, training, device):
    settings = asdict(training)
    settings.pop("epochs", None)
    settings.pop("max_steps", None)
    modules = ("transcriber.py", "transcriber_audio.py", "transcriber_data.py", "transcriber_model.py", "transcriber_runtime.py")
    return {
        "schemaVersion": 1, "manifest_sha256": dataset.manifest_sha256,
        "features": asdict(features), "model": asdict(model), "training": settings,
        "batch_size": config["data"]["batch_size"], "num_workers": config["data"]["num_workers"],
        "device": str(device), "implementationSha256": {name: sha256(Path(__file__).with_name(name)) for name in modules},
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "scipy": scipy.__version__, "soundfile": sf.__version__},
        "dataPolicy": "private-approved-pilot; no source legends or presentation settings as model inputs",
    }


def preflight(args):
    from .transcriber_model import FingerstyleTranscriber
    config, features, model_config, training = load_config(args.config)
    torch.set_num_threads(config["data"]["num_threads"])
    seed_everything(training.seed)
    model = FingerstyleTranscriber(model_config).eval() if args.forward else None
    report = {"schemaVersion": 1, "kind": "transcriber-preflight", "trainingRun": False, "weights": "untrained-in-memory-only" if model else "not-created", "splits": {}}
    for split in ("train", "validation"):
        dataset = make_dataset(config, features, model_config, split, args.data_root, args.manifest)
        counts = Counter()
        first = None
        for item in dataset:
            if first is None:
                first = item
            counts["windows"] += 1
            counts["frames"] += len(item["features"])
            counts["collisions_masked"] += item["metadata"]["stringFrameCollisionsMasked"]
            for name, mask in item["masks"].items():
                counts[f"supervised_{name}"] += int(mask.sum())
        row = dict(counts)
        if model is not None:
            batch = collate_windows([first])
            with torch.no_grad():
                outputs = model(batch["features"], batch["conditioning"], batch["lengths"])
            if not all(torch.isfinite(value).all() for value in outputs.values()):
                raise HarnessError("Model forward produced nonfinite output.")
            row["forwardShapes"] = {name: list(value.shape) for name, value in outputs.items()}
        report["splits"][split] = row
        report["manifestSha256"] = dataset.manifest_sha256
    publish_json(private_output(args.output, args.data_root), report)
    print({"preflight": "completed", "windows": {key: value["windows"] for key, value in report["splits"].items()}, "trainingRun": False})
    return report


def train(args):
    from .transcriber_model import FingerstyleTranscriber
    from .transcriber_runtime import resolve_device, run_training
    config, features, model_config, training = load_config(args.config)
    torch.set_num_threads(config["data"]["num_threads"])
    device = resolve_device(training.device)
    seed_everything(training.seed)
    train_data = make_dataset(config, features, model_config, "train", args.data_root, args.manifest)
    validation_data = make_dataset(config, features, model_config, "validation", args.data_root, args.manifest)
    if train_data.manifest_sha256 != validation_data.manifest_sha256:
        raise HarnessError("Training and validation reference different releases.")
    model = FingerstyleTranscriber(model_config)
    identity = run_identity(train_data, config, features, model_config, training, device)
    result = run_training(
        model, make_loader(train_data, config, training, shuffle=True), make_loader(validation_data, config, training, shuffle=False),
        training, private_output(args.run_dir, args.data_root), identity, resume=args.resume,
    )
    print(result)
    return result


def checkpoint_model(path, device_name):
    from .transcriber_model import FingerstyleTranscriber, ModelConfig
    from .transcriber_runtime import load_checkpoint, resolve_device
    checkpoint = load_checkpoint(path)
    if checkpoint["global_step"] <= 0:
        raise HarnessError("Inference/evaluation requires a checkpoint with actual training steps, not random initialization.")
    identity = checkpoint["identity"]
    model = FingerstyleTranscriber(ModelConfig(**identity["model"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = resolve_device(device_name)
    return model.to(device).eval(), checkpoint, device


def evaluate(args):
    from .transcriber_model import ModelConfig
    from .transcriber_runtime import evaluate_model, TrainingConfig
    model, checkpoint, device = checkpoint_model(args.checkpoint, args.device)
    identity = checkpoint["identity"]
    config = default_config()
    config["data"].update(batch_size=identity["batch_size"], num_workers=0)
    torch.set_num_threads(config["data"]["num_threads"])
    feature_config, model_config = FeatureConfig(**identity["features"]), ModelConfig(**identity["model"])
    training = TrainingConfig(**checkpoint["training_config"])
    dataset = make_dataset(config, feature_config, model_config, args.split, args.data_root, args.manifest)
    if dataset.manifest_sha256 != identity["manifest_sha256"]:
        raise HarnessError("Evaluation release differs from this run; do not silently substitute a different split.")
    metrics = evaluate_model(model, make_loader(dataset, config, training, shuffle=False), device, sparsity_weight=training.sparsity_weight)
    report = {"schemaVersion": 1, "kind": "transcriber-evaluation", "split": args.split, "checkpointSha256": sha256(Path(args.checkpoint)), "manifestSha256": dataset.manifest_sha256, "metrics": metrics, "developmentOnly": True, "visibility": "private"}
    publish_json(private_output(args.output, args.data_root), report)
    print(metrics)
    return report


def inference_metadata(value):
    required = {"openStringMidi", "capoFret", "tempo", "timeSignature"}
    if not isinstance(value, dict) or not required <= set(value) or set(value) - required - {"tempoChanges", "timeSignatureChanges"}:
        raise HarnessError("Inference metadata requires tuning, full capo, BPM/beat unit and meter; only explicit time-based schedules are optional.")
    if not isinstance(value["tempo"], dict) or not {"bpm", "beatUnit"} <= set(value["tempo"]) or set(value["tempo"]) - {"bpm", "beatUnit", "linear"}:
        raise HarnessError("Initial tempo requires explicit BPM and beat unit.")
    tempo = [{"position": 0., **value["tempo"]}]
    changes = value.get("tempoChanges", [])
    meter_changes = value.get("timeSignatureChanges", [])
    if not isinstance(changes, list) or not isinstance(meter_changes, list):
        raise HarnessError("Timing changes must be explicit event lists.")
    for event in changes:
        if not isinstance(event, dict) or not {"timeSeconds", "bpm", "beatUnit"} <= set(event) or set(event) - {"timeSeconds", "bpm", "beatUnit", "linear"}:
            raise HarnessError("A tempo change requires timeSeconds, BPM and beat unit.")
        tempo.append({"position": event["timeSeconds"], "bpm": event["bpm"], "beatUnit": event["beatUnit"], "linear": event.get("linear", False)})
    meters = [{"position": 0., "timeSignature": value["timeSignature"]}]
    for event in meter_changes:
        if not isinstance(event, dict) or set(event) != {"timeSeconds", "timeSignature"}:
            raise HarnessError("A meter change requires timeSeconds and timeSignature.")
        meters.append({"position": event["timeSeconds"], "timeSignature": event["timeSignature"]})
    conditioning_features(value["openStringMidi"], value["capoFret"], tempo, meters, [0.])
    return tempo, meters


def infer(args):
    from .transcriber_model import decode_events
    model, checkpoint, device = checkpoint_model(args.checkpoint, args.device)
    torch.set_num_threads(4)
    metadata = read_json(Path(args.metadata))
    tempos, meters = inference_metadata(metadata)
    config = FeatureConfig(**checkpoint["identity"]["features"])
    audio_path = Path(args.audio).resolve()
    if audio_path.suffix.lower() not in (".mp3", ".flac", ".wav"):
        raise HarnessError("Inference accepts local MP3, FLAC or WAV audio.")
    audio_hash = sha256(audio_path)
    info = sf.info(audio_path)
    duration = info.frames / info.samplerate
    if not 0 < duration <= 900 or not 1 <= info.channels <= 8:
        raise HarnessError("Inference audio must be nonempty, at most 15 minutes, and have one to eight channels.")
    total_frames = math.ceil(duration / config.hop_seconds)
    accumulators, total_weight = {}, torch.zeros(total_frames)
    stride_frames = max(1, round(6 / config.hop_seconds))
    window_frames = max(stride_frames, round(8 / config.hop_seconds))
    peak = 0.
    with torch.no_grad():
        for start_frame in range(0, total_frames, stride_frames):
            start_sample = round(start_frame * config.hop_seconds * info.samplerate)
            stop_sample = min(info.frames, round((start_frame + window_frames) * config.hop_seconds * info.samplerate))
            samples, rate = read_audio_window(audio_path, start_sample, stop_sample, sample_rate=info.samplerate, channels=info.channels, sample_count=info.frames)
            peak = max(peak, float(np.max(np.abs(samples))))
            features, local_times = audio_features(samples, rate, config)
            times = local_times + start_sample / rate
            conditioning = conditioning_features(metadata["openStringMidi"], metadata["capoFret"], tempos, meters, times)
            outputs = model(features[None].to(device), conditioning[None].to(device), torch.tensor([len(features)], dtype=torch.long))
            count = min(len(features), total_frames - start_frame)
            weight = torch.hann_window(max(count, 2), periodic=False)[:count].clamp_min(.05)
            for name, value in outputs.items():
                value = value[0, :count].detach().cpu()
                if not torch.isfinite(value).all():
                    raise HarnessError("Inference produced nonfinite predictions.")
                if name not in accumulators:
                    accumulators[name] = torch.zeros((total_frames, *value.shape[1:]), dtype=value.dtype)
                broadcast = weight.view(count, *([1] * (value.ndim - 1)))
                accumulators[name][start_frame:start_frame + count] += value * broadcast
            total_weight[start_frame:start_frame + count] += weight
            if start_frame + count == total_frames:
                break
    if peak <= 1e-8:
        raise HarnessError("Input audio is silent; no transcription hypotheses were published.")
    if torch.any(total_weight <= 0) or sha256(audio_path) != audio_hash:
        raise HarnessError("Inference coverage is incomplete or the audio changed during processing.")
    outputs = {name: value / total_weight.view(total_frames, *([1] * (value.ndim - 1))) for name, value in accumulators.items()}
    events = decode_events(outputs, torch.arange(total_frames) * config.hop_seconds, tuning=metadata["openStringMidi"], capo=metadata["capoFret"], onset_threshold=args.onset_threshold, percussion_threshold=args.percussion_threshold)
    report = {
        "schemaVersion": 1, "kind": "fingerstyle-transcription-hypotheses", "visibility": "private", "distributionAuthorized": False,
        "audioSha256": audio_hash, "checkpointSha256": sha256(Path(args.checkpoint)),
        "metadata": metadata, "timeUnit": "input-audio-seconds", "notatedDurationUnit": "quarter-note",
        "gpWriterImplemented": False, "modelTrainingPerformedByThisCommand": False,
        **events,
    }
    publish_json(private_output(args.output, args.data_root), report)
    print({"notes": len(report["notes"]), "percussionHypotheses": len(report["percussion"]), "gpWritten": False})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    configure = commands.add_parser("config", help="Write a generic local configuration.")
    configure.add_argument("--output", default="runs/config.json")
    for name in ("preflight", "train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--data-root", default=str(ROOT))
        command.add_argument("--manifest")
        if name in ("preflight", "train"):
            command.add_argument("--config")
        if name == "preflight":
            command.add_argument("--output", default="runs/preflight.json")
            command.add_argument("--forward", action="store_true", help="One untrained eval-mode forward per split; no optimizer or checkpoint.")
        elif name == "train":
            command.add_argument("--run-dir", required=True)
            command.add_argument("--resume")
        else:
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--split", choices=("train", "validation"), default="validation")
            command.add_argument("--device", default="auto")
            command.add_argument("--output", default="runs/evaluation.json")
    inference = commands.add_parser("infer")
    inference.add_argument("--data-root", default=str(ROOT))
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--audio", required=True)
    inference.add_argument("--metadata", required=True)
    inference.add_argument("--output", default="runs/predictions.json")
    inference.add_argument("--device", default="auto")
    inference.add_argument("--onset-threshold", type=float, default=.5)
    inference.add_argument("--percussion-threshold", type=float, default=.5)
    args = parser.parse_args(argv)
    try:
        if args.command == "config":
            publish_json(private_output(args.output), default_config())
            print("Generic configuration written under the private runs directory.")
        else:
            {"preflight": preflight, "train": train, "evaluate": evaluate, "infer": infer}[args.command](args)
    except (HarnessError, OSError, ValueError, RuntimeError) as error:
        print(f"Transcriber error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
