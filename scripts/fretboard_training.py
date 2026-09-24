"""Validate and train the six-keypoint fretboard detector."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import sys


KEYPOINT_COUNT = 6


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish(path, value):
    path = Path(path)
    pending = path.with_name("." + path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


@dataclass(frozen=True)
class DetectorTrainingConfig:
    model: str = "yolo11n-pose.pt"
    image_size: int = 3840
    batch_size: int = 1
    epochs: int = 200
    patience: int = 30
    device: str = "0"
    workers: int = 4
    seed: int = 17

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a nonempty path or Ultralytics model name.")
        for name, minimum in (
            ("image_size", 320), ("batch_size", 1), ("epochs", 1),
            ("patience", 0), ("workers", 0), ("seed", 0),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a nonempty Ultralytics device string.")


def validate_dataset(dataset):
    root = Path(dataset).resolve()
    manifest_path = root / "manifest.json"
    annotations_path = root / "annotations.json"
    yaml_path = root / "data.yaml"
    if not all(path.is_file() for path in (manifest_path, annotations_path, yaml_path)):
        raise ValueError("Dataset requires manifest.json, annotations.json, and data.yaml.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    if manifest.get("keypoints") != annotations.get("keypoints") or len(manifest.get("keypoints", ())) != KEYPOINT_COUNT:
        raise ValueError("Dataset must use the six-keypoint contract.")
    splits_by_video = {}
    counts = {"train": 0, "validation": 0, "test": 0}
    assets = {manifest_path: _sha256(manifest_path), annotations_path: _sha256(annotations_path), yaml_path: _sha256(yaml_path)}
    for record in manifest.get("records", ()):
        split = record.get("split")
        if split not in counts:
            raise ValueError("Dataset record has an unsupported split.")
        digest = record.get("sourceVideoSha256")
        if digest in splits_by_video and splits_by_video[digest] != split:
            raise ValueError("One source video crosses detector dataset splits.")
        splits_by_video[digest] = split
        label = root / "labels" / split / f"{record['id']}.txt"
        image = root / record["image"]
        if not label.is_file():
            continue
        if not image.is_file():
            raise ValueError(f"Exported label has no image: {record['id']}")
        text = label.read_text(encoding="ascii").strip()
        if text:
            fields = text.split()
            if len(fields) != 5 + KEYPOINT_COUNT * 3 or fields[0] != "0":
                raise ValueError(f"Invalid YOLO pose label: {label}")
            values = [float(value) for value in fields[1:]]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Nonfinite YOLO pose label: {label}")
            for index in range(KEYPOINT_COUNT):
                x, y, visibility = values[4 + index * 3:7 + index * 3]
                if visibility not in (0, 1, 2) or visibility and not (0 <= x <= 1 and 0 <= y <= 1):
                    raise ValueError(f"Invalid keypoint in YOLO pose label: {label}")
        counts[split] += 1
        assets[image] = _sha256(image)
        assets[label] = _sha256(label)
    if not counts["train"] or not counts["validation"]:
        raise ValueError("Detector training requires completed train and validation labels.")
    identity = hashlib.sha256(
        json.dumps(
            {str(path.relative_to(root)): digest for path, digest in assets.items()},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {"root": root, "yaml": yaml_path, "counts": counts, "sha256": identity}


def training_request(dataset, output, config):
    validated = validate_dataset(dataset)
    output = Path(output).resolve()
    return {
        "schemaVersion": 1,
        "kind": "fretboard-detector-training-request",
        "dataset": str(validated["root"]),
        "datasetSha256": validated["sha256"],
        "counts": validated["counts"],
        "output": str(output),
        "config": asdict(config),
    }


def train_detector(dataset, output, config, *, resume=None, dry_run=False):
    request = training_request(dataset, output, config)
    print(json.dumps(request, indent=2), flush=True)
    if dry_run:
        return request
    output = Path(request["output"])
    receipt_path = output / "training-request.json"
    if output.exists():
        if not receipt_path.is_file() or json.loads(receipt_path.read_text(encoding="utf-8")) != request:
            raise ValueError("Training output exists with different inputs or settings.")
    else:
        output.mkdir(parents=True)
        _publish(receipt_path, request)
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Install requirements-fretboard.txt before detector training.") from error
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cudaAvailable": torch.cuda.is_available(),
    }
    _publish(output / "runtime.json", runtime)
    model = YOLO(str(resume) if resume else config.model)
    print(f"Starting detector training: {output}", flush=True)
    results = model.train(
        data=str(Path(request["dataset"]) / "data.yaml"),
        imgsz=config.image_size,
        batch=config.batch_size,
        epochs=config.epochs,
        patience=config.patience,
        device=config.device,
        workers=config.workers,
        seed=config.seed,
        project=str(output.parent),
        name=output.name,
        exist_ok=True,
        resume=bool(resume),
        verbose=True,
        plots=True,
    )
    print(f"Detector training completed: {output}", flush=True)
    return results


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--model", default="yolo11n-pose.pt")
    result.add_argument("--image-size", type=int, default=3840)
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--epochs", type=int, default=200)
    result.add_argument("--patience", type=int, default=30)
    result.add_argument("--device", default="0")
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--resume")
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    config = DetectorTrainingConfig(
        model=args.model, image_size=args.image_size, batch_size=args.batch_size,
        epochs=args.epochs, patience=args.patience, device=args.device,
        workers=args.workers, seed=args.seed,
    )
    train_detector(args.dataset, args.output, config, resume=args.resume, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
