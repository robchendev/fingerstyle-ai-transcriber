"""Provision explicit local vision models with immutable receipts."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from core import EvidenceError, PRIVATE_OUTPUT_ROOT, private_output, sha256


HAND_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"


def _provision_model(output, source_url, model_name):
    try:
        import requests
    except ImportError as error:
        raise EvidenceError("Install the isolated video-evidence requirements before provisioning models.") from error
    output = private_output(output)
    receipt = output.with_suffix(output.suffix + ".receipt.json")
    if receipt.exists():
        raise EvidenceError(f"Refusing to overwrite existing model receipt: {receipt}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(f".{output.name}.{uuid4().hex}.part")
    try:
        with requests.get(source_url, stream=True, timeout=(15, 300)) as response:
            response.raise_for_status()
            with pending.open("xb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        stream.write(chunk)
        prefix = pending.read_bytes()[:16]
        if pending.stat().st_size < 1_000_000 or not 0 <= prefix.find(b"PK\x03\x04") <= 8:
            raise EvidenceError("Downloaded hand-landmarker model has an unexpected format.")
        pending.replace(output)
        record = {
            "schemaVersion": 1,
            "kind": "video-evidence-model",
            "model": model_name,
            "sourceUrl": source_url,
            "sha256": sha256(output),
            "size": output.stat().st_size,
        }
        receipt.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        return output, receipt
    finally:
        pending.unlink(missing_ok=True)


def provision_hand_landmarker(output=PRIVATE_OUTPUT_ROOT / "models" / "hand_landmarker.task"):
    return _provision_model(output, HAND_LANDMARKER_URL, "MediaPipe Hand Landmarker float16")


def provision_pose_landmarker(output=PRIVATE_OUTPUT_ROOT / "models" / "pose_landmarker_lite.task"):
    return _provision_model(output, POSE_LANDMARKER_URL, "MediaPipe Pose Landmarker Lite float16")
