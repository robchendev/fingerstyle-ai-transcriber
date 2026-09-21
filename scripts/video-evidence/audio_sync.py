"""Align a downloaded source-video soundtrack to the already-trimmed audio."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.signal import correlate

from catalog import check_audio_asset, source_for_pair, source_video_id
from core import EvidenceError, publish_json, sha256
from intake import ffmpeg_executables


def _decode_mono(path, sample_rate):
    ffmpeg, _ = ffmpeg_executables()
    command = [
        str(ffmpeg), "-v", "error", "-i", str(path), "-map", "0:a:0",
        "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-",
    ]
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise EvidenceError(f"Unable to decode audio for synchronization: {message}")
    samples = np.frombuffer(result.stdout, dtype="<f4").astype(np.float64)
    if not len(samples) or not np.isfinite(samples).all() or np.max(np.abs(samples)) <= 1e-8:
        raise EvidenceError("Synchronization audio is empty, silent, or nonfinite.")
    return samples


def _envelope(samples, sample_rate, envelope_rate):
    block = sample_rate // envelope_rate
    count = len(samples) // block
    if block < 1 or count < 100:
        raise EvidenceError("Synchronization audio is too short for the requested envelope rate.")
    values = np.abs(samples[:count * block]).reshape(count, block)
    return np.sqrt(np.mean(values * values, axis=1))


def _normalized_valid_correlation(source, clip):
    if len(source) < len(clip):
        raise EvidenceError("Trimmed audio is longer than the source-video soundtrack.")
    target = clip - clip.mean()
    target_energy = float(np.dot(target, target))
    if target_energy <= 1e-12:
        raise EvidenceError("Trimmed audio envelope has insufficient variation.")
    numerator = correlate(source, target, mode="valid", method="fft")
    cumulative = np.concatenate(([0.], np.cumsum(source)))
    cumulative_square = np.concatenate(([0.], np.cumsum(source * source)))
    window_sum = cumulative[len(clip):] - cumulative[:-len(clip)]
    window_square = cumulative_square[len(clip):] - cumulative_square[:-len(clip)]
    source_energy = window_square - window_sum * window_sum / len(clip)
    denominator = np.sqrt(np.maximum(source_energy * target_energy, 1e-24))
    return numerator / denominator


def load_source_provenance(mapping_path, receipt_path, pair_id, video_path, audio_path):
    mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or type(mapping.get("schemaVersion")) is not int or mapping["schemaVersion"] != 2 or mapping.get("kind") != "active-pair-video-sources":
        raise EvidenceError("Retained trims require a version2 source mapping; migrate the archived catalog into a new file.")
    url = source_for_pair(mapping, pair_id)
    row = next(row for row in mapping["pairs"] if row["pairId"] == pair_id)
    asset = check_audio_asset(row.get("audioAsset"), audio_path)
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    if (not isinstance(receipt, dict) or receipt.get("kind") != "authorized-source-video" or type(receipt.get("schemaVersion")) is not int or receipt["schemaVersion"] != 1
            or receipt.get("videoSha256") != sha256(video_path)
            or receipt.get("videoFile") != Path(video_path).name):
        raise EvidenceError("Source-video receipt does not bind the supplied video.")
    if not isinstance(receipt.get("metadata"), dict) or receipt["metadata"].get("id") != asset["sourceVideoId"] or source_video_id(url) != asset["sourceVideoId"]:
        raise EvidenceError("Source-video receipt ID differs from the retained audio's original video.")
    return {
        "pairId": pair_id, "mappingSha256": sha256(mapping_path),
        "videoReceiptSha256": sha256(receipt_path), "audioAsset": asset,
    }


def align_retained_source(video_path, audio_path, output_path, *, mapping_path, receipt_path, pair_id):
    provenance = load_source_provenance(mapping_path, receipt_path, pair_id, video_path, audio_path)
    asset = provenance["audioAsset"]
    first = asset["sourceSampleBounds"]["startSample"]
    report = {
        "schemaVersion": 1, "kind": "video-to-trimmed-audio-alignment", "visibility": "private",
        "videoSha256": sha256(video_path), "trimmedAudioSha256": asset["sha256"],
        "method": "retained-source-samples", "sourceProvenance": provenance,
        "sampleRate": asset["sampleRate"], "trimmedAudioDurationSeconds": asset["sampleCount"] / asset["sampleRate"],
        "videoStartSecondsForTrimmedAudioZero": first / asset["sampleRate"],
        "mapping": "trimmedAudioSample = sourceTimeSeconds * sampleRate - retainedStartSample",
        "correlation": None, "status": "supported", "reviewRequired": False,
        "rate": [1, 1], "trainingPerformed": False,
    }
    return publish_json(output_path, report), report


def align_soundtrack(video_path, trimmed_audio_path, output_path, *, sample_rate=8000, envelope_rate=200):
    video_path, trimmed_audio_path = Path(video_path), Path(trimmed_audio_path)
    if not video_path.is_file() or not trimmed_audio_path.is_file():
        raise EvidenceError("Soundtrack alignment requires local video and trimmed-audio files.")
    if type(sample_rate) is not int or type(envelope_rate) is not int or sample_rate < 1000 or envelope_rate < 20 or sample_rate % envelope_rate:
        raise EvidenceError("Synchronization rates must be compatible positive integers.")
    source = _envelope(_decode_mono(video_path, sample_rate), sample_rate, envelope_rate)
    clip = _envelope(_decode_mono(trimmed_audio_path, sample_rate), sample_rate, envelope_rate)
    scores = _normalized_valid_correlation(source, clip)
    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])
    exclusion = max(1, round(2 * envelope_rate))
    competing = np.array(scores, copy=True)
    competing[max(0, best_index - exclusion):min(len(competing), best_index + exclusion + 1)] = -np.inf
    second_score = None if not np.isfinite(competing).any() else float(np.max(competing))
    ambiguity = None if second_score is None else second_score / max(best_score, 1e-12)
    offset_seconds = best_index / envelope_rate
    status = "supported" if best_score >= .75 and (ambiguity is None or ambiguity <= .90) else "ambiguous"
    report = {
        "schemaVersion": 1,
        "kind": "video-to-trimmed-audio-alignment",
        "visibility": "private",
        "videoSha256": sha256(video_path),
        "trimmedAudioSha256": sha256(trimmed_audio_path),
        "sampleRate": sample_rate,
        "envelopeRate": envelope_rate,
        "videoSoundtrackDurationSeconds": len(source) / envelope_rate,
        "trimmedAudioDurationSeconds": len(clip) / envelope_rate,
        "videoStartSecondsForTrimmedAudioZero": offset_seconds,
        "mapping": "trimmedAudioSeconds = videoSeconds - videoStartSecondsForTrimmedAudioZero",
        "correlation": best_score,
        "secondCorrelationOutsideTwoSeconds": second_score,
        "ambiguityRatio": ambiguity,
        "status": status,
        "reviewRequired": True,
        "rate": [1, 1],
        "trainingPerformed": False,
    }
    return publish_json(output_path, report), report
