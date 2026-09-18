"""Align a downloaded source-video soundtrack to the already-trimmed audio."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
from uuid import uuid4

import numpy as np
from scipy.signal import correlate

from core import EvidenceError, private_output, sha256
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
    output_path = private_output(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending = output_path.with_name(f".{output_path.name}.{uuid4().hex}.part")
    try:
        pending.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        pending.replace(output_path)
    finally:
        pending.unlink(missing_ok=True)
    return output_path, report
