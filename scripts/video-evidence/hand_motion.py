"""Source-clock and bounded-clip validation for numerical hand inputs."""

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction
import json
import math
from pathlib import Path

import av
import numpy as np

from core import ClockMapping, EvidenceError, sha256


def validate_clips(clips, pts, shots, *, allow_partial_end=False):
    if not isinstance(clips, list) or not clips:
        raise EvidenceError("Specify at least one bounded clip.")
    bounds = set(int(value) for value in pts)
    end_bounds = bounds | {shots["shots"][-1]["endPtsExclusive"]}
    selected = np.zeros(len(pts), bool)
    clip_ids = np.full(len(pts), -1, np.int32)
    for index, clip in enumerate(clips):
        if not isinstance(clip, dict) or set(clip) != {"label", "startPts", "endPtsExclusive"}:
            raise EvidenceError("Each clip needs label, startPts and endPtsExclusive.")
        start, end = clip["startPts"], clip["endPtsExclusive"]
        if not isinstance(clip["label"], str) or not clip["label"].strip() or type(start) is not int or type(end) is not int or start >= end:
            raise EvidenceError("Clips need a nonempty label and increasing integer PTS boundaries.")
        if start not in bounds or (end not in end_bounds and not allow_partial_end):
            raise EvidenceError("Clip bounds must match source PTS or the final end-exclusive boundary.")
        mask = (pts >= start) & (pts < end)
        if (selected & mask).any() or mask.sum() < 3:
            raise EvidenceError("Clips must not overlap and must contain at least three frames.")
        selected |= mask
        clip_ids[mask] = index
    return selected, clip_ids


def normalize_audio_asset(asset):
    if not isinstance(asset, dict):
        raise EvidenceError("Missing retained audio extraction provenance.")
    for name in ("sampleRate", "channels", "sampleCount"):
        if type(asset.get(name)) is not int or asset[name] < 1:
            raise EvidenceError(f"Audio provenance requires a positive integer {name}.")
    for name in ("sha256", "sourceSha256"):
        value = asset.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise EvidenceError(f"Audio provenance requires {name}.")
    if "appliedRangeSeconds" not in asset:
        raise EvidenceError("Audio provenance must explicitly record its range or null for full source.")
    selection = asset["appliedRangeSeconds"]
    first, stop = 0, asset["sampleCount"]
    if selection is not None:
        if not isinstance(selection, list) or len(selection) != 2 or any(type(value) not in (int, float) for value in selection):
            raise EvidenceError("Retained source range needs two finite inclusive endpoints.")
        start, end = (Decimal(str(value)) for value in selection)
        if not start.is_finite() or not end.is_finite() or start < 0 or end < start:
            raise EvidenceError("Retained source range is invalid.")
        first = int((start * asset["sampleRate"]).to_integral_value(rounding=ROUND_CEILING))
        stop = int((end * asset["sampleRate"]).to_integral_value(rounding=ROUND_FLOOR)) + 1
        if stop - first != asset["sampleCount"]:
            raise EvidenceError("Retained source range does not match the recorded audio sample count.")
    bounds = {"startSample": first, "stopSampleExclusive": stop}
    if "sourceSampleBounds" in asset:
        saved = asset["sourceSampleBounds"]
        if not isinstance(saved, dict) or saved != bounds or any(type(value) is not int for value in saved.values()):
            raise EvidenceError("Saved source sample bounds disagree with the retained range.")
    return {**asset, "sourceSampleBounds": bounds}


def load_audio_clock(video_hash, audio_path, alignment_path):
    audio_path, alignment_path = Path(audio_path), Path(alignment_path)
    hashes = {"trimmedAudio": sha256(audio_path), "alignment": sha256(alignment_path)}
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    if alignment.get("kind") != "video-to-trimmed-audio-alignment" or type(alignment.get("schemaVersion")) is not int or alignment["schemaVersion"] != 1:
        raise EvidenceError("Paired inputs require a supported video/audio alignment report.")
    if alignment.get("videoSha256") != video_hash or alignment.get("trimmedAudioSha256") != hashes["trimmedAudio"]:
        raise EvidenceError("Alignment report does not match source video and trimmed audio.")
    if alignment.get("status") != "supported" or alignment.get("rate") != [1, 1] or any(type(value) is not int for value in alignment["rate"]):
        raise EvidenceError("Paired inputs require a supported unit-rate audio alignment.")
    offset = alignment.get("videoStartSecondsForTrimmedAudioZero")
    if type(offset) not in (int, float) or not math.isfinite(offset):
        raise EvidenceError("Alignment offset must be finite.")
    asset = None
    if alignment.get("method") == "retained-source-samples":
        provenance = alignment.get("sourceProvenance")
        if not isinstance(provenance, dict):
            raise EvidenceError("Retained-source alignment requires its extraction provenance.")
        asset = normalize_audio_asset(provenance.get("audioAsset"))
        if (asset["sha256"] != hashes["trimmedAudio"] or alignment.get("correlation") is not None
                or offset != asset["sourceSampleBounds"]["startSample"] / asset["sampleRate"]):
            raise EvidenceError("Retained-source alignment disagrees with its saved audio samples.")
    else:
        correlation = alignment.get("correlation")
        if type(correlation) not in (int, float) or not math.isfinite(correlation) or not -1 <= correlation <= 1:
            raise EvidenceError("Alignment correlation must be finite and between -1 and 1.")
    with av.open(str(audio_path)) as container:
        if len(container.streams.audio) != 1:
            raise EvidenceError("Paired inputs require exactly one trimmed-audio stream.")
        stream = container.streams.audio[0]
        if not stream.sample_rate or stream.duration is None or stream.time_base is None:
            raise EvidenceError("Trimmed audio must expose its sample rate and duration.")
        sample_rate = stream.sample_rate
        duration = Fraction(stream.duration) * stream.time_base
    if asset is not None and (sample_rate != asset["sampleRate"] or duration * sample_rate != asset["sampleCount"]):
        raise EvidenceError("Retained-source clock differs from actual audio sample rate or count.")
    offset_samples = -asset["sourceSampleBounds"]["startSample"] if asset is not None else round(-Fraction(str(offset)) * sample_rate)
    clock = ClockMapping(sample_rate, offset_samples)
    return clock, duration, alignment, hashes
