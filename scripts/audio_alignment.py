"""CPU matching heuristics, not note/technique recognition or approved labels."""

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks


class AlignmentError(ValueError):
    """Invalid or unsupported alignment inputs; no timing fallback is produced."""


@dataclass(frozen=True)
class Features:
    """Uniform seconds (T,), nonnegative chroma (T,12), onset/activity (T,) in [0,1]."""

    times: np.ndarray
    chroma: np.ndarray
    onset: np.ndarray
    activity: np.ndarray


_MAX_SECONDS = 900
_MAX_FRAMES = 90_000
_HARD_MAX_CELLS = 120_000_000
_TONAL_WEIGHT = 0.8


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise AlignmentError(f"{name} must be a finite real number.")
    try:
        result = float(value)
    except OverflowError as error:
        raise AlignmentError(f"{name} must be a finite real number.") from error
    if not math.isfinite(result):
        raise AlignmentError(f"{name} must be a finite real number.")
    return result


def _times(duration, hop):
    duration = _number(duration, "duration_seconds")
    hop = _number(hop, "hop_seconds")
    if not 0.01 <= hop <= 1:
        raise AlignmentError("hop_seconds must be between 0.01 and 1.")
    if not max(0.1, 2 * hop) <= duration <= _MAX_SECONDS:
        raise AlignmentError("Duration must cover at least 0.1 seconds and two hops, and at most 900 seconds.")
    count = math.ceil(duration / hop)
    if count > _MAX_FRAMES:
        raise AlignmentError(f"Feature extraction exceeds {_MAX_FRAMES} frames.")
    times = np.arange(count, dtype=np.float64) * hop
    return times[times < duration]


def _unit_rows(values):
    # Scale first so even finite, unusually large external features cannot overflow.
    peak = values.max(axis=1, keepdims=True)
    scaled = np.divide(values, peak, out=np.zeros_like(values, dtype=np.float64), where=peak > 0)
    norm = np.linalg.norm(scaled, axis=1, keepdims=True)
    return np.divide(scaled, norm, out=scaled, where=norm > 0)


def _strength(values):
    maximum = float(values.max())
    if maximum == 0:
        return np.zeros_like(values)
    scale = float(np.percentile(values[values > maximum * 0.05], 95))
    return np.clip(values / scale, 0, 1)


def _power_blocks(samples, centers, size, means, scale):
    window = np.hanning(size) / (size * scale)
    offsets = np.arange(size) - size // 2
    for start in range(0, len(centers), 64):
        indices = centers[start:start + 64, None] + offsets
        valid = (indices >= 0) & (indices < len(samples))
        np.clip(indices, 0, len(samples) - 1, out=indices)
        power = np.zeros((len(indices), size // 2 + 1), dtype=np.float64)
        for channel in range(samples.shape[1]):
            frames = samples[indices, channel].astype(np.float64)
            frames -= means[channel]
            frames *= valid
            frames *= window
            spectrum = rfft(frames, axis=1, workers=1)
            power += spectrum.real ** 2 + spectrum.imag ** 2
        # Pool channel powers, never waveforms: anti-phase stereo remains audible.
        yield start, power / samples.shape[1]


def _chroma_bank(size, sample_rate):
    frequencies = rfftfreq(size, 1 / sample_rate)
    selected = np.flatnonzero((frequencies >= 40) & (frequencies <= min(5000, sample_rate / 2)))
    pitches = 69 + 12 * np.log2(frequencies[selected] / 440)
    lower = np.floor(pitches).astype(int)
    fraction = pitches - lower
    bank = np.zeros((len(frequencies), 12), dtype=np.float64)
    # Octaves are deliberately folded. Mild frequency weighting retains low G1
    # evidence without treating harmonics, tuning deviations or timbre as notes.
    weight = np.sqrt(440 / frequencies[selected])
    bank[selected, lower % 12] = (1 - fraction) * weight
    bank[selected, (lower + 1) % 12] += fraction * weight
    return bank


def audio_features(samples, sample_rate, *, hop_seconds=0.05):
    """Extract centered, zero-padded spectral cues without changing audio timing.

    The long window is 8192 samples at 22050 Hz (low G1); an independent
    1024-sample window localizes attacks. Neither is an acoustic note detector.
    Digital silence/DC and recordings shorter than two hops/0.1 s fail explicitly.
    """
    if isinstance(sample_rate, (bool, np.bool_)) or not isinstance(sample_rate, Integral) or not 8000 <= sample_rate <= 48000:
        raise AlignmentError("sample_rate must be an integer between 8000 and 48000 Hz.")
    if (
        not isinstance(samples, np.ndarray) or samples.ndim != 2
        or samples.dtype.kind != "f" or not 1 <= samples.shape[1] <= 8
        or len(samples) == 0
    ):
        raise AlignmentError("samples must be a nonempty floating (samples, 1..8 channels) array.")
    times = _times(len(samples) / sample_rate, hop_seconds)
    if not np.isfinite(samples).all():
        raise AlignmentError("samples contain nonfinite values.")
    means = np.mean(samples, axis=0, dtype=np.float64)
    scale = max(float(np.max(np.abs(samples[:, channel].astype(np.float64) - means[channel])))
                for channel in range(samples.shape[1]))
    if not math.isfinite(scale) or scale <= 1e-8:
        raise AlignmentError("Audio is silent, DC-only, or below the 1e-8 sample-amplitude floor.")
    centers = np.rint(times * sample_rate).astype(np.int64)
    long_size = 2 ** math.ceil(math.log2(sample_rate * 8192 / 22050))
    short_size = 2 ** math.ceil(math.log2(sample_rate * 1024 / 22050))
    bank = _chroma_bank(long_size, sample_rate)
    chroma = np.empty((len(times), 12), dtype=np.float64)
    for start, power in _power_blocks(samples, centers, long_size, means, scale):
        # einsum avoids spawning BLAS threads for many small block projections.
        chroma[start:start + len(power)] = np.einsum("tf,fc->tc", np.sqrt(power), bank, optimize=False)

    onset = np.empty(len(times), dtype=np.float64)
    energy = np.empty(len(times), dtype=np.float64)
    previous = np.zeros(short_size // 2 + 1, dtype=np.float64)
    for start, power in _power_blocks(samples, centers, short_size, means, scale):
        magnitude = np.sqrt(power)
        compressed = np.log1p(100 * magnitude)
        difference = np.diff(compressed, axis=0, prepend=previous[None, :])
        onset[start:start + len(power)] = np.maximum(difference, 0).sum(axis=1)
        energy[start:start + len(power)] = np.sqrt(power.sum(axis=1))
        previous = compressed[-1]
    if float(energy.max()) <= 1e-12:
        raise AlignmentError("Audio has no usable frame energy.")
    activity = _strength(energy)
    chroma[activity < 0.02] = 0
    return Features(times, _unit_rows(chroma), _strength(onset), activity)


def reference_features(events, duration_seconds, *, hop_seconds=0.05):
    """Build matching cues only; never fill in or alter training label durations.

    Known notation intervals form a coarse constant chroma roll, NOT an acoustic
    decay/silence target. Missing ends use a 150 ms transient feature kernel.
    Unpitched/percussive events supply onset cues only. Sub-hop events are
    projected onto their nearest frame so every supplied event contributes.
    """
    times = _times(duration_seconds, hop_seconds)
    if not isinstance(events, list) or not events:
        raise AlignmentError("events must be a nonempty list of event dictionaries.")
    chroma = np.zeros((len(times), 12), dtype=np.float64)
    onset_feature = np.zeros(len(times), dtype=np.float64)
    activity = np.zeros(len(times), dtype=np.float64)
    for index, event in enumerate(events):
        name = f"events[{index}]"
        if not isinstance(event, dict) or not {"onset", "end", "pitch", "percussive"} <= event.keys():
            raise AlignmentError(f"{name} must supply onset, end, pitch and percussive.")
        onset = _number(event["onset"], f"{name}.onset")
        if not 0 <= onset < duration_seconds:
            raise AlignmentError(f"{name}.onset must lie in [0, duration_seconds).")
        end = event["end"]
        if end is not None:
            end = _number(end, f"{name}.end")
            if not onset < end <= duration_seconds:
                raise AlignmentError(f"{name}.end must follow its onset and not exceed duration_seconds.")
        pitch = event["pitch"]
        if pitch is not None:
            pitch = _number(pitch, f"{name}.pitch")
            if not 0 <= pitch <= 127:
                raise AlignmentError(f"{name}.pitch must be a MIDI pitch in [0, 127].")
        if type(event["percussive"]) is not bool:
            raise AlignmentError(f"{name}.percussive must be boolean.")
        if event["percussive"] and pitch is not None:
            raise AlignmentError(f"{name} is percussive but also supplies a pitch; split compound cues explicitly.")

        nearest = min(int(math.floor(onset / hop_seconds + 0.5)), len(times) - 1)
        width = max(0.04, hop_seconds)
        left = max(0, nearest - math.ceil(3 * width / hop_seconds))
        right = min(len(times), nearest + math.ceil(3 * width / hop_seconds) + 1)
        pulse = np.exp(-0.5 * ((times[left:right] - onset) / width) ** 2)
        onset_feature[left:right] = np.maximum(onset_feature[left:right], pulse)
        activity[left:right] = np.maximum(activity[left:right], pulse)
        if pitch is not None:
            if end is None:
                stop = min(len(times), nearest + max(1, math.ceil(0.15 / hop_seconds)))
                envelope = np.exp(-np.arange(stop - nearest) * hop_seconds / 0.075)
            else:
                stop = min(len(times), max(nearest + 1, math.ceil(end / hop_seconds)))
                envelope = np.ones(stop - nearest)
            lower = math.floor(pitch)
            fraction = pitch - lower
            chroma[nearest:stop, lower % 12] += envelope * (1 - fraction)
            chroma[nearest:stop, (lower + 1) % 12] += envelope * fraction
            activity[nearest:stop] = np.maximum(activity[nearest:stop], envelope)
    return Features(times, _unit_rows(chroma), _strength(onset_feature), np.clip(activity, 0, 1))


def _validate_features(features, name):
    if not isinstance(features, Features):
        raise AlignmentError(f"{name} must be Features.")
    arrays = (features.times, features.chroma, features.onset, features.activity)
    if any(not isinstance(value, np.ndarray) or value.dtype.kind not in "fi" for value in arrays):
        raise AlignmentError(f"{name} fields must be real numeric numpy arrays.")
    times, chroma, onset, activity = arrays
    if (
        times.ndim != 1 or not 2 <= len(times) <= _MAX_FRAMES
        or chroma.shape != (len(times), 12)
        or onset.shape != times.shape or activity.shape != times.shape
    ):
        raise AlignmentError(f"{name} requires 2..{_MAX_FRAMES} times, (T,12) chroma, and (T,) onset/activity.")
    if any(not np.isfinite(value).all() for value in arrays):
        raise AlignmentError(f"{name} contains nonfinite values.")
    differences = np.diff(times.astype(np.float64))
    if (
        times[0] != 0 or np.any(differences <= 0)
        or not np.allclose(differences, differences[0], rtol=1e-8, atol=1e-10)
        or not 0.01 <= differences[0] <= 1 or times[-1] >= _MAX_SECONDS
    ):
        raise AlignmentError(f"{name}.times must start at zero and use a uniform 0.01..1 second hop below 900 seconds.")
    if np.any(chroma < 0) or np.any(onset < 0) or np.any(onset > 1) or np.any(activity < 0) or np.any(activity > 1):
        raise AlignmentError(f"{name} requires nonnegative chroma and onset/activity in [0,1].")
    if not np.any(chroma > 0) and not np.any(onset > 0):
        raise AlignmentError(f"{name} has no tonal or onset matching evidence.")
    return _unit_rows(chroma)


def _cost(similarity, reference_present, audio_present, onset_difference):
    tonal = np.clip(1 - similarity, 0, 1)
    tonal = np.where(audio_present, tonal, 1.0)
    # An absent score pitch is unknown acoustic content, not a silence label.
    # A neutral cost also prevents score rests from being free audio sinks.
    tonal = np.where(reference_present, tonal, 0.5)
    return _TONAL_WEIGHT * tonal + (1 - _TONAL_WEIGHT) * onset_difference


def _cost_block(reference_chroma, audio_chroma, reference_onset, audio_onset):
    similarity = np.einsum("ik,jk->ij", reference_chroma, audio_chroma, optimize=False)
    return _cost(
        similarity, np.any(reference_chroma > 0, axis=1)[:, None],
        np.any(audio_chroma > 0, axis=1)[None, :],
        np.abs(reference_onset[:, None] - audio_onset[None, :]),
    )


def _dtw_row(cost, previous, penalty, free_start):
    diagonal = np.empty_like(previous)
    diagonal[0] = 0 if free_start else np.inf
    diagonal[1:] = previous[:-1]
    vertical = previous + penalty
    entry = cost + np.minimum(diagonal, vertical)
    # D[j] = min(entry[j], D[j-1] + cost[j] + penalty). Subtracting
    # cumulative horizontal costs turns this into a C-level cumulative minimum.
    prefix = np.cumsum(cost + penalty)
    transformed = entry - prefix
    minimum = np.minimum.accumulate(transformed)
    prior_minimum = np.concatenate(([np.inf], minimum[:-1]))
    directions = np.where(transformed <= prior_minimum, np.where(diagonal <= vertical, 0, 1), 2)
    return prefix + minimum, directions.astype(np.uint8)


def _longest_run(steps):
    edges = np.diff(np.concatenate(([False], steps, [False])).astype(np.int8))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return int(lengths.max()) if len(lengths) else 0


def _information(chroma, onset):
    active = onset >= 0.4
    onset_regions = int(np.count_nonzero(active & ~np.concatenate(([False], active[:-1]))))
    changes = int(np.count_nonzero(np.linalg.norm(np.diff(chroma, axis=0), axis=1) >= 0.35))
    return onset_regions, changes


def align_first_attack(reference, audio, *, first_reference_seconds, max_cells=60_000_000, warp_penalty=0.08):
    """Anchor the earliest plausible attack, then match the remaining sequence.

    Spectral-flux peaks, activity and first-event pitch evidence propose an attack;
    this is not a musical approval or a waveform trim. Quiet/ambiguous openings
    can require human correction. Unmatched prefixes never become silence labels.
    """
    reference_chroma = _validate_features(reference, "reference")
    audio_chroma = _validate_features(audio, "audio")
    first_reference_seconds = _number(first_reference_seconds, "first_reference_seconds")
    if not 0 <= first_reference_seconds <= reference.times[-1]:
        raise AlignmentError("First notated attack is outside the score reference.")
    reference_index = int(np.argmin(np.abs(reference.times - first_reference_seconds)))
    if reference_index >= len(reference.times) - 1 or reference.onset[reference_index] < .1:
        raise AlignmentError("First notated attack has no usable reference onset cue.")
    peaks, _ = find_peaks(np.r_[0., audio.onset, 0.], height=.12, prominence=.06, distance=2)
    peaks = peaks - 1
    peaks = peaks[(peaks >= 0) & (peaks < len(audio.times) - 1)]
    present = bool(np.any(reference_chroma[reference_index] > 0))
    similarity = audio_chroma[peaks] @ reference_chroma[reference_index]
    eligible = (audio.activity[peaks] >= .025) & (audio.times[peaks] <= 15)
    if present:
        eligible &= similarity >= .35
    matches = peaks[eligible]
    if not len(matches):
        raise AlignmentError("No plausible first attack in the first 15 seconds; an explicit reviewed anchor is needed.")
    audio_index = int(matches[0])

    def suffix(features, start):
        return Features(features.times[start:] - features.times[start], features.chroma[start:], features.onset[start:], features.activity[start:])

    result = align_features(suffix(reference, reference_index), suffix(audio, audio_index), max_cells=max_cells, warp_penalty=warp_penalty)
    result["reference_indices"] += reference_index
    result["audio_indices"] += audio_index
    result["fixedFrameAnchors"] = [(reference_index, audio_index)]
    result["diagnostics"].update(
        algorithm="first_attack_then_full_dtw",
        mode="first-attack",
        reference_frame_range=[reference_index, len(reference.times) - 1],
        audio_frame_range=[audio_index, len(audio.times) - 1],
        reference_coverage_fraction=(len(reference.times) - reference_index) / len(reference.times),
        audio_coverage_fraction=(len(audio.times) - audio_index) / len(audio.times),
        first_attack_reference_seconds=float(reference.times[reference_index]),
        first_attack_audio_seconds=float(audio.times[audio_index]),
        first_attack_onset_strength=float(audio.onset[audio_index]),
        first_attack_pitch_similarity=float(audio_chroma[audio_index] @ reference_chroma[reference_index]) if present else None,
        first_attack_is_human_approved=False,
    )
    if audio_index:
        result["diagnostics"]["flags"].append("unmatched_audio_lead_in")
    return result


def align_attack_bounds(reference, audio, *, first_reference_seconds, last_reference_seconds, max_cells=60_000_000, warp_penalty=0.08):
    """Match the attack span without stretching notation into post-attack decay."""
    reference_chroma = _validate_features(reference, "reference")
    audio_chroma = _validate_features(audio, "audio")
    last_reference_seconds = _number(last_reference_seconds, "last_reference_seconds")
    if not first_reference_seconds < last_reference_seconds <= reference.times[-1]:
        raise AlignmentError("The last notated attack must follow the first and lie within the reference.")
    last_reference = int(np.argmin(np.abs(reference.times - last_reference_seconds)))
    peaks, _ = find_peaks(np.r_[0., audio.onset, 0.], height=.12, prominence=.06, distance=2)
    peaks = peaks - 1
    peaks = peaks[(peaks > 0) & (peaks < len(audio.times))]
    compatible = audio.activity[peaks] >= .025
    if np.any(reference_chroma[last_reference] > 0):
        compatible &= audio_chroma[peaks] @ reference_chroma[last_reference] >= .35
    peaks = peaks[compatible]
    if not len(peaks):
        raise AlignmentError("No plausible final attack; no audio-end fallback was used.")
    last_audio = int(peaks[-1])

    def prefix(features, stop):
        return Features(features.times[:stop + 1], features.chroma[:stop + 1], features.onset[:stop + 1], features.activity[:stop + 1])

    result = align_first_attack(prefix(reference, last_reference), prefix(audio, last_audio), first_reference_seconds=first_reference_seconds, max_cells=max_cells, warp_penalty=warp_penalty)
    result["fixedFrameAnchors"].append((last_reference, last_audio))
    first_reference, first_audio = int(result["reference_indices"][0]), int(result["audio_indices"][0])
    result["diagnostics"].update(
        algorithm="first_and_last_attack_then_dtw", mode="attack-span",
        reference_frame_range=[first_reference, last_reference], audio_frame_range=[first_audio, last_audio],
        reference_coverage_fraction=(last_reference - first_reference + 1) / len(reference.times),
        audio_coverage_fraction=(last_audio - first_audio + 1) / len(audio.times),
        last_attack_reference_seconds=float(reference.times[last_reference]),
        last_attack_audio_seconds=float(audio.times[last_audio]),
        last_attack_is_human_approved=False,
        unmatched_audio_after_last_attack_seconds=float(audio.times[-1] - audio.times[last_audio]),
    )
    result["diagnostics"]["flags"].append("post_attack_sustain_unmapped")
    return result


def align_features(reference, audio, *, mode="global", max_cells=60_000_000, warp_penalty=0.08):
    """Return an unapproved monotonic DTW candidate, without proportional fallback.

    Subsequence mode consumes ALL audio, allowing free score prefix/suffix.
    Exact floating-point ties prefer diagonal, score advance, then audio advance;
    subsequence endpoint ties choose the earliest score end. Warping is penalized,
    not slope-constrained. Diagnostics expose stalls instead of approving them.
    Memory is one uint8 backpointer/cell plus bounded float64 blocks and rows,
    not a dense float cost matrix. Work is O(score frames * audio frames).
    """
    if not isinstance(mode, str) or mode not in ("global", "subsequence"):
        raise AlignmentError("mode must be 'global' or 'subsequence'.")
    if isinstance(max_cells, (bool, np.bool_)) or not isinstance(max_cells, Integral) or not 1 <= max_cells <= _HARD_MAX_CELLS:
        raise AlignmentError(f"max_cells must be an integer in [1, {_HARD_MAX_CELLS}].")
    warp_penalty = _number(warp_penalty, "warp_penalty")
    if not 0 <= warp_penalty <= 1:
        raise AlignmentError("warp_penalty must lie in [0,1].")
    reference_chroma = _validate_features(reference, "reference")
    audio_chroma = _validate_features(audio, "audio")
    hop = float(reference.times[1] - reference.times[0])
    if not np.isclose(hop, audio.times[1] - audio.times[0], rtol=1e-8, atol=1e-10):
        raise AlignmentError("Reference and audio must use the same hop.")
    n, m = len(reference.times), len(audio.times)
    cells = n * m
    if cells > max_cells:
        raise AlignmentError(f"DTW needs {cells} cells, exceeding max_cells={max_cells}; no fallback was run.")
    directions = np.empty((n, m), dtype=np.uint8)
    previous = np.full(m, np.inf, dtype=np.float64)
    endpoints = np.empty(n, dtype=np.float64)
    block_size = max(1, min(64, 1_000_000 // m))
    for start in range(0, n, block_size):
        block = _cost_block(reference_chroma[start:start + block_size], audio_chroma,
                            reference.onset[start:start + block_size], audio.onset)
        for offset, cost in enumerate(block):
            i = start + offset
            previous, directions[i] = _dtw_row(cost, previous, warp_penalty, mode == "subsequence" or i == 0)
            endpoints[i] = previous[-1]
    i = n - 1 if mode == "global" else int(np.argmin(endpoints))
    j = m - 1
    objective = float(endpoints[i])
    if not math.isfinite(objective):
        raise AlignmentError("DTW produced no finite complete path.")
    reference_indices = np.empty(n + m, dtype=np.int64)
    audio_indices = np.empty(n + m, dtype=np.int64)
    length = 0
    while i >= 0 and j >= 0:
        reference_indices[length], audio_indices[length] = i, j
        length += 1
        direction = directions[i, j]
        if direction != 2:
            i -= 1
        if direction != 1:
            j -= 1
    reference_indices = reference_indices[:length][::-1].copy()
    audio_indices = audio_indices[:length][::-1].copy()
    if j != -1 or mode == "global" and i != -1:
        raise AlignmentError("DTW backtracking failed to cover the required boundaries.")
    r, a = reference_chroma[reference_indices], audio_chroma[audio_indices]
    similarity = np.clip(np.einsum("ij,ij->i", r, a), 0, 1)
    tonal_frames = np.any(r > 0, axis=1)
    local_costs = _cost(similarity, tonal_frames, np.any(a > 0, axis=1),
                        np.abs(reference.onset[reference_indices] - audio.onset[audio_indices]))
    mean_cost = float(local_costs.mean())
    reference_steps, audio_steps = np.diff(reference_indices), np.diff(audio_indices)
    warped = (reference_steps == 0) | (audio_steps == 0)
    warp_fraction = float(np.mean(warped))
    reference_stall = _longest_run(reference_steps == 0) * hop
    audio_stall = _longest_run(audio_steps == 0) * hop
    first, last = int(reference_indices[0]), int(reference_indices[-1])
    ratio = (last - first) / (m - 1)
    degenerate = (
        ratio < 0.25 or ratio > 4 or warp_fraction > 0.75
        or reference_stall > max(1, 0.03 * (m - 1) * hop)
        or audio_stall > max(1, 0.03 * (last - first) * hop)
    )
    reference_information = _information(reference_chroma[first:last + 1], reference.onset[first:last + 1])
    audio_information = _information(audio_chroma, audio.onset)
    uninformative = (
        max(reference_information) <= 1 or max(audio_information) <= 1
        or np.mean(audio.activity >= 0.05) < 0.05
    )
    tonal_mismatch = float(np.mean(1 - similarity[tonal_frames])) if np.any(tonal_frames) else None
    poor_match = mean_cost > 0.5 or tonal_mismatch is not None and tonal_mismatch > 0.55
    flags = [name for name, present in (
        ("degenerate_path", degenerate), ("uninformative_features", uninformative), ("high_mismatch", poor_match),
    ) if present]
    diagnostics = {
        "algorithm": "full_dtw_cumulative_minimum",
        "mode": mode,
        "candidate_only": True,
        "requires_human_review": True,
        "cost_is_probability": False,
        "quality": "flagged" if flags else "unreviewed_candidate",
        "flags": flags,
        "degenerate": bool(degenerate),
        "uninformative": bool(uninformative),
        "poor_match": bool(poor_match),
        "cells": cells,
        "path_length": length,
        "reference_frame_range": [first, last],
        "audio_frame_range": [0, m - 1],
        "reference_coverage_fraction": (last - first + 1) / n,
        "audio_coverage_fraction": 1.0,
        "score_seconds_per_audio_second": ratio,
        "non_diagonal_step_fraction": warp_fraction,
        "max_reference_stall_seconds": reference_stall,
        "max_audio_stall_seconds": audio_stall,
        "objective_cost": objective,
        "warp_penalty": warp_penalty,
        "mean_tonal_mismatch": tonal_mismatch,
        "tonal_reference_path_fraction": float(np.mean(tonal_frames)),
        "local_cost_quantiles": [float(value) for value in np.quantile(local_costs, [0.1, 0.5, 0.9])],
        "reference_onset_regions": reference_information[0],
        "reference_chroma_changes": reference_information[1],
        "audio_onset_regions": audio_information[0],
        "audio_chroma_changes": audio_information[1],
    }
    return {
        "reference_indices": reference_indices,
        "audio_indices": audio_indices,
        "local_costs": local_costs,
        "mean_cost": mean_cost,
        "diagnostics": diagnostics,
    }
