"""Read immutable normalized labels and construct a nominal score-time clock."""

from bisect import bisect_right
from collections import Counter
import math

import numpy as np

from .canonical_events import fraction
from .gp_events import validate_provided_timing


CLOCK_POLICY = "Constant quarter-BPM segments; linear GP ramps use quarter-BPM linear in score position as a nominal matching reference, not a performed clock."


class AlignmentInputError(ValueError):
    """An alignment input is invalid, stale, or outside its declared scope."""


class ScoreClock:
    """Integrate the supplied tempo schedule without claiming audio alignment."""

    def __init__(self, labels, normalization):
        self.total_quarter = fraction(normalization["durationQuarter"], "score duration")
        visits = labels["measureVisits"]
        self.measure_starts = [fraction(visit["onsetQuarter"], "measure onset") for visit in visits]
        if (
            not visits or self.measure_starts[0] != 0 or self.total_quarter <= self.measure_starts[-1]
            or any(a >= b for a, b in zip(self.measure_starts, self.measure_starts[1:]))
            or any(visit["visitIndex"] != index or visit["measureIndex"] != index for index, visit in enumerate(visits))
        ):
            raise AlignmentInputError("Alignment requires a contiguous normalized linear score.")
        ends = [*self.measure_starts[1:], self.total_quarter]
        events = []
        for event in normalization["normalizedTempoEvents"]:
            index = event["measureIndex"]
            if type(index) is not int or not 0 <= index < len(visits) or type(event["linear"]) is not bool:
                raise AlignmentInputError("Invalid normalized tempo event.")
            offset = fraction(event["offsetQuarter"], "tempo offset")
            ratio = fraction(event["positionRatio"], "tempo position")
            if not 0 <= ratio <= 1 or offset != ratio * (ends[index] - self.measure_starts[index]):
                raise AlignmentInputError("Tempo anchor disagrees with its normalized bar coordinates.")
            bpm = validate_provided_timing({"bpm": event["bpm"], "beatUnit": event["beatUnit"]}, labels["conditioning"]["providedTiming"]["timeSignature"])
            if bpm != fraction(event["quarterBpm"], "quarter BPM"):
                raise AlignmentInputError("Tempo beat-unit conversion disagrees with its quarter BPM.")
            events.append((self.measure_starts[index] + offset, float(bpm), event["linear"]))
        events.sort(key=lambda event: event[0])
        if not events or events[0][0] != 0 or events[-1][0] > self.total_quarter or any(a[0] >= b[0] for a, b in zip(events, events[1:])):
            raise AlignmentInputError("The score clock needs a unique initial tempo and ordered in-range anchors.")
        self.segments = []
        elapsed = 0.0
        for index, (start, bpm, linear) in enumerate(events):
            if start == self.total_quarter:
                if linear:
                    raise AlignmentInputError("A terminal tempo anchor cannot start an unresolved ramp.")
                continue
            end = events[index + 1][0] if index + 1 < len(events) else self.total_quarter
            if linear and index + 1 == len(events):
                raise AlignmentInputError("A tempo ramp needs an explicit endpoint.")
            slope = (events[index + 1][1] - bpm) / float(end - start) if linear else 0.0
            duration = self._integrate(float(end - start), bpm, slope)
            self.segments.append({"start": float(start), "end": float(end), "bpm": bpm, "slope": slope, "seconds": elapsed, "endSeconds": elapsed + duration})
            elapsed += duration
        self.duration_seconds = elapsed
        self._quarter_starts = [segment["start"] for segment in self.segments]
        self._second_starts = [segment["seconds"] for segment in self.segments]

    @staticmethod
    def _integrate(quarters, bpm, slope):
        return 60 * quarters / bpm if abs(slope) < 1e-12 else 60 * math.log1p(slope * quarters / bpm) / slope

    def seconds(self, quarter):
        value = float(quarter)
        if not math.isfinite(value) or not 0 <= value <= float(self.total_quarter):
            raise AlignmentInputError("Score position is outside the nominal clock.")
        if value == float(self.total_quarter):
            return self.duration_seconds
        segment = self.segments[min(bisect_right(self._quarter_starts, value) - 1, len(self.segments) - 1)]
        return segment["seconds"] + self._integrate(value - segment["start"], segment["bpm"], segment["slope"])

    def quarter_at(self, seconds):
        if not math.isfinite(seconds) or not 0 <= seconds <= self.duration_seconds:
            raise AlignmentInputError("Reference time is outside the nominal clock.")
        segment = self.segments[min(bisect_right(self._second_starts, seconds) - 1, len(self.segments) - 1)]
        elapsed = seconds - segment["seconds"]
        delta = elapsed * segment["bpm"] / 60 if abs(segment["slope"]) < 1e-12 else segment["bpm"] * math.expm1(elapsed * segment["slope"] / 60) / segment["slope"]
        return min(float(self.total_quarter), segment["start"] + delta)


def matching_events(labels, clock):
    events, omitted = [], Counter()
    for note in labels["targets"]["notes"]:
        if not note["sourceSegments"] or note["sourceSegments"][0]["graceMode"] is not None or note["isAttack"] is not True:
            omitted["unknown_or_grace_attack"] += 1
            continue
        if not note["labelMask"]["pitch"] or note["soundingPitchMidi"] is None:
            omitted["unknown_pitch"] += 1
            continue
        onset = fraction(note["onsetQuarter"], note["id"])
        end = onset + fraction(note["notatedDurationQuarter"], note["id"]) if note["labelMask"]["notatedDuration"] else None
        if end is None:
            omitted["duration_uses_transient_matching_kernel"] += 1
        events.append({"onset": clock.seconds(onset), "end": clock.seconds(end) if end is not None else None, "pitch": note["soundingPitchMidi"], "percussive": False})
    for gesture in labels["targets"]["gestures"]:
        if not gesture.get("scoreOnsetKnown", False) or gesture.get("graceMode") is not None:
            omitted["unknown_gesture_onset"] += 1
            continue
        if gesture["technique"] in {"wrist_thump", "thumb_slap", "percussive_hit", "muted_strum"}:
            events.append({"onset": clock.seconds(fraction(gesture["onsetQuarter"], gesture["id"])), "end": None, "pitch": None, "percussive": True})
    if not events:
        raise AlignmentInputError("No resolved musical matching cues exist; no timing fallback is inferred.")
    return events, dict(sorted(omitted.items()))


def candidate_mapping(reference, audio, alignment, clock, *, allow_audio_prefix=False, allow_audio_suffix=False):
    reference_ids = np.asarray(alignment["reference_indices"])
    audio_ids = np.asarray(alignment["audio_indices"])
    costs = np.asarray(alignment["local_costs"], dtype=float)
    for times in (np.asarray(reference.times), np.asarray(audio.times)):
        if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0):
            raise AlignmentInputError("Matching feature times must be finite, nonnegative and strictly increasing.")
    if (
        reference_ids.ndim != 1 or audio_ids.shape != reference_ids.shape or costs.shape != reference_ids.shape or len(costs) < 2
        or reference_ids.dtype.kind not in "iu" or audio_ids.dtype.kind not in "iu"
        or not np.isfinite(costs).all() or np.any(costs < 0)
        or np.any(reference_ids < 0) or np.any(reference_ids >= len(reference.times))
        or np.any(audio_ids < 0) or np.any(audio_ids >= len(audio.times))
    ):
        raise AlignmentInputError("The aligner returned an invalid candidate path.")
    steps = np.column_stack((np.diff(reference_ids), np.diff(audio_ids)))
    if np.any(steps < 0) or np.any(steps > 1) or np.any(steps.sum(axis=1) == 0) or (not allow_audio_prefix and audio_ids[0] != 0) or (not allow_audio_suffix and audio_ids[-1] != len(audio.times) - 1):
        raise AlignmentInputError("The candidate path must monotonically account for every matched audio frame through the end.")
    counts = np.bincount(reference_ids, minlength=len(reference.times))
    present = counts > 0
    if np.count_nonzero(present) < 2:
        raise AlignmentInputError("The candidate collapsed onto a single score position.")
    times = np.bincount(reference_ids, weights=np.asarray(audio.times)[audio_ids], minlength=len(reference.times))[present] / counts[present]
    matching_costs = np.bincount(reference_ids, weights=costs, minlength=len(reference.times))[present] / counts[present]
    reference_times = np.asarray(reference.times)[present]
    present_ids = np.flatnonzero(present)
    for reference_id, audio_id in alignment.get("fixedFrameAnchors", []):
        if not np.any((reference_ids == reference_id) & (audio_ids == audio_id)):
            raise AlignmentInputError("A fixed attack anchor is absent from the candidate path.")
        times[present_ids == reference_id] = float(audio.times[audio_id])
    if np.any(np.diff(times) < 0):
        raise AlignmentInputError("Candidate score-to-audio time is not monotonic.")
    warp_regions = []
    for fixed, moving, moving_times, kind in (
        (reference_ids, audio_ids, audio.times, "long_score_position_stall"),
        (audio_ids, reference_ids, reference.times, "score_time_compression"),
    ):
        boundaries = np.r_[0, np.flatnonzero(np.diff(fixed)) + 1, len(fixed)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            duration = float(moving_times[moving[stop - 1]] - moving_times[moving[start]])
            if duration >= 2:
                warp_regions.append({
                    "kind": kind, "durationSeconds": duration,
                    "scoreQuarterStart": clock.quarter_at(float(reference.times[reference_ids[start]])),
                    "scoreQuarterEnd": clock.quarter_at(float(reference.times[reference_ids[stop - 1]])),
                    "clipSecondsStart": float(audio.times[audio_ids[start]]),
                    "clipSecondsEnd": float(audio.times[audio_ids[stop - 1]]),
                })
    return {
        "referenceSeconds": reference_times,
        "clipSeconds": times,
        "matchingCost": matching_costs,
        "scoreQuarter": np.array([clock.quarter_at(float(value)) for value in reference_times]),
        "warpRegions": warp_regions,
    }
