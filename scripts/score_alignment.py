"""Read immutable normalized labels and construct a nominal score-time clock."""

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path

from .canonical_events import fraction
from .catalogs import read_json, sha256
from .download_audio import sample_bounds
from .gp_events import validate_provided_timing
from .gp_normalization import NORMALIZATION_VERSION
from .normalize_gp import _metadata_paths, normalized_output_paths, validate_training_policy
from .prepare_labels import mapped_path


CLOCK_POLICY = "Constant quarter-BPM segments; linear GP ramps use quarter-BPM linear in score position as a nominal matching reference, not a performed clock."


class AlignmentInputError(ValueError):
    """An alignment input is invalid, stale, or outside its declared scope."""


@dataclass(frozen=True)
class AlignmentInput:
    entry: dict
    labels: dict
    normalization: dict
    audio_path: Path
    source_hashes: dict
    source_first_sample: int

    @property
    def source_offset_seconds(self):
        return self.source_first_sample / self.entry["audioAsset"]["sampleRate"]


def load_alignment_input(paths, identifier):
    catalog = paths.load_catalog()
    matches = [entry for entry in catalog["entries"] if entry["id"] == identifier]
    if len(matches) != 1:
        raise AlignmentInputError(f"{identifier}: expected one active catalog entry.")
    entry = matches[0]
    if identifier in paths.exclusions(catalog) or entry["source"] == "Produced" or entry["isBundle"]:
        raise AlignmentInputError(f"{identifier}: excluded recordings cannot enter alignment.")
    gp_path, canonical_path = normalized_output_paths(paths, identifier)
    manifest_path = paths.normalized_gp / f"{identifier}.manifest.json"
    manifest_hash = sha256(manifest_path)
    manifest = read_json(manifest_path)
    if manifest.get("status") != "completed" or manifest.get("inputMetadataUnchanged") is not True or manifest.get("trainingPolicyValidated") is not True or len(manifest.get("entries", [])) != 1:
        raise AlignmentInputError(f"{identifier}: normalization did not publish a successful score manifest.")
    expected_metadata = {str(path.relative_to(paths.root)).replace("/", "\\") for path in _metadata_paths(paths)}
    if set(manifest.get("inputMetadataSha256", {})) != expected_metadata:
        raise AlignmentInputError(f"{identifier}: incomplete normalization metadata provenance.")
    row = manifest["entries"][0]
    if row.get("status") != "normalized" or row.get("catalogId") != identifier or row.get("performerId") != paths.performer or row.get("normalizationVersion") != NORMALIZATION_VERSION:
        raise AlignmentInputError(f"{identifier}: wrong or stale normalized score identity.")
    policy_path = paths.root / "data" / "training-notation-policy.json"
    validate_training_policy(read_json(policy_path))
    source_hashes = {}

    def bind(path, expected=None):
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise AlignmentInputError(f"{identifier}: stale input {path.name}.")
        source_hashes[str(path.relative_to(paths.root)).replace("/", "\\")] = digest

    bind(manifest_path, manifest_hash)
    for relative, expected in manifest["inputMetadataSha256"].items():
        path = mapped_path(relative, paths, paths.root)
        bind(path, expected)
    bind(policy_path, row["trainingNotationPolicySha256"])
    for path, prefix in ((gp_path, "normalizedGp"), (canonical_path, "normalizedCanonical")):
        if mapped_path(row[prefix + "Path"], paths, paths.normalized_gp) != path.resolve():
            raise AlignmentInputError(f"{identifier}: normalized derivative path disagrees with its manifest.")
        bind(path, row[prefix + "Sha256"])
    audio_path = mapped_path(entry["localAudioPath"], paths, paths.audio)
    for path, digest in (
        (audio_path, entry["audioAsset"]["sha256"]),
        (mapped_path(entry["localGpPath"], paths, paths.gp), entry["gpExtraction"]["sourceGpSha256"]),
        (mapped_path(entry["gpExtraction"]["eventPath"], paths, paths.gp / "events"), entry["gpExtraction"]["eventSha256"]),
        (mapped_path(row["rawCanonicalPath"], paths, paths.gp / "canonical"), row["rawCanonicalSha256"]),
    ):
        bind(path, digest)
    if row["rawGpSha256"] != entry["gpExtraction"]["sourceGpSha256"] or row["rawEventSha256"] != entry["gpExtraction"]["eventSha256"] or row["rawAudioSha256"] != entry["audioAsset"]["sha256"] or row["rawAudioRangeSeconds"] != entry["rangeSeconds"]:
        raise AlignmentInputError(f"{identifier}: normalized provenance differs from the current recording or score.")
    labels = read_json(canonical_path)
    if labels.get("catalogId") != identifier or labels.get("performerId") != paths.performer or labels["provenance"]["sourceGpSha256"] != row["normalizedGpSha256"]:
        raise AlignmentInputError(f"{identifier}: canonical labels belong to another normalized score.")
    if labels.get("audioAlignment") is not None or not labels.get("scoreTimingResolved") or labels.get("timeUnit") != "quarter-note":
        raise AlignmentInputError(f"{identifier}: expected resolved, unaligned quarter-note labels.")
    if labels["audio"]["sha256"] != entry["audioAsset"]["sha256"] or entry["audioAsset"]["appliedRangeSeconds"] != entry["rangeSeconds"]:
        raise AlignmentInputError(f"{identifier}: audio/label cropping provenance is stale.")
    bounds = sample_bounds(entry["rangeSeconds"], entry["audioAsset"]["sampleRate"])
    if bounds and bounds[1] - bounds[0] != entry["audioAsset"]["sampleCount"]:
        raise AlignmentInputError(f"{identifier}: inclusive source bounds disagree with the retained sample count.")
    assert_inputs_current(paths.root, source_hashes)
    return AlignmentInput(entry, labels, row, audio_path, source_hashes, bounds[0] if bounds else 0)


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


def assert_inputs_current(root, hashes):
    for relative, expected in hashes.items():
        if sha256(root / relative) != expected:
            raise AlignmentInputError(f"Alignment input changed during processing: {relative}")
