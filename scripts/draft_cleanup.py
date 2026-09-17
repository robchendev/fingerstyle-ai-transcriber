"""Deterministic cleanup for broad model hypotheses before notation export."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import math

from .transcriber_audio import HarnessError


@dataclass(frozen=True)
class DraftProfile:
    note_threshold: float = 0.9
    percussion_threshold: float = 0.6
    harmonic_threshold: float = 0.8
    include_harmonics: bool = False
    chord_tolerance_seconds: float = 0.04
    same_string_gap_seconds: float = 0.08

    def __post_init__(self):
        if type(self.include_harmonics) is not bool:
            raise TypeError("include_harmonics must be boolean.")
        for name in ("note_threshold", "percussion_threshold", "harmonic_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be from zero through one.")
        for name in ("chord_tolerance_seconds", "same_string_gap_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")


def _confidence(event):
    value = event.get("confidence")
    if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise HarnessError("Draft cleanup requires finite zero-to-one event confidence.")
    return value


def _onset(event):
    value = event.get("onsetSeconds")
    if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise HarnessError("Draft cleanup requires finite nonnegative event times.")
    return value


def _time_clusters(events, tolerance):
    groups = []
    for event in sorted(events, key=lambda item: (_onset(item), -_confidence(item))):
        if groups and _onset(event) - _onset(groups[-1][0]) <= tolerance:
            groups[-1].append(event)
        else:
            groups.append([event])
    return groups


def _representative_time(events):
    ordered = sorted(events, key=lambda item: (_onset(item), -_confidence(item)))
    total = sum(_confidence(event) for event in ordered)
    position = 0
    for event in ordered:
        position += _confidence(event)
        if position * 2 >= total:
            return _onset(event)
    return _onset(ordered[-1])


def _cluster_notes(notes, tolerance, removed, onset_adjustments):
    result = []
    for cluster in _time_clusters(notes, tolerance):
        time = _representative_time(cluster)
        strings = {}
        for note in cluster:
            string = note.get("string")
            if type(string) is not int or not 1 <= string <= 6:
                raise HarnessError("Draft cleanup requires physical note strings from one through six.")
            current = strings.get(string)
            if current is None or (_confidence(note), -_onset(note)) > (_confidence(current), -_onset(current)):
                if current is not None:
                    removed.append({"reason": "same_string_chord_cluster", "event": current})
                strings[string] = note
            else:
                removed.append({"reason": "same_string_chord_cluster", "event": note})
        for note in strings.values():
            value = deepcopy(note)
            value["onsetSeconds"] = time
            if time != _onset(note):
                onset_adjustments.append({
                    "kind": "note_chord_cluster",
                    "string": note["string"],
                    "fromSeconds": _onset(note),
                    "toSeconds": time,
                })
            result.append(value)
    return sorted(result, key=lambda item: (_onset(item), -item["string"]))


def _suppress_same_string(notes, gap, removed):
    result = []
    for string in range(1, 7):
        accepted = []
        for note in [item for item in notes if item["string"] == string]:
            if not accepted or _onset(note) - _onset(accepted[-1]) >= gap:
                accepted.append(note)
                continue
            current = accepted[-1]
            if (_confidence(note), -_onset(note)) > (_confidence(current), -_onset(current)):
                accepted[-1] = note
                removed.append({"reason": "rapid_same_string_duplicate", "event": current, "keptOnsetSeconds": _onset(note)})
            else:
                removed.append({"reason": "rapid_same_string_duplicate", "event": note, "keptOnsetSeconds": _onset(current)})
        result.extend(accepted)
    return sorted(result, key=lambda item: (_onset(item), -item["string"]))


def _clean_harmonics(notes, profile):
    removed = []
    for note in notes:
        harmonic = note.get("harmonic")
        if harmonic is None:
            continue
        score = harmonic.get("confidence")
        uncertainty = note.get("uncertainty", [])
        consistent = not any(isinstance(value, str) and "mismatch" in value for value in uncertainty)
        if isinstance(score, bool) or type(score) not in (int, float) or not math.isfinite(score):
            raise HarnessError("Draft cleanup requires finite harmonic confidence.")
        if not profile.include_harmonics or score < profile.harmonic_threshold or not consistent:
            removed.append({
                "onsetSeconds": _onset(note),
                "string": note["string"],
                "harmonic": deepcopy(harmonic),
                "reason": (
                    "disabled_uncalibrated_harmonic_head" if not profile.include_harmonics
                    else "below_threshold" if score < profile.harmonic_threshold
                    else "pitch_fret_harmonic_mismatch"
                ),
                "ordinaryNoteResolution": "Preserve predicted sounding pitch; GP export reconciles a feasible fret and reports the change.",
            })
            note["harmonic"] = None
    return removed


def _clean_percussion(percussion, note_times, profile, removed, onset_adjustments):
    filtered = []
    for event in percussion:
        if _confidence(event) < profile.percussion_threshold:
            removed.append({"reason": "below_percussion_threshold", "event": event})
        else:
            filtered.append(deepcopy(event))
    result = []
    for cluster in _time_clusters(filtered, profile.chord_tolerance_seconds):
        nearest = None
        center = _representative_time(cluster)
        if note_times:
            candidate = min(note_times, key=lambda value: abs(value - center))
            if abs(candidate - center) <= profile.chord_tolerance_seconds:
                nearest = candidate
        time = center if nearest is None else nearest
        techniques = {}
        for event in cluster:
            technique = event.get("technique")
            if not isinstance(technique, str) or not technique:
                raise HarnessError("Draft cleanup requires named percussion techniques.")
            current = techniques.get(technique)
            if current is None or _confidence(event) > _confidence(current):
                if current is not None:
                    removed.append({"reason": "duplicate_percussion_cluster", "event": current})
                techniques[technique] = event
            else:
                removed.append({"reason": "duplicate_percussion_cluster", "event": event})
        for event in techniques.values():
            if time != _onset(event):
                onset_adjustments.append({
                    "kind": "percussion_cluster",
                    "technique": event["technique"],
                    "fromSeconds": _onset(event),
                    "toSeconds": time,
                })
            event["onsetSeconds"] = time
            result.append(event)
    return sorted(result, key=lambda item: (_onset(item), item["technique"]))


def clean_hypotheses(document, profile=DraftProfile()):
    if not isinstance(document, dict) or not isinstance(document.get("notes"), list) or not isinstance(document.get("percussion"), list):
        raise HarnessError("Draft cleanup requires a hypothesis document with note and percussion lists.")
    if not isinstance(profile, DraftProfile):
        raise TypeError("profile must be DraftProfile.")
    source = deepcopy(document)
    removed_notes = []
    onset_adjustments = []
    selected = []
    for note in source["notes"]:
        if _confidence(note) < profile.note_threshold:
            removed_notes.append({"reason": "below_note_threshold", "event": note})
        else:
            selected.append(note)
    selected = _cluster_notes(selected, profile.chord_tolerance_seconds, removed_notes, onset_adjustments)
    selected = _suppress_same_string(selected, profile.same_string_gap_seconds, removed_notes)
    removed_harmonics = _clean_harmonics(selected, profile)
    removed_percussion = []
    percussion = _clean_percussion(source["percussion"], sorted({_onset(note) for note in selected}), profile, removed_percussion, onset_adjustments)
    source["notes"] = selected
    source["percussion"] = percussion
    return source, {
        "schemaVersion": 1,
        "kind": "editable-draft-cleanup",
        "profile": asdict(profile),
        "sourceCounts": {"notes": len(document["notes"]), "percussion": len(document["percussion"])},
        "retainedCounts": {"notes": len(selected), "percussion": len(percussion)},
        "removedNoteCount": len(removed_notes),
        "removedPercussionCount": len(removed_percussion),
        "removedHarmonicCount": len(removed_harmonics),
        "maximumOnsetClusterDisplacementSeconds": max(
            (abs(value["toSeconds"] - value["fromSeconds"]) for value in onset_adjustments),
            default=0,
        ),
        "onsetAdjustments": onset_adjustments,
        "removedNotes": removed_notes,
        "removedPercussion": removed_percussion,
        "removedHarmonics": removed_harmonics,
        "rawHypothesesModified": False,
    }
