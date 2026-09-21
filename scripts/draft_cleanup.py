"""Deterministic cleanup for broad model hypotheses before notation export."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import math

from .transcriber_audio import HarnessError
from .gp_events import validate_provided_timing


@dataclass(frozen=True)
class DraftProfile:
    note_threshold: float = 0.9
    percussion_threshold: float = 0.6
    thumb_slap_threshold: float | None = None
    harmonic_threshold: float = 0.8
    include_harmonics: bool = False
    brush_threshold: float = 0.9
    arpeggio_threshold: float = 0.925
    pick_stroke_threshold: float = 0.8
    rasgueado_threshold: float = 0.99
    brush_membership_threshold: float = 0.6
    arpeggio_membership_threshold: float = 0.5
    pick_stroke_membership_threshold: float = 0.7
    rasgueado_membership_threshold: float = 0.8
    connection_threshold: float = 0.8
    note_technique_threshold: float = 0.8
    grace_threshold: float = 0.8
    chord_tolerance_seconds: float = 0.04
    same_string_gap_seconds: float = 0.0
    strict_note_confidence: bool = False
    rhythm_policy: str = "adaptive"

    def __post_init__(self):
        if type(self.include_harmonics) is not bool:
            raise TypeError("include_harmonics must be boolean.")
        if type(self.strict_note_confidence) is not bool:
            raise TypeError("strict_note_confidence must be boolean.")
        if self.rhythm_policy not in ("adaptive", "fingerstyle"):
            raise ValueError("rhythm_policy must be adaptive or fingerstyle.")
        if self.thumb_slap_threshold is not None and (
            type(self.thumb_slap_threshold) not in (int, float) or not math.isfinite(self.thumb_slap_threshold)
            or not 0 <= self.thumb_slap_threshold <= 1
        ):
            raise ValueError("thumb_slap_threshold must be from zero through one or omitted.")
        for name in (
            "note_threshold", "percussion_threshold", "harmonic_threshold",
            "brush_threshold", "arpeggio_threshold", "pick_stroke_threshold", "rasgueado_threshold",
            "brush_membership_threshold", "arpeggio_membership_threshold",
            "pick_stroke_membership_threshold", "rasgueado_membership_threshold",
            "connection_threshold", "note_technique_threshold", "grace_threshold",
        ):
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
    clusters = []
    for note in sorted(notes, key=lambda item: (_onset(item), -_confidence(item))):
        string = note.get("string")
        if type(string) is not int or not 1 <= string <= 6:
            raise HarnessError("Draft cleanup requires physical note strings from one through six.")
        prior = clusters[-1] if clusters else []
        reattack = any(value["string"] == string and _onset(value) != _onset(note) for value in prior)
        if not prior or _onset(note) - _onset(prior[0]) > tolerance or reattack:
            clusters.append([note])
        else:
            prior.append(note)
    for cluster in clusters:
        time = _representative_time(cluster)
        strings = {}
        for note in cluster:
            string = note.get("string")
            if type(string) is not int or not 1 <= string <= 6:
                raise HarnessError("Draft cleanup requires physical note strings from one through six.")
            key = (string, note.get("soundingPitchMidi"))
            current = strings.get(key)
            if current is None or (_confidence(note), -_onset(note)) > (_confidence(current), -_onset(current)):
                if current is not None:
                    removed.append({"reason": "same_string_chord_cluster", "event": current})
                strings[key] = note
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


def _clean_percussion(percussion, note_times, profile, removed, onset_adjustments, grouping_tolerance):
    filtered = []
    for event in percussion:
        threshold = profile.thumb_slap_threshold if event.get("technique") == "thumb_slap" and profile.thumb_slap_threshold is not None else profile.percussion_threshold
        if _confidence(event) < threshold:
            removed.append({"reason": "below_percussion_threshold", "event": event})
        else:
            filtered.append(deepcopy(event))
    result = []
    for cluster in _time_clusters(filtered, grouping_tolerance):
        nearest = None
        center = _representative_time(cluster)
        if note_times:
            candidate = min(note_times, key=lambda value: abs(value - center))
            if abs(candidate - center) <= grouping_tolerance:
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
    acoustic_attacks = {}
    for note in document["notes"]:
        if "technique_membership_completed_attack" not in note.get("uncertainty", []) and _confidence(note) >= .5:
            onset = _onset(note)
            acoustic_attacks[onset] = max(acoustic_attacks.get(onset, 0.), _confidence(note))
    source["acousticAttackEvidence"] = [{"onsetSeconds": onset, "confidence": confidence}
                                       for onset, confidence in sorted(acoustic_attacks.items())]
    grouping_tolerance = profile.chord_tolerance_seconds
    metadata = document.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict) or not {"tempo", "timeSignature"} <= metadata.keys():
            raise HarnessError("Chord grouping requires explicit tempo and meter when metadata is supplied.")
        changes = metadata.get("tempoChanges", [])
        if not isinstance(changes, list):
            raise HarnessError("Chord grouping requires a list of tempo changes.")
        tempos = [metadata["tempo"], *changes]
        fastest_quarter_bpm = max(float(validate_provided_timing(tempo, metadata["timeSignature"])) for tempo in tempos)
        grouping_tolerance = min(grouping_tolerance, 60 / fastest_quarter_bpm / 16)
    removed_notes = []
    onset_adjustments = []
    selected = []
    for note in source["notes"]:
        membership_completed = "technique_membership_completed_attack" in note.get("uncertainty", [])
        if _confidence(note) < profile.note_threshold and (profile.strict_note_confidence or not membership_completed):
            removed_notes.append({"reason": "below_note_threshold", "event": note})
        else:
            selected.append(note)
    selected = _cluster_notes(selected, grouping_tolerance, removed_notes, onset_adjustments)
    selected = _suppress_same_string(selected, profile.same_string_gap_seconds, removed_notes)
    removed_harmonics = _clean_harmonics(selected, profile)
    removed_percussion = []
    percussion = _clean_percussion(source["percussion"], sorted({_onset(note) for note in selected}), profile, removed_percussion, onset_adjustments, grouping_tolerance)
    source["notes"] = selected
    source["percussion"] = percussion
    removed_techniques = []
    retained_techniques = []
    thresholds = {
        "brush": profile.brush_threshold,
        "arpeggio": profile.arpeggio_threshold,
        "pick_stroke": profile.pick_stroke_threshold,
        "rasgueado": profile.rasgueado_threshold,
    }
    membership_thresholds = {
        "brush": profile.brush_membership_threshold,
        "arpeggio": profile.arpeggio_membership_threshold,
        "pick_stroke": profile.pick_stroke_membership_threshold,
        "rasgueado": profile.rasgueado_membership_threshold,
    }
    techniques = source.get("techniques", [])
    if not isinstance(techniques, list):
        raise HarnessError("Draft cleanup requires a technique list when technique predictions are present.")
    for event in techniques:
        technique = event.get("technique")
        if technique not in thresholds:
            raise HarnessError(f"Unsupported technique prediction: {technique}")
        if _confidence(event) < thresholds[technique]:
            removed_techniques.append({"reason": "below_technique_threshold", "event": event})
            continue
        membership = event.get("stringMembershipConfidence")
        if not isinstance(membership, dict):
            raise HarnessError("Technique predictions require per-string membership confidence.")
        strings = []
        for string in range(1, 7):
            score = membership.get(str(string))
            if isinstance(score, bool) or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise HarnessError("Technique string-membership scores must be finite values from zero through one.")
            if score >= membership_thresholds[technique]:
                strings.append(string)
        if not strings:
            removed_techniques.append({"reason": "empty_membership_after_threshold", "event": event})
            continue
        event["strings"] = strings
        retained_techniques.append(event)
    source["techniques"] = retained_techniques
    retained_notes = []
    removed_orphan_members = []
    for note in source["notes"]:
        if "technique_membership_completed_attack" not in note.get("uncertainty", []):
            retained_notes.append(note)
            continue
        parent = note.get("completionParent")
        if parent is not None and (not isinstance(parent, dict) or not {"technique", "onsetSeconds"} <= parent.keys()):
            raise HarnessError("Completed attack parent requires technique and onsetSeconds.")
        if parent is not None:
            _onset(parent)
        supported = any(
            note["string"] in event["strings"]
            and abs(_onset(note) - _onset(event)) <= profile.chord_tolerance_seconds
            and (parent is None or (event["technique"] == parent["technique"] and _onset(event) == _onset(parent)))
            for event in retained_techniques
        )
        if supported:
            retained_notes.append(note)
        else:
            removed_orphan_members.append({"reason": "parent_technique_or_membership_filtered", "event": note})
    source["notes"] = retained_notes
    removed_connections = []
    removed_note_techniques = []
    removed_grace = []
    for note in retained_notes:
        connection = note.get("connection", "none")
        if connection != "none":
            confidence = note.get("connectionConfidence")
            if isinstance(confidence, bool) or type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise HarnessError("Predicted note connections require finite zero-to-one confidence.")
            if confidence < profile.connection_threshold:
                removed_connections.append({
                    "reason": "below_connection_threshold", "onsetSeconds": _onset(note),
                    "string": note["string"], "connection": connection, "confidence": confidence,
                })
                note["connection"] = "none"
        scores = note.get("noteTechniques", {})
        if not isinstance(scores, dict):
            raise HarnessError("Predicted note techniques must be a confidence mapping.")
        retained_scores = {}
        for technique, confidence in scores.items():
            if technique not in ("bend", "tap", "left_hand_tap", "vibrato", "slide_out_down", "slide_out_up", "slide_in_below", "slide_in_above"):
                raise HarnessError(f"Unsupported predicted note technique: {technique}")
            if isinstance(confidence, bool) or type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise HarnessError("Predicted note-technique confidence must be finite and from zero through one.")
            if confidence >= profile.note_technique_threshold:
                retained_scores[technique] = confidence
            else:
                removed_note_techniques.append({
                    "reason": "below_note_technique_threshold", "onsetSeconds": _onset(note),
                    "string": note["string"], "technique": technique, "confidence": confidence,
                })
        note["noteTechniques"] = retained_scores
        if "bend" not in retained_scores:
            note["bendCurve"] = None
        grace = note.get("grace")
        if grace is not None:
            if not isinstance(grace, dict):
                raise HarnessError("An anchored grace gesture must be an object.")
            fields = ("confidence", "fretConfidence", "modeConfidence", "transitionConfidence")
            for field in fields:
                confidence = grace.get(field)
                if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise HarnessError(f"Grace {field} must be a finite zero-to-one score.")
            if any(grace[field] < profile.grace_threshold for field in fields):
                removed_grace.append({
                    "onsetSeconds": _onset(note), "string": note["string"],
                    "reason": "below_grace_presence_or_attribute_threshold", "grace": deepcopy(grace),
                })
                note["grace"] = None
    return source, {
        "schemaVersion": 1,
        "kind": "editable-draft-cleanup",
        "profile": asdict(profile),
        "effectiveChordGroupingSeconds": grouping_tolerance,
        "reattackPolicy": "Distinct same-string attack times split chord clusters; different simultaneous pitches survive for fingering resolution. Temporal suppression is disabled by default.",
        "rasgueadoPolicy": "Retain rasgueado evidence and its supported note candidates. GP conversion normalizes qualifying dense downstroke groups and never emits the native Rasgueado playback effect.",
        "attackEvidencePolicy": "Independent acoustic onset scores at least0.5 may support chord-completion timing without automatically promoting their low-confidence pitches.",
        "sourceCounts": {"notes": len(document["notes"]), "percussion": len(document["percussion"])},
        "retainedCounts": {"notes": len(retained_notes), "percussion": len(percussion)},
        "removedNoteCount": len(removed_notes) + len(removed_orphan_members),
        "removedPercussionCount": len(removed_percussion),
        "percussionThresholds": {
            "wrist_thump": profile.percussion_threshold,
            "thumb_slap": profile.percussion_threshold if profile.thumb_slap_threshold is None else profile.thumb_slap_threshold,
            "percussive_hit": profile.percussion_threshold,
        },
        "removedHarmonicCount": len(removed_harmonics),
        "sourceTechniqueCount": len(techniques),
        "retainedTechniqueCount": len(retained_techniques),
        "removedTechniqueCount": len(removed_techniques),
        "techniqueThresholds": thresholds,
        "techniqueMembershipThresholds": membership_thresholds,
        "maximumOnsetClusterDisplacementSeconds": max(
            (abs(value["toSeconds"] - value["fromSeconds"]) for value in onset_adjustments),
            default=0,
        ),
        "onsetAdjustments": onset_adjustments,
        "removedNotes": removed_notes,
        "removedOrphanTechniqueMembers": removed_orphan_members,
        "removedPercussion": removed_percussion,
        "removedHarmonics": removed_harmonics,
        "removedTechniques": removed_techniques,
        "removedConnections": removed_connections,
        "removedNoteTechniques": removed_note_techniques,
        "removedGraceGestures": removed_grace,
        "rawHypothesesModified": False,
    }
