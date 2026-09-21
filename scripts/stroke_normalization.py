"""Normalize supported, over-written downstroke bursts without native GP effects."""

from collections import defaultdict
from copy import deepcopy
from fractions import Fraction

from .transcriber_audio import HarnessError


STROKE_STEP = Fraction(1, 8)
BURST_SPAN = Fraction(1, 2)


def _position(event):
    try:
        value = Fraction(*event["scoreOnsetQuarter"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise HarnessError("Stroke normalization requires quantized score onsets.") from error
    if value < 0:
        raise HarnessError("Stroke normalization requires nonnegative score onsets.")
    return value


def _rational(value):
    return [value.numerator, value.denominator]


def normalize_downstroke_bursts(document):
    result = deepcopy(document)
    notes = result["notes"]
    techniques = result.get("techniques", [])
    by_onset = defaultdict(list)
    downstrokes = defaultdict(list)
    for note in notes:
        by_onset[_position(note)].append(note)
    for event in techniques:
        if event["technique"] == "rasgueado" or event["technique"] == "brush" and event["direction"] == "Down":
            onset = _position(event)
            if any(note["string"] in event["strings"] for note in by_onset.get(onset, [])):
                downstrokes[onset].append(event)
    changes, skipped = [], []
    used = set()
    removed_notes, removed_events = set(), set()
    added_notes, added_events = [], []
    # Respect the declared beat unit and any mapped tempo/beat-unit changes.
    beat_changes = [(Fraction(0), Fraction(*document["metadata"]["tempo"]["beatUnit"]) * 4)]
    beat_changes.extend((_position({"scoreOnsetQuarter": row["scoreQuarter"]}), Fraction(*row["beatUnit"]) * 4)
                        for row in document.get("notatedTempoChanges", []))
    beat_changes.sort()
    landings = set()
    for onset in downstrokes:
        origin, unit = max((row for row in beat_changes if row[0] <= onset), key=lambda row: row[0])
        if unit <= 0:
            raise HarnessError("Stroke normalization requires a positive beat unit.")
        beat = origin + round((onset - origin) / unit) * unit
        if beat >= 2 * STROKE_STEP and abs(onset - beat) <= STROKE_STEP:
            landings.add(beat)
    referenced = {note.get("connectionOriginNoteId") for note in notes if note.get("connectionOriginNoteId")}
    for landing in sorted(landings):
        positions = sorted(onset for onset in downstrokes if landing - BURST_SPAN <= onset <= landing + STROKE_STEP)
        if len(positions) <= 3 or used.intersection(positions):
            continue
        if positions[-1] - positions[0] > BURST_SPAN or any(right - left > STROKE_STEP for left, right in zip(positions, positions[1:])):
            continue
        source_notes = [note for onset in positions for note in by_onset[onset]]
        source_events = [event for event in techniques if positions[0] <= _position(event) <= positions[-1]]
        destinations = [landing - 2 * STROKE_STEP, landing - STROKE_STEP, landing]
        bins = [[], [], []]
        position_bins = [[], [], []]
        for onset in positions:
            index = min(range(3), key=lambda index: (abs(onset - destinations[index]), index))
            bins[index].extend(by_onset[onset])
            position_bins[index].append(onset)
        reason = None
        if any(event["technique"] not in ("brush", "rasgueado") or event["technique"] == "brush" and event["direction"] != "Down" for event in source_events):
            reason = "mixed_or_upward_strokes"
        elif any(onset not in positions for onset in by_onset if positions[0] <= onset <= positions[-1]):
            reason = "intervening_note_attacks"
        elif any(onset.denominator % 3 == 0 for onset in positions):
            reason = "tuplet_rhythm"
        elif not all(bins):
            reason = "missing_supported_component"
        elif len({note["soundingPitchMidi"] for note in source_notes}) == 1 and not any(event["technique"] == "rasgueado" for event in source_events):
            reason = "single_note_repetition_without_compound_evidence"
        elif any(note.get("grace") or note.get("connection", "none") != "none" or note.get("noteId") in referenced for note in source_notes):
            reason = "anchored_grace_or_note_relationship"
        for onsets in position_bins:
            if len(onsets) > 1:
                pitch_sets = [{note["soundingPitchMidi"] for note in by_onset[onset]} for onset in onsets]
                union = set.union(*pitch_sets)
                if not set.intersection(*pitch_sets) or not any(pitches == union for pitches in pitch_sets):
                    reason = "changing_pitch_content_within_merged_component"
        affected = set(positions)
        if any(onset in by_onset and onset not in affected for onset in destinations):
            reason = "destination_has_an_independent_attack"
        if reason:
            skipped.append({"sourceOnsetsQuarter": [_rational(p) for p in positions], "landingQuarter": _rational(landing), "reason": reason})
            continue
        normalized_notes, normalized_events = [], []
        for index, (destination, bucket) in enumerate(zip(destinations, bins)):
            best = {}
            for note in bucket:
                pitch = note["soundingPitchMidi"]
                if pitch not in best or note["confidence"] > best[pitch]["confidence"]:
                    best[pitch] = note
            for note in best.values():
                value = deepcopy(note)
                end = max(_position(source) + Fraction(*source["scoreDurationQuarter"]) for source in bucket
                          if source["soundingPitchMidi"] == note["soundingPitchMidi"])
                duration = STROKE_STEP if index < 2 else max(STROKE_STEP, end - destination)
                value["scoreOnsetQuarter"] = _rational(destination)
                value["scoreDurationQuarter"] = _rational(duration)
                value["uncertainty"] = [*value.get("uncertainty", []), "normalized_dense_ami_downstrokes"]
                normalized_notes.append(value)
            exemplar = max((event for onset in positions for event in downstrokes[onset]), key=lambda event: event["confidence"])
            normalized_events.append({
                **deepcopy(exemplar), "scoreOnsetQuarter": _rational(destination),
                "technique": "brush", "direction": "Down",
                "strings": sorted({note["string"] for note in best.values()}),
                "strokeFinger": ("a", "m", "i")[index],
            })
        removed_notes.update(map(id, source_notes))
        selected_events = [event for event in source_events if _position(event) in affected or event["technique"] == "rasgueado"]
        removed_events.update(map(id, selected_events))
        added_notes.extend(normalized_notes)
        added_events.extend(normalized_events)
        used.update(positions)
        changes.append({
            "sourceOnsetsQuarter": [_rational(onset) for onset in positions],
            "normalizedOnsetsQuarter": [_rational(onset) for onset in destinations],
            "fingers": ["a", "m", "i"], "direction": "Down", "landingQuarter": _rational(landing),
            "sourceStrokeCount": len(positions), "normalizedStrokeCount": 3,
            "sourceNotes": deepcopy(source_notes), "normalizedNotes": deepcopy(normalized_notes),
            "sourceTechniques": deepcopy(selected_events),
            "mergedRepeatedPitchAttacks": len(source_notes) - len(normalized_notes),
        })
    result["notes"] = sorted([note for note in notes if id(note) not in removed_notes] + added_notes, key=_position)
    result["techniques"] = sorted([event for event in techniques if id(event) not in removed_events] + added_events, key=_position)
    return result, {
        "kind": "fingerstyle-downstroke-normalization", "schemaVersion": 1,
        "policy": "Four or more close, compatible attacks supported by down-brush or rasgueado evidence within half a quarter become three downstrokes: two 32nds before a declared beat and one on it. Existing triples, changing passages, mixed directions and anchored effects remain unchanged. A compound score alone never creates missing attacks.",
        "changedGroupCount": len(changes), "changes": changes, "skippedCandidates": skipped,
        "nativeRasgueadoPlayback": False, "rawHypothesesModified": False,
    }
