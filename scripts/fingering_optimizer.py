"""Choose unique, playable guitar fingerings for simultaneous pitch hypotheses."""

from collections import defaultdict
from copy import deepcopy
from fractions import Fraction

from .transcriber_audio import HarnessError


MAX_FRET = 24
MAX_FRETTED_SPAN = 5


def _onset(note):
    value = note.get("scoreOnsetQuarter")
    if value is not None:
        try:
            return Fraction(*value)
        except (TypeError, ValueError, ZeroDivisionError) as error:
            raise HarnessError("Fingering optimization requires rational score onsets.") from error
    return Fraction(str(note["onsetSeconds"])).limit_denominator(1_000_000)


def _candidates(pitch, tuning, capo):
    result = []
    for string in range(1, 7):
        fret = pitch - tuning[6 - string] - capo
        if 0 <= fret <= MAX_FRET:
            result.append((string, fret))
    return result


def _deduplicate(group):
    by_pitch = {}
    removed = []
    for index, note in group:
        pitch = note["soundingPitchMidi"]
        current = by_pitch.get(pitch)
        rank = (note["confidence"], -note["fret"], note["string"], -index)
        if current is None or rank > current[0]:
            if current is not None:
                removed.append({"index": current[1], "reason": "simultaneous_unison_duplicate", "keptIndex": index, "pitch": pitch})
            by_pitch[pitch] = (rank, index, note)
        else:
            removed.append({"index": index, "reason": "simultaneous_unison_duplicate", "keptIndex": current[1], "pitch": pitch})
    return [(value[1], value[2]) for value in by_pitch.values()], removed


def _optimize_group(group, tuning, capo, onset, recent, previous_position):
    try:
        from ortools.sat.python import cp_model
    except (ImportError, ModuleNotFoundError) as error:
        raise HarnessError("OR-Tools is required for chord fingering optimization.") from error
    unique, removed = _deduplicate(group)
    candidates = {}
    for index, note in unique:
        values = _candidates(note["soundingPitchMidi"], tuning, capo)
        if not values:
            removed.append({"index": index, "reason": "pitch_has_no_playable_string", "pitch": note["soundingPitchMidi"]})
        else:
            candidates[index] = values
    playable = [(index, note) for index, note in unique if index in candidates]
    if not playable:
        return {}, removed, "empty"
    model = cp_model.CpModel()
    selected = {}
    dropped = {}
    for index, note in playable:
        dropped[index] = model.new_bool_var(f"drop_{index}")
        for candidate_index, (string, fret) in enumerate(candidates[index]):
            selected[index, candidate_index] = model.new_bool_var(f"use_{index}_{candidate_index}")
        model.add(dropped[index] + sum(selected[index, candidate_index] for candidate_index in range(len(candidates[index]))) == 1)
    for string in range(1, 7):
        model.add(sum(
            selected[index, candidate_index]
            for index, _ in playable
            for candidate_index, value in enumerate(candidates[index])
            if value[0] == string
        ) <= 1)
    options = [
        (index, candidate_index, string, fret, selected[index, candidate_index])
        for index, _ in playable
        for candidate_index, (string, fret) in enumerate(candidates[index])
    ]
    for left_index, left_candidate, _, left_fret, left_var in options:
        if left_fret == 0:
            continue
        for right_index, right_candidate, _, right_fret, right_var in options:
            if (right_index, right_candidate) <= (left_index, left_candidate) or right_fret == 0:
                continue
            if abs(left_fret - right_fret) > MAX_FRETTED_SPAN:
                model.add(left_var + right_var <= 1)
    objective = []
    for index, note in playable:
        objective.append(round(1000 * note["confidence"]) * dropped[index])
        for candidate_index, (string, fret) in enumerate(candidates[index]):
            model_prior = abs(string - note["string"]) * 4 + abs(fret - note["fret"])
            ease = (8 + fret + max(0, fret - 7) * 2) if fret else 0
            hand = abs(fret - previous_position) * 3 if fret and previous_position is not None else 0
            continuity = 0
            prior = recent.get(note["soundingPitchMidi"])
            if prior is not None and onset - prior["onset"] <= Fraction(3, 2):
                continuity = abs(string - prior["string"]) * 80 + abs(fret - prior["fret"]) * 20
            objective.append((model_prior + ease + hand + continuity) * selected[index, candidate_index])
    model.minimize(sum(objective))
    solver = cp_model.CpSolver()
    solver.parameters.max_deterministic_time = 5
    solver.parameters.num_search_workers = 1
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise HarnessError("Chord fingering optimization found no feasible assignment.")
    assignments = {}
    for index, note in playable:
        if solver.value(dropped[index]):
            removed.append({"index": index, "reason": "unplayable_chord_voice", "pitch": note["soundingPitchMidi"]})
            continue
        for candidate_index, value in enumerate(candidates[index]):
            if solver.value(selected[index, candidate_index]):
                assignments[index] = value
                break
    return assignments, removed, "optimal" if status == cp_model.OPTIMAL else "feasible"


def optimize_fingerings(document):
    if not isinstance(document, dict) or not isinstance(document.get("metadata"), dict) or not isinstance(document.get("notes"), list):
        raise HarnessError("Fingering optimization requires hypothesis metadata and notes.")
    tuning = document["metadata"].get("openStringMidi")
    capo = document["metadata"].get("capoFret")
    if not isinstance(tuning, list) or len(tuning) != 6 or any(type(value) is not int for value in tuning) or type(capo) is not int:
        raise HarnessError("Fingering optimization requires six-string tuning and full capo.")
    result = deepcopy(document)
    groups = defaultdict(list)
    for index, note in enumerate(result["notes"]):
        if not isinstance(note, dict) or type(note.get("soundingPitchMidi")) is not int or type(note.get("string")) is not int or type(note.get("fret")) is not int:
            raise HarnessError("Fingering optimization requires integer pitch, string and fret hypotheses.")
        groups[_onset(note)].append((index, note))
    removed = []
    changes = []
    difficult = []
    statuses = defaultdict(int)
    keep = set(range(len(result["notes"])))
    recent = {}
    previous_position = None
    for onset, group in sorted(groups.items()):
        assignments, local_removed, status = _optimize_group(group, tuning, capo, onset, recent, previous_position)
        statuses[status] += 1
        for value in local_removed:
            keep.discard(value["index"])
            removed.append({"onsetQuarter": [onset.numerator, onset.denominator], **value})
        for index, (string, fret) in assignments.items():
            note = result["notes"][index]
            if (note["string"], note["fret"]) != (string, fret):
                changes.append({
                    "index": index,
                    "onsetQuarter": [onset.numerator, onset.denominator],
                    "pitch": note["soundingPitchMidi"],
                    "predictedString": note["string"],
                    "predictedFret": note["fret"],
                    "selectedString": string,
                    "selectedFret": fret,
                })
                note["string"], note["fret"] = string, fret
            note["fretBasePitchMidi"] = tuning[6 - note["string"]] + capo + note["fret"]
            recent[note["soundingPitchMidi"]] = {"onset": onset, "string": note["string"], "fret": note["fret"]}
        fretted = [fret for _, fret in assignments.values() if fret]
        if fretted:
            previous_position = sorted(fretted)[len(fretted) // 2]
        by_fret = defaultdict(list)
        for index, (string, fret) in assignments.items():
            if fret:
                by_fret[fret].append(string)
        for fret, strings in by_fret.items():
            if len(strings) >= 4:
                difficult.append({
                    "onsetQuarter": [onset.numerator, onset.denominator],
                    "reason": "extended_barre_required_by_retained_pitch_set",
                    "fret": fret,
                    "strings": sorted(strings, reverse=True),
                    "noteIndices": sorted(index for index, value in assignments.items() if value[1] == fret),
                })
    result["notes"] = [note for index, note in enumerate(result["notes"]) if index in keep]
    return result, {
        "schemaVersion": 1,
        "kind": "playable-chord-fingering-optimization",
        "solver": "OR-Tools CP-SAT",
        "maximumFret": MAX_FRET,
        "maximumFrettedChordSpan": MAX_FRETTED_SPAN,
        "repeatedPitchContinuityQuarter": [3, 2],
        "sourceNoteCount": len(document["notes"]),
        "retainedNoteCount": len(result["notes"]),
        "changedCount": len(changes),
        "removedCount": len(removed),
        "difficultChordCount": len(difficult),
        "statuses": dict(sorted(statuses.items())),
        "changes": changes,
        "removed": removed,
        "difficultChords": difficult,
        "rawHypothesesModified": False,
    }
