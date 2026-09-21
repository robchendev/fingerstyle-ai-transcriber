"""Assign two readable GP voices while minimizing avoidable tied fragments."""

from copy import deepcopy
from collections import defaultdict
from fractions import Fraction

from .transcriber_audio import HarnessError


VOICE_COUNT = 2


def _interval(note):
    try:
        onset = Fraction(*note["scoreOnsetQuarter"])
        duration = Fraction(*note["scoreDurationQuarter"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise HarnessError("Voice optimization requires rational score onset and duration.") from error
    if onset < 0 or duration <= 0:
        raise HarnessError("Voice intervals must be positive and nonnegative.")
    return onset, onset + duration


def _fragment_count(notes):
    total = 0
    by_voice = {voice: [] for voice in range(VOICE_COUNT)}
    for note in notes:
        by_voice[note["voiceIndex"]].append((_interval(note), note))
    for values in by_voice.values():
        boundaries = sorted({point for interval, _ in values for point in interval})
        for (start, end), _ in values:
            total += 1 + sum(start < boundary < end for boundary in boundaries)
    return total


def optimize_voices(document):
    if not isinstance(document, dict) or not isinstance(document.get("notes"), list):
        raise HarnessError("Voice optimization requires note hypotheses.")
    try:
        from ortools.sat.python import cp_model
    except (ImportError, ModuleNotFoundError) as error:
        raise HarnessError("OR-Tools is required for voice optimization.") from error
    result = deepcopy(document)
    if not result["notes"]:
        return result, {
            "schemaVersion": 1, "kind": "readable-voice-optimization", "solver": "OR-Tools CP-SAT",
            "voiceCount": VOICE_COUNT, "changedCount": 0, "estimatedSegmentsBefore": 0,
            "estimatedSegmentsAfter": 0, "rawHypothesesModified": False,
        }
    intervals = [_interval(note) for note in result["notes"]]
    model = cp_model.CpModel()
    groups = defaultdict(list)
    for index, interval in enumerate(intervals):
        groups[interval].append(index)
    group_intervals = list(groups)
    group_voices = [model.new_int_var(0, VOICE_COUNT - 1, f"voice_{index}") for index in range(len(groups))]
    voices = [None] * len(intervals)
    for interval, voice in zip(group_intervals, group_voices):
        for index in groups[interval]:
            voices[index] = voice
    objective = []
    for index, note in enumerate(result["notes"]):
        original = note.get("voiceIndex")
        if type(original) is not int or not 0 <= original <= 3:
            raise HarnessError("Voice optimization requires integer model voice indices.")
        changed = model.new_bool_var(f"changed_{index}")
        preferred = min(original, VOICE_COUNT - 1)
        model.add(voices[index] != preferred).only_enforce_if(changed)
        model.add(voices[index] == preferred).only_enforce_if(changed.Not())
        objective.append(12 * changed)
    same_voice = {}
    for left, (start, end) in enumerate(group_intervals):
        boundaries = defaultdict(list)
        for right, interval in enumerate(group_intervals):
            points = [point for point in interval if start < point < end]
            if not points:
                continue
            pair = min(left, right), max(left, right)
            if pair not in same_voice:
                same = model.new_bool_var(f"same_{pair[0]}_{pair[1]}")
                model.add(group_voices[left] == group_voices[right]).only_enforce_if(same)
                model.add(group_voices[left] != group_voices[right]).only_enforce_if(same.Not())
                same_voice[pair] = same
            for point in points:
                boundaries[point].append(same_voice[pair])
        for ordinal, same in enumerate(boundaries.values()):
            split = model.new_bool_var(f"split_{left}_{ordinal}")
            model.add_max_equality(split, same)
            objective.append(40 * len(groups[(start, end)]) * split)
    model.minimize(sum(objective))
    solver = cp_model.CpSolver()
    solver.parameters.max_deterministic_time = 20
    solver.parameters.num_search_workers = 1
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise HarnessError("Voice optimization found no feasible assignment.")
    before = _fragment_count([{**note, "voiceIndex": min(note["voiceIndex"], VOICE_COUNT - 1)} for note in result["notes"]])
    changes = []
    for index, note in enumerate(result["notes"]):
        selected = solver.value(voices[index])
        if note["voiceIndex"] != selected:
            changes.append({
                "index": index,
                "scoreOnsetQuarter": note["scoreOnsetQuarter"],
                "pitch": note["soundingPitchMidi"],
                "predictedVoice": note["voiceIndex"],
                "selectedVoice": selected,
            })
            note["voiceIndex"] = selected
    after = _fragment_count(result["notes"])
    return result, {
        "schemaVersion": 1,
        "kind": "readable-voice-optimization",
        "solver": "OR-Tools CP-SAT",
        "status": "optimal" if status == cp_model.OPTIMAL else "feasible",
        "objectiveValue": solver.objective_value,
        "bestObjectiveBound": solver.best_objective_bound,
        "wallTimeSeconds": solver.wall_time,
        "voiceCount": VOICE_COUNT,
        "policy": "Equal onset/end chord members share a voice; minimize actual distinct internal boundaries per sustained note, not pairwise overlap counts.",
        "changedCount": len(changes),
        "estimatedSegmentsBefore": before,
        "estimatedSegmentsAfter": after,
        "changes": changes,
        "rawHypothesesModified": False,
    }
