"""Retain playable pitches before optimizing fingering across bounded phrases."""

from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
import math

from .gp_events import HARMONIC_OFFSETS
from .transcriber_audio import HarnessError


MAX_FRET = 24
MAX_FRETTED_SPAN = 5
# These are reporting thresholds, never candidate filters.
MAX_RAPID_POSITION_SHIFT = 5
RAPID_SHIFT_WINDOW_QUARTER = Fraction(1)
PHRASE_WINDOW_ONSETS = 12
PHRASE_COMMIT_ONSETS = 6


def _onset(note):
    value = note.get("scoreOnsetQuarter")
    if value is not None:
        try:
            return Fraction(*value)
        except (TypeError, ValueError, ZeroDivisionError) as error:
            raise HarnessError("Fingering optimization requires rational score onsets.") from error
    if note.get("onsetSeconds") is None:
        raise HarnessError("Fingering optimization requires a score onset or onset seconds.")
    return Fraction(str(note["onsetSeconds"])).limit_denominator(1_000_000)


def _rational(value):
    return [value.numerator, value.denominator]


def _candidates(pitch, tuning, capo):
    return [
        (string, pitch - tuning[6 - string] - capo)
        for string in range(1, 7)
        if 0 <= pitch - tuning[6 - string] - capo <= MAX_FRET
    ]


def _deduplicate(group):
    by_pitch = defaultdict(list)
    for index, note in group:
        by_pitch[note["soundingPitchMidi"]].append((index, note))
    unique, removed = [], []
    for pitch, values in by_pitch.items():
        winner = max(values, key=lambda value: (
            value[1]["confidence"], -value[1]["fret"], value[1]["string"], -value[0],
        ))
        unique.append(winner)
        removed.extend({
            "index": index, "reason": "simultaneous_unison_duplicate",
            "keptIndex": winner[0], "pitch": pitch,
        } for index, _ in values if index != winner[0])
    return unique, removed


def _harmonic_node(harmonic):
    if not isinstance(harmonic, dict):
        return None
    value = harmonic.get("fret")
    try:
        return Fraction(*value) if isinstance(value, list) else Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _hand_fret(note, candidate):
    harmonic = note.get("harmonic")
    if isinstance(harmonic, dict) and harmonic.get("type") == "Natural":
        node = _harmonic_node(harmonic)
        if node in HARMONIC_OFFSETS:
            return int(node)
    return candidate[1]


def _hand_position(fretted):
    return sorted(fretted)[len(fretted) // 2]


def _grace_spec(note, tuning, capo):
    grace = note.get("grace")
    if grace is None:
        return None
    if not isinstance(grace, dict) or type(grace.get("intervalSemitones")) is not int:
        raise HarnessError("Fingering optimization requires an integer grace interval.")
    transition = grace.get("transition", "none")
    if transition not in ("none", "hammer_on", "pull_off", "slide_1", "slide_2"):
        raise HarnessError("Fingering optimization requires a supported grace transition.")
    interval = grace["intervalSemitones"]
    source_pitch = note["soundingPitchMidi"] - interval
    declared_pitch = grace.get("sourcePitchMidi")
    if declared_pitch is not None and type(declared_pitch) is not int:
        raise HarnessError("Grace source pitch must be an integer MIDI pitch.")
    reason = None
    if (transition == "hammer_on" and interval <= 0
            or transition == "pull_off" and interval >= 0
            or transition in ("slide_1", "slide_2") and interval == 0):
        reason = "grace_transition_interval_mismatch"
    elif not 0 <= source_pitch <= 127:
        reason = "grace_source_pitch_outside_midi_range"
    elif declared_pitch is not None and declared_pitch != source_pitch:
        reason = "grace_source_pitch_interval_mismatch"
    elif note.get("noteId") is not None and grace.get("anchorNoteId") is not None and grace["anchorNoteId"] != note["noteId"]:
        reason = "grace_anchor_note_id_mismatch"
    return {
        "sourcePitch": source_pitch, "reason": reason,
        "sourceFrets": {string: fret for string, fret in _candidates(source_pitch, tuning, capo)},
    }


def _note_candidates(index, note, tuning, capo, harmonic_reports):
    harmonic = note.get("harmonic")
    if harmonic is not None:
        values = [(note["string"], note["fret"])] if 1 <= note["string"] <= 6 and 0 <= note["fret"] <= MAX_FRET else []
        node = _harmonic_node(harmonic)
        supported = isinstance(harmonic, dict) and harmonic.get("type") in ("Natural", "Artificial", "Tap", "Pinch") and node in HARMONIC_OFFSETS
        consistent = False
        if supported and values:
            base = tuning[6 - note["string"]] + capo
            if harmonic["type"] != "Natural":
                base += note["fret"]
            consistent = base + HARMONIC_OFFSETS[node] == note["soundingPitchMidi"]
        reason = "harmonic_position_preserved" if supported and consistent else "unsupported_harmonic_position_preserved"
        harmonic_reports.append({
            "index": index,
            "reason": reason if values else "harmonic_position_outside_guitar_range",
            "harmonic": deepcopy(harmonic), "pitchConsistent": consistent,
            "positionInRange": bool(values),
        })
    else:
        values = _candidates(note["soundingPitchMidi"], tuning, capo)
    return values


def _maximum_retention(group, candidates):
    """Exact maximum matching over every possible five-fret hand interval."""
    best = {}
    for low in range(1, MAX_FRET + 1):
        occupied = {}

        def augment(index, visited):
            note = group["notes"][index]
            for candidate in candidates[index]:
                string = candidate[0]
                fret = _hand_fret(note, candidate)
                if string in visited or (fret and not low <= fret <= low + MAX_FRETTED_SPAN):
                    continue
                visited.add(string)
                if string not in occupied or augment(occupied[string][0], visited):
                    occupied[string] = (index, candidate)
                    return True
            return False

        for index in group["notes"]:
            augment(index, set())
        if len(occupied) > len(best):
            best = {index: candidate for index, candidate in occupied.values()}
        if len(best) == min(6, len(group["notes"])):
            break
    return best


def _elapsed(left, right):
    seconds = None
    if left["seconds"] is not None and right["seconds"] is not None:
        difference = right["seconds"] - left["seconds"]
        if difference > 0:
            seconds = difference
    return right["onset"] - left["onset"], seconds


def _motion_weight(left, right):
    quarter, seconds = _elapsed(left, right)
    # Seconds reflect tempo/rubato; score time is the explicit fallback.
    elapsed = seconds if seconds is not None else float(quarter)
    return max(1, min(160, round(12 / max(.125, elapsed))))


def _candidate_cost(note, candidate, values):
    string, fret = candidate
    provisional = (note["string"], note["fret"])
    prior = 0
    if provisional in values:
        prior = round(note["confidence"] * (
            24 * (string != provisional[0]) + 2 * abs(fret - provisional[1])
        ))
    # There is deliberately no absolute-fret/low-position reward.
    return prior + (2 if _hand_fret(note, candidate) else 0)


def _solve_window(window, committed, groups_by_index, token_indices, previous_hand):
    try:
        from ortools.sat.python import cp_model
    except (ImportError, ModuleNotFoundError) as error:
        raise HarnessError("OR-Tools is required for phrase fingering optimization.") from error
    model = cp_model.CpModel()
    selected, retained, strings, positions = {}, {}, {}, {}
    objective, upper_bound, connection_breaks, grace_breaks = [], 0, [], []

    def cost(expression, coefficient, maximum=1):
        nonlocal upper_bound
        objective.append(coefficient * expression)
        upper_bound += coefficient * maximum

    for chord_number, group in enumerate(window):
        options, fretted_options = [], []
        for index, note in group["notes"].items():
            values = group["candidates"][index]
            retained[index] = model.new_bool_var(f"keep_{index}")
            selected[index] = {}
            for candidate_number, candidate in enumerate(values):
                use = model.new_bool_var(f"use_{index}_{candidate_number}")
                selected[index][candidate] = use
                model.add_hint(use, int(group["witness"].get(index) == candidate))
                cost(use, _candidate_cost(note, candidate, values))
                options.append((candidate[0], use))
                hand_fret = _hand_fret(note, candidate)
                if hand_fret:
                    fretted_options.append((hand_fret, use))
            model.add(sum(selected[index].values()) == retained[index])
            grace = group["graces"].get(index)
            if grace is not None and grace["reason"] is None:
                grace_breaks.append(1 - sum(
                    use for candidate, use in selected[index].items()
                    if candidate[0] in grace["sourceFrets"]
                ))
            strings[index] = model.new_int_var(0, 6, f"string_{index}")
            model.add(strings[index] == sum(candidate[0] * use for candidate, use in selected[index].items()))
            cost(1 - retained[index], round(note["confidence"] * 200))
        model.add(sum(retained[index] for index in group["notes"]) == len(group["witness"]))
        for string in range(1, 7):
            model.add(sum(use for candidate_string, use in options if candidate_string == string) <= 1)
        position = model.new_int_var(1, MAX_FRET, f"position_{chord_number}")
        positions[group["onset"]] = position
        if fretted_options:
            high = model.new_int_var(0, MAX_FRET, f"high_{chord_number}")
            low = model.new_int_var(1, MAX_FRET, f"low_{chord_number}")
            has_fret = model.new_bool_var(f"fretted_{chord_number}")
            model.add_max_equality(has_fret, [use for _, use in fretted_options])
            model.add_max_equality(high, [fret * use for fret, use in fretted_options])
            model.add_min_equality(low, [MAX_FRET - (MAX_FRET - fret) * use for fret, use in fretted_options])
            model.add(high - low <= MAX_FRETTED_SPAN).only_enforce_if(has_fret)
            # The median represents the chord's hand position. Using its lowest
            # fret would let one stretched outlier conceal a large barre shift.
            median_choices = []
            fretted_count = sum(use for _, use in fretted_options)
            for fret in range(1, MAX_FRET + 1):
                is_median = model.new_bool_var(f"median_{chord_number}_{fret}")
                median_choices.append(is_median)
                model.add(position == fret).only_enforce_if(is_median)
                below = sum(use for value, use in fretted_options if value < fret)
                above = sum(use for value, use in fretted_options if value > fret)
                model.add(2 * below <= fretted_count).only_enforce_if(is_median)
                model.add(2 * above < fretted_count).only_enforce_if([is_median, has_fret])
            model.add_exactly_one(median_choices)
            span = model.new_int_var(0, MAX_FRETTED_SPAN, f"span_{chord_number}")
            model.add(span == high - low).only_enforce_if(has_fret)
            model.add(span == 0).only_enforce_if(has_fret.Not())
            cost(span, 3, MAX_FRETTED_SPAN)
        prior_group = window[chord_number - 1] if chord_number else previous_hand
        if prior_group is not None:
            prior_position = positions[prior_group["onset"]] if chord_number else prior_group["position"]
            motion = model.new_int_var(0, MAX_FRET - 1, f"motion_{chord_number}")
            model.add_abs_equality(motion, position - prior_position)
            cost(motion, _motion_weight(prior_group, group), MAX_FRET - 1)

    def string_and_presence(index):
        if index in strings:
            return strings[index], retained[index]
        if index in committed:
            return committed[index][0], 1
        return 0, 0

    latest_pitch = {}
    for index in sorted(committed, key=lambda value: (groups_by_index[value]["onset"], value)):
        latest_pitch[groups_by_index[index]["notes"][index]["soundingPitchMidi"]] = index
    for group in window:
        for index, note in group["notes"].items():
            pitch = note["soundingPitchMidi"]
            prior_index = latest_pitch.get(pitch)
            if prior_index is not None:
                prior_group = groups_by_index[prior_index]
                quarter, _ = _elapsed(prior_group, group)
                if 0 < quarter <= Fraction(3, 2):
                    prior_string, prior_retained = string_and_presence(prior_index)
                    changed = model.new_bool_var(f"repeat_change_{index}")
                    model.add(strings[index] == prior_string).only_enforce_if([retained[index], prior_retained, changed.Not()])
                    cost(changed, 40)
            latest_pitch[pitch] = index
            origin = token_indices.get(note.get("_connectionOriginToken"))
            if origin is None or groups_by_index[origin]["onset"] >= group["onset"]:
                continue
            if groups_by_index[origin]["notes"][origin].get("voiceIndex", 0) != note.get("voiceIndex", 0):
                continue
            origin_string, origin_retained = string_and_presence(origin)
            broken = model.new_bool_var(f"connection_break_{index}")
            connection_breaks.append(broken)
            # Missing endpoints and intervening same-voice attacks invalidate this
            # exact pair; never redirect a relation to a newly adjacent note.
            model.add(retained[index] == 1).only_enforce_if(broken.Not())
            model.add(origin_retained == 1).only_enforce_if(broken.Not())
            model.add(strings[index] == origin_string).only_enforce_if(broken.Not())
            for other, other_group in groups_by_index.items():
                if not groups_by_index[origin]["onset"] < other_group["onset"] < group["onset"]:
                    continue
                if other_group["notes"][other].get("voiceIndex", 0) != note.get("voiceIndex", 0):
                    continue
                other_string, other_retained = string_and_presence(other)
                model.add(other_string != origin_string).only_enforce_if([broken.Not(), other_retained])
    # Preserve main-pitch count, then bound connections, then playable grace
    # gestures. An impossible ornament cannot buy a main-note deletion.
    grace_weight = upper_bound + 1
    connection_weight = grace_weight * (len(grace_breaks) + 1)
    model.minimize(sum(objective) + grace_weight * sum(grace_breaks) + connection_weight * sum(connection_breaks))
    solver = cp_model.CpSolver()
    solver.parameters.max_deterministic_time = 1
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.solve(model)
    if status == cp_model.UNKNOWN:
        return {index: candidate for group in window for index, candidate in group["witness"].items()}, "retention_only_solver_limit"
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise HarnessError("Phrase fingering optimization violated its proven chord-retention bounds.")
    assignments = {
        index: candidate for index, values in selected.items()
        for candidate, use in values.items() if solver.value(use)
    }
    return assignments, "optimal" if status == cp_model.OPTIMAL else "feasible_solver_limit"


def _provenance(index, note, onset):
    result = {
        "index": index, "onsetQuarter": _rational(onset),
        "pitch": note["soundingPitchMidi"], "predictedString": note["string"],
        "predictedFret": note["fret"], "confidence": note["confidence"],
    }
    for field in ("id", "noteId", "connectionOriginNoteId", "_connectionToken", "_connectionOriginToken"):
        if field in note:
            result[field] = note[field]
    return result


def optimize_fingerings(document):
    """Assign pitch-preserving phrase fingerings and report unresolved conflicts."""
    if not isinstance(document, dict) or not isinstance(document.get("metadata"), dict) or not isinstance(document.get("notes"), list):
        raise HarnessError("Fingering optimization requires hypothesis metadata and notes.")
    tuning = document["metadata"].get("openStringMidi")
    capo = document["metadata"].get("capoFret")
    if not isinstance(tuning, list) or len(tuning) != 6 or any(type(value) is not int for value in tuning) or type(capo) is not int or not 0 <= capo <= MAX_FRET:
        raise HarnessError("Fingering optimization requires six-string tuning and full capo.")
    result = deepcopy(document)
    grouped = defaultdict(list)
    token_indices = {}
    for index, note in enumerate(result["notes"]):
        if not isinstance(note, dict) or any(type(note.get(field)) is not int for field in ("soundingPitchMidi", "string", "fret")):
            raise HarnessError("Fingering optimization requires integer pitch, string and fret hypotheses.")
        if type(note.get("confidence")) not in (int, float) or not math.isfinite(note["confidence"]) or not 0 <= note["confidence"] <= 1:
            raise HarnessError("Fingering optimization requires finite confidence from zero to one.")
        seconds = note.get("onsetSeconds")
        if seconds is not None and (type(seconds) not in (int, float) or not math.isfinite(seconds)):
            raise HarnessError("Fingering optimization requires finite onset seconds.")
        token = note.get("_connectionToken")
        if token is not None:
            if not isinstance(token, str) or not token or token in token_indices:
                raise HarnessError("Connection tokens must be unique nonempty strings.")
            token_indices[token] = index
        origin = note.get("_connectionOriginToken")
        if origin is not None and (not isinstance(origin, str) or not origin):
            raise HarnessError("Connection origin tokens must be nonempty strings.")
        grouped[_onset(note)].append((index, note))
    removed, changes, difficult, movements = [], [], [], []
    harmonic_reports, connection_conflicts, grace_reports = [], [], []
    groups, groups_by_index = [], {}
    for onset, values in sorted(grouped.items()):
        unique, duplicates = _deduplicate(values)
        removed.extend(duplicates)
        seconds = [note["onsetSeconds"] for _, note in values if note.get("onsetSeconds") is not None]
        group = {
            "onset": onset, "seconds": min(seconds) if seconds else None,
            "notes": dict(unique), "candidates": {}, "graces": {},
        }
        for index, note in unique:
            group["candidates"][index] = _note_candidates(
                index, note, tuning, capo, harmonic_reports,
            )
            grace = _grace_spec(note, tuning, capo)
            if grace is not None:
                group["graces"][index] = grace
        witness = _maximum_retention(group, group["candidates"])
        if group["graces"]:
            grace_candidates = {
                index: [
                    candidate for candidate in values
                    if index not in group["graces"] or group["graces"][index]["reason"] is not None
                    or candidate[0] in group["graces"][index]["sourceFrets"]
                ]
                for index, values in group["candidates"].items()
            }
            grace_witness = _maximum_retention(group, grace_candidates)
            if len(grace_witness) == len(witness):
                witness = grace_witness
        group["witness"] = witness
        groups.append(group)
        groups_by_index.update({index: group for index, _ in unique})
    # Tokens referring to a deduplicated origin remain unresolved, not rebound
    # to another equal-pitch note.
    surviving_tokens = {token: index for token, index in token_indices.items() if index in groups_by_index}
    committed, statuses, previous_hand = {}, defaultdict(int), None
    for start in range(0, len(groups), PHRASE_COMMIT_ONSETS):
        window = groups[start:start + PHRASE_WINDOW_ONSETS]
        assignments, status = _solve_window(window, committed, groups_by_index, surviving_tokens, previous_hand)
        statuses[status] += 1
        for group in window[:PHRASE_COMMIT_ONSETS]:
            for index in group["notes"]:
                if index in assignments:
                    committed[index] = assignments[index]
            fretted = [_hand_fret(group["notes"][index], committed[index]) for index in group["notes"] if index in committed]
            fretted = [fret for fret in fretted if fret]
            if fretted:
                previous_hand = {**group, "position": _hand_position(fretted)}
    previous_hand = None
    for group in groups:
        onset = group["onset"]
        assignments = {index: committed[index] for index in group["notes"] if index in committed}
        for index, note in group["notes"].items():
            if index not in assignments:
                reason = "unplayable_chord_voice"
                if not group["candidates"][index]:
                    reason = "harmonic_position_outside_guitar_range" if note.get("harmonic") is not None else "pitch_has_no_playable_string"
                removed.append({"index": index, "reason": reason})
                continue
            string, fret = assignments[index]
            if (note["string"], note["fret"]) != (string, fret):
                changes.append({
                    **_provenance(index, note, onset), "selectedString": string, "selectedFret": fret,
                    "reason": "joint_phrase_assignment",
                })
                note["string"], note["fret"] = string, fret
            if note.get("harmonic") is None:
                note["fretBasePitchMidi"] = tuning[6 - string] + capo + fret
            grace = group["graces"].get(index)
            if grace is not None:
                original = document["notes"][index]
                required_source_fret = grace["sourcePitch"] - tuning[6 - string] - capo
                source_fret = grace["sourceFrets"].get(string)
                reason = grace["reason"]
                if reason is None and source_fret is None:
                    reason = (
                        "grace_source_below_selected_open_string" if required_source_fret < 0
                        else "grace_source_above_selected_fret_limit"
                    )
                changed = reason is None and (
                    original["string"] != string or original["grace"].get("sourceFret") != source_fret
                    or original["grace"].get("sourcePitchMidi") != grace["sourcePitch"]
                )
                grace_reports.append({
                    **_provenance(index, original, onset), "grace": deepcopy(original["grace"]),
                    "status": "suppressed" if reason else "retained", "changed": changed,
                    "reason": reason or "same_string_grace_source_assignment",
                    "selectedString": string, "selectedFret": fret, "selectedSourceFret": source_fret,
                    "requiredSourceFret": required_source_fret, "sourcePitchMidi": grace["sourcePitch"],
                    "intervalSemitones": original["grace"]["intervalSemitones"],
                })
                if reason:
                    note["grace"] = None
                else:
                    note["grace"]["sourceFret"] = source_fret
                    note["grace"]["sourcePitchMidi"] = grace["sourcePitch"]
        fretted = [_hand_fret(group["notes"][index], candidate) for index, candidate in assignments.items()]
        fretted = [fret for fret in fretted if fret]
        if fretted:
            position = _hand_position(fretted)
            if previous_hand is not None:
                quarter, seconds = _elapsed(previous_hand, group)
                shift = abs(position - previous_hand["position"])
                if shift > MAX_RAPID_POSITION_SHIFT and (
                    (seconds is not None and seconds <= .5)
                    or (seconds is None and quarter <= RAPID_SHIFT_WINDOW_QUARTER)
                ):
                    movements.append({
                        "onsetQuarter": _rational(onset), "previousOnsetQuarter": _rational(previous_hand["onset"]),
                        "previousPosition": previous_hand["position"], "selectedPosition": position,
                        "shiftFrets": shift, "elapsedQuarter": _rational(quarter),
                        "elapsedSeconds": seconds, "fretsPerSecond": shift / seconds if seconds is not None else None,
                        "reason": "rapid_movement_in_selected_phrase",
                        "noteIndices": sorted(assignments),
                    })
            previous_hand = {**group, "position": position}
        by_fret = defaultdict(list)
        for index, (string, fret) in assignments.items():
            if fret and group["notes"][index].get("harmonic") is None:
                by_fret[fret].append(string)
        for fret, strings in by_fret.items():
            if len(strings) >= 4:
                difficult.append({
                    "onsetQuarter": _rational(onset),
                    "reason": "extended_barre_in_selected_fingering",
                    "fret": fret, "strings": sorted(strings, reverse=True),
                    "noteIndices": sorted(index for index, value in assignments.items() if value[1] == fret),
                })
    for index, original in enumerate(document["notes"]):
        origin_token = original.get("_connectionOriginToken")
        if origin_token is None:
            continue
        origin = token_indices.get(origin_token)
        reason = None
        if origin is None:
            reason = "connection_origin_token_missing"
        elif origin not in committed or index not in committed:
            reason = "connection_endpoint_not_retained"
        elif groups_by_index[origin]["onset"] >= groups_by_index[index]["onset"]:
            reason = "connection_origin_not_earlier"
        elif document["notes"][origin].get("voiceIndex", 0) != original.get("voiceIndex", 0):
            reason = "connection_voice_mismatch"
        elif committed[origin][0] != committed[index][0]:
            reason = "connection_requires_different_strings"
        elif any(
            groups_by_index[origin]["onset"] < groups_by_index[other]["onset"] < groups_by_index[index]["onset"]
            and committed[other][0] == committed[index][0]
            and document["notes"][other].get("voiceIndex", 0) == original.get("voiceIndex", 0)
            for other in committed
        ):
            reason = "connection_has_intervening_same_string_attack"
        if reason:
            connection_conflicts.append({**_provenance(index, original, _onset(original)), "reason": reason, "originIndex": origin})
    removed = [
        {**_provenance(value["index"], document["notes"][value["index"]], _onset(document["notes"][value["index"]])), **value}
        for value in removed
    ]
    grace_reports.extend({
        **_provenance(value["index"], document["notes"][value["index"]], _onset(document["notes"][value["index"]])),
        "grace": deepcopy(document["notes"][value["index"]]["grace"]),
        "status": "anchor_not_retained", "changed": False, "reason": value["reason"],
        "selectedString": None, "selectedFret": None, "selectedSourceFret": None,
        "sourcePitchMidi": document["notes"][value["index"]]["grace"].get("sourcePitchMidi"),
        "intervalSemitones": document["notes"][value["index"]]["grace"].get("intervalSemitones"),
    } for value in removed if document["notes"][value["index"]].get("grace") is not None)
    grace_reports.sort(key=lambda value: value["index"])
    result["notes"] = [note for index, note in enumerate(result["notes"]) if index in committed]
    return result, {
        "schemaVersion": 2, "kind": "playable-phrase-fingering-optimization",
        "solver": "OR-Tools CP-SAT", "objectiveOrder": [
            "maximum_chord_pitch_retention", "bound_connection_pairs",
            "same_string_grace_gestures", "phrase_likelihood_and_timed_motion",
        ],
        "optimizationScope": "overlapping_bounded_phrase_windows",
        "maximumChordRetentionProven": True,
        "motionTimeBasis": "increasing_onset_seconds_else_score_quarter",
        "handPositionStatistic": "upper_median_of_non_open_contacts",
        "phraseWindowOnsets": PHRASE_WINDOW_ONSETS, "phraseCommitOnsets": PHRASE_COMMIT_ONSETS,
        "maximumFret": MAX_FRET, "maximumFrettedChordSpan": MAX_FRETTED_SPAN,
        "maximumRapidPositionShift": MAX_RAPID_POSITION_SHIFT,
        "rapidShiftPolicy": "report_only_never_drop",
        "rapidShiftWindowQuarter": _rational(RAPID_SHIFT_WINDOW_QUARTER),
        "rapidShiftWindowSeconds": .5, "repeatedPitchContinuityQuarter": [3, 2],
        "sourceNoteCount": len(document["notes"]), "retainedNoteCount": len(result["notes"]),
        "changedCount": len(changes), "removedCount": len(removed),
        "difficultChordCount": len(difficult), "difficultMovementCount": len(movements),
        "harmonicPositionCount": len(harmonic_reports),
        "connectionConflictCount": len(connection_conflicts),
        "graceFingeringCount": len(grace_reports),
        "graceChangedCount": sum(value["changed"] for value in grace_reports),
        "graceConflictCount": sum(value["status"] == "suppressed" for value in grace_reports),
        "statuses": dict(sorted(statuses.items())),
        "changes": changes, "removed": removed, "difficultChords": difficult,
        "difficultMovements": movements, "harmonicPositions": harmonic_reports,
        "connectionConflicts": connection_conflicts,
        "graceFingerings": grace_reports,
        "rawHypothesesModified": False,
    }
