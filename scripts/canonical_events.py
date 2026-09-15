"""Construct voice-preserving logical notes and source-evidenced gestures."""

from collections import Counter
from copy import deepcopy
from fractions import Fraction

from .gp_events import rational, validate_provided_timing


class CanonicalLabelError(ValueError):
    """Extracted events or interpretation rules are inconsistent."""


def fraction(value, label):
    if not isinstance(value, list) or len(value) != 2 or any(type(part) is not int for part in value) or value[0] < 0 or value[1] <= 0:
        raise CanonicalLabelError(f"Invalid nonnegative rational {label}: {value!r}")
    return Fraction(*value)


def normalized_text(value):
    return " ".join((value or "").split())


def source_indices(score):
    beats, notes = {}, {}
    for beat in score["scoreEvents"]:
        if beat["id"] in beats:
            raise CanonicalLabelError(f"Duplicate written beat: {beat['id']}")
        beats[beat["id"]] = beat
        for note in beat["notes"]:
            if note["id"] in notes:
                raise CanonicalLabelError(f"Duplicate written note: {note['id']}")
            notes[note["id"]] = (beat, note)
    return beats, notes


def validate_rules(score, annotations, beats):
    if annotations is None:
        return []
    if not isinstance(annotations, dict) or annotations.get("gpSha256") != score["sourceGpSha256"]:
        raise CanonicalLabelError("Percussion interpretations belong to another GP revision.")
    rules = annotations.get("rules")
    if not isinstance(rules, list):
        raise CanonicalLabelError("Percussion rules must be a list.")
    identifiers = set()
    for rule in rules:
        if not isinstance(rule, dict) or {"id", "technique", "evidenceBeatIds", "match", "consumesDeadNotes"} - set(rule):
            raise CanonicalLabelError("A percussion rule is missing required fields.")
        if not isinstance(rule["id"], str) or not rule["id"] or rule["id"] in identifiers:
            raise CanonicalLabelError("Percussion rule IDs must be unique.")
        identifiers.add(rule["id"])
        if not isinstance(rule["technique"], str) or not rule["technique"] or not isinstance(rule["evidenceBeatIds"], list) or not rule["evidenceBeatIds"]:
            raise CanonicalLabelError("A gesture requires a technique and reference evidence.")
        for identifier in rule["evidenceBeatIds"]:
            if not isinstance(identifier, str) or identifier not in beats or not beats[identifier]["referenceOnly"]:
                raise CanonicalLabelError(f"Gesture evidence is not a reference beat: {identifier}")
        match = rule["match"]
        if not isinstance(match, dict) or not match or set(match) - {"tokens", "text", "deadStrings", "beatTechniques", "deadNoteMarks"}:
            raise CanonicalLabelError(f"Unsupported or empty gesture match: {rule['id']}")
        if not any(match.get(key) for key in ("tokens", "text", "deadStrings", "beatTechniques")):
            raise CanonicalLabelError(f"Gesture rule would match every beat: {rule['id']}")
        if "tokens" in match and (not isinstance(match["tokens"], list) or not match["tokens"] or any(not isinstance(token, str) or not token or len(token.split()) != 1 for token in match["tokens"])):
            raise CanonicalLabelError(f"Invalid gesture tokens: {rule['id']}")
        if "text" in match and not isinstance(match["text"], str) or "beatTechniques" in match and not isinstance(match["beatTechniques"], dict):
            raise CanonicalLabelError(f"Invalid gesture text or articulations: {rule['id']}")
        if "deadNoteMarks" in match and (not isinstance(match["deadNoteMarks"], list) or any(not isinstance(mark, dict) or set(mark) != {"string", "techniques", "harmonic", "instrumentArticulation"} for mark in match["deadNoteMarks"])):
            raise CanonicalLabelError(f"Invalid dead-note articulations: {rule['id']}")
        if "deadStrings" in match and (not isinstance(match["deadStrings"], list) or any(type(string) is not int or not 1 <= string <= 6 for string in match["deadStrings"]) or len(set(match["deadStrings"])) != len(match["deadStrings"])):
            raise CanonicalLabelError(f"Invalid symbolic strings: {rule['id']}")
        if type(rule["consumesDeadNotes"]) is not bool:
            raise CanonicalLabelError(f"Missing explicit dead-note handling: {rule['id']}")
        if "consumedTokens" in rule and (not isinstance(rule["consumedTokens"], list) or any(not isinstance(token, str) or not token for token in rule["consumedTokens"])):
            raise CanonicalLabelError(f"Invalid consumed markers: {rule['id']}")
        available_tokens = set(match.get("tokens", normalized_text(match.get("text")).split()))
        if not rule_tokens(rule) <= available_tokens:
            raise CanonicalLabelError(f"Rule consumes markers it does not match: {rule['id']}")
        if "attributes" in rule and not isinstance(rule["attributes"], dict):
            raise CanonicalLabelError(f"Invalid gesture attributes: {rule['id']}")
    return rules


def matches_rule(beat, rule):
    match = rule["match"]
    text = normalized_text(beat["text"])
    tokens = Counter(text.split())
    expected_tokens = Counter(match.get("tokens", []))
    dead_strings = sorted(note["string"] for note in beat["notes"] if note["techniques"]["dead"])
    return (
        ("text" not in match or text == normalized_text(match["text"]))
        and ("tokens" not in match or tokens == expected_tokens)
        and ("deadStrings" not in match or dead_strings == sorted(match["deadStrings"]))
        and ("beatTechniques" not in match or beat["techniques"] == match["beatTechniques"])
        and ("deadNoteMarks" not in match or dead_note_marks(beat) == match["deadNoteMarks"])
    )


def dead_note_marks(beat):
    return [{key: note[key] for key in ("string", "techniques", "harmonic", "instrumentArticulation")} for note in sorted(beat["notes"], key=lambda note: note["string"]) if note["techniques"]["dead"]]


def rule_tokens(rule):
    return set(rule.get("consumedTokens", rule["match"].get("tokens", normalized_text(rule["match"].get("text")).split())))


def logical_notes(score, written_notes):
    chains, owners, segments, successors = {}, {}, {}, {}
    for event in score["playback"]["noteEvents"]:
        identifier = event["id"]
        if identifier in segments:
            raise CanonicalLabelError(f"Duplicate playback note: {identifier}")
        if event["sourceEventId"] not in written_notes:
            raise CanonicalLabelError(f"Missing written note: {event['sourceEventId']}")
        beat, note = written_notes[event["sourceEventId"]]
        if beat["referenceOnly"]:
            raise CanonicalLabelError("A reference example entered the performance.")
        onset = fraction(event["onsetQuarter"], identifier)
        duration = event["durationQuarter"]
        if duration is not None and fraction(duration, identifier) <= 0:
            raise CanonicalLabelError(f"Nonpositive note duration: {identifier}")
        if (duration is None) != (beat["graceMode"] is not None):
            raise CanonicalLabelError(f"Grace duration differs from written source: {identifier}")
        if duration is not None and fraction(duration, identifier) != fraction(beat["notatedDurationQuarter"], beat["id"]):
            raise CanonicalLabelError(f"Note duration differs from written source: {identifier}")
        attack = event["isAttack"]
        if attack is not None and type(attack) is not bool:
            raise CanonicalLabelError(f"Invalid attack status: {identifier}")
        prior_id = event["tieFrom"]
        if prior_id is not None:
            if prior_id not in segments or prior_id in successors:
                raise CanonicalLabelError(f"Missing, forward, cyclic, or branching tie: {identifier}")
            prior, prior_beat, prior_note = segments[prior_id]
            prior_end = fraction(prior["onsetQuarter"], prior_id)
            if prior["durationQuarter"] is not None:
                prior_end += fraction(prior["durationQuarter"], prior_id)
            if attack is not False or not note["tie"]["destination"]:
                raise CanonicalLabelError(f"Tie flags contradict the link: {identifier}")
            if (beat["voiceIndex"], note["string"], note["basePitchMidi"]) != (prior_beat["voiceIndex"], prior_note["string"], prior_note["basePitchMidi"]) or prior_end != onset:
                raise CanonicalLabelError(f"Tie crosses voice, string, pitch, or a timing gap: {identifier}")
            root = owners[prior_id]
            successors[prior_id] = identifier
        else:
            if attack is False or (attack is None) != bool(note["tie"]["destination"]):
                raise CanonicalLabelError(f"Attack status contradicts an unlinked note: {identifier}")
            root = identifier
            chains[root] = []
        owners[identifier] = root
        segments[identifier] = (event, beat, note)
        chains[root].append(identifier)

    notes, symbols, issues = [], [], []
    for root, identifiers in chains.items():
        first, beat, source_note = segments[root]
        parts = [segments[identifier] for identifier in identifiers]
        duration_known = first["isAttack"] is True and all(event["durationQuarter"] is not None for event, _, _ in parts)
        if first["isAttack"] is None:
            issues.append({"code": "unresolved_logical_note_start", "performanceId": root})
        pitches = {note["soundingPitchMidi"] for _, _, note in parts}
        dead_states = {note["techniques"]["dead"] for _, _, note in parts}
        if len(dead_states) != 1:
            raise CanonicalLabelError(f"Tie changes between a pitched note and symbolic percussion: {root}")
        pitch_known = first["isAttack"] is True and len(pitches) == 1 and None not in pitches
        if len(pitches) > 1:
            issues.append({"code": "changing_tied_pitch", "performanceId": root})
        observed = sum((fraction(event["durationQuarter"], event["id"]) for event, _, _ in parts if event["durationQuarter"] is not None), Fraction(0))
        dead = source_note["techniques"]["dead"]
        result = {
            "id": root, "voiceIndex": beat["voiceIndex"],
            "onsetQuarter": first["onsetQuarter"], "isAttack": first["isAttack"],
            "notatedDurationQuarter": rational(observed) if duration_known else None,
            "observedDurationQuarter": rational(observed),
            "string": source_note["string"], "fret": source_note["fret"],
            "basePitchMidi": source_note["basePitchMidi"],
            "soundingPitchMidi": next(iter(pitches)) if pitch_known else None,
            "labelMask": {"attack": first["isAttack"] is True, "notatedDuration": duration_known, "pitch": pitch_known and not dead, "fingering": first["isAttack"] is True and not dead},
            "sourceSegments": [
                {
                    "performanceId": event["id"], "writtenNoteId": note["id"], "writtenBeatId": source_beat["id"],
                    "measureIndex": source_beat["measureIndex"], "visitIndex": event["visitIndex"],
                    "onsetQuarter": event["onsetQuarter"], "durationQuarter": event["durationQuarter"],
                    "notatedDurationQuarter": source_beat["notatedDurationQuarter"],
                    "rhythm": source_beat["rhythm"], "graceMode": source_beat["graceMode"],
                    "harmonic": note["harmonic"], "bend": note["bend"], "techniques": note["techniques"],
                    "basePitchMidi": note["basePitchMidi"], "soundingPitchMidi": note["soundingPitchMidi"],
                    "string": note["string"], "fret": note["fret"],
                    "storedMidi": note["storedMidi"], "instrumentArticulation": note["instrumentArticulation"],
                    "tie": note["tie"],
                    "techniqueMask": {key: value is not False and value is not None for key, value in note["techniques"].items()},
                    "beatTechniques": source_beat["techniques"], "dynamic": source_beat["dynamic"],
                }
                for event, source_beat, note in parts
            ],
        }
        (symbols if dead else notes).append(result)
    return notes, symbols, issues, segments


def performance_beats(score):
    for visit in score["playback"]["measureVisits"]:
        index = visit["measureIndex"]
        measure = score["measures"][index]
        if measure["referenceOnly"]:
            raise CanonicalLabelError("A reference measure entered the performance.")
        first, stop = measure["eventSlice"]
        for beat in score["scoreEvents"][first:stop]:
            if beat["measureIndex"] != index or beat["referenceOnly"]:
                raise CanonicalLabelError("Playback measure contains an inconsistent beat slice.")
            yield visit, beat, rational(fraction(visit["onsetQuarter"], "measure visit") + fraction(beat["offsetQuarter"], beat["id"]))


def gesture_events(score, rules, segments):
    gestures, unresolved, rests = [], [], []
    for visit, beat, onset in performance_beats(score):
        performance_id = f"p{visit['visitIndex']}:{beat['id']}"
        if beat["isRest"]:
            rests.append({"id": performance_id, "voiceIndex": beat["voiceIndex"], "onsetQuarter": onset, "notatedDurationQuarter": beat["notatedDurationQuarter"], "writtenBeatId": beat["id"], "measureIndex": beat["measureIndex"], "visitIndex": visit["visitIndex"]})
        matched = [rule for rule in rules if matches_rule(beat, rule)]
        dead_ids = [f"p{visit['visitIndex']}:{note['id']}" for note in beat["notes"] if note["techniques"]["dead"]]
        candidates = []
        for rule in matched:
            if rule["consumesDeadNotes"] and dead_ids:
                attacks = [segments[identifier][0]["isAttack"] for identifier in dead_ids]
                if all(attack is False for attack in attacks):
                    continue
                if any(attack is not True for attack in attacks):
                    continue
            candidates.append(rule)
        conflicting = set()
        for index, first_rule in enumerate(candidates):
            first_tokens = rule_tokens(first_rule)
            for second_rule in candidates[index + 1:]:
                second_tokens = rule_tokens(second_rule)
                if first_tokens & second_tokens or dead_ids and first_rule["consumesDeadNotes"] and second_rule["consumesDeadNotes"]:
                    conflicting.update((first_rule["id"], second_rule["id"]))
        if conflicting:
            unresolved.append({"performanceBeatId": performance_id, "onsetQuarter": onset, "reason": "ambiguous_gesture_rules", "ruleIds": sorted(conflicting), "symbolicNoteIds": dead_ids})
            candidates = [rule for rule in candidates if rule["id"] not in conflicting]
        consumed = set()
        for rule in candidates:
            consumed.update(rule_tokens(rule))
            gestures.append({
                "id": f"{performance_id}:{rule['id']}", "technique": rule["technique"],
                "attributes": rule.get("attributes", {}),
                "voiceIndex": beat["voiceIndex"], "onsetQuarter": onset, "durationQuarter": None,
                "writtenBeatId": beat["id"], "visitIndex": visit["visitIndex"],
                "symbolicNoteIds": dead_ids if rule["consumesDeadNotes"] else [],
                "interpretationRuleId": rule["id"], "evidenceBeatIds": rule["evidenceBeatIds"],
            })
        uncovered_dead = dead_ids and not any(rule["consumesDeadNotes"] for rule in candidates)
        if uncovered_dead and not conflicting and any(segments[identifier][0]["isAttack"] is not False for identifier in dead_ids):
            unresolved.append({"performanceBeatId": performance_id, "onsetQuarter": onset, "reason": "uninterpreted_dead_note_cluster", "symbolicNoteIds": dead_ids})
        if set(normalized_text(beat["text"]).split()) - consumed:
            unresolved.append({"performanceBeatId": performance_id, "onsetQuarter": onset, "reason": "uninterpreted_annotation", "writtenBeatId": beat["id"]})
    return gestures, unresolved, rests


def canonicalize(score, annotations=None):
    if score["schemaVersion"] != 1 or score["timeUnit"] != "quarter-note" or score["playback"] is None:
        raise CanonicalLabelError("Canonical labels require version-1 score events with resolved playback.")
    if score.get("audioAlignment") is not None:
        raise CanonicalLabelError("Prepare canonical labels before attaching audio alignment.")
    validate_provided_timing(score["providedTiming"]["tempo"], score["providedTiming"]["timeSignature"])
    instrument = score["instrument"]
    if instrument["stringOrder"] != [6, 5, 4, 3, 2, 1] or len(instrument["openStringMidi"]) != 6 or any(type(pitch) is not int or not 0 <= pitch <= 127 for pitch in instrument["openStringMidi"]):
        raise CanonicalLabelError("Canonical labels require six explicit bass-to-treble string pitches.")
    if type(instrument["capoFret"]) is not int or not 0 <= instrument["capoFret"] <= 24 or instrument["fretConvention"] != "capo-relative":
        raise CanonicalLabelError("Canonical labels require an explicit full capo and normalized frets.")
    beats, written_notes = source_indices(score)
    expected = {}
    for visit, beat, onset in performance_beats(score):
        for note in beat["notes"]:
            identifier = f"p{visit['visitIndex']}:{note['id']}"
            if identifier in expected:
                raise CanonicalLabelError(f"Duplicate playback visit: {identifier}")
            expected[identifier] = (note["id"], visit["visitIndex"], fraction(onset, identifier))
    for event in score["playback"]["noteEvents"]:
        actual = (event["sourceEventId"], event["visitIndex"], fraction(event["onsetQuarter"], event["id"]))
        if expected.get(event["id"]) != actual:
            raise CanonicalLabelError(f"Playback note disagrees with its written occurrence: {event['id']}")
    if set(expected) != {event["id"] for event in score["playback"]["noteEvents"]}:
        raise CanonicalLabelError("Playback omits written note occurrences.")
    rules = validate_rules(score, annotations, beats)
    notes, symbols, issues, segments = logical_notes(score, written_notes)
    gestures, unresolved, rests = gesture_events(score, rules, segments)
    timing_resolved = not any(issue["code"] in {"underfull_measure", "overfull_measure", "unresolved_playback_order"} for issue in score["issues"])
    output = {
        "schemaVersion": 1, "catalogId": score["catalogId"], "timeUnit": "quarter-note",
        "audioAlignment": None, "scoreTimingResolved": timing_resolved,
        "supervisionPolicy": {"unmarkedTechniquesAreNegatives": False, "gestureAbsenceSupervised": False, "restsAreAcousticSilence": False},
        "conditioning": {
            "instrument": {key: instrument[key] for key in ("stringOrder", "openStringMidi", "capoFret")},
            "providedTiming": {key: score["providedTiming"][key] for key in ("tempo", "timeSignature", "sourceTempoChanges", "sourceTimeSignatureChanges")},
        },
        "measureVisits": score["playback"]["measureVisits"],
        "targets": {"notes": notes, "rests": rests, "gestures": gestures},
        "review": {"notationSymbols": symbols, "unresolvedGestures": unresolved, "issues": issues, "sourceIssues": score["issues"]},
        "provenance": {"sourceGpSha256": score["sourceGpSha256"], "fretConvention": instrument["fretConvention"], "ruleIds": [rule["id"] for rule in rules], "referenceBeatIds": [beat["id"] for beat in score["scoreEvents"] if beat["referenceOnly"]]},
    }
    return deepcopy(output)


def canonical_counts(labels):
    notes = labels["targets"]["notes"]
    symbols = labels["review"]["notationSymbols"]
    unresolved = Counter(item["reason"] for item in labels["review"]["unresolvedGestures"])
    return {
        "logicalNoteCount": len(notes),
        "symbolicNoteCount": len(symbols),
        "sourceSegmentCount": sum(len(note["sourceSegments"]) for note in notes + symbols),
        "mergedContinuationCount": sum(len(note["sourceSegments"]) - 1 for note in notes + symbols),
        "unknownAttackCount": sum(note["isAttack"] is None for note in notes + symbols),
        "unknownDurationCount": sum(not note["labelMask"]["notatedDuration"] for note in notes + symbols),
        "gestureCount": len(labels["targets"]["gestures"]),
        "gestureTypes": dict(sorted(Counter(gesture["technique"] for gesture in labels["targets"]["gestures"]).items())),
        "unresolvedGestureCount": len(labels["review"]["unresolvedGestures"]),
        "uninterpretedClusterCount": unresolved["uninterpreted_dead_note_cluster"],
        "uninterpretedAnnotationCount": unresolved["uninterpreted_annotation"],
        "ambiguousGestureCount": unresolved["ambiguous_gesture_rules"],
        "scoreTimingResolved": labels["scoreTimingResolved"],
    }
