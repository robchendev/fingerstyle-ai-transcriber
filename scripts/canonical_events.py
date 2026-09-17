"""Construct voice-preserving logical notes and source-evidenced gestures."""

from collections import Counter
from copy import deepcopy
from fractions import Fraction
import json

from .gp_events import rational, validate_provided_timing
from . import settings


class CanonicalLabelError(ValueError):
    """Extracted events or interpretation rules are inconsistent."""


def fraction(value, label):
    if not isinstance(value, list) or len(value) != 2 or any(type(part) is not int for part in value) or value[0] < 0 or value[1] <= 0:
        raise CanonicalLabelError(f"Invalid nonnegative rational {label}: {value!r}")
    return Fraction(*value)


def normalized_text(value):
    return " ".join((value or "").split())


def percussive_hit_text_max_len():
    limit = settings.PERCUSSIVE_HIT_TEXT_DETECTION_MAX_LEN
    if type(limit) is not int or limit < 1:
        raise CanonicalLabelError("PERCUSSIVE_HIT_TEXT_DETECTION_MAX_LEN must be a positive integer.")
    return limit


def is_percussive_hit_text(value):
    limit = percussive_hit_text_max_len()
    text = normalized_text(value)
    return bool(text) and len(text) <= limit and set(text.replace(" ", "")) != {"?"}


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
        if not isinstance(match, dict) or not match or set(match) - {"tokens", "text", "deadStrings", "beatTechniques", "deadNoteMarks", "pitchedNoteMarks", "simultaneousPitchedNoteMarks", "writtenBeatIds"}:
            raise CanonicalLabelError(f"Unsupported or empty gesture match: {rule['id']}")
        if not any(match.get(key) for key in ("tokens", "text", "deadStrings", "beatTechniques", "pitchedNoteMarks", "simultaneousPitchedNoteMarks")):
            raise CanonicalLabelError(f"Gesture rule would match every beat: {rule['id']}")
        if "tokens" in match and (not isinstance(match["tokens"], list) or not match["tokens"] or any(not isinstance(token, str) or not token or len(token.split()) != 1 for token in match["tokens"])):
            raise CanonicalLabelError(f"Invalid gesture tokens: {rule['id']}")
        if "text" in match and not isinstance(match["text"], str) or "beatTechniques" in match and not isinstance(match["beatTechniques"], dict):
            raise CanonicalLabelError(f"Invalid gesture text or articulations: {rule['id']}")
        for field in ("deadNoteMarks", "pitchedNoteMarks", "simultaneousPitchedNoteMarks"):
            if field in match and (not isinstance(match[field], list) or any(not isinstance(mark, dict) or set(mark) != {"string", "techniques", "harmonic", "instrumentArticulation"} for mark in match[field])):
                raise CanonicalLabelError(f"Invalid note articulations: {rule['id']}")
            for mark in match.get(field, []):
                if type(mark["string"]) is not int or not 1 <= mark["string"] <= 6 or not isinstance(mark["techniques"], dict) or mark["techniques"].get("dead") is not (field == "deadNoteMarks"):
                    raise CanonicalLabelError(f"Invalid note kind/string in rule: {rule['id']}")
                harmonic = mark["harmonic"]
                if harmonic is not None:
                    if not isinstance(harmonic, dict) or set(harmonic) != {"type", "fret"} or not isinstance(harmonic["type"], str) or not harmonic["type"]:
                        raise CanonicalLabelError(f"Invalid harmonic predicate: {rule['id']}")
                    if fraction(harmonic["fret"], rule["id"]) <= 0:
                        raise CanonicalLabelError(f"Nonpositive harmonic fret in rule: {rule['id']}")
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
        if "writtenBeatIds" in match and (not isinstance(match["writtenBeatIds"], list) or not match["writtenBeatIds"] or any(not isinstance(identifier, str) or identifier not in beats or beats[identifier]["referenceOnly"] for identifier in match["writtenBeatIds"])):
            raise CanonicalLabelError(f"Invalid musical-beat scope: {rule['id']}")
    return rules


def matches_rule(beat, rule, simultaneous_notes=None):
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
        and ("pitchedNoteMarks" not in match or pitched_note_marks(beat) == match["pitchedNoteMarks"])
        and ("simultaneousPitchedNoteMarks" not in match or simultaneous_notes is not None and note_marks(simultaneous_notes, dead=False) == match["simultaneousPitchedNoteMarks"])
        and ("writtenBeatIds" not in match or beat["id"] in match["writtenBeatIds"])
    )


def note_marks(notes, *, dead):
    marks = [{key: note[key] for key in ("string", "techniques", "harmonic", "instrumentArticulation")} for note in notes if note["techniques"]["dead"] is dead]
    return sorted(marks, key=lambda mark: (mark["string"], json.dumps(mark, sort_keys=True)))


def dead_note_marks(beat):
    return note_marks(beat["notes"], dead=True)


def pitched_note_marks(beat):
    return note_marks(beat["notes"], dead=False)


def rule_tokens(rule):
    return set(rule.get("consumedTokens", rule["match"].get("tokens", normalized_text(rule["match"].get("text")).split())))


def logical_notes(score, written_notes):
    chains, owners, segments, successors = {}, {}, {}, {}
    silenced = []
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
        if event.get("isSilent"):
            if score["playback"].get("silenceRepeatedEntryTies") is not True or event.get("isSilent") is not True or event["isAttack"] is not False or event["tieFrom"] is not None or not note["tie"]["destination"] or event.get("suppressionReason") not in {"repeated_entry_tie_without_original_origin", "continuation_of_silenced_repeat_entry"} or duration is None:
                raise CanonicalLabelError(f"Invalid silenced repeat-entry tie: {identifier}")
            segments[identifier] = (event, beat, note)
            silenced.append({
                "performanceId": identifier, "writtenNoteId": note["id"], "writtenBeatId": beat["id"],
                "voiceIndex": beat["voiceIndex"], "measureIndex": beat["measureIndex"], "visitIndex": event["visitIndex"],
                "onsetQuarter": event["onsetQuarter"], "durationQuarter": duration,
                "sourceString": note["string"], "sourceFret": note["fret"], "reason": event["suppressionReason"],
            })
            continue
        attack = event["isAttack"]
        if attack is not None and type(attack) is not bool:
            raise CanonicalLabelError(f"Invalid attack status: {identifier}")
        prior_id = event["tieFrom"]
        if prior_id is not None:
            if prior_id not in owners or prior_id in successors:
                raise CanonicalLabelError(f"Missing, forward, cyclic, or branching tie: {identifier}")
            prior, prior_beat, prior_note = segments[prior_id]
            prior_end = fraction(prior["onsetQuarter"], prior_id)
            if prior["durationQuarter"] is not None:
                prior_end += fraction(prior["durationQuarter"], prior_id)
            if attack is not False or not note["tie"]["destination"]:
                raise CanonicalLabelError(f"Tie flags contradict the link: {identifier}")
            same_kind = note["techniques"]["dead"] == prior_note["techniques"]["dead"]
            same_pitch = note["techniques"]["dead"] or note["basePitchMidi"] == prior_note["basePitchMidi"]
            if (beat["voiceIndex"], note["string"]) != (prior_beat["voiceIndex"], prior_note["string"]) or not same_kind or not same_pitch or prior_end != onset:
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
    return notes, symbols, issues, segments, silenced


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


def performance_slot_keys(beats):
    grace_ordinals = Counter()
    keys = []
    for visit, beat, onset in beats:
        position = (visit["visitIndex"], fraction(onset, beat["id"]), beat["graceMode"])
        ordinal = 0
        if beat["graceMode"] is not None:
            voice_position = (*position, beat["voiceIndex"])
            ordinal = grace_ordinals[voice_position]
            grace_ordinals[voice_position] += 1
        keys.append((*position, ordinal))
    return keys


def normalized_performance_text(score, conventions):
    beats = list(performance_beats(score))
    if conventions is None:
        return beats, []
    if conventions.get("schemaVersion") != 1 or conventions.get("uppercaseOIsWristThump") is not True or conventions.get("simultaneousTextPriority") != "lowest-voice-index":
        raise CanonicalLabelError("Unsupported owner notation conventions.")
    groups = {}
    for index, ((visit, beat, onset), slot) in enumerate(zip(beats, performance_slot_keys(beats))):
        if normalized_text(beat["text"]):
            groups.setdefault(slot, []).append(index)
    overrides, suppressed = {}, []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        voices = [beats[index][1]["voiceIndex"] for index in indices]
        if len(voices) != len(set(voices)):
            raise CanonicalLabelError("Multiple annotated regular beats share one voice/onset.")
        winner = min(indices, key=lambda index: beats[index][1]["voiceIndex"])
        chosen = beats[winner][1]
        for index in indices:
            if index == winner:
                continue
            visit, beat, onset = beats[index]
            overrides[index] = {**beat, "text": None}
            suppressed.append({
                "performanceBeatId": f"p{visit['visitIndex']}:{beat['id']}",
                "writtenBeatId": beat["id"], "voiceIndex": beat["voiceIndex"],
                "onsetQuarter": onset, "rawText": beat["text"],
                "effectiveTextBeatId": chosen["id"], "effectiveText": chosen["text"],
                "reason": "higher_priority_voice_text",
            })
    return [(visit, overrides.get(index, beat), onset) for index, (visit, beat, onset) in enumerate(beats)], suppressed


def gesture_events(score, rules, segments, conventions=None):
    gestures, unresolved, rests = [], [], []
    performed, suppressed = normalized_performance_text(score, conventions)
    filtered = []
    for visit, beat, onset in performed:
        notes = [note for note in beat["notes"] if not segments[f"p{visit['visitIndex']}:{note['id']}"][0].get("isSilent")]
        filtered.append((visit, {**beat, "notes": notes, "isRest": not notes}, onset))
    performed = filtered
    suppressed_ids = {item["performanceBeatId"] for item in suppressed}
    slots = performance_slot_keys(performed)
    simultaneous = {}
    for (visit, beat, _), slot in zip(performed, slots):
        simultaneous.setdefault(slot, []).extend((f"p{visit['visitIndex']}:{note['id']}", note) for note in beat["notes"])
    emitted = set()
    for (visit, beat, onset), slot in zip(performed, slots):
        performance_id = f"p{visit['visitIndex']}:{beat['id']}"
        if beat["isRest"]:
            rests.append({"id": performance_id, "voiceIndex": beat["voiceIndex"], "onsetQuarter": onset, "notatedDurationQuarter": beat["notatedDurationQuarter"], "writtenBeatId": beat["id"], "measureIndex": beat["measureIndex"], "visitIndex": visit["visitIndex"]})
        # A hidden annotation must not be reinterpreted as an unmarked pattern.
        context = simultaneous[slot]
        matched = [] if performance_id in suppressed_ids else [rule for rule in rules if matches_rule(beat, rule, [note for _, note in context])]
        owner_wrist = conventions is not None and normalized_text(beat["text"]) == "O"
        if owner_wrist:
            gestures.append({
                "id": f"{performance_id}:owner-uppercase-O", "technique": "wrist_thump",
                "attributes": {}, "voiceIndex": beat["voiceIndex"], "onsetQuarter": onset,
                "durationQuarter": None, "writtenBeatId": beat["id"], "visitIndex": visit["visitIndex"],
                "graceMode": beat["graceMode"], "scoreOnsetKnown": beat["graceMode"] is None,
                "symbolicNoteIds": [], "interpretationRuleId": "owner-uppercase-O",
                "evidenceBeatIds": [], "evidenceSource": "owner-notation-conventions",
            })
            matched = [rule for rule in matched if not (rule["technique"] == "wrist_thump" and "O" in rule_tokens(rule))]
        dead_ids = [f"p{visit['visitIndex']}:{note['id']}" for note in beat["notes"] if note["techniques"]["dead"]]
        text = normalized_text(beat["text"])
        if not matched and not owner_wrist and "O" not in text.split() and is_percussive_hit_text(text):
            matched.append({
                "id": "owner-short-text-hit", "technique": "percussive_hit",
                "attributes": {}, "evidenceBeatIds": [],
                "match": {"text": text}, "consumedTokens": text.split(),
                "consumesDeadNotes": bool(dead_ids),
                "evidenceSource": "owner-short-text-length-policy",
            })
        candidates = []
        for rule in matched:
            if any(field in rule["match"] for field in ("pitchedNoteMarks", "simultaneousPitchedNoteMarks")):
                relevant = [(f"p{visit['visitIndex']}:{note['id']}", note) for note in beat["notes"]] if "pitchedNoteMarks" in rule["match"] else context
                attacks = [segments[identifier][0]["isAttack"] for identifier, note in relevant if not note["techniques"]["dead"]]
                if not attacks or any(attack is not True for attack in attacks):
                    continue
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
                shared_pitched_scope = all(any(field in rule["match"] for field in ("pitchedNoteMarks", "simultaneousPitchedNoteMarks")) for rule in (first_rule, second_rule))
                if first_tokens & second_tokens or dead_ids and first_rule["consumesDeadNotes"] and second_rule["consumesDeadNotes"] or shared_pitched_scope:
                    conflicting.update((first_rule["id"], second_rule["id"]))
        if conflicting:
            unresolved.append({"performanceBeatId": performance_id, "onsetQuarter": onset, "reason": "ambiguous_gesture_rules", "ruleIds": sorted(conflicting), "symbolicNoteIds": dead_ids})
            candidates = [rule for rule in candidates if rule["id"] not in conflicting]
        consumed = {"O"} if owner_wrist else set()
        for rule in candidates:
            consumed.update(rule_tokens(rule))
            grouped = "simultaneousPitchedNoteMarks" in rule["match"]
            key = (slot, rule["id"])
            if grouped and key in emitted:
                continue
            if grouped:
                emitted.add(key)
            pitched = [(f"p{visit['visitIndex']}:{note['id']}", note) for note in beat["notes"]] if "pitchedNoteMarks" in rule["match"] else context if grouped else []
            gesture = {
                "id": f"{performance_id}:{rule['id']}", "technique": rule["technique"],
                "attributes": rule.get("attributes", {}),
                "voiceIndex": beat["voiceIndex"], "onsetQuarter": onset, "durationQuarter": None,
                "graceMode": beat["graceMode"], "scoreOnsetKnown": beat["graceMode"] is None,
                "writtenBeatId": beat["id"], "visitIndex": visit["visitIndex"],
                "symbolicNoteIds": dead_ids if rule["consumesDeadNotes"] else [],
                "sourcePitchedNoteIds": [identifier for identifier, note in pitched if not note["techniques"]["dead"]] if any(field in rule["match"] for field in ("pitchedNoteMarks", "simultaneousPitchedNoteMarks")) else [],
                "contextPitchedNoteIds": [identifier for identifier, note in context if not note["techniques"]["dead"]] if grouped else [],
                "interpretationRuleId": rule["id"], "evidenceBeatIds": rule["evidenceBeatIds"],
            }
            if "evidenceSource" in rule:
                gesture["evidenceSource"] = rule["evidenceSource"]
            gestures.append(gesture)
        uncovered_dead = dead_ids and not any(rule["consumesDeadNotes"] for rule in candidates)
        conventional = {}
        if uncovered_dead and not conflicting and performance_id not in suppressed_ids and not (set(normalized_text(beat["text"]).split()) - consumed):
            for identifier in dead_ids:
                event, _, note = segments[identifier]
                if event["isAttack"] is False:
                    continue
                ghost = note["techniques"].get("antiAccent") == "Normal"
                convention = "ghostXIsPercussiveHit" if ghost else "plainXIsThumbSlap"
                if conventions is not None and conventions.get(convention) is True:
                    conventional.setdefault("percussive_hit" if ghost else "thumb_slap", []).append(identifier)
            for technique, identifiers in conventional.items():
                known = beat["graceMode"] is None and all(segments[identifier][0]["isAttack"] is True for identifier in identifiers)
                rule_id = "owner-ghost-X-generic" if technique == "percussive_hit" else "owner-plain-X-thumb"
                gestures.append({
                    "id": f"{performance_id}:{rule_id}", "technique": technique, "attributes": {},
                    "voiceIndex": beat["voiceIndex"], "onsetQuarter": onset, "durationQuarter": None,
                    "writtenBeatId": beat["id"], "visitIndex": visit["visitIndex"], "graceMode": beat["graceMode"],
                    "scoreOnsetKnown": known, "symbolicNoteIds": identifiers,
                    "sourcePitchedNoteIds": [], "contextPitchedNoteIds": [],
                    "labelMask": {"gesture": True, "fingering": False, "pitch": False, "onset": known},
                    "interpretationRuleId": rule_id, "evidenceBeatIds": [], "evidenceSource": "owner-notation-conventions",
                })
            classified = {identifier for identifiers in conventional.values() for identifier in identifiers}
            dead_ids = [identifier for identifier in dead_ids if identifier not in classified]
            uncovered_dead = bool(dead_ids)
        if uncovered_dead and not conflicting and any(segments[identifier][0]["isAttack"] is not False for identifier in dead_ids):
            unresolved.append({"performanceBeatId": performance_id, "onsetQuarter": onset, "reason": "uninterpreted_dead_note_cluster", "symbolicNoteIds": dead_ids})
        if set(normalized_text(beat["text"]).split()) - consumed:
            unresolved.append({"performanceBeatId": performance_id, "onsetQuarter": onset, "reason": "uninterpreted_annotation", "writtenBeatId": beat["id"]})
    if conventions is not None and conventions.get("plainXIsThumbSlap") is True:
        by_position, merged = {}, []
        positions = {f"p{visit['visitIndex']}:{beat['id']}": slot for (visit, beat, _), slot in zip(performed, slots)}
        for gesture in gestures:
            if gesture["technique"] != "thumb_slap" or not gesture["symbolicNoteIds"]:
                merged.append(gesture)
                continue
            position = positions[f"p{gesture['visitIndex']}:{gesture['writtenBeatId']}"]
            if position not in by_position:
                by_position[position] = gesture
                merged.append(gesture)
            else:
                first = by_position[position]
                first.setdefault("coincidentSourceGestures", []).append(deepcopy(gesture))
                first["symbolicNoteIds"].extend(identifier for identifier in gesture["symbolicNoteIds"] if identifier not in first["symbolicNoteIds"])
                first["scoreOnsetKnown"] = first["scoreOnsetKnown"] and gesture["scoreOnsetKnown"]
                first.setdefault("labelMask", {"gesture": True, "fingering": False, "pitch": False})["onset"] = first["scoreOnsetKnown"]
        gestures = merged
    return gestures, unresolved, rests, suppressed


def canonicalize(score, annotations=None, conventions=None):
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
    notes, symbols, issues, segments, silenced = logical_notes(score, written_notes)
    gestures, unresolved, rests, suppressed = gesture_events(score, rules, segments, conventions)
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
        "targets": {"notes": notes, "rests": rests, "noteRests": silenced, "gestures": gestures},
        "review": {"notationSymbols": symbols, "unresolvedGestures": unresolved, "suppressedAnnotations": suppressed, "issues": issues, "sourceIssues": score["issues"]},
        "provenance": {"sourceGpSha256": score["sourceGpSha256"], "fretConvention": instrument["fretConvention"], "ruleIds": [rule["id"] for rule in rules], "referenceBeatIds": [beat["id"] for beat in score["scoreEvents"] if beat["referenceOnly"]], "percussiveHitTextDetectionMaxLen": percussive_hit_text_max_len()},
    }
    if conventions is not None:
        for name in ("plainXIsThumbSlap", "ghostXIsPercussiveHit"):
            if name in conventions:
                if type(conventions[name]) is not bool:
                    raise CanonicalLabelError(f"{name} must be an explicit boolean.")
                output["provenance"][name] = conventions[name]
    return deepcopy(output)


def canonical_counts(labels):
    notes = labels["targets"]["notes"]
    symbols = labels["review"]["notationSymbols"]
    unresolved = Counter(item["reason"] for item in labels["review"]["unresolvedGestures"])
    return {
        "logicalNoteCount": len(notes),
        "symbolicNoteCount": len(symbols),
        "sourceSegmentCount": sum(len(note["sourceSegments"]) for note in notes + symbols) + len(labels["targets"]["noteRests"]),
        "silencedRepeatTieCount": len(labels["targets"]["noteRests"]),
        "mergedContinuationCount": sum(len(note["sourceSegments"]) - 1 for note in notes + symbols),
        "unknownAttackCount": sum(note["isAttack"] is None for note in notes + symbols),
        "unknownDurationCount": sum(not note["labelMask"]["notatedDuration"] for note in notes + symbols),
        "gestureCount": len(labels["targets"]["gestures"]),
        "gestureTypes": dict(sorted(Counter(gesture["technique"] for gesture in labels["targets"]["gestures"]).items())),
        "unresolvedGestureCount": len(labels["review"]["unresolvedGestures"]),
        "uninterpretedClusterCount": unresolved["uninterpreted_dead_note_cluster"],
        "uninterpretedAnnotationCount": unresolved["uninterpreted_annotation"],
        "ambiguousGestureCount": unresolved["ambiguous_gesture_rules"],
        "suppressedAnnotationCount": len(labels["review"]["suppressedAnnotations"]),
        "scoreTimingResolved": labels["scoreTimingResolved"],
    }
