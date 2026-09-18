"""Derive note-connection and note-technique targets from canonical provenance."""

from collections import defaultdict
from copy import deepcopy
from fractions import Fraction

from .canonical_events import fraction
from .score_alignment import AlignmentInputError


# Freeze v3's vocabulary: its historical logits must never acquire v4 meanings.
SLIDE_FLAGS = (1, 2, 4, 8, 16, 20, 32)
CONNECTION_TYPES = ("none", "hammer_on", "pull_off", *(f"slide_{value}" for value in SLIDE_FLAGS))
NOTE_TECHNIQUE_TYPES = ("bend", "tap", "left_hand_tap", "vibrato")
BEND_FIELDS = ("OriginOffset", "OriginValue", "MiddleOffset1", "MiddleOffset2", "MiddleValue", "DestinationOffset", "DestinationValue")
SUPERVISION_VERSION = "note-relations-local-slides-anchored-grace-v4"
RELATION_TYPES = ("none", "hammer_on", "pull_off", "slide_1", "slide_2")
SLIDE_NOTE_FLAGS = {
    "slide_out_down": 4, "slide_out_up": 8,
    "slide_in_below": 16, "slide_in_above": 32,
}
V4_NOTE_TECHNIQUE_TYPES = (*NOTE_TECHNIQUE_TYPES, *SLIDE_NOTE_FLAGS)
GRACE_MODES = ("BeforeBeat", "OnBeat")


def vocabularies(architecture_version):
    if architecture_version not in (1, 2, 3, 4):
        raise AlignmentInputError("Unsupported connection architecture version.")
    return (RELATION_TYPES, V4_NOTE_TECHNIQUE_TYPES) if architecture_version == 4 else (CONNECTION_TYPES, NOTE_TECHNIQUE_TYPES)


def _flags(segment):
    techniques = segment.get("techniques", {})
    if not isinstance(techniques, dict):
        raise AlignmentInputError("Canonical note techniques must be a mapping.")
    flags = techniques.get("slideFlags", 0)
    if type(flags) is not int or flags < 0:
        raise AlignmentInputError("Slide flags must be a nonnegative integer.")
    return flags


def _known_note(note):
    return (
        note["isAttack"] is True and note["labelMask"].get("attack") is True
        and note["labelMask"].get("pitch") is True
        and type(note.get("soundingPitchMidi")) is int
        and not any(part.get("techniques", {}).get("dead") for part in note["sourceSegments"])
    )


def _relation(origin, destination, *, adjacent):
    hopo = bool(destination["sourceSegments"][0].get("techniques", {}).get("hopoDestination"))
    flags = _flags(origin["sourceSegments"][-1]) if origin is not None else 0
    slide = flags & 3
    if flags & ~63:
        return "none", False
    if not hopo and not slide:
        return "none", True
    if not adjacent or origin is None or not _known_note(origin) or not _known_note(destination):
        return "none", False
    interval = destination["soundingPitchMidi"] - origin["soundingPitchMidi"]
    if flags & ~63 or slide == 3 or (hopo and slide) or interval == 0:
        return "none", False
    return (("hammer_on" if interval > 0 else "pull_off") if hopo else f"slide_{slide}"), True


def _canonical_connections_v4(labels):
    groups = defaultdict(list)
    # Unknown starts and unpitched carriers remain adjacency barriers, never
    # silently bridge them to the preceding known pitched attack.
    all_notes = labels["targets"]["notes"] + labels.get("review", {}).get("notationSymbols", [])
    occupied = defaultdict(int)
    for order, note in enumerate(all_notes):
        if note.get("sourceSegments"):
            groups[(note["voiceIndex"], note["string"])].append((order, note))
            if note["sourceSegments"][0].get("graceMode") is None:
                occupied[(note["string"], fraction(note["onsetQuarter"], note["id"]))] += 1
    result = []
    for (voice, string), items in groups.items():
        items.sort(key=lambda item: (
            fraction(item[1]["onsetQuarter"], item[1]["id"]),
            item[1]["sourceSegments"][0].get("graceMode") is None, item[0],
        ))
        regular = [note for _, note in items if note["sourceSegments"][0].get("graceMode") is None]
        grace_by_onset = defaultdict(list)
        for _, note in items:
            if note["sourceSegments"][0].get("graceMode") is not None:
                grace_by_onset[fraction(note["onsetQuarter"], note["id"])].append(note)
        prior = None
        for note in regular:
            onset = fraction(note["onsetQuarter"], note["id"])
            parts = note["sourceSegments"]
            segment = parts[0]
            if not _known_note(note):
                prior = note
                continue
            flags = [_flags(part) for part in parts]
            techniques = segment.get("techniques", {})
            bend = segment.get("bend")
            local = {
                "bend": bend is not None, "tap": bool(techniques.get("tapped")),
                "left_hand_tap": bool(techniques.get("leftHandTapped")),
                "vibrato": bool(techniques.get("vibrato")),
                **{name: any(value & bit for value in flags) for name, bit in SLIDE_NOTE_FLAGS.items()},
            }
            masks = {name: True for name in local}
            if any(value & ~63 for value in flags):
                masks.update({name: False for name in SLIDE_NOTE_FLAGS})
            for names in (("slide_out_down", "slide_out_up"), ("slide_in_below", "slide_in_above")):
                if all(local[name] for name in names):
                    masks.update({name: False for name in names})
            # A changed effect on a tied continuation is not an attack-local label.
            for name, key in (("tap", "tapped"), ("left_hand_tap", "leftHandTapped"), ("vibrato", "vibrato")):
                masks[name] = all(bool(part.get("techniques", {}).get(key)) == local[name] for part in parts)
            masks["bend"] = all(part.get("bend") == bend for part in parts)
            masks["slide_in_below"] &= not any(value & 16 for value in flags[1:])
            masks["slide_in_above"] &= not any(value & 32 for value in flags[1:])
            prior_end = None
            if prior is not None and prior.get("notatedDurationQuarter") is not None:
                prior_end = fraction(prior["onsetQuarter"], prior["id"]) + fraction(prior["notatedDurationQuarter"], prior["id"])
            orphan_grace = any(
                position < onset and (prior is None or position > fraction(prior["onsetQuarter"], prior["id"]))
                for position in grace_by_onset
            )
            adjacent = not orphan_grace and prior_end == onset and (prior is None or occupied[(string, fraction(prior["onsetQuarter"], prior["id"]))] == 1)
            graces = grace_by_onset.get(onset, [])
            connection, connection_mask = _relation(prior, note, adjacent=adjacent)
            grace = None
            grace_mask = len(graces) <= 1 and not orphan_grace
            grace_masks = {"fret": False, "mode": False, "transition": False}
            if graces:
                # The score fixes the main-note anchor, not the grace audio time.
                # Multiple grace notes need a sequence head; censor, do not pick one.
                connection, connection_mask = "none", len(graces) == 1 and not orphan_grace
                if len(graces) == 1:
                    source = graces[0]
                    source_segment = source["sourceSegments"][0]
                    transition, transition_mask = _relation(source, note, adjacent=True)
                    known = _known_note(source)
                    simple_pitch = known and source.get("basePitchMidi") == source["soundingPitchMidi"]
                    grace = {
                        "sourceNoteId": source["id"], "sourceFret": source.get("fret"),
                        "sourcePitchMidi": source.get("soundingPitchMidi"),
                        "intervalSemitones": note["soundingPitchMidi"] - source["soundingPitchMidi"] if known else None,
                        "mode": source_segment.get("graceMode"), "transition": transition,
                        "onsetSeconds": None, "timingKnown": False,
                    }
                    grace_masks = {
                        "fret": simple_pitch and source["labelMask"].get("fingering") is True,
                        "mode": source_segment.get("graceMode") in GRACE_MODES,
                        "transition": simple_pitch and transition_mask,
                    }
            if not grace_mask:
                grace_masks = {name: False for name in grace_masks}
            if occupied[(string, onset)] != 1:
                connection_mask = grace_mask = False
                masks = {name: False for name in masks}
                grace_masks = {name: False for name in grace_masks}
            result.append({
                "id": f"connection:{note['id']}", "sourceNoteId": note["id"],
                "supervisionVersion": SUPERVISION_VERSION, "techniqueSchemaVersion": 4,
                "string": string, "voiceIndex": voice, "fret": note["fret"],
                "soundingPitchMidi": note["soundingPitchMidi"],
                "onsetQuarter": deepcopy(note["onsetQuarter"]),
                "connection": connection, "connectionMask": connection_mask,
                "priorSourceNoteId": prior["id"] if prior is not None and adjacent and not graces else None,
                "techniques": local, "techniqueMasks": masks,
                "bendCurve": [bend[field] for field in BEND_FIELDS] if isinstance(bend, dict) and set(bend) == set(BEND_FIELDS) else None,
                "bendCurveMask": masks["bend"] and isinstance(bend, dict) and set(bend) == set(BEND_FIELDS),
                "grace": grace, "graceMask": grace_mask, "graceAttributeMasks": grace_masks,
                "unresolvedGraceCount": len(graces) if len(graces) > 1 else 0,
                "scoreOnsetKnown": True,
            })
            prior = note
    return sorted(result, key=lambda event: (fraction(event["onsetQuarter"], event["id"]), -event["string"], event["voiceIndex"]))


def canonical_connections(labels, *, architecture_version=3):
    vocabularies(architecture_version)
    if architecture_version == 4:
        return _canonical_connections_v4(labels)
    by_string = defaultdict(list)
    for note in labels["targets"]["notes"]:
        if note["isAttack"] is not True or not note["sourceSegments"] or note["sourceSegments"][0].get("graceMode") is not None:
            continue
        if not note["labelMask"].get("attack") or note["soundingPitchMidi"] is None:
            continue
        by_string[note["string"]].append(note)
    result = []
    for string, notes in by_string.items():
        notes.sort(key=lambda note: fraction(note["onsetQuarter"], note["id"]))
        for index, note in enumerate(notes):
            segment = note["sourceSegments"][0]
            segment_techniques = segment.get("techniques", {})
            if not isinstance(segment_techniques, dict):
                raise AlignmentInputError("Canonical note techniques must be a mapping when present.")
            prior = notes[index - 1] if index else None
            connection = "none"
            connection_mask = True
            if segment_techniques.get("hopoDestination"):
                if prior is None:
                    connection_mask = False
                elif note["soundingPitchMidi"] > prior["soundingPitchMidi"]:
                    connection = "hammer_on"
                elif note["soundingPitchMidi"] < prior["soundingPitchMidi"]:
                    connection = "pull_off"
                else:
                    connection_mask = False
            elif prior is not None and prior["sourceSegments"][0].get("techniques", {}).get("slideFlags"):
                flags = prior["sourceSegments"][0]["techniques"]["slideFlags"]
                connection = f"slide_{flags}" if flags in SLIDE_FLAGS else "none"
                connection_mask = flags in SLIDE_FLAGS
            bend = segment.get("bend")
            techniques = {
                "bend": segment.get("bend") is not None,
                "tap": bool(segment_techniques.get("tapped")),
                "left_hand_tap": bool(segment_techniques.get("leftHandTapped")),
                "vibrato": bool(segment_techniques.get("vibrato")),
            }
            result.append({
                "id": f"connection:{note['id']}",
                "sourceNoteId": note["id"],
                "string": string,
                "onsetQuarter": deepcopy(note["onsetQuarter"]),
                "connection": connection,
                "connectionMask": connection_mask,
                "priorSourceNoteId": prior["id"] if prior is not None else None,
                "techniques": techniques,
                "bendCurve": [bend[field] for field in BEND_FIELDS] if bend is not None and set(bend) == set(BEND_FIELDS) else None,
                "bendCurveMask": bend is not None and set(bend) == set(BEND_FIELDS),
                "scoreOnsetKnown": True,
            })
    return sorted(result, key=lambda event: (fraction(event["onsetQuarter"], event["id"]), -event["string"]))


def projected_connections(labels, candidate, clock, *, architecture_version=3):
    mapping = candidate["denseMapping"]
    reference = [point["referenceSeconds"] for point in mapping]
    audio = [point["clipSeconds"] for point in mapping]
    if len(reference) < 2 or any(left >= right for left, right in zip(reference, reference[1:])) or any(left > right for left, right in zip(audio, audio[1:])):
        raise AlignmentInputError("Connection projection requires a monotone mapping.")
    result = []
    for event in canonical_connections(labels, architecture_version=architecture_version):
        seconds = clock.seconds(fraction(event["onsetQuarter"], event["id"]))
        if not reference[0] <= seconds <= reference[-1]:
            clip = None
        else:
            import numpy as np
            clip = float(np.interp(seconds, reference, audio))
        result.append({**deepcopy(event), "proposedOnsetClipSeconds": clip})
    by_id = {event["sourceNoteId"]: event for event in result}
    if architecture_version == 4:
        for event in result:
            origin = by_id.get(event["priorSourceNoteId"])
            event["origin"] = {
                key: deepcopy(origin[key]) for key in ("sourceNoteId", "string", "voiceIndex", "soundingPitchMidi", "fret", "proposedOnsetClipSeconds")
            } if origin is not None else None
            if event["connection"] != "none" and (origin is None or origin["proposedOnsetClipSeconds"] is None or event["proposedOnsetClipSeconds"] is None
                                                  or origin["proposedOnsetClipSeconds"] >= event["proposedOnsetClipSeconds"]):
                event["connectionMask"] = False
    return result


def connections_in_window(events, start, stop, rate):
    left, right = start / rate, stop / rate
    result = []
    for event in events:
        onset = event["proposedOnsetClipSeconds"]
        if onset is not None and left <= onset < right:
            connection_known = bool(event["scoreOnsetKnown"] and event["connectionMask"])
            if event.get("techniqueSchemaVersion") == 4 and event["connection"] != "none":
                origin = event.get("origin")
                connection_known &= origin is not None and origin["proposedOnsetClipSeconds"] is not None and left <= origin["proposedOnsetClipSeconds"] < onset
            result.append({
                **deepcopy(event),
                "onsetWindowSeconds": onset - left,
                "supervisionMask": {
                    "connection": connection_known,
                    "techniques": bool(event["scoreOnsetKnown"]),
                    "bendCurve": bool(event["scoreOnsetKnown"] and event["bendCurveMask"]),
                    **({
                        "grace": bool(event["scoreOnsetKnown"] and event["graceMask"]),
                        "graceAttributes": deepcopy(event["graceAttributeMasks"]),
                    } if event.get("techniqueSchemaVersion") == 4 else {}),
                },
            })
    return result
