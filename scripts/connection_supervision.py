"""Derive note-connection and note-technique targets from canonical provenance."""

from collections import defaultdict
from copy import deepcopy
from fractions import Fraction

from .canonical_events import fraction
from .score_alignment import AlignmentInputError


SLIDE_FLAGS = (1, 2, 4, 8, 16, 20, 32)
CONNECTION_TYPES = ("none", "hammer_on", "pull_off", *(f"slide_{value}" for value in SLIDE_FLAGS))
NOTE_TECHNIQUE_TYPES = ("bend", "tap", "left_hand_tap", "vibrato")
BEND_FIELDS = ("OriginOffset", "OriginValue", "MiddleOffset1", "MiddleOffset2", "MiddleValue", "DestinationOffset", "DestinationValue")


def canonical_connections(labels):
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


def projected_connections(labels, candidate, clock):
    mapping = candidate["denseMapping"]
    reference = [point["referenceSeconds"] for point in mapping]
    audio = [point["clipSeconds"] for point in mapping]
    if len(reference) < 2 or any(left >= right for left, right in zip(reference, reference[1:])) or any(left > right for left, right in zip(audio, audio[1:])):
        raise AlignmentInputError("Connection projection requires a monotone mapping.")
    result = []
    for event in canonical_connections(labels):
        seconds = clock.seconds(fraction(event["onsetQuarter"], event["id"]))
        if not reference[0] <= seconds <= reference[-1]:
            clip = None
        else:
            import numpy as np
            clip = float(np.interp(seconds, reference, audio))
        result.append({**deepcopy(event), "proposedOnsetClipSeconds": clip})
    return result


def connections_in_window(events, start, stop, rate):
    left, right = start / rate, stop / rate
    result = []
    for event in events:
        onset = event["proposedOnsetClipSeconds"]
        if onset is not None and left <= onset < right:
            result.append({
                **deepcopy(event),
                "onsetWindowSeconds": onset - left,
                "supervisionMask": {
                    "connection": bool(event["scoreOnsetKnown"] and event["connectionMask"]),
                    "techniques": bool(event["scoreOnsetKnown"]),
                    "bendCurve": bool(event["scoreOnsetKnown"] and event["bendCurveMask"]),
                },
            })
    return result
