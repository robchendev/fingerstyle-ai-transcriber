"""Derive acoustic strum/roll targets from canonical GP beat evidence."""

from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
import math

import numpy as np

from .canonical_events import fraction
from .score_alignment import AlignmentInputError


TECHNIQUE_TYPES = ("brush", "arpeggio", "pick_stroke", "rasgueado")
TECHNIQUE_DIRECTIONS = ("Down", "Up")
SOURCE_KEYS = {
    "brush": "brush",
    "arpeggio": "arpeggio",
    "pickStroke": "pick_stroke",
    "rasgueado": "rasgueado",
}


def canonical_techniques(labels):
    notes = labels["targets"]["notes"]
    groups = defaultdict(list)
    for note in notes:
        if note.get("isAttack") is not True or not note.get("sourceSegments"):
            continue
        segment = note["sourceSegments"][0]
        if segment.get("graceMode") is not None or not note["labelMask"].get("attack"):
            continue
        groups[fraction(note["onsetQuarter"], note["id"])].append(note)
    result = []
    for onset, group in sorted(groups.items()):
        marks = defaultdict(list)
        beat_ids = set()
        for note in group:
            segment = note["sourceSegments"][0]
            written = segment.get("writtenBeatId")
            visit = segment.get("visitIndex")
            if isinstance(written, str):
                beat_ids.add(f"p{visit}:{written}" if type(visit) is int else written)
            for source, target in SOURCE_KEYS.items():
                value = segment.get("beatTechniques", {}).get(source)
                if value not in (None, False, ""):
                    marks[target].append(value)
        if not marks:
            continue
        techniques = sorted(marks, key=TECHNIQUE_TYPES.index)
        directions = {}
        direction_masks = {}
        for technique in techniques:
            values = {value for value in marks[technique] if value in TECHNIQUE_DIRECTIONS}
            direction_masks[technique] = len(values) == 1
            if len(values) == 1:
                directions[technique] = next(iter(values))
        all_strings = sorted({note["string"] for note in group})
        all_pitches = sorted({note["soundingPitchMidi"] for note in group if note["soundingPitchMidi"] is not None})
        strings_by_technique = {technique: all_strings for technique in techniques}
        pitches_by_technique = {technique: all_pitches for technique in techniques}
        result.append({
            "id": f"technique:{onset.numerator}/{onset.denominator}",
            "onsetQuarter": [onset.numerator, onset.denominator],
            "techniques": techniques,
            "directions": directions,
            "directionMasks": direction_masks,
            "stringsByTechnique": strings_by_technique,
            "soundingPitchesMidiByTechnique": pitches_by_technique,
            "sourceBeatIds": sorted(beat_ids),
            "scoreOnsetKnown": True,
            "membershipComplete": True,
            "annotationCompleteAtResolvedNoteAttacks": True,
        })
    return result
def projected_techniques(labels, candidate, clock):
    mapping = candidate["denseMapping"]
    reference = np.asarray([point["referenceSeconds"] for point in mapping], dtype=np.float64)
    audio = np.asarray([point["clipSeconds"] for point in mapping], dtype=np.float64)
    if len(reference) < 2 or not np.isfinite(reference).all() or not np.isfinite(audio).all() or np.any(np.diff(reference) <= 0) or np.any(np.diff(audio) < 0):
        raise AlignmentInputError("Technique projection requires a finite monotone mapping.")
    result = []
    for event in canonical_techniques(labels):
        nominal = clock.seconds(fraction(event["onsetQuarter"], event["id"]))
        seconds = float(np.interp(nominal, reference, audio)) if reference[0] <= nominal <= reference[-1] else None
        result.append({**deepcopy(event), "proposedOnsetClipSeconds": seconds})
    return result


def techniques_in_window(events, start, stop, rate):
    if any(type(value) is not int or value < 0 for value in (start, stop, rate)) or start >= stop or rate <= 0:
        raise AlignmentInputError("Technique windows require positive integer sample geometry.")
    left, right = start / rate, stop / rate
    result = []
    for event in events:
        onset = event["proposedOnsetClipSeconds"]
        if onset is not None and left <= onset < right:
            result.append({
                **deepcopy(event),
                "onsetWindowSeconds": onset - left,
                "supervisionMask": {
                    "onset": bool(event["scoreOnsetKnown"]),
                    "technique": bool(event["scoreOnsetKnown"]),
                    "direction": deepcopy(event["directionMasks"]),
                    "strings": bool(event["scoreOnsetKnown"] and event["membershipComplete"]),
                },
            })
    return result
