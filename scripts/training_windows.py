"""Project canonical supervision into bounded, overlapping audio windows."""

from copy import deepcopy

import numpy as np

from .canonical_events import fraction
from .score_alignment import AlignmentInputError


SETTINGS = {"windowSeconds": 8, "strideSeconds": 6, "minimumWindowSeconds": 2}


def sample_windows(spans, rate, settings=SETTINGS):
    length = round(settings["windowSeconds"] * rate)
    stride = round(settings["strideSeconds"] * rate)
    minimum = round(settings["minimumWindowSeconds"] * rate)
    if not 0 < stride <= length or not 0 < minimum <= length:
        raise AlignmentInputError("Invalid training window geometry.")
    windows = []
    for left, right in spans:
        start = left
        while right - start >= minimum:
            stop = min(start + length, right)
            windows.append((start, stop))
            if stop == right:
                break
            start += stride
    return windows


def projected_targets(labels, candidate, clock):
    mapping = candidate["denseMapping"]
    reference = np.array([point["referenceSeconds"] for point in mapping])
    audio = np.array([point["clipSeconds"] for point in mapping])
    if len(reference) < 2 or not np.isfinite(reference).all() or not np.isfinite(audio).all() or np.any(np.diff(reference) <= 0) or np.any(np.diff(audio) < 0):
        raise AlignmentInputError("Window projection requires a finite monotone mapping.")

    def mapped(quarter):
        nominal = clock.seconds(fraction(quarter, "target position"))
        return float(np.interp(nominal, reference, audio)) if reference[0] <= nominal <= reference[-1] else None

    notes = []
    for note in labels["targets"]["notes"]:
        onset = mapped(note["onsetQuarter"])
        duration = note["notatedDurationQuarter"]
        end = None
        if duration is not None and note["labelMask"]["notatedDuration"]:
            total = fraction(note["onsetQuarter"], note["id"]) + fraction(duration, note["id"])
            end = mapped([total.numerator, total.denominator])
        known_onset = note["isAttack"] is True and note["sourceSegments"][0]["graceMode"] is None and note["labelMask"]["attack"]
        notes.append({
            "sourceNoteId": note["id"], "voiceIndex": note["voiceIndex"], "string": note["string"], "fret": note["fret"],
            "soundingPitchMidi": note["soundingPitchMidi"], "onsetQuarter": note["onsetQuarter"], "notatedDurationQuarter": duration,
            "proposedOnsetClipSeconds": onset, "proposedNotatedEndClipSeconds": end,
            "onsetTimingKnownInScore": bool(known_onset), "sourceLabelMask": deepcopy(note["labelMask"]),
        })
    gestures = []
    for gesture in labels["targets"]["gestures"]:
        gestures.append({
            "sourceGestureId": gesture["id"], "technique": gesture["technique"], "voiceIndex": gesture["voiceIndex"],
            "onsetQuarter": gesture["onsetQuarter"], "proposedOnsetClipSeconds": mapped(gesture["onsetQuarter"]),
            "onsetTimingKnownInScore": gesture.get("scoreOnsetKnown", False) and gesture.get("graceMode") is None,
            "sourceLabelMask": deepcopy(gesture.get("labelMask", {})),
        })
    return notes, gestures


def targets_in_window(notes, gestures, start, stop, rate):
    left, right = start / rate, stop / rate
    selected_notes, selected_gestures = [], []
    for note in notes:
        onset, end = note["proposedOnsetClipSeconds"], note["proposedNotatedEndClipSeconds"]
        if onset is None or onset >= right or (onset < left and (end is None or end <= left)):
            continue
        attack = left <= onset < right and note["onsetTimingKnownInScore"]
        complete_duration = attack and end is not None and onset < end <= right
        mask = note["sourceLabelMask"]
        selected_notes.append({
            **deepcopy(note), "onsetWindowSeconds": onset - left, "carryIn": onset < left,
            "supervisionMask": {
                "onset": bool(attack), "pitch": bool(attack and mask["pitch"]),
                "fingering": bool(attack and mask["fingering"]),
                "notatedDuration": bool(complete_duration and mask["notatedDuration"]), "acousticRelease": False,
            },
        })
    for gesture in gestures:
        onset = gesture["proposedOnsetClipSeconds"]
        if onset is not None and left <= onset < right:
            selected_gestures.append({
                **deepcopy(gesture), "onsetWindowSeconds": onset - left,
                "supervisionMask": {
                    "onset": bool(gesture["onsetTimingKnownInScore"]),
                    "gesture": bool(gesture["onsetTimingKnownInScore"] and gesture["sourceLabelMask"].get("gesture", True)),
                    "physicalFingering": False,
                },
            })
    return {"notes": selected_notes, "gestures": selected_gestures, "negativePercussionSupervision": False, "restsAreAcousticSilence": False}
