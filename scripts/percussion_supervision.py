"""Resolved percussion annotation coverage, never an authorization to use it."""

from bisect import bisect_right
import re

import numpy as np

from .canonical_events import fraction
from .score_alignment import AlignmentInputError, ScoreClock


ATTACK_GUARD_SECONDS = .1
PERCUSSION_TECHNIQUES = frozenset({"wrist_thump", "thumb_slap", "percussive_hit", "muted_strum"})


def _attack_known(value):
    mask = value.get("labelMask", {})
    if any(mask.get(key, True) is not True for key in ("attack", "onset")):
        return False
    if "scoreOnsetKnown" in value:
        return value["scoreOnsetKnown"] is True and value.get("graceMode") is None
    segments = value.get("sourceSegments", [])
    return (
        value.get("isAttack") is True and mask.get("attack") is True
        and bool(segments) and "graceMode" in segments[0] and segments[0]["graceMode"] is None
    )


def percussion_annotation_coverage(canonical, candidate, normalization):
    """Return absolute half-open clip intervals; callers must confirm completeness.

    Resolved unknown attacks censor a transient neighborhood, not notated sustain.
    Unknown/grace timing censors its source measures; unlocatable uncertainty
    censors the entire mapping. Positive events remain inside annotation coverage.
    """
    clock = ScoreClock(canonical, normalization)
    if canonical.get("scoreTimingResolved") is not True:
        raise AlignmentInputError("Percussion coverage requires resolved normalized score timing.")
    mapping = candidate["denseMapping"]
    reference = np.array([point["referenceSeconds"] for point in mapping], dtype=float)
    audio = np.array([point["clipSeconds"] for point in mapping], dtype=float)
    if (
        len(reference) < 2 or not np.isfinite(reference).all() or not np.isfinite(audio).all()
        or np.any(np.diff(reference) <= 0) or np.any(np.diff(audio) < 0)
        or reference[0] < 0 or reference[-1] > clock.duration_seconds or audio[0] < 0
    ):
        raise AlignmentInputError("Percussion coverage requires a finite bounded monotone mapping.")
    holes = []

    def censor_reference(left, right):
        if right < reference[0] or left > reference[-1]:
            return
        start, stop = np.interp([max(left, reference[0]), min(right, reference[-1])], reference, audio)
        holes.append((float(start) - ATTACK_GUARD_SECONDS, float(stop) + ATTACK_GUARD_SECONDS))

    targets = canonical["targets"]
    review = canonical["review"]
    symbols = {symbol["id"]: symbol for symbol in review["notationSymbols"]}
    beat_context = {}
    for value in [*targets["notes"], *symbols.values()]:
        for segment in value["sourceSegments"]:
            if "visitIndex" in segment and "writtenBeatId" in segment:
                key = f"p{segment['visitIndex']}:{segment['writtenBeatId']}"
                beat_context.setdefault(key, []).append(segment)
    for rest in targets.get("rests", []):
        key = f"p{rest['visitIndex']}:{rest['writtenBeatId']}"
        beat_context.setdefault(key, []).append(rest)
    for gesture in targets["gestures"]:
        if "visitIndex" in gesture and "writtenBeatId" in gesture:
            key = f"p{gesture['visitIndex']}:{gesture['writtenBeatId']}"
            beat_context.setdefault(key, []).append(gesture)

    def measure_indices(value):
        indices = set()
        for item in [value, *value.get("sourceSegments", [])]:
            index = item.get("visitIndex", item.get("measureIndex"))
            performance_id = item.get("performanceBeatId")
            if index is None and isinstance(performance_id, str):
                match = re.match(r"p(\d+):", performance_id)
                if match:
                    index = int(match[1])
            if index is not None:
                if type(index) is not int or not 0 <= index < len(clock.measure_starts):
                    raise AlignmentInputError("Uncertain percussion references an invalid normalized measure.")
                indices.add(index)
            onset = item.get("onsetQuarter")
            if onset is not None:
                quarter = fraction(onset, "uncertain percussion position")
                clock.seconds(quarter)
                index = min(bisect_right(clock.measure_starts, quarter) - 1, len(clock.measure_starts) - 1)
                indices.add(index)
                # An unplaced attack at a barline can precede its nominal anchor.
                if index > 0 and quarter == clock.measure_starts[index]:
                    indices.add(index - 1)
        return indices

    def censor(value, timing_known, contexts=()):
        onset = value.get("onsetQuarter")
        if timing_known and onset is not None:
            seconds = clock.seconds(fraction(onset, "uncertain percussion attack"))
            censor_reference(seconds, seconds)
            return True
        indices = measure_indices(value)
        for context in contexts:
            indices.update(measure_indices(context))
        if not indices:
            return False
        for index in indices:
            left = clock.measure_starts[index]
            right = clock.measure_starts[index + 1] if index + 1 < len(clock.measure_starts) else clock.total_quarter
            censor_reference(clock.seconds(left), clock.seconds(right))
        return True

    for symbol in symbols.values():
        timing_known = _attack_known(symbol)
        if (symbol.get("labelMask", {}).get("gesture") is not True or not timing_known) and not censor(symbol, timing_known):
            return []
    for gesture in targets["gestures"]:
        if gesture["technique"] not in PERCUSSION_TECHNIQUES:
            continue
        timing_known = _attack_known(gesture)
        if (not timing_known or gesture.get("labelMask", {}).get("gesture", True) is not True) and not censor(gesture, timing_known):
            return []
    for unresolved in review["unresolvedGestures"]:
        linked = unresolved.get("symbolicNoteIds", [])
        contexts = beat_context.get(unresolved.get("performanceBeatId"), [])
        if "scoreOnsetKnown" in unresolved or "isAttack" in unresolved:
            timing_known = _attack_known(unresolved)
        elif linked:
            timing_known = all(identifier in symbols and _attack_known(symbols[identifier]) for identifier in linked)
        else:
            timing_known = bool(contexts) and all(
                _attack_known(context) if "scoreOnsetKnown" in context else
                "graceMode" in context and context["graceMode"] is None and context.get("notatedDurationQuarter") is not None
                for context in contexts
            )
        mask = unresolved.get("labelMask", {})
        timing_known = timing_known and unresolved.get("graceMode") is None and all(mask.get(key, True) is True for key in ("onset", "attack"))
        if not censor(unresolved, timing_known, [*contexts, *(symbols[identifier] for identifier in linked if identifier in symbols)]):
            return []

    # A score-time plateau has no unique inverse attack position in the clip.
    for left, right in zip(mapping, mapping[1:]):
        if left["clipSeconds"] == right["clipSeconds"]:
            seconds = left["clipSeconds"]
            holes.append((seconds - ATTACK_GUARD_SECONDS, seconds + ATTACK_GUARD_SECONDS))
    coverage, cursor = [], float(audio[0])
    for start, stop in sorted(holes):
        start, stop = max(start, float(audio[0])), min(stop, float(audio[-1]))
        if stop <= cursor or start >= audio[-1]:
            continue
        if cursor < start:
            coverage.append([cursor, start])
        cursor = max(cursor, stop)
    if cursor < audio[-1]:
        coverage.append([cursor, float(audio[-1])])
    return coverage
