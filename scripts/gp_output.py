"""Create deterministic full-voice and single-voice GP files from hypotheses."""

from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from .gp_events import HARMONIC_OFFSETS, NOTE_VALUES, TEMPO_BEAT_UNITS, decode_score
from .draft_cleanup import DraftProfile, clean_hypotheses
from .rhythm_inference import TICKS_PER_QUARTER, _fine_stroke_position
from .gp_normalization import (
    GPIF_ENTRY,
    NormalizationError,
    _archive_bytes,
    _parse_archive,
    _remove,
    _set_text,
    ghost_dead_note,
    semantic_node,
)
from .transcriber_audio import HarnessError
from .transcriber_model import HARMONIC_FRETS, HARMONIC_TYPES, PERCUSSION_TYPES
from .technique_supervision import TECHNIQUE_DIRECTIONS, TECHNIQUE_TYPES


GRID = Fraction(1, 8)
RHYTHM_GRIDS = ((Fraction(1, 4), 0.0, "sixteenth"), (Fraction(1, 8), 0.02, "thirty-second"))
PERCUSSION_DURATION = Fraction(1, 4)
SINGLE_VOICE_BRIDGE_QUARTER = Fraction(2)
INSTANT_BRUSH_DURATION_TICKS = 0
BRUSH_DURATION_XPROPERTY = "687935489"
BRUSH_START_XPROPERTY = "687935490"
MAX_FRET = 36
MAX_MEASURES = 4096
KEY_ORDER = (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7)
SHARP_ORDER = "FCGDAEB"
FLAT_ORDER = "BEADGCF"
NATURAL_PITCH = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
LETTERS = tuple(NATURAL_PITCH)


def _finite(value, name, *, minimum=None):
    if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value):
        raise HarnessError(f"{name} must be a finite number.")
    if minimum is not None and value < minimum:
        raise HarnessError(f"{name} must be at least {minimum}.")
    return value


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise HarnessError(f"{name} must be an integer from {minimum} through {maximum}.")
    return value


def _fraction(value):
    return Fraction(str(value)).limit_denominator(1_000_000)


def _quantize(value, grid=GRID, origin=Fraction(0)):
    units = (value - origin) / grid
    return max(Fraction(0), origin + (units.numerator * 2 + units.denominator) // (2 * units.denominator) * grid)


def _rhythmic_positions(values):
    grouped = defaultdict(set)
    for value in values:
        grouped[value.numerator // value.denominator].add(value)
    result = {}
    counts = defaultdict(int)
    for beat, positions in grouped.items():
        origin = Fraction(beat)
        candidates = []
        for grid, penalty, name in RHYTHM_GRIDS:
            snapped = {value: _quantize(value, grid, origin) for value in positions}
            collisions = len(snapped) - len(set(snapped.values()))
            error = sum(abs(snapped[value] - value) for value in positions)
            score = float(error) + penalty * len(positions) + collisions * 0.25
            candidates.append((score, penalty, -grid, name, grid, snapped))
        _, _, _, name, grid, snapped = min(candidates)
        result.update({value: (position, grid) for value, position in snapped.items()})
        counts[name] += len(positions)
    return result, dict(sorted(counts.items()))


def _rational(value):
    return [value.numerator, value.denominator]


class TempoMap:
    def __init__(self, metadata):
        initial = metadata.get("tempo")
        changes = metadata.get("tempoChanges", [])
        if not isinstance(initial, dict) or not isinstance(changes, list):
            raise HarnessError("Tempo metadata requires an initial object and a list of changes.")
        events = [{"timeSeconds": 0, **initial}]
        events.extend(changes)
        self.events = []
        previous = -1.0
        for index, event in enumerate(events):
            required = {"timeSeconds", "bpm", "beatUnit"}
            if not isinstance(event, dict) or not required <= set(event) or set(event) - required - {"linear"}:
                raise HarnessError("Each tempo event requires only timeSeconds, BPM, beat unit, and optional linear interpolation.")
            time = _finite(event["timeSeconds"], "Tempo time", minimum=0)
            if time <= previous:
                raise HarnessError("Tempo event times must be strictly increasing.")
            unit = event["beatUnit"]
            if not isinstance(unit, list) or len(unit) != 2 or any(type(term) is not int or term <= 0 for term in unit):
                raise HarnessError("Tempo beat units must be positive [numerator, denominator] pairs.")
            bpm = _finite(event["bpm"], "Tempo BPM", minimum=0)
            if bpm == 0:
                raise HarnessError("Tempo BPM must be positive.")
            linear = event.get("linear", False)
            if type(linear) is not bool or linear and index + 1 == len(events):
                raise HarnessError("A linear tempo event requires a later endpoint.")
            self.events.append({
                "time": _fraction(time),
                "bpm": _fraction(bpm),
                "beatUnit": Fraction(*unit),
                "quarterBpm": _fraction(bpm) * Fraction(*unit) * 4,
                "linear": linear,
            })
            previous = time
        self.quarters = [Fraction(0)]
        for index in range(1, len(self.events)):
            self.quarters.append(self.quarters[-1] + self._segment(index - 1, self.events[index]["time"]))

    def _segment(self, index, stop):
        event = self.events[index]
        elapsed = stop - event["time"]
        if elapsed < 0:
            raise HarnessError("Tempo integration cannot move backward.")
        if event["linear"]:
            following = self.events[index + 1]
            width = float(following["time"] - event["time"])
            start_bpm = float(event["quarterBpm"])
            end_bpm = float(following["quarterBpm"])
            if abs(end_bpm - start_bpm) < 1e-12:
                return _fraction(float(elapsed) * start_bpm / 60)
            quarter_width = width * (end_bpm - start_bpm) / (60 * math.log(end_bpm / start_bpm))
            slope = (end_bpm - start_bpm) / quarter_width
            quarter = start_bpm * math.expm1(float(elapsed) * slope / 60) / slope
            return _fraction(quarter)
        return elapsed * event["quarterBpm"] / 60

    def quarter_at(self, seconds):
        seconds = _fraction(_finite(seconds, "Event time", minimum=0))
        index = bisect_right([event["time"] for event in self.events], seconds) - 1
        return self.quarters[index] + self._segment(index, seconds)


def _validate_metadata(metadata):
    if not isinstance(metadata, dict):
        raise HarnessError("Prediction metadata must be an object.")
    required = {"openStringMidi", "capoFret", "tempo", "timeSignature"}
    if not required <= set(metadata):
        raise HarnessError("Prediction metadata requires tuning, full capo, tempo, and time signature.")
    tuning = metadata["openStringMidi"]
    if not isinstance(tuning, list) or len(tuning) != 6:
        raise HarnessError("Prediction metadata requires six open-string MIDI pitches.")
    for pitch in tuning:
        _integer(pitch, "Open-string MIDI pitch", 0, 127)
    capo = _integer(metadata["capoFret"], "Capo fret", 0, 24)
    if any(pitch + capo > 127 for pitch in tuning):
        raise HarnessError("Tuning plus capo exceeds the MIDI range.")
    meter = metadata["timeSignature"]
    _meter_duration(meter)
    changes = metadata.get("timeSignatureChanges", [])
    if not isinstance(changes, list):
        raise HarnessError("Time-signature changes must be a list.")
    previous = 0
    for event in changes:
        if not isinstance(event, dict) or set(event) != {"timeSeconds", "timeSignature"}:
            raise HarnessError("Each time-signature change requires only timeSeconds and timeSignature.")
        time = _finite(event["timeSeconds"], "Time-signature change time", minimum=0)
        if time <= previous:
            raise HarnessError("Time-signature change times must be strictly increasing and after time zero.")
        _meter_duration(event["timeSignature"])
        previous = time
    return tuning, capo


def _meter_duration(meter):
    if not isinstance(meter, list) or len(meter) != 2 or any(type(term) is not int or term <= 0 for term in meter):
        raise HarnessError("Time signatures must be positive [numerator, denominator] pairs.")
    if meter[1] & (meter[1] - 1):
        raise HarnessError("Time-signature denominators must be powers of two.")
    return Fraction(meter[0] * 4, meter[1])


def _validate_predictions(document):
    if not isinstance(document, dict) or document.get("kind") != "fingerstyle-transcription-hypotheses":
        raise HarnessError("Expected a fingerstyle-transcription-hypotheses document.")
    tuning, capo = _validate_metadata(document.get("metadata"))
    tempo = TempoMap(document["metadata"])
    duration_seconds = _finite(document.get("audioDurationSeconds"), "Audio duration", minimum=0)
    if duration_seconds == 0:
        raise HarnessError("Audio duration must be positive.")
    if not isinstance(document.get("notes"), list) or not isinstance(document.get("percussion"), list):
        raise HarnessError("Prediction notes and percussion must be lists.")
    notes = []
    structured_flags = []
    for index, value in enumerate(document.get("notes", [])):
        required = {"onsetSeconds", "string", "fret", "soundingPitchMidi", "voiceIndex", "notatedDurationQuarter", "harmonic", "confidence"}
        if not isinstance(value, dict) or not required <= set(value):
            raise HarnessError("Every predicted note requires time, string, fret, pitch, voice, duration, harmonic, and confidence fields.")
        if _finite(value["onsetSeconds"], "Note onset", minimum=0) > duration_seconds:
            raise HarnessError("A note prediction falls after the declared audio duration.")
        harmonic = value.get("harmonic")
        if harmonic is not None:
            if not isinstance(harmonic, dict) or harmonic.get("type") not in HARMONIC_TYPES or harmonic.get("fret") not in HARMONIC_FRETS:
                raise HarnessError("Predicted harmonics require a supported type and node.")
        score_onset = value.get("scoreOnsetQuarter")
        score_duration = value.get("scoreDurationQuarter")
        structured_flags.append(score_onset is not None)
        if (score_onset is None) != (score_duration is None):
            raise HarnessError("Structured note timing requires both score onset and score duration.")
        if score_onset is not None:
            try:
                onset_raw = Fraction(*score_onset)
                duration_raw = Fraction(*score_duration)
            except (TypeError, ValueError, ZeroDivisionError) as error:
                raise HarnessError("Structured note timing requires rational pairs.") from error
            if (
                onset_raw < 0 or duration_raw <= 0
                or onset_raw.denominator > TICKS_PER_QUARTER or TICKS_PER_QUARTER % onset_raw.denominator
                or duration_raw.denominator > TICKS_PER_QUARTER or TICKS_PER_QUARTER % duration_raw.denominator
            ):
                raise HarnessError("Structured note timing is outside the supported 24-tick quarter grid.")
        else:
            onset_raw = tempo.quarter_at(value["onsetSeconds"])
            duration_raw = _fraction(_finite(value["notatedDurationQuarter"], "Note duration", minimum=0))
        note = {
            "id": f"note-{index}",
            "onsetRaw": onset_raw,
            "durationRaw": duration_raw,
            "string": _integer(value["string"], "Note string", 1, 6),
            "fret": _integer(value["fret"], "Note fret", 0, MAX_FRET),
            "soundingPitchMidi": _integer(value["soundingPitchMidi"], "Sounding pitch", 0, 127),
            "voice": _integer(value["voiceIndex"], "Voice index", 0, 3),
            "harmonic": deepcopy(harmonic),
            "confidence": _finite(value["confidence"], "Note confidence", minimum=0),
            "uncertainty": list(value.get("uncertainty", [])),
            "connection": value.get("connection", "none"),
            "connectionToken": value.get("_connectionToken"),
            "connectionOriginToken": value.get("_connectionOriginToken"),
            "techniqueSchemaVersion": value.get("techniqueSchemaVersion", 3),
            "noteTechniques": deepcopy(value.get("noteTechniques", {})),
            "bendCurve": deepcopy(value.get("bendCurve")),
            "grace": deepcopy(value.get("grace")),
        }
        if note["connection"] not in ("none", "hammer_on", "pull_off", *(f"slide_{value}" for value in (1, 2, 4, 8, 16, 20, 32))):
            raise HarnessError("Predicted note connection is unsupported.")
        if note["confidence"] > 1:
            raise HarnessError("Note confidence must be from zero through one.")
        grace = note["grace"]
        if grace is not None:
            if not isinstance(grace, dict) or grace.get("mode") not in ("OnBeat", "BeforeBeat") or grace.get("transition") not in ("none", "hammer_on", "pull_off", "slide_1", "slide_2"):
                raise HarnessError("Anchored grace notation requires a supported mode and transition.")
            source_pitch = _integer(grace.get("sourcePitchMidi"), "Grace source pitch", 0, 127)
            if grace.get("onsetSeconds") is not None or grace.get("timingKnown") is not False:
                raise HarnessError("Anchored grace gestures must not invent an independent audio onset.")
            if grace.get("anchorNoteId") != value.get("noteId"):
                raise HarnessError("Grace gesture refers to a different main-note anchor.")
            source_fret = source_pitch - tuning[6 - note["string"]] - capo
            delta = note["soundingPitchMidi"] - source_pitch
            valid = (
                note["harmonic"] is None and 0 <= source_fret <= 24
                and grace.get("intervalSemitones") == delta
                and (grace["transition"] != "hammer_on" or delta > 0)
                and (grace["transition"] != "pull_off" or delta < 0)
                and (not grace["transition"].startswith("slide_") or delta != 0)
            )
            if valid:
                grace["selectedFret"] = source_fret
            else:
                note["rejectedGrace"] = {"reason": "grace_pitch_or_transition_infeasible_after_fingering", "gesture": grace}
                note["grace"] = None
        notes.append(note)
    percussion = []
    for index, value in enumerate(document.get("percussion", [])):
        required = {"onsetSeconds", "technique", "confidence"}
        if not isinstance(value, dict) or not required <= set(value) or value.get("technique") not in PERCUSSION_TYPES:
            raise HarnessError("Every percussion prediction requires a supported technique.")
        if _finite(value["onsetSeconds"], "Percussion onset", minimum=0) > duration_seconds:
            raise HarnessError("A percussion prediction falls after the declared audio duration.")
        score_onset = value.get("scoreOnsetQuarter")
        structured_flags.append(score_onset is not None)
        if score_onset is not None:
            try:
                onset_raw = Fraction(*score_onset)
            except (TypeError, ValueError, ZeroDivisionError) as error:
                raise HarnessError("Structured percussion timing requires a rational score onset.") from error
            if onset_raw < 0 or onset_raw.denominator > TICKS_PER_QUARTER or TICKS_PER_QUARTER % onset_raw.denominator:
                raise HarnessError("Structured percussion timing is outside the supported 24-tick quarter grid.")
        else:
            onset_raw = tempo.quarter_at(value["onsetSeconds"])
        event = {
            "id": f"percussion-{index}",
            "technique": value["technique"],
            "onsetRaw": onset_raw,
            "confidence": _finite(value["confidence"], "Percussion confidence", minimum=0),
        }
        if event["confidence"] > 1:
            raise HarnessError("Percussion confidence must be from zero through one.")
        percussion.append(event)
    techniques = []
    for index, value in enumerate(document.get("techniques", [])):
        required = {"onsetSeconds", "technique", "direction", "strings", "confidence"}
        if not isinstance(value, dict) or not required <= set(value) or value["technique"] not in TECHNIQUE_TYPES:
            raise HarnessError("Every acoustic technique prediction requires onset, type, direction, strings, and confidence.")
        if value["direction"] not in TECHNIQUE_DIRECTIONS or not isinstance(value["strings"], list) or any(type(string) is not int or not 1 <= string <= 6 for string in value["strings"]):
            raise HarnessError("Technique direction or string membership is invalid.")
        score_onset = value.get("scoreOnsetQuarter")
        structured_flags.append(score_onset is not None)
        onset_raw = Fraction(*score_onset) if score_onset is not None else tempo.quarter_at(value["onsetSeconds"])
        techniques.append({
            "id": f"technique-{index}",
            "onsetRaw": onset_raw,
            "onset": onset_raw,
            "technique": value["technique"],
            "direction": value["direction"],
            "strings": sorted(set(value["strings"]), reverse=True),
            "confidence": _finite(value["confidence"], "Technique confidence", minimum=0),
            "strokeFinger": value.get("strokeFinger"),
        })
        if techniques[-1]["strokeFinger"] not in (None, "a", "m", "i"):
            raise HarnessError("A normalized downstroke finger must be a, m or i.")
    if not notes and not percussion:
        raise HarnessError("GP output requires at least one decoded note or percussion event.")
    if any(structured_flags) and not all(structured_flags):
        raise HarnessError("Every event must use the same structured or nominal score-time coordinate system.")
    structured = all(structured_flags)
    if structured:
        rhythm_counts = {"constrained-24-tick": len({note["onsetRaw"] for note in notes} | {event["onsetRaw"] for event in percussion} | {event["onsetRaw"] for event in techniques})}
        for note in notes:
            note["onset"] = note["onsetRaw"]
            note["rhythmGrid"] = Fraction(1, TICKS_PER_QUARTER) if document.get("rhythmPolicy") != "fingerstyle" else (
                GRID if _fine_stroke_position(document, note["onset"]) else Fraction(1, 4))
            note["duration"] = note["durationRaw"]
            note["durationQuantized"] = note["duration"]
            note["end"] = note["onset"] + note["duration"]
        for event in percussion:
            event["onset"] = event["onsetRaw"]
            event["rhythmGrid"] = Fraction(1, TICKS_PER_QUARTER)
        for event in techniques:
            event["onset"] = event["onsetRaw"]
        declared_end = document.get("scoreAudioEndQuarter")
        if not isinstance(declared_end, list) or len(declared_end) != 2:
            raise HarnessError("Structured timing requires a rational score audio end.")
        score_end = Fraction(*declared_end)
        if score_end <= 0 or score_end.denominator > TICKS_PER_QUARTER or TICKS_PER_QUARTER % score_end.denominator:
            raise HarnessError("Structured score audio end must be positive.")
    else:
        positions, rhythm_counts = _rhythmic_positions([note["onsetRaw"] for note in notes] + [event["onsetRaw"] for event in percussion] + [event["onsetRaw"] for event in techniques])
        for note in notes:
            note["onset"], note["rhythmGrid"] = positions[note["onsetRaw"]]
            note["duration"] = _quantize(note["durationRaw"], note["rhythmGrid"])
            note["durationQuantized"] = note["duration"]
            note["duration"] = max(note["rhythmGrid"], note["duration"])
            note["end"] = note["onset"] + note["duration"]
        for event in percussion:
            event["onset"], event["rhythmGrid"] = positions[event["onsetRaw"]]
        for event in techniques:
            event["onset"], _ = positions[event["onsetRaw"]]
        score_end = tempo.quarter_at(duration_seconds)
    return tuning, capo, tempo, score_end, notes, percussion, techniques, rhythm_counts, structured


def _resolve_notes(notes, tuning, capo, audio_end):
    reconciled = []
    unresolved = []
    shortened = []
    for note in notes:
        if note["durationQuantized"] < note["rhythmGrid"]:
            shortened.append({
                "id": note["id"], "reason": "minimum_notated_duration",
                "fromDurationQuarter": _rational(note["durationQuantized"]),
                "toDurationQuarter": _rational(note["rhythmGrid"]),
            })
    dropped = []
    by_attack = {}
    for note in notes:
        key = (note["onset"], note["string"])
        current = by_attack.get(key)
        if current is None or (note["confidence"], -note["voice"], note["id"]) > (current["confidence"], -current["voice"], current["id"]):
            if current is not None:
                dropped.append({"id": current["id"], "reason": "same_string_same_quantized_attack", "kept": note["id"]})
            by_attack[key] = note
        else:
            dropped.append({"id": note["id"], "reason": "same_string_same_quantized_attack", "kept": current["id"]})
    resolved = sorted(by_attack.values(), key=lambda item: (item["onset"], -item["string"], item["voice"], item["id"]))
    for note in resolved:
        if note["end"] > audio_end:
            available = max(Fraction(0), audio_end - note["onset"])
            minimum = max(note["rhythmGrid"], Fraction(1, 6) if note["onset"].denominator % 3 == 0 else GRID)
            bounded = max(minimum, available // note["rhythmGrid"] * note["rhythmGrid"])
            shortened.append({
                "id": note["id"], "reason": "clamped_to_audio_end",
                "fromDurationQuarter": _rational(note["duration"]),
                "toDurationQuarter": _rational(bounded),
            })
            note["end"] = note["onset"] + bounded
            note["duration"] = note["end"] - note["onset"]
        if note["harmonic"] is None:
            derived = note["soundingPitchMidi"] - tuning[6 - note["string"]] - capo
            if derived != note["fret"] and 0 <= derived <= MAX_FRET:
                reconciled.append({
                    "id": note["id"], "string": note["string"], "predictedFret": note["fret"],
                    "resolvedFret": derived, "soundingPitchMidi": note["soundingPitchMidi"],
                })
                note["fret"] = derived
        note["basePitchMidi"] = tuning[6 - note["string"]] + capo + note["fret"]
        expected = note["basePitchMidi"]
        if note["harmonic"] is not None:
            offset = HARMONIC_OFFSETS[Fraction(note["harmonic"]["fret"])]
            expected = (tuning[6 - note["string"]] + capo if note["harmonic"]["type"] == "Natural" else note["basePitchMidi"]) + offset
        if expected != note["soundingPitchMidi"]:
            unresolved.append({
                "id": note["id"], "string": note["string"], "writtenFret": note["fret"],
                "predictedSoundingPitchMidi": note["soundingPitchMidi"], "writtenExpectedPitchMidi": expected,
                "harmonic": deepcopy(note["harmonic"]),
            })
    by_string = defaultdict(list)
    for note in resolved:
        by_string[note["string"]].append(note)
    for string_notes in by_string.values():
        for current, following in zip(string_notes, string_notes[1:]):
            if current["end"] > following["onset"]:
                shortened.append({
                    "id": current["id"], "reason": "same_string_reattack",
                    "fromDurationQuarter": _rational(current["duration"]),
                    "toDurationQuarter": _rational(following["onset"] - current["onset"]),
                    "following": following["id"],
                })
                current["end"] = following["onset"]
                current["duration"] = current["end"] - current["onset"]
    kept = []
    for note in resolved:
        if note["end"] <= note["onset"]:
            dropped.append({"id": note["id"], "reason": "same_string_reattack_removed_duration"})
        else:
            kept.append(note)
    return kept, reconciled, unresolved, shortened, dropped


def _resolve_percussion(percussion):
    selected = {}
    dropped = []
    for event in percussion:
        key = (event["onset"], event["technique"])
        current = selected.get(key)
        if current is None or (event["confidence"], event["id"]) > (current["confidence"], current["id"]):
            if current is not None:
                dropped.append({"id": current["id"], "reason": "duplicate_quantized_percussion", "kept": event["id"]})
            selected[key] = event
        else:
            dropped.append({"id": event["id"], "reason": "duplicate_quantized_percussion", "kept": current["id"]})
    return sorted(selected.values(), key=lambda item: (item["onset"], PERCUSSION_TYPES.index(item["technique"]))), dropped


def _attach_techniques(notes, techniques):
    attacks = sorted({note["onset"] for note in notes})
    selected, changes = {}, []
    for event in techniques:
        if not attacks:
            changes.append({"id": event["id"], "reason": "no_pitched_attack"})
            continue
        nearest = min(attacks, key=lambda onset: (abs(onset - event["onset"]), onset))
        if abs(nearest - event["onset"]) > Fraction(1, 8):
            changes.append({"id": event["id"], "reason": "no_nearby_pitched_attack", "onsetQuarter": _rational(event["onset"])})
            continue
        if nearest != event["onset"]:
            changes.append({"id": event["id"], "reason": "attach_to_existing_attack",
                            "fromQuarter": _rational(event["onset"]), "toQuarter": _rational(nearest)})
        event = {**event, "onset": nearest}
        key = nearest, event["technique"]
        current = selected.get(key)
        if current is None or event["confidence"] > current["confidence"]:
            if current is not None:
                changes.append({"id": current["id"], "reason": "duplicate_attack_articulation", "kept": event["id"]})
            selected[key] = event
        else:
            changes.append({"id": event["id"], "reason": "duplicate_attack_articulation", "kept": current["id"]})
    return sorted(selected.values(), key=lambda event: (event["onset"], event["technique"])), changes


def _bind_connection_origins(document):
    notes = document["notes"]
    seen = set()
    for index, note in enumerate(notes):
        token = note.get("noteId") if note.get("techniqueSchemaVersion") == 4 else f"connection-note-{index}"
        if not isinstance(token, str) or not token or token in seen:
            raise HarnessError("V4 notes require unique stable decoder note IDs.")
        seen.add(token)
        note["_connectionToken"] = token
        if note.get("techniqueSchemaVersion") == 4:
            note["_connectionOriginToken"] = note.get("connectionOriginNoteId")
    by_string = defaultdict(list)
    for note in notes:
        by_string[note["string"]].append(note)
    for values in by_string.values():
        values.sort(key=lambda note: (
            Fraction(*note["scoreOnsetQuarter"]) if note.get("scoreOnsetQuarter") is not None
            else Fraction(str(note["onsetSeconds"])),
            note["_connectionToken"],
        ))
        for prior, note in zip(values, values[1:]):
            if note.get("connection", "none") != "none" and note.get("techniqueSchemaVersion") != 4:
                note["_connectionOriginToken"] = prior["_connectionToken"]
    return document


def _meter_changes(document, tempo):
    metadata = document["metadata"]
    changes = [(Fraction(0), metadata["timeSignature"])]
    structured = document.get("notatedTimeSignatureChanges")
    events = structured if structured is not None else metadata.get("timeSignatureChanges", [])
    for event in events:
        if structured is not None:
            raw = Fraction(*event["scoreQuarter"])
            previous_start, previous_meter = changes[-1]
            measure = _meter_duration(previous_meter)
            pickup = Fraction(*document.get("pickupDurationQuarter", [0, 1])) if len(changes) == 1 else Fraction(0)
            base = previous_start + pickup
            quantized = base + max(1, round((raw - base) / measure)) * measure
        else:
            quantized = _quantize(tempo.quarter_at(event["timeSeconds"]))
        changes.append((quantized, event["timeSignature"]))
    return changes


def _measures(document, tempo, content_end):
    changes = _meter_changes(document, tempo)
    result = []
    cursor = Fraction(0)
    pickup_value = document.get("pickupDurationQuarter", [0, 1])
    try:
        pickup = Fraction(*pickup_value)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise HarnessError("Pickup duration must be a rational pair.") from error
    if not 0 <= pickup < _meter_duration(changes[0][1]):
        raise HarnessError("Pickup duration must be shorter than the initial measure.")
    if pickup:
        result.append({"index": 0, "start": cursor, "end": pickup, "meter": changes[0][1], "pickup": True})
        cursor = pickup
    for index, (change, meter) in enumerate(changes):
        if index == 0:
            change = cursor
        if change != cursor:
            raise HarnessError("Each time-signature change must fall on the next generated measure boundary.")
        stop = changes[index + 1][0] if index + 1 < len(changes) else None
        duration = _meter_duration(meter)
        if stop is not None:
            while cursor < stop:
                if cursor + duration > stop:
                    raise HarnessError("A time-signature change splits a generated measure.")
                result.append({"index": len(result), "start": cursor, "end": cursor + duration, "meter": meter})
                if len(result) > MAX_MEASURES:
                    raise HarnessError("Generated score exceeds the 4096-measure safety limit.")
                cursor += duration
        else:
            while cursor < content_end or not result:
                result.append({"index": len(result), "start": cursor, "end": cursor + duration, "meter": meter})
                if len(result) > MAX_MEASURES:
                    raise HarnessError("Generated score exceeds the 4096-measure safety limit.")
                cursor += duration
    return result


def _simplified_notes(notes, percussion):
    attacks = defaultdict(list)
    for note in notes:
        attacks[note["onset"]].append(note)
    positions = sorted(set(attacks) | {event["onset"] for event in percussion})
    result = []
    for position, group in sorted(attacks.items()):
        next_index = bisect_right(positions, position)
        next_position = positions[next_index] if next_index < len(positions) else None
        end = max(note["end"] for note in group)
        if next_position is not None:
            if next_position - position <= SINGLE_VOICE_BRIDGE_QUARTER:
                end = next_position
            else:
                end = min(end, next_position)
        end = max(position + GRID, end)
        for note in group:
            result.append({**deepcopy(note), "voice": 0, "end": end, "duration": end - position})
    return result


def _key_defaults(count):
    defaults = {letter: 0 for letter in LETTERS}
    order = SHARP_ORDER if count > 0 else FLAT_ORDER
    for letter in order[:abs(count)]:
        defaults[letter] = 1 if count > 0 else -1
    return defaults


def _spelling_candidates(pitch_class):
    result = []
    for letter, natural in NATURAL_PITCH.items():
        for accidental in range(-2, 3):
            if (natural + accidental) % 12 == pitch_class:
                result.append((letter, accidental))
    return result


def _spelling_for_key(pitch_class, count, preference):
    defaults = _key_defaults(count)
    direction = 1 if count > 0 or count == 0 and preference == "Sharps" else -1

    def rank(item):
        letter, accidental = item
        return (
            accidental != defaults[letter],
            abs(accidental),
            0 if accidental == 0 or accidental * direction > 0 else 1,
            LETTERS.index(letter),
        )

    return min(_spelling_candidates(pitch_class), key=rank)


def _pitch_octave(midi, spelling):
    letter, accidental = spelling
    return (midi - NATURAL_PITCH[letter] - accidental) // 12


def _note_measure(note, measures):
    starts = [measure["start"] for measure in measures]
    index = bisect_right(starts, note["onset"]) - 1
    if index < 0 or not measures[index]["start"] <= note["onset"] < measures[index]["end"]:
        raise HarnessError("A quantized note falls outside generated measures.")
    return index


def _key_cost(notes, measures, count, preference, transposed_offset):
    spelling = {pitch: _spelling_for_key(pitch % 12, count, preference) for pitch in {note["basePitchMidi"] for note in notes}}
    defaults = _key_defaults(count)
    grouped = defaultdict(list)
    for note in notes:
        grouped[(_note_measure(note, measures), note["voice"])].append(note)
    displayed = 0
    for group in grouped.values():
        state = {}
        for note in sorted(group, key=lambda item: (item["onset"], -item["string"], item["id"])):
            letter, accidental = spelling[note["basePitchMidi"]]
            octave = _pitch_octave(note["basePitchMidi"] + transposed_offset, (letter, accidental))
            key = (letter, octave)
            current = state.get(key, defaults[letter])
            if accidental != current:
                displayed += 1
                state[key] = accidental
    return displayed, spelling


def _select_key(notes, measures, preference, transposed_offset):
    results = []
    for count in range(-7, 8):
        cost, spelling = _key_cost(notes, measures, count, preference, transposed_offset)
        direction_penalty = 0 if count == 0 or count > 0 and preference == "Sharps" or count < 0 and preference == "Flats" else 1
        results.append(((cost, abs(count), direction_penalty, KEY_ORDER.index(count)), count, spelling))
    rank, count, spelling = min(results, key=lambda item: item[0])
    return count, spelling, rank[0]


def _rhythm_palette():
    result = {}
    for value, base in NOTE_VALUES.items():
        for dots in range(3):
            duration = base * (2 - Fraction(1, 2 ** dots))
            result.setdefault(duration, (value, dots, None))
        result.setdefault(base * Fraction(2, 3), (value, 0, (3, 2)))
    return sorted(result.items(), key=lambda item: (-item[0], item[1][2] is not None, item[1][1]))


RHYTHM_PALETTE = _rhythm_palette()


def _split_duration(duration):
    def decompose(palette):
        result = []
        remaining = duration
        for value, notation in palette:
            while value <= remaining:
                result.append((value, notation))
                remaining -= value
        return result, remaining

    binary = [item for item in RHYTHM_PALETTE if item[1][2] is None]
    # A triplet interval must not first consume binary values and leave an
    # arbitrary tiny tuplet remainder (e.g. 1/3 quarter is one triplet eighth).
    palette = binary if duration.denominator & (duration.denominator - 1) == 0 else [
        item for item in RHYTHM_PALETTE if item[1][2] is not None
    ]
    result, remaining = decompose(palette)
    if not remaining:
        return result
    raise HarnessError(f"Duration {duration} cannot be spelled using supported binary or triplet values.")


def _spell_interval(start, stop, measure, *, rest=False):
    meter = measure["meter"]
    numerator, denominator = meter
    if numerator not in (2, 3, 4) and not (numerator > 3 and numerator % 3 == 0):
        return _split_duration(stop - start)
    group = Fraction(12, denominator) if numerator > 3 and numerator % 3 == 0 else Fraction(4, denominator)
    origin = measure["start"]
    if measure.get("pickup"):
        origin = measure["end"] - _meter_duration(meter)

    @lru_cache(None)
    def spell(cursor):
        if cursor == stop:
            return ()
        offset = cursor - origin
        boundary = origin + (offset // group + 1) * group
        local_stop = min(stop, boundary)
        triplet = cursor.denominator % 3 == 0 or local_stop.denominator % 3 == 0
        best = None
        for duration, notation in RHYTHM_PALETTE:
            if measure.get("rhythmPolicy") == "fingerstyle" and (notation[1] > 1 or notation[1] and NOTE_VALUES[notation[0]] < Fraction(1, 2)):
                continue
            end = cursor + duration
            if end > stop:
                continue
            if triplet:
                if notation[2] is None or (duration * 6).denominator != 1 or end > boundary:
                    continue
            elif notation[2] is not None or (duration * 8).denominator != 1:
                continue
            if offset % group and end > boundary:
                continue
            if rest and end > boundary and (cursor != measure["start"] or end != measure["end"]):
                continue
            compound_group = numerator > 3 and numerator % 3 == 0 and not offset % group and not duration % group
            if not notation[2] and not compound_group and duration >= group and NOTE_VALUES[notation[0]] > group and offset % NOTE_VALUES[notation[0]]:
                continue
            tail = spell(end)
            if tail is None:
                continue
            candidate = ((duration, notation), *tail)
            rank = (len(candidate), sum(row[1][1] > 1 for row in candidate), sum(row[1][2] is not None for row in candidate))
            if best is None or rank < best[0]:
                best = rank, candidate
        return None if best is None else best[1]

    result = spell(start)
    if result is None:
        raise HarnessError(f"Cannot spell metrical interval {start}..{stop} in {numerator}/{denominator}.")
    return list(result)


def _prototype_pitch_offsets(root):
    transpose = root.find("./Tracks/Track/Transpose")
    if transpose is None:
        raise HarnessError("The GP template requires explicit track transposition metadata.")
    chromatic = int(transpose.findtext("Chromatic", "0"))
    octave = int(transpose.findtext("Octave", "0"))
    if chromatic != 0:
        raise HarnessError("Chromatically transposing GP templates are not supported by deterministic pitch spelling.")
    return {"ConcertPitch": 0, "TransposedPitch": -chromatic - 12 * octave}


def _pitch_property_spelled(name, midi, spelling):
    letter, accidental = spelling
    symbols = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "x"}
    prop = ET.Element("Property", name=name)
    pitch = ET.SubElement(prop, "Pitch")
    ET.SubElement(pitch, "Step").text = letter
    ET.SubElement(pitch, "Accidental").text = symbols[accidental]
    ET.SubElement(pitch, "Octave").text = str(_pitch_octave(midi, spelling))
    return prop


def _template_invariant(root):
    value = deepcopy(root)
    for section, tag in (("MasterBars", "MasterBar"), ("Bars", "Bar"), ("Voices", "Voice"), ("Beats", "Beat"), ("Notes", "Note"), ("Rhythms", "Rhythm")):
        parent = value.find(section)
        if parent is not None:
            _remove(parent, tag)
    automations = value.find("./MasterTrack/Automations")
    if automations is not None:
        for node in list(automations):
            if node.tag == "Automation" and node.findtext("Type") == "Tempo":
                automations.remove(node)
    properties = value.find("./Tracks/Track/Staves/Staff/Properties")
    if properties is not None:
        for prop in list(properties):
            if prop.tag == "Property" and prop.get("name") in {"Tuning", "CapoFret", "FretCount"}:
                properties.remove(prop)
    for path in ("./Tracks/Track/SystemsLayout", "./Score/MultiVoice"):
        node = value.find(path)
        if node is not None:
            node.text = "__GENERATED__"
    return semantic_node(value)


class _ScoreWriter:
    def __init__(self, root, tuning, capo, measures, notes, percussion, techniques, key_count, spellings, *, simplified):
        notes = deepcopy(notes)
        self.root = root
        self.tuning = tuning
        self.capo = capo
        self.measures = measures
        self.notes = notes
        self.percussion = percussion
        self.techniques = techniques
        self.key_count = key_count
        self.spellings = spellings
        self.simplified = simplified
        self.pitch_offsets = _prototype_pitch_offsets(root)
        self.pitch_profile = {"offsets": self.pitch_offsets, "spellings": {}}
        self.fallbacks = []
        self.connection_fallbacks = []
        self.articulation_fallbacks = []
        for note in notes:
            if note.get("rejectedGrace"):
                self.articulation_fallbacks.append({
                    "id": note["id"], "onsetQuarter": _rational(note["onset"]), **note["rejectedGrace"],
                })
            grace = note.get("grace")
            if grace is not None and grace["transition"] in ("hammer_on", "pull_off"):
                note["hopoDestination"] = True
        by_string = defaultdict(list)
        for note in notes:
            by_string[note["string"]].append(note)
        previous = {}
        for values in by_string.values():
            values.sort(key=lambda note: (note["onset"], note["id"]))
            for prior, note in zip(values, values[1:]):
                previous[note["connectionToken"]] = prior
        by_token = {note["connectionToken"]: note for note in notes}
        for note in notes:
            connection = note.get("connection", "none")
            if connection == "none":
                continue
            prior = by_token.get(note.get("connectionOriginToken"))
            valid = prior is not None and previous.get(note["connectionToken"]) is prior
            valid = valid and prior["string"] == note["string"] and prior["onset"] < note["onset"]
            if note.get("techniqueSchemaVersion") == 4:
                valid = valid and prior["voice"] == note["voice"] and prior["end"] >= note["onset"] and note.get("grace") is None
            else:
                valid = valid and note["onset"] - prior["onset"] <= 2
            if connection == "hammer_on":
                valid = valid and note["soundingPitchMidi"] > prior["soundingPitchMidi"]
            elif connection == "pull_off":
                valid = valid and note["soundingPitchMidi"] < prior["soundingPitchMidi"]
            elif connection.startswith("slide_"):
                valid = valid and note["soundingPitchMidi"] != prior["soundingPitchMidi"]
            if not valid:
                self.connection_fallbacks.append({
                    "id": note["id"], "onsetQuarter": _rational(note["onset"]),
                    "connection": connection, "reason": "invalid_connection_relationship",
                })
                continue
            if connection in ("hammer_on", "pull_off"):
                prior["hopoOrigin"] = True
                note["hopoDestination"] = True
            elif connection.startswith("slide_"):
                prior["slideFlags"] = int(connection.split("_", 1)[1])
        self.ids = defaultdict(int)
        self.rhythms = {}
        voices = sorted({note["voice"] for note in notes} | {0})
        self.percussion_voices = {}
        for event in percussion:
            def host_cost(voice):
                local = [note for note in notes if note["voice"] == voice]
                boundary = any(event["onset"] in (note["onset"], note["end"]) for note in local)
                active = sum(note["onset"] < event["onset"] < note["end"] for note in local)
                return (0 if boundary else 1 + 100 * active, voice)

            self.percussion_voices[event["id"]] = min(voices, key=host_cost)
        self.technique_voices = {}
        for event in techniques:
            def technique_host_cost(voice):
                at_onset = [note for note in notes if note["voice"] == voice and note["onset"] == event["onset"]]
                overlap = len(set(event["strings"]) & {note["string"] for note in at_onset})
                return (-overlap, voice)

            self.technique_voices[event["id"]] = min(voices, key=technique_host_cost)
        masters = root.findall("./MasterBars/MasterBar")
        bars = root.findall("./Bars/Bar")
        if not masters or not bars:
            raise HarnessError("The GP template requires at least one placeholder master bar and bar.")
        self.master_prototype = deepcopy(masters[0])
        self.bar_prototype = deepcopy(bars[0])
        for section, tag in (("MasterBars", "MasterBar"), ("Bars", "Bar"), ("Voices", "Voice"), ("Beats", "Beat"), ("Notes", "Note"), ("Rhythms", "Rhythm")):
            parent = root.find(section)
            if parent is None:
                raise HarnessError(f"The GP template has no {section} section.")
            _remove(parent, tag)

    def identifier(self, section):
        value = self.ids[section]
        self.ids[section] += 1
        return str(value)

    def rhythm(self, notation):
        if notation in self.rhythms:
            return self.rhythms[notation]
        value, dots, tuplet = notation
        identifier = self.identifier("Rhythms")
        node = ET.Element("Rhythm", id=identifier)
        ET.SubElement(node, "NoteValue").text = value
        if dots:
            ET.SubElement(node, "AugmentationDot", count=str(dots))
        if tuplet:
            ET.SubElement(node, "PrimaryTuplet", num=str(tuplet[0]), den=str(tuplet[1]))
        self.root.find("Rhythms").append(node)
        self.rhythms[notation] = identifier
        return identifier

    def pitched_note(self, note, start, stop):
        identifier = self.identifier("Notes")
        node = ET.Element("Note", id=identifier)
        incoming = start > note["onset"]
        outgoing = stop < note["end"]
        if incoming or outgoing:
            ET.SubElement(node, "Tie", origin=str(outgoing).lower(), destination=str(incoming).lower())
        ET.SubElement(node, "InstrumentArticulation").text = "0"
        props = ET.SubElement(node, "Properties")
        midi = note["basePitchMidi"]
        spelling = self.spellings[midi]
        for name in ("ConcertPitch", "TransposedPitch"):
            props.append(_pitch_property_spelled(name, midi + self.pitch_offsets[name], spelling))
        for name, tag, value in (
            ("Fret", "Fret", note["fret"]),
            ("Midi", "Number", midi),
            ("String", "String", 6 - note["string"]),
        ):
            ET.SubElement(ET.SubElement(props, "Property", name=name), tag).text = str(value)
        for name in ("hopoOrigin", "hopoDestination"):
            attack_only = name == "hopoDestination"
            if note.get(name) and (not incoming if attack_only else not outgoing):
                native = "HopoOrigin" if name == "hopoOrigin" else "HopoDestination"
                ET.SubElement(ET.SubElement(props, "Property", name=native), "Enable")
        technique_scores = note.get("noteTechniques", {})
        slide_flags = note.get("slideFlags", 0) if not outgoing else 0
        for name, bit, enabled in (
            ("slide_in_below", 16, not incoming), ("slide_in_above", 32, not incoming),
            ("slide_out_down", 4, not outgoing), ("slide_out_up", 8, not outgoing),
        ):
            if enabled and technique_scores.get(name, 0) >= .5:
                slide_flags |= bit
        if slide_flags & 48 == 48 or sum(bool(slide_flags & bit) for bit in (1, 2, 4, 8)) > 1:
            self.articulation_fallbacks.append({
                "id": note["id"], "onsetQuarter": _rational(start),
                "reason": "conflicting_native_slide_flags", "predictedFlags": slide_flags,
            })
            slide_flags = (slide_flags & 48 if slide_flags & 48 != 48 else 0) | (note.get("slideFlags", 0) if not outgoing else 0)
        if slide_flags:
            ET.SubElement(ET.SubElement(props, "Property", name="Slide"), "Flags").text = str(slide_flags)
        for name, native in (("tap", "Tapped"), ("left_hand_tap", "LeftHandTapped")):
            if technique_scores.get(name, 0) >= .5 and not incoming:
                ET.SubElement(ET.SubElement(props, "Property", name=native), "Enable")
        if technique_scores.get("vibrato", 0) >= .5 and not incoming:
            ET.SubElement(node, "Vibrato").text = "Slight"
        if technique_scores.get("bend", 0) >= .5 and not incoming:
            curve = note.get("bendCurve")
            if outgoing:
                self.articulation_fallbacks.append({
                    "id": note["id"], "onsetQuarter": _rational(note["onset"]),
                    "technique": "bend", "reason": "bend_curve_crosses_tie",
                })
            elif not isinstance(curve, dict) or set(curve) != {
                "OriginOffset", "OriginValue", "MiddleOffset1", "MiddleOffset2",
                "MiddleValue", "DestinationOffset", "DestinationValue",
            }:
                note.setdefault("uncertainty", []).append("bend_curve_unavailable")
            else:
                ET.SubElement(ET.SubElement(props, "Property", name="Bended"), "Enable")
                for name, value in curve.items():
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise HarnessError("Predicted bend curve contains a nonfinite value.")
                    ET.SubElement(ET.SubElement(props, "Property", name=f"Bend{name}"), "Float").text = f"{max(0, min(100, value)):.6f}"
        if note["harmonic"] is not None:
            ET.SubElement(ET.SubElement(props, "Property", name="Harmonic"), "Enable")
            ET.SubElement(ET.SubElement(props, "Property", name="HarmonicFret"), "HFret").text = f"{note['harmonic']['fret']:.6f}"
            ET.SubElement(ET.SubElement(props, "Property", name="HarmonicType"), "HType").text = note["harmonic"]["type"]
        self.root.find("Notes").append(node)
        return identifier

    def grace_beats(self, active, onset, voice):
        groups = defaultdict(list)
        for note in active:
            if note["onset"] == onset and note.get("grace") is not None:
                groups[note["grace"]["mode"]].append(note)
        result = []
        for mode, notes in sorted(groups.items()):
            beat = ET.Element("Beat", id=self.identifier("Beats"))
            ET.SubElement(beat, "Dynamic").text = "MF"
            ET.SubElement(beat, "Rhythm", ref=self.rhythm(("32nd", 0, None)))
            ET.SubElement(beat, "GraceNotes").text = mode
            ET.SubElement(beat, "TransposedPitchStemOrientation").text = "Upward" if voice == 0 else "Downward"
            ids = []
            for note in sorted(notes, key=lambda item: -item["string"]):
                grace = note["grace"]
                source = {
                    **note, "fret": grace["selectedFret"], "basePitchMidi": grace["sourcePitchMidi"],
                    "soundingPitchMidi": grace["sourcePitchMidi"], "onset": onset, "end": onset,
                    "harmonic": None, "noteTechniques": {}, "bendCurve": None,
                    "hopoOrigin": grace["transition"] in ("hammer_on", "pull_off"),
                    "hopoDestination": False,
                    "slideFlags": int(grace["transition"].split("_")[1]) if grace["transition"].startswith("slide_") else 0,
                }
                ids.append(self.pitched_note(source, onset, onset))
            ET.SubElement(beat, "Notes").text = " ".join(ids)
            self.root.find("Beats").append(beat)
            result.append(beat.get("id"))
        return result

    def percussion_note(self, event, string):
        identifier = self.identifier("Notes")
        node = ghost_dead_note(identifier, string, self.tuning, self.capo, pitch_profile=self.pitch_profile)
        if event["technique"] == "thumb_slap":
            anti = node.find("AntiAccent")
            if anti is not None:
                node.remove(anti)
        self.root.find("Notes").append(node)
        return identifier

    def beat(self, notation, active, carriers, text, techniques, voice, start, stop):
        identifier = self.identifier("Beats")
        beat = ET.Element("Beat", id=identifier)
        ET.SubElement(beat, "Dynamic").text = "MF"
        ET.SubElement(beat, "Rhythm", ref=self.rhythm(notation))
        ET.SubElement(beat, "TransposedPitchStemOrientation").text = "Upward" if voice == 0 else "Downward"
        ET.SubElement(beat, "ConcertPitchStemOrientation").text = "Undefined"
        if text:
            ET.SubElement(beat, "FreeText").text = text
        self.apply_techniques(beat, techniques)
        note_ids = [self.pitched_note(note, start, stop) for note in sorted(active, key=lambda item: -item["string"])]
        note_ids.extend(self.percussion_note(event, string) for event, string in carriers)
        if note_ids:
            ET.SubElement(beat, "Notes").text = " ".join(note_ids)
        self.root.find("Beats").append(beat)
        return identifier

    def percussion_at(self, onset, voice):
        return [
            event for event in self.percussion
            if event["onset"] == onset and self.percussion_voices[event["id"]] == voice
        ]

    def active_strings(self, onset):
        return {note["string"] for note in self.notes if note["onset"] <= onset < note["end"]}

    def percussion_content(self, onset, voice):
        events = self.percussion_at(onset, voice)
        occupied = self.active_strings(onset)
        carriers = []
        text = []
        for event in events:
            if event["technique"] == "wrist_thump":
                text.append("O")
                continue
            string = next((value for value in (6, 5, 4, 3, 2, 1) if value not in occupied), None)
            if string is None:
                marker = "X" if event["technique"] == "thumb_slap" else "(X)"
                text.append(marker)
                self.fallbacks.append({"id": event["id"], "onsetQuarter": _rational(onset), "text": marker, "reason": "all_strings_occupied"})
            else:
                occupied.add(string)
                carriers.append((event, string))
        return carriers, " ".join(text)

    def technique_content(self, onset, voice):
        return [
            event for event in self.techniques
            if event["onset"] == onset and self.technique_voices[event["id"]] == voice
        ]

    @staticmethod
    def apply_techniques(beat, events):
        if not events:
            return
        properties = beat.find("Properties")
        if properties is None:
            properties = ET.SubElement(beat, "Properties")
        for event in events:
            technique, direction = event["technique"], event["direction"]
            if technique == "rasgueado":
                raise HarnessError("Native GP Rasgueado playback is forbidden; use independently timed brush downstrokes.")
            if event.get("strokeFinger"):
                existing = beat.findtext("FreeText", "")
                _set_text(beat, "FreeText", " ".join((*existing.split(), event["strokeFinger"])))
            if technique == "arpeggio":
                _set_text(beat, "Arpeggio", direction)
                continue
            name, child, value = {
                "brush": ("Brush", "Direction", direction),
                "pick_stroke": ("PickStroke", "Direction", direction),
            }[technique]
            prop = properties.find(f"./Property[@name='{name}']")
            if prop is None:
                prop = ET.SubElement(properties, "Property", name=name)
            _set_text(prop, child, value)
            if technique == "brush":
                xproperties = beat.find("XProperties")
                if xproperties is None:
                    xproperties = ET.SubElement(beat, "XProperties")
                for identifier, child, value in ((BRUSH_DURATION_XPROPERTY, "Int", str(INSTANT_BRUSH_DURATION_TICKS)), (BRUSH_START_XPROPERTY, "Float", "0")):
                    prop = xproperties.find(f"./XProperty[@id='{identifier}']")
                    if prop is None:
                        prop = ET.SubElement(xproperties, "XProperty", id=identifier)
                    _set_text(prop, child, value)

    def voice_beats(self, measure, voice):
        notes = [note for note in self.notes if note["voice"] == voice and note["onset"] < measure["end"] and note["end"] > measure["start"]]
        boundaries = {measure["start"], measure["end"]}
        for note in notes:
            boundaries.update((max(measure["start"], note["onset"]), min(measure["end"], note["end"])))
        hosted = [event for event in self.percussion if self.percussion_voices[event["id"]] == voice]
        hosted_techniques = [event for event in self.techniques if self.technique_voices[event["id"]] == voice]
        for event in hosted_techniques:
            if measure["start"] <= event["onset"] < measure["end"]:
                boundaries.add(event["onset"])
        if hosted:
            for event in hosted:
                if measure["start"] <= event["onset"] < measure["end"]:
                    boundaries.add(event["onset"])
            for event in hosted:
                if measure["start"] <= event["onset"] < measure["end"]:
                    active = any(note["onset"] <= event["onset"] < note["end"] for note in notes)
                    if not active:
                        later = min((value for value in boundaries if value > event["onset"]), default=measure["end"])
                        maximum = SINGLE_VOICE_BRIDGE_QUARTER if self.simplified else PERCUSSION_DURATION
                        beat = event["onset"] // 1
                        if not self.simplified and any(
                            point // 1 == beat and point.denominator % 3 == 0 for point in boundaries
                        ):
                            maximum = Fraction(1, 6)
                        if later - event["onset"] > maximum:
                            boundaries.add(event["onset"] + maximum)
        points = sorted(boundaries)
        beats = []
        for left, right in zip(points, points[1:]):
            active = [note for note in notes if note["onset"] <= left < note["end"]]
            carriers, text = self.percussion_content(left, voice)
            techniques = self.technique_content(left, voice)
            cursor = left
            for duration, notation in _spell_interval(left, right, measure, rest=not active and not carriers and not text):
                stop = cursor + duration
                local_carriers = carriers if cursor == left else []
                local_text = text if cursor == left else ""
                beats.extend(self.grace_beats(active, cursor, voice))
                beats.append(self.beat(notation, active, local_carriers, local_text, techniques if cursor == left else [], voice, cursor, stop))
                cursor = stop
        return beats

    def build(self):
        master_parent = self.root.find("MasterBars")
        bar_parent = self.root.find("Bars")
        voices_used = {note["voice"] for note in self.notes} | set(self.percussion_voices.values()) | set(self.technique_voices.values())
        for measure in self.measures:
            master = deepcopy(self.master_prototype)
            for tag in ("Repeat", "AlternateEndings", "Directions", "Section", "Fermatas", "TripletFeel", "DoubleBar", "XProperties"):
                _remove(master, tag)
            key = master.find("Key")
            if key is None:
                key = ET.Element("Key")
                master.insert(0, key)
            _set_text(key, "AccidentalCount", str(self.key_count))
            _set_text(key, "Mode", "Major")
            _set_text(key, "TransposeAs", "Sharps" if self.key_count >= 0 else "Flats")
            _set_text(master, "Time", f"{measure['meter'][0]}/{measure['meter'][1]}")
            bar = deepcopy(self.bar_prototype)
            bar.set("id", self.identifier("Bars"))
            _remove(bar, "XProperties")
            voice_refs = ["-1"] * 4
            active_voices = sorted(voices_used) if not self.simplified else [0]
            if 0 not in active_voices:
                active_voices.insert(0, 0)
            for voice in active_voices:
                if voice != 0 and all(
                    not (note["voice"] == voice and note["onset"] < measure["end"] and note["end"] > measure["start"])
                    for note in self.notes
                ) and all(
                    self.percussion_voices[event["id"]] != voice or not measure["start"] <= event["onset"] < measure["end"]
                    for event in self.percussion
                ) and all(
                    self.technique_voices[event["id"]] != voice or not measure["start"] <= event["onset"] < measure["end"]
                    for event in self.techniques
                ):
                    continue
                beat_ids = self.voice_beats(measure, voice)
                voice_node = ET.Element("Voice", id=self.identifier("Voices"))
                ET.SubElement(voice_node, "Beats").text = " ".join(beat_ids)
                self.root.find("Voices").append(voice_node)
                voice_refs[voice] = voice_node.get("id")
            _set_text(bar, "Voices", " ".join(voice_refs))
            bar_parent.append(bar)
            _set_text(master, "Bars", bar.get("id"))
            master_parent.append(master)
        return self.fallbacks


def _tempo_automations(root, tempo, measures, notated_changes=None, initial_tempo=None):
    parent = root.find("./MasterTrack/Automations")
    if parent is None:
        master = root.find("MasterTrack")
        if master is None:
            raise HarnessError("The GP template has no MasterTrack.")
        parent = ET.SubElement(master, "Automations")
    for node in list(parent):
        if node.tag == "Automation" and node.findtext("Type") == "Tempo":
            parent.remove(node)
    reverse_units = {value: key for key, value in TEMPO_BEAT_UNITS.items()}
    events = [(initial_tempo or tempo.events[0], Fraction(0))]
    if notated_changes is None:
        events.extend(zip(tempo.events[1:], tempo.quarters[1:]))
    else:
        events.extend((event, Fraction(*event["scoreQuarter"])) for event in notated_changes)
    for event, quarter in events:
        beat_unit = event["beatUnit"]
        beat_unit = Fraction(*beat_unit) if isinstance(beat_unit, list) else beat_unit
        if beat_unit not in reverse_units:
            raise HarnessError("The GP writer supports tempo beat units 1/8, 1/4, dotted 1/4, 1/2, and dotted 1/2.")
        index = bisect_right([measure["start"] for measure in measures], quarter) - 1
        if index < 0:
            index = 0
        if index >= len(measures):
            continue
        measure = measures[index]
        position = (quarter - measure["start"]) / (measure["end"] - measure["start"])
        if not 0 <= position <= 1:
            raise HarnessError("A tempo event falls outside generated measures.")
        node = ET.SubElement(parent, "Automation")
        for tag, value in (
            ("Type", "Tempo"),
            ("Linear", str(event["linear"]).lower()),
            ("Bar", index),
            ("Position", f"{float(position):.12g}"),
            ("Visible", "true"),
            ("Value", f"{float(event['bpm']):.12g} {reverse_units[beat_unit]}"),
        ):
            ET.SubElement(node, tag).text = str(value)


def _instrument(root, tuning, capo, fret_count):
    tracks = root.findall("./Tracks/Track")
    if len(tracks) != 1 or len(tracks[0].findall("./Staves/Staff")) != 1:
        raise HarnessError("The GP output template must contain one track with one staff.")
    properties = tracks[0].find("./Staves/Staff/Properties")
    if properties is None:
        raise HarnessError("The GP output template has no staff properties.")
    values = {
        "Tuning": ("Pitches", " ".join(str(value) for value in tuning)),
        "CapoFret": ("Fret", capo),
        "FretCount": ("Number", fret_count),
    }
    for name, (tag, value) in values.items():
        prop = properties.find(f"Property[@name='{name}']")
        if prop is None:
            prop = ET.SubElement(properties, "Property", name=name)
        _set_text(prop, tag, value)


def _systems_layout(root, measure_count):
    track = root.find("./Tracks/Track")
    layout = track.find("SystemsLayout")
    if layout is None:
        return
    original = [int(value) for value in (layout.text or "").split() if value.isdecimal() and int(value) > 0]
    width = original[0] if original else 4
    systems = [width] * (measure_count // width)
    if measure_count % width:
        systems.append(measure_count % width)
    layout.text = " ".join(str(value) for value in systems)


def _multivoice(root, enabled):
    node = root.find("./Score/MultiVoice")
    if node is not None:
        suffix = ">" if (node.text or "").endswith(">") else ""
        node.text = ("1" if enabled else "0") + suffix


def _build_variant(template_raw, notes, percussion, techniques, measures, tuning, capo, tempo, *, simplified, notated_tempo_changes=None, initial_tempo=None):
    from .gp_stylesheet import clear_template_attribution

    root, payloads, comment = _parse_archive(template_raw)
    payloads, cleared_attribution = clear_template_attribution(root, payloads)
    before = _template_invariant(root)
    offsets = _prototype_pitch_offsets(root)
    preference = root.findtext("./MasterBars/MasterBar/Key/TransposeAs", "Sharps")
    if preference not in ("Sharps", "Flats"):
        preference = "Sharps"
    key_notes = notes + [
        {**note, "basePitchMidi": note["grace"]["sourcePitchMidi"]}
        for note in notes if note.get("grace") is not None
    ]
    key_count, spellings, accidental_count = _select_key(key_notes, measures, preference, offsets["TransposedPitch"])
    writer = _ScoreWriter(root, tuning, capo, measures, notes, percussion, techniques, key_count, spellings, simplified=simplified)
    fallbacks = writer.build()
    _tempo_automations(root, tempo, measures, notated_tempo_changes, initial_tempo)
    _instrument(root, tuning, capo, max(24, max((note["fret"] for note in notes), default=0)))
    _systems_layout(root, len(measures))
    _multivoice(root, not simplified and any(note["voice"] > 0 for note in notes))
    if _template_invariant(root) != before:
        raise HarnessError("GP output changed template content outside the allowed musical and bar-layout fields.")
    output = _archive_bytes(root, payloads, comment)
    parsed, _, _ = _parse_archive(output)
    decoded = decode_score(ET.fromstring(ET.tostring(parsed)), tuning, capo)
    if len(decoded["measures"]) != len(measures):
        raise HarnessError("Generated GP measure count failed validation.")
    if any(issue["code"] in {"underfull_measure", "overfull_measure"} for issue in decoded["issues"]):
        raise HarnessError("Generated GP contains an incomplete measure voice.")
    return output, {
        "clearedTemplateAttribution": cleared_attribution,
        "keyAccidentalCount": key_count,
        "displayedNoteAccidentalCount": accidental_count,
        "measureCount": len(measures),
        "noteSegmentCount": len(parsed.findall("./Notes/Note")),
        "voiceCount": 1 if simplified else max((note["voice"] for note in notes), default=0) + 1,
        "percussionTextFallbacks": fallbacks,
        "connectionFallbacks": writer.connection_fallbacks,
        "articulationFallbacks": writer.articulation_fallbacks,
    }


def _atomic_bytes(path, content):
    path = Path(path).absolute()
    if path.resolve() != path or any(part.exists() and (part.is_symlink() or getattr(part.lstat(), "st_file_attributes", 0) & 0x400) for part in (path, *path.parents)) or path.exists() and path.stat().st_nlink > 1:
        raise HarnessError(f"GP output aliases are not allowed: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise HarnessError(f"Refusing to overwrite existing GP output: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_gp_outputs(template_path, predictions, full_path, single_path, *, profile=DraftProfile(), beat_evidence=None, include_unsupported_tail=False, progress=None):
    def emit(message):
        if progress is not None:
            progress(f"GP export: {message}")

    emit("reading template and validating export inputs...")
    if profile.rhythm_policy == "fingerstyle" and beat_evidence is None:
        raise HarnessError("The fingerstyle rhythm policy requires beat evidence to locate supported 32nd-note stroke figures.")
    template_path = Path(template_path).resolve()
    if not template_path.is_file():
        raise HarnessError(f"GP output template does not exist: {template_path}")
    template_raw = template_path.read_bytes()
    template_hash = hashlib.sha256(template_raw).hexdigest()
    if Path(full_path).absolute() == Path(single_path).absolute():
        raise HarnessError("Full-voice and single-voice GP outputs require different paths.")
    emit("template and inputs ready; filtering note, percussion and technique candidates...")
    cleaned, cleanup = clean_hypotheses(predictions, profile)
    emit(f"confidence filtering completed ({len(cleaned['notes'])} note candidates).")
    rhythm_inference = None
    stroke_normalization = None
    if beat_evidence is not None:
        if beat_evidence.get("audioSha256") != predictions.get("audioSha256"):
            raise HarnessError("Beat evidence and transcription hypotheses reference different audio.")
        from .rhythm_inference import infer_notated_timing

        emit("quantizing rhythm and selecting durations...")
        cleaned, rhythm_inference = infer_notated_timing(cleaned, beat_evidence, include_unsupported_tail=include_unsupported_tail, rhythm_policy=profile.rhythm_policy)
        emit("rhythm quantization completed; normalizing downstroke groups...")
        from .stroke_normalization import normalize_downstroke_bursts

        cleaned, stroke_normalization = normalize_downstroke_bursts(cleaned)
        emit("downstroke normalization completed.")
    cleaned = _bind_connection_origins(cleaned)
    from .fingering_optimizer import optimize_fingerings

    emit("optimizing playable fingering candidates...")
    cleaned, fingering_optimization = optimize_fingerings(cleaned)
    emit("fingering optimization completed.")
    suppressed_rasgueado = [event for event in cleaned.get("techniques", []) if event["technique"] == "rasgueado"]
    cleaned["techniques"] = [event for event in cleaned.get("techniques", []) if event["technique"] != "rasgueado"]
    tuning, capo, tempo, audio_end, raw_notes, raw_percussion, raw_techniques, rhythm_counts, structured = _validate_predictions(cleaned)
    notes, reconciled, unresolved, shortened, dropped_notes = _resolve_notes(raw_notes, tuning, capo, audio_end)
    percussion, dropped_percussion = _resolve_percussion(raw_percussion)
    raw_techniques, technique_attachment = _attach_techniques(notes, raw_techniques)
    voice_optimization = None
    if rhythm_inference is not None:
        from .voice_optimizer import optimize_voices

        voice_input = {"notes": [{
            "scoreOnsetQuarter": _rational(note["onset"]), "scoreDurationQuarter": _rational(note["duration"]),
            "soundingPitchMidi": note["soundingPitchMidi"], "voiceIndex": note["voice"],
        } for note in notes]}
        emit("assigning voices and reducing tied fragments...")
        voiced, voice_optimization = optimize_voices(voice_input)
        for note, value in zip(notes, voiced["notes"]):
            note["voice"] = value["voiceIndex"]
        emit("voice assignment completed.")
    content_end = max(audio_end, max([note["onset"] + note["rhythmGrid"] for note in notes] + [event["onset"] + PERCUSSION_DURATION for event in percussion] + [event["onset"] + GRID for event in raw_techniques], default=GRID))
    measures = _measures(cleaned, tempo, content_end)
    for measure in measures:
        measure["rhythmPolicy"] = profile.rhythm_policy
    notated_tempo_changes = cleaned.get("notatedTempoChanges")
    initial_tempo = cleaned.get("notatedInitialTempo")
    emit("building full-voice GP notation...")
    full, full_report = _build_variant(template_raw, notes, percussion, raw_techniques, measures, tuning, capo, tempo, simplified=False, notated_tempo_changes=notated_tempo_changes, initial_tempo=initial_tempo)
    emit("full-voice notation completed; building the single-voice version...")
    simple_notes = _simplified_notes(notes, percussion)
    single, single_report = _build_variant(template_raw, simple_notes, percussion, raw_techniques, measures, tuning, capo, tempo, simplified=True, notated_tempo_changes=notated_tempo_changes, initial_tempo=initial_tempo)
    emit("single-voice notation completed.")
    if hashlib.sha256(template_path.read_bytes()).hexdigest() != template_hash:
        raise HarnessError("GP template changed while outputs were generated.")
    report = {
        "schemaVersion": 1,
        "kind": "gp-output-report",
        "templateSha256": template_hash,
        "templateModified": False,
        "finestCandidateQuarterGrid": _rational(GRID),
        "sourceHypotheses": {"notes": len(predictions["notes"]), "percussion": len(predictions["percussion"])},
        "cleanedHypotheses": {"notes": len(raw_notes), "percussion": len(raw_percussion)},
        "resolvedHypotheses": {"notes": len(notes), "percussion": len(percussion)},
        "techniqueHypotheses": len(raw_techniques),
        "draftCleanup": cleanup,
        "rhythmInference": rhythm_inference,
        "fingeringOptimization": fingering_optimization,
        "strokePolicy": "Never emit native GP Rasgueado. Normalize supported overly dense downstroke groups to a-m-i ending on a beat; preserve other note candidates and never synthesize missing strokes from a compound score alone.",
        "strokeNormalization": stroke_normalization,
        "suppressedPostprocessingRasgueado": suppressed_rasgueado,
        "rhythmSpellingPolicy": "Minimize written fragments under simple/compound beat grouping. Off-beat carries and rests expose beat boundaries; binary/triplet portions use their own values. Preserve logical attacks and sustain. Unspecified additive groupings retain duration-only spelling.",
        "voiceOptimization": voice_optimization,
        "techniqueAttachment": technique_attachment,
        "rhythmicGridPolicy": {
            "mode": "beat-anchored-constrained" if structured else "nominal-tempo-fallback",
            "candidateQuarterGrids": [_rational(value[0]) for value in RHYTHM_GRIDS] if not structured else None,
            "complexityPenalties": {value[2]: value[1] for value in RHYTHM_GRIDS},
            "selectedUniqueOnsetsByGrid": rhythm_counts,
            "percussionDurationQuarter": _rational(PERCUSSION_DURATION),
        },
        "pitchFretReconciliations": reconciled,
        "unresolvedPitchFretConflicts": unresolved,
        "shortenedSustainHypotheses": shortened,
        "maximumOnsetQuantizationErrorQuarter": float(max(
            [abs(note["onset"] - note["onsetRaw"]) for note in raw_notes]
            + [abs(event["onset"] - event["onsetRaw"]) for event in raw_percussion],
            default=Fraction(0),
        )),
        "maximumDurationQuantizationErrorQuarter": float(max(
            (abs(note["durationQuantized"] - note["durationRaw"]) for note in raw_notes),
            default=Fraction(0),
        )),
        "droppedNoteHypotheses": dropped_notes,
        "droppedPercussionHypotheses": dropped_percussion,
        "fullVoices": {**full_report, "path": str(Path(full_path))},
        "singleVoice": {**single_report, "path": str(Path(single_path))},
        "linearPerformanceOrder": True,
        "trainingPerformed": False,
    }
    try:
        json.dumps(report, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise HarnessError(f"GP output report is not safely serializable: {error}") from error
    emit("writing both GP files...")
    _atomic_bytes(full_path, full)
    try:
        _atomic_bytes(single_path, single)
    except Exception:
        Path(full_path).unlink(missing_ok=True)
        raise
    emit("both GP files saved.")
    return report
