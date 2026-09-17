"""Create deterministic full-voice and single-voice GP files from hypotheses."""

from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from .gp_events import HARMONIC_OFFSETS, NOTE_VALUES, TEMPO_BEAT_UNITS, decode_score
from .draft_cleanup import DraftProfile, clean_hypotheses
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


GRID = Fraction(1, 8)
RHYTHM_GRIDS = ((Fraction(1, 4), 0.0, "sixteenth"), (Fraction(1, 8), 0.02, "thirty-second"))
PERCUSSION_DURATION = Fraction(1, 4)
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
    for index, value in enumerate(document.get("notes", [])):
        required = {"onsetSeconds", "string", "fret", "soundingPitchMidi", "voiceIndex", "notatedDurationQuarter", "harmonic", "confidence"}
        if not isinstance(value, dict) or not required <= set(value):
            raise HarnessError("Every predicted note requires time, string, fret, pitch, voice, duration, harmonic, and confidence fields.")
        harmonic = value.get("harmonic")
        if harmonic is not None:
            if not isinstance(harmonic, dict) or harmonic.get("type") not in HARMONIC_TYPES or harmonic.get("fret") not in HARMONIC_FRETS:
                raise HarnessError("Predicted harmonics require a supported type and node.")
        note = {
            "id": f"note-{index}",
            "onsetRaw": tempo.quarter_at(value["onsetSeconds"]),
            "durationRaw": _fraction(_finite(value["notatedDurationQuarter"], "Note duration", minimum=0)),
            "string": _integer(value["string"], "Note string", 1, 6),
            "fret": _integer(value["fret"], "Note fret", 0, MAX_FRET),
            "soundingPitchMidi": _integer(value["soundingPitchMidi"], "Sounding pitch", 0, 127),
            "voice": _integer(value["voiceIndex"], "Voice index", 0, 3),
            "harmonic": deepcopy(harmonic),
            "confidence": _finite(value["confidence"], "Note confidence", minimum=0),
            "uncertainty": list(value.get("uncertainty", [])),
        }
        if note["confidence"] > 1:
            raise HarnessError("Note confidence must be from zero through one.")
        notes.append(note)
    percussion = []
    for index, value in enumerate(document.get("percussion", [])):
        required = {"onsetSeconds", "technique", "confidence"}
        if not isinstance(value, dict) or not required <= set(value) or value.get("technique") not in PERCUSSION_TYPES:
            raise HarnessError("Every percussion prediction requires a supported technique.")
        event = {
            "id": f"percussion-{index}",
            "technique": value["technique"],
            "onsetRaw": tempo.quarter_at(value["onsetSeconds"]),
            "confidence": _finite(value["confidence"], "Percussion confidence", minimum=0),
        }
        if event["confidence"] > 1:
            raise HarnessError("Percussion confidence must be from zero through one.")
        percussion.append(event)
    if not notes and not percussion:
        raise HarnessError("GP output requires at least one decoded note or percussion event.")
    if any(note["onsetRaw"] > tempo.quarter_at(duration_seconds) for note in notes) or any(event["onsetRaw"] > tempo.quarter_at(duration_seconds) for event in percussion):
        raise HarnessError("A prediction falls after the declared audio duration.")
    positions, rhythm_counts = _rhythmic_positions([note["onsetRaw"] for note in notes] + [event["onsetRaw"] for event in percussion])
    for note in notes:
        note["onset"], note["rhythmGrid"] = positions[note["onsetRaw"]]
        note["duration"] = _quantize(note["durationRaw"], note["rhythmGrid"])
        note["durationQuantized"] = note["duration"]
        note["duration"] = max(note["rhythmGrid"], note["duration"])
        note["end"] = note["onset"] + note["duration"]
    for event in percussion:
        event["onset"], event["rhythmGrid"] = positions[event["onsetRaw"]]
    return tuning, capo, tempo, tempo.quarter_at(duration_seconds), notes, percussion, rhythm_counts


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
            bounded = max(note["rhythmGrid"], available // note["rhythmGrid"] * note["rhythmGrid"])
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


def _meter_changes(metadata, tempo):
    changes = [(Fraction(0), metadata["timeSignature"])]
    for event in metadata.get("timeSignatureChanges", []):
        raw = tempo.quarter_at(event["timeSeconds"])
        quantized = _quantize(raw)
        changes.append((quantized, event["timeSignature"]))
    return changes


def _measures(metadata, tempo, content_end):
    changes = _meter_changes(metadata, tempo)
    result = []
    cursor = Fraction(0)
    for index, (change, meter) in enumerate(changes):
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

    result, remaining = decompose([item for item in RHYTHM_PALETTE if item[1][2] is None])
    if not remaining:
        return result
    result, remaining = decompose(RHYTHM_PALETTE)
    if remaining:
        result.append((remaining, ("Quarter", 0, (remaining.denominator, remaining.numerator))))
    return result


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
    def __init__(self, root, tuning, capo, measures, notes, percussion, key_count, spellings, *, simplified):
        self.root = root
        self.tuning = tuning
        self.capo = capo
        self.measures = measures
        self.notes = notes
        self.percussion = percussion
        self.key_count = key_count
        self.spellings = spellings
        self.simplified = simplified
        self.pitch_offsets = _prototype_pitch_offsets(root)
        self.pitch_profile = {"offsets": self.pitch_offsets, "spellings": {}}
        self.ids = defaultdict(int)
        self.rhythms = {}
        self.fallbacks = []
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
        if note["harmonic"] is not None:
            ET.SubElement(ET.SubElement(props, "Property", name="Harmonic"), "Enable")
            ET.SubElement(ET.SubElement(props, "Property", name="HarmonicFret"), "HFret").text = f"{note['harmonic']['fret']:.6f}"
            ET.SubElement(ET.SubElement(props, "Property", name="HarmonicType"), "HType").text = note["harmonic"]["type"]
        self.root.find("Notes").append(node)
        return identifier

    def percussion_note(self, event, string):
        identifier = self.identifier("Notes")
        node = ghost_dead_note(identifier, string, self.tuning, self.capo, pitch_profile=self.pitch_profile)
        if event["technique"] == "thumb_slap":
            anti = node.find("AntiAccent")
            if anti is not None:
                node.remove(anti)
        self.root.find("Notes").append(node)
        return identifier

    def beat(self, notation, active, carriers, text, voice, start, stop):
        identifier = self.identifier("Beats")
        beat = ET.Element("Beat", id=identifier)
        ET.SubElement(beat, "Dynamic").text = "MF"
        ET.SubElement(beat, "Rhythm", ref=self.rhythm(notation))
        ET.SubElement(beat, "TransposedPitchStemOrientation").text = "Upward" if voice == 0 else "Downward"
        ET.SubElement(beat, "ConcertPitchStemOrientation").text = "Undefined"
        if text:
            ET.SubElement(beat, "FreeText").text = text
        note_ids = [self.pitched_note(note, start, stop) for note in sorted(active, key=lambda item: -item["string"])]
        note_ids.extend(self.percussion_note(event, string) for event, string in carriers)
        if note_ids:
            ET.SubElement(beat, "Notes").text = " ".join(note_ids)
        self.root.find("Beats").append(beat)
        return identifier

    def percussion_at(self, onset):
        return [event for event in self.percussion if event["onset"] == onset]

    def active_strings(self, onset):
        return {note["string"] for note in self.notes if note["onset"] <= onset < note["end"]}

    def percussion_content(self, onset):
        events = self.percussion_at(onset)
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

    def voice_beats(self, measure, voice):
        notes = [note for note in self.notes if note["voice"] == voice and note["onset"] < measure["end"] and note["end"] > measure["start"]]
        boundaries = {measure["start"], measure["end"]}
        for note in notes:
            boundaries.update((max(measure["start"], note["onset"]), min(measure["end"], note["end"])))
        if voice == 0:
            for event in self.percussion:
                if measure["start"] <= event["onset"] < measure["end"]:
                    boundaries.add(event["onset"])
                    boundaries.add(min(measure["end"], event["onset"] + PERCUSSION_DURATION))
        points = sorted(boundaries)
        beats = []
        for left, right in zip(points, points[1:]):
            active = [note for note in notes if note["onset"] <= left < note["end"]]
            carriers, text = self.percussion_content(left) if voice == 0 else ([], "")
            cursor = left
            for duration, notation in _split_duration(right - left):
                stop = cursor + duration
                local_carriers = carriers if cursor == left else []
                local_text = text if cursor == left else ""
                beats.append(self.beat(notation, active, local_carriers, local_text, voice, cursor, stop))
                cursor = stop
        return beats

    def build(self):
        master_parent = self.root.find("MasterBars")
        bar_parent = self.root.find("Bars")
        voices_used = {note["voice"] for note in self.notes}
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


def _tempo_automations(root, tempo, measures):
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
    for event, quarter in zip(tempo.events, tempo.quarters):
        if event["beatUnit"] not in reverse_units:
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
            ("Value", f"{float(event['bpm']):.12g} {reverse_units[event['beatUnit']]}"),
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


def _build_variant(template_raw, notes, percussion, measures, tuning, capo, tempo, *, simplified):
    root, payloads, comment = _parse_archive(template_raw)
    before = _template_invariant(root)
    offsets = _prototype_pitch_offsets(root)
    preference = root.findtext("./MasterBars/MasterBar/Key/TransposeAs", "Sharps")
    if preference not in ("Sharps", "Flats"):
        preference = "Sharps"
    key_count, spellings, accidental_count = _select_key(notes, measures, preference, offsets["TransposedPitch"])
    writer = _ScoreWriter(root, tuning, capo, measures, notes, percussion, key_count, spellings, simplified=simplified)
    fallbacks = writer.build()
    _tempo_automations(root, tempo, measures)
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
        "keyAccidentalCount": key_count,
        "displayedNoteAccidentalCount": accidental_count,
        "measureCount": len(measures),
        "noteSegmentCount": len(parsed.findall("./Notes/Note")),
        "voiceCount": 1 if simplified else max((note["voice"] for note in notes), default=0) + 1,
        "percussionTextFallbacks": fallbacks,
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


def write_gp_outputs(template_path, predictions, full_path, single_path, *, profile=DraftProfile()):
    template_path = Path(template_path).resolve()
    if not template_path.is_file():
        raise HarnessError(f"GP output template does not exist: {template_path}")
    template_raw = template_path.read_bytes()
    template_hash = hashlib.sha256(template_raw).hexdigest()
    if Path(full_path).absolute() == Path(single_path).absolute():
        raise HarnessError("Full-voice and single-voice GP outputs require different paths.")
    cleaned, cleanup = clean_hypotheses(predictions, profile)
    tuning, capo, tempo, audio_end, raw_notes, raw_percussion, rhythm_counts = _validate_predictions(cleaned)
    notes, reconciled, unresolved, shortened, dropped_notes = _resolve_notes(raw_notes, tuning, capo, audio_end)
    percussion, dropped_percussion = _resolve_percussion(raw_percussion)
    content_end = max(audio_end, max([note["onset"] + note["rhythmGrid"] for note in notes] + [event["onset"] + PERCUSSION_DURATION for event in percussion], default=GRID))
    measures = _measures(predictions["metadata"], tempo, content_end)
    full, full_report = _build_variant(template_raw, notes, percussion, measures, tuning, capo, tempo, simplified=False)
    simple_notes = _simplified_notes(notes, percussion)
    single, single_report = _build_variant(template_raw, simple_notes, percussion, measures, tuning, capo, tempo, simplified=True)
    if hashlib.sha256(template_path.read_bytes()).hexdigest() != template_hash:
        raise HarnessError("GP template changed while outputs were generated.")
    _atomic_bytes(full_path, full)
    try:
        _atomic_bytes(single_path, single)
    except Exception:
        Path(full_path).unlink(missing_ok=True)
        raise
    return {
        "schemaVersion": 1,
        "kind": "gp-output-report",
        "templateSha256": template_hash,
        "templateModified": False,
        "finestCandidateQuarterGrid": _rational(GRID),
        "sourceHypotheses": {"notes": len(predictions["notes"]), "percussion": len(predictions["percussion"])},
        "cleanedHypotheses": {"notes": len(raw_notes), "percussion": len(raw_percussion)},
        "resolvedHypotheses": {"notes": len(notes), "percussion": len(percussion)},
        "draftCleanup": cleanup,
        "rhythmicGridPolicy": {
            "candidateQuarterGrids": [_rational(value[0]) for value in RHYTHM_GRIDS],
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
