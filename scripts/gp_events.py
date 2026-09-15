"""Decode GPIF musical events in rational score time, without rewriting a GP."""

from fractions import Fraction
import math
import re

from .inspect_gp_files import GpInspectionError, number, properties


class EventExtractionError(GpInspectionError):
    """A score cannot be decoded without guessing its structure."""


NOTE_VALUES = {
    "Whole": Fraction(4), "Half": Fraction(2), "Quarter": Fraction(1),
    "Eighth": Fraction(1, 2), "16th": Fraction(1, 4), "32nd": Fraction(1, 8),
    "64th": Fraction(1, 16), "128th": Fraction(1, 32),
}
HARMONIC_OFFSETS = {Fraction(5): 24, Fraction(7): 19, Fraction(9): 28, Fraction(12): 12, Fraction(19): 19, Fraction(24): 24}
TEMPO_BEAT_UNITS = {1: Fraction(1, 8), 2: Fraction(1, 4), 3: Fraction(3, 8), 4: Fraction(1, 2), 5: Fraction(3, 4)}
NOTE_FLAGS = {"Muted": "dead", "PalmMuted": "palmMuted", "HopoOrigin": "hopoOrigin", "HopoDestination": "hopoDestination", "LeftHandTapped": "leftHandTapped", "Tapped": "tapped"}
NOTE_KNOWN = set(NOTE_FLAGS) | {"ConcertPitch", "TransposedPitch", "Fret", "String", "Midi", "Harmonic", "HarmonicFret", "HarmonicType", "Slide", "Bended", "BendDestinationOffset", "BendDestinationValue", "BendMiddleOffset1", "BendMiddleOffset2", "BendMiddleValue", "BendOriginOffset", "BendOriginValue"}
BEAT_KNOWN = {"PrimaryPickupVolume", "PrimaryPickupTone", "PickStroke", "Brush", "Slapped", "Rasgueado"}


def rational(value):
    value = Fraction(value)
    return [value.numerator, value.denominator]


def required_text(element, path):
    text = element.findtext(path)
    if text is None or not text.strip():
        raise EventExtractionError(f"Missing {path} in {element.tag}.")
    return text.strip()


def integer(text, label):
    try:
        return int(text)
    except (TypeError, ValueError) as error:
        raise EventExtractionError(f"Invalid integer for {label}: {text}") from error


def enabled(element):
    return element is not None and element.find("Enable") is not None


def boolean_attribute(element, name):
    if element is None:
        return False
    value = element.get(name, "false")
    if value not in ("true", "false"):
        raise EventExtractionError(f"Invalid {name} boolean: {value}")
    return value == "true"


def index_section(root, section, tag):
    result = {}
    for item in root.findall(f"./{section}/{tag}"):
        identifier = item.get("id")
        if identifier is None or identifier in result:
            raise EventExtractionError(f"Missing or duplicate {tag} ID.")
        result[identifier] = item
    return result


def lookup(items, reference, label):
    if reference not in items:
        raise EventExtractionError(f"Unknown {label} reference: {reference}")
    return items[reference]


def rhythm_duration(rhythm):
    value = required_text(rhythm, "NoteValue")
    if value not in NOTE_VALUES:
        raise EventExtractionError(f"Unsupported note value: {value}")
    dots = rhythm.find("AugmentationDot")
    count = integer(dots.get("count"), "dot count") if dots is not None else 0
    if not 0 <= count <= 4:
        raise EventExtractionError("Unsupported augmentation-dot count.")
    duration = NOTE_VALUES[value] * (2 - Fraction(1, 2 ** count))
    tuplet = rhythm.find("PrimaryTuplet")
    ratio = None
    if tuplet is not None:
        numerator, denominator = integer(tuplet.get("num"), "tuplet numerator"), integer(tuplet.get("den"), "tuplet denominator")
        if numerator <= 0 or denominator <= 0:
            raise EventExtractionError("Tuplet terms must be positive.")
        ratio = [numerator, denominator]
        duration *= Fraction(denominator, numerator)
    unknown = {node.tag for node in rhythm} - {"NoteValue", "AugmentationDot", "PrimaryTuplet"}
    if unknown:
        raise EventExtractionError(f"Unsupported rhythm fields: {sorted(unknown)}")
    return duration, {"value": value, "dots": count, "tuplet": ratio}


def note_event(note, identifier, tuning, capo, issues):
    props = properties(note)
    unknown = set(props) - NOTE_KNOWN
    if unknown:
        issues.append({"code": "unknown_note_properties", "eventId": identifier, "properties": sorted(unknown)})
    unknown_elements = {child.tag for child in note} - {"InstrumentArticulation", "Properties", "Tie", "Vibrato", "Accent", "AntiAccent", "XProperties"}
    if unknown_elements:
        issues.append({"code": "unknown_note_elements", "eventId": identifier, "elements": sorted(unknown_elements)})
    string_index = number(props.get("String"), "String")
    fret = number(props.get("Fret"), "Fret")
    if string_index is None or not 0 <= string_index < len(tuning) or fret is None or fret < 0:
        raise EventExtractionError(f"Invalid string/fret at {identifier}.")
    base_pitch = tuning[string_index] + capo + fret
    stored_midi = number(props.get("Midi"), "Number")
    if stored_midi is not None and stored_midi != base_pitch:
        issues.append({"code": "stored_midi_disagrees_with_fret", "eventId": identifier, "storedMidi": stored_midi, "derivedMidi": base_pitch})
    techniques = {target: enabled(props.get(source)) for source, target in NOTE_FLAGS.items()}
    slide = number(props.get("Slide"), "Flags")
    if slide is not None:
        techniques["slideFlags"] = slide
    for tag in ("Vibrato", "Accent", "AntiAccent"):
        text = note.findtext(tag)
        if text is not None:
            techniques[tag[0].lower() + tag[1:]] = text
    bend = None
    if enabled(props.get("Bended")):
        bend = {}
        for key, prop in props.items():
            if key.startswith("Bend") and key != "Bended":
                value = float(required_text(prop, "Float"))
                if not math.isfinite(value):
                    raise EventExtractionError(f"Nonfinite bend value at {identifier}.")
                bend[key[4:]] = value
    harmonic = None
    sounding_pitch = base_pitch
    if enabled(props.get("Harmonic")):
        if "HarmonicType" not in props or "HarmonicFret" not in props:
            raise EventExtractionError(f"Incomplete harmonic at {identifier}.")
        kind = required_text(props["HarmonicType"], "HType")
        harmonic_fret = Fraction(required_text(props["HarmonicFret"], "HFret"))
        harmonic = {"type": kind, "fret": rational(harmonic_fret)}
        offset = HARMONIC_OFFSETS.get(harmonic_fret)
        if kind not in ("Natural", "Artificial", "Tap", "Pinch") or offset is None:
            sounding_pitch = None
            issues.append({"code": "unsupported_harmonic_pitch", "eventId": identifier})
        elif kind == "Natural":
            sounding_pitch = tuning[string_index] + capo + offset
        else:
            sounding_pitch += offset
    if techniques["dead"]:
        sounding_pitch = None
    if sounding_pitch is not None and not 0 <= sounding_pitch <= 127:
        raise EventExtractionError(f"Pitch is outside MIDI range at {identifier}.")
    tie = note.find("Tie")
    return {
        "id": identifier, "sourceNoteId": note.get("id"),
        "string": 6 - string_index, "gpStringIndex": string_index, "fret": fret,
        "storedMidi": stored_midi, "basePitchMidi": base_pitch, "soundingPitchMidi": sounding_pitch,
        "harmonic": harmonic, "bend": bend, "techniques": techniques,
        "tie": {"origin": boolean_attribute(tie, "origin"), "destination": boolean_attribute(tie, "destination")},
        "instrumentArticulation": note.findtext("InstrumentArticulation"),
    }


def beat_techniques(beat, identifier, issues):
    props = properties(beat)
    unknown = set(props) - BEAT_KNOWN
    if unknown:
        issues.append({"code": "unknown_beat_properties", "eventId": identifier, "properties": sorted(unknown)})
    known_elements = {"Dynamic", "Rhythm", "Notes", "Properties", "GraceNotes", "Arpeggio", "FreeText", "Chord", "Hairpin", "XProperties", "TransposedPitchStemOrientation", "ConcertPitchStemOrientation", "UserTransposedPitchStemOrientation", "TransposedPitchStemOrientationUserDefined"}
    unknown_elements = {child.tag for child in beat} - known_elements
    if unknown_elements:
        issues.append({"code": "unknown_beat_elements", "eventId": identifier, "elements": sorted(unknown_elements)})
    result = {}
    for prop_name, child in (("Brush", "Direction"), ("PickStroke", "Direction"), ("Rasgueado", "Rasgueado")):
        if prop_name in props:
            result[prop_name[0].lower() + prop_name[1:]] = required_text(props[prop_name], child)
    if enabled(props.get("Slapped")):
        result["slapped"] = True
    for tag in ("Arpeggio", "Hairpin"):
        text = beat.findtext(tag)
        if text is not None:
            result[tag[0].lower() + tag[1:]] = text
    return result


def tempo_events(root, measures):
    result = []
    for automation in root.findall("./MasterTrack/Automations/Automation"):
        if automation.findtext("Type") != "Tempo":
            continue
        bar = integer(required_text(automation, "Bar"), "tempo measure")
        if not 0 <= bar < len(measures):
            raise EventExtractionError("Tempo refers to an absent measure.")
        position = Fraction(required_text(automation, "Position"))
        values = required_text(automation, "Value").split()
        if len(values) != 2 or not 0 <= position <= 1:
            raise EventExtractionError("Invalid tempo value or position.")
        tempo, reference = Fraction(values[0]), integer(values[1], "tempo unit")
        if tempo <= 0 or reference not in TEMPO_BEAT_UNITS:
            raise EventExtractionError("Unsupported tempo reference or nonpositive tempo.")
        linear = required_text(automation, "Linear")
        if linear not in ("true", "false"):
            raise EventExtractionError("Invalid tempo interpolation.")
        result.append({
            "measureIndex": bar, "positionRatio": rational(position),
            "offsetQuarter": rational(position * Fraction(*measures[bar]["durationQuarter"])),
            "quarterBpm": rational(tempo * TEMPO_BEAT_UNITS[reference] * 4),
            "bpm": int(tempo) if tempo.denominator == 1 else float(tempo),
            "beatUnit": rational(TEMPO_BEAT_UNITS[reference]),
            "gpReference": reference, "linear": linear == "true",
        })
    if not result:
        raise EventExtractionError("No explicit tempo metadata.")
    return result


def validate_provided_timing(tempo, time_signature):
    if not isinstance(tempo, dict) or "bpm" not in tempo or "beatUnit" not in tempo:
        raise EventExtractionError("BPM and its beat unit are required inputs.")
    bpm, unit = tempo["bpm"], tempo["beatUnit"]
    if isinstance(bpm, bool) or not isinstance(bpm, (int, float)) or not math.isfinite(bpm) or bpm <= 0:
        raise EventExtractionError("BPM must be a finite positive number.")
    if not isinstance(unit, list) or len(unit) != 2 or any(type(value) is not int or value <= 0 for value in unit):
        raise EventExtractionError("Tempo beat unit must be a positive [numerator, denominator] pair.")
    if not isinstance(time_signature, list) or len(time_signature) != 2 or any(type(value) is not int or value <= 0 for value in time_signature):
        raise EventExtractionError("Time signature is a required positive [numerator, denominator] pair.")
    denominator = time_signature[1]
    if denominator & (denominator - 1):
        raise EventExtractionError("Unsupported time-signature denominator; do not silently normalize it.")
    return Fraction(str(bpm)) * Fraction(*unit) * 4


def catalog_timing(decoded):
    musical_measures = [measure for measure in decoded["measures"] if not measure["referenceOnly"]]
    if not musical_measures:
        raise EventExtractionError("No performed measures to establish timing inputs.")
    first = musical_measures[0]["index"]
    tempo_events = sorted(decoded["tempoEvents"], key=lambda event: (event["measureIndex"], Fraction(*event["positionRatio"])))
    initial_candidates = [event for event in tempo_events if (event["measureIndex"], Fraction(*event["positionRatio"])) <= (first, Fraction(0))]
    if not initial_candidates:
        raise EventExtractionError("No explicit tempo at the start of the performance.")
    initial = initial_candidates[-1]
    tempo = {"bpm": initial["bpm"], "beatUnit": initial["beatUnit"]}
    signature = musical_measures[0]["timeSignature"]
    validate_provided_timing(tempo, signature)
    changes = []
    initial_position = (initial["measureIndex"], Fraction(*initial["positionRatio"]))
    for event in tempo_events:
        position = (event["measureIndex"], Fraction(*event["positionRatio"]))
        if position < initial_position or decoded["measures"][event["measureIndex"]]["referenceOnly"]:
            continue
        if event == initial and not event["linear"]:
            continue
        changes.append({key: event[key] for key in ("measureIndex", "positionRatio", "bpm", "beatUnit", "linear")})
    meter_changes = []
    previous = signature
    for measure in musical_measures[1:]:
        if measure["timeSignature"] != previous:
            meter_changes.append({"measureIndex": measure["index"], "timeSignature": measure["timeSignature"]})
            previous = measure["timeSignature"]
    return {"tempo": tempo, "timeSignature": signature, "sourceTempoChanges": changes, "sourceTimeSignatureChanges": meter_changes}


def decode_score(root, tuning, capo):
    issues = []
    definitions = {section: index_section(root, section, tag) for section, tag in (("Bars", "Bar"), ("Voices", "Voice"), ("Beats", "Beat"), ("Notes", "Note"), ("Rhythms", "Rhythm"))}
    measures, events = [], []
    score_position = Fraction(0)
    reference_only = False
    performed_measure_seen = False
    for measure_index, master in enumerate(root.findall("./MasterBars/MasterBar")):
        section = master.findtext("./Section/Text")
        if section and section.strip():
            reference_only = re.search(r"\b(instructions?|intructions|legend)\b", section, re.IGNORECASE) is not None
        unknown_elements = {child.tag for child in master} - {"Key", "Time", "Fermatas", "Bars", "Repeat", "AlternateEndings", "Section", "XProperties", "Directions", "TripletFeel", "DoubleBar"}
        if unknown_elements:
            issues.append({"code": "unknown_measure_elements", "measureIndex": measure_index, "elements": sorted(unknown_elements)})
        meter = required_text(master, "Time").split("/")
        if len(meter) != 2:
            raise EventExtractionError("Invalid time signature.")
        numerator, denominator = [integer(term, "meter") for term in meter]
        if numerator <= 0 or denominator <= 0:
            raise EventExtractionError("Nonpositive time signature.")
        nominal_duration = Fraction(numerator * 4, denominator)
        references = required_text(master, "Bars").split()
        if len(references) != 1:
            raise EventExtractionError("Only a single track/staff is supported.")
        bar = lookup(definitions["Bars"], references[0], "bar")
        voice_lengths = []
        first_event = len(events)
        for voice_index, voice_id in enumerate(required_text(bar, "Voices").split()):
            if voice_id == "-1":
                continue
            voice = lookup(definitions["Voices"], voice_id, "voice")
            position = Fraction(0)
            for beat_index, beat_id in enumerate(voice.findtext("Beats", "").split()):
                beat = lookup(definitions["Beats"], beat_id, "beat")
                rhythm_ref = beat.find("Rhythm")
                if rhythm_ref is None:
                    raise EventExtractionError(f"Beat {beat_id} has no rhythm reference.")
                rhythm_id = rhythm_ref.get("ref")
                duration, notation = rhythm_duration(lookup(definitions["Rhythms"], rhythm_id, "rhythm"))
                grace = beat.findtext("GraceNotes")
                if grace not in (None, "OnBeat", "BeforeBeat"):
                    raise EventExtractionError(f"Unsupported grace-note mode: {grace}")
                identifier = f"m{measure_index}:v{voice_index}:b{beat_index}"
                notes = [note_event(lookup(definitions["Notes"], note_id, "note"), f"{identifier}:n{note_index}", tuning, capo, issues) for note_index, note_id in enumerate(beat.findtext("Notes", "").split())]
                events.append({
                    "id": identifier, "measureIndex": measure_index, "voiceIndex": voice_index, "beatIndex": beat_index,
                    "sourceBarId": bar.get("id"), "sourceVoiceId": voice_id, "sourceBeatId": beat_id, "sourceRhythmId": rhythm_id,
                    "offsetQuarter": rational(position), "scoreOnsetQuarter": rational(score_position + position),
                    "notatedDurationQuarter": rational(duration), "advanceQuarter": rational(0 if grace else duration),
                    "rhythm": notation, "graceMode": grace, "isRest": not notes,
                    "referenceOnly": reference_only,
                    "dynamic": beat.findtext("Dynamic"), "techniques": beat_techniques(beat, identifier, issues),
                    "text": beat.findtext("FreeText"), "chordRef": beat.findtext("Chord"), "notes": notes,
                })
                if grace is None:
                    position += duration
            voice_lengths.append(position)
        actual = max(voice_lengths, default=Fraction(0))
        pickup = not performed_measure_seen and not reference_only and 0 < actual < nominal_duration
        duration = actual if pickup else max(nominal_duration, actual)
        if pickup:
            issues.append({"code": "pickup_inferred_from_short_first_measure", "measureIndex": measure_index})
        elif not reference_only and actual != nominal_duration:
            issues.append({"code": "underfull_measure" if actual < nominal_duration else "overfull_measure", "measureIndex": measure_index, "writtenDurationQuarter": rational(actual), "meterDurationQuarter": rational(nominal_duration)})
        repeat = master.find("Repeat")
        repeat_end = integer(repeat.get("count"), "repeat count") if repeat is not None and boolean_attribute(repeat, "end") else 0
        if repeat_end and not 2 <= repeat_end <= 100:
            raise EventExtractionError("Unsupported repeat count.")
        endings = [integer(value, "alternate ending") for value in master.findtext("AlternateEndings", "").split()]
        if any(value < 1 or value > 100 for value in endings):
            raise EventExtractionError("Invalid alternate ending.")
        fermatas = []
        for fermata in master.findall("./Fermatas/Fermata"):
            fermatas.append({"type": fermata.findtext("Type"), "offset": required_text(fermata, "Offset"), "length": required_text(fermata, "Length")})
        measures.append({
            "index": measure_index, "sourceBarId": bar.get("id"), "timeSignature": [numerator, denominator],
            "scoreStartQuarter": rational(score_position), "nominalDurationQuarter": rational(nominal_duration),
            "durationQuarter": rational(duration), "inferredPickup": pickup,
            "referenceOnly": reference_only,
            "voiceDurationsQuarter": [rational(value) for value in voice_lengths],
            "eventSlice": [first_event, len(events)],
            "repeatStart": boolean_attribute(repeat, "start"), "repeatEndCount": repeat_end, "alternateEndings": endings,
            "directions": [{"kind": node.tag, "value": node.text} for node in master.findall("./Directions/*")],
            "tripletFeel": master.findtext("TripletFeel"), "fermatas": fermatas,
            "key": {"accidentalCount": master.findtext("./Key/AccidentalCount"), "mode": master.findtext("./Key/Mode")},
            "section": section,
        })
        performed_measure_seen = performed_measure_seen or not reference_only
        score_position += duration
    if not measures:
        raise EventExtractionError("No musical measures.")
    return {"measures": measures, "scoreEvents": events, "tempoEvents": tempo_events(root, measures), "issues": issues}


def playback_order(measures, fine_measure_index=None):
    """Expand flat repeats and common D.C./D.S./Coda navigation."""
    if fine_measure_index is not None:
        if type(fine_measure_index) is not int or not 0 <= fine_measure_index < len(measures) or measures[fine_measure_index].get("referenceOnly", False):
            raise EventExtractionError("Confirmed Fine must identify a musical measure.")
        if any(direction["kind"] == "Target" and direction["value"] == "Fine" for measure in measures for direction in measure["directions"]):
            raise EventExtractionError("A confirmed Fine override cannot replace an explicit Fine target.")
    for measure in measures:
        for direction in measure["directions"]:
            if direction["kind"] not in ("Jump", "Target"):
                raise EventExtractionError("Unknown navigation element.")
            if direction["kind"] == "Target" and direction["value"] not in {"Segno", "SegnoSegno", "Coda", "DoubleCoda", "Fine"}:
                raise EventExtractionError(f"Unknown direction target: {direction['value']}")
    index, anchor, iteration = 0, 0, 1
    ending = []
    active = False
    following_ending = False
    return_mode = None
    used_jumps = set()
    visits = []
    steps = 0
    while index < len(measures):
        steps += 1
        if steps > max(1000, len(measures) * 200):
            raise EventExtractionError("Playback navigation exceeded its safety bound.")
        bar = measures[index]
        if bar.get("referenceOnly", False):
            index += 1
            continue
        navigation_playback = return_mode is not None
        if not navigation_playback:
            if bar["repeatStart"] and not (active and index == anchor):
                if active:
                    raise EventExtractionError("Nested or overlapping repeat groups need explicit handling.")
                anchor, iteration, ending, active, following_ending = index, 1, [], True, False
            if bar["alternateEndings"]:
                ending = bar["alternateEndings"]
            elif following_ending:
                ending, following_ending = [], False
            play = not ending or iteration in ending
        else:
            play = True
        if play:
            visits.append(index)
        directions = bar["directions"]
        jumps = [item["value"] for item in directions if item["kind"] == "Jump"]
        targets = [item["value"] for item in directions if item["kind"] == "Target"]
        commands = [jump for jump in jumps if jump not in ("DaCoda", "DaDoubleCoda") and (index, jump) not in used_jumps]
        if len(commands) > 1:
            raise EventExtractionError("Multiple competing direction jumps on one bar.")
        if return_mode == "Fine" and ("Fine" in targets or index == fine_measure_index):
            return_mode = None
            break
        if return_mode in ("Coda", "DoubleCoda") and "Da" + return_mode in jumps:
            matches = [position for position, item in enumerate(measures) if not item.get("referenceOnly", False) and any(d["kind"] == "Target" and d["value"] == return_mode for d in item["directions"])]
            later = [position for position in matches if position > index]
            if not later:
                raise EventExtractionError(f"No forward {return_mode} target.")
            index, return_mode, active, ending = later[0], None, False, []
            continue
        if return_mode in (None, "End") and play and commands:
            jump = commands[0]
            marker = (index, jump)
            match = re.fullmatch(r"(DaCapo|DaSegno|DaSegnoSegno)(?:Al(Coda|DoubleCoda|Fine))?", jump)
            if not match:
                raise EventExtractionError(f"Unsupported navigation: {jump}")
            used_jumps.add(marker)
            kind, suffix = match.groups()
            if kind == "DaCapo":
                destination = 0
            else:
                target = "SegnoSegno" if kind == "DaSegnoSegno" else "Segno"
                matches = [position for position, item in enumerate(measures[:index + 1]) if not item.get("referenceOnly", False) and any(d["kind"] == "Target" and d["value"] == target for d in item["directions"])]
                if not matches:
                    raise EventExtractionError(f"No preceding {target} target.")
                destination = matches[-1]
            index, return_mode, active, ending = destination, suffix or "End", False, []
            continue
        if navigation_playback:
            index += 1
            continue
        if bar["repeatEndCount"]:
            count = bar["repeatEndCount"]
            active = True
            if play and iteration < count:
                iteration += 1
                index, ending = anchor, []
                continue
            if iteration >= count:
                active, following_ending = False, True
        if bar["repeatEndCount"] and not active and not ending:
            anchor = index + 1
        index += 1
    if return_mode in ("Coda", "DoubleCoda", "Fine"):
        raise EventExtractionError(f"Unresolved navigation destination: {return_mode}")
    if active:
        raise EventExtractionError("Unclosed repeat group.")
    return visits


def performance_events(decoded, order):
    visits, notes = [], []
    position = Fraction(0)
    previous = {}
    for visit_index, measure_index in enumerate(order):
        measure = decoded["measures"][measure_index]
        visits.append({"visitIndex": visit_index, "measureIndex": measure_index, "onsetQuarter": rational(position)})
        first, stop = measure["eventSlice"]
        for beat in decoded["scoreEvents"][first:stop]:
            onset = position + Fraction(*beat["offsetQuarter"])
            duration = Fraction(*beat["notatedDurationQuarter"])
            for note in beat["notes"]:
                identifier = f"p{visit_index}:{note['id']}"
                prior = previous.get((beat["voiceIndex"], note["string"]))
                tie_origin = None
                attack = not note["tie"]["destination"]
                if note["tie"]["destination"]:
                    if prior and prior["isOrigin"] and prior["end"] == onset and prior["pitch"] == note["basePitchMidi"]:
                        tie_origin = prior["id"]
                    else:
                        attack = None
                        decoded["issues"].append({"code": "unresolved_tie_destination", "eventId": identifier})
                notes.append({
                    "id": identifier, "sourceEventId": note["id"], "visitIndex": visit_index,
                    "onsetQuarter": rational(onset), "durationQuarter": None if beat["graceMode"] else rational(duration),
                    "isAttack": attack, "tieFrom": tie_origin,
                })
                previous[(beat["voiceIndex"], note["string"])] = {
                    "id": identifier, "isOrigin": note["tie"]["origin"],
                    "end": onset if beat["graceMode"] else onset + duration, "pitch": note["basePitchMidi"],
                }
        position += Fraction(*measure["durationQuarter"])
    notes.sort(key=lambda event: (Fraction(*event["onsetQuarter"]), event["visitIndex"]))
    return {"policy": "Repeats/voltas on first pass; D.C./D.S. return ignores repeats and voltas until its Coda/Fine/end; repeats resume at Coda.", "measureVisits": visits, "noteEvents": notes, "durationQuarter": rational(position)}
