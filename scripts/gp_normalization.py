"""Lossless-template, score-time normalization of reviewed GP percussion."""

from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
from io import BytesIO
import xml.etree.ElementTree as ET
from xml.dom import Node, minidom
from zipfile import BadZipFile, ZipFile

from .canonical_events import canonical_counts, canonicalize, fraction, normalized_text, rule_tokens, source_indices
from .gp_events import NOTE_VALUES, beat_techniques, catalog_timing, decode_score, index_section, performance_events, rational


GPIF_ENTRY = "Content/score.gpif"
NORMALIZATION_VERSION = 2
GENERIC_TECHNIQUES = frozenset({
    "body_tap", "body_flam", "body_scratch", "body_slap", "body_rasgueado",
    "string_slap", "nail_attack", "slap_pluck", "finger_snap", "snare_tap",
    "string_tap", "string_scrape",
})
PLACEMENT_LIMITATION = (
    "Strings are free only under full notated sustain intervals [start, end). "
    "A notated end is not evidence of acoustic decay. Unresolved/grace durations "
    "and retained note symbols reserve their strings conservatively; generic strings are "
    "notation carriers, not fingering labels."
)


class NormalizationError(ValueError):
    """Normalization cannot preserve the source contract without guessing."""

    def __init__(self, message, *, conflicts=None):
        super().__init__(message)
        self.conflicts = conflicts or []


def semantic_node(node):
    """Ignore XML indentation, but retain unknown elements, attributes and text."""
    if node is None:
        return None
    text = lambda value: value if value and value.strip() else None
    return (
        node.tag, tuple(sorted(node.attrib.items())), text(node.text), text(node.tail),
        tuple(semantic_node(child) for child in node),
    )


def _pitch_value(prop):
    pitch = prop.find("Pitch")
    if pitch is None:
        raise NormalizationError("A native pitch property has no Pitch element.")
    step = pitch.findtext("Step")
    accidental = pitch.findtext("Accidental") or ""
    offsets = {"": 0, "Natural": 0, "#": 1, "b": -1, "x": 2, "##": 2, "bb": -2}
    if not isinstance(step, str) or len(step) != 1 or step not in "CDEFGAB" or accidental not in offsets:
        raise NormalizationError("Unsupported native GP pitch spelling.")
    return int(pitch.findtext("Octave")) * 12 + {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[step] + offsets[accidental]


def native_pitch_profile(root):
    offsets = {name: set() for name in ("ConcertPitch", "TransposedPitch")}
    spellings = {}
    for note in root.findall("./Notes/Note"):
        props = {prop.get("name"): prop for prop in note.findall("./Properties/Property")}
        if "Midi" not in props:
            continue
        midi = int(props["Midi"].findtext("Number"))
        if not all(name in props for name in offsets):
            raise NormalizationError("Native GP normalization requires complete source concert/transposed pitch properties.")
        for name in offsets:
            offsets[name].add(_pitch_value(props[name]) - midi)
        spellings.setdefault(midi, {name: deepcopy(props[name]) for name in offsets})
    if any(len(values) != 1 for values in offsets.values()):
        raise NormalizationError("Source pitch display offsets are missing or inconsistent; no transposition default is inferred.")
    return {"offsets": {name: next(iter(values)) for name, values in offsets.items()}, "spellings": spellings}


def _pitch_property(name, midi):
    step, accidental = (("C", ""), ("C", "#"), ("D", ""), ("D", "#"), ("E", ""), ("F", ""), ("F", "#"), ("G", ""), ("G", "#"), ("A", ""), ("A", "#"), ("B", ""))[midi % 12]
    prop = ET.Element("Property", name=name)
    pitch = ET.SubElement(prop, "Pitch")
    for tag, text in (("Step", step), ("Accidental", accidental), ("Octave", str(midi // 12))):
        ET.SubElement(pitch, tag).text = text
    return prop


def ghost_dead_note(identifier, string, tuning, capo, *, pitch_profile):
    """Create the complete native GP ghost/dead-note representation."""
    if type(string) is not int or not 1 <= string <= 6 or len(tuning) != 6:
        raise NormalizationError("A ghost carrier requires one of six physical strings.")
    midi = tuning[6 - string] + capo
    if not 0 <= midi <= 127:
        raise NormalizationError("Ghost carrier MIDI is outside the GP note range.")
    note = ET.Element("Note", id=str(identifier))
    ET.SubElement(note, "AntiAccent").text = "Normal"
    ET.SubElement(note, "InstrumentArticulation").text = "0"
    props = ET.SubElement(note, "Properties")
    spellings = pitch_profile["spellings"].get(midi)
    concert = deepcopy(spellings["ConcertPitch"]) if spellings else _pitch_property("ConcertPitch", midi + pitch_profile["offsets"]["ConcertPitch"])
    transposed = deepcopy(spellings["TransposedPitch"]) if spellings else _pitch_property("TransposedPitch", midi + pitch_profile["offsets"]["TransposedPitch"])
    props.append(concert)
    for name, tag, value in (("Fret", "Fret", 0), ("Midi", "Number", midi)):
        ET.SubElement(ET.SubElement(props, "Property", name=name), tag).text = str(value)
    ET.SubElement(ET.SubElement(props, "Property", name="Muted"), "Enable")
    ET.SubElement(ET.SubElement(props, "Property", name="String"), "String").text = str(6 - string)
    props.append(transposed)
    return note


def select_free_string(notes, onset, *, reserved=()):
    """Choose bass first; unknown duration reserves a string for the whole score."""
    onset = fraction(onset, "hit onset") if isinstance(onset, list) else Fraction(onset)
    occupied = set(reserved)
    for note in notes:
        duration = note.get("notatedDurationQuarter")
        if duration is None or not note.get("labelMask", {}).get("notatedDuration", False):
            occupied.add(note["string"])
            continue
        start = fraction(note["onsetQuarter"], note["id"])
        if start <= onset < start + fraction(duration, note["id"]):
            occupied.add(note["string"])
    for string in (6, 5, 4, 3, 2, 1):
        if string not in occupied:
            return string
    raise NormalizationError(
        f"No free notated string at {rational(onset)}; held notes were not shortened.",
        conflicts=[{"onsetQuarter": rational(onset), "occupiedStrings": sorted(occupied)}],
    )


def _parse_archive(raw):
    try:
        archive = ZipFile(BytesIO(raw))
    except BadZipFile as error:
        raise NormalizationError("Normalization requires a modern GP ZIP template.") from error
    with archive:
        infos = archive.infolist()
        if len({item.filename for item in infos}) != len(infos):
            raise NormalizationError("Duplicate ZIP entry names are not safe to patch.")
        matches = [item for item in infos if item.filename == GPIF_ENTRY]
        if len(matches) != 1 or matches[0].file_size > 32 * 1024 * 1024:
            raise NormalizationError("Missing or oversized Content/score.gpif.")
        payloads = [(deepcopy(item), archive.read(item)) for item in infos]
        xml = next(payload for info, payload in payloads if info.filename == GPIF_ENTRY)
        comment = archive.comment
    if b"<!DOCTYPE" in xml.upper() or b"<!ENTITY" in xml.upper():
        raise NormalizationError("DTD/entity declarations are not supported.")
    try:
        root = ET.fromstring(xml, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True, insert_pis=True)))
    except ET.ParseError as error:
        raise NormalizationError(f"Malformed GPIF XML: {error}") from error
    if root.tag != "GPIF":
        raise NormalizationError("The archive does not contain a GPIF score.")
    return root, payloads, comment


def _archive_bytes(root, payloads, comment):
    original_xml = next(payload for info, payload in payloads if info.filename == GPIF_ENTRY)
    gpif = _native_gpif_bytes(root, original_xml)
    result = BytesIO()
    with ZipFile(result, "w") as archive:
        archive.comment = comment
        for info, payload in payloads:
            archive.writestr(info, gpif if info.filename == GPIF_ENTRY else payload)
    output = result.getvalue()
    with ZipFile(BytesIO(output)) as archive:
        for info, payload in payloads:
            if info.filename != GPIF_ENTRY and archive.read(info.filename) != payload:
                raise NormalizationError(f"Archive resource changed: {info.filename}")
    return output


def _native_gpif_bytes(root, original_xml):
    document = minidom.parseString(original_xml)
    cdata_paths = set()

    def discover(node, path):
        if any(child.nodeType == Node.CDATA_SECTION_NODE for child in node.childNodes):
            if any(child.nodeType == Node.ELEMENT_NODE for child in node.childNodes):
                raise NormalizationError("Mixed XML/CDATA fields require a native-aware content mapper; text is not flattened.")
            cdata_paths.add(path)
        for child in node.childNodes:
            if child.nodeType == Node.ELEMENT_NODE:
                discover(child, (*path, child.tagName))

    try:
        discover(document.documentElement, (document.documentElement.tagName,))
    finally:
        document.unlink()
    output = deepcopy(root)
    serialized = ET.tostring(output, encoding="utf-8")
    prefix = b"__GP_CDATA_"
    while prefix in serialized:
        prefix += b"_"
    replacements = {}

    def preserve(node, path):
        if path in cdata_paths:
            token = prefix + str(len(replacements)).encode("ascii") + b"__"
            value = (node.text or "").encode("utf-8").replace(b"]]>", b"]]]]><![CDATA[>")
            replacements[token] = b"<![CDATA[" + value + b"]]>"
            node.text = token.decode("ascii")
        for child in node:
            if isinstance(child.tag, str):
                preserve(child, (*path, child.tag))

    preserve(output, (output.tag,))
    result = ET.tostring(output, encoding="utf-8", xml_declaration=True)
    for token, value in replacements.items():
        result = result.replace(token, value)
    return result


def _remove(parent, tag):
    for child in list(parent):
        if child.tag == tag:
            parent.remove(child)


def _set_text(parent, tag, text):
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    child.text = str(text)


def _protected_tree(root):
    result = deepcopy(root)
    for section, tag in (("MasterBars", "MasterBar"), ("Bars", "Bar"), ("Voices", "Voice"), ("Beats", "Beat"), ("Notes", "Note"), ("Rhythms", "Rhythm")):
        parent = result.find(section)
        if parent is not None:
            _remove(parent, tag)
    automations = result.find("./MasterTrack/Automations")
    if automations is not None:
        for node in list(automations):
            if node.tag == "Automation" and node.findtext("Type") == "Tempo":
                automations.remove(node)
    return semantic_node(result)


def _decode_musical_tree(root, instrument):
    # The existing reader ignores XML comments/PIs; retain them in the template,
    # but use that same reader view for source and output event comparisons.
    musical = ET.fromstring(ET.tostring(root, encoding="utf-8"))
    return decode_score(musical, instrument["openStringMidi"], instrument["capoFret"])


def _conversion_candidates(labels, annotations):
    rules = {rule["id"]: rule for rule in (annotations or {}).get("rules", [])}
    result = []
    for gesture in labels["targets"]["gestures"]:
        technique = gesture["technique"]
        if technique == "compound_gesture":
            components = gesture.get("attributes", {}).get("components", [])
            percussive = [component["technique"] in GENERIC_TECHNIQUES for component in components]
            if not any(percussive):
                continue
            if not all(percussive) or gesture["attributes"].get("timing") != "simultaneous":
                raise NormalizationError(
                    f"{gesture['id']}: mixed/ordered percussion compound cannot be collapsed "
                    "without losing another technique or inventing subevent timing."
                )
        elif technique not in GENERIC_TECHNIQUES:
            continue
        if gesture.get("scoreOnsetKnown") is not True or gesture.get("graceMode") is not None:
            raise NormalizationError(f"{gesture['id']}: a generic hit needs a resolved regular score onset.")
        rule = rules.get(gesture["interpretationRuleId"])
        if rule is None:
            raise NormalizationError(f"{gesture['id']}: no checksum-scoped reviewed rule.")
        if "O" in rule_tokens(rule):
            raise NormalizationError(f"{gesture['id']}: converting this rule would erase the owner's wrist-thump O.")
        result.append({"gesture": gesture, "rule": rule})
    return result


def _consumed_symbols(candidates, labels):
    symbols = {
        segment["performanceId"]: symbol
        for symbol in labels["review"]["notationSymbols"] for segment in symbol["sourceSegments"]
    }
    consumed, owners = set(), {}
    for candidate in candidates:
        identifiers = set()
        for identifier in candidate["gesture"]["symbolicNoteIds"]:
            if identifier not in symbols or symbols[identifier]["id"] != identifier:
                raise NormalizationError(f"{identifier}: consumed percussion is not a logical dead-note attack.")
            identifiers.update(segment["performanceId"] for segment in symbols[identifier]["sourceSegments"])
        for identifier in identifiers:
            if identifier in owners:
                raise NormalizationError(f"{identifier}: two generic gestures consume the same symbol.")
            owners[identifier] = candidate["gesture"]["id"]
        candidate["consumedNoteIds"] = sorted(identifiers)
        consumed.update(identifiers)
    converted = {candidate["gesture"]["id"] for candidate in candidates}
    for record in [
        *(gesture for gesture in labels["targets"]["gestures"] if gesture["id"] not in converted),
        *labels["review"]["unresolvedGestures"],
    ]:
        if consumed.intersection(record.get("symbolicNoteIds", [])):
            raise NormalizationError("A converted dead symbol also carries an unrelated or unresolved meaning.")
    return consumed


def _validate_inputs(root, raw, score, labels, annotations, conventions):
    digest = hashlib.sha256(raw).hexdigest()
    if digest != score["sourceGpSha256"] or digest != labels["provenance"]["sourceGpSha256"]:
        raise NormalizationError("Raw GP, events and canonical labels have different revision hashes.")
    tracks = root.findall("./Tracks/Track")
    if len(tracks) != 1 or len(tracks[0].findall("./Staves/Staff")) != 1:
        raise NormalizationError("Normalization supports one track and one staff; it never adds a percussion track.")
    instrument = score["instrument"]
    decoded = _decode_musical_tree(root, instrument)
    for key in ("measures", "scoreEvents", "tempoEvents"):
        if decoded[key] != score[key]:
            raise NormalizationError(f"Extracted {key} do not match the immutable GP template.")
    expected = canonicalize(score, annotations, conventions)
    for key, value in expected.items():
        if key == "provenance":
            if any(labels[key].get(name) != field for name, field in value.items()):
                raise NormalizationError("Canonical source provenance is stale.")
        elif labels.get(key) != value:
            raise NormalizationError(f"Canonical source {key} is stale.")
    if not labels["scoreTimingResolved"]:
        raise NormalizationError("Unresolved source measure lengths/playback cannot be normalized safely.")
    visits = score["playback"]["measureVisits"]
    position = Fraction(0)
    for index, visit in enumerate(visits):
        if visit["visitIndex"] != index or fraction(visit["onsetQuarter"], "visit") != position:
            raise NormalizationError("Playback visits are not a contiguous chronological sequence.")
        position += fraction(score["measures"][visit["measureIndex"]]["durationQuarter"], "measure duration")
    if not visits or rational(position) != score["playback"]["durationQuarter"]:
        raise NormalizationError("Playback visit durations disagree with the score length.")
    # Static sound/DSP settings remain byte-semantically intact. Moving later automation
    # or nonempty lyrics requires a dedicated mapper, not silent index reinterpretation.
    tempo_nodes = root.findall("./MasterTrack/Automations/Automation")
    tempo_ids = {id(node) for node in tempo_nodes if node.findtext("Type") == "Tempo"}
    for node in root.findall(".//Automation"):
        if id(node) not in tempo_ids and (node.findtext("Bar") != "0" or Fraction(node.findtext("Position", "0")) != 0):
            raise NormalizationError(f"Nonstatic {node.findtext('Type')} automation needs occurrence mapping.")
    if any(normalized_text(node.text) for node in root.findall("./Tracks/Track/Lyrics/Line/Text")):
        raise NormalizationError("Nonempty lyrics need occurrence mapping before a linear copy can be made.")
    if any(event["linear"] for event in score["tempoEvents"]):
        raise NormalizationError("Tempo ramps require an interpolation-aware playback mapper; no constant-tempo fallback.")


def _linear_tempos(root, source, score):
    original = [
        node for node in source.findall("./MasterTrack/Automations/Automation")
        if node.findtext("Type") == "Tempo"
    ]
    original.sort(key=lambda node: (int(node.findtext("Bar")), Fraction(node.findtext("Position"))))
    output = root.find("./MasterTrack/Automations")
    for node in list(output):
        if node.tag == "Automation" and node.findtext("Type") == "Tempo":
            output.remove(node)
    previous = None
    for index, visit in enumerate(score["playback"]["measureVisits"]):
        measure = visit["measureIndex"]
        active = [node for node in original if (int(node.findtext("Bar")), Fraction(node.findtext("Position"))) <= (measure, Fraction(0))]
        if not active:
            raise NormalizationError(f"Visit {index} has no explicit source tempo.")
        within = [node for node in original if int(node.findtext("Bar")) == measure]
        baseline = active[-1]
        has_start = any(Fraction(node.findtext("Position")) == 0 for node in within)
        selected = []
        if not has_start and (index == 0 or previous != baseline.findtext("Value")):
            reset = deepcopy(baseline)
            _set_text(reset, "Position", "0")
            selected.append(reset)
        selected.extend(deepcopy(node) for node in within)
        for node in selected:
            _set_text(node, "Bar", index)
            output.append(node)
        previous = (within[-1] if within else baseline).findtext("Value")


class _OccurrenceWriter:
    def __init__(self, source, score, omitted):
        self.source, self.score = source, score
        self.pitch_profile = native_pitch_profile(source)
        self.root = deepcopy(source)
        self.definitions = {
            section: index_section(source, section, tag)
            for section, tag in (("Bars", "Bar"), ("Voices", "Voice"), ("Beats", "Beat"), ("Notes", "Note"), ("Rhythms", "Rhythm"))
        }
        self.next_ids = {}
        for section, definitions in self.definitions.items():
            if any(not identifier.isdecimal() for identifier in definitions):
                raise NormalizationError(f"{section} has nonnumeric entity IDs.")
            self.next_ids[section] = max((int(identifier) for identifier in definitions), default=-1) + 1
        self.records, self.note_clones, self.rest_splits = {}, {}, []
        for section, tag in (("MasterBars", "MasterBar"), ("Bars", "Bar"), ("Voices", "Voice"), ("Beats", "Beat"), ("Notes", "Note")):
            _remove(self.root.find(section), tag)
        masters = source.findall("./MasterBars/MasterBar")
        for visit in score["playback"]["measureVisits"]:
            measure_index, visit_index = visit["measureIndex"], visit["visitIndex"]
            master = deepcopy(masters[measure_index])
            for tag in ("Repeat", "AlternateEndings", "Directions"):
                _remove(master, tag)
            bar = self.clone("Bars", master.findtext("Bars").strip())
            _set_text(master, "Bars", bar.get("id"))
            self.root.find("MasterBars").append(master)
            voice_refs = bar.findtext("Voices").split()
            for voice_index, voice_ref in enumerate(voice_refs):
                if voice_ref == "-1":
                    continue
                voice = self.clone("Voices", voice_ref)
                voice_refs[voice_index] = voice.get("id")
                beat_refs = []
                for beat_index, beat_ref in enumerate(voice.findtext("Beats", "").split()):
                    written_id = f"m{measure_index}:v{voice_index}:b{beat_index}"
                    beat = self.clone("Beats", beat_ref)
                    performance_id = f"p{visit_index}:{written_id}"
                    self.records[performance_id] = {
                        "node": beat, "voice": voice, "visit": visit, "writtenBeatId": written_id,
                    }
                    note_refs = []
                    for note_index, note_ref in enumerate(beat.findtext("Notes", "").split()):
                        note_id = f"{performance_id}:n{note_index}"
                        if note_id in omitted:
                            continue
                        note = self.clone("Notes", note_ref)
                        self.note_clones[note_id] = note
                        note_refs.append(note.get("id"))
                    if note_refs:
                        _set_text(beat, "Notes", " ".join(note_refs))
                    else:
                        _remove(beat, "Notes")
                    beat_refs.append(beat.get("id"))
                _set_text(voice, "Beats", " ".join(beat_refs))
            _set_text(bar, "Voices", " ".join(voice_refs))
        _linear_tempos(self.root, source, score)

    def identifier(self, section):
        identifier = self.next_ids[section]
        self.next_ids[section] += 1
        return str(identifier)

    def clone(self, section, reference):
        node = deepcopy(self.definitions[section][reference])
        node.set("id", self.identifier(section))
        self.root.find(section).append(node)
        return node

    def duration(self, beat, duration):
        rhythm = ET.Element("Rhythm", id=self.identifier("Rhythms"))
        for value, base in NOTE_VALUES.items():
            for dots in range(5):
                if base * (2 - Fraction(1, 2 ** dots)) == duration:
                    ET.SubElement(rhythm, "NoteValue").text = value
                    if dots:
                        ET.SubElement(rhythm, "AugmentationDot", count=str(dots))
                    break
            else:
                continue
            break
        else:
            ET.SubElement(rhythm, "NoteValue").text = "Quarter"
            ET.SubElement(rhythm, "PrimaryTuplet", num=str(duration.denominator), den=str(duration.numerator))
        self.root.find("Rhythms").append(rhythm)
        beat.find("Rhythm").set("ref", rhythm.get("id"))

    def split_rest(self, record, offset, duration):
        original = record["node"]
        carrier = deepcopy(original)
        carrier.set("id", self.identifier("Beats"))
        self.duration(original, offset)
        self.duration(carrier, duration - offset)
        self.root.find("Beats").append(carrier)
        refs = record["voice"].findtext("Beats").split()
        refs.insert(refs.index(original.get("id")) + 1, carrier.get("id"))
        _set_text(record["voice"], "Beats", " ".join(refs))
        self.rest_splits.append({
            "sourceWrittenBeatId": record["writtenBeatId"], "visitIndex": record["visit"]["visitIndex"],
            "prefixDurationQuarter": rational(offset), "carrierDurationQuarter": rational(duration - offset),
        })
        return carrier


def _strip_converted_text(writer, candidates, labels):
    by_beat = {}
    for candidate in candidates:
        gesture = candidate["gesture"]
        key = f"p{gesture['visitIndex']}:{gesture['writtenBeatId']}"
        by_beat.setdefault(key, set()).update(rule_tokens(candidate["rule"]))
    cleared = []
    for key, tokens in by_beat.items():
        beat = writer.records[key]["node"]
        text = beat.findtext("FreeText")
        if tokens:
            remaining = [token for token in normalized_text(text).split() if token not in tokens]
            if remaining:
                _set_text(beat, "FreeText", " ".join(remaining))
            else:
                _remove(beat, "FreeText")
    for suppressed in labels["review"]["suppressedAnnotations"]:
        visit = suppressed["performanceBeatId"].split(":", 1)[0]
        if f"{visit}:{suppressed['effectiveTextBeatId']}" in by_beat:
            _remove(writer.records[suppressed["performanceBeatId"]]["node"], "FreeText")
            cleared.append(deepcopy(suppressed))
    return cleared


def _strip_consumed_marks(writer, candidates, labels):
    converted = {item["gesture"]["id"] for item in candidates}
    other_meanings = {
        f"p{gesture['visitIndex']}:{gesture['writtenBeatId']}"
        for gesture in labels["targets"]["gestures"] if gesture["id"] not in converted
    } | {item["performanceBeatId"] for item in labels["review"]["unresolvedGestures"]}
    removed = []
    for candidate in candidates:
        gesture = candidate["gesture"]
        key = f"p{gesture['visitIndex']}:{gesture['writtenBeatId']}"
        beat = writer.records[key]["node"]
        if beat.findtext("Notes", "").strip() or key in other_meanings:
            continue
        reviewed_marks = candidate["rule"]["match"].get("beatTechniques", {})
        props = beat.find("Properties")
        for mark, prop_name in (("slapped", "Slapped"), ("brush", "Brush"), ("rasgueado", "Rasgueado")):
            if mark not in reviewed_marks or props is None:
                continue
            for prop in list(props):
                if prop.tag == "Property" and prop.get("name") == prop_name:
                    props.remove(prop)
                    removed.append({"sourceGestureId": gesture["id"], "mark": mark, "value": reviewed_marks[mark]})
        if "arpeggio" in reviewed_marks and beat.find("Arpeggio") is not None:
            _remove(beat, "Arpeggio")
            removed.append({"sourceGestureId": gesture["id"], "mark": "arpeggio", "value": reviewed_marks["arpeggio"]})
    return removed


def _place_hits(writer, candidates, labels, consumed):
    beats, _ = source_indices(writer.score)
    symbol_notes = [
        symbol for symbol in labels["review"]["notationSymbols"]
        if not any(segment["performanceId"] in consumed for segment in symbol["sourceSegments"])
    ]
    occupied_notes = labels["targets"]["notes"] + symbol_notes
    meanings = {
        f"p{gesture['visitIndex']}:{gesture['writtenBeatId']}" for gesture in labels["targets"]["gestures"]
    } | {item["performanceBeatId"] for item in labels["review"]["unresolvedGestures"]}
    reserved, placements = {}, []
    used_hosts = set()
    for candidate in sorted(candidates, key=lambda item: (
        fraction(item["gesture"]["onsetQuarter"], "gesture"), item["gesture"]["voiceIndex"], item["gesture"]["id"],
    )):
        gesture = candidate["gesture"]
        onset = fraction(gesture["onsetQuarter"], gesture["id"])
        try:
            string = select_free_string(occupied_notes, onset, reserved=reserved.get(onset, set()))
        except NormalizationError as error:
            for conflict in error.conflicts:
                conflict["sourceGestureId"] = gesture["id"]
            raise
        key = f"p{gesture['visitIndex']}:{gesture['writtenBeatId']}"
        record = writer.records[key]
        source_beat = beats[gesture["writtenBeatId"]]
        host = record["node"]
        remaining_dead = any(
            node["techniques"]["dead"] and f"p{gesture['visitIndex']}:{node['id']}" not in consumed
            for node in source_beat["notes"]
        )
        musical_host = ET.fromstring(ET.tostring(host, encoding="utf-8"))
        conflicting_marks = set(beat_techniques(musical_host, key, [])) & {"brush", "rasgueado", "slapped", "arpeggio"}
        if remaining_dead or conflicting_marks or host.get("id") in used_hosts:
            host = None
            for alternative_key, alternative in writer.records.items():
                if alternative["visit"]["visitIndex"] != gesture["visitIndex"] or alternative_key in meanings:
                    continue
                beat = beats[alternative["writtenBeatId"]]
                node = alternative["node"]
                start = fraction(alternative["visit"]["onsetQuarter"], "visit") + fraction(beat["offsetQuarter"], beat["id"])
                duration = fraction(beat["notatedDurationQuarter"], beat["id"])
                if (
                    not beat["isRest"] or beat["graceMode"] is not None or beat["techniques"]
                    or beat["chordRef"] is not None or normalized_text(node.findtext("FreeText"))
                    or node.get("id") in used_hosts or not start <= onset < start + duration
                ):
                    continue
                host = node if start == onset else writer.split_rest(alternative, onset - start, duration)
                used_hosts.add(node.get("id"))
                break
            if host is None:
                raise NormalizationError(
                    f"{gesture['id']}: its beat has other symbolic/strum/slap meanings and no safe existing rest voice "
                    "can carry the generic hit; sounding beats were not split."
                )
        note = ghost_dead_note(
            writer.identifier("Notes"), string,
            writer.score["instrument"]["openStringMidi"], writer.score["instrument"]["capoFret"],
            pitch_profile=writer.pitch_profile,
        )
        writer.root.find("Notes").append(note)
        _set_text(host, "Notes", " ".join([*host.findtext("Notes", "").split(), note.get("id")]))
        used_hosts.add(host.get("id"))
        reserved.setdefault(onset, set()).add(string)
        placements.append({
            "sourceGestureId": gesture["id"], "sourceTechnique": gesture["technique"],
            "sourceWrittenBeatId": gesture["writtenBeatId"], "sourceVisitIndex": gesture["visitIndex"],
            "sourceVoiceIndex": gesture["voiceIndex"], "onsetQuarter": gesture["onsetQuarter"],
            "consumedSourceNoteIds": candidate["consumedNoteIds"], "gpNoteId": note.get("id"),
            "gpBeatId": host.get("id"), "string": string, "gpStringIndex": 6 - string,
            "stringRole": "notation-carrier-only", "fingeringLabelMask": False,
        })
    return placements


def _output_indices(writer, decoded):
    by_beat = {beat["sourceBeatId"]: beat for beat in decoded["scoreEvents"]}
    by_note = {
        note["sourceNoteId"]: (beat, note) for beat in decoded["scoreEvents"] for note in beat["notes"]
    }
    beat_map = {
        identifier: by_beat[record["node"].get("id")]
        for identifier, record in writer.records.items()
    }
    note_map = {
        identifier: f"p{by_note[node.get('id')][0]['measureIndex']}:{by_note[node.get('id')][1]['id']}"
        for identifier, node in writer.note_clones.items()
    }
    return by_beat, by_note, beat_map, note_map


def _verify_music(writer, decoded, playback, consumed, by_note, note_map):
    _, written = source_indices(writer.score)
    output_events = {event["id"]: event for event in playback["noteEvents"]}
    for beat in decoded["scoreEvents"]:
        duplicates = sorted(string for string, count in Counter(note["string"] for note in beat["notes"]).items() if count > 1)
        if duplicates:
            raise NormalizationError(
                f"{beat['id']}: duplicate-string GP notes on physical strings {duplicates}; "
                "retained symbols were not relabeled or deleted to hide the conflict."
            )
    for event in writer.score["playback"]["noteEvents"]:
        if event.get("isSilent") or event["id"] in consumed:
            if event["id"] in note_map:
                raise NormalizationError("An omitted/silent symbol survived in the performance.")
            continue
        output = output_events[note_map[event["id"]]]
        for field in ("onsetQuarter", "durationQuarter", "isAttack"):
            if output[field] != event[field]:
                raise NormalizationError(f"{event['id']}: normalization changed {field}.")
        if output["tieFrom"] != (note_map[event["tieFrom"]] if event["tieFrom"] is not None else None):
            raise NormalizationError(f"{event['id']}: normalization changed a first-pass tie link.")
        source_beat, note = written[event["sourceEventId"]]
        new_beat, new_note = by_note[writer.note_clones[event["id"]].get("id")]
        if new_beat["voiceIndex"] != source_beat["voiceIndex"]:
            raise NormalizationError(f"{event['id']}: normalization changed the voice.")
        if any(new_note[key] != value for key, value in note.items() if key not in {"id", "sourceNoteId"}):
            raise NormalizationError(f"{event['id']}: note pitch/fingering/articulation changed.")
        original = deepcopy(writer.definitions["Notes"][note["sourceNoteId"]])
        original.set("id", new_note["sourceNoteId"])
        if semantic_node(original) != semantic_node(writer.note_clones[event["id"]]):
            raise NormalizationError(f"{event['id']}: an unmodified note's GPIF fields changed.")
    if playback["durationQuarter"] != writer.score["playback"]["durationQuarter"]:
        raise NormalizationError("Linearization changed the score duration.")
    visits = writer.score["playback"]["measureVisits"]
    for measure, visit in zip(decoded["measures"], visits):
        source = writer.score["measures"][visit["measureIndex"]]
        for field in ("durationQuarter", "timeSignature", "tripletFeel", "fermatas", "key"):
            if measure[field] != source[field]:
                raise NormalizationError(f"Visit {visit['visitIndex']}: normalization changed {field}.")
        if measure["repeatStart"] or measure["repeatEndCount"] or measure["directions"] or measure["alternateEndings"] or measure["referenceOnly"]:
            raise NormalizationError("Navigation or a reference section survived into played bars.")


def _verify_native_carriers(writer, placements):
    for placement in placements:
        note = writer.root.find(f"./Notes/Note[@id='{placement['gpNoteId']}']")
        props = {prop.get("name"): prop for prop in note.findall("./Properties/Property")}
        if note.findtext("AntiAccent") != "Normal" or {"ConcertPitch", "TransposedPitch", "String", "Fret", "Midi", "Muted"} - set(props):
            raise NormalizationError("Generic percussion lacks native GP ghost-note fields.")
        midi = int(props["Midi"].findtext("Number"))
        for name, offset in writer.pitch_profile["offsets"].items():
            if _pitch_value(props[name]) != midi + offset:
                raise NormalizationError("Generic percussion pitch coordinates disagree with the source display transposition.")
        if props["Muted"].find("Enable") is None:
            raise NormalizationError("Generic percussion was not serialized as a dead note.")


def _normalized_labels(source, score, placements, beat_map, note_map, by_note):
    result = canonicalize(score)
    generic = {placement["sourceGestureId"]: placement for placement in placements}
    gestures = []
    provenance_gestures = []
    for source_gesture in source["targets"]["gestures"]:
        placement = generic.get(source_gesture["id"])
        source_beat_key = f"p{source_gesture['visitIndex']}:{source_gesture['writtenBeatId']}"
        beat = beat_map[source_beat_key]
        gesture = deepcopy(source_gesture)
        if placement:
            beat, note = by_note[placement["gpNoteId"]]
            note_id = f"p{beat['measureIndex']}:{note['id']}"
            placement.update(
                normalizedWrittenBeatId=beat["id"], normalizedNoteId=note_id,
                normalizedWrittenNoteId=note["id"], normalizedVoiceIndex=beat["voiceIndex"],
                normalizedMeasureIndex=beat["measureIndex"],
            )
            gesture.update(
                technique="percussive_hit", attributes={}, symbolicNoteIds=[note_id],
                sourcePitchedNoteIds=[], contextPitchedNoteIds=[],
                notation={"kind": "ghost-dead", "gpNoteId": placement["gpNoteId"], "string": placement["string"], "stringRole": "notation-carrier-only"},
                labelMask={"gesture": True, "fingering": False},
            )
        else:
            for field in ("symbolicNoteIds", "sourcePitchedNoteIds", "contextPitchedNoteIds"):
                if field in gesture:
                    gesture[field] = [note_map[identifier] for identifier in gesture[field]]
        gesture.update(
            id=f"normalized:{source_gesture['id']}", writtenBeatId=beat["id"],
            visitIndex=beat["measureIndex"], voiceIndex=beat["voiceIndex"],
        )
        for field in ("interpretationRuleId", "evidenceBeatIds", "evidenceSource"):
            gesture.pop(field, None)
        gestures.append(gesture)
        provenance_gestures.append({"normalizedGestureId": gesture["id"], "source": deepcopy(source_gesture)})
    result["targets"]["gestures"] = gestures
    result["targets"]["noteRests"] = []
    for rest in source["targets"]["noteRests"]:
        mapped = deepcopy(rest)
        beat = beat_map[f"p{rest['visitIndex']}:{rest['writtenBeatId']}"]
        mapped.update(
            sourceWrittenNoteId=rest["writtenNoteId"], sourceWrittenBeatId=rest["writtenBeatId"],
            writtenNoteId=None, writtenBeatId=beat["id"], measureIndex=beat["measureIndex"],
            sourceVisitIndex=rest["visitIndex"], visitIndex=beat["measureIndex"],
            sourceOnly=True, normalizedNoteId=None,
        )
        result["targets"]["noteRests"].append(mapped)
    result["review"]["unresolvedGestures"] = []
    for unresolved in source["review"]["unresolvedGestures"]:
        mapped = deepcopy(unresolved)
        beat = beat_map[unresolved["performanceBeatId"]]
        mapped.update(
            sourcePerformanceBeatId=unresolved["performanceBeatId"],
            performanceBeatId=f"p{beat['measureIndex']}:{beat['id']}", labelMask={"gesture": False, "fingering": False},
        )
        if "writtenBeatId" in mapped:
            mapped["writtenBeatId"] = beat["id"]
        if "symbolicNoteIds" in mapped:
            mapped["symbolicNoteIds"] = [note_map[identifier] for identifier in mapped["symbolicNoteIds"]]
        result["review"]["unresolvedGestures"].append(mapped)
    bound = {identifier for gesture in gestures for identifier in gesture["symbolicNoteIds"]}
    for symbol in result["review"]["notationSymbols"]:
        symbol["labelMask"]["gesture"] = symbol["id"] in bound
        symbol["labelMask"]["fingering"] = False
    # Suppressed source text is provenance, not newly visible training supervision.
    result["review"]["suppressedAnnotations"] = deepcopy(source["review"]["suppressedAnnotations"])
    for record in result["review"]["suppressedAnnotations"]:
        record["sourceOnly"] = True
    result["review"]["sourceIssues"] = deepcopy(source["review"]["sourceIssues"])
    result["provenance"].update(
        sourceGestures=provenance_gestures, rawCanonicalProvenance=deepcopy(source["provenance"]),
        sourceMeasureVisits=deepcopy(source["measureVisits"]),
        silentRepeatedEntrySegments=deepcopy(source["targets"]["noteRests"]),
    )
    for field in ("performerId", "audio"):
        if field in source:
            result[field] = deepcopy(source[field])
    return result


def normalize_gp_bytes(raw, score, labels, annotations=None, conventions=None):
    """Return (GP archive bytes, normalized canonical labels, manifest details).

    This function performs no filesystem writes, alignment, inference or training.
    All source material is validated before occurrence-specific edits are made.
    """
    source, payloads, comment = _parse_archive(raw)
    _validate_inputs(source, raw, score, labels, annotations, conventions)
    candidates = _conversion_candidates(labels, annotations)
    consumed = _consumed_symbols(candidates, labels)
    silenced = {event["id"] for event in score["playback"]["noteEvents"] if event.get("isSilent")}
    writer = _OccurrenceWriter(source, score, consumed | silenced)
    cleared = _strip_converted_text(writer, candidates, labels)
    removed_marks = _strip_consumed_marks(writer, candidates, labels)
    placements = _place_hits(writer, candidates, labels, consumed)
    if _protected_tree(source) != _protected_tree(writer.root):
        raise NormalizationError("Nonmusical GPIF nodes or settings changed.")
    for identifier, original in writer.definitions["Rhythms"].items():
        actual = writer.root.find(f"./Rhythms/Rhythm[@id='{identifier}']")
        if semantic_node(actual) != semantic_node(original):
            raise NormalizationError("An original rhythm definition changed globally.")
    instrument = score["instrument"]
    decoded = _decode_musical_tree(writer.root, instrument)
    playback = performance_events(decoded, list(range(len(decoded["measures"]))))
    _, by_note, beat_map, note_map = _output_indices(writer, decoded)
    _verify_music(writer, decoded, playback, consumed, by_note, note_map)
    _verify_native_carriers(writer, placements)
    output = _archive_bytes(writer.root, payloads, comment)
    normalized_score = {
        **deepcopy(score), **decoded, "playback": playback,
        "sourceGpSha256": hashlib.sha256(output).hexdigest(),
        "providedTiming": catalog_timing(decoded), "audioAlignment": None,
    }
    normalized = _normalized_labels(labels, normalized_score, placements, beat_map, note_map, by_note)
    unresolved = canonical_counts(normalized)["unresolvedGestureCount"]
    blockers = ["audio_alignment_absent"]
    if unresolved:
        blockers.append("unresolved_source_gestures")
    if any(not all(note["labelMask"].values()) for note in normalized["targets"]["notes"]):
        blockers.append("unresolved_note_labels")
    report = {
        "normalizationVersion": NORMALIZATION_VERSION, "trainingReady": False, "audioAligned": False,
        "trainingBlockers": blockers, "rawGpSha256": hashlib.sha256(raw).hexdigest(),
        "normalizedGpSha256": hashlib.sha256(output).hexdigest(),
        "nativePitchOffsets": dict(writer.pitch_profile["offsets"]),
        "nativeGhostEncoding": "AntiAccent=Normal; Muted Enable; complete ConcertPitch/TransposedPitch",
        "nativeTextEncoding": "Preserve source CDATA fields, including titles and retained FreeText",
        "rawCounts": canonical_counts(labels), "normalizedCounts": canonical_counts(normalized),
        "rawWrittenMeasureCount": len(score["measures"]),
        "rawWrittenBeatCount": len(score["scoreEvents"]),
        "normalizedWrittenBeatCount": len(decoded["scoreEvents"]),
        "genericHitCount": len(placements), "consumedSymbolicSegmentCount": len(consumed),
        "silencedRepeatedEntrySegmentCount": len(silenced), "placements": placements,
        "unresolvedCounts": dict(sorted(Counter(item["reason"] for item in normalized["review"]["unresolvedGestures"]).items())),
        "placementPolicy": PLACEMENT_LIMITATION,
        "placementAvoidsRetainedSymbols": True,
        "omittedReferenceMeasureIndices": [measure["index"] for measure in score["measures"] if measure["referenceOnly"]],
        "referencePolicy": "Reference-only examples omitted from the performance copy; original GP and canonical evidence retained unchanged.",
        "legendPolicy": "Native ghost-dead notation needs no arbitrary percussion text or replacement instruction bars.",
        "keyPolicy": "Raw key choice preserved; accidental optimization belongs to the later final writer.",
        "clearedSuppressedAnnotations": cleared, "removedPercussionMarks": removed_marks, "restSplits": writer.rest_splits,
        "linearMeasureCount": len(playback["measureVisits"]),
        "durationQuarter": playback["durationQuarter"],
        "sourceMeasureVisits": deepcopy(score["playback"]["measureVisits"]),
        "normalizedTempoEvents": decoded["tempoEvents"],
        "normalizedTimeSignatureChanges": normalized_score["providedTiming"]["sourceTimeSignatureChanges"],
        "archiveResourcesPreserved": True, "nonmusicalGpifPreserved": True,
    }
    normalized["normalization"] = {
        "version": NORMALIZATION_VERSION, "trainingReady": False, "trainingBlockers": blockers,
        "placementPolicy": PLACEMENT_LIMITATION, "genericHitCount": len(placements),
    }
    return output, normalized, report
