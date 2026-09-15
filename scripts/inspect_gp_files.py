"""Inspect modern Guitar Pro archives without modifying their contents."""

from collections import Counter
import re
import xml.etree.ElementTree as ET
from zipfile import BadZipFile, ZipFile


class GpInspectionError(Exception):
    """A GP archive cannot be inspected reliably."""


def read_gp(path):
    try:
        with ZipFile(path) as archive:
            members = [item for item in archive.infolist() if item.filename == "Content/score.gpif"]
            if len(members) != 1 or members[0].file_size > 32 * 1024 * 1024:
                raise GpInspectionError("Missing, duplicated, or oversized score.gpif.")
            raw = archive.read(members[0])
    except BadZipFile as error:
        raise GpInspectionError("Not a valid modern GP ZIP archive.") from error
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise GpInspectionError("DTD/entity declarations are not supported.")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as error:
        raise GpInspectionError(f"Malformed GPIF XML: {error}") from error
    if root.tag != "GPIF":
        raise GpInspectionError(f"Unsupported GPIF root: {root.tag}")
    return root


def properties(element):
    return {prop.get("name"): prop for prop in element.findall("./Properties/Property")}


def number(prop, name):
    if prop is None:
        return None
    text = prop.findtext(name)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError as error:
        raise GpInspectionError(f"Noninteger {name}: {text}") from error


def pitch_to_midi(value):
    match = re.fullmatch(r"([A-G])([#b]?)(-?\d+)", value)
    if not match:
        raise GpInspectionError(f"Invalid scientific pitch: {value}")
    pitch, accidental, octave = match.groups()
    return (int(octave) + 1) * 12 + {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[pitch] + {"": 0, "#": 1, "b": -1}[accidental]


def compare_tuning(entry, inspection):
    selected = entry["selectedTuningIndex"]
    if selected is None:
        return "catalog_tuning_not_selected"
    expected = [pitch_to_midi(pitch) for pitch in entry["tunings"][selected]["strings"]]
    staves = [staff for track in inspection["tracks"] if track["instrument"] != "drumKit" for staff in track["staves"]]
    if not staves or any(staff["openStringMidi"] != expected for staff in staves):
        return "catalog_gp_tuning_mismatch"
    return None


def apply_owner_confirmation(result, rules):
    confirmation = rules.get("ownerConfirmations", {}).get(result["catalogId"])
    if confirmation is None:
        return
    if confirmation["gpSha256"] != result["sha256"]:
        result["issues"].append("owner_confirmation_revision_mismatch")
        return
    if confirmation["capoType"] != "full":
        raise GpInspectionError("Unsupported owner capo confirmation.")
    result["ownerConfirmation"] = dict(confirmation)
    result["resolvedInspectionIssues"] = [issue for issue in result["inspection"]["warnings"] if issue == "inconsistent_partial_capo_metadata"]
    result["issues"] = [issue for issue in result["issues"] if issue not in result["resolvedInspectionIssues"]]


def notes_per_track(root, tracks):
    bars = {item.get("id"): item for item in root.findall("./Bars/Bar")}
    voices = {item.get("id"): item for item in root.findall("./Voices/Voice")}
    beats = {item.get("id"): item for item in root.findall("./Beats/Beat")}
    order = root.findtext("./MasterTrack/Tracks", "").split()
    if order:
        by_id = {track["id"]: track for track in tracks}
        if len(order) != len(tracks) or set(order) != set(by_id):
            raise GpInspectionError("Master-track order does not match the track definitions.")
        ordered_tracks = [by_id[identifier] for identifier in order]
    else:
        ordered_tracks = tracks
    columns = [track for track in ordered_tracks for _ in track["staves"]]
    found = {track["id"]: set() for track in tracks}
    for master_bar in root.findall("./MasterBars/MasterBar"):
        references = master_bar.findtext("Bars", "").split()
        if len(references) != len(columns):
            raise GpInspectionError("Master-bar staff count does not match track staves.")
        for track, bar_id in zip(columns, references):
            if bar_id == "-1":
                continue
            bar = bars.get(bar_id)
            if bar is None:
                raise GpInspectionError("Missing bar reference.")
            for voice_id in bar.findtext("Voices", "").split():
                if voice_id == "-1":
                    continue
                voice = voices.get(voice_id)
                if voice is None:
                    raise GpInspectionError("Missing voice reference.")
                for beat_id in voice.findtext("Beats", "").split():
                    beat = beats.get(beat_id)
                    if beat is None:
                        raise GpInspectionError("Missing beat reference.")
                    found[track["id"]].update(beat.findtext("Notes", "").split())
    return {identifier: len(notes) for identifier, notes in found.items()}


def inspect_gp(path):
    root = read_gp(path)
    tracks = root.findall("./Tracks/Track")
    if not tracks:
        raise GpInspectionError("No track definitions.")
    result = {
        "gpVersion": root.findtext("GPVersion"),
        "title": root.findtext("./Score/Title", ""),
        "subtitle": root.findtext("./Score/SubTitle", ""),
        "artist": root.findtext("./Score/Artist", ""),
        "album": root.findtext("./Score/Album", ""),
        "trackCount": len(tracks),
        "tracks": [],
        "warnings": [],
    }
    for track in tracks:
        staves = track.findall("./Staves/Staff")
        track_info = {
            "id": track.get("id"),
            "name": track.findtext("Name", ""),
            "instrument": track.findtext("./InstrumentSet/Type"),
            "playbackState": track.findtext("PlaybackState"),
            "staffCount": len(staves),
            "staves": [],
        }
        if not staves:
            result["warnings"].append("missing_staff_metadata")
        for staff in staves:
            props = properties(staff)
            tuning = props.get("Tuning")
            raw_pitches = tuning.findtext("Pitches", "") if tuning is not None else ""
            try:
                pitches = [int(pitch) for pitch in raw_pitches.split()]
            except ValueError as error:
                raise GpInspectionError("Noninteger tuning pitches.") from error
            partial_fret = number(props.get("PartialCapoFret"), "Fret")
            partial_flags = props.get("PartialCapoStringFlags")
            bitset = partial_flags.findtext("Bitset") if partial_flags is not None else None
            staff_info = {
                "openStringMidi": pitches,
                "capoFret": number(props.get("CapoFret"), "Fret"),
                "partialCapoFret": partial_fret,
                "partialCapoStringFlags": bitset,
                "partialCapoActive": bool(partial_fret and bitset and "1" in bitset),
            }
            track_info["staves"].append(staff_info)
            if len(pitches) != 6:
                result["warnings"].append("not_six_strings")
            if any(pitch < 0 or pitch > 127 for pitch in pitches):
                result["warnings"].append("invalid_tuning_pitch")
            if partial_fret not in (None, 0) or (bitset is not None and set(bitset) - {"0"}):
                result["warnings"].append("partial_capo_active" if staff_info["partialCapoActive"] else "inconsistent_partial_capo_metadata")
            if staff_info["capoFret"] is None:
                result["warnings"].append("missing_full_capo_metadata")
            elif not 0 <= staff_info["capoFret"] <= 24:
                result["warnings"].append("invalid_full_capo")
        if len(staves) > 1:
            result["warnings"].append("multiple_staves")
        result["tracks"].append(track_info)
    result["noteCount"] = len(root.findall("./Notes/Note"))
    track_notes = notes_per_track(root, result["tracks"])
    for track in result["tracks"]:
        track["noteCount"] = track_notes[track["id"]]
    if not result["noteCount"]:
        result["warnings"].append("empty_score")
    result["automationTypes"] = sorted({item.findtext("Type", "") for item in root.findall(".//Automation")})
    if any("capo" in value.casefold() or "tuning" in value.casefold() for value in result["automationTypes"]):
        result["warnings"].append("tuning_or_capo_automation")
    result["capoOrTuningText"] = sorted({node.text for node in root.findall("./Beats/Beat/FreeText") if node.text and re.search(r"\b(capo|retun\w*|tuning)\b", node.text, re.IGNORECASE)})
    if result["capoOrTuningText"]:
        result["warnings"].append("capo_or_tuning_text_requires_review")
    if any(re.search(r"\bpartial\s+capo\b", text, re.IGNORECASE) for text in result["capoOrTuningText"]):
        result["warnings"].append("partial_capo_in_text")
    result["warnings"] = sorted(set(result["warnings"]))
    note_properties = Counter(prop.get("name") for prop in root.findall("./Notes/Note/Properties/Property"))
    result["notePropertyCounts"] = dict(sorted(note_properties.items()))
    return result
