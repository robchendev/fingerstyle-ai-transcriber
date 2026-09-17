from copy import deepcopy
from fractions import Fraction
import hashlib
from io import BytesIO
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

from scripts.canonical_events import canonicalize
from scripts.gp_events import catalog_timing, decode_score, note_event, performance_events, playback_order
from scripts.gp_normalization import (
    GPIF_ENTRY, NormalizationError, ghost_dead_note, normalize_gp_bytes,
    select_free_string, semantic_node, native_pitch_profile, notated_uncertainty_intervals,
)
from scripts import settings
from tests.test_gp_events import TUNING, musical_score, note_xml


CONVENTIONS = {
    "schemaVersion": 1, "uppercaseOIsWristThump": True,
    "simultaneousTextPriority": "lowest-voice-index",
}

def template_score():
    root = musical_score()
    root.find("./Beats/Beat[@id='0']/FreeText").text = "*"
    root.find("./Beats/Beat[@id='0']/Properties").clear()
    root.find("MasterBars").append(ET.fromstring(
        "<MasterBar><Time>4/4</Time><Bars>20</Bars><Section><Text>Instructions</Text></Section></MasterBar>"
    ))
    root.find("Bars").append(ET.fromstring('<Bar id="20"><Voices>20 -1 -1 -1</Voices></Bar>'))
    root.find("Voices").append(ET.fromstring('<Voice id="20"><Beats>20</Beats></Voice>'))
    root.find("Beats").append(ET.fromstring(
        '<Beat id="20"><Rhythm ref="0"/><FreeText>Tap the side of the guitar *</FreeText></Beat>'
    ))
    ET.SubElement(root, "FuturePresentation", mode="keep").text = "Unknown font and page settings"
    ET.SubElement(root.find("./Beats/Beat[@id='0']"), "UserTransposedPitchStemOrientation").text = "Downward"
    ET.SubElement(root.find("./Bars/Bar[@id='0']"), "XProperties").append(
        ET.fromstring('<XProperty id="12345"><Int>17</Int></XProperty>')
    )
    for master in root.findall("./MasterBars/MasterBar"):
        key = ET.SubElement(master, "Key")
        ET.SubElement(key, "AccidentalCount").text = "5"
        ET.SubElement(key, "Mode").text = "Major"
        ET.SubElement(key, "TransposeAs").text = "Sharps"
    return root


def archive_bytes(root):
    root = deepcopy(root)
    for note in root.findall("./Notes/Note"):
        props = note.find("Properties")
        if props is None or props.find("Property[@name='Midi']/Number") is None:
            continue
        midi = int(props.findtext("Property[@name='Midi']/Number"))
        spellings = (("C", ""), ("C", "#"), ("D", ""), ("D", "#"), ("E", ""), ("F", ""), ("F", "#"), ("G", ""), ("G", "#"), ("A", ""), ("A", "#"), ("B", ""))
        for name, offset in (("ConcertPitch", 0), ("TransposedPitch", 12)):
            if props.find(f"Property[@name='{name}']") is not None:
                continue
            value = midi + offset
            pitch = ET.SubElement(ET.SubElement(props, "Property", name=name), "Pitch")
            for key, text in (("Step", spellings[value % 12][0]), ("Accidental", spellings[value % 12][1]), ("Octave", str(value // 12))):
                ET.SubElement(pitch, key).text = text
    result = BytesIO()
    with ZipFile(result, "w") as archive:
        archive.comment = b"preserve unknown archive comment"
        for name, payload in (
            ("Content/BinaryStylesheet", b"\x00\xfffont-size=17;future-settings\x80"),
            ("Content/Preferences.json", b'{"font":"source font","unknown":[1,2]}'),
            ("Content/Stylesheets/score.gpss", b"arbitrary stylesheet bytes"),
            ("Content/ScoreViews/1.gpsv", b"untouched view resource"),
            ("future/resource", b"\x01\x02opaque"),
            (GPIF_ENTRY, ET.tostring(root, encoding="utf-8", xml_declaration=True)),
        ):
            info = ZipInfo(name, date_time=(2025, 2, 3, 4, 5, 6))
            info.compress_type = ZIP_DEFLATED
            info.comment = b"entry comment"
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload)
    return result.getvalue()


def prepared(root=None, *, technique="body_tap", attributes=None, order=None, conventions=None, rule=True):
    root = template_score() if root is None else root
    raw = archive_bytes(root)
    digest = hashlib.sha256(raw).hexdigest()
    decoded = decode_score(ET.fromstring(ET.tostring(root, encoding="utf-8")), TUNING, 2)
    playback = performance_events(
        decoded, playback_order(decoded["measures"]) if order is None else order,
        silence_repeated_entry_ties=True,
    )
    score = {
        "schemaVersion": 1, "catalogId": "example", "sourceGpSha256": digest,
        "audioPath": "audio.flac", "sourceRangeSeconds": None, "timeUnit": "quarter-note",
        "audioAlignment": None,
        "instrument": {
            "stringOrder": [6, 5, 4, 3, 2, 1], "openStringMidi": TUNING,
            "capoFret": 2, "fretConvention": "capo-relative",
        },
        "providedTiming": catalog_timing(decoded), **decoded, "playback": playback,
    }
    annotations = {
        "gpSha256": digest, "rules": [{
            "id": "local-body", "technique": technique,
            "attributes": attributes if attributes is not None else {"finger": "middle", "location": "side"},
            "evidenceBeatIds": ["m2:v0:b0"], "match": {"tokens": ["*"]},
            "consumedTokens": ["*"], "consumesDeadNotes": False,
        }] if rule else [],
    }
    return raw, score, canonicalize(score, annotations, conventions), annotations, conventions


def normalized_root(raw):
    with ZipFile(BytesIO(raw)) as archive:
        return ET.fromstring(archive.read(GPIF_ENTRY))


class GpNormalizationTests(unittest.TestCase):
    def test_all_dead_carriers_use_zero_fret_but_keep_pitch_note_and_ghost_status(self):
        from tests.test_canonical_events import X_CONVENTIONS

        root = template_score()
        root.find("Notes").clear()
        root.find("Notes").extend([note_xml("0", fret=99, dead=True), note_xml("1", fret=5, dead=True), note_xml("2", string=3, fret=8)])
        ET.SubElement(root.find("./Notes/Note[@id='1']"), "AntiAccent").text = "Normal"
        root.find("./Beats/Beat[@id='0']/FreeText").text = ""
        root.find("./Beats/Beat[@id='0']/Notes").text = "0 2"
        arguments = prepared(root, conventions=X_CONVENTIONS, rule=False)
        before = arguments[0]
        output, labels, report = normalize_gp_bytes(*arguments)
        decoded = normalized_root(output)
        parsed = decode_score(decoded, TUNING, 2)
        dead = [note for beat in parsed["scoreEvents"] for note in beat["notes"] if note["techniques"]["dead"]]
        pitched = [note for beat in parsed["scoreEvents"] for note in beat["notes"] if not note["techniques"]["dead"]]
        self.assertEqual([note["fret"] for note in dead], [0, 0])
        self.assertEqual([note["soundingPitchMidi"] for note in dead], [None, None])
        self.assertNotIn("antiAccent", dead[0]["techniques"])
        self.assertEqual(dead[1]["techniques"]["antiAccent"], "Normal")
        self.assertEqual([note["fret"] for note in pitched], [8])
        self.assertEqual([g["technique"] for g in labels["targets"]["gestures"]], ["thumb_slap", "percussive_hit"])
        self.assertEqual(report["normalizationVersion"], 4)
        self.assertEqual(len(report["unpitchedCarrierNormalizations"]), 2)
        self.assertEqual(arguments[0], before)
        with ZipFile(BytesIO(before)) as old, ZipFile(BytesIO(output)) as new:
            for name in old.namelist():
                if name != GPIF_ENTRY:
                    self.assertEqual(old.read(name), new.read(name))
        for note in decoded.findall("./Notes/Note"):
            if note.find("./Properties/Property[@name='Muted']") is not None:
                props = {prop.get("name"): prop for prop in note.findall("./Properties/Property")}
                midi = int(props["Midi"].findtext("Number"))
                for name, offset in report["nativePitchOffsets"].items():
                    from scripts.gp_normalization import _pitch_value
                    self.assertEqual(_pitch_value(props[name]), midi + offset)

    def test_native_cdata_title_and_retained_wrist_text_are_preserved(self):
        root = template_score()
        ET.SubElement(root.find("./Beats/Beat[@id='1']"), "FreeText").text = "O"
        raw, score, labels, annotations, conventions = prepared(root, conventions=CONVENTIONS)
        rebuilt = BytesIO()
        with ZipFile(BytesIO(raw)) as source, ZipFile(rebuilt, "w") as target:
            target.comment = source.comment
            for info in source.infolist():
                payload = source.read(info.filename)
                if info.filename == GPIF_ENTRY:
                    payload = payload.replace(b"<Title>Example</Title>", b"<Title><![CDATA[Example]]></Title>")
                    payload = payload.replace(b"<FreeText>O</FreeText>", b"<FreeText><![CDATA[O]]></FreeText>")
                target.writestr(info, payload)
        raw = rebuilt.getvalue()
        digest = hashlib.sha256(raw).hexdigest()
        score["sourceGpSha256"] = digest
        labels["provenance"]["sourceGpSha256"] = digest
        annotations["gpSha256"] = digest
        output, _, _ = normalize_gp_bytes(raw, score, labels, annotations, conventions)
        with ZipFile(BytesIO(output)) as archive:
            xml = archive.read(GPIF_ENTRY)
        self.assertIn(b"<Title><![CDATA[Example]]></Title>", xml)
        self.assertIn(b"<FreeText><![CDATA[O]]></FreeText>", xml)
        self.assertNotIn(b"<FreeText>*</FreeText>", xml)

    def test_pitch_profile_preserves_native_display_transposition_instead_of_assuming_octave(self):
        root = ET.fromstring('''<GPIF><Notes><Note id="0"><Properties>
          <Property name="Midi"><Number>42</Number></Property>
          <Property name="ConcertPitch"><Pitch><Step>F</Step><Accidental>#</Accidental><Octave>3</Octave></Pitch></Property>
          <Property name="TransposedPitch"><Pitch><Step>D</Step><Accidental/><Octave>4</Octave></Pitch></Property>
        </Properties></Note></Notes></GPIF>''')
        profile = native_pitch_profile(root)
        self.assertEqual(profile["offsets"], {"ConcertPitch": 0, "TransposedPitch": 8})
        ghost = ghost_dead_note("1", 6, TUNING, 2, pitch_profile=profile)
        self.assertEqual(ghost.findtext("./Properties/Property[@name='TransposedPitch']/Pitch/Step"), "D")
        self.assertEqual(ghost.findtext("./Properties/Property[@name='TransposedPitch']/Pitch/Octave"), "4")
        new_pitch = ghost_dead_note("2", 5, TUNING, 2, pitch_profile=profile)
        self.assertEqual(new_pitch.findtext("./Properties/Property[@name='ConcertPitch']/Pitch/Step"), "B")
        self.assertEqual(new_pitch.findtext("./Properties/Property[@name='TransposedPitch']/Pitch/Step"), "G")
        self.assertEqual(new_pitch.findtext("./Properties/Property[@name='TransposedPitch']/Pitch/Octave"), "4")

    def test_missing_native_pitch_metadata_fails_rather_than_writing_invisible_carriers(self):
        with self.assertRaisesRegex(NormalizationError, "complete source concert/transposed"):
            native_pitch_profile(musical_score())

    def test_reader_compatible_but_native_incomplete_carrier_is_rejected(self):
        original = ghost_dead_note

        def incomplete(*args, **kwargs):
            note = original(*args, **kwargs)
            properties = note.find("Properties")
            properties.remove(properties.find("Property[@name='ConcertPitch']"))
            return note

        with patch("scripts.gp_normalization.ghost_dead_note", side_effect=incomplete):
            with self.assertRaisesRegex(NormalizationError, "native GP ghost-note fields"):
                normalize_gp_bytes(*prepared())

    def test_ghost_dead_serialization_is_native_and_has_valid_pitch_properties(self):
        profile = native_pitch_profile(normalized_root(archive_bytes(template_score())))
        node = ghost_dead_note("17", 6, TUNING, 2, pitch_profile=profile)
        self.assertEqual(node.findtext("AntiAccent"), "Normal")
        self.assertIsNotNone(node.find("./Properties/Property[@name='Muted']/Enable"))
        self.assertEqual(node.findtext("./Properties/Property[@name='ConcertPitch']/Pitch/Step"), "F")
        self.assertEqual(node.findtext("./Properties/Property[@name='ConcertPitch']/Pitch/Accidental"), "#")
        self.assertEqual(node.findtext("./Properties/Property[@name='ConcertPitch']/Pitch/Octave"), "3")
        self.assertEqual(node.findtext("./Properties/Property[@name='TransposedPitch']/Pitch/Octave"), "4")
        self.assertIsNone(node.find("FreeText"))
        issues = []
        note = note_event(node, "ghost", TUNING, 2, issues)
        self.assertEqual((note["gpStringIndex"], note["string"], note["fret"], note["storedMidi"]), (0, 6, 0, 42))
        self.assertTrue(note["techniques"]["dead"])
        self.assertEqual(note["techniques"]["antiAccent"], "Normal")
        self.assertIsNone(note["soundingPitchMidi"])
        self.assertEqual(issues, [])

    def test_archive_payloads_unknown_settings_and_source_styling_are_preserved(self):
        args = prepared()
        output, labels, report = normalize_gp_bytes(*args)
        self.assertEqual(hashlib.sha256(args[0]).hexdigest(), args[1]["sourceGpSha256"])
        with ZipFile(BytesIO(args[0])) as original, ZipFile(BytesIO(output)) as normalized:
            self.assertEqual(original.namelist(), normalized.namelist())
            self.assertEqual(original.comment, normalized.comment)
            for name in original.namelist():
                if name != GPIF_ENTRY:
                    self.assertEqual(original.read(name), normalized.read(name))
                for attr in ("date_time", "compress_type", "comment", "external_attr", "extra"):
                    self.assertEqual(getattr(original.getinfo(name), attr), getattr(normalized.getinfo(name), attr))
        before, after = normalized_root(args[0]), normalized_root(output)
        for path in ("Score", "Tracks", "FuturePresentation"):
            self.assertEqual(semantic_node(before.find(path)), semantic_node(after.find(path)))
        first = decode_score(after, TUNING, 2)["scoreEvents"][0]
        self.assertEqual(after.find(f"./Beats/Beat[@id='{first['sourceBeatId']}']/UserTransposedPitchStemOrientation").text, "Downward")
        self.assertEqual(
            semantic_node(before.find("./Bars/Bar[@id='0']/XProperties")),
            semantic_node(after.find("./Bars/Bar/XProperties")),
        )
        self.assertEqual(report["omittedReferenceMeasureIndices"], [2])
        self.assertEqual(report["linearMeasureCount"], 2)
        self.assertFalse(report["trainingReady"])
        self.assertIn("audio_alignment_absent", report["trainingBlockers"])
        self.assertEqual(labels["provenance"]["sourceGpSha256"], hashlib.sha256(output).hexdigest())

    def test_generic_target_drops_detailed_attributes_and_binds_new_note(self):
        output, labels, report = normalize_gp_bytes(*prepared())
        gesture = labels["targets"]["gestures"][0]
        placement = report["placements"][0]
        self.assertEqual(gesture["technique"], "percussive_hit")
        self.assertEqual(gesture["attributes"], {})
        self.assertEqual(gesture["symbolicNoteIds"], [placement["normalizedNoteId"]])
        self.assertFalse(gesture["labelMask"]["fingering"])
        self.assertEqual(placement["string"], 5)
        self.assertEqual(placement["gpStringIndex"], 1)
        self.assertEqual(placement["stringRole"], "notation-carrier-only")
        source = labels["provenance"]["sourceGestures"][0]["source"]
        self.assertEqual(source["attributes"]["finger"], "middle")
        self.assertNotIn("interpretationRuleId", gesture)
        self.assertEqual(report["genericHitCount"], 1)
        node = normalized_root(output).find(f"./Notes/Note[@id='{placement['gpNoteId']}']")
        self.assertEqual(node.findtext("AntiAccent"), "Normal")
        self.assertIsNone(normalized_root(output).find(f"./Beats/Beat[@id='{placement['gpBeatId']}']/FreeText"))
        symbol = next(symbol for symbol in labels["review"]["notationSymbols"] if symbol["id"] == placement["normalizedNoteId"])
        self.assertTrue(symbol["labelMask"]["gesture"])
        self.assertFalse(symbol["labelMask"]["fingering"])

    def test_unknown_xml_comments_and_processing_instructions_survive_entity_cloning(self):
        root = template_score()
        root.find("./Notes/Note[@id='0']").append(ET.Comment("retain source note comment"))
        root.find("./Beats/Beat[@id='0']").append(ET.ProcessingInstruction("future", 'style="keep"'))
        root.find("Score").append(ET.Comment("retain presentation comment"))
        output, _, _ = normalize_gp_bytes(*prepared(root))
        with ZipFile(BytesIO(output)) as archive:
            xml = archive.read(GPIF_ENTRY)
        self.assertIn(b"<!--retain source note comment-->", xml)
        self.assertIn(b'<?future style="keep"?>', xml)
        self.assertIn(b"<!--retain presentation comment-->", xml)

    def test_selection_uses_full_notated_intervals_and_half_open_boundaries(self):
        notes = [
            {"id": "bass", "string": 6, "onsetQuarter": [0, 1], "notatedDurationQuarter": [2, 1], "labelMask": {"notatedDuration": True}},
            {"id": "fifth", "string": 5, "onsetQuarter": [2, 1], "notatedDurationQuarter": [2, 1], "labelMask": {"notatedDuration": True}},
        ]
        self.assertEqual(select_free_string(notes, Fraction(199, 100)), 5)
        self.assertEqual(select_free_string(notes, [2, 1]), 6)
        self.assertEqual(select_free_string(notes, [0, 1], reserved={5}), 4)

    def test_unresolved_and_grace_durations_reserve_strings_conservatively(self):
        note = {"id": "grace", "string": 6, "onsetQuarter": [10, 1], "notatedDurationQuarter": None, "labelMask": {"notatedDuration": False}}
        self.assertEqual(select_free_string([note], 0), 5)
        self.assertEqual(select_free_string([note], 1000), 5)
        note.update(notatedDurationQuarter=[1, 1])
        self.assertEqual(select_free_string([note], 1000), 5)

    def test_all_strings_busy_uses_text_and_records_conflict_without_shortening(self):
        root = template_score()
        for string in range(1, 6):
            root.find("Notes").append(note_xml(str(string + 30), string=string))
        root.find("./Beats/Beat[@id='0']/Notes").text = "0 31 32 33 34 35"
        args = prepared(root)
        original = args[0]
        output, labels, report = normalize_gp_bytes(*args)
        placement = report["placements"][0]
        conflict = placement["fallbackDetails"][0]
        self.assertEqual(conflict["onsetQuarter"], [0, 1])
        self.assertEqual(conflict["occupiedStrings"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(placement["sourceGestureId"], "p0:m0:v0:b0:local-body")
        self.assertEqual(report["textFallbackHitCount"], 1)
        self.assertEqual(report["textFallbackReasons"], {"no_free_notated_string": 1})
        self.assertIsNone(placement["string"])
        self.assertIsNone(placement["normalizedNoteId"])
        self.assertEqual(labels["targets"]["gestures"][0]["symbolicNoteIds"], [])
        self.assertEqual(labels["targets"]["gestures"][0]["notation"]["kind"], "generic-hit-text")
        self.assertEqual(normalized_root(output).findtext("./Beats/Beat/FreeText"), "(X)")
        self.assertEqual(len(labels["targets"]["notes"]), len(args[2]["targets"]["notes"]))
        self.assertEqual(args[0], original)

    def test_pitched_note_ties_attacks_voices_and_harmonic_articulations_survive(self):
        root = template_score()
        for identifier, destination in (("0", "false"), ("1", "true")):
            original = root.find(f"./Notes/Note[@id='{identifier}']")
            root.find("Notes").remove(original)
            replacement = note_xml(identifier, harmonic=("Artificial", 12), tie={"origin": "true", "destination": destination})
            ET.SubElement(replacement, "Vibrato").text = "Slight"
            ET.SubElement(replacement, "Accent").text = "Heavy"
            ET.SubElement(ET.SubElement(replacement.find("Properties"), "Property", name="PalmMuted"), "Enable")
            root.find("Notes").append(replacement)
        args = prepared(root, technique="string_slap", attributes={"producesPitchedNote": True})
        _, labels, report = normalize_gp_bytes(*args)
        before, after = args[2]["targets"]["notes"][0], labels["targets"]["notes"][0]
        for key in ("voiceIndex", "onsetQuarter", "notatedDurationQuarter", "string", "fret", "basePitchMidi", "soundingPitchMidi", "isAttack"):
            self.assertEqual(before[key], after[key])
        self.assertEqual(after["notatedDurationQuarter"], [8, 1])
        self.assertEqual(len(after["sourceSegments"]), 2)
        for first, second in zip(before["sourceSegments"], after["sourceSegments"]):
            for key in ("tie", "harmonic", "techniques", "bend", "beatTechniques", "dynamic"):
                self.assertEqual(first[key], second[key])
        self.assertEqual(report["genericHitCount"], 1)

    def test_consumed_cluster_becomes_one_ghost_without_removing_chord_notes(self):
        root = template_score()
        root.find("Notes").append(note_xml("40", string=3, dead=True))
        root.find("Notes").append(note_xml("41", string=4, dead=True))
        root.find("./Beats/Beat[@id='0']/Notes").text = "0 40 41"
        args = list(prepared(root))
        args[3]["rules"][0]["consumesDeadNotes"] = True
        args[2] = canonicalize(args[1], args[3])
        _, labels, report = normalize_gp_bytes(*args)
        self.assertEqual(report["consumedSymbolicSegmentCount"], 2)
        self.assertEqual(report["genericHitCount"], 1)
        self.assertEqual(len(labels["review"]["notationSymbols"]), 1)
        self.assertEqual(len(labels["targets"]["notes"]), 1)
        self.assertEqual(labels["targets"]["notes"][0]["notatedDurationQuarter"], [8, 1])

    def test_reviewed_geometry_on_consumed_percussion_does_not_survive_as_extra_marks(self):
        root = template_score()
        root.find("./Beats/Beat[@id='0']/Notes").text = "40"
        root.find("Notes").append(note_xml("40", string=3, dead=True))
        root.find("./Notes/Note[@id='1']").remove(root.find("./Notes/Note[@id='1']/Tie"))
        root.find("./Beats/Beat[@id='0']/Properties").append(ET.fromstring('<Property name="Slapped"><Enable/></Property>'))
        args = list(prepared(root, technique="string_slap"))
        args[3]["rules"][0]["consumesDeadNotes"] = True
        args[3]["rules"][0]["match"]["beatTechniques"] = {"slapped": True}
        args[2] = canonicalize(args[1], args[3])
        output, _, report = normalize_gp_bytes(*args)
        placement = report["placements"][0]
        host = normalized_root(output).find(f"./Beats/Beat[@id='{placement['gpBeatId']}']")
        self.assertEqual(placement["normalizedVoiceIndex"], 0)
        self.assertIsNone(host.find("./Properties/Property[@name='Slapped']"))
        self.assertEqual(report["removedPercussionMarks"], [{
            "sourceGestureId": "p0:m0:v0:b0:local-body", "mark": "slapped", "value": True,
        }])

    def test_unknown_symbols_remain_visible_masked_and_are_not_generic_hits(self):
        root = template_score()
        unknown = note_xml("40", string=3, dead=True)
        ET.SubElement(unknown, "AntiAccent").text = "Normal"
        root.find("Notes").append(unknown)
        root.find("./Beats/Beat[@id='1']/Notes").text = "1 40"
        output, labels, report = normalize_gp_bytes(*prepared(root))
        self.assertEqual(report["unresolvedCounts"], {"uninterpreted_dead_note_cluster": 1})
        self.assertEqual(report["genericHitCount"], 1)
        self.assertFalse(report["trainingReady"])
        self.assertIn("unresolved_source_gestures", report["trainingBlockers"])
        unknowns = [symbol for symbol in labels["review"]["notationSymbols"] if not symbol["labelMask"]["gesture"]]
        self.assertEqual(len(unknowns), 1)
        self.assertEqual(unknowns[0]["string"], 3)
        self.assertFalse(unknowns[0]["labelMask"]["pitch"])
        self.assertFalse(unknowns[0]["labelMask"]["fingering"])
        self.assertEqual(len([n for n in normalized_root(output).find("Notes") if n.findtext("AntiAccent") == "Normal"]), 2)

    def test_generic_hit_avoids_simultaneous_retained_thumb_or_unknown_x(self):
        for reviewed_thumb in (False, True):
            with self.subTest(reviewed_thumb=reviewed_thumb):
                root = template_score()
                root.find("Notes").append(note_xml("40", string=1, dead=True))
                root.find("./Beats/Beat[@id='0']/Notes").text = "0 40"
                args = list(prepared(root))
                if reviewed_thumb:
                    args[3]["rules"].append({
                        "id": "local-thumb", "technique": "thumb_slap", "attributes": {},
                        "evidenceBeatIds": ["m2:v0:b0"],
                        "match": {"deadStrings": [5], "text": "*"},
                        "consumedTokens": [], "consumesDeadNotes": True,
                    })
                    args[2] = canonicalize(args[1], args[3])
                output, labels, report = normalize_gp_bytes(*args)
                placement = report["placements"][0]
                self.assertEqual(placement["string"], 4)
                self.assertEqual(placement["normalizedVoiceIndex"], 0)
                self.assertEqual(report["genericHitCount"], 1)
                self.assertTrue(report["placementAvoidsRetainedSymbols"])
                symbols = labels["review"]["notationSymbols"]
                retained = next(symbol for symbol in symbols if symbol["string"] == 5)
                self.assertNotIn("antiAccent", retained["sourceSegments"][0]["techniques"])
                self.assertEqual(retained["labelMask"]["gesture"], reviewed_thumb)
                self.assertFalse(retained["labelMask"]["fingering"])
                self.assertEqual(
                    [gesture["technique"] for gesture in labels["targets"]["gestures"]],
                    ["percussive_hit", "thumb_slap"] if reviewed_thumb else ["percussive_hit"],
                )
                for beat in decode_score(normalized_root(output), TUNING, 2)["scoreEvents"]:
                    strings = [note["string"] for note in beat["notes"]]
                    self.assertEqual(len(strings), len(set(strings)))

    def test_inherited_duplicate_string_notes_fail_instead_of_erasing_unknown_x(self):
        root = template_score()
        root.find("Notes").append(note_xml("40", string=0, dead=True))
        root.find("./Beats/Beat[@id='0']/Notes").text = "0 40"
        root.find("./Notes/Note[@id='1']").remove(root.find("./Notes/Note[@id='1']/Tie"))
        with self.assertRaisesRegex(NormalizationError, "duplicate-string GP notes"):
            normalize_gp_bytes(*prepared(root))

    def test_long_instruction_is_not_treated_as_body_percussion(self):
        root = template_score()
        root.find("./Beats/Beat[@id='0']/FreeText").text = "capo to fret 5"
        output, labels, report = normalize_gp_bytes(*prepared(root, rule=False))
        self.assertEqual(report["genericHitCount"], 0)
        self.assertEqual(labels["targets"]["gestures"], [])
        self.assertEqual(report["unresolvedCounts"], {"uninterpreted_annotation": 1})
        self.assertTrue(any(node.text == "capo to fret 5" for node in normalized_root(output).findall("./Beats/Beat/FreeText")))

    def test_short_text_without_a_legend_or_x_becomes_native_generic_percussion(self):
        output, labels, report = normalize_gp_bytes(*prepared(rule=False))
        self.assertEqual(report["genericHitCount"], 1)
        self.assertEqual([g["technique"] for g in labels["targets"]["gestures"]], ["percussive_hit"])
        self.assertFalse(any(node.text == "*" for node in normalized_root(output).findall("./Beats/Beat/FreeText")))
        ghost = normalized_root(output).find("./Notes/Note[AntiAccent='Normal']")
        self.assertIsNotNone(ghost)
        self.assertIsNotNone(ghost.find("./Properties/Property[@name='ConcertPitch']/Pitch"))
        self.assertEqual(labels["provenance"]["percussiveHitTextDetectionMaxLen"], 3)

    def test_changed_text_limit_rejects_stale_canonical_inputs(self):
        args = prepared()
        with patch.object(settings, "PERCUSSIVE_HIT_TEXT_DETECTION_MAX_LEN", 2):
            with self.assertRaisesRegex(NormalizationError, "provenance is stale"):
                normalize_gp_bytes(*args)

    def test_silent_repeated_entry_removes_only_suppressed_tie_not_other_chord_notes(self):
        root = template_score()
        root.find("Notes").append(note_xml("40", string=2, fret=7))
        root.find("./Beats/Beat[@id='1']/Notes").text = "1 40"
        args = prepared(root, order=[0, 1, 1])
        output, labels, report = normalize_gp_bytes(*args)
        self.assertEqual(report["durationQuarter"], [12, 1])
        self.assertEqual(report["silencedRepeatedEntrySegmentCount"], 1)
        returning = [note for note in labels["targets"]["notes"] if note["onsetQuarter"] == [8, 1]]
        self.assertEqual(len(returning), 1)
        self.assertEqual(returning[0]["string"], 4)
        self.assertTrue(returning[0]["isAttack"])
        self.assertEqual(labels["targets"]["noteRests"][0]["durationQuarter"], [4, 1])
        self.assertIsNone(labels["targets"]["noteRests"][0]["normalizedNoteId"])
        self.assertEqual(labels["targets"]["notes"][0]["notatedDurationQuarter"], [8, 1])
        decoded = decode_score(normalized_root(output), TUNING, 2)
        first_pass = next(beat for beat in decoded["scoreEvents"] if beat["measureIndex"] == 1 and beat["voiceIndex"] == 0)
        last_pass = next(beat for beat in decoded["scoreEvents"] if beat["measureIndex"] == 2 and beat["voiceIndex"] == 0)
        self.assertTrue(first_pass["notes"][0]["tie"]["destination"])
        self.assertEqual(len(last_pass["notes"]), 1)

    def test_fully_silenced_repeated_entry_is_an_exact_duration_rest(self):
        output, labels, report = normalize_gp_bytes(*prepared(order=[0, 1, 1]))
        rests = [rest for rest in labels["targets"]["rests"] if rest["voiceIndex"] == 0 and rest["onsetQuarter"] == [8, 1]]
        self.assertEqual(len(rests), 1)
        self.assertEqual(rests[0]["notatedDurationQuarter"], [4, 1])
        self.assertEqual(report["durationQuarter"], [12, 1])
        self.assertEqual(len(normalized_root(output).findall("./Tracks/Track")), 1)

    def test_replayed_original_origins_keep_real_ties_and_clones_are_occurrence_local(self):
        output, labels, report = normalize_gp_bytes(*prepared(order=[0, 1, 0, 1]))
        self.assertEqual(report["genericHitCount"], 2)
        self.assertEqual(len({item["gpNoteId"] for item in report["placements"]}), 2)
        self.assertEqual(len({item["gpBeatId"] for item in report["placements"]}), 2)
        self.assertEqual([note["notatedDurationQuarter"] for note in labels["targets"]["notes"]], [[8, 1], [8, 1]])
        self.assertEqual(report["silencedRepeatedEntrySegmentCount"], 0)
        root = normalized_root(output)
        for section in ("Beats", "Notes", "Voices", "Bars"):
            ids = [node.get("id") for node in root.find(section)]
            self.assertEqual(len(ids), len(set(ids)))

    def test_tempo_beat_units_meter_and_key_follow_playback_visits(self):
        root = template_score()
        root.find("./MasterTrack/Automations").append(ET.fromstring(
            "<Automation><Type>Tempo</Type><Bar>1</Bar><Position>0</Position><Value>72 2</Value><Linear>false</Linear></Automation>"
        ))
        root.findall("./MasterBars/MasterBar")[1].find("Time").text = "2/2"
        output, labels, report = normalize_gp_bytes(*prepared(root, order=[0, 1, 0, 1]))
        self.assertEqual([(e["measureIndex"], e["bpm"], e["beatUnit"]) for e in report["normalizedTempoEvents"]], [
            (0, 60, [3, 8]), (1, 72, [1, 4]), (2, 60, [3, 8]), (3, 72, [1, 4]),
        ])
        self.assertEqual(report["normalizedTimeSignatureChanges"], [
            {"measureIndex": 1, "timeSignature": [2, 2]},
            {"measureIndex": 2, "timeSignature": [4, 4]},
            {"measureIndex": 3, "timeSignature": [2, 2]},
        ])
        self.assertEqual(labels["conditioning"]["providedTiming"]["tempo"], {"bpm": 60, "beatUnit": [3, 8]})
        for master in normalized_root(output).findall("./MasterBars/MasterBar"):
            self.assertEqual(master.findtext("./Key/AccidentalCount"), "5")
            self.assertEqual(master.findtext("./Key/TransposeAs"), "Sharps")

    def test_navigation_is_cleared_only_in_copy_and_reference_bars_never_play(self):
        root = template_score()
        ET.SubElement(root.findall("./MasterBars/MasterBar")[0], "Repeat", start="true", end="false")
        ET.SubElement(root.findall("./MasterBars/MasterBar")[1], "Repeat", start="false", end="true", count="2")
        args = prepared(root)
        output, _, report = normalize_gp_bytes(*args)
        self.assertEqual(report["linearMeasureCount"], 4)
        self.assertIsNotNone(normalized_root(args[0]).find("./MasterBars/MasterBar/Repeat"))
        for master in normalized_root(output).findall("./MasterBars/MasterBar"):
            for tag in ("Repeat", "AlternateEndings", "Directions"):
                self.assertIsNone(master.find(tag))
            self.assertNotEqual(master.findtext("./Section/Text"), "Instructions")

    def test_removing_winning_text_does_not_unhide_suppressed_lower_voice(self):
        root = template_score()
        ET.SubElement(root.find("./Beats/Beat[@id='2']"), "FreeText").text = "hidden arbitrary annotation"
        args = prepared(root, conventions=CONVENTIONS)
        output, labels, report = normalize_gp_bytes(*args)
        self.assertEqual(len(report["clearedSuppressedAnnotations"]), 1)
        decoded = decode_score(normalized_root(output), TUNING, 2)
        first = [beat for beat in decoded["scoreEvents"] if beat["measureIndex"] == 0]
        self.assertTrue(all(beat["text"] is None for beat in first))
        self.assertEqual(len(labels["review"]["suppressedAnnotations"]), 1)
        self.assertTrue(labels["review"]["suppressedAnnotations"][0]["sourceOnly"])

    def test_owner_uppercase_o_and_thumb_slap_are_preserved(self):
        root = template_score()
        ET.SubElement(root.find("./Beats/Beat[@id='1']"), "FreeText").text = "O"
        output, labels, report = normalize_gp_bytes(*prepared(root, conventions=CONVENTIONS))
        self.assertEqual([g["technique"] for g in labels["targets"]["gestures"]], ["percussive_hit", "wrist_thump"])
        self.assertTrue(any(node.text == "O" for node in normalized_root(output).findall("./Beats/Beat/FreeText")))
        _, thumb_labels, thumb_report = normalize_gp_bytes(*prepared(technique="thumb_slap"))
        self.assertEqual(thumb_report["genericHitCount"], 0)
        self.assertEqual(thumb_labels["targets"]["gestures"][0]["technique"], "thumb_slap")
        self.assertEqual(report["genericHitCount"], 1)

    def test_musical_strums_and_slap_harmonics_are_not_generic_dead_notes(self):
        for technique in ("strum", "rasgueado", "harmonic_strum", "slap_harmonic", "strumming_pull_off"):
            with self.subTest(technique=technique):
                args = prepared(technique=technique)
                _, labels, report = normalize_gp_bytes(*args)
                self.assertEqual(report["genericHitCount"], 0)
                self.assertEqual(labels["targets"]["gestures"][0]["technique"], technique)
                self.assertEqual(len(labels["review"]["notationSymbols"]), 0)
                self.assertEqual(labels["targets"]["notes"][0]["notatedDurationQuarter"], [8, 1])

    def test_explicit_simultaneous_percussive_compound_is_one_generic_hit(self):
        attrs = {"components": [{"technique": "string_slap", "attributes": {}}, {"technique": "body_tap", "attributes": {}}], "timing": "simultaneous"}
        _, labels, report = normalize_gp_bytes(*prepared(technique="compound_gesture", attributes=attrs))
        self.assertEqual(report["genericHitCount"], 1)
        self.assertEqual(labels["targets"]["gestures"][0]["attributes"], {})
        self.assertEqual(labels["targets"]["gestures"][0]["technique"], "percussive_hit")
        for changed in ("ordered", "mixed"):
            with self.subTest(changed=changed):
                copy = deepcopy(attrs)
                if changed == "ordered":
                    copy["timing"] = "ordered"
                else:
                    copy["components"][1]["technique"] = "wrist_thump"
                output, labels, report = normalize_gp_bytes(*prepared(technique="compound_gesture", attributes=copy))
                generic = labels["targets"]["gestures"][0]
                self.assertEqual(report["genericHitCount"], 1)
                self.assertEqual(generic["labelMask"]["onset"], changed != "ordered")
                self.assertEqual(generic["scoreOnsetKnown"], changed != "ordered")
                if changed == "mixed":
                    self.assertEqual([g["technique"] for g in labels["targets"]["gestures"]], ["percussive_hit", "wrist_thump"])
                    self.assertEqual(normalized_root(output).findtext("./Beats/Beat/FreeText"), "O")
                else:
                    self.assertIn("unresolved_gesture_timing", report["trainingBlockers"])

    def test_reviewed_unusual_percussion_reduces_to_the_same_coarse_target(self):
        for technique in ("body_flam", "body_scratch", "string_slap", "nail_attack", "slap_pluck", "string_tap", "string_scrape"):
            with self.subTest(technique=technique):
                _, labels, report = normalize_gp_bytes(*prepared(technique=technique))
                self.assertEqual(report["genericHitCount"], 1)
                self.assertEqual(labels["targets"]["gestures"][0]["technique"], "percussive_hit")
                self.assertEqual(labels["targets"]["gestures"][0]["attributes"], {})

    def test_conflicting_beat_uses_existing_rest_without_changing_strum(self):
        root = template_score()
        root.find("./Beats/Beat[@id='0']/Properties").append(ET.fromstring('<Property name="Brush"><Direction>Down</Direction></Property>'))
        output, labels, report = normalize_gp_bytes(*prepared(root))
        self.assertEqual(report["placements"][0]["sourceVoiceIndex"], 0)
        self.assertEqual(report["placements"][0]["normalizedVoiceIndex"], 1)
        first = decode_score(normalized_root(output), TUNING, 2)["scoreEvents"][0]
        self.assertEqual(first["techniques"]["brush"], "Down")
        self.assertFalse(any(note["techniques"]["dead"] for note in first["notes"]))
        self.assertEqual(labels["targets"]["notes"][0]["notatedDurationQuarter"], [8, 1])

    def test_rest_split_is_exact_and_never_splits_sounding_notes(self):
        root = template_score()
        root.find("Rhythms").append(ET.fromstring('<Rhythm id="3"><NoteValue>Half</NoteValue></Rhythm>'))
        root.find("./Beats/Beat[@id='0']/Rhythm").set("ref", "3")
        root.find("./Beats/Beat[@id='0']/FreeText").text = None
        root.find("Notes").append(note_xml("3", string=0, fret=1))
        root.find("./Notes/Note[@id='1']").remove(root.find("./Notes/Note[@id='1']/Tie"))
        root.find("Beats").append(ET.fromstring(
            '<Beat id="3"><Rhythm ref="3"/><Notes>3</Notes><FreeText>*</FreeText>'
            '<Properties><Property name="Brush"><Direction>Down</Direction></Property></Properties></Beat>'
        ))
        root.find("./Voices/Voice[@id='0']/Beats").text = "0 3"
        args = prepared(root)
        output, labels, report = normalize_gp_bytes(*args)
        self.assertEqual(report["restSplits"], [{
            "sourceWrittenBeatId": "m0:v1:b0", "visitIndex": 0,
            "prefixDurationQuarter": [2, 1], "carrierDurationQuarter": [2, 1],
        }])
        self.assertEqual(report["placements"][0]["onsetQuarter"], [2, 1])
        notes = labels["targets"]["notes"]
        self.assertEqual([note["notatedDurationQuarter"] for note in notes], [[2, 1], [2, 1], [4, 1]])
        first_measure = [beat for beat in decode_score(normalized_root(output), TUNING, 2)["scoreEvents"] if beat["measureIndex"] == 0]
        self.assertEqual([beat["notatedDurationQuarter"] for beat in first_measure], [[2, 1]] * 4)

    def test_same_onset_pitched_alternative_preserves_voices_and_compound_components(self):
        root = template_score()
        root.find("./Beats/Beat[@id='0']/Properties").append(ET.fromstring('<Property name="Brush"><Direction>Down</Direction></Property>'))
        root.find("Notes").append(note_xml("3", string=2))
        ET.SubElement(root.find("./Beats/Beat[@id='2']"), "Notes").text = "3"
        attrs = {"components": [{"technique": "body_tap"}, {"technique": "strum"}], "timing": "simultaneous"}
        output, labels, report = normalize_gp_bytes(*prepared(root, technique="compound_gesture", attributes=attrs))
        generic, strum = labels["targets"]["gestures"]
        self.assertEqual((generic["voiceIndex"], strum["voiceIndex"]), (1, 0))
        self.assertNotEqual(generic["writtenBeatId"], strum["writtenBeatId"])
        self.assertEqual(report["nativeGhostHitCount"], 1)
        self.assertEqual([note["voiceIndex"] for note in labels["targets"]["notes"]], [0, 1, 1])
        self.assertEqual(decode_score(normalized_root(output), TUNING, 2)["scoreEvents"][0]["techniques"]["brush"], "Down")

    def test_unsafe_hosts_use_text_preserving_wrist_and_articulations(self):
        root = template_score()
        root.find("./Bars/Bar[@id='0']/Voices").text = "0 -1 -1 -1"
        root.find("./Beats/Beat[@id='0']/Properties").append(ET.fromstring('<Property name="Brush"><Direction>Down</Direction></Property>'))
        attrs = {"components": [{"technique": "body_tap"}, {"technique": "wrist_thump"}], "timing": "simultaneous"}
        output, labels, report = normalize_gp_bytes(*prepared(root, technique="compound_gesture", attributes=attrs))
        self.assertEqual(report["textFallbackReasons"], {"no_safe_voice_beat": 1})
        self.assertEqual(normalized_root(output).findtext("./Beats/Beat/FreeText"), "O (X)")
        self.assertEqual([g["technique"] for g in labels["targets"]["gestures"]], ["percussive_hit", "wrist_thump"])
        self.assertEqual(labels["targets"]["notes"][0]["notatedDurationQuarter"], [8, 1])

    def test_grace_text_fallback_keeps_slots_and_timing_uncertainty(self):
        root = template_score()
        ET.SubElement(root.find("./Beats/Beat[@id='0']"), "GraceNotes").text = "OnBeat"
        root.find("./Voices/Voice[@id='0']/Beats").text = "0 1"
        ET.SubElement(root.find("./Beats/Beat[@id='1']"), "FreeText").text = "O"
        output, labels, report = normalize_gp_bytes(*prepared(root, order=[0], conventions=CONVENTIONS))
        self.assertEqual(report["textFallbackReasons"], {"unknown_score_onset": 1})
        generic = labels["targets"]["gestures"][0]
        self.assertFalse(generic["labelMask"]["onset"])
        self.assertFalse(generic["scoreOnsetKnown"])
        self.assertEqual(generic["graceMode"], "OnBeat")
        first = decode_score(normalized_root(output), TUNING, 2)["scoreEvents"]
        self.assertEqual([beat["text"] for beat in first[:2]], ["(X)", "O"])
        self.assertIn("unresolved_gesture_timing", report["trainingBlockers"])

    def test_text_fallback_is_visible_beside_upper_voice_wrist_annotation(self):
        root = template_score()
        root.find("./Beats/Beat[@id='0']/FreeText").text = "O"
        for string in range(1, 6):
            root.find("Notes").append(note_xml(str(string + 30), string=string))
        root.find("./Beats/Beat[@id='0']/Notes").text = "0 31 32 33 34 35"
        root.find("Notes").append(note_xml("40", string=1, dead=True))
        ET.SubElement(root.find("./Beats/Beat[@id='2']"), "Notes").text = "40"
        args = list(prepared(root, conventions=CONVENTIONS))
        args[3]["rules"][0].update(match={"deadStrings": [5]}, consumesDeadNotes=True, consumedTokens=[])
        args[2] = canonicalize(args[1], args[3], CONVENTIONS)
        output, labels, report = normalize_gp_bytes(*args)
        placement = report["placements"][0]
        self.assertEqual((placement["sourceVoiceIndex"], placement["normalizedVoiceIndex"]), (1, 0))
        self.assertEqual(normalized_root(output).findtext("./Beats/Beat/FreeText"), "O (X)")
        self.assertEqual(sum(g["technique"] == "wrist_thump" for g in labels["targets"]["gestures"]), 1)

    def test_grace_uncertainty_is_bounded_without_fabricating_duration(self):
        root = template_score()
        ET.SubElement(root.find("./Beats/Beat[@id='1']"), "GraceNotes").text = "BeforeBeat"
        root.find("./Notes/Note[@id='1']").remove(root.find("./Notes/Note[@id='1']/Tie"))
        root.find("./Voices/Voice[@id='1']/Beats").text = "1 2"
        root.find("Notes").append(note_xml("3", string=1))
        root.find("./Beats/Beat[@id='1']/Notes").text = "3"
        _, score, labels, _, _ = prepared(root)
        grace = next(note for note in labels["targets"]["notes"] if note["notatedDurationQuarter"] is None)
        bounds = notated_uncertainty_intervals([grace], score)
        self.assertEqual(bounds[grace["id"]], (Fraction(0), Fraction(8)))
        self.assertEqual(select_free_string([grace], 2, reserved={6}, uncertainty_intervals=bounds), 4)
        self.assertEqual(select_free_string([grace], 8, reserved={6}, uncertainty_intervals=bounds), 5)
        grace["sourceSegments"][0]["graceMode"] = "OnBeat"
        bounds = notated_uncertainty_intervals([grace], score)
        self.assertEqual(bounds[grace["id"]], (Fraction(4), Fraction(8)))
        self.assertEqual(select_free_string([grace], 2, reserved={6}, uncertainty_intervals=bounds), 5)
        self.assertIsNone(grace["notatedDurationQuarter"])
        trailing_grace = {
            "id": "trailing", "string": 5, "voiceIndex": 0, "isAttack": True,
            "onsetQuarter": [0, 1], "notatedDurationQuarter": None,
            "sourceSegments": [
                {"graceMode": None, "onsetQuarter": [0, 1], "durationQuarter": [4, 1]},
                {"graceMode": "OnBeat", "onsetQuarter": [4, 1], "durationQuarter": None},
            ],
        }
        attack = {"id": "later", "string": 5, "isAttack": True, "onsetQuarter": [6, 1], "notatedDurationQuarter": [1, 1], "labelMask": {"notatedDuration": True}, "sourceSegments": [{"graceMode": None}]}
        bounds = notated_uncertainty_intervals([trailing_grace, attack], score)
        self.assertEqual(bounds["trailing"], (Fraction(0), Fraction(6)))

    def test_native_ramps_repeat_with_endpoints_but_reject_cut_navigation(self):
        root = template_score()
        root.find("./MasterTrack/Automations/Automation/Linear").text = "true"
        root.find("./MasterTrack/Automations").append(ET.fromstring(
            "<Automation><Type>Tempo</Type><Bar>1</Bar><Position>0</Position><Value>72 3</Value><Linear>false</Linear></Automation>"
        ))
        _, _, report = normalize_gp_bytes(*prepared(root, order=[0, 1, 0, 1]))
        self.assertEqual([(event["bpm"], event["linear"]) for event in report["normalizedTempoEvents"]], [(60, True), (72, False), (60, True), (72, False)])
        for order in ([0], [0, 0, 1]):
            with self.subTest(order=order), self.assertRaisesRegex(NormalizationError, "contiguous endpoint"):
                normalize_gp_bytes(*prepared(root, order=order))

    def test_ramp_endpoint_at_omitted_reference_boundary_stays_at_score_end(self):
        root = template_score()
        root.find("./MasterTrack/Automations/Automation/Linear").text = "true"
        root.find("./MasterTrack/Automations").append(ET.fromstring(
            "<Automation><Type>Tempo</Type><Bar>2</Bar><Position>0</Position><Value>40.6823 3</Value><Linear>false</Linear></Automation>"
        ))
        output, _, report = normalize_gp_bytes(*prepared(root))
        self.assertEqual(report["durationQuarter"], [8, 1])
        endpoint = report["normalizedTempoEvents"][-1]
        self.assertEqual((endpoint["measureIndex"], endpoint["positionRatio"], endpoint["bpm"], endpoint["linear"]), (1, [1, 1], 40.6823, False))
        self.assertEqual(len(normalized_root(output).findall("./MasterBars/MasterBar")), 2)

    def test_stale_revision_canonical_and_unsupported_tempo_never_get_fallbacks(self):
        args = list(prepared())
        args[1]["sourceGpSha256"] = "0" * 64
        with self.assertRaisesRegex(NormalizationError, "revision hashes"):
            normalize_gp_bytes(*args)
        args = list(prepared())
        args[2]["targets"]["notes"][0]["fret"] = 24
        with self.assertRaisesRegex(NormalizationError, "Canonical source targets"):
            normalize_gp_bytes(*args)
        root = template_score()
        root.find("./MasterTrack/Automations/Automation/Linear").text = "true"
        with self.assertRaisesRegex(NormalizationError, "Tempo ramps"):
            normalize_gp_bytes(*prepared(root))

if __name__ == "__main__":
    unittest.main()
