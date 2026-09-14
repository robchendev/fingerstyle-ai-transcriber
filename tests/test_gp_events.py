from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from scripts.download_audio import sha256
from scripts.extract_gp_events import extract_entry
from scripts.gp_events import EventExtractionError, catalog_timing, decode_score, note_event, performance_events, playback_order, rhythm_duration, validate_provided_timing


TUNING = [40, 45, 50, 55, 59, 64]


def note_xml(identifier="0", string=0, fret=0, capo=2, harmonic=None, dead=False, tie=None):
    node = ET.Element("Note", id=identifier)
    ET.SubElement(node, "InstrumentArticulation").text = "0"
    props = ET.SubElement(node, "Properties")
    for name, child, value in (("String", "String", string), ("Fret", "Fret", fret), ("Midi", "Number", TUNING[string] + capo + fret)):
        ET.SubElement(ET.SubElement(props, "Property", name=name), child).text = str(value)
    if harmonic:
        ET.SubElement(ET.SubElement(props, "Property", name="Harmonic"), "Enable")
        ET.SubElement(ET.SubElement(props, "Property", name="HarmonicType"), "HType").text = harmonic[0]
        ET.SubElement(ET.SubElement(props, "Property", name="HarmonicFret"), "HFret").text = str(harmonic[1])
    if dead:
        ET.SubElement(ET.SubElement(props, "Property", name="Muted"), "Enable")
    if tie:
        ET.SubElement(node, "Tie", **tie)
    return node


def musical_score():
    root = ET.fromstring("""
    <GPIF><GPVersion>8</GPVersion>
    <Score><Title>Example</Title><FirstPageHeader>DO NOT EXPORT FONT SETTINGS</FirstPageHeader></Score>
    <Tracks><Track id="0"><Name>Guitar</Name><InstrumentSet><Type>steelGuitar</Type></InstrumentSet><Staves><Staff><Properties>
      <Property name="Tuning"><Pitches>40 45 50 55 59 64</Pitches></Property>
      <Property name="CapoFret"><Fret>2</Fret></Property>
      <Property name="PartialCapoFret"><Fret>0</Fret></Property>
      <Property name="PartialCapoStringFlags"><Bitset>000000</Bitset></Property>
    </Properties></Staff></Staves></Track></Tracks>
    <MasterTrack><Tracks>0</Tracks><Automations><Automation><Type>Tempo</Type><Bar>0</Bar><Position>0</Position><Value>60 3</Value><Linear>false</Linear></Automation></Automations></MasterTrack>
    <MasterBars><MasterBar><Time>4/4</Time><Bars>0</Bars></MasterBar><MasterBar><Time>4/4</Time><Bars>1</Bars></MasterBar></MasterBars>
    <Bars><Bar id="0"><Voices>0 2 -1 -1</Voices></Bar><Bar id="1"><Voices>1 2 -1 -1</Voices></Bar></Bars>
    <Voices><Voice id="0"><Beats>0</Beats></Voice><Voice id="1"><Beats>1</Beats></Voice><Voice id="2"><Beats>2</Beats></Voice></Voices>
    <Beats>
      <Beat id="0"><Rhythm ref="0"/><Notes>0</Notes><Dynamic>MF</Dynamic><FreeText>O</FreeText><Properties><Property name="Slapped"><Enable/></Property></Properties></Beat>
      <Beat id="1"><Rhythm ref="0"/><Notes>1</Notes></Beat>
      <Beat id="2"><Rhythm ref="0"/></Beat>
    </Beats>
    <Notes/>
    <Rhythms><Rhythm id="0"><NoteValue>Whole</NoteValue></Rhythm></Rhythms>
    </GPIF>""")
    root.find("Notes").append(note_xml(tie={"origin": "true", "destination": "false"}))
    root.find("Notes").append(note_xml("1", tie={"origin": "false", "destination": "true"}))
    return root


def bar(start=False, end=0, endings=(), targets=(), jumps=()):
    return {
        "repeatStart": start, "repeatEndCount": end, "alternateEndings": list(endings),
        "directions": [{"kind": "Target", "value": value} for value in targets] + [{"kind": "Jump", "value": value} for value in jumps],
    }


class MusicalEventTests(unittest.TestCase):
    def test_supplied_tempo_preserves_dotted_beat_unit_and_unreduced_meter(self):
        root = musical_score()
        for measure in root.findall("./MasterBars/MasterBar"):
            measure.find("Time").text = "6/8"
        timing = catalog_timing(decode_score(root, TUNING, 2))
        self.assertEqual(timing["tempo"], {"bpm": 60, "beatUnit": [3, 8]})
        self.assertEqual(timing["timeSignature"], [6, 8])
        self.assertEqual(validate_provided_timing(timing["tempo"], timing["timeSignature"]), Fraction(90))
        self.assertEqual(timing["sourceTempoChanges"], [])
        self.assertEqual(timing["sourceTimeSignatureChanges"], [])
        for tempo, meter in ((None, [4, 4]), ({"bpm": 120}, [4, 4]), ({"bpm": 120, "beatUnit": [1, 4]}, None), ({"bpm": 0, "beatUnit": [1, 4]}, [4, 4]), ({"bpm": True, "beatUnit": [1, 4]}, [4, 4]), ({"bpm": 90, "beatUnit": [1, 4]}, [4, 0])):
            with self.subTest(tempo=tempo, meter=meter), self.assertRaises(EventExtractionError):
                validate_provided_timing(tempo, meter)

    def test_source_tempo_and_meter_changes_are_not_flattened(self):
        root = musical_score()
        root.findall("./MasterBars/MasterBar")[1].find("Time").text = "3/4"
        root.find("./MasterTrack/Automations").append(ET.fromstring("<Automation><Type>Tempo</Type><Bar>1</Bar><Position>0.5</Position><Value>72 2</Value><Linear>true</Linear></Automation>"))
        timing = catalog_timing(decode_score(root, TUNING, 2))
        self.assertEqual(timing["sourceTempoChanges"], [{"measureIndex": 1, "positionRatio": [1, 2], "bpm": 72, "beatUnit": [1, 4], "linear": True}])
        self.assertEqual(timing["sourceTimeSignatureChanges"], [{"measureIndex": 1, "timeSignature": [3, 4]}])
        section = ET.SubElement(root.findall("./MasterBars/MasterBar")[1], "Section")
        ET.SubElement(section, "Text").text = "Instructions"
        reference = catalog_timing(decode_score(root, TUNING, 2))
        self.assertEqual(reference["sourceTempoChanges"], [])
        self.assertEqual(reference["sourceTimeSignatureChanges"], [])

    def test_absent_initial_tempo_is_not_replaced_with_a_default(self):
        root = musical_score()
        root.find("./MasterTrack/Automations/Automation/Bar").text = "1"
        with self.assertRaisesRegex(EventExtractionError, "start of the performance"):
            catalog_timing(decode_score(root, TUNING, 2))

    def test_rational_rhythms_preserve_dots_and_tuplets(self):
        rhythm = ET.fromstring('<Rhythm><NoteValue>Quarter</NoteValue><AugmentationDot count="2"/><PrimaryTuplet num="3" den="2"/></Rhythm>')
        duration, notation = rhythm_duration(rhythm)
        self.assertEqual(duration, Fraction(7, 6))
        self.assertEqual(notation, {"value": "Quarter", "dots": 2, "tuplet": [3, 2]})
        with self.assertRaises(EventExtractionError):
            rhythm_duration(ET.fromstring("<Rhythm><NoteValue>Unknown</NoteValue></Rhythm>"))

    def test_pitch_capo_harmonics_and_dead_notes_are_distinct(self):
        ordinary = note_event(note_xml(), "a", TUNING, 2, [])
        self.assertEqual((ordinary["string"], ordinary["fret"], ordinary["soundingPitchMidi"]), (6, 0, 42))
        natural = note_event(note_xml(fret=7, harmonic=("Natural", 7)), "b", TUNING, 2, [])
        self.assertEqual(natural["storedMidi"], 49)
        self.assertEqual(natural["soundingPitchMidi"], 61)
        artificial = note_event(note_xml(fret=2, harmonic=("Artificial", 12)), "c", TUNING, 2, [])
        self.assertEqual(artificial["soundingPitchMidi"], 56)
        self.assertIsNone(note_event(note_xml(dead=True), "d", TUNING, 2, [])["soundingPitchMidi"])
        issues = []
        self.assertIsNone(note_event(note_xml(harmonic=("Natural", 11)), "e", TUNING, 2, issues)["soundingPitchMidi"])
        self.assertEqual(issues[0]["code"], "unsupported_harmonic_pitch")

    def test_voices_reused_definitions_rests_and_ties_keep_occurrence_identity(self):
        decoded = decode_score(musical_score(), TUNING, 2)
        self.assertEqual(len(decoded["scoreEvents"]), 4)
        rests = [beat for beat in decoded["scoreEvents"] if beat["isRest"]]
        self.assertEqual([beat["sourceBeatId"] for beat in rests], ["2", "2"])
        self.assertNotEqual(rests[0]["id"], rests[1]["id"])
        self.assertEqual(decoded["scoreEvents"][0]["text"], "O")
        self.assertTrue(decoded["scoreEvents"][0]["techniques"]["slapped"])
        self.assertEqual(decoded["tempoEvents"][0]["quarterBpm"], [90, 1])
        playback = performance_events(decoded, playback_order(decoded["measures"]))
        self.assertEqual(playback["noteEvents"][1]["tieFrom"], playback["noteEvents"][0]["id"])
        self.assertFalse(playback["noteEvents"][1]["isAttack"])
        self.assertEqual(playback["noteEvents"][1]["onsetQuarter"], [4, 1])
        self.assertNotIn("FONT SETTINGS", json.dumps(decoded))

    def test_grace_tie_links_without_inventing_performed_duration(self):
        root = musical_score()
        ET.SubElement(root.find("./Beats/Beat[@id='0']"), "GraceNotes").text = "OnBeat"
        root.find("./Voices/Voice[@id='0']/Beats").text = "0 1"
        decoded = decode_score(root, TUNING, 2)
        self.assertEqual(decoded["scoreEvents"][0]["advanceQuarter"], [0, 1])
        playback = performance_events(decoded, [0])
        self.assertIsNone(playback["noteEvents"][0]["durationQuarter"])
        self.assertFalse(playback["noteEvents"][1]["isAttack"])
        self.assertEqual(playback["noteEvents"][1]["tieFrom"], playback["noteEvents"][0]["id"])

    def test_playback_notes_are_time_ordered_across_overlapping_voices(self):
        root = musical_score()
        root.find("./Rhythms/Rhythm/NoteValue").text = "Half"
        root.find("./Voices/Voice[@id='0']/Beats").text = "0 1"
        root.find("Notes").append(note_xml("2", string=2))
        ET.SubElement(root.find("./Beats/Beat[@id='2']"), "Notes").text = "2"
        decoded = decode_score(root, TUNING, 2)
        playback = performance_events(decoded, [0])
        self.assertEqual([event["onsetQuarter"] for event in playback["noteEvents"]], [[0, 1], [0, 1], [2, 1]])

    def test_bad_references_fail_and_short_first_measure_is_explicit(self):
        root = musical_score()
        root.find("./Rhythms/Rhythm/NoteValue").text = "Quarter"
        decoded = decode_score(root, TUNING, 2)
        self.assertTrue(decoded["measures"][0]["inferredPickup"])
        self.assertEqual(decoded["measures"][0]["durationQuarter"], [1, 1])
        self.assertIn("underfull_measure", [issue["code"] for issue in decoded["issues"]])
        root.find("./Beats/Beat/Rhythm").set("ref", "absent")
        with self.assertRaisesRegex(EventExtractionError, "rhythm reference"):
            decode_score(root, TUNING, 2)

    def test_instruction_examples_remain_in_score_but_not_performance_events(self):
        root = musical_score()
        section = ET.SubElement(root.findall("./MasterBars/MasterBar")[1], "Section")
        ET.SubElement(section, "Text").text = "Instructions"
        decoded = decode_score(root, TUNING, 2)
        self.assertEqual(len(decoded["scoreEvents"]), 4)
        self.assertTrue(decoded["measures"][1]["referenceOnly"])
        order = playback_order(decoded["measures"])
        self.assertEqual(order, [0])
        self.assertEqual(len(performance_events(decoded, order)["noteEvents"]), 1)

    def test_extractor_preserves_gp_bytes_and_honors_revision_and_capo_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gp_root = root / "data" / "guitar-perf-gp" / "dataset-a"
            gp_root.mkdir(parents=True)
            path = gp_root / "example.gp"
            with ZipFile(path, "w") as archive:
                archive.writestr("Content/score.gpif", ET.tostring(musical_score()))
                archive.writestr("Content/Preferences.json", '{"keep":true}')
                archive.writestr("Content/Stylesheets/score.gpss", b"\0custom font\xff")
            before = path.read_bytes()
            entry = {"id": "example", "title": "Example", "localGpPath": str(path.relative_to(root)), "localAudioPath": None, "rangeSeconds": None, "selectedTuningIndex": 0, "tunings": [{"strings": ["E2", "A2", "D3", "G3", "B3", "E4"]}], "capoFret": None}
            audit = {"sha256": sha256(path)}
            with patch("scripts.extract_gp_events.ROOT", root), patch("scripts.extract_gp_events.GP_ROOT", gp_root):
                output = extract_entry(entry, audit, {})
                self.assertEqual(output["instrument"]["capoFret"], 2)
                self.assertIsNone(output["audioAlignment"])
                self.assertIsNone(entry["capoFret"])
                self.assertEqual(path.read_bytes(), before)
                entry["tempo"] = {"bpm": 61, "beatUnit": [3, 8]}
                with self.assertRaisesRegex(EventExtractionError, "tempo conflicts"):
                    extract_entry(entry, audit, {})
                del entry["tempo"]
                entry["capoFret"] = 3
                with self.assertRaisesRegex(EventExtractionError, "conflicts"):
                    extract_entry(entry, audit, {})
                with self.assertRaisesRegex(EventExtractionError, "revision"):
                    extract_entry(entry, {"sha256": "0" * 64}, {})

    def test_owner_confirmed_fine_preserves_source_and_excludes_instruction_measures(self):
        score = musical_score()
        measures = score.find("MasterBars")
        ET.SubElement(ET.SubElement(measures[0], "Directions"), "Target").text = "Segno"
        ET.SubElement(ET.SubElement(measures[1], "Directions"), "Jump").text = "DaSegnoAlFine"
        measures.append(ET.fromstring("<MasterBar><Time>4/4</Time><Bars>1</Bars><Section><Text>Instructions</Text></Section></MasterBar>"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "example.gp"
            with ZipFile(path, "w") as archive:
                archive.writestr("Content/score.gpif", ET.tostring(score))
            before = path.read_bytes()
            entry = {"id": "example", "title": "Example", "localGpPath": path.name, "localAudioPath": None, "rangeSeconds": None, "selectedTuningIndex": 0, "tunings": [{"strings": ["E2", "A2", "D3", "G3", "B3", "E4"]}], "capoFret": None}
            audit = {"sha256": sha256(path)}
            confirmation = {"gpSha256": audit["sha256"], "fineTarget": "last-musical-measure", "note": "Owner confirmed."}
            rules = {"navigationOverrides": {"example": confirmation}}
            with patch("scripts.extract_gp_events.ROOT", root), patch("scripts.extract_gp_events.GP_ROOT", root):
                self.assertIsNone(extract_entry(entry, audit, {})["playback"])
                output = extract_entry(entry, audit, rules)
                self.assertEqual([visit["measureIndex"] for visit in output["playback"]["measureVisits"]], [0, 1, 0, 1])
                self.assertEqual(output["navigationOverride"]["fineMeasureIndex"], 1)
                self.assertFalse(any(direction["value"] == "Fine" for measure in output["measures"] for direction in measure["directions"]))
                self.assertEqual(path.read_bytes(), before)
                confirmation["gpSha256"] = "0" * 64
                with self.assertRaisesRegex(EventExtractionError, "Navigation confirmation"):
                    extract_entry(entry, audit, rules)


class NavigationTests(unittest.TestCase):
    def test_explicit_fine_is_a_successful_destination(self):
        measures = [bar(targets=["Segno"]), bar(targets=["Fine"]), bar(jumps=["DaSegnoAlFine"])]
        self.assertEqual(playback_order(measures), [0, 1, 2, 0, 1])
        with self.assertRaisesRegex(EventExtractionError, "explicit Fine"):
            playback_order(measures, fine_measure_index=2)

    def test_confirmed_fine_must_be_a_musical_measure(self):
        measures = [bar(targets=["Segno"]), bar(jumps=["DaSegnoAlFine"]), dict(bar(), referenceOnly=True)]
        for index in (-1, 2, 3, True):
            with self.subTest(index=index), self.assertRaisesRegex(EventExtractionError, "musical measure"):
                playback_order(measures, fine_measure_index=index)

    def test_repeats_inherited_voltas_and_multiple_closings(self):
        self.assertEqual(playback_order([bar(start=True), bar(), bar(end=2, endings=[1]), bar(endings=[2]), bar()]), [0, 1, 2, 0, 1, 3, 4])
        self.assertEqual(playback_order([bar(start=True), bar(endings=[1]), bar(end=2), bar(endings=[2])]), [0, 1, 2, 0, 3])
        self.assertEqual(playback_order([bar(start=True), bar(end=4, endings=[1]), bar(end=4, endings=[2]), bar(end=4, endings=[3]), bar(endings=[4])]), [0, 1, 0, 2, 0, 3, 0, 4])
        self.assertEqual(playback_order([bar(start=True, end=3)]), [0, 0, 0])
        self.assertEqual(playback_order([bar(), bar(end=2)]), [0, 1, 0, 1])

    def test_dal_segno_al_coda_ignores_initial_to_coda(self):
        measures = [bar(targets=["Segno"]), bar(jumps=["DaCoda"]), bar(jumps=["DaSegnoAlCoda"]), bar(targets=["Coda"])]
        self.assertEqual(playback_order(measures), [0, 1, 2, 0, 1, 3])

    def test_sequential_dc_and_ds_on_shared_navigation_bars(self):
        measures = [bar(targets=["Segno"]), bar(jumps=["DaCapo", "DaCoda"]), bar(jumps=["DaSegnoAlCoda"]), bar(targets=["Coda"])]
        self.assertEqual(playback_order(measures), [0, 1, 0, 1, 2, 0, 1, 3])

    def test_missing_fine_or_coda_never_yields_success_shaped_order(self):
        for measures in ([bar(targets=["Segno"]), bar(jumps=["DaSegnoAlFine"])], [bar(targets=["Segno"]), bar(jumps=["DaCoda"]), bar(jumps=["DaSegnoAlCoda"])]):
            with self.assertRaises(EventExtractionError):
                playback_order(measures)


if __name__ == "__main__":
    unittest.main()
