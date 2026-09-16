from fractions import Fraction
import json
import unittest
import xml.etree.ElementTree as ET
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

    def test_explicit_tie_destination_does_not_require_redundant_origin_flag(self):
        root = musical_score()
        root.find("./Notes/Note[@id='0']/Tie").set("origin", "false")
        decoded = decode_score(root, TUNING, 2)
        playback = performance_events(decoded, playback_order(decoded["measures"]))
        self.assertFalse(decoded["scoreEvents"][0]["notes"][0]["tie"]["origin"])
        self.assertFalse(playback["noteEvents"][1]["isAttack"])
        self.assertEqual(playback["noteEvents"][1]["tieFrom"], playback["noteEvents"][0]["id"])
        self.assertNotIn("unresolved_tie_destination", [issue["code"] for issue in decoded["issues"]])

    def test_tie_destination_still_requires_matching_pitch_and_adjacent_score_time(self):
        for change in ("pitch", "gap"):
            root = musical_score()
            root.find("./Notes/Note[@id='0']/Tie").set("origin", "false")
            if change == "pitch":
                root.find("Notes").remove(root.find("./Notes/Note[@id='1']"))
                root.find("Notes").append(note_xml("1", fret=1, tie={"origin": "false", "destination": "true"}))
            else:
                root.find("Rhythms").append(ET.fromstring('<Rhythm id="1"><NoteValue>Half</NoteValue></Rhythm>'))
                root.find("./Beats/Beat[@id='0']/Rhythm").set("ref", "1")
            decoded = decode_score(root, TUNING, 2)
            playback = performance_events(decoded, playback_order(decoded["measures"]))
            with self.subTest(change=change):
                self.assertIsNone(playback["noteEvents"][1]["isAttack"])
                self.assertIsNone(playback["noteEvents"][1]["tieFrom"])

    def test_repeat_entry_tie_is_silent_without_its_original_first_pass_origin(self):
        decoded = decode_score(musical_score(), TUNING, 2)
        playback = performance_events(decoded, [0, 1, 1], silence_repeated_entry_ties=True)
        first, tied, returning = playback["noteEvents"]
        self.assertEqual(tied["tieFrom"], first["id"])
        self.assertFalse(returning["isAttack"])
        self.assertIsNone(returning["tieFrom"])
        self.assertTrue(returning["isSilent"])
        self.assertEqual(returning["durationQuarter"], [4, 1])
        self.assertEqual(playback["durationQuarter"], [12, 1])
        self.assertEqual(decoded["issues"], [])
        self.assertTrue(decoded["scoreEvents"][2]["notes"][0]["tie"]["destination"])

    def test_replayed_original_origin_keeps_its_real_tie(self):
        decoded = decode_score(musical_score(), TUNING, 2)
        playback = performance_events(decoded, [0, 1, 0, 1], silence_repeated_entry_ties=True)
        self.assertFalse(any(event.get("isSilent") for event in playback["noteEvents"]))
        self.assertEqual(playback["noteEvents"][3]["tieFrom"], playback["noteEvents"][2]["id"])

    def test_unknown_first_tie_is_not_silenced_by_an_unrelated_repeat(self):
        decoded = decode_score(musical_score(), TUNING, 2)
        playback = performance_events(decoded, [1, 1], silence_repeated_entry_ties=True)
        self.assertIsNone(playback["noteEvents"][0]["isAttack"])
        self.assertFalse(any(event.get("isSilent") for event in playback["noteEvents"]))

    def test_silenced_repeat_entry_keeps_its_following_tie_segments_silent(self):
        root = musical_score()
        root.find("./Notes/Note[@id='1']/Tie").set("origin", "true")
        root.find("Notes").append(note_xml("3", tie={"origin": "false", "destination": "true"}))
        root.find("Beats").append(ET.fromstring('<Beat id="3"><Rhythm ref="0"/><Notes>3</Notes></Beat>'))
        root.find("Voices").append(ET.fromstring('<Voice id="3"><Beats>3</Beats></Voice>'))
        root.find("Bars").append(ET.fromstring('<Bar id="2"><Voices>3 2 -1 -1</Voices></Bar>'))
        root.find("MasterBars").append(ET.fromstring('<MasterBar><Time>4/4</Time><Bars>2</Bars></MasterBar>'))
        decoded = decode_score(root, TUNING, 2)
        playback = performance_events(decoded, [0, 1, 2, 1, 2], silence_repeated_entry_ties=True)
        self.assertEqual([event.get("isSilent", False) for event in playback["noteEvents"]], [False, False, False, True, True])
        self.assertEqual(playback["noteEvents"][-1]["suppressionReason"], "continuation_of_silenced_repeat_entry")

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
        for label in ("Instructions", "\nIntructions\n", "LEGEND"):
            with self.subTest(label=label):
                root = musical_score()
                section = ET.SubElement(root.findall("./MasterBars/MasterBar")[1], "Section")
                ET.SubElement(section, "Text").text = label
                decoded = decode_score(root, TUNING, 2)
                self.assertEqual(len(decoded["scoreEvents"]), 4)
                self.assertTrue(decoded["measures"][1]["referenceOnly"])
                order = playback_order(decoded["measures"])
                self.assertEqual(order, [0])
                self.assertEqual(len(performance_events(decoded, order)["noteEvents"]), 1)

    def test_instruction_rest_anchors_preserve_text_positions_without_playback_silence(self):
        root = musical_score()
        section = ET.SubElement(root.findall("./MasterBars/MasterBar")[1], "Section")
        ET.SubElement(section, "Text").text = "Instructions"
        for identifier, value in (("1", "Quarter"), ("2", "Half")):
            rhythm = ET.SubElement(root.find("Rhythms"), "Rhythm", id=identifier)
            ET.SubElement(rhythm, "NoteValue").text = value
        labels = ("Thumb slap: X", "Wrist thump: O", "Technique explanation")
        for identifier, rhythm, label in zip(("3", "4", "5"), ("1", "1", "2"), labels):
            beat = ET.SubElement(root.find("Beats"), "Beat", id=identifier)
            ET.SubElement(beat, "Rhythm", ref=rhythm)
            ET.SubElement(beat, "FreeText").text = label
        root.find("./Voices/Voice[@id='1']/Beats").text = "3 4 5"
        decoded = decode_score(root, TUNING, 2)
        anchors = [beat for beat in decoded["scoreEvents"] if beat["referenceOnly"] and beat["voiceIndex"] == 0]
        self.assertTrue(all(beat["isRest"] for beat in anchors))
        self.assertEqual([beat["text"] for beat in anchors], list(labels))
        self.assertEqual([beat["offsetQuarter"] for beat in anchors], [[0, 1], [1, 1], [2, 1]])
        self.assertEqual([beat["notatedDurationQuarter"] for beat in anchors], [[1, 1], [1, 1], [2, 1]])
        playback = performance_events(decoded, playback_order(decoded["measures"]))
        self.assertEqual([visit["measureIndex"] for visit in playback["measureVisits"]], [0])
        self.assertEqual(playback["durationQuarter"], [4, 1])

    def test_additional_remarks_remain_reference_only_until_music_resumes(self):
        root = musical_score()
        measures = root.find("MasterBars")
        ET.SubElement(ET.SubElement(measures[1], "Section"), "Text").text = "Instructions"
        measures.append(ET.fromstring("<MasterBar><Time>4/4</Time><Bars>2</Bars><Section><Text>Additional Remarks</Text></Section></MasterBar>"))
        measures.append(ET.fromstring("<MasterBar><Time>4/4</Time><Bars>0</Bars><Section><Text>Chorus</Text></Section></MasterBar>"))
        ET.SubElement(ET.SubElement(root.find("Bars"), "Bar", id="2"), "Voices").text = "2 -1 -1 -1"
        ET.SubElement(root.find("./Beats/Beat[@id='2']"), "FreeText").text = "Sections differ; do not repeat them identically."
        decoded = decode_score(root, TUNING, 2)
        self.assertEqual([measure["referenceOnly"] for measure in decoded["measures"]], [False, True, True, False])
        remarks = [beat for beat in decoded["scoreEvents"] if beat["measureIndex"] == 2]
        self.assertEqual(len(remarks), 1)
        self.assertTrue(remarks[0]["isRest"])
        self.assertEqual(remarks[0]["text"], "Sections differ; do not repeat them identically.")
        order = playback_order(decoded["measures"])
        self.assertEqual(order, [0, 3])
        self.assertEqual(performance_events(decoded, order)["durationQuarter"], [8, 1])

    def test_owner_confirmed_fine_preserves_source_and_excludes_instruction_measures(self):
        score = musical_score()
        measures = score.find("MasterBars")
        ET.SubElement(ET.SubElement(measures[0], "Directions"), "Target").text = "Segno"
        ET.SubElement(ET.SubElement(measures[1], "Directions"), "Jump").text = "DaSegnoAlFine"
        measures.append(ET.fromstring("<MasterBar><Time>4/4</Time><Bars>1</Bars><Section><Text>Instructions</Text></Section></MasterBar>"))
        before = ET.tostring(score)
        decoded = decode_score(score, TUNING, 2)
        with self.assertRaises(EventExtractionError):
            playback_order(decoded["measures"])
        order = playback_order(decoded["measures"], fine_measure_index=1)
        self.assertEqual(order, [0, 1, 0, 1])
        self.assertEqual([visit["measureIndex"] for visit in performance_events(decoded, order)["measureVisits"]], order)
        self.assertEqual(ET.tostring(score), before)


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
