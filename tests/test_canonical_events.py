from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from scripts.canonical_events import CanonicalLabelError, canonical_counts, canonicalize, dead_note_marks, matches_rule
from scripts.catalogs import PerformerPaths, sha256, write_json
from scripts.gp_events import catalog_timing, decode_score, performance_events, playback_order
from scripts.prepare_labels import prepare_entry
from tests.test_gp_events import TUNING, musical_score, note_xml


def extracted(root=None, order=None):
    decoded = decode_score(root if root is not None else musical_score(), TUNING, 2)
    playback = performance_events(decoded, playback_order(decoded["measures"]) if order is None else order)
    return {
        "schemaVersion": 1, "catalogId": "example", "sourceGpSha256": "a" * 64,
        "audioPath": "audio.flac", "sourceRangeSeconds": None, "timeUnit": "quarter-note",
        "instrument": {"stringOrder": [6, 5, 4, 3, 2, 1], "openStringMidi": TUNING, "capoFret": 2, "fretConvention": "capo-relative"},
        "providedTiming": catalog_timing(decoded), **decoded, "playback": playback,
    }


def gesture_score():
    root = musical_score()
    for note in root.findall("./Notes/Note"):
        note.remove(note.find("Tie"))
    root.find("Notes").remove(root.find("./Notes/Note[@id='0']"))
    root.find("Notes").append(note_xml("0", dead=True))
    root.find("Notes").append(note_xml("2", string=1, dead=True))
    root.find("Notes").append(note_xml("3", string=3))
    root.find("./Beats/Beat[@id='0']/Notes").text = "0 2 3"
    root.find("./Beats/Beat[@id='0']/FreeText").text = ""
    root.find("./Beats/Beat[@id='1']/Notes").text = "0 2"
    ET.SubElement(root.find("./Beats/Beat[@id='1']"), "FreeText").text = "Slap strings with right hand"
    ET.SubElement(ET.SubElement(root.findall("./MasterBars/MasterBar")[1], "Section"), "Text").text = "Instructions"
    return extracted(root)


def gesture_rules(technique="right_hand_string_slap"):
    return {"gpSha256": "a" * 64, "rules": [{"id": "slap", "technique": technique, "evidenceBeatIds": ["m1:v0:b0"], "match": {"deadStrings": [5, 6], "text": ""}, "consumesDeadNotes": True}]}


class CanonicalEventTests(unittest.TestCase):
    def test_ties_become_one_note_without_mutating_the_source(self):
        score = extracted()
        before = deepcopy(score)
        result = canonicalize(score)
        self.assertEqual(score, before)
        self.assertEqual(len(result["targets"]["notes"]), 1)
        note = result["targets"]["notes"][0]
        self.assertEqual(note["notatedDurationQuarter"], [8, 1])
        self.assertTrue(note["isAttack"])
        self.assertEqual(len(note["sourceSegments"]), 2)
        self.assertEqual(canonical_counts(result)["mergedContinuationCount"], 1)
        self.assertEqual(len(result["targets"]["rests"]), 2)
        result["conditioning"]["instrument"]["openStringMidi"][0] = 0
        self.assertEqual(score, before)

    def test_equal_pitch_reattacks_and_repeat_occurrences_remain_distinct(self):
        root = musical_score()
        for note in root.findall("./Notes/Note"):
            note.remove(note.find("Tie"))
        labels = canonicalize(extracted(root, order=[0, 1, 0, 1]))
        notes = labels["targets"]["notes"]
        self.assertEqual(len(notes), 4)
        self.assertEqual(len({note["id"] for note in notes}), 4)
        self.assertEqual([note["onsetQuarter"] for note in notes], [[0, 1], [4, 1], [8, 1], [12, 1]])

    def test_overlapping_voice_durations_are_independent(self):
        root = musical_score()
        root.find("Notes").append(note_xml("2", string=2))
        root.find("./Beats/Beat[@id='2']").append(ET.fromstring("<Notes>2</Notes>"))
        labels = canonicalize(extracted(root))
        bass = next(note for note in labels["targets"]["notes"] if note["voiceIndex"] == 0)
        other = [note for note in labels["targets"]["notes"] if note["voiceIndex"] == 1]
        self.assertEqual(bass["notatedDurationQuarter"], [8, 1])
        self.assertEqual([note["notatedDurationQuarter"] for note in other], [[4, 1], [4, 1]])

    def test_unresolved_destination_is_not_an_attack_or_complete_duration(self):
        root = musical_score()
        root.find("./Notes/Note[@id='0']/Tie").set("origin", "false")
        root.find("Rhythms").append(ET.fromstring('<Rhythm id="1"><NoteValue>Half</NoteValue></Rhythm>'))
        root.find("./Beats/Beat[@id='0']/Rhythm").set("ref", "1")
        labels = canonicalize(extracted(root))
        unresolved = labels["targets"]["notes"][1]
        self.assertIsNone(unresolved["isAttack"])
        self.assertFalse(unresolved["labelMask"]["attack"])
        self.assertFalse(unresolved["labelMask"]["pitch"])
        self.assertFalse(unresolved["labelMask"]["fingering"])
        self.assertIsNone(unresolved["soundingPitchMidi"])
        self.assertIsNone(unresolved["notatedDurationQuarter"])
        self.assertEqual(unresolved["observedDurationQuarter"], [4, 1])

    def test_redundant_origin_without_destination_does_not_invent_missing_sustain(self):
        labels = canonicalize(extracted(order=[0]))
        note = labels["targets"]["notes"][0]
        self.assertTrue(note["isAttack"])
        self.assertTrue(note["labelMask"]["notatedDuration"])
        self.assertEqual(note["notatedDurationQuarter"], [4, 1])
        self.assertTrue(note["sourceSegments"][0]["tie"]["origin"])
        self.assertEqual(labels["review"]["issues"], [])

    def test_chain_merges_when_only_the_destination_carries_the_tie(self):
        root = musical_score()
        root.find("./Notes/Note[@id='0']/Tie").set("origin", "false")
        labels = canonicalize(extracted(root))
        self.assertEqual(len(labels["targets"]["notes"]), 1)
        self.assertEqual(labels["targets"]["notes"][0]["notatedDurationQuarter"], [8, 1])

    def test_grace_chain_retains_spelling_without_inventing_a_duration(self):
        root = musical_score()
        ET.SubElement(root.find("./Beats/Beat[@id='0']"), "GraceNotes").text = "OnBeat"
        root.find("./Voices/Voice[@id='0']/Beats").text = "0 1"
        labels = canonicalize(extracted(root, order=[0]))
        note = labels["targets"]["notes"][0]
        self.assertIsNone(note["notatedDurationQuarter"])
        self.assertEqual(note["observedDurationQuarter"], [4, 1])
        self.assertEqual(note["sourceSegments"][0]["graceMode"], "OnBeat")
        self.assertEqual(note["sourceSegments"][0]["notatedDurationQuarter"], [4, 1])

    def test_invalid_tie_links_and_duplicate_events_fail(self):
        score = extracted()
        for change in ("missing", "branch", "duplicate", "voice", "gap"):
            broken = deepcopy(score)
            notes = broken["playback"]["noteEvents"]
            if change == "missing":
                notes[1]["tieFrom"] = "absent"
            elif change == "branch":
                extra = deepcopy(notes[1])
                extra["id"] += ":branch"
                notes.append(extra)
            elif change == "duplicate":
                notes.append(deepcopy(notes[0]))
            elif change == "voice":
                broken["scoreEvents"][2]["voiceIndex"] = 3
            else:
                notes[1]["onsetQuarter"] = [5, 1]
            with self.subTest(change=change), self.assertRaises(CanonicalLabelError):
                canonicalize(broken)

    def test_missing_or_displaced_playback_notes_fail(self):
        for mode in ("missing", "visit", "onset"):
            score = extracted()
            if mode == "missing":
                score["playback"]["noteEvents"].pop()
            elif mode == "visit":
                score["playback"]["noteEvents"][0]["visitIndex"] = 9
            else:
                score["playback"]["noteEvents"][0]["onsetQuarter"] = [1, 1]
            with self.subTest(mode=mode), self.assertRaises(CanonicalLabelError):
                canonicalize(score)

    def test_one_symbolic_cluster_produces_one_gesture_and_keeps_pitched_notes(self):
        labels = canonicalize(gesture_score(), gesture_rules())
        self.assertEqual(len(labels["targets"]["notes"]), 1)
        self.assertEqual(len(labels["review"]["notationSymbols"]), 2)
        self.assertEqual(len(labels["targets"]["gestures"]), 1)
        self.assertEqual(len(labels["targets"]["gestures"][0]["symbolicNoteIds"]), 2)
        self.assertEqual(labels["review"]["unresolvedGestures"], [])
        self.assertTrue(all(not note["labelMask"]["fingering"] for note in labels["review"]["notationSymbols"]))
        self.assertNotIn("Slap strings with right hand", json.dumps(labels["conditioning"]))
        self.assertNotIn("Slap strings with right hand", json.dumps(labels["targets"]))

    def test_same_symbols_have_score_scoped_meanings_and_revision_guards(self):
        first = canonicalize(gesture_score(), gesture_rules())
        score = gesture_score()
        score["sourceGpSha256"] = "b" * 64
        rules = gesture_rules("muted_strum")
        with self.assertRaisesRegex(CanonicalLabelError, "revision"):
            canonicalize(score, rules)
        rules["gpSha256"] = "b" * 64
        second = canonicalize(score, rules)
        self.assertNotEqual(first["targets"]["gestures"][0]["technique"], second["targets"]["gestures"][0]["technique"])
        self.assertEqual(len(canonicalize(score)["review"]["unresolvedGestures"]), 1)

    def test_dead_note_slide_marks_disambiguate_identical_x_geometry(self):
        score = gesture_score()
        rules = gesture_rules()
        rules["rules"][0]["match"]["deadNoteMarks"] = deepcopy(dead_note_marks(score["scoreEvents"][0]))
        self.assertEqual(len(canonicalize(score, rules)["targets"]["gestures"]), 1)
        score["scoreEvents"][0]["notes"][0]["techniques"]["slideFlags"] = 2
        labels = canonicalize(score, rules)
        self.assertEqual(labels["targets"]["gestures"], [])
        self.assertEqual(labels["review"]["unresolvedGestures"][0]["reason"], "uninterpreted_dead_note_cluster")

    def test_a_reused_finger_letter_requires_its_local_note_context(self):
        beat = gesture_score()["scoreEvents"][0]
        beat["text"] = "i"
        rule = {"match": {"tokens": ["i"], "deadStrings": [5, 6], "beatTechniques": beat["techniques"]}}
        self.assertTrue(matches_rule(beat, rule))
        strum = deepcopy(beat)
        strum["notes"] = [note for note in strum["notes"] if not note["techniques"]["dead"]]
        strum["techniques"] = {"brush": "Up"}
        self.assertFalse(matches_rule(strum, rule))

    def test_competing_cluster_or_token_meanings_remain_unresolved(self):
        score = gesture_score()
        rules = gesture_rules()
        other = deepcopy(rules["rules"][0])
        other.update(id="other", technique="muted_strum")
        rules["rules"].append(other)
        labels = canonicalize(score, rules)
        self.assertEqual(labels["targets"]["gestures"], [])
        self.assertEqual(labels["review"]["unresolvedGestures"][0]["reason"], "ambiguous_gesture_rules")
        score["scoreEvents"][0]["text"] = "O"
        for rule in rules["rules"]:
            rule["match"] = {"tokens": ["O"]}
            rule["consumesDeadNotes"] = False
        self.assertEqual(canonicalize(score, rules)["targets"]["gestures"], [])

    def test_wrist_marker_coexists_with_notes_without_claiming_symbolic_strings(self):
        score = gesture_score()
        score["scoreEvents"][0]["text"] = "O"
        rules = gesture_rules("wrist_thump")
        rules["rules"][0]["match"] = {"tokens": ["O"]}
        rules["rules"][0]["consumesDeadNotes"] = False
        labels = canonicalize(score, rules)
        self.assertEqual(len(labels["targets"]["notes"]), 1)
        self.assertEqual(labels["targets"]["gestures"][0]["symbolicNoteIds"], [])
        self.assertEqual(labels["review"]["unresolvedGestures"][0]["reason"], "uninterpreted_dead_note_cluster")

    def test_independent_wrist_and_string_hits_share_a_beat_without_double_counting(self):
        score = gesture_score()
        score["scoreEvents"][0]["text"] = "O"
        rules = gesture_rules()
        rules["rules"][0]["match"]["text"] = "O"
        rules["rules"][0]["consumedTokens"] = []
        rules["rules"].append({"id": "wrist", "technique": "wrist_thump", "attributes": {"location": "above_soundhole"}, "evidenceBeatIds": ["m1:v0:b0"], "match": {"tokens": ["O"]}, "consumesDeadNotes": False})
        labels = canonicalize(score, rules)
        self.assertEqual(len(labels["targets"]["gestures"]), 2)
        self.assertEqual(labels["review"]["unresolvedGestures"], [])
        self.assertEqual(labels["targets"]["gestures"][1]["attributes"], {"location": "above_soundhole"})

    def test_repeated_markers_are_not_silently_collapsed_to_one_hit(self):
        rules = gesture_rules("wrist_thump")
        rules["rules"][0]["match"] = {"tokens": ["O"]}
        rules["rules"][0]["consumesDeadNotes"] = False
        for text in ("O O", "O explanation", "O L"):
            score = gesture_score()
            score["scoreEvents"][0]["text"] = text
            with self.subTest(text=text):
                labels = canonicalize(score, rules)
                self.assertEqual(labels["targets"]["gestures"], [])
                self.assertIn("uninterpreted_annotation", [item["reason"] for item in labels["review"]["unresolvedGestures"]])

    def test_explicit_combinations_can_produce_independent_gestures(self):
        score = gesture_score()
        score["scoreEvents"][0]["text"] = "O L"
        rules = gesture_rules("string_slap")
        rules["rules"][0]["match"] = {"tokens": ["O", "L"], "deadStrings": [5, 6]}
        rules["rules"][0]["consumedTokens"] = ["L"]
        rules["rules"].append({"id": "wrist", "technique": "wrist_thump", "evidenceBeatIds": ["m1:v0:b0"], "match": {"tokens": ["O", "L"]}, "consumedTokens": ["O"], "consumesDeadNotes": False})
        labels = canonicalize(score, rules)
        self.assertEqual(len(labels["targets"]["gestures"]), 2)
        self.assertEqual(labels["review"]["unresolvedGestures"], [])

    def test_conditioning_allowlist_excludes_legend_and_key_metadata(self):
        score = extracted()
        score["instrument"]["legendText"] = "Preparation-only secret"
        score["providedTiming"]["sourceKey"] = "Not an acoustic model input"
        conditioning = canonicalize(score)["conditioning"]
        self.assertNotIn("legendText", conditioning["instrument"])
        self.assertNotIn("sourceKey", conditioning["providedTiming"])
        score["instrument"]["fretConvention"] = "nut-relative"
        with self.assertRaisesRegex(CanonicalLabelError, "normalized frets"):
            canonicalize(score)

    def test_interpretation_evidence_must_be_reference_material(self):
        rules = gesture_rules()
        rules["rules"][0]["evidenceBeatIds"] = ["m0:v0:b0"]
        with self.assertRaisesRegex(CanonicalLabelError, "reference beat"):
            canonicalize(gesture_score(), rules)

    def test_malformed_or_overbroad_interpretation_rules_fail(self):
        for change in ({"match": {}}, {"match": {"deadStrings": []}}, {"match": {"beatTechniques": []}}, {"match": {"tokens": [""]}}, {"consumedTokens": ["unmatched"]}, {"attributes": []}, {"technique": []}):
            rules = gesture_rules()
            rules["rules"][0].update(change)
            with self.subTest(change=change), self.assertRaises(CanonicalLabelError):
                canonicalize(gesture_score(), rules)

    def test_reference_text_and_rests_never_become_targets(self):
        root = musical_score()
        ET.SubElement(ET.SubElement(root.findall("./MasterBars/MasterBar")[1], "Section"), "Text").text = "Instructions"
        root.find("./Beats/Beat[@id='1']").append(ET.fromstring("<FreeText>Private legend prose</FreeText>"))
        labels = canonicalize(extracted(root))
        self.assertEqual(len(labels["targets"]["rests"]), 1)
        self.assertNotIn("Private legend prose", json.dumps(labels["targets"]))
        self.assertNotIn("Private legend prose", json.dumps(labels["conditioning"]))
        self.assertEqual(len(labels["provenance"]["referenceBeatIds"]), 2)

    def test_duration_warning_blocks_score_timing_not_pitch_information(self):
        score = extracted()
        score["issues"].append({"code": "overfull_measure", "measureIndex": 0})
        labels = canonicalize(score)
        self.assertFalse(labels["scoreTimingResolved"])
        self.assertTrue(labels["targets"]["notes"][0]["labelMask"]["pitch"])
        self.assertFalse(labels["targets"]["notes"][0]["sourceSegments"][0]["techniqueMask"]["palmMuted"])

    def test_changing_tied_harmonic_pitch_preserves_segments_but_masks_constant_pitch(self):
        root = musical_score()
        root.find("Notes").remove(root.find("./Notes/Note[@id='1']"))
        root.find("Notes").append(note_xml("1", harmonic=("Natural", 12), tie={"origin": "false", "destination": "true"}))
        labels = canonicalize(extracted(root))
        note = labels["targets"]["notes"][0]
        self.assertEqual(len(labels["targets"]["notes"]), 1)
        self.assertIsNone(note["soundingPitchMidi"])
        self.assertFalse(note["labelMask"]["pitch"])
        self.assertEqual([segment["soundingPitchMidi"] for segment in note["sourceSegments"]], [42, 54])
        self.assertEqual(labels["review"]["issues"][0]["code"], "changing_tied_pitch")

    def test_preparation_binds_hashes_and_never_alters_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = PerformerPaths("dataset-a", Path(directory))
            paths.gp.mkdir(parents=True)
            paths.audio.mkdir(parents=True)
            gp = paths.gp / "example.gp"
            gp.write_bytes(b"immutable GP archive stand-in")
            audio = paths.audio / "example.flac"
            audio.write_bytes(b"immutable audio stand-in")
            events = paths.gp / "events" / "example.json"
            score = extracted()
            score["sourceGpSha256"] = sha256(gp)
            score["audioPath"] = str(audio.relative_to(paths.root))
            write_json(events, score)
            entry = {
                "id": "example", "localAudioPath": score["audioPath"], "localGpPath": str(gp.relative_to(paths.root)),
                "rangeSeconds": None, "selectedTuningIndex": 0, "tunings": [{"strings": ["E2", "A2", "D3", "G3", "B3", "E4"]}],
                "capoFret": 2, **score["providedTiming"],
                "audioAsset": {"sha256": sha256(audio), "appliedRangeSeconds": None},
                "gpExtraction": {"eventPath": str(events.relative_to(paths.root)), "eventSha256": sha256(events), "sourceGpSha256": sha256(gp)},
            }
            audit = {"sha256": sha256(gp), "localGpPath": entry["localGpPath"]}
            before = [path.read_bytes() for path in (gp, audio, events)]
            prepared = prepare_entry(entry, audit, None, paths)
            self.assertEqual(prepared["provenance"]["eventSha256"], sha256(events))
            self.assertEqual([path.read_bytes() for path in (gp, audio, events)], before)
            entry["rangeSeconds"] = [0, 1]
            with self.assertRaisesRegex(CanonicalLabelError, "bounds"):
                prepare_entry(entry, audit, None, paths)
            entry["rangeSeconds"] = None
            audio.write_bytes(b"changed")
            with self.assertRaisesRegex(CanonicalLabelError, "checksum"):
                prepare_entry(entry, audit, None, paths)


if __name__ == "__main__":
    unittest.main()
