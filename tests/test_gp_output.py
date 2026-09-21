from io import BytesIO
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib
import math
import unittest
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from scripts.dataset_io import ROOT
from scripts.dataset_io import publish_json, read_json
from scripts.gp_events import decode_score, rhythm_duration
from scripts.gp_normalization import GPIF_ENTRY
from scripts.gp_output import GRID, TempoMap, _rhythmic_positions, _split_duration, _spell_interval, write_gp_outputs
from scripts.draft_cleanup import DraftProfile
from scripts import transcriber
from scripts.transcriber_audio import HarnessError
from tests.test_gp_normalization import archive_bytes, template_score


def hypotheses():
    return {
        "schemaVersion": 1,
        "kind": "fingerstyle-transcription-hypotheses",
        "audioSha256": "audio",
        "audioDurationSeconds": 5.0,
        "metadata": {
            "openStringMidi": [40, 45, 50, 55, 59, 64],
            "capoFret": 0,
            "tempo": {"bpm": 120, "beatUnit": [1, 4]},
            "timeSignature": [4, 4],
            "tempoChanges": [],
            "timeSignatureChanges": [],
        },
        "notes": [
            {
                "onsetSeconds": 0.5,
                "string": 6,
                "fret": 2,
                "soundingPitchMidi": 43,
                "voiceIndex": 1,
                "notatedDurationQuarter": 5,
                "harmonic": None,
                "confidence": 0.9,
                "uncertainty": ["sounding_pitch_fret_mismatch"],
            },
            {
                "onsetSeconds": 1.0,
                "string": 1,
                "fret": 12,
                "soundingPitchMidi": 76,
                "voiceIndex": 0,
                "notatedDurationQuarter": 0.5,
                "harmonic": {"type": "Natural", "fret": 12, "confidence": 0.8},
                "confidence": 0.95,
                "uncertainty": ["harmonic_presence_uncalibrated"],
            },
        ],
        "percussion": [
            {"onsetSeconds": 1.5, "technique": "wrist_thump", "confidence": 0.8},
            {"onsetSeconds": 2.0, "technique": "thumb_slap", "confidence": 0.9},
            {"onsetSeconds": 2.5, "technique": "percussive_hit", "confidence": 0.7},
        ],
    }


def output_template():
    root = template_score()
    track = root.find("./Tracks/Track")
    if track.find("Transpose") is None:
        transpose = ET.SubElement(track, "Transpose")
        ET.SubElement(transpose, "Chromatic").text = "0"
        ET.SubElement(transpose, "Octave").text = "-1"
    return root


def gp_root(path):
    with ZipFile(path) as archive:
        return ET.fromstring(archive.read(GPIF_ENTRY))


class GpOutputTests(unittest.TestCase):
    def test_fingerstyle_export_does_not_write_double_dots_or_dotted_sixteenths(self):
        document = hypotheses()
        document["notes"] = [{**document["notes"][0], "onsetSeconds": time, "notatedDurationQuarter": .37,
                              "string": 1, "fret": 0, "soundingPitchMidi": 64, "voiceIndex": 0, "confidence": .99, "uncertainty": []}
                             for time in (0., .31, .57, .81, 1.31)]
        document["percussion"] = []
        beats = {"kind": "audio-beat-evidence", "audioSha256": "audio",
                 "beatSeconds": [i / 2 for i in range(11)], "downbeatSeconds": [0., 2., 4.]}
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            template, full, single = directory / "template.gpt", directory / "full.gp", directory / "single.gp"
            template.write_bytes(archive_bytes(output_template()))
            with self.assertRaisesRegex(HarnessError, "requires beat evidence"):
                write_gp_outputs(template, document, full, single, profile=DraftProfile(rhythm_policy="fingerstyle"))
            write_gp_outputs(template, document, full, single, profile=DraftProfile(rhythm_policy="fingerstyle"), beat_evidence=beats)
            for path in (full, single):
                root = gp_root(path)
                for rhythm in root.findall("./Rhythms/Rhythm"):
                    dots = rhythm.find("AugmentationDot")
                    self.assertFalse(dots is not None and (int(dots.get("count")) > 1 or rhythm.findtext("NoteValue") in ("16th", "32nd")))
                    self.assertIsNone(rhythm.find("PrimaryTuplet"))
                    self.assertNotIn(rhythm.findtext("NoteValue"), ("32nd", "64th", "128th"))

    def test_strict_note_threshold_also_applies_to_optional_symbolic_completion(self):
        document = hypotheses()
        document["notes"] = [{**document["notes"][0], "confidence": .99}]
        profile = DraftProfile(note_threshold=.98, strict_note_confidence=True)
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            template = directory / "template.gpt"
            template.write_bytes(archive_bytes(output_template()))
            def complete(model, source, **kwargs):
                self.assertEqual(kwargs["threshold"], .98)
                return source, {"addedCount": 0, "replacedCount": 0, "removedCount": 0}
            with patch("scripts.symbolic_completer.complete_document", side_effect=complete) as completion:
                write_gp_outputs(template, document, directory / "full.gp", directory / "single.gp", profile=profile, completer=object())
            completion.assert_called_once()

    def test_v4_grace_slide_has_zero_score_advance_and_terminal_slides_stay_local(self):
        document = hypotheses()
        document["notes"] = [{
            **document["notes"][0], "noteId": "anchor", "techniqueSchemaVersion": 4,
            "onsetSeconds": 0., "string": 1, "fret": 2, "soundingPitchMidi": 66,
            "voiceIndex": 0, "notatedDurationQuarter": 1., "harmonic": None,
            "confidence": .99, "connection": "none", "connectionOriginNoteId": None,
            "noteTechniques": {"slide_out_down": .99, "slide_in_below": .99},
            "grace": {
                "anchorNoteId": "anchor", "confidence": .99,
                "sourceFret": 1, "sourcePitchMidi": 65, "intervalSemitones": 1,
                "mode": "OnBeat", "transition": "slide_2",
                "fretConfidence": .99, "modeConfidence": .99, "transitionConfidence": .99,
                "onsetSeconds": None, "timingKnown": False,
            },
        }]
        document["percussion"] = []
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            template, full, single = directory / "template.gpt", directory / "full.gp", directory / "single.gp"
            template.write_bytes(archive_bytes(output_template()))
            write_gp_outputs(template, document, full, single)
            for path in (full, single):
                root = gp_root(path)
                decoded = decode_score(root, document["metadata"]["openStringMidi"], 0)
                grace = [beat for beat in decoded["scoreEvents"] if beat["graceMode"]]
                self.assertEqual(len(grace), 1)
                self.assertEqual(grace[0]["advanceQuarter"], [0, 1])
                self.assertEqual(grace[0]["notes"][0]["soundingPitchMidi"], 65)
                self.assertEqual(grace[0]["notes"][0]["techniques"]["slideFlags"], 2)
                main = [beat for beat in decoded["scoreEvents"] if beat["graceMode"] is None and beat["notes"]]
                self.assertEqual(main[0]["notes"][0]["techniques"]["slideFlags"], 20)

    def test_v4_filtered_origin_is_never_rebound_to_remaining_note(self):
        document = hypotheses()
        document["percussion"] = []
        document["notes"] = [{
            **document["notes"][0], "noteId": name, "techniqueSchemaVersion": 4,
            "onsetSeconds": seconds, "string": 1, "fret": pitch - 64, "soundingPitchMidi": pitch,
            "voiceIndex": 0, "notatedDurationQuarter": 1., "harmonic": None,
            "confidence": confidence, "connection": "hammer_on" if name == "destination" else "none",
            "connectionConfidence": .99, "connectionOriginNoteId": "filtered-origin" if name == "destination" else None,
        } for name, seconds, pitch, confidence in (
            ("unrelated", 0., 64, .99), ("filtered-origin", .5, 65, .2), ("destination", 1., 67, .99),
        )]
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            template, full, single = directory / "template.gpt", directory / "full.gp", directory / "single.gp"
            template.write_bytes(archive_bytes(output_template()))
            report = write_gp_outputs(template, document, full, single)
            self.assertFalse(gp_root(single).findall("./Notes/Note/Properties/Property[@name='HopoDestination']"))
            self.assertTrue(report["singleVoice"]["connectionFallbacks"])

    def test_paired_32nds_keep_distinct_attacks_without_inventing_brushes(self):
        document = hypotheses()
        document["notes"] = [{
            **document["notes"][0], "onsetSeconds": onset, "string": 1, "fret": 0,
            "soundingPitchMidi": 64, "voiceIndex": 0, "harmonic": None,
            "confidence": .99, "notatedDurationQuarter": .125, "uncertainty": [],
        } for onset in (1.875, 1.9375, 2.)]
        document["percussion"] = []
        document["techniques"] = [{
            "onsetSeconds": 1.875, "technique": "brush", "direction": "Down",
            "strings": [1], "confidence": .99,
            "stringMembershipConfidence": {str(i): .99 if i == 1 else .01 for i in range(1, 7)},
        }]
        beat_evidence = {
            "kind": "audio-beat-evidence", "audioSha256": "audio",
            "beatSeconds": [i / 2 for i in range(11)], "downbeatSeconds": [0., 2., 4.],
        }
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            template, full, single = directory / "template.gpt", directory / "full.gp", directory / "single.gp"
            template.write_bytes(archive_bytes(output_template()))
            write_gp_outputs(template, document, full, single, beat_evidence=beat_evidence)
            for path in (full, single):
                root = gp_root(path)
                score = decode_score(root, document["metadata"]["openStringMidi"], 0)
                positions = [
                    Fraction(*beat["scoreOnsetQuarter"]) for beat in score["scoreEvents"]
                    for note in beat["notes"] if not note["tie"]["destination"]
                ]
                self.assertEqual(positions, [Fraction(15, 4), Fraction(31, 8), Fraction(4)])
                self.assertEqual(len(root.findall("./Beats/Beat/Properties/Property[@name='Brush']")), 1)

    def test_three_downstrokes_do_not_trigger_native_rasgueado_or_extra_attacks(self):
        document = hypotheses()
        onsets = (1.875, 1.9375, 2.)
        document["notes"] = [{
            **document["notes"][0], "onsetSeconds": onset, "string": 1, "fret": 0,
            "soundingPitchMidi": 64, "voiceIndex": 0, "harmonic": None,
            "confidence": .99, "notatedDurationQuarter": .125, "uncertainty": [],
        } for onset in onsets]
        document["percussion"] = []
        document["techniques"] = [{
            "onsetSeconds": onset, "technique": "brush", "direction": "Down",
            "strings": [1], "confidence": 1.,
            "stringMembershipConfidence": {str(i): float(i == 1) for i in range(1, 7)},
        } for onset in onsets]
        document["techniques"].append({**document["techniques"][0], "technique": "rasgueado"})
        document["techniques"].append(dict(document["techniques"][0]))
        beats = {"kind": "audio-beat-evidence", "audioSha256": "audio",
                 "beatSeconds": [i / 2 for i in range(11)], "downbeatSeconds": [0., 2., 4.]}
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            template, full, single = directory / "template.gpt", directory / "full.gp", directory / "single.gp"
            template.write_bytes(archive_bytes(output_template()))
            write_gp_outputs(template, document, full, single, beat_evidence=beats)
            for path in (full, single):
                root = gp_root(path)
                self.assertFalse(root.findall(".//Property[@name='Rasgueado']"))
                self.assertEqual([p.findtext("Direction") for p in root.findall("./Beats/Beat/Properties/Property[@name='Brush']")], ["Down"] * 3)
                self.assertEqual(len(root.findall("./Beats/Beat/XProperties/XProperty[@id='687935489']")), 3)
                score = decode_score(root, document["metadata"]["openStringMidi"], 0)
                attacks = [Fraction(*b["scoreOnsetQuarter"]) for b in score["scoreEvents"]
                           for n in b["notes"] if not n["tie"]["destination"]]
                self.assertEqual(attacks, [Fraction(15, 4), Fraction(31, 8), Fraction(4)])

    def test_rhythm_spelling_preserves_dots_triplets_and_offbeat_boundaries(self):
        self.assertEqual(_split_duration(Fraction(3, 2)), [(Fraction(3, 2), ("Quarter", 1, None))])
        self.assertEqual(_split_duration(Fraction(1, 3)), [(Fraction(1, 3), ("Eighth", 0, (3, 2)))])
        simple = {"start": Fraction(0), "meter": [4, 4]}
        spelled = _spell_interval(Fraction(5, 2), Fraction(7, 2), simple)
        self.assertEqual(spelled, [(Fraction(1, 2), ("Eighth", 0, None))] * 2)
        compound = {"start": Fraction(0), "meter": [6, 8]}
        self.assertEqual(_spell_interval(Fraction(0), Fraction(3, 2), compound), [(Fraction(3, 2), ("Quarter", 1, None))])
        self.assertEqual(_spell_interval(Fraction(3, 2), Fraction(3), compound), [(Fraction(3, 2), ("Quarter", 1, None))])
        self.assertEqual(_spell_interval(Fraction(2, 3), Fraction(1), compound), [(Fraction(1, 3), ("Eighth", 0, (3, 2)))])
        triplet = _spell_interval(Fraction(1, 3), Fraction(3, 2), simple)
        self.assertEqual(triplet, [(Fraction(2, 3), ("Quarter", 0, (3, 2))), (Fraction(1, 2), ("Eighth", 0, None))])
        self.assertEqual(_spell_interval(Fraction(3, 4), Fraction(5, 4), simple),
                         [(Fraction(1, 4), ("16th", 0, None))] * 2)
        with self.assertRaisesRegex(HarnessError, "cannot be spelled"):
            _split_duration(Fraction(1, 5))

    def test_metrical_ties_preserve_note_endpoints_and_do_not_split_plain_quavers(self):
        for onset, duration, expected in (
            (Fraction(0), Fraction(1, 2), [Fraction(1, 2)]),
            (Fraction(5, 2), Fraction(1), [Fraction(1, 2), Fraction(1, 2)]),
            (Fraction(3, 4), Fraction(1, 2), [Fraction(1, 4), Fraction(1, 4)]),
            (Fraction(0), Fraction(3, 2), [Fraction(3, 2)]),
        ):
            with self.subTest(onset=onset, duration=duration), TemporaryDirectory(dir=ROOT) as directory:
                directory = Path(directory)
                document = hypotheses()
                document["notes"] = [{
                    **document["notes"][0], "onsetSeconds": float(onset) / 2, "notatedDurationQuarter": float(duration),
                    "string": 1, "fret": 0, "soundingPitchMidi": 64, "voiceIndex": 0, "confidence": .99, "uncertainty": [],
                }]
                document["percussion"] = []
                template, full, single = directory / "template.gpt", directory / "full.gp", directory / "single.gp"
                template.write_bytes(archive_bytes(output_template()))
                write_gp_outputs(template, document, full, single)
                for path in (full, single):
                    score = decode_score(gp_root(path), document["metadata"]["openStringMidi"], 0)
                    fragments = [(b, n) for b in score["scoreEvents"] for n in b["notes"]]
                    self.assertEqual([Fraction(*b["advanceQuarter"]) for b, _ in fragments], expected)
                    self.assertEqual(Fraction(*fragments[0][0]["scoreOnsetQuarter"]), onset)
                    self.assertEqual(sum(expected), duration)
                    self.assertEqual([bool(n["tie"]["destination"]) for _, n in fragments], [False] + [True] * (len(expected) - 1))
                    self.assertFalse([i for i in score["issues"] if i["code"] in ("underfull_measure", "overfull_measure")])

    def test_unattached_technique_cannot_fragment_a_sustained_note(self):
        document = hypotheses()
        document["notes"] = [{
            **document["notes"][0], "onsetSeconds": 0., "notatedDurationQuarter": .5,
            "string": 1, "fret": 0, "soundingPitchMidi": 64, "voiceIndex": 0, "confidence": .99, "uncertainty": [],
        }]
        document["percussion"] = []
        document["techniques"] = [{
            "onsetSeconds": .1875, "technique": "brush", "direction": "Down", "confidence": .99, "strings": [1],
            "stringMembershipConfidence": {str(i): float(i == 1) for i in range(1, 7)},
        }]
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            template, full, single = directory / "template.gpt", directory / "full.gp", directory / "single.gp"
            template.write_bytes(archive_bytes(output_template()))
            report = write_gp_outputs(template, document, full, single)
            self.assertEqual(report["techniqueAttachment"][0]["reason"], "no_nearby_pitched_attack")
            for path in (full, single):
                root = gp_root(path)
                self.assertEqual(len(root.findall("./Notes/Note")), 1)
                self.assertFalse(root.findall("./Notes/Note/Tie"))
                self.assertFalse(root.findall("./Beats/Beat/Properties/Property[@name='Brush']"))

    def test_final_supported_beat_keeps_its_attack_with_a_spellable_duration(self):
        document = hypotheses()
        document["audioDurationSeconds"] = 2.1
        document["notes"] = [{
            **document["notes"][0], "onsetSeconds": seconds, "notatedDurationQuarter": .5,
            "string": 1, "fret": 0, "soundingPitchMidi": 64, "voiceIndex": 0, "confidence": .99, "uncertainty": [],
        } for seconds in (0., 2.)]
        document["percussion"] = []
        beats = {"kind": "audio-beat-evidence", "audioSha256": "audio",
                 "beatSeconds": [0., .5, 1., 1.5, 2.], "downbeatSeconds": [0., 2.]}
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            template, full, single = directory / "template.gpt", directory / "full.gp", directory / "single.gp"
            template.write_bytes(archive_bytes(output_template()))
            write_gp_outputs(template, document, full, single, beat_evidence=beats)
            for path in (full, single):
                score = decode_score(gp_root(path), document["metadata"]["openStringMidi"], 0)
                attacks = [b for b in score["scoreEvents"] if any(not n["tie"]["destination"] for n in b["notes"])]
                self.assertEqual([b["scoreOnsetQuarter"] for b in attacks], [[0, 1], [4, 1]])
                self.assertEqual(attacks[-1]["advanceQuarter"], [1, 8])

    def test_percussion_rest_carrier_uses_the_triplet_grid_instead_of_a_binary_fragment(self):
        document = hypotheses()
        document["notes"] = [{
            **document["notes"][0], "onsetSeconds": float(q) / 2,
            "scoreOnsetQuarter": [q.numerator, q.denominator], "scoreDurationQuarter": [1, 3],
            "string": 1, "fret": 0, "soundingPitchMidi": 64, "voiceIndex": 0, "confidence": .99, "uncertainty": [],
        } for q in (Fraction(1, 3), Fraction(2, 3))]
        document["scoreAudioEndQuarter"] = [4, 1]
        document["percussion"] = [{"onsetSeconds": 0., "scoreOnsetQuarter": [0, 1], "technique": "thumb_slap", "confidence": .99}]
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            template, full, single = directory / "template.gpt", directory / "full.gp", directory / "single.gp"
            template.write_bytes(archive_bytes(output_template()))
            write_gp_outputs(template, document, full, single)
            for path in (full, single):
                root = gp_root(path)
                self.assertFalse([rhythm for rhythm in root.findall("./Rhythms/Rhythm") if rhythm.findtext("NoteValue") in ("64th", "128th")])
                decoded = decode_score(root, document["metadata"]["openStringMidi"], 0)
                self.assertFalse([i for i in decoded["issues"] if i["code"] in ("underfull_measure", "overfull_measure")])

    def test_dense_downstroke_normalization_reaches_both_gp_outputs(self):
        from tests.test_stroke_normalization import burst

        document = {**hypotheses(), **burst()}
        beats = {"kind": "audio-beat-evidence", "audioSha256": "audio",
                 "beatSeconds": [i / 2 for i in range(11)], "downbeatSeconds": [0., 2., 4.]}
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            template, full, single = directory / "template.gpt", directory / "full.gp", directory / "single.gp"
            template.write_bytes(archive_bytes(output_template()))
            report = write_gp_outputs(template, document, full, single, beat_evidence=beats)
            self.assertEqual(report["strokeNormalization"]["changedGroupCount"], 1)
            for path in (full, single):
                root = gp_root(path)
                score = decode_score(root, document["metadata"]["openStringMidi"], 0)
                attacks = [Fraction(*b["scoreOnsetQuarter"]) for b in score["scoreEvents"]
                           for n in b["notes"] if not n["tie"]["destination"]]
                self.assertEqual(sorted(attacks), sorted([Fraction(15, 4), Fraction(31, 8), Fraction(4)] * 2))
                self.assertEqual([b.findtext("FreeText") for b in root.findall("./Beats/Beat") if b.findtext("FreeText")], ["a", "m", "i"])
                self.assertFalse(root.findall(".//Property[@name='Rasgueado']"))
                self.assertEqual(len(root.findall("./Beats/Beat/Properties/Property[@name='Brush']")), 3)

    def test_tempo_map_integrates_constant_and_linear_tempo(self):
        mapping = TempoMap({
            "tempo": {"bpm": 60, "beatUnit": [1, 4], "linear": True},
            "tempoChanges": [{"timeSeconds": 2, "bpm": 120, "beatUnit": [1, 4]}],
        })
        self.assertAlmostEqual(float(mapping.quarter_at(1)), 1.1951677046)
        self.assertAlmostEqual(float(mapping.quarter_at(2)), 2 / math.log(2))
        self.assertAlmostEqual(float(mapping.quarter_at(3)), 2 / math.log(2) + 2)

    def test_beat_relative_grid_prefers_sixteenths_but_retains_needed_thirty_seconds(self):
        positions, counts = _rhythmic_positions([Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)])
        self.assertEqual({value[1] for value in positions.values()}, {Fraction(1, 4)})
        self.assertEqual(counts, {"sixteenth": 4})
        positions, counts = _rhythmic_positions([Fraction(0), Fraction(1, 8), Fraction(1, 4)])
        self.assertEqual({value[1] for value in positions.values()}, {Fraction(1, 8)})
        self.assertEqual(counts, {"thirty-second": 3})

    def test_template_bar_scaffold_is_rebuilt_for_full_audio_duration(self):
        root = output_template()
        template = archive_bytes(root)
        non_gpif = {}
        with ZipFile(BytesIO(template)) as archive:
            non_gpif = {item.filename: archive.read(item) for item in archive.infolist() if item.filename != GPIF_ENTRY}
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            template_path = directory / "template.gpt"
            full = directory / "result.full-voices.gp"
            single = directory / "result.single-voice.gp"
            template_path.write_bytes(template)
            before = hashlib.sha256(template_path.read_bytes()).hexdigest()
            report = write_gp_outputs(template_path, hypotheses(), full, single)
            self.assertEqual(hashlib.sha256(template_path.read_bytes()).hexdigest(), before)
            self.assertEqual(report["fullVoices"]["measureCount"], 3)
            self.assertEqual(report["singleVoice"]["measureCount"], 3)
            self.assertFalse(report["pitchFretReconciliations"])
            self.assertEqual(report["fingeringOptimization"]["changes"][0]["selectedFret"], 3)
            for output in (full, single):
                with ZipFile(output) as archive:
                    self.assertEqual(
                        {item.filename: archive.read(item) for item in archive.infolist() if item.filename != GPIF_ENTRY},
                        non_gpif,
                    )
                generated = gp_root(output)
                self.assertEqual(len(generated.findall("./MasterBars/MasterBar")), 3)
                self.assertEqual(len(generated.findall("./Bars/Bar")), 3)
                decoded = decode_score(generated, [40, 45, 50, 55, 59, 64], 0)
                self.assertFalse([issue for issue in decoded["issues"] if issue["code"] in ("underfull_measure", "overfull_measure")])
                self.assertFalse(any(master.find("Repeat") is not None or master.find("Directions") is not None for master in generated.findall("./MasterBars/MasterBar")))
            full_root = gp_root(full)
            single_root = gp_root(single)
            self.assertTrue(any(voice != "-1" for bar in full_root.findall("./Bars/Bar") for voice in bar.findtext("Voices").split()[1:]))
            self.assertTrue(all(bar.findtext("Voices").split()[1:] == ["-1", "-1", "-1"] for bar in single_root.findall("./Bars/Bar")))
            ties = full_root.findall("./Notes/Note/Tie")
            self.assertTrue(any(tie.get("destination") == "true" for tie in ties))
            bass = [
                note for note in full_root.findall("./Notes/Note")
                if note.findtext("./Properties/Property[@name='String']/String") == "0"
            ]
            self.assertEqual(len(bass), 3)
            self.assertEqual(sum(n.find("Tie").get("destination") != "true" for n in bass), 1)
            empty_voice = single_root.find("./Voices/Voice[Beats]")
            rhythm_ids = empty_voice.findtext("Beats").split()
            self.assertLess(len(rhythm_ids), 12)
            self.assertEqual({beat.findtext("FreeText") for beat in full_root.findall("./Beats/Beat") if beat.findtext("FreeText")}, {"O"})
            dead = [note for note in full_root.findall("./Notes/Note") if note.find("./Properties/Property[@name='Muted']/Enable") is not None]
            self.assertEqual(len(dead), 2)
            self.assertEqual(sum(note.findtext("AntiAccent") == "Normal" for note in dead), 1)
            self.assertEqual(report["finestCandidateQuarterGrid"], [GRID.numerator, GRID.denominator])
            self.assertFalse(full_root.findall("./Rhythms/Rhythm/PrimaryTuplet"))
            self.assertEqual(report["draftCleanup"]["profile"]["note_threshold"], 0.9)
            rhythms = {node.get("id"): node for node in full_root.findall("./Rhythms/Rhythm")}
            beats = {node.get("id"): node for node in full_root.findall("./Beats/Beat")}
            dead_ids = {note.get("id") for note in dead}
            carrier_beats = [
                beat for beat in beats.values()
                if dead_ids & set(beat.findtext("Notes", "").split())
            ]
            self.assertEqual([rhythm_duration(rhythms[beat.find("Rhythm").get("ref")])[0] for beat in carrier_beats], [Fraction(1, 4), Fraction(1, 4)])

    def test_consistency_gated_harmonic_export_remains_available_explicitly(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            template = directory / "template.gpt"
            full, single = directory / "full.gp", directory / "single.gp"
            template.write_bytes(archive_bytes(output_template()))
            write_gp_outputs(
                template,
                hypotheses(),
                full,
                single,
                profile=DraftProfile(note_threshold=.5, include_harmonics=True, harmonic_threshold=.5),
            )
            self.assertTrue(gp_root(full).findall("./Notes/Note/Properties/Property[@name='Harmonic']"))

    def test_acoustic_technique_hypotheses_serialize_native_gp_marks(self):
        document = hypotheses()
        document["techniques"] = [
            {
                "onsetSeconds": .5, "technique": "brush", "direction": "Down",
                "strings": [1, 2, 6], "stringMembershipConfidence": {str(string): 1. for string in range(1, 7)},
                "confidence": .95,
            },
            {
                "onsetSeconds": 1., "technique": "arpeggio", "direction": "Up",
                "strings": [1, 2], "stringMembershipConfidence": {str(string): 1. for string in range(1, 7)},
                "confidence": .95,
            },
        ]
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            template = directory / "template.gpt"
            full, single = directory / "full.gp", directory / "single.gp"
            template.write_bytes(archive_bytes(output_template()))
            report = write_gp_outputs(template, document, full, single)
            root = gp_root(full)
            self.assertEqual(root.findtext("./Beats/Beat/Properties/Property[@name='Brush']/Direction"), "Down")
            brush = next(
                beat for beat in root.findall("./Beats/Beat")
                if beat.find("./Properties/Property[@name='Brush']") is not None
            )
            self.assertEqual(brush.findtext("./XProperties/XProperty[@id='687935489']/Int"), "0")
            self.assertEqual(brush.findtext("./XProperties/XProperty[@id='687935490']/Float"), "0")
            self.assertEqual(root.findtext("./Beats/Beat/Arpeggio"), "Up")
            self.assertEqual(report["techniqueHypotheses"], 2)

    def test_note_connections_and_supported_note_techniques_serialize_natively(self):
        document = hypotheses()
        document["notes"] = [
            {
                **document["notes"][0],
                "onsetSeconds": 0,
                "string": 1,
                "fret": 0,
                "soundingPitchMidi": 64,
                "notatedDurationQuarter": .5,
                "connection": "none",
                "noteTechniques": {},
            },
            {
                **document["notes"][0],
                "onsetSeconds": .5,
                "string": 1,
                "fret": 2,
                "soundingPitchMidi": 66,
                "notatedDurationQuarter": .5,
                "connection": "hammer_on",
                "connectionConfidence": .95,
                "noteTechniques": {"left_hand_tap": .9, "vibrato": .8, "bend": .9},
                "bendCurve": {
                    "OriginOffset": 0, "OriginValue": 0, "MiddleOffset1": 12,
                    "MiddleOffset2": 12, "MiddleValue": 12,
                    "DestinationOffset": 99, "DestinationValue": 25,
                },
            },
        ]
        document["percussion"] = []
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            template = directory / "template.gpt"
            full, single = directory / "full.gp", directory / "single.gp"
            template.write_bytes(archive_bytes(output_template()))
            write_gp_outputs(template, document, full, single)
            root = gp_root(full)
            notes = root.findall("./Notes/Note")
            self.assertIsNotNone(notes[0].find("./Properties/Property[@name='HopoOrigin']/Enable"))
            self.assertIsNotNone(notes[1].find("./Properties/Property[@name='HopoDestination']/Enable"))
            self.assertIsNotNone(notes[1].find("./Properties/Property[@name='LeftHandTapped']/Enable"))
            self.assertEqual(notes[1].findtext("Vibrato"), "Slight")
            self.assertIsNotNone(notes[1].find("./Properties/Property[@name='Bended']/Enable"))
            self.assertEqual(notes[1].findtext("./Properties/Property[@name='BendDestinationValue']/Float"), "25.000000")

    def test_connection_keeps_original_pair_or_is_suppressed_after_fingering(self):
        document = hypotheses()
        document["notes"] = [
            {
                **document["notes"][0], "onsetSeconds": 0, "string": 3, "fret": 2,
                "soundingPitchMidi": 57, "notatedDurationQuarter": 2, "connection": "none",
            },
            {
                **document["notes"][0], "onsetSeconds": .5, "string": 2, "fret": 0,
                "soundingPitchMidi": 59, "notatedDurationQuarter": .5, "connection": "none",
            },
            {
                **document["notes"][0], "onsetSeconds": 1, "string": 3, "fret": 5,
                "soundingPitchMidi": 60, "notatedDurationQuarter": .5,
                "connection": "hammer_on", "connectionConfidence": .95,
            },
        ]
        document["percussion"] = []
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            template = directory / "template.gpt"
            full, single = directory / "full.gp", directory / "single.gp"
            template.write_bytes(archive_bytes(output_template()))
            report = write_gp_outputs(template, document, full, single)
            root = gp_root(full)
            origins = [n for n in root.findall("./Notes/Note") if n.find("./Properties/Property[@name='HopoOrigin']") is not None]
            destinations = [n for n in root.findall("./Notes/Note") if n.find("./Properties/Property[@name='HopoDestination']") is not None]
            if origins:
                self.assertEqual([n.findtext("./Properties/Property[@name='Midi']/Number") for n in origins], ["57"])
                self.assertEqual([n.findtext("./Properties/Property[@name='Midi']/Number") for n in destinations], ["60"])
            else:
                self.assertFalse(destinations)
                self.assertEqual(report["fullVoices"]["connectionFallbacks"][0]["reason"], "invalid_connection_relationship")

    def test_attack_articulations_are_not_repeated_on_tied_continuations(self):
        document = hypotheses()
        document["notes"] = [
            {
                **document["notes"][0], "onsetSeconds": 0, "string": 1, "fret": 0,
                "soundingPitchMidi": 64, "notatedDurationQuarter": .5, "connection": "none",
            },
            {
                **document["notes"][0], "onsetSeconds": .5, "string": 1, "fret": 2,
                "soundingPitchMidi": 66, "notatedDurationQuarter": 5,
                "connection": "hammer_on", "connectionConfidence": .95,
                "noteTechniques": {"tap": .9, "bend": .9},
                "bendCurve": {
                    "OriginOffset": 0, "OriginValue": 0, "MiddleOffset1": 12,
                    "MiddleOffset2": 12, "MiddleValue": 12,
                    "DestinationOffset": 99, "DestinationValue": 25,
                },
            },
        ]
        document["percussion"] = []
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            template = directory / "template.gpt"
            full, single = directory / "full.gp", directory / "single.gp"
            template.write_bytes(archive_bytes(output_template()))
            report = write_gp_outputs(template, document, full, single)
            root = gp_root(full)
            self.assertEqual(len(root.findall("./Notes/Note/Properties/Property[@name='HopoDestination']")), 1)
            self.assertEqual(len(root.findall("./Notes/Note/Properties/Property[@name='Tapped']")), 1)
            self.assertFalse(root.findall("./Notes/Note/Properties/Property[@name='Bended']"))
            self.assertEqual(report["fullVoices"]["articulationFallbacks"][0]["reason"], "bend_curve_crosses_tie")

    def test_beat_evidence_drives_constrained_score_timing(self):
        document = hypotheses()
        beat_evidence = {
            "kind": "audio-beat-evidence",
            "audioSha256": "audio",
            "beatSeconds": [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
            "downbeatSeconds": [0, 2, 4],
        }
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            template = directory / "template.gpt"
            full, single = directory / "full.gp", directory / "single.gp"
            template.write_bytes(archive_bytes(output_template()))
            report = write_gp_outputs(template, document, full, single, beat_evidence=beat_evidence)
            self.assertEqual(report["rhythmicGridPolicy"]["mode"], "beat-anchored-constrained")
            self.assertEqual(report["rhythmInference"]["onsetOptimization"]["status"], "optimal")
            self.assertTrue(report["rhythmInference"]["durationOptimization"]["changedCount"])
            self.assertFalse(report["rhythmInference"]["rawHypothesesModified"])
            different = dict(beat_evidence, audioSha256="different")
            with self.assertRaisesRegex(HarnessError, "different audio"):
                write_gp_outputs(template, document, directory / "other.gp", directory / "other-single.gp", beat_evidence=different)

    def test_pickup_creates_a_short_first_measure_and_mixed_timing_fails(self):
        document = hypotheses()
        document["notes"][0]["onsetSeconds"] = 0
        beat_evidence = {
            "kind": "audio-beat-evidence",
            "audioSha256": "audio",
            "beatSeconds": [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
            "downbeatSeconds": [.5, 2.5, 4.5],
        }
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            template = directory / "template.gpt"
            full, single = directory / "full.gp", directory / "single.gp"
            template.write_bytes(archive_bytes(output_template()))
            write_gp_outputs(template, document, full, single, beat_evidence=beat_evidence)
            decoded = decode_score(gp_root(full), [40, 45, 50, 55, 59, 64], 0)
            self.assertTrue(decoded["measures"][0]["inferredPickup"])
            self.assertEqual(decoded["measures"][0]["durationQuarter"], [1, 1])
            mixed = hypotheses()
            mixed["notes"][0]["scoreOnsetQuarter"] = [0, 1]
            mixed["notes"][0]["scoreDurationQuarter"] = [1, 1]
            with self.assertRaisesRegex(HarnessError, "same structured"):
                write_gp_outputs(template, mixed, directory / "mixed.gp", directory / "mixed-single.gp")

    def test_percussion_uses_existing_voice_boundary_without_splitting_sustain(self):
        document = hypotheses()
        document["audioDurationSeconds"] = 3
        document["notes"] = [
            {
                "onsetSeconds": 0,
                "string": 6,
                "fret": 0,
                "soundingPitchMidi": 40,
                "voiceIndex": 0,
                "notatedDurationQuarter": 2,
                "harmonic": None,
                "confidence": .99,
                "uncertainty": [],
            },
            {
                "onsetSeconds": .5,
                "string": 1,
                "fret": 0,
                "soundingPitchMidi": 64,
                "voiceIndex": 1,
                "notatedDurationQuarter": 1,
                "harmonic": None,
                "confidence": .99,
                "uncertainty": [],
            },
        ]
        document["percussion"] = [{"onsetSeconds": .5, "technique": "wrist_thump", "confidence": .99}]
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            template = directory / "template.gpt"
            full, single = directory / "full.gp", directory / "single.gp"
            template.write_bytes(archive_bytes(output_template()))
            write_gp_outputs(template, document, full, single)
            score = decode_score(gp_root(full), [40, 45, 50, 55, 59, 64], 0)
            voice_zero = [event for event in score["scoreEvents"] if event["voiceIndex"] == 0 and not event["isRest"]]
            self.assertEqual(len(voice_zero), 1)
            self.assertEqual(voice_zero[0]["notatedDurationQuarter"], [2, 1])
            self.assertFalse(voice_zero[0]["notes"][0]["tie"]["origin"])
            wrist = [event for event in score["scoreEvents"] if event["text"] == "O"]
            self.assertEqual([(event["voiceIndex"], event["offsetQuarter"]) for event in wrist], [(1, [1, 1])])

    def test_single_voice_bridges_short_note_and_percussion_gaps_without_rests(self):
        document = hypotheses()
        document["audioDurationSeconds"] = 2
        document["notes"] = [
            {
                "onsetSeconds": 0,
                "string": 1,
                "fret": 0,
                "soundingPitchMidi": 64,
                "voiceIndex": 0,
                "notatedDurationQuarter": .25,
                "harmonic": None,
                "confidence": .99,
                "uncertainty": [],
            },
            {
                "onsetSeconds": .5,
                "string": 2,
                "fret": 1,
                "soundingPitchMidi": 60,
                "voiceIndex": 0,
                "notatedDurationQuarter": .25,
                "harmonic": None,
                "confidence": .99,
                "uncertainty": [],
            },
        ]
        document["percussion"] = [{"onsetSeconds": .25, "technique": "thumb_slap", "confidence": .99}]
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            template = directory / "template.gpt"
            full, single = directory / "full.gp", directory / "single.gp"
            template.write_bytes(archive_bytes(output_template()))
            write_gp_outputs(template, document, full, single)
            score = decode_score(gp_root(single), [40, 45, 50, 55, 59, 64], 0)
            events = score["scoreEvents"][:3]
            self.assertEqual([(event["offsetQuarter"], event["notatedDurationQuarter"], event["isRest"]) for event in events], [
                ([0, 1], [1, 2], False),
                ([1, 2], [1, 2], False),
                ([1, 1], [1, 4], False),
            ])

    def test_duration_is_bounded_by_audio_and_same_string_reattacks_are_reported(self):
        document = hypotheses()
        document["audioDurationSeconds"] = 2
        document["notes"] = [
            {
                "onsetSeconds": 0,
                "string": 6,
                "fret": 0,
                "soundingPitchMidi": 40,
                "voiceIndex": 0,
                "notatedDurationQuarter": 64,
                "harmonic": None,
                "confidence": 0.9,
                "uncertainty": ["duration_clipped"],
            },
            {
                "onsetSeconds": 1,
                "string": 6,
                "fret": 2,
                "soundingPitchMidi": 42,
                "voiceIndex": 0,
                "notatedDurationQuarter": 1,
                "harmonic": None,
                "confidence": 0.9,
                "uncertainty": [],
            },
        ]
        document["percussion"] = []
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            template = directory / "template.gpt"
            full, single = directory / "full.gp", directory / "single.gp"
            template.write_bytes(archive_bytes(output_template()))
            report = write_gp_outputs(template, document, full, single)
            self.assertEqual(report["fullVoices"]["measureCount"], 1)
            self.assertEqual(
                {item["reason"] for item in report["shortenedSustainHypotheses"]},
                {"clamped_to_audio_end", "same_string_reattack"},
            )

    def test_invalid_or_empty_hypotheses_and_existing_outputs_fail(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            template = directory / "template.gpt"
            template.write_bytes(archive_bytes(output_template()))
            full, single = directory / "full.gp", directory / "single.gp"
            empty = hypotheses()
            empty["notes"] = []
            empty["percussion"] = []
            with self.assertRaisesRegex(HarnessError, "at least one"):
                write_gp_outputs(template, empty, full, single)
            full.write_bytes(b"existing")
            with self.assertRaisesRegex(HarnessError, "overwrite"):
                write_gp_outputs(template, hypotheses(), full, single)
            with self.assertRaisesRegex(HarnessError, "different paths"):
                write_gp_outputs(template, hypotheses(), single, single)

    def test_meter_changes_and_cli_export_publish_private_outputs(self):
        document = hypotheses()
        document["audioDurationSeconds"] = 6
        document["metadata"]["timeSignature"] = [3, 4]
        document["metadata"]["timeSignatureChanges"] = [{"timeSeconds": 3, "timeSignature": [4, 4]}]
        with TemporaryDirectory(dir=ROOT / "runs") as directory:
            directory = Path(directory)
            template = directory / "template.gpt"
            predictions = directory / "predictions.json"
            template.write_bytes(archive_bytes(output_template()))
            publish_json(predictions, document)
            relative = directory.relative_to(ROOT)
            arguments = [
                "export-gp",
                "--predictions", str(predictions),
                "--template", str(template),
                "--full-output", str(relative / "full.gp"),
                "--single-output", str(relative / "single.gp"),
                "--report", str(relative / "report.json"),
            ]
            self.assertEqual(transcriber.main(arguments), 0)
            report = read_json(directory / "report.json")
            self.assertEqual(report["fullVoices"]["measureCount"], 4)
            self.assertTrue((directory / "full.gp").is_file())
            self.assertTrue((directory / "single.gp").is_file())
            self.assertEqual(report["fullOutputSha256"], hashlib.sha256((directory / "full.gp").read_bytes()).hexdigest())

    def test_full_voice_generic_hit_uses_text_when_all_strings_are_sustained(self):
        document = hypotheses()
        document["notes"] = [{
            "onsetSeconds": 0,
            "string": string,
            "fret": 0,
            "soundingPitchMidi": document["metadata"]["openStringMidi"][6 - string],
            "voiceIndex": 0,
            "notatedDurationQuarter": 4,
            "harmonic": None,
            "confidence": 0.9,
            "uncertainty": [],
        } for string in range(1, 7)]
        document["percussion"] = [{"onsetSeconds": 1, "technique": "percussive_hit", "confidence": 0.9}]
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            template = directory / "template.gpt"
            full, single = directory / "full.gp", directory / "single.gp"
            template.write_bytes(archive_bytes(output_template()))
            report = write_gp_outputs(template, document, full, single)
            self.assertEqual(report["fullVoices"]["percussionTextFallbacks"][0]["text"], "(X)")
            self.assertIn("(X)", {beat.findtext("FreeText") for beat in gp_root(full).findall("./Beats/Beat")})


if __name__ == "__main__":
    unittest.main()
