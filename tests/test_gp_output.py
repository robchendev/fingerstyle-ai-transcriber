from io import BytesIO
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import math
import unittest
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from scripts.dataset_io import ROOT
from scripts.dataset_io import publish_json, read_json
from scripts.gp_events import decode_score, rhythm_duration
from scripts.gp_normalization import GPIF_ENTRY
from scripts.gp_output import GRID, TempoMap, _rhythmic_positions, write_gp_outputs
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
            self.assertEqual(len(bass), 2)
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
