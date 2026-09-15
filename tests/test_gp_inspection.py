from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from scripts.inspect_gp_files import apply_owner_confirmation, candidate_priority, compare_tuning, inspect_gp, matches_entry


def score(track_count=1, partial_fret=0, flags="000000", annotation=""):
    tracks = "".join(
        f'<Track id="{index}"><Name>Guitar {index + 1}</Name><InstrumentSet><Type>steelGuitar</Type></InstrumentSet><Staves><Staff><Properties>'
        '<Property name="Tuning"><Pitches>40 45 50 55 59 64</Pitches></Property>'
        '<Property name="CapoFret"><Fret>2</Fret></Property>'
        f'<Property name="PartialCapoFret"><Fret>{partial_fret}</Fret></Property>'
        f'<Property name="PartialCapoStringFlags"><Bitset>{flags}</Bitset></Property>'
        '</Properties></Staff></Staves></Track>'
        for index in range(track_count)
    )
    bars = "".join(f'<Bar id="{index}"><Voices>{index} -1 -1 -1</Voices></Bar>' for index in range(track_count))
    voices = "".join(f'<Voice id="{index}"><Beats>{index}</Beats></Voice>' for index in range(track_count))
    beats = "".join(f'<Beat id="{index}"><Notes>{index}</Notes><FreeText>{annotation}</FreeText></Beat>' for index in range(track_count))
    notes = "".join(f'<Note id="{index}"/>' for index in range(track_count))
    references = " ".join(str(index) for index in range(track_count))
    return f'<GPIF><GPVersion>8</GPVersion><Score><Title>Example</Title></Score><Tracks>{tracks}</Tracks><MasterBars><MasterBar><Bars>{references}</Bars></MasterBar></MasterBars><Bars>{bars}</Bars><Voices>{voices}</Voices><Beats>{beats}</Beats><Notes>{notes}</Notes></GPIF>'


class GpInspectionTests(unittest.TestCase):
    def inspect(self, xml):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "score.gp"
            with ZipFile(path, "w") as archive:
                archive.writestr("Content/score.gpif", xml)
            return inspect_gp(path)

    def test_track_count_uses_track_definitions_not_voice_slots(self):
        for count in (1, 2, 3):
            with self.subTest(count=count):
                result = self.inspect(score(count))
                self.assertEqual(result["trackCount"], count)
                self.assertEqual([track["noteCount"] for track in result["tracks"]], [1] * count)
                self.assertEqual(result["tracks"][0]["staves"][0]["capoFret"], 2)

    def test_note_references_follow_master_track_order(self):
        xml = score(2).replace("<MasterBars>", "<MasterTrack><Tracks>1 0</Tracks></MasterTrack><MasterBars>")
        xml = xml.replace("<Notes>0</Notes>", "<Notes>0 2</Notes>").replace('<Note id="1"/>', '<Note id="1"/><Note id="2"/>')
        result = self.inspect(xml)
        self.assertEqual([track["noteCount"] for track in result["tracks"]], [1, 2])

    def test_partial_capo_configuration_and_text_are_not_silently_normalized(self):
        inactive = self.inspect(score(partial_fret=4))
        self.assertIn("inconsistent_partial_capo_metadata", inactive["warnings"])
        self.assertFalse(inactive["tracks"][0]["staves"][0]["partialCapoActive"])
        active = self.inspect(score(partial_fret=4, flags="001111", annotation="Capo to fret 8"))
        self.assertIn("partial_capo_active", active["warnings"])
        self.assertIn("capo_or_tuning_text_requires_review", active["warnings"])
        self.assertEqual(active["capoOrTuningText"], ["Capo to fret 8"])
        self.assertNotIn("partial_capo_active", self.inspect(score())["warnings"])
        textual = self.inspect(score(annotation="Partial Capo on fret 5, on strings 1-5."))
        self.assertIn("partial_capo_in_text", textual["warnings"])
        self.assertNotIn("partial_capo_active", textual["warnings"])

    def test_catalog_tuning_uses_exact_open_string_pitches(self):
        entry = {"selectedTuningIndex": 0, "tunings": [{"strings": ["E2", "A2", "D3", "G3", "B3", "E4"]}]}
        self.assertIsNone(compare_tuning(entry, self.inspect(score())))
        entry["tunings"][0]["strings"][0] = "D2"
        self.assertEqual(compare_tuning(entry, self.inspect(score())), "catalog_gp_tuning_mismatch")

    def test_owner_confirmation_resolves_only_stale_metadata_on_the_confirmed_revision(self):
        inspection = self.inspect(score(partial_fret=4))
        result = {"catalogId": "set-a-item-0218", "sha256": "a" * 64, "inspection": inspection, "issues": [*inspection["warnings"], "catalog_gp_tuning_mismatch"]}
        rules = {"ownerConfirmations": {"set-a-item-0218": {"gpSha256": "a" * 64, "capoType": "full", "note": "Owner confirmed normal capo."}}}
        apply_owner_confirmation(result, rules)
        self.assertEqual(result["issues"], ["catalog_gp_tuning_mismatch"])
        self.assertIn("inconsistent_partial_capo_metadata", inspection["warnings"])
        apply_owner_confirmation(result, rules)
        self.assertEqual(result["resolvedInspectionIssues"], ["inconsistent_partial_capo_metadata"])
        other = {"catalogId": "set-a-item-0218", "sha256": "b" * 64, "issues": ["inconsistent_partial_capo_metadata"]}
        apply_owner_confirmation(other, rules)
        self.assertIn("inconsistent_partial_capo_metadata", other["issues"])
        self.assertIn("owner_confirmation_revision_mismatch", other["issues"])
        self.assertNotIn("ownerConfirmation", other)

    def test_inspection_leaves_style_resources_untouched_and_out_of_the_music_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            inspections = []
            for index in (1, 2):
                path = Path(directory) / f"styled-{index}.gp"
                xml = score().replace("<Score>", f"<Score><FirstPageHeader>Custom font size {index}</FirstPageHeader>")
                with ZipFile(path, "w") as archive:
                    archive.writestr("Content/score.gpif", xml)
                    archive.writestr("Content/Preferences.json", f'{{"customSetting": {index}}}')
                    archive.writestr("Content/Stylesheets/score.gpss", bytes([index, 0, 255]))
                    archive.writestr("Content/UnknownSetting.bin", bytes([index, 42]))
                before = path.read_bytes()
                inspections.append(inspect_gp(path))
                self.assertEqual(path.read_bytes(), before)
            self.assertEqual(inspections[0]["musicXmlSha256"], inspections[1]["musicXmlSha256"])

    def test_matching_respects_full_title_boundaries_artists_and_years(self):
        rules = {"titleAliases": {}, "requiredFilenameText": {}, "excludedPathText": ["\\Noble\\"], "sourcePreference": ["Tabs\\Transcriptions 2022+\\", "Tabs\\MyMusicSheet Migration\\"]}
        entry = {"id": "x", "title": "Here"}
        self.assertFalse(matches_entry(entry, {"sourcePath": "Tabs\\Sample Artist - Other Example.gp", "title": "Other Example"}, rules))
        self.assertTrue(matches_entry(entry, {"sourcePath": "Tabs\\JUNNA - Here.gp", "title": "Here"}, rules))
        self.assertFalse(matches_entry(entry, {"sourcePath": "Tabs\\Here (2025).gp", "title": "Here"}, rules))
        rules["requiredFilenameText"]["x"] = "JUNNA"
        self.assertFalse(matches_entry(entry, {"sourcePath": "Tabs\\Other - Here.gp", "title": "Here"}, rules))
        original = {"sourcePath": "Tabs\\Transcriptions 2022+\\Here.gp", "trackCount": 2}
        publishing = {"sourcePath": "Tabs\\MyMusicSheet Migration\\Here.gp", "trackCount": 1}
        self.assertLess(candidate_priority(original, rules), candidate_priority(publishing, rules))

    def test_newest_selection_uses_mtime_not_publishing_priority_or_track_count(self):
        rules = {"selectionPolicy": "newest-modified"}
        older = {"sourcePath": "A.gp", "mtimeNs": 10, "trackCount": 1}
        newest = {"sourcePath": "Z.gp", "mtimeNs": 20, "trackCount": 2}
        self.assertLess(candidate_priority(newest, rules), candidate_priority(older, rules))
        self.assertLess(candidate_priority(dict(older, mtimeNs=20), rules), candidate_priority(newest, rules))

    def test_dataset-b_context_prefix_does_not_match_arbitrary_title_suffixes(self):
        rules = {"filenameTitleAfterLeadingContext": True, "titleAliases": {}, "requiredFilenameText": {}, "excludedPathText": []}
        self.assertTrue(matches_entry({"id": "set-b-item-0013", "title": "Example Work"}, {"sourcePath": r"2021\(Source Work A OP) Example Work.gp", "title": "Expanded score title"}, rules))
        self.assertFalse(matches_entry({"id": "x", "title": "Here"}, {"sourcePath": r"2021\(Source Work B ED) Other Example.gp", "title": "Another title"}, rules))


if __name__ == "__main__":
    unittest.main()
