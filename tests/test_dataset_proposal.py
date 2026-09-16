from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.dataset_io import ROOT, read_json
from scripts.dataset_release import candidate_digest
from scripts.dataset_proposal import coverage_profile, propose_dataset, proposed_ranges, select_validation, subtract_intervals
from scripts.prepare_training_data import write_json
from tests.test_score_alignment import clock_fixture


def profile(identifier, *, group=None, split=None, counts=None, performer="performer", meter=(4, 4)):
    return {
        "id": identifier, "title": identifier, "groupId": group or identifier,
        "performerId": performer, "status": "proposed", "blockers": [],
        "existingSplitConstraint": split, "existingRangeApproval": False, "percussionCompletenessConfirmed": False,
        "coverage": {
            "counts": {"notes": 100, "wrist_thump": 10, "thumb_slap": 10, "percussive_hit": 10, "harmonics": 10, **(counts or {})},
            "voices": [0, 1], "windows": [{"startSample": 0, "stopSampleExclusive": 1000}], "windowCount": 1,
        },
        "ranges": [[0., 10.]], "excludedPassages": [], "proposedAudioSeconds": 10., "audioDurationSeconds": 10.,
        "tuning": [40, 45, 50, 55, 59, 64], "capo": 0, "meters": [list(meter)],
        "sourceBindings": {}, "reviewWasPresent": False,
    }


class DatasetProposalTests(unittest.TestCase):
    def test_local_risks_are_guarded_and_unbounded_risks_are_quarantined(self):
        candidate = {
            "denseMapping": [{"clipSeconds": 0.}, {"clipSeconds": 10.}],
            "triage": {"reasons": ["sustained_local_mismatch"], "passages": [{"reason": "sustained_local_mismatch", "clipSecondsStart": 4., "clipSecondsEnd": 5.}]},
        }
        ranges, exclusions, blockers, reviewed = proposed_ranges(candidate, {}, 10.)
        self.assertEqual(ranges, [[0., 3.5], [5.5, 10.]])
        self.assertEqual(exclusions[0]["guardedClipSeconds"], [3.5, 5.5])
        self.assertEqual(blockers, [])
        self.assertFalse(reviewed)
        candidate["triage"]["reasons"].append("owner_rejected_placement")
        self.assertEqual(proposed_ranges(candidate, {}, 10.)[0], [])
        self.assertIn("owner_rejected_placement", proposed_ranges(candidate, {}, 10.)[2])

    def test_unbounded_local_flags_cannot_be_silently_accepted(self):
        candidate = {"denseMapping": [{"clipSeconds": 1.}, {"clipSeconds": 10.}], "triage": {"reasons": ["long_score_position_stall"], "passages": []}}
        self.assertEqual(proposed_ranges(candidate, {}, 10.)[2], ["unbounded_long_score_position_stall"])

    def test_existing_approved_ranges_are_preserved_not_expanded_or_retrimmed(self):
        candidate = {"denseMapping": [{"clipSeconds": 0.}, {"clipSeconds": 10.}], "triage": {"reasons": ["owner_rejected_placement"]}}
        approval = {name: True for name in ("authorizedUse", "recordingAndTargetPitchConfirmed", "notationReviewed", "approveExperimentalRangesAndSplit", "groupingConfirmed")}
        approval["approvedClipRanges"] = [[2., 3.], [6., 9.]]
        original = deepcopy(approval)
        result = proposed_ranges(candidate, {"approval": approval}, 10.)
        self.assertEqual(result, (approval["approvedClipRanges"], [], [], True))
        self.assertEqual(approval, original)

    def test_subtraction_handles_overlapping_exclusions(self):
        self.assertEqual(subtract_intervals([[0., 10.]], [[2., 4.], [3., 6.], [8., 12.]]), [[0., 2.], [6., 8.]])

    def test_group_selection_keeps_related_records_and_reviewed_assignments(self):
        rows = [profile("a", group="t", split="train", counts={name: 1000 for name in ("wrist_thump", "thumb_slap", "percussive_hit", "harmonics")}), profile("b", group="v", split="validation"), profile("c", group="v"), profile("d"), profile("e")]
        groups = select_validation(rows, 2)
        self.assertIn("v", groups)
        self.assertNotIn("t", groups)
        self.assertEqual(sum(row["groupId"] in groups for row in rows), 3)
        self.assertEqual(groups, select_validation(rows, 2))
        with self.assertRaisesRegex(ValueError, "exceed"):
            select_validation([*rows, profile("f", split="validation")], 1)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            select_validation(rows, 3, ["a", "b"])

    def test_explicit_ids_expand_whole_groups_without_second_split_mechanism(self):
        rows = [profile("a", group="same"), profile("b", group="same"), profile("c")]
        self.assertEqual(select_validation(rows, 1, ["b"]), ["same"])
        for selected in ([], ["missing"], ["a", "a"]):
            with self.assertRaises(ValueError):
                select_validation(rows, 1, selected)

    def test_overlap_window_counts_do_not_duplicate_unique_event_coverage(self):
        labels, normalization = clock_fixture()
        labels["targets"] = {
            "notes": [{
                "id": "n", "voiceIndex": 0, "string": 6, "fret": 0, "soundingPitchMidi": 52,
                "onsetQuarter": [3, 1], "notatedDurationQuarter": [1, 1], "isAttack": True,
                "labelMask": {"attack": True, "pitch": True, "fingering": True, "notatedDuration": True},
                "sourceSegments": [{"graceMode": None, "harmonic": {"type": "Natural", "fret": [12, 1]}}],
            }],
            "gestures": [{"id": "g", "voiceIndex": 0, "technique": "wrist_thump", "onsetQuarter": [3, 1], "scoreOnsetKnown": True, "graceMode": None}],
        }
        mapping = {"denseMapping": [{"referenceSeconds": float(value), "clipSeconds": float(value), "scoreQuarter": float(value)} for value in (0, 3, 6)]}
        result = coverage_profile(labels, mapping, normalization, [[0., 4.], [2., 6.]], 100)
        self.assertEqual(result["windowCount"], 2)
        self.assertEqual(sum(window["noteAttacks"] for window in result["windows"]), 2)
        self.assertEqual(result["counts"]["notes"], 1)
        self.assertEqual(result["counts"]["harmonics"], 1)
        self.assertEqual(result["counts"]["wrist_thump"], 1)
        self.assertEqual(result["pitchCounts"], {"52": 1})
        self.assertEqual(result["fretCounts"], {"0": 1})

    def test_proposal_is_private_nontraining_bound_and_refuses_changed_named_output(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            rows = [profile("a"), profile("b", performer="other", meter=(3, 4))]
            registry = {"schemaVersion": 1, "kind": "local-pairs", "pairs": [
                {"id": row["id"], "groupId": row["groupId"], "gpPath": f"pairs\\{row['id']}\\raw.gp", "audioPath": f"pairs\\{row['id']}\\trimmed.flac"} for row in rows
            ]}
            write_json(root / "pairs.json", registry)
            with patch("scripts.dataset_proposal.inspect_pair", side_effect=deepcopy(rows)):
                result = propose_dataset(root, "example", validation_group_count=1)
            proposal = read_json(Path(result["proposalPath"]))
            self.assertEqual(proposal["proposalSha256"], candidate_digest({key: value for key, value in proposal.items() if key != "proposalSha256"}))
            self.assertIs(proposal["trainingReady"], False)
            self.assertEqual(proposal["kind"], "training-dataset-proposal")
            self.assertEqual(proposal["visibility"], "private")
            self.assertFalse((root / "releases").exists())
            self.assertTrue(Path(result["reviewPath"]).is_file())
            self.assertEqual(proposal["splitSummary"]["train"]["recordings"], 1)
            before = Path(result["proposalPath"]).read_bytes()
            with patch("scripts.dataset_proposal.inspect_pair", side_effect=deepcopy(rows)):
                propose_dataset(root, "example", validation_group_count=1)
            self.assertEqual(Path(result["proposalPath"]).read_bytes(), before)
            duplicate = deepcopy(rows)
            for row in duplicate:
                row["audioSha256"] = "identical-audio"
            with patch("scripts.dataset_proposal.inspect_pair", side_effect=duplicate), self.assertRaisesRegex(ValueError, "Identical source"):
                propose_dataset(root, "duplicate", validation_group_count=1)
            rows[0]["title"] = "changed"
            with patch("scripts.dataset_proposal.inspect_pair", side_effect=rows), self.assertRaisesRegex(ValueError, "different proposal"):
                propose_dataset(root, "example", validation_group_count=1)


if __name__ == "__main__":
    unittest.main()
