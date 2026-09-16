from copy import deepcopy
import hashlib
from io import StringIO
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from scripts import transcriber
from scripts.dataset_io import ROOT, read_json, sha256, publish_json
from scripts.dataset_release import candidate_digest, release_scope, validate_mapping, validate_release
from scripts.training_windows import projected_targets, targets_in_window
from scripts.score_alignment import ScoreClock
from scripts.transcriber_audio import FeatureConfig
from scripts.transcriber_data import TrainingDataset, collate_windows
from scripts.transcriber_model import ModelConfig
from tests.test_score_alignment import clock_fixture


def synthetic_release(root, *, plateau=False):
    entries, records = [], []
    for index, split in enumerate(("train", "validation")):
        identifier = f"piece-{index}"
        rate = 8000
        audio = root / "audio" / f"{identifier}.flac"
        audio.parent.mkdir(parents=True, exist_ok=True)
        samples = .1 * np.sin(2 * np.pi * (220 + index * 110) * np.arange(rate * 6) / rate)
        sf.write(audio, np.column_stack((samples, -samples)), rate, subtype="PCM_24")
        labels, normalization = clock_fixture()
        labels.update(timeUnit="quarter-note", scoreTimingResolved=True)
        labels["provenance"] = {"sourceGpPath": "unavailable\\source.gp", "eventPath": "unavailable\\events.json"}
        normalization["rawCanonicalPath"] = "unavailable\\canonical.json"
        labels["conditioning"]["instrument"] = {"openStringMidi": [40, 45, 50, 55, 59, 64], "capoFret": 0}
        labels["conditioning"]["providedTiming"]["sourceTimeSignatureChanges"] = []
        labels["targets"] = {
            "notes": [{"id": "n", "voiceIndex": 0, "string": 6, "fret": 0, "soundingPitchMidi": 40, "onsetQuarter": [1, 1], "notatedDurationQuarter": [2, 1], "isAttack": True, "sourceSegments": [{"graceMode": None, "harmonic": None}], "labelMask": {"attack": True, "pitch": True, "fingering": True, "notatedDuration": True}}],
            "gestures": [{"id": "g", "voiceIndex": 0, "technique": "wrist_thump", "onsetQuarter": [1, 1], "scoreOnsetKnown": True, "graceMode": None}],
        }
        candidate = {"denseMapping": [{"clipSeconds": float(t), "referenceSeconds": float(t), "scoreQuarter": float(t)} for t in (0, 3, 6)]}
        if plateau:
            labels["targets"]["notes"].append({**deepcopy(labels["targets"]["notes"][0]), "id": "n2", "onsetQuarter": [2, 1], "fret": 2, "soundingPitchMidi": 42})
            candidate["denseMapping"] = [{"clipSeconds": clip, "referenceSeconds": reference, "scoreQuarter": reference} for clip, reference in ((0., 0.), (1., 1.), (1., 2.), (6., 6.))]
        clock = ScoreClock(labels, normalization)
        notes, gestures = projected_targets(labels, candidate, clock)
        approval = {
            "authorizedUse": True, "recordingAndTargetPitchConfirmed": True, "notationReviewed": True,
            "approveExperimentalRangesAndSplit": True, "groupingConfirmed": True,
            "groupId": identifier, "split": split,
            "sourceGpSha256": hashlib.sha256(identifier.encode()).hexdigest(), "audioSha256": sha256(audio),
            "candidateSha256": candidate_digest(candidate), "approvedClipRanges": [[0., 6.]],
        }
        if plateau:
            approval["uncertaintyAcknowledged"] = True
        payload = {
            "schemaVersion": 1, "kind": "local-training-targets", "id": identifier,
            "canonical": labels, "normalization": normalization, "candidate": candidate, "approval": approval,
            "windows": [{"windowId": f"{identifier}:0-{rate * 6}", "startSample": 0, "stopSampleExclusive": rate * 6, "targets": targets_in_window(notes, gestures, 0, rate * 6, rate)}],
        }
        entries.append({
            "id": identifier, "groupId": identifier, "split": split,
            "audioPath": f"audio\\{identifier}.flac", "audioSha256": sha256(audio),
            "sampleRate": rate, "channels": 2, "sampleCount": rate * 6,
            "targetsPath": f"targets\\{identifier}.json",
        })
        records.append((entries[-1], payload))
    authorization = {
        "schemaVersion": 1, "kind": "local-release-authorization", "reviewer": "Synthetic reviewer",
        "version": "v1", "validationGroup": "piece-1", "authorizedUse": True,
        "approveExperimentalRangesAndSplit": True, "groupingConfirmed": True,
        "distributionAuthorized": False, "trainingExecution": "human-owner-only",
        "selectedScope": release_scope(records),
    }
    authorization["sha256"] = candidate_digest(authorization)
    for entry, payload in records:
        payload["approval"].update(releaseReviewer=authorization["reviewer"], releaseAuthorizationSha256=authorization["sha256"])
        target = root / "targets" / f"{entry['id']}.json"
        publish_json(target, payload)
        entry["targetsSha256"] = sha256(target)
    manifest = root / "manifest.json"
    publish_json(manifest, {"schemaVersion": 1, "kind": "local-training-dataset", "trainingReady": True, "visibility": "private", "distributionAuthorized": False, "entries": entries, "counts": {"windowsBySplit": {"train": 1, "validation": 1}}, "version": "v1", "validationGroup": "piece-1", "trainingExecution": "human-owner-only", "releaseAuthorization": authorization})
    return manifest


class DatasetReleaseTests(unittest.TestCase):
    def test_mapping_bounds_finiteness_clock_and_reference_order_are_required(self):
        clock = ScoreClock(*clock_fixture())
        mapping = [{"clipSeconds": clip, "referenceSeconds": reference, "scoreQuarter": reference} for clip, reference in ((0., 0.), (1., 1.), (1., 2.), (6., 6.))]
        validate_mapping(mapping, clock, 6.)
        for index, field, value in ((1, "clipSeconds", float("nan")), (1, "referenceSeconds", float("inf")), (0, "clipSeconds", -1.), (3, "clipSeconds", 7.), (3, "referenceSeconds", 7.), (2, "clipSeconds", .9), (2, "referenceSeconds", 1.), (1, "scoreQuarter", 1.000001)):
            changed = deepcopy(mapping)
            changed[index][field] = value
            with self.subTest(index=index, field=field, value=value), self.assertRaises(ValueError):
                validate_mapping(changed, clock, 6.)

    def test_plateau_preserves_projection_and_exact_collision_masks(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory).resolve()
            manifest_path = synthetic_release(root, plateau=True)
            manifest, records, _ = validate_release(manifest_path)
            payload = records[0][1]
            self.assertEqual([note["proposedOnsetClipSeconds"] for note in payload["windows"][0]["targets"]["notes"]], [1., 1.])
            features = FeatureConfig(sample_rate=8000, n_fft=512, hop_length=160, n_mels=16, f_max=3000)
            item = TrainingDataset(manifest_path, "train", features, ModelConfig(n_mels=16), root=root)[0]
            self.assertEqual(item["metadata"]["stringFrameCollisionsMasked"], 1)
            expected = torch.zeros((300, 6))
            expected[50, 0] = 1
            torch.testing.assert_close(item["targets"]["note_onset"], expected)
            mask = torch.zeros((300, 6), dtype=torch.bool)
            mask[25:274] = True
            mask[47:54, 0] = False
            mask[50, 0] = True
            torch.testing.assert_close(item["masks"]["note_onset"], mask)
            for name in ("fret", "pitch", "voice", "duration_log", "harmonic", "harmonic_kind", "harmonic_node"):
                torch.testing.assert_close(item["masks"][name], torch.zeros((300, 6), dtype=torch.bool))
            target = root / "targets" / "piece-0.json"
            payload["approval"]["uncertaintyAcknowledged"] = False
            publish_json(target, payload)
            manifest["entries"][0]["targetsSha256"] = sha256(target)
            publish_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "plateaus"):
                validate_release(manifest_path)

    def test_single_loader_rejects_retired_schema_without_source_resolution(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory).resolve()
            manifest_path = root / "manifest.json"
            publish_json(manifest_path, {"kind": "released-experimental-pilot-dataset", "proposalPath": "must-not-read.json"})
            with self.assertRaisesRegex(ValueError, "training release"):
                TrainingDataset(manifest_path, "train", FeatureConfig(), ModelConfig(), root=root)
            with self.assertRaisesRegex(ValueError, "training release"):
                transcriber.make_dataset(transcriber.default_config(), FeatureConfig(), ModelConfig(), "train", root, manifest_path)

    def test_portable_release_loads_after_original_workspace_is_removed(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory).resolve()
            source = root / "old-workspace" / "release"
            manifest = synthetic_release(source)
            digest = sha256(manifest)
            moved = root / "relocated"
            shutil.copytree(source, moved)
            shutil.rmtree(source.parent)
            features = FeatureConfig(sample_rate=8000, n_fft=512, hop_length=160, n_mels=16, f_max=3000)
            model = ModelConfig(n_mels=16)
            dataset = TrainingDataset(moved / "manifest.json", "train", features, model, root=moved)
            self.assertEqual(dataset.manifest_sha256, digest)
            item = dataset[0]
            self.assertEqual(item["features"].shape, (300, 16))
            self.assertEqual(item["conditioning"].shape, (300, 12))
            self.assertTrue(torch.isfinite(item["features"]).all())
            self.assertGreater(float(item["features"].std()), .1)
            self.assertEqual(int(item["masks"]["percussion"].sum()), 1)
            self.assertEqual(int(item["targets"]["note_onset"].sum()), 1)
            self.assertEqual(collate_windows([item])["lengths"].tolist(), [300])
            with patch("scripts.transcriber_runtime.run_training", side_effect=AssertionError("No training in preflight")), patch("torch.optim.AdamW", side_effect=AssertionError("No optimizer in preflight")), patch("sys.stdout", new=StringIO()):
                self.assertEqual(transcriber.main(["preflight", "--data-root", str(moved), "--manifest", "manifest.json", "--forward"]), 0)
            report = read_json(moved / "runs" / "preflight.json")
            self.assertFalse(report["trainingRun"])
            self.assertEqual(report["weights"], "untrained-in-memory-only")
            self.assertEqual(report["splits"]["validation"]["forwardShapes"]["note_onset_logits"], [1, 300, 6])

    def test_explicit_approval_and_connected_split_guards(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            manifest_path = synthetic_release(Path(directory).resolve())
            manifest = read_json(manifest_path)
            for changed in (
                {**manifest, "distributionAuthorized": True},
                {**manifest, "counts": {"windowsBySplit": {"train": 2, "validation": 1}}},
                {**manifest, "entries": [manifest["entries"][0], {**manifest["entries"][1], "groupId": "piece-0"}]},
                {**manifest, "entries": [manifest["entries"][0], {**manifest["entries"][1], "split": "test"}]},
                {**manifest, "releaseAuthorization": {**manifest["releaseAuthorization"], "selectedScope": []}},
                {**manifest, "validationGroup": "piece-0"},
            ):
                publish_json(manifest_path, changed)
                with self.assertRaises(ValueError):
                    validate_release(manifest_path)
            publish_json(manifest_path, manifest)
            target = manifest_path.parent / "targets" / "piece-0.json"
            payload = read_json(target)
            payload["approval"]["authorizedUse"] = False
            publish_json(target, payload)
            manifest["entries"][0]["targetsSha256"] = sha256(target)
            publish_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "approval"):
                validate_release(manifest_path)

    def test_rehashed_mutations_cannot_weaken_masks_or_change_approved_timing(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            manifest_path = synthetic_release(Path(directory).resolve())
            original_manifest = read_json(manifest_path)
            target = manifest_path.parent / "targets" / "piece-0.json"
            original_payload = read_json(target)

            def change_mask(payload):
                payload["windows"][0]["targets"]["negativePercussionSupervision"] = True

            def change_candidate(payload):
                payload["candidate"]["denseMapping"][1]["clipSeconds"] = 3.5

            def change_range(payload):
                payload["approval"]["approvedClipRanges"] = [[1., 6.]]

            def change_quarter(payload):
                payload["candidate"]["denseMapping"][1]["scoreQuarter"] = 4.
                payload["approval"]["candidateSha256"] = candidate_digest(payload["candidate"])

            for mutation in (change_mask, change_candidate, change_range, change_quarter):
                payload, manifest = deepcopy(original_payload), deepcopy(original_manifest)
                mutation(payload)
                publish_json(target, payload)
                manifest["entries"][0]["targetsSha256"] = sha256(target)
                publish_json(manifest_path, manifest)
                with self.subTest(mutation=mutation.__name__), self.assertRaises(ValueError):
                    validate_release(manifest_path)

    def test_changed_bytes_and_escaping_paths_fail(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            manifest_path = synthetic_release(Path(directory).resolve())
            manifest = read_json(manifest_path)
            target = manifest_path.parent / "targets" / "piece-0.json"
            target.write_bytes(target.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_release(manifest_path)
            manifest["entries"][0]["targetsSha256"] = sha256(target)
            manifest["entries"][0]["targetsPath"] = "targets\\..\\manifest.json"
            publish_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "stay under"):
                validate_release(manifest_path)


if __name__ == "__main__":
    unittest.main()
