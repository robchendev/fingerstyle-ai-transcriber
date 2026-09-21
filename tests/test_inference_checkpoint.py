from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import pickle
import pickletools
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

import torch

from scripts.dataset_io import ROOT
from scripts.transcriber import checkpoint_model
from scripts.transcriber_audio import FeatureConfig
from scripts.transcriber_model import FingerstyleTranscriber, ModelConfig
from scripts import transcriber_runtime as runtime
from scripts.transcriber_video import AudioVideoTranscriber, VideoConfig


class InferenceCheckpointTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory(prefix="inference-package-", dir=ROOT)
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "model.pt"
        self.features = FeatureConfig(n_mels=16)
        self.config = ModelConfig(architecture_version=4, n_mels=16, hidden_size=8, recurrent_layers=1)
        self.video = VideoConfig(hidden_size=8)
        self.model = AudioVideoTranscriber(FingerstyleTranscriber(self.config), self.video).eval()
        self.payload = {
            "format": runtime.INFERENCE_FORMAT, "schema_version": runtime.INFERENCE_SCHEMA_VERSION,
            "model_config": asdict(self.config), "feature_config": asdict(self.features),
            "video_config": asdict(self.video),
            "model_state": dict(self.model.state_dict()),
        }

    def save(self, payload=None):
        torch.save(self.payload if payload is None else payload, self.path)
        return self.path

    def test_exact_tensor_and_no_gradient_output_roundtrip(self):
        restored, checkpoint, device = checkpoint_model(self.save(), "cpu")
        for name, expected in self.payload["model_state"].items():
            actual = restored.state_dict()[name]
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(torch.equal(actual, expected), name)
        self.assertEqual(str(device), "cpu")
        self.assertNotIn("identity", checkpoint)
        from scripts.transcription_pipeline import _checkpoint
        self.assertEqual(_checkpoint(self.path, require_video=True)["format"], runtime.INFERENCE_FORMAT)
        inputs = (torch.randn(2, 7, 16), torch.randn(2, 7, 12), torch.tensor([7, 5]))
        available = torch.ones(2, 5, 4, self.video.structured_dim, dtype=torch.bool)
        available[:, :, 2:, 186:] = False
        video = {
            "structured": torch.randn(2, 5, 4, self.video.structured_dim),
            "structured_available": available,
            "technique_available": torch.ones(2, 5, dtype=torch.bool),
            "segment_id": torch.zeros(2, 5, 4, dtype=torch.long),
            "frame_indices": torch.tensor([[0, 1, 2, 3, 4, 4, 4]]).expand(2, -1),
        }
        for kwargs in ({}, {"video": video}):
            with self.subTest(video=bool(kwargs)), torch.no_grad():
                expected = self.model(*inputs, **kwargs)
                actual = restored(*inputs, **kwargs)
                repeated = restored(*inputs, **kwargs)
            for name in expected:
                self.assertTrue(torch.equal(expected[name], actual[name]), name)
                self.assertTrue(torch.equal(repeated[name], actual[name]), name)
                self.assertFalse(actual[name].requires_grad)

    def test_audio_model_export_has_no_video_branch(self):
        model = FingerstyleTranscriber(self.config)
        self.payload.update(video_config=None, model_state=dict(model.state_dict()))
        restored, checkpoint, _ = checkpoint_model(self.save(), "cpu")
        self.assertIsInstance(restored, FingerstyleTranscriber)
        self.assertNotIn("video", runtime.checkpoint_identity(checkpoint))

    def test_inference_artifact_never_supplies_resume_or_evaluation_identity(self):
        self.save()
        for kwargs in ({}, {"allow_inference": True, "expected_identity": {"model": {}}}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "cannot resume"):
                runtime.load_checkpoint(self.path, **kwargs)
        with self.assertRaisesRegex(ValueError, "cannot resume"):
            checkpoint_model(self.path, "cpu", allow_inference=False)

    def test_strict_version_config_and_state_validation(self):
        cases = []
        for key, value in (("schema_version", True), ("schema_version", 99), ("format", "other"), ("extra", "not permitted")):
            payload = deepcopy(self.payload)
            payload[key] = value
            cases.append(payload)
        for config_key in ("model_config", "feature_config", "video_config"):
            payload = deepcopy(self.payload)
            payload[config_key]["extra"] = 1
            cases.append(payload)
            payload = deepcopy(self.payload)
            payload[config_key].pop(next(iter(payload[config_key])))
            cases.append(payload)
        payload = deepcopy(self.payload)
        payload["feature_config"]["n_mels"] = 32
        cases.append(payload)
        name = next(iter(self.payload["model_state"]))
        for value in (torch.zeros(1), self.payload["model_state"][name].double(),
                      torch.full_like(self.payload["model_state"][name], float("nan"))):
            payload = deepcopy(self.payload)
            payload["model_state"][name] = value
            cases.append(payload)
        payload = deepcopy(self.payload)
        del payload["model_state"][name]
        cases.append(payload)
        for index, payload in enumerate(cases):
            with self.subTest(index=index), self.assertRaises((ValueError, TypeError)):
                runtime.load_checkpoint(self.save(payload), allow_inference=True)

    def test_restricted_load_has_no_unsafe_retry_and_validation_preserves_rng(self):
        self.save()
        with patch.object(runtime.torch, "load", side_effect=pickle.UnpicklingError("unsupported object")) as load:
            with self.assertRaises(pickle.UnpicklingError):
                runtime.load_checkpoint(self.path, allow_inference=True)
        self.assertEqual(load.call_count, 1)
        self.assertIs(load.call_args.kwargs["weights_only"], True)
        before = torch.get_rng_state().clone()
        runtime.load_checkpoint(self.path, allow_inference=True)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


    def test_bundled_artifact_metadata_has_only_allowlisted_strings(self):
        path = ROOT / "models" / "transcriber.pt"
        payload = runtime.load_checkpoint(path, allow_inference=True)
        self.assertLess(path.stat().st_size, 7_000_000)
        allowed = {
            *payload, runtime.INFERENCE_FORMAT, *payload["model_state"],
            *payload["model_config"], *payload["feature_config"], *payload["video_config"],
            "storage", "cpu",
        }
        with zipfile.ZipFile(path) as archive:
            metadata = [name for name in archive.namelist() if name.endswith("/data.pkl")]
            self.assertEqual(len(metadata), 1)
            strings = {
                value for opcode, value, _ in pickletools.genops(archive.read(metadata[0]))
                if opcode.name in ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE", "BINUNICODE8")
            }
        self.assertTrue(all(value in allowed or value.isdecimal() for value in strings))


if __name__ == "__main__":
    unittest.main()
