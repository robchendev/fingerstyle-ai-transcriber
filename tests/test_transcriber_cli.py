from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from scripts import transcriber
from scripts.catalogs import ROOT, read_json, write_json
from scripts.transcriber_audio import FeatureConfig, HarnessError


class HarnessCommandTests(unittest.TestCase):
    def test_config_path_roundtrip_and_worker_contract(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = transcriber.default_config()
            write_json(path, config)
            loaded, features, model, training = transcriber.load_config(str(path))
            self.assertEqual(loaded, config)
            self.assertEqual(features.n_mels, model.n_mels)
            config["data"]["num_workers"] = 1
            write_json(path, config)
            with self.assertRaisesRegex(HarnessError, "num_workers"):
                transcriber.load_config(str(path))

    def metadata(self):
        return {"openStringMidi": [40, 45, 50, 55, 59, 64], "capoFret": 0, "tempo": {"bpm": 100, "beatUnit": [1, 4]}, "timeSignature": [4, 4]}

    def test_training_requires_intentional_human_invocation(self):
        with self.assertRaisesRegex(HarnessError, "human owner"):
            transcriber.train(SimpleNamespace(human_run=False))

    def test_inference_metadata_never_defaults_missing_tuning_or_timing(self):
        value = self.metadata()
        tempos, meters = transcriber.inference_metadata(value)
        self.assertEqual(tempos[0]["bpm"], 100)
        self.assertEqual(meters[0]["timeSignature"], [4, 4])
        for field in value:
            invalid = dict(value)
            del invalid[field]
            with self.subTest(field=field), self.assertRaises(HarnessError):
                transcriber.inference_metadata(invalid)
        for change in ({"tempo": None}, {"tempoChanges": [{"bpm": 120}]}, {"timeSignatureChanges": None}, {"capoChanges": []}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                transcriber.inference_metadata({**value, **change})

    def test_output_must_remain_in_private_runtime_directories(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.assertEqual(transcriber.private_output("runs/result.json", root), root / "runs" / "result.json")
            for path in ("data/output.json", "docs/output.json", "..\\public.json"):
                with self.subTest(path=path), self.assertRaises(HarnessError):
                    transcriber.private_output(path, root)

    def test_whole_audio_stitching_has_no_uncovered_or_duplicated_frame_positions(self):
        class ConstantModel(torch.nn.Module):
            def forward(self, features, conditioning, lengths):
                return {"note_onset_logits": torch.full((1, features.shape[1], 6), 2., device=features.device)}

        with TemporaryDirectory(prefix="harness-inference-test-", dir=ROOT) as directory:
            root = Path(directory)
            rate = 8000
            samples = (.1 * np.sin(2 * np.pi * 440 * np.arange(rate * 14) / rate)).astype(np.float32)
            audio = root / "synthetic.wav"
            sf.write(audio, samples, rate)
            metadata = root / "metadata.json"
            write_json(metadata, self.metadata())
            checkpoint_path = root / "synthetic.pt"
            checkpoint_path.write_bytes(b"synthetic checkpoint identity")
            feature_config = FeatureConfig(sample_rate=rate, n_fft=512, hop_length=160, n_mels=16, f_max=3000.)
            captured = {}

            def decode(outputs, times, **kwargs):
                captured.update(outputs=outputs, times=times)
                return {"notes": [], "percussion": [], "policy": {"synthetic": True}}

            args = SimpleNamespace(checkpoint=str(checkpoint_path), device="cpu", metadata=str(metadata), audio=str(audio), output="runs/prediction.json", data_root=root, onset_threshold=.5, percussion_threshold=.5)
            with patch.object(transcriber, "checkpoint_model", return_value=(ConstantModel(), {"identity": {"features": asdict(feature_config)}}, torch.device("cpu"))), patch.dict("sys.modules", {"scripts.transcriber_model": SimpleNamespace(decode_events=decode)}):
                result = transcriber.infer(args)
            self.assertEqual(len(captured["times"]), 700)
            torch.testing.assert_close(captured["outputs"]["note_onset_logits"], torch.full((700, 6), 2.))
            self.assertFalse(result["gpWriterImplemented"])
            self.assertFalse(result["modelTrainingPerformedByThisCommand"])
            self.assertEqual(read_json(root / "runs" / "prediction.json")["visibility"], "private")


if __name__ == "__main__":
    unittest.main()
