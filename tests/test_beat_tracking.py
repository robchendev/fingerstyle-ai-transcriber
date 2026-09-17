from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from scripts.beat_tracking import (
    HOP_LENGTH,
    N_MELS,
    SAMPLE_RATE,
    _peak_times,
    _predict,
    _read_channel,
    log_mel_spectrogram,
    mel_filter_bank,
    track_beats,
)
from scripts.dataset_io import ROOT
from scripts.transcriber_audio import HarnessError


class BeatTrackingTests(unittest.TestCase):
    def test_slaney_features_are_finite_and_have_expected_shape(self):
        signal = np.sin(2 * np.pi * 110 * np.arange(SAMPLE_RATE) / SAMPLE_RATE).astype(np.float32)
        features = log_mel_spectrogram(signal)
        self.assertEqual(features.shape[1], N_MELS)
        self.assertTrue(torch.isfinite(features).all())
        filters = mel_filter_bank()
        self.assertEqual(filters.shape, (513, N_MELS))
        self.assertTrue(torch.all(filters >= 0))

    def test_channel_selection_does_not_cancel_antiphase_audio(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "stereo.wav"
            signal = np.sin(2 * np.pi * 220 * np.arange(8000) / 8000).astype(np.float32)
            sf.write(path, np.stack((signal, -signal), axis=1), 8000)
            selected, source_rate, channels, index, energies, mixed_energy, mode = _read_channel(path)
            self.assertEqual((source_rate, channels, index, mode), (8000, 2, 0, "highest-energy-channel"))
            self.assertGreater(np.max(np.abs(selected)), .9)
            self.assertAlmostEqual(energies[0], energies[1], places=5)
            self.assertLess(mixed_energy, 1e-6)

    def test_short_and_long_chunk_prediction_cover_every_frame(self):
        class Stub(torch.nn.Module):
            def forward(self, features):
                values = torch.arange(features.shape[1], dtype=torch.float32)[None]
                return {"beat": values, "downbeat": values + 1}

        for length in (500, 2000, 4000):
            with self.subTest(length=length):
                beat, downbeat = _predict(Stub(), torch.zeros(length, N_MELS), torch.device("cpu"))
                self.assertEqual(beat.shape, (length,))
                self.assertEqual(downbeat.shape, (length,))
                self.assertTrue(torch.isfinite(beat).all())

    def test_peak_times_deduplicate_adjacent_frames(self):
        logits = torch.full((20,), -5.0)
        logits[4:6] = 3
        logits[15] = 2
        times = _peak_times(logits)
        np.testing.assert_allclose(times, [4.5 * HOP_LENGTH / SAMPLE_RATE, 15 * HOP_LENGTH / SAMPLE_RATE])

    def test_track_beats_binds_audio_checkpoint_and_local_model_results(self):
        class Model:
            pass

        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            audio = directory / "audio.wav"
            checkpoint = directory / "beat.ckpt"
            sf.write(audio, np.sin(2 * np.pi * 110 * np.arange(8000) / 8000), 8000)
            checkpoint.write_bytes(b"local checkpoint")
            beat_logits = torch.full((50,), -5.0)
            downbeat_logits = torch.full((50,), -5.0)
            beat_logits[[0, 25, 49]] = 5
            downbeat_logits[[0, 49]] = 5
            with patch("scripts.beat_tracking._load_model", return_value=(Model(), {"synthetic": True})), patch("scripts.beat_tracking._predict", return_value=(beat_logits, downbeat_logits)):
                report = track_beats(audio, checkpoint)
            self.assertEqual(report["beatCount"], 3)
            self.assertEqual(report["downbeatCount"], 2)
            self.assertEqual(report["analysis"]["modelParameters"], {"synthetic": True})
            self.assertIsNone(report["selectedChannelIndex"])
            self.assertEqual(report["channelSelection"], "channel-mean")
            self.assertFalse(report["trainingPerformed"])

    def test_invalid_audio_and_missing_checkpoint_fail(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            audio = directory / "silent.wav"
            sf.write(audio, np.zeros(8000), 8000)
            with self.assertRaises(HarnessError):
                track_beats(audio, directory / "missing.ckpt")


if __name__ == "__main__":
    unittest.main()
