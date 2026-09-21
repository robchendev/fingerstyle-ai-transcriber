from fractions import Fraction
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import wave

import numpy as np

from core import EvidenceError, PRIVATE_OUTPUT_ROOT, sha256
from hand_motion import load_audio_clock


class MotionAudioTests(unittest.TestCase):
    def test_source_bound_audio_clock_uses_actual_sample_rate(self):
        PRIVATE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="motion-audio-test-", dir=PRIVATE_OUTPUT_ROOT) as directory:
            root = Path(directory)
            audio, alignment = root / "trimmed.wav", root / "alignment.json"
            with wave.open(str(audio), "wb") as stream:
                stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                stream.writeframes(np.zeros(32000, np.int16).tobytes())
            document = {
                "schemaVersion": 1, "kind": "video-to-trimmed-audio-alignment",
                "videoSha256": "a" * 64, "trimmedAudioSha256": sha256(audio),
                "videoStartSecondsForTrimmedAudioZero": 2.92, "sampleRate": 8000,
                "status": "supported", "rate": [1, 1], "correlation": .9,
            }
            alignment.write_text(json.dumps(document))
            clock, duration, _, _ = load_audio_clock("a" * 64, audio, alignment)
            self.assertEqual(clock.sample_rate, 16000)
            self.assertEqual(duration, 2)
            self.assertEqual(clock.map_pts(4004, Fraction(1, 1000)), 17344)
            for key, value in (("videoSha256", "b" * 64), ("trimmedAudioSha256", "b" * 64), ("rate", [2, 1]), ("status", "ambiguous")):
                alignment.write_text(json.dumps({**document, key: value}))
                with self.assertRaises(EvidenceError):
                    load_audio_clock("a" * 64, audio, alignment)
            asset = {
                "sha256": sha256(audio), "sourceSha256": "a" * 64, "sampleRate": 16000,
                "channels": 1, "sampleCount": 32000, "appliedRangeSeconds": [2.92, 4.9199375],
                "sourceSampleBounds": {"startSample": 46720, "stopSampleExclusive": 78720},
            }
            retained = {**document, "method": "retained-source-samples", "correlation": None,
                        "sourceProvenance": {"audioAsset": asset}}
            alignment.write_text(json.dumps(retained))
            retained_clock, retained_duration, _, _ = load_audio_clock("a" * 64, audio, alignment)
            self.assertEqual((retained_clock, retained_duration), (clock, duration))
            for changed in ({**asset, "sha256": "b" * 64}, {**asset, "sampleCount": 32001},
                            {**asset, "sourceSampleBounds": {"startSample": 46721, "stopSampleExclusive": 78721}}):
                alignment.write_text(json.dumps({**retained, "sourceProvenance": {"audioAsset": changed}}))
                with self.assertRaises(EvidenceError):
                    load_audio_clock("a" * 64, audio, alignment)

if __name__ == "__main__":
    unittest.main()
