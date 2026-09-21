from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from scripts.audio_tools import executable, run_media
from scripts.dataset_io import ROOT, read_json, sha256
from scripts.local_media import prepare_soundtrack, soundtrack_timeline


def make_video(video, audio=None, *, offset=1.25, duration=3.5):
    arguments = [
        executable("ffmpeg"), "-v", "error", "-nostdin", "-n",
        "-f", "lavfi", "-i", f"color=c=blue:s=64x48:r=5:d={duration}",
        "-itsoffset", str(offset),
    ]
    arguments += ["-i", str(audio)] if audio else ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=2"]
    run_media([*arguments, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "ffv1", "-c:a", "pcm_s24le", str(video)])


class LocalMediaTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="local-media-tests-", dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.video, self.cache = self.root / "trimmed.mkv", self.root / "owned-cache"

    def test_nonzero_audio_pts_are_preserved_without_trimming_or_repeated_conversion(self):
        make_video(self.video)
        before = sha256(self.video), self.video.stat().st_mtime_ns
        audio, receipt_path = prepare_soundtrack(self.video, self.cache)
        receipt = read_json(receipt_path)
        stream = receipt["stream"]
        origin = stream["firstDecodedPts"] * Fraction(*stream["timeBase"])
        self.assertEqual(origin, Fraction(5, 4))
        self.assertEqual(stream["sampleCount"], 32000)
        self.assertEqual(stream["sampleRate"], 16000)
        self.assertEqual(receipt["videoSha256"], before[0])
        self.assertEqual(receipt["audioSha256"], sha256(audio))
        snapshot = {path: (sha256(path), path.stat().st_mtime_ns) for path in self.cache.rglob("*") if path.is_file()}
        with patch("scripts.local_media.run_media", side_effect=AssertionError("Unchanged sources must not be decoded again")):
            self.assertEqual(prepare_soundtrack(self.video, self.cache), (audio, receipt_path))
            self.assertEqual(prepare_soundtrack(self.video, self.cache, create=False), (audio, receipt_path))
        self.assertEqual(snapshot, {path: (sha256(path), path.stat().st_mtime_ns) for path in self.cache.rglob("*") if path.is_file()})
        self.assertEqual(before, (sha256(self.video), self.video.stat().st_mtime_ns))

    def test_extraction_preserves_every_decoded_sample_and_channel(self):
        source = self.root / "native.flac"
        samples = np.arange(32000, dtype=np.int32).reshape(16000, 2) * 256
        sf.write(source, samples, 8000, subtype="PCM_24")
        make_video(self.video, source)
        audio, receipt = prepare_soundtrack(self.video, self.cache)
        np.testing.assert_array_equal(sf.read(audio, dtype="int32")[0], samples)
        self.assertEqual(read_json(receipt)["stream"]["channels"], 2)
        second, _ = prepare_soundtrack(self.video, self.root / "second-cache")
        self.assertEqual(sha256(audio), sha256(second))

    def test_compressed_mp4_soundtrack_uses_decoded_pts_and_sample_count(self):
        video = self.video.with_suffix(".mp4")
        run_media([
            executable("ffmpeg"), "-v", "error", "-nostdin", "-n",
            "-f", "lavfi", "-i", "color=c=blue:s=64x48:r=5:d=3.5",
            "-itsoffset", "1.25", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=2",
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "mpeg4", "-c:a", "aac", str(video),
        ])
        audio, receipt = prepare_soundtrack(video, self.cache)
        stream = read_json(receipt)["stream"]
        self.assertGreater(stream["firstDecodedPts"], 0)
        self.assertEqual(sf.info(audio).frames, stream["sampleCount"])

    def test_read_only_missing_cache_does_not_create_anything(self):
        make_video(self.video)
        with self.assertRaisesRegex(ValueError, "Run the batch first"):
            prepare_soundtrack(self.video, self.cache, create=False)
        self.assertFalse(self.cache.exists())

    def test_changed_receipt_audio_or_partial_cache_is_not_adopted(self):
        make_video(self.video)
        audio, receipt = prepare_soundtrack(self.video, self.cache)
        original = audio.read_bytes()
        audio.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "soundtrack changed"):
            prepare_soundtrack(self.video, self.cache)
        audio.write_bytes(original)
        document = read_json(receipt)
        document["stream"]["firstDecodedPts"] += 1
        receipt.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "receipt changed"):
            prepare_soundtrack(self.video, self.cache)
        receipt.unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete owned"):
            prepare_soundtrack(self.video, self.cache)

    def test_missing_pts_multiple_streams_and_discontinuous_clocks_are_not_guessed(self):
        document = {
            "streams": [{"index": 1, "sample_rate": "8000", "channels": 1, "time_base": "1/1000"}],
            "frames": [{"stream_index": 1, "pts": 1250, "nb_samples": 800}, {"stream_index": 1, "pts": 1350, "nb_samples": 800}],
        }
        for problem in ("missing-pts", "multiple", "gap", "reset"):
            changed = deepcopy(document)
            if problem == "missing-pts":
                del changed["frames"][0]["pts"]
            elif problem == "multiple":
                changed["streams"].append(dict(changed["streams"][0]))
            else:
                changed["frames"][1]["pts"] = 1400 if problem == "gap" else 0
            with self.subTest(problem=problem), patch("scripts.local_media.run_media", return_value=json.dumps(changed)):
                with self.assertRaises(ValueError):
                    soundtrack_timeline(self.video, "ffprobe")

    def test_failed_decode_never_publishes_or_leaves_an_owned_cache_entry(self):
        make_video(self.video)
        from scripts.local_media import soundtrack_timeline as inspect

        stream = inspect(self.video, executable("ffprobe"))
        with patch("scripts.local_media.soundtrack_timeline", return_value=stream), patch(
            "scripts.local_media.run_media", side_effect=ValueError("decode failure"),
        ):
            with self.assertRaisesRegex(ValueError, "decode failure"):
                prepare_soundtrack(self.video, self.cache)
        self.assertEqual(list(self.cache.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
