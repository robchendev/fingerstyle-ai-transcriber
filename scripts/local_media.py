"""Owned, immutable soundtrack extraction from already-trimmed local video."""

from fractions import Fraction
import json
from pathlib import Path
import shutil
from uuid import uuid4

import soundfile as sf

from .audio_tools import executable, run_media
from .dataset_io import read_json, sha256
from .dataset_release import candidate_digest
from .prepare_training_data import regular_path, write_json


EXTRACTION = "native-pcm24-flac-v1"


def soundtrack_timeline(video, ffprobe):
    data = json.loads(run_media([
        ffprobe, "-v", "error", "-select_streams", "a", "-show_streams", "-show_frames",
        "-show_entries", "stream=index,sample_rate,channels,time_base:frame=stream_index,pts,nb_samples",
        "-of", "json", str(video),
    ]))
    streams, frames = data.get("streams", []), data.get("frames", [])
    if len(streams) != 1 or not frames:
        raise ValueError("Trimmed video must contain exactly one nonempty audio stream; supply explicit audio otherwise.")
    stream = streams[0]
    try:
        index, rate, channels = (int(stream[key]) for key in ("index", "sample_rate", "channels"))
        time_base = Fraction(stream["time_base"])
        first_pts = int(frames[0]["pts"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("The soundtrack needs an explicit decoded PTS clock; supply audio and review alignment otherwise.") from error
    if index < 0 or rate <= 0 or not 1 <= channels <= 8 or time_base <= 0:
        raise ValueError("Unsupported soundtrack stream properties.")
    count, timeline, previous = 0, [], None
    for frame in frames:
        try:
            pts, samples, frame_index = (int(frame[key]) for key in ("pts", "nb_samples", "stream_index"))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Every decoded soundtrack frame needs PTS and a sample count.") from error
        if frame_index != index or samples <= 0 or previous is not None and pts <= previous:
            raise ValueError("Soundtrack frames must belong to one monotonic, nonempty stream.")
        # Container clocks may quantize frame PTS by one tick; do not repair real gaps.
        if abs((pts - first_pts) * time_base - Fraction(count, rate)) > time_base:
            raise ValueError("Discontinuous soundtrack PTS cannot be flattened; supply explicit audio and reviewed alignment.")
        timeline.append([pts, samples])
        count += samples
        previous = pts
    if not 0 < count / rate <= 900:
        raise ValueError("Trimmed video soundtrack must last at most fifteen minutes.")
    return {
        "index": index, "timeBase": [time_base.numerator, time_base.denominator],
        "firstDecodedPts": first_pts, "sampleRate": rate, "channels": channels,
        "sampleCount": count, "frameTimelineSha256": candidate_digest(timeline),
    }


def _cached(directory, video_hash):
    receipt_path = regular_path(directory / "receipt.json")
    audio = regular_path(directory / "soundtrack.flac")
    if not receipt_path.is_file() or not audio.is_file():
        raise ValueError(f"Incomplete owned soundtrack cache: {directory}. Use a new batch output, never adopt or overwrite partial files.")
    receipt = read_json(receipt_path)
    if (not isinstance(receipt, dict) or receipt.get("schemaVersion") != 1
            or receipt.get("kind") != "local-video-soundtrack" or receipt.get("extraction") != EXTRACTION
            or receipt.get("videoSha256") != video_hash or receipt.get("audioFile") != audio.name):
        raise ValueError("The soundtrack receipt does not bind this video and extraction policy.")
    if receipt.get("receiptDigest") != candidate_digest({key: value for key, value in receipt.items() if key != "receiptDigest"}):
        raise ValueError("The soundtrack receipt changed.")
    if receipt.get("audioSha256") != sha256(audio):
        raise ValueError("The owned soundtrack changed; use a new batch output.")
    stream = receipt["stream"]
    with sf.SoundFile(audio) as decoded:
        if (decoded.format != "FLAC" or decoded.subtype != "PCM_24"
                or (decoded.samplerate, decoded.channels, len(decoded))
                != (stream["sampleRate"], stream["channels"], stream["sampleCount"])):
            raise ValueError("Owned soundtrack properties differ from its source stream receipt.")
    return audio, receipt_path


def prepare_soundtrack(video, cache, *, ffmpeg_dir=None, create=True):
    """Extract all decoded samples once, preserving their separate source PTS origin."""
    video, cache = regular_path(video), regular_path(cache)
    if not video.is_file():
        raise ValueError("Soundtrack extraction requires an existing trimmed local video.")
    digest = sha256(video)
    directory = regular_path(cache / candidate_digest({"videoSha256": digest, "extraction": EXTRACTION}))
    if directory.exists():
        return _cached(directory, digest)
    if not create:
        raise ValueError("Run the batch first to extract this trimmed video's soundtrack.")
    ffmpeg, ffprobe = executable("ffmpeg", ffmpeg_dir), executable("ffprobe", ffmpeg_dir)
    stream = soundtrack_timeline(video, ffprobe)
    cache.mkdir(parents=True, exist_ok=True)
    staging = regular_path(cache / f".extracting-{uuid4().hex}")
    staging.mkdir()
    try:
        audio = staging / "soundtrack.flac"
        run_media([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-i", str(video),
            "-map", f"0:{stream['index']}", "-vn", "-map_metadata", "-1",
            "-c:a", "flac", "-sample_fmt", "s32", "-bits_per_raw_sample", "24", str(audio),
        ])
        if sha256(video) != digest:
            raise ValueError("The source video changed during extraction; no soundtrack was adopted.")
        receipt = {
            "schemaVersion": 1, "kind": "local-video-soundtrack", "extraction": EXTRACTION,
            "videoSha256": digest, "audioSha256": sha256(audio), "audioFile": audio.name, "stream": stream,
        }
        receipt["receiptDigest"] = candidate_digest(receipt)
        write_json(staging / "receipt.json", receipt)
        _cached(staging, digest)
        staging.rename(directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return _cached(directory, digest)
