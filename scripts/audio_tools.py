"""Local media-tool discovery and decoded audio inspection; no acquisition."""

from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess

from .dataset_io import ROOT


class AcquisitionError(Exception):
    """Local audio could not be decoded or inspected reliably."""


def executable(name, location=None):
    if location:
        candidate = Path(location) / (name + (".exe" if os.name == "nt" else ""))
        if candidate.is_file():
            return str(candidate.resolve())
        raise AcquisitionError(f"{name} was not found in the supplied directory.")
    installed = shutil.which(name)
    if installed:
        return installed
    candidates = list((ROOT / ".tools" / "ffmpeg").rglob(f"{name}.exe"))
    if len(candidates) == 1:
        return str(candidates[0])
    raise AcquisitionError(f"Install {name} or supply --ffmpeg-dir.")


def run_media(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    if result.returncode:
        raise AcquisitionError(f"{Path(arguments[0]).stem} failed: {result.stderr.strip()[-1200:]}")
    return result.stdout


def probe(path, ffprobe):
    data = json.loads(run_media([
        ffprobe, "-v", "error", "-select_streams", "a", "-show_entries",
        "stream=codec_name,sample_rate,channels,duration_ts,time_base:format=duration",
        "-of", "json", str(path),
    ]))
    streams = data.get("streams", [])
    if len(streams) != 1:
        raise AcquisitionError("Expected exactly one audio stream.")
    stream = streams[0]
    rate, channels = int(stream["sample_rate"]), int(stream["channels"])
    if rate <= 0 or channels <= 0:
        raise AcquisitionError("Invalid audio sample rate or channel count.")
    count = None
    if "duration_ts" in stream:
        samples = Fraction(stream["duration_ts"]) * Fraction(stream["time_base"]) * rate
        if samples.denominator != 1:
            raise AcquisitionError("Audio sample count is not integral.")
        count = int(samples)
    return {"codec": stream["codec_name"], "sampleRate": rate, "channels": channels, "sampleCount": count}
