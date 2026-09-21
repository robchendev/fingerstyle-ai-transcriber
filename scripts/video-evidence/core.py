"""Source integrity, rational clocks and coordinate schema primitives."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_OUTPUT_ROOT = REPOSITORY_ROOT / "runs" / "video-evidence"

class EvidenceError(ValueError):
    pass


def sha256(path):
    path = Path(path)
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def video_stream(container):
    """Select one video track, excluding embedded cover artwork."""
    from av.stream import Disposition

    streams = [stream for stream in container.streams.video if not stream.disposition & Disposition.attached_pic]
    if len(streams) != 1:
        raise EvidenceError("Exactly one video stream is required, excluding attached pictures.")
    return streams[0]


def frames_in_shot_range(container, stream, shots):
    first = shots["shots"][0]["startPts"]
    end = shots["shots"][-1]["endPtsExclusive"]
    container.seek(first, stream=stream, backward=True, any_frame=False)
    for frame in container.decode(stream):
        if frame.pts is None:
            raise EvidenceError("Decoded frame has no presentation timestamp.")
        if frame.pts < first:
            continue
        if frame.pts >= end:
            break
        yield frame


def private_output(path):
    root = PRIVATE_OUTPUT_ROOT.resolve()
    value = Path(path)
    value = value if value.is_absolute() else REPOSITORY_ROOT / value
    absolute = value.absolute()
    if absolute.resolve() != absolute or not absolute.is_relative_to(root):
        raise EvidenceError(f"Output must be an unaliased path under {root}.")
    for component in (absolute, *absolute.parents):
        if component.exists() and (component.is_symlink() or getattr(component.lstat(), "st_file_attributes", 0) & 0x400):
            raise EvidenceError(f"Output path aliases another location: {component}")
    if absolute.exists():
        raise EvidenceError(f"Refusing to overwrite existing output: {absolute}")
    return absolute


def publish_json(path, value):
    path = private_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid4().hex}.part")
    try:
        with staging.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)
    return path


def _integer(value, name, minimum=None):
    if type(value) is not int or minimum is not None and value < minimum:
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise EvidenceError(f"{name} must be an integer{suffix}.")
    return value


def _finite(value, name, minimum=None, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise EvidenceError(f"{name} must be finite.")
    result = float(value)
    if minimum is not None and result < minimum or maximum is not None and result > maximum:
        raise EvidenceError(f"{name} is outside its permitted range.")
    return result


def _point(value, name):
    if not isinstance(value, dict) or set(value) != {"x", "y", "confidence"}:
        raise EvidenceError(f"{name} must contain x, y and confidence.")
    return (
        _finite(value["x"], f"{name}.x"),
        _finite(value["y"], f"{name}.y"),
        _finite(value["confidence"], f"{name}.confidence", 0, 1),
    )


def _round_fraction(value):
    if value >= 0:
        return (2 * value.numerator + value.denominator) // (2 * value.denominator)
    return -_round_fraction(-value)


@dataclass(frozen=True)
class ClockMapping:
    sample_rate: int
    offset_samples: int
    rate: Fraction = Fraction(1)

    def __post_init__(self):
        _integer(self.sample_rate, "sample rate", 1)
        _integer(self.offset_samples, "offset samples")
        if not isinstance(self.rate, Fraction) or self.rate <= 0:
            raise EvidenceError("Clock rate must be a positive rational value.")

    def map_pts(self, pts, time_base):
        _integer(pts, "PTS")
        if not isinstance(time_base, Fraction) or time_base <= 0:
            raise EvidenceError("Time base must be a positive rational value.")
        sample = Fraction(pts) * time_base * self.rate * self.sample_rate + self.offset_samples
        return _round_fraction(sample)


@dataclass(frozen=True)
class GuitarCoordinateFrame:
    origin_x: float
    origin_y: float
    neck_x: float
    neck_y: float
    cross_x: float
    cross_y: float
    neck_length: float
    fretboard_width: float
    confidence: float

    @classmethod
    def from_landmarks(cls, landmarks):
        required = {"nut", "neckBody", "fretboardUpper", "fretboardLower"}
        if not isinstance(landmarks, dict) or not required <= set(landmarks):
            raise EvidenceError(f"Guitar geometry requires {sorted(required)}.")
        points = {name: _point(landmarks[name], f"guitarGeometry.{name}") for name in required}
        nut, neck_body = points["nut"], points["neckBody"]
        upper, lower = points["fretboardUpper"], points["fretboardLower"]
        neck_dx, neck_dy = nut[0] - neck_body[0], nut[1] - neck_body[1]
        neck_length = math.hypot(neck_dx, neck_dy)
        cross_dx, cross_dy = lower[0] - upper[0], lower[1] - upper[1]
        projection = (cross_dx * neck_dx + cross_dy * neck_dy) / max(neck_length * neck_length, 1e-12)
        cross_dx -= projection * neck_dx
        cross_dy -= projection * neck_dy
        fretboard_width = math.hypot(cross_dx, cross_dy)
        if neck_length <= 1e-6 or fretboard_width <= 1e-6:
            raise EvidenceError("Guitar geometry axes are degenerate.")
        midpoint_x = (upper[0] + lower[0]) / 2
        midpoint_y = (upper[1] + lower[1]) / 2
        confidence = min(point[2] for point in points.values())
        return cls(
            origin_x=midpoint_x,
            origin_y=midpoint_y,
            neck_x=neck_dx / neck_length,
            neck_y=neck_dy / neck_length,
            cross_x=cross_dx / fretboard_width,
            cross_y=cross_dy / fretboard_width,
            neck_length=neck_length,
            fretboard_width=fretboard_width,
            confidence=confidence,
        )

    def transform(self, point):
        x, y, confidence = _point(point, "hand landmark")
        dx, dy = x - self.origin_x, y - self.origin_y
        return {
            "alongNeck": (dx * self.neck_x + dy * self.neck_y) / self.neck_length,
            "acrossFretboard": (dx * self.cross_x + dy * self.cross_y) / self.fretboard_width,
            "confidence": min(confidence, self.confidence),
        }
