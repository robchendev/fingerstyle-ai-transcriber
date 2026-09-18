"""Validated, model-independent evidence for edited modern fingerstyle video."""

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

FRAME_STATES = frozenset({
    "trackable",
    "guitar_partial",
    "guitar_absent",
    "hands_occluded",
    "calibration_unstable",
    "transition",
    "decode_failure",
    "av_mismatch",
    "unknown",
})

FINGERSTYLE_ACTIONS = frozenset({
    "pluck",
    "thumb_bass",
    "brush",
    "arpeggio_roll",
    "rasgueado",
    "pick_stroke",
    "thumb_slap",
    "wrist_thump",
    "body_tap",
    "string_tap",
    "hammer_on",
    "pull_off",
    "slide",
    "bend",
    "vibrato",
    "harmonic_touch",
})


class EvidenceError(ValueError):
    pass


def sha256(path):
    path = Path(path)
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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


def _fraction(value, name, *, positive=False):
    if not isinstance(value, list) or len(value) != 2:
        raise EvidenceError(f"{name} must be [numerator, denominator].")
    numerator = _integer(value[0], f"{name} numerator")
    denominator = _integer(value[1], f"{name} denominator", 1)
    result = Fraction(numerator, denominator)
    if positive and result <= 0:
        raise EvidenceError(f"{name} must be positive.")
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


def _hand_evidence(hand, geometry, name):
    if hand is None:
        return None
    if not isinstance(hand, dict) or set(hand) != {"role", "anatomicalHandedness", "landmarks", "candidateActions"}:
        raise EvidenceError(f"{name} hand evidence has an invalid schema.")
    if hand["role"] not in ("fretting", "plucking", "unknown"):
        raise EvidenceError(f"{name} hand role is invalid.")
    if hand["anatomicalHandedness"] not in ("left", "right", "unknown"):
        raise EvidenceError(f"{name} anatomical handedness is invalid.")
    landmarks = hand["landmarks"]
    if not isinstance(landmarks, list):
        raise EvidenceError(f"{name} hand landmarks must be a list.")
    actions = hand["candidateActions"]
    if not isinstance(actions, list):
        raise EvidenceError(f"{name} candidate actions must be a list.")
    normalized_actions = []
    for action in actions:
        if not isinstance(action, dict) or set(action) != {"type", "confidence"} or action["type"] not in FINGERSTYLE_ACTIONS:
            raise EvidenceError(f"{name} contains an unsupported fingerstyle action.")
        normalized_actions.append({
            "type": action["type"],
            "confidence": _finite(action["confidence"], f"{name} action confidence", 0, 1),
        })
    return {
        "role": hand["role"],
        "anatomicalHandedness": hand["anatomicalHandedness"],
        "landmarks": [geometry.transform(point) for point in landmarks] if geometry is not None else [],
        "landmarksUnavailableWithoutGeometry": geometry is None and bool(landmarks),
        "candidateActions": normalized_actions,
    }


def _coverage(frames, end_sample):
    if not frames:
        raise EvidenceError("At least one frame observation is required.")
    if end_sample <= frames[-1]["audioSample"]:
        raise EvidenceError("Coverage end must follow the final mapped frame.")
    intervals = []
    for index, frame in enumerate(frames):
        stop = frames[index + 1]["audioSample"] if index + 1 < len(frames) else end_sample
        if stop <= frame["audioSample"]:
            raise EvidenceError("Mapped frame samples must increase strictly.")
        key = (frame["state"], frame["cameraSegmentId"])
        if intervals and (intervals[-1]["state"], intervals[-1]["cameraSegmentId"]) == key:
            intervals[-1]["endAudioSample"] = stop
        else:
            intervals.append({
                "startAudioSample": frame["audioSample"],
                "endAudioSample": stop,
                "state": frame["state"],
                "cameraSegmentId": frame["cameraSegmentId"],
            })
    return intervals


def build_evidence(video_path, audio_path, observations):
    video_path, audio_path = Path(video_path), Path(audio_path)
    if not video_path.is_file() or not audio_path.is_file():
        raise EvidenceError("Evidence requires existing local video and trimmed-audio files.")
    if not isinstance(observations, dict) or set(observations) != {
        "timeBase", "clockMapping", "endPts", "frames", "review"
    }:
        raise EvidenceError("Observation document has an invalid top-level schema.")
    time_base = _fraction(observations["timeBase"], "time base", positive=True)
    clock = observations["clockMapping"]
    if not isinstance(clock, dict) or set(clock) != {"sampleRate", "offsetSamples", "rate"}:
        raise EvidenceError("Clock mapping has an invalid schema.")
    mapping = ClockMapping(
        sample_rate=_integer(clock["sampleRate"], "sample rate", 1),
        offset_samples=_integer(clock["offsetSamples"], "offset samples"),
        rate=_fraction(clock["rate"], "clock rate", positive=True),
    )
    raw_frames = observations["frames"]
    if not isinstance(raw_frames, list) or not raw_frames:
        raise EvidenceError("Frames must be a nonempty list.")
    frames = []
    prior_pts = None
    segment = -1
    for index, frame in enumerate(raw_frames):
        expected = {"pts", "cutBefore", "state", "guitarGeometry", "frettingHand", "pluckingHand", "sync"}
        if not isinstance(frame, dict) or set(frame) != expected:
            raise EvidenceError(f"Frame {index} has an invalid schema.")
        pts = _integer(frame["pts"], f"frame {index} PTS")
        if prior_pts is not None and pts <= prior_pts:
            raise EvidenceError("Frame PTS values must increase strictly.")
        if type(frame["cutBefore"]) is not bool or index == 0 and frame["cutBefore"] is not True:
            raise EvidenceError("The first frame must start a camera segment; cutBefore must be boolean.")
        if frame["cutBefore"]:
            segment += 1
        state = frame["state"]
        if state not in FRAME_STATES:
            raise EvidenceError(f"Unsupported frame state: {state}")
        geometry = None
        if frame["guitarGeometry"] is not None:
            geometry = GuitarCoordinateFrame.from_landmarks(frame["guitarGeometry"])
        if state == "trackable" and geometry is None:
            raise EvidenceError("Trackable frames require guitar geometry.")
        sync = frame["sync"]
        if not isinstance(sync, dict) or set(sync) != {"status", "lagSeconds", "support", "ambiguity"}:
            raise EvidenceError("Frame synchronization evidence has an invalid schema.")
        if sync["status"] not in ("supported", "contradicted", "unassessable"):
            raise EvidenceError("Synchronization status is invalid.")
        normalized_sync = {
            "status": sync["status"],
            "lagSeconds": None if sync["lagSeconds"] is None else _finite(sync["lagSeconds"], "synchronization lag"),
            "support": _integer(sync["support"], "synchronization support", 0),
            "ambiguity": _finite(sync["ambiguity"], "synchronization ambiguity", 0),
        }
        if sync["status"] == "unassessable" and sync["lagSeconds"] is not None:
            raise EvidenceError("Unassessable synchronization cannot claim a lag.")
        audio_sample = mapping.map_pts(pts, time_base)
        frames.append({
            "pts": pts,
            "audioSample": audio_sample,
            "audioSeconds": audio_sample / mapping.sample_rate,
            "cutBefore": frame["cutBefore"],
            "cameraSegmentId": segment,
            "trackingGeneration": segment,
            "state": state,
            "guitarGeometryConfidence": None if geometry is None else geometry.confidence,
            "frettingHand": _hand_evidence(frame["frettingHand"], geometry, "fretting"),
            "pluckingHand": _hand_evidence(frame["pluckingHand"], geometry, "plucking"),
            "sync": normalized_sync,
        })
        prior_pts = pts
    end_pts = _integer(observations["endPts"], "end PTS")
    if end_pts <= raw_frames[-1]["pts"]:
        raise EvidenceError("endPts must follow the final frame.")
    end_sample = mapping.map_pts(end_pts, time_base)
    review = observations["review"]
    if not isinstance(review, dict) or set(review) != {"timingReviewed", "coverageReviewed", "sameTakeConfirmed"}:
        raise EvidenceError("Review status has an invalid schema.")
    if any(type(review[name]) is not bool for name in review):
        raise EvidenceError("Review statuses must be boolean.")
    return {
        "schemaVersion": 1,
        "kind": "fingerstyle-video-evidence",
        "visibility": "private",
        "trainingPerformed": False,
        "videoSha256": sha256(video_path),
        "trimmedAudioSha256": sha256(audio_path),
        "clock": {
            "timeBase": [time_base.numerator, time_base.denominator],
            "sampleRate": mapping.sample_rate,
            "offsetSamples": mapping.offset_samples,
            "rate": [mapping.rate.numerator, mapping.rate.denominator],
            "mapping": "trimmedAudioSeconds = rate * (videoPTS * timeBase) + offsetSamples / sampleRate",
        },
        "review": dict(review),
        "sameTakeInferencePolicy": "Soundtrack alignment and local motion agreement do not prove same-take fingering.",
        "frames": frames,
        "coverage": _coverage(frames, end_sample),
    }
