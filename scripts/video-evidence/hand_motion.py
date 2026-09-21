"""Bounded, audio-aligned motion review from cached hand-role observations."""

from dataclasses import asdict, dataclass
from fractions import Fraction
import json
import math
from pathlib import Path

import av
import numpy as np

from core import ClockMapping, EvidenceError, sha256
from catalog import normalize_audio_asset
from geometry import _cleanup, _publish, _safe_directory
from hand_roles import PALM, TIPS, load_role_inputs, load_role_observations


FEATURE_NAMES = ("palm", "thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip")
ROLE_ORDER = ("fretting", "plucking")
MOTION_REASONS = {
    "outside_clip": 0, "role_unavailable": 1, "coordinates_unavailable": 2,
    "segment_start": 3, "continuous": 4, "detector_change": 5, "timestamp_gap": 6,
}


@dataclass(frozen=True)
class MotionConfig:
    maximum_gap_seconds: float = .09
    smoothing_radius_seconds: float = .06

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise EvidenceError(f"{name} must be a positive finite number.")


def validate_clips(clips, pts, shots, *, allow_partial_end=False):
    if not isinstance(clips, list) or not clips:
        raise EvidenceError("Specify at least one bounded motion clip.")
    bounds = set(int(value) for value in pts)
    end_bounds = bounds | {shots["shots"][-1]["endPtsExclusive"]}
    selected = np.zeros(len(pts), bool)
    clip_ids = np.full(len(pts), -1, np.int32)
    for index, clip in enumerate(clips):
        if not isinstance(clip, dict) or set(clip) != {"label", "startPts", "endPtsExclusive"}:
            raise EvidenceError("Each clip needs label, startPts and endPtsExclusive.")
        start, end = clip["startPts"], clip["endPtsExclusive"]
        if not isinstance(clip["label"], str) or not clip["label"].strip() or type(start) is not int or type(end) is not int or start >= end:
            raise EvidenceError("Clips need a nonempty label and increasing integer PTS boundaries.")
        if start not in bounds or (end not in end_bounds and not allow_partial_end):
            raise EvidenceError("Clip bounds must match source PTS or the final end-exclusive boundary.")
        mask = (pts >= start) & (pts < end)
        if (selected & mask).any() or mask.sum() < 3:
            raise EvidenceError("Clips must not overlap and must contain at least three frames.")
        selected |= mask
        clip_ids[mask] = index
    return selected, clip_ids


def extract_motion_arrays(hand, roles, shots, clips, config=MotionConfig()):
    pts = roles["pts"]
    if not np.array_equal(hand["pts"], pts) or not np.array_equal(hand["shot_id"], roles["shot_id"]):
        raise EvidenceError("Motion hand and role timestamps/shot IDs must match exactly.")
    if len(pts) == 0 or np.any(np.diff(pts) <= 0):
        raise EvidenceError("Motion timestamps must be strictly increasing.")
    selected, clip_ids = validate_clips(clips, pts, shots)
    count = len(pts)
    shape = (count, 2, len(FEATURE_NAMES))
    output = {
        "pts": pts.copy(), "shot_id": roles["shot_id"].copy(), "clip_id": clip_ids,
        "selected": selected, "role_available": np.zeros((count, 2), bool),
        "track_id": np.full((count, 2), -1, np.int32),
        "position_raw": np.full((*shape, 2), np.nan, np.float32),
        "position_smooth": np.full((*shape, 2), np.nan, np.float32),
        "velocity_raw": np.full((*shape, 2), np.nan, np.float32),
        "velocity": np.full((*shape, 2), np.nan, np.float32),
        "smoothing_residual": np.full((*shape, 2), np.nan, np.float32),
        "available": np.zeros(shape, bool), "velocity_available": np.zeros(shape, bool),
        "smoothed": np.zeros(shape, bool), "segment_id": np.full(shape, -1, np.int32),
        "reason": np.full(shape, MOTION_REASONS["outside_clip"], np.int8),
        "source_slot": np.full((count, 2), -1, np.int8),
    }
    output["reason"][selected] = MOTION_REASONS["role_unavailable"]
    for i in np.flatnonzero(selected):
        for slot in range(2):
            role = int(roles["role"][i, slot])
            if role == 0:
                continue
            if role not in (1, 2) or output["role_available"][i, role - 1] or roles["track_id"][i, slot] < 0:
                raise EvidenceError("Motion requires unique roles with valid track identities.")
            target = role - 1
            output["role_available"][i, target] = True
            output["track_id"][i, target] = roles["track_id"][i, slot]
            output["source_slot"][i, target] = slot
            output["reason"][i, target] = MOTION_REASONS["coordinates_unavailable"]
            coordinates = roles["guitar_landmarks"][i, slot]
            palm = coordinates[list(PALM)]
            palm = palm[np.isfinite(palm).all(-1)]
            if len(palm) >= 3:
                output["position_raw"][i, target, 0] = np.median(palm, axis=0)
            output["position_raw"][i, target, 1:] = coordinates[list(TIPS)]
    output["available"] = np.isfinite(output["position_raw"]).all(-1)
    time_base = Fraction(*shots["timeBase"])
    seconds = np.asarray([float(int(value) * time_base) for value in pts])
    next_segment = 0
    for role in range(2):
        for feature in range(len(FEATURE_NAMES)):
            previous = None
            for i in np.flatnonzero(output["available"][:, role, feature]):
                same_track = previous is not None and i == previous + 1 and clip_ids[i] == clip_ids[previous] and roles["shot_id"][i] == roles["shot_id"][previous] and output["track_id"][i, role] == output["track_id"][previous, role]
                reason = MOTION_REASONS["segment_start"]
                if same_track:
                    if seconds[i] - seconds[previous] > config.maximum_gap_seconds:
                        reason = MOTION_REASONS["timestamp_gap"]
                    elif hand["detection_source"][i] != hand["detection_source"][previous]:
                        reason = MOTION_REASONS["detector_change"]
                    else:
                        reason = MOTION_REASONS["continuous"]
                if reason != MOTION_REASONS["continuous"]:
                    segment = next_segment
                    next_segment += 1
                output["segment_id"][i, role, feature] = segment
                output["reason"][i, role, feature] = reason
                previous = i
            segments = output["segment_id"][:, role, feature]
            for segment in np.unique(segments[segments >= 0]):
                indices = np.flatnonzero(segments == segment)
                raw = output["position_raw"][indices, role, feature]
                smoothed = raw.copy()
                # Centered local linear fits preserve a steady trajectory on irregular PTS.
                for j in range(1, len(indices) - 1):
                    center = seconds[indices[j]]
                    neighbors = np.flatnonzero(np.abs(seconds[indices] - center) <= config.smoothing_radius_seconds)
                    if len(neighbors) < 3 or not (neighbors[0] < j < neighbors[-1]):
                        continue
                    delta = seconds[indices[neighbors]] - center
                    matrix = np.column_stack((np.ones(len(delta)), delta))
                    smoothed[j] = np.linalg.lstsq(matrix, raw[neighbors], rcond=None)[0][0]
                    output["smoothed"][indices[j], role, feature] = True
                output["position_smooth"][indices, role, feature] = smoothed
                output["smoothing_residual"][indices, role, feature] = raw - smoothed
                if len(indices) > 1:
                    elapsed = np.diff(seconds[indices])[:, None]
                    output["velocity_raw"][indices[1:], role, feature] = np.diff(raw, axis=0) / elapsed
                    output["velocity"][indices[1:], role, feature] = np.diff(smoothed, axis=0) / elapsed
                    output["velocity_available"][indices[1:], role, feature] = True
    return output


def load_audio_clock(video_hash, audio_path, alignment_path):
    audio_path, alignment_path = Path(audio_path), Path(alignment_path)
    hashes = {"trimmedAudio": sha256(audio_path), "alignment": sha256(alignment_path)}
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    if alignment.get("kind") != "video-to-trimmed-audio-alignment" or type(alignment.get("schemaVersion")) is not int or alignment["schemaVersion"] != 1:
        raise EvidenceError("Motion review requires a supported video/audio alignment report.")
    if alignment.get("videoSha256") != video_hash or alignment.get("trimmedAudioSha256") != hashes["trimmedAudio"]:
        raise EvidenceError("Alignment report does not match source video and trimmed audio.")
    if alignment.get("status") != "supported" or alignment.get("rate") != [1, 1] or any(type(value) is not int for value in alignment["rate"]):
        raise EvidenceError("Motion pilot requires a supported unit-rate audio alignment.")
    offset = alignment.get("videoStartSecondsForTrimmedAudioZero")
    if type(offset) not in (int, float) or not math.isfinite(offset):
        raise EvidenceError("Alignment offset must be finite.")
    asset = None
    if alignment.get("method") == "retained-source-samples":
        provenance = alignment.get("sourceProvenance")
        if not isinstance(provenance, dict):
            raise EvidenceError("Retained-source alignment requires its extraction provenance.")
        asset = normalize_audio_asset(provenance.get("audioAsset"))
        if (asset["sha256"] != hashes["trimmedAudio"] or alignment.get("correlation") is not None
                or offset != asset["sourceSampleBounds"]["startSample"] / asset["sampleRate"]):
            raise EvidenceError("Retained-source alignment disagrees with its saved audio samples.")
    else:
        correlation = alignment.get("correlation")
        if type(correlation) not in (int, float) or not math.isfinite(correlation) or not -1 <= correlation <= 1:
            raise EvidenceError("Alignment correlation must be finite and between -1 and 1.")
    with av.open(str(audio_path)) as container:
        if len(container.streams.audio) != 1:
            raise EvidenceError("Motion review requires exactly one trimmed-audio stream.")
        stream = container.streams.audio[0]
        if not stream.sample_rate or stream.duration is None or stream.time_base is None:
            raise EvidenceError("Trimmed audio must expose its sample rate and duration.")
        sample_rate = stream.sample_rate
        duration = Fraction(stream.duration) * stream.time_base
    if asset is not None and (sample_rate != asset["sampleRate"] or duration * sample_rate != asset["sampleCount"]):
        raise EvidenceError("Retained-source clock differs from actual audio sample rate or count.")
    offset_samples = -asset["sourceSampleBounds"]["startSample"] if asset is not None else round(-Fraction(str(offset)) * sample_rate)
    clock = ClockMapping(sample_rate, offset_samples)
    return clock, duration, alignment, hashes


def _motion_summary(motion, clip):
    selected = (motion["pts"] >= clip["startPts"]) & (motion["pts"] < clip["endPtsExclusive"])
    rows = {}
    for role, name in enumerate(ROLE_ORDER):
        available = motion["available"][selected, role]
        velocity = motion["velocity"][selected, role]
        rows[name] = {
            "framesWithRole": int(motion["role_available"][selected, role].sum()),
            "framesWithPalmCoordinates": int(available[:, 0].sum()),
            "framesWithPalmVelocity": int(motion["velocity_available"][selected, role, 0].sum()),
            "framesWithAnyTipCoordinates": int(available[:, 1:].any(-1).sum()),
            "featureSegmentCount": int(len(np.unique(motion["segment_id"][selected, role][available]))),
            "palmAlongNeckSpeedP95": float(np.nanquantile(np.abs(velocity[:, 0, 0]), .95)) if np.isfinite(velocity[:, 0, 0]).any() else None,
            "tipAcrossBoardSpeedP95": float(np.nanquantile(np.abs(velocity[:, 1:, 1]), .95)) if np.isfinite(velocity[:, 1:, 1]).any() else None,
        }
    return {"frameCount": int(selected.sum()), "roles": rows}


def build_motion_pilot(video_path, shots_path, hands_path, geometry_path, annotations_path, roles_path, audio_path, alignment_path, output_directory, clips, config=MotionConfig()):
    if not isinstance(config, MotionConfig):
        raise EvidenceError("Motion configuration must be MotionConfig.")
    inputs = load_role_inputs(video_path, shots_path, hands_path, geometry_path, annotations_path)
    roles_path = Path(roles_path)
    roles_hash = sha256(roles_path)
    role_report, roles = load_role_observations(inputs, roles_path)
    clock, audio_duration, alignment, audio_hashes = load_audio_clock(inputs["hashes"]["video"], audio_path, alignment_path)
    shots = inputs["documents"]["shots"]
    motion = extract_motion_arrays(inputs["hand"], roles, shots, clips, config)
    time_base = Fraction(*shots["timeBase"])
    motion["trimmed_audio_sample"] = np.asarray([clock.map_pts(int(pts), time_base) for pts in motion["pts"]], np.int64)
    for clip in clips:
        start = clock.map_pts(clip["startPts"], time_base)
        end = clock.map_pts(clip["endPtsExclusive"], time_base)
        if start < 0 or Fraction(end, clock.sample_rate) > audio_duration:
            raise EvidenceError("Motion clip falls outside the aligned trimmed audio.")
    output = _safe_directory(output_directory, "motion pilot")
    complete = False
    try:
        from motion_review import render_motion_clips

        np.savez_compressed(output / "motion.npz", **motion)
        rendered = render_motion_clips(video_path, audio_path, output, clips, time_base, inputs["hand"], inputs["geometry"], roles, motion, clock)
        for clip, artifact in zip(clips, rendered):
            artifact.update(_motion_summary(motion, clip))
        if len(rendered) != len(clips):
            raise EvidenceError("Motion renderer did not return all requested clips.")
        if any(sha256(path) != inputs["hashes"][name] for name, path in inputs["paths"].items()) or sha256(roles_path) != roles_hash or sha256(roles_path.with_name("roles.npz")) != role_report["arraysSha256"]:
            raise EvidenceError("An observation input changed during motion review.")
        if sha256(audio_path) != audio_hashes["trimmedAudio"] or sha256(alignment_path) != audio_hashes["alignment"]:
            raise EvidenceError("Audio or alignment changed during motion review.")
        adjacent = motion["selected"][1:] & motion["selected"][:-1] & (motion["clip_id"][1:] == motion["clip_id"][:-1])
        intervals = np.diff(motion["pts"])[adjacent] * float(time_base)
        report = {
            "schemaVersion": 1, "kind": "guitar-hand-motion-pilot", "visibility": "private",
            "inputSha256": {**inputs["hashes"], **audio_hashes, "roles": roles_hash, "roleArrays": role_report["arraysSha256"]},
            "timeBase": shots["timeBase"], "roleOrder": list(ROLE_ORDER), "featureOrder": list(FEATURE_NAMES),
            "reasonEncoding": MOTION_REASONS, "config": asdict(config),
            "clock": {"sampleRate": clock.sample_rate, "offsetSamples": clock.offset_samples, "rate": [1, 1]},
            "alignmentCorrelation": alignment["correlation"],
            "frameCount": len(motion["pts"]), "selectedFrameCount": int(motion["selected"].sum()),
            "medianFrameIntervalSeconds": float(np.median(intervals)),
            "maximumFrameIntervalSeconds": float(intervals.max()),
            "clips": rendered, "arrays": "motion.npz", "arraysSha256": sha256(output / "motion.npz"),
            "positionUnits": ["neckBody-to-nut lengths", "fretboard widths"],
            "velocityUnits": ["neckBody-to-nut lengths/second", "fretboard widths/second"],
            "smoothingPolicy": "Centered local linear fit within each continuous feature/role/track/shot/clip/detector segment. Raw values retained. No filling missing points; edges stay raw.",
            "velocityPolicy": "Backward difference on actual PTS. First point and gaps unavailable; raw and smoothed velocities retained. Smoothing can attenuate brief real gestures.",
            "limitations": [
                "Hand role availability does not imply guitar-coordinate availability.",
                "Landmark jitter and optical-flow geometry errors can resemble real movement.",
                "Soundtrack alignment is not proof of identical physical take or local movement/audio synchrony.",
                "These trajectories are not string contact, exact fret positions, attack timestamps or musical technique labels.",
                "Frame sampling and smoothing limit timing resolution; brief contacts can occur between frames.",
            ],
            "repeatMatchingPolicy": "Future repeated-passage correspondence uses audio, not visual camera-angle matching.",
            "implementationSha256": {name: sha256(Path(__file__).with_name(name)) for name in ("hand_motion.py", "motion_review.py", "hand_roles.py", "core.py")},
            "trainingPerformed": False, "detectorRerun": False, "transcriptionModified": False,
            "sameTakeConfirmed": False, "reviewRequired": True,
        }
        _publish(output / "motion.json", report)
        complete = True
        return output / "motion.json", report
    finally:
        if not complete:
            _cleanup(output)
