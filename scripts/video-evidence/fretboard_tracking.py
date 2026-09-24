"""Scene-aware fretboard detection and optical-flow tracking."""

from dataclasses import asdict, dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import sys

import av
import cv2
import numpy as np

from core import EvidenceError, frames_in_shot_range, sha256, video_stream
from geometry import _cleanup, _publish, _safe_directory


_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from scripts.fretboard_detector import FretboardDetectorConfig, YoloFretboardDetector
from scripts.fretboard_geometry import FretboardGeometry, FretboardGeometryError


SOURCE_CODES = {"unavailable": 0, "detector": 1, "optical_flow": 2}
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


@dataclass(frozen=True)
class TrackingConfig:
    detector_interval_seconds: float = .5
    reacquire_interval_frames: int = 6
    flow_width: int = 1280
    flow_error_pixels: float = 2.
    minimum_flow_points: int = 4

    def __post_init__(self):
        if not isinstance(self.detector_interval_seconds, (int, float)) or self.detector_interval_seconds <= 0:
            raise ValueError("detector_interval_seconds must be positive.")
        for name, minimum in (("reacquire_interval_frames", 1), ("flow_width", 320), ("minimum_flow_points", 4)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if not isinstance(self.flow_error_pixels, (int, float)) or self.flow_error_pixels <= 0:
            raise ValueError("flow_error_pixels must be positive.")


def _gray(image, width):
    scale = min(1., width / image.shape[1])
    size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
    resized = image if size == (image.shape[1], image.shape[0]) else cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), np.asarray(size, np.float32) / np.asarray([image.shape[1], image.shape[0]], np.float32)


def track_keypoints(previous_gray, current_gray, points, available, previous_scale, current_scale,
                    frame_size, config):
    source = np.ascontiguousarray(points.reshape(-1, 2) * previous_scale, dtype=np.float32)
    flags = available.reshape(-1)
    indices = np.flatnonzero(flags)
    if len(indices) < config.minimum_flow_points:
        return None
    forward, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, source[indices, None], None,
        winSize=(31, 31), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01),
    )
    if forward is None or status is None:
        return None
    backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray, previous_gray, forward, None,
        winSize=(31, 31), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01),
    )
    if backward is None or reverse_status is None:
        return None
    errors = np.linalg.norm(backward[:, 0] - source[indices], axis=1)
    valid = status[:, 0].astype(bool) & reverse_status[:, 0].astype(bool) & (errors <= config.flow_error_pixels)
    if int(valid.sum()) < config.minimum_flow_points:
        return None
    matrix, inliers = cv2.findHomography(
        source[indices][valid], forward[valid, 0], cv2.RANSAC, config.flow_error_pixels,
    )
    if matrix is None or inliers is None or int(inliers.sum()) < config.minimum_flow_points:
        return None
    mapped = cv2.perspectiveTransform(source[:, None], matrix)[:, 0] / current_scale
    mapped_available = flags.copy()
    mapped_available[indices[~valid]] = False
    mapped = mapped.reshape(3, 2, 2)
    mapped_available = mapped_available.reshape(3, 2)
    try:
        geometry = FretboardGeometry.from_keypoints(mapped, mapped_available, frame_size)
    except FretboardGeometryError:
        return None
    widths = np.linalg.norm(mapped[:, 1] - mapped[:, 0], axis=1)
    error = float(np.median(errors[valid]) / max(float(np.median(widths)), 1.))
    return mapped.astype(np.float32), mapped_available, geometry, error


def track_fretboard(video_path, shots_path, model_path, output_directory,
                    tracking_config=TrackingConfig(), detector_config=FretboardDetectorConfig(),
                    detector=None):
    video_path, shots_path, model_path = map(Path, (video_path, shots_path, model_path))
    shots = json.loads(shots_path.read_text(encoding="utf-8"))
    detector = detector or YoloFretboardDetector(model_path, detector_config)
    output = _safe_directory(output_directory, "fretboard tracking")
    complete = False
    rows = {name: [] for name in ("pts", "shot_id", "keypoints", "available", "confidence", "source", "age_seconds", "flow_error", "detector_anchor")}
    detector_calls = accepted_calls = tracked_frames = 0
    container = av.open(str(video_path))
    try:
        stream = video_stream(container)
        time_base = Fraction(stream.time_base)
        shot_index = 0
        previous = None
        last_detector_pts = None
        last_anchor_pts = None
        frames_since_attempt = tracking_config.reacquire_interval_frames
        for frame in frames_in_shot_range(container, stream, shots):
            pts = int(frame.pts)
            cut = False
            while shot_index + 1 < len(shots["shots"]) and pts >= shots["shots"][shot_index + 1]["startPts"]:
                shot_index += 1
                cut = True
            if cut:
                previous = None
                last_detector_pts = last_anchor_pts = None
                frames_since_attempt = tracking_config.reacquire_interval_frames
            image = frame.to_ndarray(format="bgr24")
            frame_size = (image.shape[1], image.shape[0])
            gray, scale = _gray(image, tracking_config.flow_width)
            tracked = None
            if previous is not None:
                tracked = track_keypoints(
                    previous["gray"], gray, previous["keypoints"], previous["available"],
                    previous["scale"], scale, frame_size, tracking_config,
                )
            elapsed = math.inf if last_detector_pts is None else float((pts - last_detector_pts) * time_base)
            due = elapsed >= tracking_config.detector_interval_seconds
            if tracked is None:
                due |= frames_since_attempt >= tracking_config.reacquire_interval_frames
            detection = None
            if due:
                detector_calls += 1
                detection = detector.detect(image)
                last_detector_pts = pts
                frames_since_attempt = 0
            else:
                frames_since_attempt += 1
            if detection is not None and detection.score >= detector_config.acceptance_score:
                keypoints, available = detection.keypoints, detection.available
                confidence, source, flow_error = detection.score, SOURCE_CODES["detector"], 0.
                last_anchor_pts = pts
                accepted_calls += 1
            elif tracked is not None:
                keypoints, available, _, flow_error = tracked
                confidence = previous["confidence"] * .995
                source = SOURCE_CODES["optical_flow"]
                tracked_frames += 1
            else:
                keypoints = np.full((3, 2, 2), np.nan, np.float32)
                available = np.zeros((3, 2), bool)
                confidence, source, flow_error = 0., SOURCE_CODES["unavailable"], 0.
            age = 0. if last_anchor_pts is None else float((pts - last_anchor_pts) * time_base)
            for name, value in (
                ("pts", pts), ("shot_id", shot_index), ("keypoints", keypoints / np.asarray(frame_size, np.float32)),
                ("available", available), ("confidence", confidence), ("source", source),
                ("age_seconds", age), ("flow_error", flow_error), ("detector_anchor", source == SOURCE_CODES["detector"]),
            ):
                rows[name].append(value)
            previous = {
                "gray": gray, "scale": scale, "keypoints": keypoints,
                "available": available, "confidence": confidence,
            } if available.any() else None
        arrays = {
            "pts": np.asarray(rows["pts"], np.int64),
            "shot_id": np.asarray(rows["shot_id"], np.int32),
            "keypoints": np.asarray(rows["keypoints"], np.float32),
            "available": np.asarray(rows["available"], bool),
            "confidence": np.asarray(rows["confidence"], np.float32),
            "source": np.asarray(rows["source"], np.int8),
            "age_seconds": np.asarray(rows["age_seconds"], np.float32),
            "flow_error": np.asarray(rows["flow_error"], np.float32),
            "detector_anchor": np.asarray(rows["detector_anchor"], bool),
        }
        arrays_path = output / "fretboard.npz"
        np.savez_compressed(arrays_path, **arrays)
        coverage = arrays["available"].all(axis=-1).sum(axis=-1) >= 2
        report = {
            "schemaVersion": 1, "kind": "six-point-fretboard-observations",
            "videoSha256": sha256(video_path), "shotsSha256": sha256(shots_path),
            "modelSha256": sha256(model_path), "timeBase": shots["timeBase"],
            "frameCount": len(arrays["pts"]), "shotCount": len(shots["shots"]),
            "sourceEncoding": SOURCE_CODES, "arrays": "fretboard.npz",
            "arraysSha256": sha256(arrays_path),
            "detectorCalls": detector_calls, "acceptedDetectorCalls": accepted_calls,
            "trackedFrames": tracked_frames, "geometryFrames": int(coverage.sum()),
            "geometryCoverage": float(coverage.mean()),
            "trackingConfig": asdict(tracking_config), "detectorConfig": asdict(detector_config),
        }
        path = output / "fretboard.json"
        _publish(path, report)
        complete = True
        return path, report
    finally:
        container.close()
        if not complete:
            _cleanup(output)


def render_overlay(video_path, observations_path, output_path, hands_path=None):
    video_path, observations_path, output_path = map(Path, (video_path, observations_path, output_path))
    with np.load(observations_path.with_name("fretboard.npz"), allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    by_pts = {int(value): index for index, value in enumerate(arrays["pts"])}
    hand_by_pts, hand_landmarks = {}, None
    if hands_path is not None:
        with np.load(Path(hands_path).with_name("hands.npz"), allow_pickle=False) as hands:
            hand_by_pts = {int(value): index for index, value in enumerate(hands["pts"])}
            hand_landmarks = hands["image_landmarks"].copy()
    source = av.open(str(video_path))
    target = av.open(str(output_path), "w")
    try:
        stream = video_stream(source)
        output = target.add_stream("libx264", rate=stream.average_rate or 24)
        output.width, output.height, output.pix_fmt = stream.width, stream.height, "yuv420p"
        for frame in source.decode(stream):
            image = frame.to_ndarray(format="bgr24")
            index = by_pts.get(int(frame.pts))
            if index is not None:
                points = arrays["keypoints"][index] * [image.shape[1], image.shape[0]]
                available = arrays["available"][index]
                try:
                    geometry = FretboardGeometry.from_keypoints(points, available, (image.shape[1], image.shape[0]))
                except FretboardGeometryError:
                    geometry = None
                if geometry is not None:
                    for line in geometry.fret_lines(24):
                        cv2.line(image, tuple(np.round(line[0]).astype(int)), tuple(np.round(line[1]).astype(int)), (0, 255, 0), 1)
                    for string in geometry.string_paths():
                        cv2.polylines(image, [np.round(string).astype(np.int32)], False, (255, 255, 0), 1)
                    source_name = next(name for name, code in SOURCE_CODES.items() if code == int(arrays["source"][index]))
                    cv2.putText(
                        image,
                        f"fretboard {source_name} confidence={arrays['confidence'][index]:.2f} age={arrays['age_seconds'][index]:.2f}s",
                        (20, 36), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2, cv2.LINE_AA,
                    )
                else:
                    cv2.putText(image, "fretboard unavailable", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2, cv2.LINE_AA)
            hand_index = hand_by_pts.get(int(frame.pts))
            if hand_index is not None:
                for slot, color in enumerate(((255, 80, 80), (80, 160, 255))):
                    landmarks = hand_landmarks[hand_index, slot, :, :2] * [image.shape[1], image.shape[0]]
                    valid = np.isfinite(landmarks).all(axis=1)
                    for left, right in HAND_CONNECTIONS:
                        if valid[left] and valid[right]:
                            cv2.line(image, tuple(np.round(landmarks[left]).astype(int)), tuple(np.round(landmarks[right]).astype(int)), color, 2)
                    for point in landmarks[valid]:
                        cv2.circle(image, tuple(np.round(point).astype(int)), 3, color, -1)
            rendered = av.VideoFrame.from_ndarray(image, format="bgr24")
            rendered.pts, rendered.time_base = frame.pts, frame.time_base
            for packet in output.encode(rendered):
                target.mux(packet)
        for packet in output.encode():
            target.mux(packet)
    finally:
        source.close()
        target.close()
