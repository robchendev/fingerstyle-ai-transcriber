"""Optional local YOLO adapter for fretboard keypoints."""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .fretboard_geometry import FretboardGeometry, FretboardGeometryError, fret_scale_position


@dataclass(frozen=True)
class FretboardDetectorConfig:
    image_sizes: tuple[int, ...] = (1920, 3840, 640)
    proposal_score: float = .05
    acceptance_score: float = .25
    keypoint_score: float = .25
    iou: float = .5
    device: str = "cpu"

    def __post_init__(self):
        if not self.image_sizes or any(type(value) is not int or value < 320 for value in self.image_sizes):
            raise ValueError("image_sizes must contain integers >= 320.")
        if len(set(self.image_sizes)) != len(self.image_sizes):
            raise ValueError("image_sizes must be unique.")
        for name in ("proposal_score", "acceptance_score", "keypoint_score", "iou"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one.")
        if self.proposal_score > self.acceptance_score:
            raise ValueError("proposal_score cannot exceed acceptance_score.")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a nonempty string.")


@dataclass(frozen=True)
class FretboardDetection:
    keypoints: np.ndarray
    available: np.ndarray
    score: float
    geometry: FretboardGeometry
    image_size: int


def select_detection(keypoints, keypoint_scores, scores, frame_size, config, image_size):
    points = np.asarray(keypoints, dtype=np.float64)
    point_scores = np.asarray(keypoint_scores, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (7, 2):
        raise ValueError("keypoints must have shape (N, 7, 2).")
    if point_scores.shape != points.shape[:2] or scores.shape != (len(points),):
        raise ValueError("Detector score shapes disagree.")
    best = None
    for candidate, confidences, score in zip(points, point_scores, scores, strict=True):
        available = confidences >= config.keypoint_score
        anchors = candidate[:6].reshape(3, 2, 2)
        anchor_available = available[:6].reshape(3, 2)
        try:
            geometry = FretboardGeometry.from_keypoints(anchors, anchor_available, frame_size)
        except FretboardGeometryError:
            continue
        if available[6]:
            fret5 = geometry.coordinate(candidate[6])
            expected_scale = fret_scale_position(5)
            if abs(fret5.scale_position - expected_scale) > .12 or abs(fret5.string_position - 2.5) > 1.5:
                continue
        value = FretboardDetection(anchors.astype(np.float32), anchor_available, float(score), geometry, image_size)
        key = int(anchor_available.all(axis=1).sum()), int(available[6]), float(score), -geometry.reprojection_error
        if best is None or key > best[0]:
            best = key, value
    return None if best is None else best[1]


class YoloFretboardDetector:
    def __init__(self, model_path, config=FretboardDetectorConfig()):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Fretboard detector does not exist: {path}")
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Install requirements-fretboard.txt before detector inference.") from error
        self.path = path.resolve()
        self.config = config
        self.model = YOLO(str(self.path))
        if list(getattr(self.model.model, "kpt_shape", ())) != [7, 3]:
            raise ValueError("Fretboard detector must emit seven keypoints with visibility.")

    def detect(self, frame):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be an HxWx3 uint8 image.")
        best = None
        for maximum in self.config.image_sizes:
            size = min(maximum, max(frame.shape[:2]))
            result = self.model.predict(
                frame, verbose=False, conf=self.config.proposal_score,
                iou=self.config.iou, imgsz=size, device=self.config.device,
            )[0]
            if result.boxes is None or result.keypoints is None:
                continue
            scores = result.boxes.conf.detach().cpu().numpy()
            data = result.keypoints.data.detach().cpu().numpy()
            count = min(len(scores), len(data))
            if not count:
                continue
            selected = select_detection(
                data[:count, :, :2],
                data[:count, :, 2],
                scores[:count], (frame.shape[1], frame.shape[0]), self.config, size,
            )
            if selected is not None and (best is None or selected.score > best.score):
                best = selected
            if selected is not None and selected.score >= self.config.acceptance_score:
                return selected
        return best
