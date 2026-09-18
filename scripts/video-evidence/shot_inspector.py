"""PTS-preserving shot inspection for edited performance video."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
from uuid import uuid4

import av
import cv2
import numpy as np

from core import EvidenceError, PRIVATE_OUTPUT_ROOT, private_output, sha256


@dataclass(frozen=True)
class ShotConfig:
    analysis_width: int = 320
    cut_threshold: float = .28
    candidate_cut_threshold: float = .21
    min_shot_seconds: float = .20
    histogram_weight: float = .65
    dissolve_window_seconds: float = 1.0
    dissolve_endpoint_threshold: float = .12
    dissolve_residual_threshold: float = .60

    def __post_init__(self):
        if type(self.analysis_width) is not int or self.analysis_width < 64:
            raise EvidenceError("Shot analysis width must be an integer >= 64.")
        for name in (
            "cut_threshold",
            "candidate_cut_threshold",
            "min_shot_seconds",
            "histogram_weight",
            "dissolve_window_seconds",
            "dissolve_endpoint_threshold",
            "dissolve_residual_threshold",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise EvidenceError(f"{name} must be finite and nonnegative.")
        if any(getattr(self, name) > 1 for name in (
            "cut_threshold",
            "candidate_cut_threshold",
            "histogram_weight",
            "dissolve_endpoint_threshold",
            "dissolve_residual_threshold",
        )):
            raise EvidenceError("Shot thresholds and weights cannot exceed one.")
        if self.candidate_cut_threshold > self.cut_threshold:
            raise EvidenceError("Candidate cut threshold cannot exceed the accepted cut threshold.")
        if self.dissolve_window_seconds <= 0:
            raise EvidenceError("Dissolve window must be positive.")


def _analysis_frame(image, width):
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, alpha=1, norm_type=cv2.NORM_L1)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    return histogram, gray


def cut_components(previous, current, histogram_weight=.65):
    prior_histogram, prior_gray = previous
    histogram, gray = current
    if prior_gray.shape != gray.shape:
        raise EvidenceError("Shot analysis frames must have the same shape.")
    histogram_distance = cv2.compareHist(prior_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA)
    pixel_distance = float(np.mean(cv2.absdiff(prior_gray, gray))) / 255
    score = float(histogram_weight * histogram_distance + (1 - histogram_weight) * pixel_distance)
    return score, float(histogram_distance), pixel_distance


def cut_score(previous, current, histogram_weight=.65):
    return cut_components(previous, current, histogram_weight)[0]


def dissolve_measurement(start, middle, end, histogram_weight=.65):
    start_gray = start[1].astype(np.float32).ravel()
    middle_gray = middle[1].astype(np.float32).ravel()
    end_gray = end[1].astype(np.float32).ravel()
    endpoint_delta = end_gray - start_gray
    denominator = float(endpoint_delta @ endpoint_delta)
    alpha = 0.0 if denominator == 0 else float(np.clip(
        ((middle_gray - start_gray) @ endpoint_delta) / denominator,
        0,
        1,
    ))
    prediction = start_gray + alpha * endpoint_delta
    residual = float(np.sqrt(np.mean(np.square(middle_gray - prediction))))
    endpoint_rms = float(np.sqrt(np.mean(np.square(endpoint_delta))))
    residual_ratio = residual / (endpoint_rms + 1e-6)
    return {
        "endpointScore": cut_score(start, end, histogram_weight),
        "blendAlpha": alpha,
        "residualRatio": residual_ratio,
    }


def _fraction(value):
    value = Fraction(value)
    return [value.numerator, value.denominator]


def _safe_directory(path):
    marker = private_output(Path(path) / ".directory-check")
    directory = marker.parent
    if directory.exists():
        raise EvidenceError(f"Refusing to overwrite existing shot inspection: {directory}")
    directory.mkdir(parents=True)
    return directory


def _write_thumbnail(directory, shot_id, pts, image):
    path = directory / "review" / f"shot-{shot_id:04d}-pts-{pts}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    height = max(1, round(image.shape[0] * 960 / image.shape[1]))
    preview = cv2.resize(image, (960, height), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise EvidenceError(f"Unable to write review frame: {path}")
    return str(path.relative_to(directory))


def _load_review(path, video_sha256):
    if path is None:
        return []
    review = json.loads(Path(path).read_text(encoding="utf-8"))
    if review.get("schemaVersion") != 1 or review.get("kind") != "video-shot-boundary-review":
        raise EvidenceError("Shot review has an unsupported schema.")
    if review.get("videoSha256") != video_sha256:
        raise EvidenceError("Shot review does not match the source video.")
    additions = review.get("addBoundaries")
    if not isinstance(additions, list):
        raise EvidenceError("Shot review addBoundaries must be a list.")
    for addition in additions:
        if not isinstance(addition, dict):
            raise EvidenceError("Shot review boundaries must be objects.")
        seconds = addition.get("seconds")
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise EvidenceError("Reviewed boundary seconds must be finite and positive.")
        if addition.get("type") not in ("hard_cut", "dissolve"):
            raise EvidenceError("Reviewed boundary type must be hard_cut or dissolve.")
    return additions


def _nearest_frame(frames, seconds):
    return min(frames, key=lambda frame: abs(frame["seconds"] - seconds))


def _select_local_maxima(candidates, score_key, radius_seconds):
    selected = []
    for candidate in candidates:
        score = candidate[score_key]
        if all(
            other is candidate
            or abs(other["seconds"] - candidate["seconds"]) > radius_seconds
            or score >= other[score_key]
            for other in candidates
        ):
            selected.append(candidate)
    return selected


def inspect_shots(video_path, output_directory, config=ShotConfig(), review_path=None):
    video_path = Path(video_path)
    if not video_path.is_file():
        raise EvidenceError(f"Video does not exist: {video_path}")
    video_sha256 = sha256(video_path)
    reviewed_additions = _load_review(review_path, video_sha256)
    output = _safe_directory(output_directory)
    container = av.open(str(video_path))
    try:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if len(streams) != 1:
            raise EvidenceError("Shot inspection requires exactly one video stream.")
        stream = streams[0]
        if stream.time_base is None:
            raise EvidenceError("Video stream has no presentation time base.")
        time_base = Fraction(stream.time_base)
        stream_metadata = {
            "index": stream.index,
            "codec": stream.codec_context.name,
            "width": stream.codec_context.width,
            "height": stream.codec_context.height,
            "averageRate": None if stream.average_rate is None else _fraction(stream.average_rate),
        }
        previous = None
        frames_by_time = []
        cut_candidates = []
        dissolve_candidates = []
        dissolve_window = deque()
        frames = 0
        first_pts = last_pts = None
        for frame in container.decode(stream):
            if frame.pts is None:
                raise EvidenceError("Decoded frame has no presentation timestamp.")
            pts = int(frame.pts)
            image = frame.to_ndarray(format="rgb24")
            analysis = _analysis_frame(image, config.analysis_width)
            seconds = float(pts * time_base)
            frame_time = {"pts": pts, "seconds": seconds}
            frames_by_time.append(frame_time)
            if previous is not None:
                score, histogram_distance, pixel_distance = cut_components(
                    previous,
                    analysis,
                    config.histogram_weight,
                )
                if score >= config.candidate_cut_threshold:
                    cut_candidates.append({
                        **frame_time,
                        "score": score,
                        "histogramDistance": histogram_distance,
                        "pixelDistance": pixel_distance,
                        "accepted": score >= config.cut_threshold,
                    })
            dissolve_window.append((frame_time, analysis))
            while (
                len(dissolve_window) >= 3
                and dissolve_window[-1][0]["seconds"] - dissolve_window[0][0]["seconds"]
                >= config.dissolve_window_seconds
            ):
                start = dissolve_window[0]
                end = dissolve_window[-1]
                midpoint = (start[0]["seconds"] + end[0]["seconds"]) / 2
                middle = min(dissolve_window, key=lambda item: abs(item[0]["seconds"] - midpoint))
                measurement = dissolve_measurement(
                    start[1],
                    middle[1],
                    end[1],
                    config.histogram_weight,
                )
                if (
                    measurement["endpointScore"] >= config.dissolve_endpoint_threshold
                    and .15 <= measurement["blendAlpha"] <= .85
                    and measurement["residualRatio"] <= config.dissolve_residual_threshold
                ):
                    dissolve_candidates.append({
                        "pts": middle[0]["pts"],
                        "seconds": middle[0]["seconds"],
                        "startSeconds": start[0]["seconds"],
                        "endSeconds": end[0]["seconds"],
                        **measurement,
                        "reviewRequired": True,
                    })
                dissolve_window.popleft()
            previous = analysis
            frames += 1
            first_pts = pts if first_pts is None else first_pts
            last_pts = pts
        if frames == 0:
            raise EvidenceError("Video contains no decoded frames.")
        hard_boundaries = [{
            "pts": frames_by_time[0]["pts"],
            "seconds": frames_by_time[0]["seconds"],
            "type": "start",
            "provenance": "stream",
            "score": None,
        }]
        hard_boundaries.extend({
            "pts": candidate["pts"],
            "seconds": candidate["seconds"],
            "type": "hard_cut",
            "provenance": "automatic",
            "score": candidate["score"],
        } for candidate in cut_candidates if candidate["accepted"])
        reviewed_boundaries = []
        for addition in reviewed_additions:
            frame_time = _nearest_frame(frames_by_time, float(addition["seconds"]))
            reviewed_boundaries.append({
                "pts": frame_time["pts"],
                "seconds": frame_time["seconds"],
                "type": addition["type"],
                "provenance": "reviewed",
                "score": None,
                "reviewNote": addition.get("note"),
            })
        boundaries = sorted(hard_boundaries + reviewed_boundaries, key=lambda item: item["pts"])
        deduplicated = []
        for boundary in boundaries:
            if deduplicated and boundary["seconds"] - deduplicated[-1]["seconds"] < config.min_shot_seconds:
                if boundary["provenance"] == "reviewed":
                    deduplicated[-1] = boundary
                continue
            deduplicated.append(boundary)
        boundary_by_pts = {boundary["pts"]: boundary for boundary in deduplicated}
        container.close()
        container = av.open(str(video_path))
        stream = [stream for stream in container.streams if stream.type == "video"][0]
        for frame in container.decode(stream):
            pts = int(frame.pts)
            boundary = boundary_by_pts.get(pts)
            if boundary is not None:
                boundary["reviewFrame"] = _write_thumbnail(
                    output,
                    deduplicated.index(boundary),
                    pts,
                    frame.to_ndarray(format="rgb24"),
                )
        cuts = []
        for shot_id, boundary in enumerate(deduplicated):
            cuts.append({
                "shotId": shot_id,
                "startPts": boundary["pts"],
                "startSeconds": boundary["seconds"],
                "boundaryType": boundary["type"],
                "boundaryProvenance": boundary["provenance"],
                "cutScore": boundary["score"],
                "reviewFrame": boundary["reviewFrame"],
                **({"reviewNote": boundary["reviewNote"]} if boundary.get("reviewNote") else {}),
            })
        for index, shot in enumerate(cuts):
            stop = cuts[index + 1]["startPts"] if index + 1 < len(cuts) else last_pts + 1
            shot["endPtsExclusive"] = stop
            shot["endSecondsExclusive"] = float(stop * time_base)
            shot["reviewState"] = "unknown"
            shot["guitarVisibility"] = "unknown"
    except Exception:
        for path in sorted(output.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        output.rmdir()
        raise
    finally:
        container.close()
    report = {
        "schemaVersion": 1,
        "kind": "video-shot-inspection",
        "visibility": "private",
        "videoSha256": video_sha256,
        "videoStreamIndex": stream_metadata["index"],
        "codec": stream_metadata["codec"],
        "width": stream_metadata["width"],
        "height": stream_metadata["height"],
        "averageRate": stream_metadata["averageRate"],
        "timeBase": _fraction(time_base),
        "firstPts": first_pts,
        "lastPts": last_pts,
        "frameCount": frames,
        "shotCount": len(cuts),
        "config": asdict(config),
        "reviewPath": None if review_path is None else str(Path(review_path)),
        "lowContrastCutCandidates": [
            candidate for candidate in cut_candidates if not candidate["accepted"]
        ],
        "dissolveCandidates": _select_local_maxima(
            [
                candidate for candidate in dissolve_candidates
                if all(
                    abs(candidate["seconds"] - boundary["seconds"])
                    > config.dissolve_window_seconds / 2
                    for boundary in hard_boundaries[1:]
                )
            ],
            "endpointScore",
            config.dissolve_window_seconds / 2,
        ),
        "shots": cuts,
        "trainingPerformed": False,
    }
    report_path = output / "shots.json"
    pending = report_path.with_name(f".{report_path.name}.{uuid4().hex}.part")
    try:
        pending.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        pending.replace(report_path)
    finally:
        pending.unlink(missing_ok=True)
    return report_path, report
