"""PTS-preserving shot inspection for edited performance video."""

from __future__ import annotations

from collections import deque
from bisect import bisect_left
from copy import deepcopy
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
from uuid import uuid4

import av
import cv2
import numpy as np

from core import EvidenceError, frames_in_shot_range, private_output, publish_json, sha256, video_stream
from geometry import _cleanup


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


def _feature_coherence(source, target):
    orb = cv2.ORB_create(nfeatures=600, edgeThreshold=15, fastThreshold=12)
    source_points, source_descriptors = orb.detectAndCompute(source, None)
    target_points, target_descriptors = orb.detectAndCompute(target, None)
    result = {
        "sourceFeatures": len(source_points), "targetFeatures": len(target_points),
        "matches": 0, "inliers": 0, "inlierFraction": 0., "featureSupport": 0.,
        "gridCells": 0, "coherent": False,
    }
    if source_descriptors is None or target_descriptors is None or len(target_points) < 2:
        return result, None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(source_descriptors, target_descriptors, k=2)
    matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .75 * pair[1].distance]
    # Distinct source features must not all vote for the same target feature.
    matches = list({match.trainIdx: match for match in sorted(matches, key=lambda match: -match.distance)}.values())
    result["matches"] = len(matches)
    if len(matches) < 4:
        return result, None
    left = np.float32([source_points[match.queryIdx].pt for match in matches])
    right = np.float32([target_points[match.trainIdx].pt for match in matches])
    matrix, mask = cv2.estimateAffinePartial2D(
        left, right, method=cv2.RANSAC, ransacReprojThreshold=3., maxIters=1000, confidence=.99,
    )
    if matrix is None or mask is None:
        return result, None
    accepted = mask.ravel().astype(bool)
    height, width = source.shape
    cells = {(min(3, int(x * 4 / width)), min(3, int(y * 4 / height))) for x, y in left[accepted]}
    result.update(
        inliers=int(accepted.sum()), inlierFraction=float(accepted.mean()),
        featureSupport=float(accepted.sum() / max(1, min(len(source_points), len(target_points)))),
        gridCells=len(cells),
    )
    result["coherent"] = (
        result["inliers"] >= 12 and result["inlierFraction"] >= .65
        and result["featureSupport"] >= .20 and result["gridCells"] >= 6
    )
    return result, matrix


def transition_evidence(start, middle, end, kind):
    """Compare scene correspondences and an aligned blend, not score/GP targets."""
    endpoints, _ = _feature_coherence(start, end)
    evidence = {"endpoints": endpoints}
    if endpoints["coherent"]:
        return {"disposition": "coherent-motion", "reason": "spatially-distributed-endpoint-correspondence", "evidence": evidence}
    if kind == "low-contrast-cut":
        if (
            min(endpoints["sourceFeatures"], endpoints["targetFeatures"]) >= 30
            and endpoints["featureSupport"] < .05
            and float(np.mean(cv2.absdiff(start, end))) / 255 >= .03
        ):
            return {"disposition": "accepted-hard-cut", "reason": "textured-views-with-lost-feature-correspondence", "evidence": evidence}
    else:
        left, left_matrix = _feature_coherence(start, middle)
        right, right_matrix = _feature_coherence(end, middle)
        evidence.update(startToMiddle=left, endToMiddle=right)
        aligned = left["coherent"] and right["coherent"]
        textured_discontinuity = (
            min(endpoints["sourceFeatures"], endpoints["targetFeatures"]) >= 30
            and endpoints["featureSupport"] < .05
        )
        if aligned or textured_discontinuity:
            if not aligned:
                # An almost exact cross-fade can erase midpoint descriptors.
                # Without geometric support require a much stronger pixel model.
                left_matrix = right_matrix = np.float32([[1, 0, 0], [0, 1, 0]])
            height, width = middle.shape
            source_mask = np.ones_like(middle)
            valid = (
                cv2.warpAffine(source_mask, left_matrix, (width, height)) > 0
            ) & (cv2.warpAffine(source_mask, right_matrix, (width, height)) > 0)
            overlap = float(valid.mean())
            evidence["alignedOverlap"] = overlap
            if overlap >= .5:
                a = cv2.warpAffine(start, left_matrix, (width, height))[valid].astype(np.float32)
                b = cv2.warpAffine(end, right_matrix, (width, height))[valid].astype(np.float32)
                observed = middle[valid].astype(np.float32)
                delta = b - a
                denominator = float(delta @ delta)
                alpha = float(np.clip(((observed - a) @ delta) / denominator, 0, 1)) if denominator else 0.
                residual = float(np.sqrt(np.mean((observed - (a + alpha * delta)) ** 2)))
                single = min(float(np.sqrt(np.mean((observed - view) ** 2))) for view in (a, b))
                improvement = 1 - residual / max(single, 1e-6)
                evidence.update(blendAlpha=alpha, blendResidual=residual / 255, blendImprovement=improvement,
                                blendAlignment="feature-affine" if aligned else "identity")
                if .2 <= alpha <= .8 and improvement >= (.45 if aligned else .85) and single >= 8:
                    reason = "two-distinct-views-supported-by-aligned-middle-blend" if aligned else "distinct-textured-views-with-near-exact-pixel-blend"
                    return {"disposition": "accepted-dissolve", "reason": reason, "evidence": evidence}
    return {"disposition": "uncertain", "reason": "insufficient-independent-transition-evidence", "evidence": evidence}


def _cache_candidate_frame(frame, cache):
    pts = int(frame.pts)
    if pts in cache:
        return
    image = frame.to_ndarray(format="rgb24")
    gray = cv2.cvtColor(cv2.resize(image, (320, max(1, round(image.shape[0] * 320 / image.shape[1]))),
                                  interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    preview = cv2.resize(image, (960, max(1, round(image.shape[0] * 960 / image.shape[1]))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise EvidenceError("Unable to encode candidate review frame.")
    cache[pts] = {"gray": gray, "preview": encoded.tobytes()}


def _candidate_times(report):
    time_base = Fraction(*report["timeBase"])
    times = []
    for field in ("lowContrastCutCandidates", "dissolveCandidates"):
        for candidate in report.get(field, []):
            if type(candidate.get("pts")) is not int:
                raise EvidenceError("Automatic cut candidates must have integer source PTS.")
            times.append(float(candidate["pts"] * time_base))
            if field == "dissolveCandidates":
                for pts_key, seconds_key in (("startPts", "startSeconds"), ("endPts", "endSeconds")):
                    value = float(candidate[pts_key] * time_base) if type(candidate.get(pts_key)) is int else candidate.get(seconds_key)
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise EvidenceError("Dissolve bounds must have finite source times.")
                    times.append(value)
    return deque(sorted(set(times)))


def _read_candidate_frames(video_path, report, wanted, cache):
    wanted = sorted(set(wanted) - cache.keys())
    if not wanted:
        return 0
    indices = {pts: index for index, pts in enumerate(report["framePts"])}
    decoded = 0
    with av.open(str(video_path)) as container:
        stream = _video_stream(container, report)
        previous = None
        for target in wanted:
            if previous is None or indices[target] - indices[previous] > 32:
                container.seek(target, stream=stream, backward=True, any_frame=False)
                frames = container.decode(stream)
            found = False
            for frame in frames:
                decoded += 1
                if frame.pts is None:
                    raise EvidenceError("Decoded candidate frame has no presentation timestamp.")
                if frame.pts < target:
                    continue
                if frame.pts == target:
                    _cache_candidate_frame(frame, cache)
                    found = True
                break
            if not found:
                raise EvidenceError(f"Candidate source frame {target} was not decoded exactly.")
            previous = target
    return decoded


def _merge_intervals(intervals):
    merged = []
    for row in sorted(intervals, key=lambda row: (row["startPts"], row["endPtsExclusive"])):
        if merged and row["startPts"] <= merged[-1]["endPtsExclusive"]:
            merged[-1]["endPtsExclusive"] = max(merged[-1]["endPtsExclusive"], row["endPtsExclusive"])
            merged[-1]["reason"] = ";".join(sorted(set(merged[-1]["reason"].split(";")) | set(row["reason"].split(";"))))
        else:
            merged.append(dict(row))
    return merged


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


def validate_frame_timeline(report):
    """Return (native PTS, exclusive end), or None for a legacy uncached report."""
    if "framePts" not in report and "frameEndPtsExclusive" not in report:
        return None
    pts, end = report.get("framePts"), report.get("frameEndPtsExclusive")
    time_base = report.get("timeBase")
    if (
        not isinstance(time_base, list) or len(time_base) != 2
        or any(type(value) is not int or value <= 0 for value in time_base)
        or not isinstance(pts, list) or not pts
        or any(type(value) is not int for value in pts)
        or any(left >= right for left, right in zip(pts, pts[1:]))
        or type(report.get("frameCount")) is not int or len(pts) != report["frameCount"]
        or type(report.get("firstPts")) is not int or pts[0] != report["firstPts"]
        or type(report.get("lastPts")) is not int or pts[-1] != report["lastPts"]
        or type(end) is not int or end <= pts[-1]
    ):
        raise EvidenceError("Cached inspection timeline is malformed.")
    shots = report.get("shots")
    native = set(pts)
    if (
        not isinstance(shots, list) or not shots
        or type(report.get("shotCount")) is not int or len(shots) != report["shotCount"]
        or any(
            not isinstance(shot, dict)
            or type(shot.get("shotId")) is not int or shot["shotId"] != index
            or type(shot.get("startPts")) is not int or shot["startPts"] not in native
            or type(shot.get("endPtsExclusive")) is not int
            or shot["startPts"] >= shot["endPtsExclusive"]
            for index, shot in enumerate(shots)
        )
        or shots[0]["startPts"] != pts[0] or shots[-1]["endPtsExclusive"] != end
        or any(left["endPtsExclusive"] != right["startPts"] for left, right in zip(shots, shots[1:]))
    ):
        raise EvidenceError("Cached inspection boundaries do not match its native timeline.")
    if "frameTimelineSha256" in report and report["frameTimelineSha256"] != frame_timeline_sha256(report):
        raise EvidenceError("Cached inspection timeline hash differs from its contents.")
    return pts, end


def frame_timeline_sha256(report):
    """Hash source identity, rational clock, native PTS and the declared end."""
    value = {key: report[key] for key in ("videoSha256", "timeBase", "framePts", "frameEndPtsExclusive")}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def _frame_end(frame, stream):
    if frame.duration is not None and frame.duration > 0:
        duration = Fraction(frame.duration) * Fraction(frame.time_base or stream.time_base) / Fraction(stream.time_base)
        if duration.denominator != 1:
            raise EvidenceError("Final frame duration is not representable in native stream PTS.")
        return int(frame.pts) + int(duration), "decoded-final-frame-duration"
    if stream.duration is not None and stream.start_time is not None:
        end = int(stream.start_time) + int(stream.duration)
        if end > frame.pts:
            return end, "video-stream-duration"
    raise EvidenceError("Source has no trustworthy final frame duration; use an explicit source-bounded interval.")


def _video_stream(container, report):
    stream = video_stream(container)
    if stream.time_base is None or _fraction(stream.time_base) != report["timeBase"] or stream.index != report["videoStreamIndex"]:
        raise EvidenceError("Inspection stream and native source time base differ.")
    return stream


def _seek_thumbnails(video_path, report, output, shots):
    """Decode only the keyframe-to-boundary preroll needed for missing previews."""
    if not shots:
        return 0
    decoded = 0
    indices = {value: index for index, value in enumerate(report.get("framePts", []))}
    with av.open(str(video_path)) as container:
        stream = _video_stream(container, report)
        previous_target = None
        for shot in sorted(shots, key=lambda row: row["startPts"]):
            target = shot["startPts"]
            # Nearby previews share decoder preroll instead of repeatedly decoding
            # the same GOP. Distant previews still seek rather than scanning gaps.
            if previous_target is None or target not in indices or indices[target] - indices[previous_target] > 32:
                container.seek(target, stream=stream, backward=True, any_frame=False)
                frames = container.decode(stream)
            found = False
            for frame in frames:
                decoded += 1
                if frame.pts is None:
                    raise EvidenceError("Decoded frame has no presentation timestamp.")
                if frame.pts < target:
                    continue
                if frame.pts == target:
                    shot["reviewFrame"] = _write_thumbnail(
                        output, shot["shotId"], target, frame.to_ndarray(format="rgb24"),
                    )
                    found = True
                break
            if not found:
                raise EvidenceError(f"Source boundary PTS {target} was not decoded exactly.")
            previous_target = target
    return decoded


def inspect_shots(video_path, output_directory, config=ShotConfig(), review_path=None):
    video_path = Path(video_path)
    if not video_path.is_file():
        raise EvidenceError(f"Video does not exist: {video_path}")
    video_sha256 = sha256(video_path)
    reviewed_additions = _load_review(review_path, video_sha256)
    output = _safe_directory(output_directory)
    container = av.open(str(video_path))
    try:
        stream = video_stream(container)
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
        thumbnails = {}
        for frame in container.decode(stream):
            if frame.pts is None:
                raise EvidenceError("Decoded frame has no presentation timestamp.")
            pts = int(frame.pts)
            if last_pts is not None and pts <= last_pts:
                raise EvidenceError("Source PTS must be strictly increasing.")
            if frame.time_base is not None and Fraction(frame.time_base) != time_base:
                raise EvidenceError("Decoded frame time base differs from its video stream.")
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
                        "startPts": start[0]["pts"],
                        "endPts": end[0]["pts"],
                    })
                dissolve_window.popleft()
            if previous is None or cut_candidates and cut_candidates[-1]["pts"] == pts and cut_candidates[-1]["accepted"]:
                thumbnails[pts] = _write_thumbnail(output, frames, pts, image)
            previous = analysis
            frames += 1
            first_pts = pts if first_pts is None else first_pts
            last_pts = pts
        if frames == 0:
            raise EvidenceError("Video contains no decoded frames.")
        frame_end, end_source = _frame_end(frame, stream)
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
        cuts = []
        for shot_id, boundary in enumerate(deduplicated):
            cuts.append({
                "shotId": shot_id,
                "startPts": boundary["pts"],
                "startSeconds": boundary["seconds"],
                "boundaryType": boundary["type"],
                "boundaryProvenance": boundary["provenance"],
                "cutScore": boundary["score"],
                **({"reviewFrame": thumbnails[boundary["pts"]]} if boundary["pts"] in thumbnails else {}),
                **({"reviewNote": boundary["reviewNote"]} if boundary.get("reviewNote") else {}),
            })
        for index, shot in enumerate(cuts):
            stop = cuts[index + 1]["startPts"] if index + 1 < len(cuts) else frame_end
            shot["endPtsExclusive"] = stop
            shot["endSecondsExclusive"] = float(stop * time_base)
            shot["reviewState"] = "unknown"
            shot["guitarVisibility"] = "unknown"
        container.close()
        _seek_thumbnails(video_path, {"timeBase": _fraction(time_base), "videoStreamIndex": stream_metadata["index"]},
                         output, [shot for shot in cuts if "reviewFrame" not in shot])
        retained = {shot["reviewFrame"] for shot in cuts}
        for thumbnail in thumbnails.values():
            if thumbnail not in retained:
                (output / thumbnail).unlink()
        for shot in cuts:
            shot["reviewFrameSha256"] = sha256(output / shot["reviewFrame"])
        if sha256(video_path) != video_sha256:
            raise EvidenceError("Source video changed during inspection.")
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
        "framePts": [frame["pts"] for frame in frames_by_time],
        "frameEndPtsExclusive": frame_end,
        "frameTimelineProvenance": {"method": "decoded-native-pts", "endSource": end_source},
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
    report["frameTimelineSha256"] = frame_timeline_sha256(report)
    report_path = output / "shots.json"
    pending = report_path.with_name(f".{report_path.name}.{uuid4().hex}.part")
    try:
        pending.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        pending.replace(report_path)
    finally:
        pending.unlink(missing_ok=True)
    return report_path, report


def _decode_inspection_timeline(video_path, report, candidate_frames=None):
    first, last = report["firstPts"], report["lastPts"]
    pts = []
    decoded = 0
    requested_times = _candidate_times(report) if candidate_frames is not None else deque()
    capture_next = False
    previous_frame = None
    with av.open(str(video_path)) as container:
        stream = _video_stream(container, report)
        container.seek(first, stream=stream, backward=True, any_frame=False)
        previous = None
        for frame in container.decode(stream):
            decoded += 1
            if frame.pts is None or previous is not None and frame.pts <= previous:
                raise EvidenceError("Source PTS must be present and strictly increasing.")
            if frame.time_base is not None and Fraction(frame.time_base) != Fraction(stream.time_base):
                raise EvidenceError("Decoded frame time base differs from its video stream.")
            previous = frame.pts
            if frame.pts < first:
                continue
            if frame.pts > last:
                break
            pts.append(int(frame.pts))
            if capture_next:
                _cache_candidate_frame(frame, candidate_frames)
                capture_next = False
            while requested_times and requested_times[0] <= float(frame.pts * stream.time_base):
                requested_times.popleft()
                _cache_candidate_frame(frame, candidate_frames)
                if previous_frame is not None:
                    _cache_candidate_frame(previous_frame, candidate_frames)
                capture_next = True
            previous_frame = frame
            if frame.pts == last:
                break
        if not pts or pts[0] != first or pts[-1] != last or len(pts) != report["frameCount"]:
            raise EvidenceError("Legacy inspection frame bounds/count differ from the decoded source.")
        end = report["shots"][-1]["endPtsExclusive"]
        if end == last + 1 and not report.get("coveragePolicy"):
            end, end_source = _frame_end(frame, stream)
        else:
            if type(end) is not int or end <= last:
                raise EvidenceError("Inspection has an invalid explicit source interval end.")
            end_source = "explicit-inspection-interval"
            if frame.duration and end > _frame_end(frame, stream)[0]:
                raise EvidenceError("Inspection end extends beyond its last source frame.")
    report["shots"][-1]["endPtsExclusive"] = end
    report["shots"][-1]["endSecondsExclusive"] = float(end * Fraction(*report["timeBase"]))
    report.update(framePts=pts, frameEndPtsExclusive=end)
    report["frameTimelineProvenance"] = {
        "method": "decoded-legacy-inspection-range",
        "endSource": end_source,
        "decodedFrameCount": decoded,
    }
    report["frameTimelineSha256"] = frame_timeline_sha256(report)
    validate_frame_timeline(report)
    return decoded


def _copy_review_frame(inspection_path, output, source, shot):
    relative = Path(source["reviewFrame"])
    root = inspection_path.parent.resolve()
    path = root / relative
    if relative.is_absolute() or path.resolve() != path.absolute() or not path.resolve().is_relative_to(root) or not path.is_file():
        raise EvidenceError("Inspection review thumbnail must be an existing, unaliased local image.")
    digest = sha256(path)
    if source.get("reviewFrameSha256", digest) != digest:
        raise EvidenceError("Inspection review thumbnail hash differs from its contents.")
    target = output / "review" / f"shot-{shot['shotId']:04d}-pts-{shot['startPts']}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    if sha256(target) != digest:
        raise EvidenceError("Review thumbnail changed during copying.")
    shot["reviewFrame"] = str(target.relative_to(output))


def prepare_automatic_shots(video_path, inspection_path, output_directory):
    """Split supported unreviewed transitions, keeping weak evidence diagnostic.

    False positives shorten temporal context. Missed cuts remain possible: this
    policy reuses inspection candidates, not a calibrated transition detector.
    Reviewed/already-automatic inputs are returned unchanged.
    """
    video_path, inspection_path = Path(video_path), Path(inspection_path)
    source_hash = sha256(inspection_path)
    original = json.loads(inspection_path.read_text(encoding="utf-8"))
    if (
        original.get("schemaVersion") != 1 or original.get("kind") != "video-shot-inspection"
        or original.get("videoSha256") != sha256(video_path)
        or not isinstance(original.get("shots"), list) or not original["shots"]
    ):
        raise EvidenceError("Automatic shots require an inspection bound to this source video.")
    cached = validate_frame_timeline(original)
    if (
        original.get("reviewComplete") is True or original.get("reviewPath")
        or any(shot.get("boundaryProvenance") == "reviewed"
               or shot.get("reviewState") in ("reviewed", "accepted", "confirmed") for shot in original["shots"])
        or original.get("automaticReview", {}).get("method") == "conservative-cut-boundaries-v1"
    ):
        return inspection_path, original
    report = deepcopy(original)
    candidate_frames = {}
    timeline_decoded = 0 if cached is not None else _decode_inspection_timeline(video_path, report, candidate_frames)
    pts, end = validate_frame_timeline(report)
    native = set(pts)
    time_base = Fraction(*report["timeBase"])
    boundaries = {shot["startPts"]: deepcopy(shot) for shot in report["shots"]}
    original_boundaries = set(boundaries)
    intervals = []
    diagnostic_intervals = []
    plans = []

    def add_boundary(pts_value, boundary_type, reason, score=None):
        if pts_value == end or pts_value in boundaries:
            return
        boundaries[pts_value] = {
            "startPts": pts_value, "startSeconds": float(pts_value * time_base),
            "boundaryType": boundary_type, "boundaryProvenance": "automatic-conservative",
            "cutScore": score, "reviewState": "unknown", "guitarVisibility": "unknown",
            "automaticBoundaryReason": reason,
        }

    def candidate_index(candidate):
        value = candidate.get("pts")
        if type(value) is not int or value not in native:
            raise EvidenceError("Automatic cut candidates must have native in-range frame PTS.")
        return bisect_left(pts, value)

    for candidate in report.get("lowContrastCutCandidates", []):
        index = candidate_index(candidate)
        start = pts[max(0, index - 1)]
        stop = pts[index + 1] if index + 1 < len(pts) else end
        plans.append({
            "candidate": candidate, "kind": "low-contrast-cut", "startPts": start,
            "endPtsExclusive": stop, "samplePts": [start, candidate["pts"], candidate["pts"]],
        })
    for candidate in report.get("dissolveCandidates", []):
        candidate_index(candidate)
        bounds = []
        for field, seconds_field in (("startPts", "startSeconds"), ("endPts", "endSeconds")):
            if field in candidate:
                value = candidate[field]
                if type(value) is not int:
                    raise EvidenceError("Dissolve bounds must use integer source PTS.")
                value = max(pts[0], min(pts[-1], value))
                if value not in native:
                    raise EvidenceError("Dissolve bounds must match native source PTS.")
            else:
                seconds = candidate.get(seconds_field)
                if type(seconds) not in (int, float) or not math.isfinite(seconds):
                    raise EvidenceError("Legacy dissolve bounds must have finite source seconds.")
                value = min(pts, key=lambda value: abs(float(value * time_base) - seconds))
            bounds.append(value)
        start, last = bounds
        if not start <= candidate["pts"] <= last:
            raise EvidenceError("Dissolve bounds do not contain the candidate frame.")
        index = bisect_left(pts, last) + 1
        stop = pts[index] if index < len(pts) else end
        plans.append({
            "candidate": candidate, "kind": "dissolve", "startPts": start,
            "endPtsExclusive": stop, "samplePts": [start, candidate["pts"], last],
        })
    wanted = [value for plan in plans for value in (*plan["samplePts"], plan["endPtsExclusive"]) if value != end]
    evidence_decoded = _read_candidate_frames(video_path, report, wanted, candidate_frames)
    dispositions = []
    for plan in plans:
        candidate = plan["candidate"]
        evidence = transition_evidence(*(candidate_frames[value]["gray"] for value in plan["samplePts"]), plan["kind"])
        dispositions.append({
            "pts": candidate["pts"], "kind": plan["kind"], "startPts": plan["startPts"],
            "endPtsExclusive": plan["endPtsExclusive"], **evidence,
        })
        interval = {"startPts": plan["startPts"], "endPtsExclusive": plan["endPtsExclusive"], "reason": plan["kind"]}
        if evidence["disposition"] == "accepted-hard-cut":
            add_boundary(candidate["pts"], "hard_cut", evidence["reason"], candidate.get("score"))
        elif evidence["disposition"] == "accepted-dissolve":
            intervals.append(interval)
        elif evidence["disposition"] == "uncertain":
            diagnostic_intervals.append(interval)
    intervals = _merge_intervals(intervals)
    # Only isolate supported blends. No midpoint resets inside already-masked spans.
    for interval in intervals:
        add_boundary(interval["startPts"], "transition", interval["reason"])
        add_boundary(interval["endPtsExclusive"], "transition", interval["reason"])
    shots = [boundaries[value] for value in sorted(boundaries)]
    source_index = 0
    for index, shot in enumerate(shots):
        while source_index + 1 < len(original["shots"]) and original["shots"][source_index + 1]["startPts"] <= shot["startPts"]:
            source_index += 1
        shot.update(
            shotId=index, sourceShotId=original["shots"][source_index]["shotId"],
            endPtsExclusive=shots[index + 1]["startPts"] if index + 1 < len(shots) else end,
        )
        shot["endSecondsExclusive"] = float(shot["endPtsExclusive"] * time_base)
    report.update(shots=shots, shotCount=len(shots), sourceInspectionSha256=source_hash,
                  sourceInspectionPath=str(inspection_path.resolve()), trainingPerformed=False)
    report["automaticReview"] = {
        "method": "conservative-cut-boundaries-v1",
        "policyVersion": "motion-coherence-v2",
        "reviewRequired": False,
        "manualReviewPerformed": False,
        "uncertainIntervals": intervals,
        "diagnosticIntervals": _merge_intervals(diagnostic_intervals),
        "candidateDispositions": dispositions,
        "sourceShotCount": original["shotCount"],
        "addedBoundaryCount": len(boundaries) - len(original_boundaries),
        "lowContrastCandidateCount": len(report.get("lowContrastCutCandidates", [])),
        "dissolveCandidateCount": len(report.get("dissolveCandidates", [])),
        "acceptedHardCutCount": sum(row["disposition"] == "accepted-hard-cut" for row in dispositions),
        "acceptedDissolveCount": sum(row["disposition"] == "accepted-dissolve" for row in dispositions),
        "coherentMotionCount": sum(row["disposition"] == "coherent-motion" for row in dispositions),
        "unresolvedCandidateCount": sum(row["disposition"] == "uncertain" for row in dispositions),
        "maskedDurationSeconds": float(sum(row["endPtsExclusive"] - row["startPts"] for row in intervals) * time_base),
        "timelineSource": "cached-native-pts" if cached is not None else "decoded-legacy-inspection-range",
        "timelineDecodedFrameCount": timeline_decoded,
        "candidateEvidenceDecodedFrameCount": evidence_decoded,
        "policy": "Preserve strong cuts. Reject spatially coherent motion; require feature discontinuity for low-contrast cuts or motion-compensated/near-exact dual-view blend evidence for dissolves. Merge supported blend masks and split only their edges. Unconfirmed candidates remain diagnostic, not camera shots or masks.",
        "limitation": "Unconfirmed transitions remain possible and unmasked. Feature-poor, moving, occluded or similar-view transitions can be missed. Correspondence thresholds are engineering criteria, not calibrated cut accuracy or manual review.",
    }
    report["frameTimelineSha256"] = frame_timeline_sha256(report)
    validate_frame_timeline(report)
    output = _safe_directory(output_directory)
    complete = False
    try:
        originals_by_pts = {shot["startPts"]: shot for shot in original["shots"]}
        missing = []
        for shot in shots:
            source = originals_by_pts.get(shot["startPts"])
            if source is not None and source.get("reviewFrame"):
                _copy_review_frame(inspection_path, output, source, shot)
            elif shot["startPts"] in candidate_frames:
                target = output / "review" / f"shot-{shot['shotId']:04d}-pts-{shot['startPts']}.jpg"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(candidate_frames[shot["startPts"]]["preview"])
                shot["reviewFrame"] = str(target.relative_to(output))
            else:
                missing.append(shot)
        report["automaticReview"]["thumbnailDecodedFrameCount"] = _seek_thumbnails(video_path, report, output, missing)
        for shot in shots:
            shot["reviewFrameSha256"] = sha256(output / shot["reviewFrame"])
        if sha256(inspection_path) != source_hash or sha256(video_path) != report["videoSha256"]:
            raise EvidenceError("Source video or inspection changed during automatic preparation.")
        path = publish_json(output / "shots.json", report)
        complete = True
        return path, report
    finally:
        if not complete:
            _cleanup(output)


def select_shots(video_path, shots_path, first_shot, last_shot, output_directory):
    shots_path, video_path = Path(shots_path), Path(video_path)
    original = json.loads(shots_path.read_text(encoding="utf-8"))
    if original.get("kind") != "video-shot-inspection" or original.get("videoSha256") != sha256(video_path):
        raise EvidenceError("Shot selection requires an inspection bound to this source video.")
    if type(first_shot) is not int or type(last_shot) is not int or not 1 <= first_shot <= last_shot <= len(original["shots"]):
        raise EvidenceError("Select an ordered, one-based shot range from this inspection.")
    cached = validate_frame_timeline(original)
    report = deepcopy(original)
    report["shots"] = [
        {**shot, "shotId": i, "sourceShotId": shot["shotId"]}
        for i, shot in enumerate(original["shots"][first_shot - 1:last_shot])
    ]
    report["sourceInspectionSha256"] = sha256(shots_path)
    report["sourceInspectionPath"] = str(shots_path.resolve())
    report["sourceFrameCount"] = original.get("sourceFrameCount", original["frameCount"])
    report["sourceShotCount"] = original.get("sourceShotCount", original["shotCount"])
    report["shotCount"] = len(report["shots"])
    report["coveragePolicy"] = "Explicit contiguous shot subset; original video PTS, bytes and time base retained."
    start, end = report["shots"][0]["startPts"], report["shots"][-1]["endPtsExclusive"]
    for field in ("lowContrastCutCandidates", "dissolveCandidates"):
        report[field] = [row for row in original.get(field, []) if start <= row["pts"] < end]
    if "automaticReview" in report:
        automatic = report["automaticReview"]
        for field in ("uncertainIntervals", "diagnosticIntervals"):
            if field in automatic:
                automatic[field] = [
                    {**row, "startPts": max(start, row["startPts"]), "endPtsExclusive": min(end, row["endPtsExclusive"])}
                    for row in automatic[field] if row["startPts"] < end and row["endPtsExclusive"] > start
                ]
        if "maskedDurationSeconds" in automatic:
            automatic["maskedDurationSeconds"] = float(sum(
                row["endPtsExclusive"] - row["startPts"] for row in automatic["uncertainIntervals"]
            ) * Fraction(*report["timeBase"]))
    output = _safe_directory(output_directory)
    complete = False
    try:
        boundaries = {shot["startPts"]: shot for shot in report["shots"]}
        if cached is not None:
            pts = [value for value in cached[0] if start <= value < end]
            missing = []
            for shot in report["shots"]:
                source = original["shots"][shot["sourceShotId"]]
                if source.get("reviewFrame"):
                    _copy_review_frame(shots_path, output, source, shot)
                else:
                    missing.append(shot)
            _seek_thumbnails(video_path, report, output, missing)
        else:
            pts = []
            with av.open(str(video_path)) as container:
                stream = _video_stream(container, report)
                for frame in frames_in_shot_range(container, stream, report):
                    if pts and frame.pts <= pts[-1]:
                        raise EvidenceError("Source PTS must be strictly increasing.")
                    pts.append(int(frame.pts))
                    if frame.pts in boundaries:
                        shot = boundaries[frame.pts]
                        shot["reviewFrame"] = _write_thumbnail(output, shot["shotId"], frame.pts, frame.to_ndarray(format="rgb24"))
                    final_frame = frame
                if pts and end == pts[-1] + 1 and not original.get("coveragePolicy"):
                    end, _ = _frame_end(final_frame, stream)
                    report["shots"][-1]["endPtsExclusive"] = end
                    report["shots"][-1]["endSecondsExclusive"] = float(end * Fraction(*report["timeBase"]))
        if not pts or pts[0] != start or any(shot["startPts"] not in pts for shot in report["shots"]):
            raise EvidenceError("Selected source shot boundaries were not decoded exactly.")
        report.update(firstPts=pts[0], lastPts=pts[-1], frameCount=len(pts), framePts=pts, frameEndPtsExclusive=end)
        report["frameTimelineProvenance"] = {
            "method": "selected-native-pts" if cached is not None else "decoded-selected-range",
            "endSource": "source-shot-interval",
        }
        report["frameTimelineSha256"] = frame_timeline_sha256(report)
        for shot in report["shots"]:
            shot["reviewFrameSha256"] = sha256(output / shot["reviewFrame"])
        validate_frame_timeline(report)
        if sha256(shots_path) != report["sourceInspectionSha256"] or sha256(video_path) != report["videoSha256"]:
            raise EvidenceError("Source video or inspection changed during selection.")
        path = publish_json(output / "shots.json", report)
        complete = True
        return path, report
    finally:
        if not complete:
            _cleanup(output)
