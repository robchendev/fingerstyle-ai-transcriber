"""Reviewed guitar geometry preparation, tracking, and review."""

from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path
from uuid import uuid4

import av
import cv2
import numpy as np

from core import EvidenceError, GuitarCoordinateFrame, private_output, sha256


GEOMETRY_POINTS = (
    "nut",
    "neckBody",
    "fretboardUpper",
    "fretboardLower",
    "soundhole",
    "bridge",
)
REQUIRED_FRAME_POINTS = GEOMETRY_POINTS[:4]
GEOMETRY_STATES = (
    "trackable",
    "guitar_partial",
    "guitar_absent",
    "transition",
    "calibration_unstable",
    "unknown",
)
STATE_CODES = {state: index for index, state in enumerate(GEOMETRY_STATES)}


def _safe_directory(path, kind):
    marker = private_output(Path(path) / ".directory-check")
    directory = marker.parent
    if directory.exists():
        raise EvidenceError(f"Refusing to overwrite existing {kind}: {directory}")
    directory.mkdir(parents=True)
    return directory


def _publish(path, value):
    pending = path.with_name(f".{path.name}.{uuid4().hex}.part")
    try:
        pending.write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def _cleanup(directory):
    for path in sorted(directory.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    directory.rmdir()


def _documents(video_path, shots_path, hand_review_path=None):
    video_path, shots_path = Path(video_path), Path(shots_path)
    if not video_path.is_file() or not shots_path.is_file():
        raise EvidenceError("Geometry preparation requires video and shot report files.")
    video_hash = sha256(video_path)
    shots_hash = sha256(shots_path)
    shots = json.loads(shots_path.read_text(encoding="utf-8"))
    if shots.get("kind") != "video-shot-inspection" or shots.get("videoSha256") != video_hash:
        raise EvidenceError("Shot report is not bound to the geometry source video.")
    review = None
    if hand_review_path is not None:
        hand_review_path = Path(hand_review_path)
        review = json.loads(hand_review_path.read_text(encoding="utf-8"))
        if (
            review.get("kind") != "fingerstyle-hand-review"
            or review.get("videoSha256") != video_hash
            or review.get("shotsSha256") != shots_hash
        ):
            raise EvidenceError("Hand review is not bound to the geometry video and shots.")
    return video_path, shots_path, shots, review, video_hash, shots_hash


def prepare_geometry_annotations(
    video_path,
    shots_path,
    output_directory,
    *,
    hand_review_path=None,
    existing_annotations_path=None,
):
    video_path, shots_path, shots, hand_review, video_hash, shots_hash = _documents(
        video_path,
        shots_path,
        hand_review_path,
    )
    output = _safe_directory(output_directory, "geometry annotation preparation")
    try:
        existing_document = None
        existing_rows = {}
        if existing_annotations_path is not None:
            existing_annotations_path = Path(existing_annotations_path)
            existing_document = json.loads(existing_annotations_path.read_text(encoding="utf-8"))
            if (
                existing_document.get("kind") != "guitar-geometry-annotations"
                or existing_document.get("videoSha256") != video_hash
                or existing_document.get("pointOrder") != list(GEOMETRY_POINTS)
                or existing_document.get("coordinateSpace") != "normalized_full_frame"
            ):
                raise EvidenceError("Existing geometry annotations do not match the source video.")
            existing_rows = migrate_annotation_rows(
                existing_document.get("shots"),
                shots["shots"],
            )
        hand_review_pts = {} if hand_review is None else {
            row["shotId"]: row["bestReviewPts"]
            for row in hand_review["shots"]
        }
        selected = {}
        for shot in shots["shots"]:
            existing = existing_rows.get(shot["shotId"])
            selected[shot["shotId"]] = (
                existing["keyframePts"]
                if existing is not None
                else hand_review_pts.get(
                    shot["shotId"],
                    (shot["startPts"] + shot["endPtsExclusive"] - 1) // 2,
                )
            )
        rows = []
        container = av.open(str(video_path))
        try:
            stream = container.streams.video[0]
            best_frames = {}
            shot_index = 0
            for frame in container.decode(stream):
                pts = int(frame.pts)
                while (
                    shot_index + 1 < len(shots["shots"])
                    and pts >= shots["shots"][shot_index + 1]["startPts"]
                ):
                    shot_index += 1
                distance = abs(pts - selected[shot_index])
                if shot_index in best_frames and distance >= best_frames[shot_index][0]:
                    continue
                image = frame.to_ndarray(format="rgb24")
                height = max(1, round(image.shape[0] * 1280 / image.shape[1]))
                preview = cv2.resize(image, (1280, height), interpolation=cv2.INTER_AREA)
                best_frames[shot_index] = (distance, pts, preview)
            if len(best_frames) != len(shots["shots"]):
                missing = sorted(set(range(len(shots["shots"]))) - set(best_frames))
                raise EvidenceError(f"Geometry keyframes were not decoded for shots: {missing}")
            for shot_id, shot in enumerate(shots["shots"]):
                _, pts, preview = best_frames[shot_id]
                relative = Path("keyframes") / f"shot-{shot_id:04d}-pts-{pts}.jpg"
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(
                    str(path),
                    cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92],
                ):
                    raise EvidenceError(f"Unable to write geometry keyframe: {path}")
                existing = existing_rows.get(shot_id)
                rows.append({
                    "shotId": shot_id,
                    "startPts": shot["startPts"],
                    "endPtsExclusive": shot["endPtsExclusive"],
                    "keyframePts": pts,
                    "keyframeImage": str(relative),
                    "state": "unknown" if existing is None else existing["state"],
                    "geometry": None if existing is None else existing["geometry"],
                    "reviewNote": None if existing is None else existing["reviewNote"],
                })
        finally:
            container.close()
        rows.sort(key=lambda row: row["shotId"])
        document = {
            "schemaVersion": 1,
            "kind": "guitar-geometry-annotations",
            "visibility": "private",
            "videoSha256": video_hash,
            "shotsSha256": shots_hash,
            "pointOrder": list(GEOMETRY_POINTS),
            "coordinateSpace": "normalized_full_frame",
            "shots": rows,
            "reviewComplete": all(geometry_row_complete(row) for row in rows),
            "migratedFromAnnotationsSha256": (
                None
                if existing_annotations_path is None
                else sha256(existing_annotations_path)
            ),
            "trainingPerformed": False,
        }
        path = output / "annotations.json"
        _publish(path, document)
        return path, document
    except Exception:
        _cleanup(output)
        raise


def migrate_annotation_rows(existing_rows, new_shots):
    if not isinstance(existing_rows, list):
        raise EvidenceError("Existing geometry annotations contain no shot rows.")
    result = {}
    for shot in new_shots:
        matches = [
            row for row in existing_rows
            if shot["startPts"] <= row.get("keyframePts", -1) < shot["endPtsExclusive"]
        ]
        if len(matches) > 1:
            raise EvidenceError(f"Multiple existing geometry seeds map to shot {shot['shotId']}.")
        if matches:
            result[shot["shotId"]] = matches[0]
    return result


def geometry_row_complete(row):
    geometry = row.get("geometry")
    if row.get("state") == "trackable":
        return isinstance(geometry, dict) and set(geometry) == set(GEOMETRY_POINTS)
    if row.get("state") == "guitar_partial":
        return isinstance(geometry, dict) and bool(geometry)
    return row.get("state") in ("guitar_absent", "transition") and geometry is None


def _point(value, name):
    if not isinstance(value, dict) or set(value) != {"x", "y", "confidence"}:
        raise EvidenceError(f"{name} must contain x, y, and confidence.")
    result = []
    for field in ("x", "y", "confidence"):
        item = value[field]
        if type(item) not in (int, float) or not math.isfinite(item) or not 0 <= item <= 1:
            raise EvidenceError(f"{name}.{field} must be between zero and one.")
        result.append(float(item))
    return result


def validate_geometry_annotations(document, shots, video_hash, shots_hash):
    if not isinstance(document, dict) or document.get("schemaVersion") != 1:
        raise EvidenceError("Geometry annotations have an unsupported schema.")
    if (
        document.get("kind") != "guitar-geometry-annotations"
        or document.get("videoSha256") != video_hash
        or document.get("shotsSha256") != shots_hash
        or document.get("pointOrder") != list(GEOMETRY_POINTS)
        or document.get("coordinateSpace") != "normalized_full_frame"
    ):
        raise EvidenceError("Geometry annotations do not match the source video and shots.")
    rows = document.get("shots")
    if not isinstance(rows, list) or len(rows) != len(shots["shots"]):
        raise EvidenceError("Geometry annotations must contain every shot exactly once.")
    normalized = []
    for expected_shot, row in zip(shots["shots"], rows):
        if not isinstance(row, dict) or set(row) != {
            "shotId",
            "startPts",
            "endPtsExclusive",
            "keyframePts",
            "keyframeImage",
            "state",
            "geometry",
            "reviewNote",
        }:
            raise EvidenceError("Geometry annotation row has an invalid schema.")
        if (
            row["shotId"] != expected_shot["shotId"]
            or row["startPts"] != expected_shot["startPts"]
            or row["endPtsExclusive"] != expected_shot["endPtsExclusive"]
            or not row["startPts"] <= row["keyframePts"] < row["endPtsExclusive"]
        ):
            raise EvidenceError(f"Geometry annotation does not match shot {expected_shot['shotId']}.")
        state = row["state"]
        if state not in GEOMETRY_STATES:
            raise EvidenceError(f"Unsupported guitar geometry state: {state}")
        geometry = row["geometry"]
        if state in ("guitar_absent", "transition", "unknown"):
            if geometry is not None:
                raise EvidenceError(f"{state} shots cannot claim guitar geometry.")
            points = {}
        else:
            if not isinstance(geometry, dict) or not geometry:
                raise EvidenceError(f"{state} shots require reviewed guitar geometry.")
            if not set(geometry) <= set(GEOMETRY_POINTS):
                raise EvidenceError("Geometry contains an unsupported guitar point.")
            if state == "trackable" and set(geometry) != set(GEOMETRY_POINTS):
                raise EvidenceError("Trackable shots require all guitar geometry points.")
            points = {
                name: _point(value, f"shot {row['shotId']} {name}")
                for name, value in geometry.items()
            }
            if all(name in geometry for name in REQUIRED_FRAME_POINTS):
                GuitarCoordinateFrame.from_landmarks({
                    name: geometry[name]
                    for name in REQUIRED_FRAME_POINTS
                })
        normalized.append({
            **row,
            "geometry": points,
        })
    return normalized


def _analysis_gray(image, width):
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)


def _track_step(previous_gray, next_gray, points, valid, confidence, threshold):
    output_points = np.full_like(points, np.nan)
    output_confidence = np.zeros_like(confidence)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return output_points, np.zeros_like(valid), output_confidence
    source = points[indices].astype(np.float32).reshape(-1, 1, 2)
    target, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        next_gray,
        source,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01),
    )
    if target is None or status is None:
        return output_points, np.zeros_like(valid), output_confidence
    reverse, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
        next_gray,
        previous_gray,
        target,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01),
    )
    if reverse is None or reverse_status is None:
        return output_points, np.zeros_like(valid), output_confidence
    target = target.reshape(-1, 2)
    reverse = reverse.reshape(-1, 2)
    errors = np.linalg.norm(reverse - source.reshape(-1, 2), axis=1)
    inside = (
        (target[:, 0] >= 0)
        & (target[:, 0] < next_gray.shape[1])
        & (target[:, 1] >= 0)
        & (target[:, 1] < next_gray.shape[0])
    )
    accepted = (
        status.reshape(-1).astype(bool)
        & reverse_status.reshape(-1).astype(bool)
        & np.isfinite(target).all(axis=1)
        & np.isfinite(errors)
        & (errors <= threshold)
        & inside
    )
    accepted_indices = indices[accepted]
    output_points[accepted_indices] = target[accepted]
    output_confidence[accepted_indices] = np.minimum(
        confidence[accepted_indices],
        np.exp(-errors[accepted] / threshold),
    )
    output_valid = np.zeros_like(valid)
    output_valid[accepted_indices] = True
    return output_points, output_valid, output_confidence


def track_segment(grays, seed_index, seed_points, seed_confidence, threshold=1.5):
    frame_count, point_count = len(grays), len(seed_points)
    coordinates = np.full((frame_count, point_count, 2), np.nan, dtype=np.float32)
    confidence = np.zeros((frame_count, point_count), dtype=np.float32)
    source = np.zeros((frame_count, point_count), dtype=np.int8)
    coordinates[seed_index] = seed_points
    confidence[seed_index] = seed_confidence
    source[seed_index, seed_confidence > 0] = 1
    for direction in (1, -1):
        current_points = seed_points.copy()
        current_confidence = seed_confidence.copy()
        valid = np.isfinite(seed_points).all(axis=1) & (seed_confidence > 0)
        index = seed_index
        while 0 <= index + direction < frame_count:
            next_index = index + direction
            current_points, valid, current_confidence = _track_step(
                grays[index],
                grays[next_index],
                current_points,
                valid,
                current_confidence,
                threshold,
            )
            coordinates[next_index] = current_points
            confidence[next_index] = current_confidence
            source[next_index, valid] = 2
            index = next_index
    return coordinates, confidence, source


def _process_segment(grays, pts, annotation, analysis_width, threshold):
    point_count = len(GEOMETRY_POINTS)
    coordinates = np.full((len(pts), point_count, 2), np.nan, dtype=np.float32)
    confidence = np.zeros((len(pts), point_count), dtype=np.float32)
    source = np.zeros((len(pts), point_count), dtype=np.int8)
    states = np.full(len(pts), STATE_CODES[annotation["state"]], dtype=np.int8)
    if not annotation["geometry"]:
        return coordinates, confidence, source, states
    try:
        seed_index = pts.index(annotation["keyframePts"])
    except ValueError as error:
        raise EvidenceError(
            f"Geometry keyframe PTS was not decoded for shot {annotation['shotId']}."
        ) from error
    height, width = grays[0].shape
    seed_points = np.full((point_count, 2), np.nan, dtype=np.float32)
    seed_confidence = np.zeros(point_count, dtype=np.float32)
    for point_index, name in enumerate(GEOMETRY_POINTS):
        point = annotation["geometry"].get(name)
        if point is None:
            continue
        seed_points[point_index] = (point[0] * width, point[1] * height)
        seed_confidence[point_index] = point[2]
    coordinates, confidence, source = track_segment(
        grays,
        seed_index,
        seed_points,
        seed_confidence,
        threshold,
    )
    coordinates[:, :, 0] /= analysis_width
    coordinates[:, :, 1] /= height
    if annotation["state"] == "trackable":
        required = confidence[:, :len(REQUIRED_FRAME_POINTS)]
        states[np.any(required <= 0, axis=1)] = STATE_CODES["calibration_unstable"]
    elif annotation["state"] == "guitar_partial":
        states[np.all(confidence <= 0, axis=1)] = STATE_CODES["calibration_unstable"]
    return coordinates, confidence, source, states


def track_geometry(
    video_path,
    shots_path,
    annotations_path,
    output_directory,
    *,
    analysis_width=960,
    flow_threshold=1.5,
):
    video_path, shots_path, shots, _, video_hash, shots_hash = _documents(
        video_path,
        shots_path,
    )
    annotations_path = Path(annotations_path)
    if not annotations_path.is_file():
        raise EvidenceError("Geometry tracking requires reviewed annotations.")
    if type(analysis_width) is not int or analysis_width < 320:
        raise EvidenceError("Geometry analysis width must be an integer >= 320.")
    if type(flow_threshold) not in (int, float) or not math.isfinite(flow_threshold) or flow_threshold <= 0:
        raise EvidenceError("Geometry optical-flow threshold must be positive.")
    document = json.loads(annotations_path.read_text(encoding="utf-8"))
    annotations = validate_geometry_annotations(document, shots, video_hash, shots_hash)
    if document.get("reviewComplete") is not True:
        raise EvidenceError("Geometry annotations must be explicitly marked reviewComplete.")
    output = _safe_directory(output_directory, "geometry tracking")
    all_pts = []
    all_shot_ids = []
    all_coordinates = []
    all_confidence = []
    all_source = []
    all_states = []
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        shot_index = 0
        segment_pts = []
        segment_grays = []

        def flush():
            if not segment_pts:
                return
            coordinates, confidence, source, states = _process_segment(
                segment_grays,
                segment_pts,
                annotations[shot_index],
                analysis_width,
                float(flow_threshold),
            )
            all_pts.extend(segment_pts)
            all_shot_ids.extend([shot_index] * len(segment_pts))
            all_coordinates.extend(coordinates)
            all_confidence.extend(confidence)
            all_source.extend(source)
            all_states.extend(states)

        for frame in container.decode(stream):
            pts = int(frame.pts)
            while (
                shot_index + 1 < len(shots["shots"])
                and pts >= shots["shots"][shot_index + 1]["startPts"]
            ):
                flush()
                segment_pts = []
                segment_grays = []
                shot_index += 1
            segment_pts.append(pts)
            segment_grays.append(_analysis_gray(frame.to_ndarray(format="rgb24"), analysis_width))
        flush()
        if len(all_pts) != shots["frameCount"]:
            raise EvidenceError("Geometry tracking frame count differs from shot inspection.")
    except Exception:
        _cleanup(output)
        raise
    finally:
        container.close()
    arrays_path = output / "geometry.npz"
    np.savez_compressed(
        arrays_path,
        pts=np.asarray(all_pts, dtype=np.int64),
        shot_id=np.asarray(all_shot_ids, dtype=np.int32),
        coordinates=np.asarray(all_coordinates, dtype=np.float32),
        confidence=np.asarray(all_confidence, dtype=np.float32),
        source=np.asarray(all_source, dtype=np.int8),
        state=np.asarray(all_states, dtype=np.int8),
    )
    confidence = np.asarray(all_confidence)
    required = confidence[:, :len(REQUIRED_FRAME_POINTS)]
    report = {
        "schemaVersion": 1,
        "kind": "guitar-geometry-observations",
        "visibility": "private",
        "videoSha256": video_hash,
        "shotsSha256": shots_hash,
        "annotationsSha256": sha256(annotations_path),
        "timeBase": shots["timeBase"],
        "frameCount": len(all_pts),
        "shotCount": len(shots["shots"]),
        "pointOrder": list(GEOMETRY_POINTS),
        "stateEncoding": STATE_CODES,
        "analysisWidth": analysis_width,
        "flowThresholdPixels": float(flow_threshold),
        "framesWithCoordinateFrame": int(np.count_nonzero(np.all(required > 0, axis=1))),
        "coordinateFrameCoverage": float(np.mean(np.all(required > 0, axis=1))),
        "trackingResetCount": len(shots["shots"]),
        "reviewRequired": True,
        "trainingPerformed": False,
        "arrays": arrays_path.name,
    }
    report_path = output / "geometry.json"
    _publish(report_path, report)
    return report_path, report


def review_geometry(video_path, shots_path, geometry_path, output_directory):
    video_path, shots_path, shots, _, video_hash, shots_hash = _documents(
        video_path,
        shots_path,
    )
    geometry_path = Path(geometry_path)
    arrays_path = geometry_path.with_name("geometry.npz")
    if not geometry_path.is_file() or not arrays_path.is_file():
        raise EvidenceError("Geometry review requires a geometry report and arrays.")
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if (
        geometry.get("kind") != "guitar-geometry-observations"
        or geometry.get("videoSha256") != video_hash
        or geometry.get("shotsSha256") != shots_hash
    ):
        raise EvidenceError("Geometry observations do not match the review video and shots.")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        pts = np.array(arrays["pts"], copy=True)
        shot_ids = np.array(arrays["shot_id"], copy=True)
        coordinates = np.array(arrays["coordinates"], copy=True)
        confidence = np.array(arrays["confidence"], copy=True)
        states = np.array(arrays["state"], copy=True)
    output = _safe_directory(output_directory, "geometry review")
    rows = []
    selected = {}
    try:
        for shot in shots["shots"]:
            indices = np.flatnonzero(shot_ids == shot["shotId"])
            required = confidence[indices, :len(REQUIRED_FRAME_POINTS)]
            frame_confidence = np.min(required, axis=1)
            best_offset = max(
                range(len(indices)),
                key=lambda offset: (
                    float(frame_confidence[offset]),
                    -abs(int(pts[indices[offset]]) - (shot["startPts"] + shot["endPtsExclusive"]) // 2),
                ),
            )
            best = int(indices[best_offset])
            review_frames = []
            for label, index in (
                ("start", int(indices[0])),
                ("best", best),
                ("end", int(indices[-1])),
            ):
                if any(frame["pts"] == int(pts[index]) for frame in review_frames):
                    continue
                relative = f"review\\shot-{shot['shotId']:04d}-{label}-pts-{int(pts[index])}.jpg"
                selected[int(pts[index])] = (shot["shotId"], index, relative)
                review_frames.append({
                    "label": label,
                    "pts": int(pts[index]),
                    "state": GEOMETRY_STATES[int(states[index])],
                    "coordinateFrameConfidence": float(np.min(
                        confidence[index, :len(REQUIRED_FRAME_POINTS)]
                    )),
                    "reviewFrame": relative,
                })
            rows.append({
                "shotId": shot["shotId"],
                "startSeconds": shot["startSeconds"],
                "endSecondsExclusive": shot["endSecondsExclusive"],
                "state": GEOMETRY_STATES[int(states[best])],
                "coordinateFrameCoverage": float(np.mean(np.all(required > 0, axis=1))),
                "medianCoordinateFrameConfidence": float(np.median(frame_confidence)),
                "bestReviewPts": int(pts[best]),
                "bestReviewConfidence": float(frame_confidence[best_offset]),
                "reviewFrame": next(
                    frame["reviewFrame"]
                    for frame in review_frames
                    if frame["pts"] == int(pts[best])
                ),
                "reviewFrames": review_frames,
            })
        container = av.open(str(video_path))
        try:
            stream = container.streams.video[0]
            remaining = dict(selected)
            colors = (
                (255, 160, 80),
                (80, 220, 255),
                (80, 255, 120),
                (255, 120, 220),
                (80, 80, 255),
                (255, 255, 80),
            )
            for frame in container.decode(stream):
                selection = remaining.pop(int(frame.pts), None)
                if selection is None:
                    continue
                shot_id, index, relative = selection
                image = frame.to_ndarray(format="bgr24")
                height = max(1, round(image.shape[0] * 1280 / image.shape[1]))
                image = cv2.resize(image, (1280, height), interpolation=cv2.INTER_AREA)
                pixels = []
                for point_index, name in enumerate(GEOMETRY_POINTS):
                    point = coordinates[index, point_index]
                    if not np.isfinite(point).all() or confidence[index, point_index] <= 0:
                        pixels.append(None)
                        continue
                    pixel = (round(point[0] * image.shape[1]), round(point[1] * image.shape[0]))
                    pixels.append(pixel)
                    cv2.circle(image, pixel, 6, colors[point_index], -1, cv2.LINE_AA)
                    cv2.putText(
                        image,
                        name,
                        (pixel[0] + 8, pixel[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        .55,
                        colors[point_index],
                        2,
                        cv2.LINE_AA,
                    )
                for left, right in ((0, 1), (2, 3), (4, 5)):
                    if pixels[left] is not None and pixels[right] is not None:
                        cv2.line(image, pixels[left], pixels[right], (255, 255, 255), 2, cv2.LINE_AA)
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise EvidenceError(f"Unable to write geometry review frame: {path}")
                if not remaining:
                    break
            if remaining:
                raise EvidenceError(f"Geometry review frames were not decoded: {sorted(remaining)}")
        finally:
            container.close()
        report = {
            "schemaVersion": 1,
            "kind": "guitar-geometry-review",
            "visibility": "private",
            "videoSha256": video_hash,
            "shotsSha256": shots_hash,
            "geometrySha256": sha256(geometry_path),
            "shotCount": len(rows),
            "shotsWithCoordinateFrame": sum(row["coordinateFrameCoverage"] > 0 for row in rows),
            "shots": rows,
            "reviewRequired": True,
            "trainingPerformed": False,
        }
        report_path = output / "review.json"
        _publish(report_path, report)
        return report_path, report
    except Exception:
        _cleanup(output)
        raise
