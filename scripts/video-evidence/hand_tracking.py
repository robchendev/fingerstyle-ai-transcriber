"""Per-shot MediaPipe hand observations with optional pose-guided crops."""

from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path
from uuid import uuid4

import av
import cv2
import mediapipe as mp
import numpy as np

from core import EvidenceError, private_output, sha256


def _safe_directory(path):
    marker = private_output(Path(path) / ".directory-check")
    directory = marker.parent
    if directory.exists():
        raise EvidenceError(f"Refusing to overwrite existing hand tracking: {directory}")
    directory.mkdir(parents=True)
    return directory


def _hand_landmarker(model_path, running_mode=mp.tasks.vision.RunningMode.VIDEO, confidence=.45):
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=running_mode,
        num_hands=2,
        min_hand_detection_confidence=confidence,
        min_hand_presence_confidence=confidence,
        min_tracking_confidence=confidence,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def _pose_landmarker(model_path):
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=.35,
        min_pose_presence_confidence=.35,
        min_tracking_confidence=.35,
        output_segmentation_masks=False,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def _shot_for_pts(shots, pts, current):
    while current + 1 < len(shots) and pts >= shots[current + 1]["startPts"]:
        current += 1
    if not shots[current]["startPts"] <= pts < shots[current]["endPtsExclusive"]:
        raise EvidenceError("Decoded PTS falls outside the declared shot coverage.")
    return current


def _scaled(image, maximum_dimension):
    scale = min(1, maximum_dimension / max(image.shape[:2]))
    if scale == 1:
        return image
    return cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _landmark_confidence(landmark):
    values = [
        value for value in (
            getattr(landmark, "visibility", None),
            getattr(landmark, "presence", None),
        )
        if value is not None
    ]
    return min(values, default=1.0)


def _square_crop(center_x, center_y, side, image_width, image_height):
    side = min(round(side), image_width, image_height)
    left = round(max(0, min(image_width - side, center_x - side / 2)))
    top = round(max(0, min(image_height - side, center_y - side / 2)))
    return left, top, left + side, top + side


def pose_hand_crops(landmarks, image_width, image_height):
    if len(landmarks) < 17:
        return []
    shoulders = landmarks[11], landmarks[12]
    if any(_landmark_confidence(landmark) < .25 for landmark in shoulders):
        return []
    shoulder_span = math.hypot(
        (shoulders[0].x - shoulders[1].x) * image_width,
        (shoulders[0].y - shoulders[1].y) * image_height,
    )
    crops = []
    for elbow_index, wrist_index in ((13, 15), (14, 16)):
        elbow = landmarks[elbow_index]
        wrist = landmarks[wrist_index]
        if min(_landmark_confidence(elbow), _landmark_confidence(wrist)) < .25:
            continue
        elbow_x, elbow_y = elbow.x * image_width, elbow.y * image_height
        wrist_x, wrist_y = wrist.x * image_width, wrist.y * image_height
        forearm = math.hypot(wrist_x - elbow_x, wrist_y - elbow_y)
        center_x = wrist_x + .5 * (wrist_x - elbow_x)
        center_y = wrist_y + .5 * (wrist_y - elbow_y)
        side = max(256, min(640, shoulder_span * 2.3, forearm * 3))
        crops.append({
            "crop": _square_crop(center_x, center_y, side, image_width, image_height),
            "expectedWrist": (wrist.x, wrist.y),
        })
    return crops


def remap_image_landmarks(
    landmarks,
    crop,
    image_width,
    image_height,
    inverse_affine=None,
):
    result = np.array(landmarks, dtype=np.float32, copy=True)
    if inverse_affine is not None:
        crop_width = crop[2] - crop[0]
        pixels = result[:, :2] * crop_width
        pixels = cv2.transform(pixels[np.newaxis], inverse_affine)[0]
        result[:, :2] = pixels / crop_width
    if crop is None:
        return result
    left, top, right, bottom = crop
    result[:, 0] = (left + result[:, 0] * (right - left)) / image_width
    result[:, 1] = (top + result[:, 1] * (bottom - top)) / image_height
    result[:, 2] *= (right - left) / image_width
    return result


def _observations(result, image_width, image_height, crop=None, inverse_affine=None):
    frame_image = np.full((2, 21, 3), np.nan, dtype=np.float32)
    frame_world = np.full((2, 21, 3), np.nan, dtype=np.float32)
    frame_handedness = np.full(2, -1, dtype=np.int8)
    frame_scores = np.zeros(2, dtype=np.float32)
    hands = min(2, len(result.hand_landmarks))
    for hand_index in range(hands):
        image_points = [
            (landmark.x, landmark.y, landmark.z)
            for landmark in result.hand_landmarks[hand_index]
        ]
        frame_image[hand_index] = remap_image_landmarks(
            image_points,
            crop,
            image_width,
            image_height,
            inverse_affine,
        )
        if inverse_affine is None:
            for landmark_index, landmark in enumerate(result.hand_world_landmarks[hand_index]):
                frame_world[hand_index, landmark_index] = (landmark.x, landmark.y, landmark.z)
        category = result.handedness[hand_index][0]
        frame_handedness[hand_index] = 0 if category.category_name.lower() == "left" else 1
        frame_scores[hand_index] = category.score
    return hands, frame_image, frame_world, frame_handedness, frame_scores


def merge_observations(*observations):
    candidates = []
    for hands, image, world, handedness, scores in observations:
        for hand_index in range(hands):
            wrist = image[hand_index, 0, :2]
            duplicate = next(
                (
                    candidate_index for candidate_index, candidate in enumerate(candidates)
                    if np.linalg.norm(candidate[0][0, :2] - wrist) < .08
                ),
                None,
            )
            value = (
                image[hand_index],
                world[hand_index],
                handedness[hand_index],
                scores[hand_index],
            )
            if duplicate is None:
                candidates.append(value)
    candidates.sort(key=lambda candidate: float(candidate[3]), reverse=True)
    frame_image = np.full((2, 21, 3), np.nan, dtype=np.float32)
    frame_world = np.full((2, 21, 3), np.nan, dtype=np.float32)
    frame_handedness = np.full(2, -1, dtype=np.int8)
    frame_scores = np.zeros(2, dtype=np.float32)
    for index, candidate in enumerate(candidates[:2]):
        frame_image[index], frame_world[index], frame_handedness[index], frame_scores[index] = candidate
    hands = min(2, len(candidates))
    return hands, frame_image, frame_world, frame_handedness, frame_scores


def observation_near_wrist(
    observation,
    expected_wrist,
    crop,
    image_width,
    image_height,
    maximum_crop_distance=.38,
):
    hands, image, world, handedness, scores = observation
    if hands == 0:
        return observation
    crop_width = (crop[2] - crop[0]) / image_width
    crop_height = (crop[3] - crop[1]) / image_height
    distances = [
        math.hypot(
            (image[index, 0, 0] - expected_wrist[0]) / crop_width,
            (image[index, 0, 1] - expected_wrist[1]) / crop_height,
        )
        for index in range(hands)
    ]
    selected = int(np.argmin(distances))
    if distances[selected] > maximum_crop_distance:
        return _observations_from_candidates([])
    return _observations_from_candidates([(
        image[selected],
        world[selected],
        handedness[selected],
        scores[selected],
    )])


def _observations_from_candidates(candidates):
    frame_image = np.full((2, 21, 3), np.nan, dtype=np.float32)
    frame_world = np.full((2, 21, 3), np.nan, dtype=np.float32)
    frame_handedness = np.full(2, -1, dtype=np.int8)
    frame_scores = np.zeros(2, dtype=np.float32)
    for index, candidate in enumerate(candidates[:2]):
        frame_image[index], frame_world[index], frame_handedness[index], frame_scores[index] = candidate
    return min(2, len(candidates)), frame_image, frame_world, frame_handedness, frame_scores


def prefer_guided_detection(full_count, guided_count):
    return guided_count > full_count


def track_hands(
    video_path,
    shots_path,
    model_path,
    output_directory,
    *,
    maximum_dimension=1280,
    pose_model_path=None,
):
    video_path, shots_path, model_path = map(Path, (video_path, shots_path, model_path))
    pose_model_path = None if pose_model_path is None else Path(pose_model_path)
    required = (video_path, shots_path, model_path) if pose_model_path is None else (
        video_path,
        shots_path,
        model_path,
        pose_model_path,
    )
    if not all(path.is_file() for path in required):
        raise EvidenceError("Hand tracking requires video, shot report, and hand-landmarker model files.")
    if type(maximum_dimension) is not int or maximum_dimension < 256:
        raise EvidenceError("Hand tracking maximum dimension must be an integer >= 256.")
    shots_document = json.loads(shots_path.read_text(encoding="utf-8"))
    if shots_document.get("kind") != "video-shot-inspection" or shots_document.get("videoSha256") != sha256(video_path):
        raise EvidenceError("Shot report is not bound to this video.")
    shots = shots_document.get("shots")
    if not isinstance(shots, list) or not shots:
        raise EvidenceError("Shot report contains no shots.")
    output = _safe_directory(output_directory)
    count = 0
    detected_frames = 0
    pts_values = []
    shot_ids = []
    hand_counts = []
    image_landmarks = []
    world_landmarks = []
    handedness = []
    handedness_scores = []
    detection_sources = []
    guided_rois = []
    pose_attempts = 0
    pose_frames = 0
    guided_attempts = 0
    guided_improvements = 0
    container = av.open(str(video_path))
    full_tracker = None
    guided_tracker = None
    pose_tracker = None
    try:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if len(streams) != 1 or streams[0].index != shots_document["videoStreamIndex"]:
            raise EvidenceError("Video stream differs from the shot report.")
        stream = streams[0]
        time_base = Fraction(stream.time_base)
        shot_index = 0
        prior_shot = None
        segment_start_pts = None
        for frame in container.decode(stream):
            if frame.pts is None:
                raise EvidenceError("Decoded frame has no presentation timestamp.")
            pts = int(frame.pts)
            shot_index = _shot_for_pts(shots, pts, shot_index)
            if shot_index != prior_shot:
                for tracker in (full_tracker, guided_tracker, pose_tracker):
                    if tracker is not None:
                        tracker.close()
                full_tracker = _hand_landmarker(model_path)
                guided_tracker = (
                    _hand_landmarker(
                        model_path,
                        running_mode=mp.tasks.vision.RunningMode.IMAGE,
                        confidence=.25,
                    )
                    if pose_model_path is not None
                    else None
                )
                pose_tracker = _pose_landmarker(pose_model_path) if pose_model_path is not None else None
                segment_start_pts = pts
                prior_shot = shot_index
            original = frame.to_ndarray(format="rgb24")
            image_height, image_width = original.shape[:2]
            image = _scaled(original, maximum_dimension)
            timestamp_ms = round(float((pts - segment_start_pts) * time_base) * 1000)
            full_result = full_tracker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=image),
                timestamp_ms,
            )
            selected = _observations(full_result, image_width, image_height)
            source = 1 if selected[0] else 0
            crops = []
            if pose_tracker is not None and selected[0] < 2:
                pose_attempts += 1
                pose_result = pose_tracker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=image),
                    timestamp_ms,
                )
                if pose_result.pose_landmarks:
                    pose_frames += 1
                    crops = pose_hand_crops(
                        pose_result.pose_landmarks[0],
                        image_width,
                        image_height,
                    )
                if crops:
                    guided_attempts += 1
                    combined = selected
                    for crop_evidence in crops:
                        crop = crop_evidence["crop"]
                        left, top, right, bottom = crop
                        crop_image = original[top:bottom, left:right]
                        side = right - left
                        best = None
                        for angle in (0, 45, -45):
                            affine = cv2.getRotationMatrix2D((side / 2, side / 2), angle, 1)
                            rotated = cv2.warpAffine(crop_image, affine, (side, side))
                            guided_image = cv2.resize(
                                rotated,
                                (768, 768),
                                interpolation=cv2.INTER_CUBIC,
                            )
                            guided_result = guided_tracker.detect(
                                mp.Image(image_format=mp.ImageFormat.SRGB, data=guided_image)
                            )
                            observation = _observations(
                                guided_result,
                                image_width,
                                image_height,
                                crop,
                                cv2.invertAffineTransform(affine),
                            )
                            observation = observation_near_wrist(
                                observation,
                                crop_evidence["expectedWrist"],
                                crop,
                                image_width,
                                image_height,
                            )
                            if best is None or observation[0] > best[0]:
                                best = observation
                            if best[0] == 2:
                                break
                        combined = merge_observations(combined, best)
                        if combined[0] == 2:
                            break
                    if prefer_guided_detection(selected[0], combined[0]):
                        selected = combined
                        source = 2
                        guided_improvements += 1
            hands, frame_image, frame_world, frame_handedness, frame_scores = selected
            pts_values.append(pts)
            shot_ids.append(shot_index)
            hand_counts.append(hands)
            image_landmarks.append(frame_image)
            world_landmarks.append(frame_world)
            handedness.append(frame_handedness)
            handedness_scores.append(frame_scores)
            detection_sources.append(source)
            normalized_crops = [
                (
                    crop_evidence["crop"][0] / image_width,
                    crop_evidence["crop"][1] / image_height,
                    crop_evidence["crop"][2] / image_width,
                    crop_evidence["crop"][3] / image_height,
                )
                for crop_evidence in crops[:2]
            ]
            normalized_crops.extend(
                [(np.nan, np.nan, np.nan, np.nan)] * (2 - len(normalized_crops))
            )
            guided_rois.append(normalized_crops)
            count += 1
            detected_frames += hands > 0
        for tracker in (full_tracker, guided_tracker, pose_tracker):
            if tracker is not None:
                tracker.close()
        full_tracker = guided_tracker = pose_tracker = None
        if count == 0:
            raise EvidenceError("Hand tracking decoded no frames.")
        arrays_path = output / "hands.npz"
        np.savez_compressed(
            arrays_path,
            pts=np.asarray(pts_values, dtype=np.int64),
            shot_id=np.asarray(shot_ids, dtype=np.int32),
            hand_count=np.asarray(hand_counts, dtype=np.int8),
            image_landmarks=np.asarray(image_landmarks, dtype=np.float32),
            world_landmarks=np.asarray(world_landmarks, dtype=np.float32),
            handedness=np.asarray(handedness, dtype=np.int8),
            handedness_score=np.asarray(handedness_scores, dtype=np.float32),
            detection_source=np.asarray(detection_sources, dtype=np.int8),
            guided_rois=np.asarray(guided_rois, dtype=np.float32),
        )
        report = {
            "schemaVersion": 1,
            "kind": "fingerstyle-hand-observations",
            "visibility": "private",
            "videoSha256": sha256(video_path),
            "shotsSha256": sha256(shots_path),
            "modelSha256": sha256(model_path),
            "poseModelSha256": None if pose_model_path is None else sha256(pose_model_path),
            "timeBase": [time_base.numerator, time_base.denominator],
            "frameCount": count,
            "framesWithHands": detected_frames,
            "handFrameCoverage": detected_frames / count,
            "shotCount": len(shots),
            "trackerResetCount": len(shots),
            "poseTrackerResetCount": 0 if pose_model_path is None else len(shots),
            "maximumInputDimension": maximum_dimension,
            "poseAttempts": pose_attempts,
            "poseFrames": pose_frames,
            "guidedCropAttempts": guided_attempts,
            "framesImprovedByGuidedCrop": guided_improvements,
            "guidedCropSelectionRule": "accept_only_when_hand_count_increases",
            "guidedCropRotationsDegrees": [0, 45, -45],
            "guidedHandConfidence": .25,
            "guidedMaximumPoseWristDistance": .38,
            "guidedWorldLandmarksAvailable": False,
            "anatomicalHandednessOnly": True,
            "playingRolesAssigned": False,
            "guitarRelativeCoordinatesAvailable": False,
            "trainingPerformed": False,
            "arrays": arrays_path.name,
        }
        report_path = output / "hands.json"
        pending = report_path.with_name(f".{report_path.name}.{uuid4().hex}.part")
        try:
            pending.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
            pending.replace(report_path)
        finally:
            pending.unlink(missing_ok=True)
        return report_path, report
    except Exception:
        for tracker in (full_tracker, guided_tracker, pose_tracker):
            if tracker is not None:
                tracker.close()
        for path in sorted(output.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        output.rmdir()
        raise
    finally:
        container.close()
