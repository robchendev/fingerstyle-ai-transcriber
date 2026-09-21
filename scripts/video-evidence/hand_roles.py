"""Geometry-based hand roles with per-shot continuity and explicit unknowns."""

from dataclasses import asdict, dataclass
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path

import av
import cv2
import numpy as np

from core import EvidenceError, GuitarCoordinateFrame, sha256
from geometry import GEOMETRY_POINTS, STATE_CODES, _cleanup, _publish, _safe_directory, annotations_ready, validate_geometry_annotations
from hand_tracking import same_hand_landmarks


ROLE_CODES = {"unknown": 0, "fretting": 1, "plucking": 2}
ASSIGNMENT_SOURCES = {"unavailable": 0, "guitar_geometry": 1, "reviewed_screen_order": 2}
REASON_CODES = {
    "assigned": 0, "hand_absent": 1, "geometry_unavailable": 2,
    "landmarks_unavailable": 3, "outside_role_regions": 4,
    "ambiguous_role": 5, "competing_hands": 6, "temporal_confirmation": 7,
    "body_geometry_unavailable": 8,
}
ANCHOR_NAMES = ("wrist", "palm", "fingertips")
PALM = (0, 5, 9, 13, 17)
TIPS = (4, 8, 12, 16, 20)


@dataclass(frozen=True)
class RoleConfig:
    plucking_screen_side: str = "geometry"
    minimum_geometry_confidence: float = .5
    minimum_role_score: float = .6
    ambiguity_margin: float = .2
    confirmation_seconds: float = .08
    maximum_gap_seconds: float = .25
    minimum_neck_pixels: float = 24.
    minimum_board_pixels: float = 4.

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "plucking_screen_side":
                if value not in ("geometry", "left", "right"):
                    raise EvidenceError("Plucking screen side must be geometry, left or right.")
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise EvidenceError(f"{name} must be a finite nonnegative number.")
        if not 0 < self.minimum_geometry_confidence <= 1 or not 0 < self.minimum_role_score <= 1 or not 0 < self.ambiguity_margin <= 1:
            raise EvidenceError("Geometry, role and ambiguity thresholds must be above zero through one.")
        if self.maximum_gap_seconds <= 0 or self.minimum_neck_pixels <= 0 or self.minimum_board_pixels <= 0:
            raise EvidenceError("Continuity gap and minimum geometry sizes must be positive.")


def _read_arrays(path, shapes):
    try:
        with np.load(path, allow_pickle=False) as source:
            values = {name: np.array(source[name], copy=True) for name in shapes}
    except (ValueError, KeyError) as error:
        raise EvidenceError(f"Malformed observation archive {path}: {error}") from error
    for name, (dtype, shape) in shapes.items():
        value = values[name]
        if value.dtype != dtype or value.shape != shape:
            raise EvidenceError(f"{path.name}:{name} requires {dtype} and shape {shape}, not {value.dtype}/{value.shape}.")
        if value.dtype.kind == "f" and np.isinf(value).any():
            raise EvidenceError(f"{path.name}:{name} contains infinite values.")
    return values


def load_role_inputs(video_path, shots_path, hands_path, geometry_path, annotations_path):
    paths = {name: Path(path).resolve() for name, path in (
        ("video", video_path), ("shots", shots_path), ("hands", hands_path),
        ("geometry", geometry_path), ("annotations", annotations_path),
    )}
    paths["handArrays"] = paths["hands"].with_name("hands.npz")
    paths["geometryArrays"] = paths["geometry"].with_name("geometry.npz")
    if not all(path.is_file() for path in paths.values()):
        raise EvidenceError("Role assignment requires video, shots, hand/geometry reports and arrays, and reviewed annotations.")
    hashes = {name: sha256(path) for name, path in paths.items()}
    documents = {name: json.loads(paths[name].read_text(encoding="utf-8")) for name in ("shots", "hands", "geometry", "annotations")}
    if any(not isinstance(document, dict) for document in documents.values()):
        raise EvidenceError("Observation and annotation reports must be JSON objects.")
    shots, hands, geometry, annotations = (documents[name] for name in ("shots", "hands", "geometry", "annotations"))
    for name, kind in (("shots", "video-shot-inspection"), ("hands", "fingerstyle-hand-observations"), ("geometry", "guitar-geometry-observations")):
        document = documents[name]
        if type(document.get("schemaVersion")) is not int or document["schemaVersion"] != 1 or document.get("kind") != kind or document.get("videoSha256") != hashes["video"]:
            raise EvidenceError(f"{name} report does not match the source video or schema.")
    for name in ("hands", "geometry"):
        if documents[name].get("shotsSha256") != hashes["shots"]:
            raise EvidenceError(f"{name} report references different shot boundaries.")
        if documents[name].get("timeBase") != shots.get("timeBase"):
            raise EvidenceError(f"{name} and shot report time bases differ.")
    if geometry.get("annotationsSha256") != hashes["annotations"] or not annotations_ready(annotations):
        raise EvidenceError("Geometry must reference these completed reviewed or automatically prepared annotations.")
    if not isinstance(shots.get("shots"), list) or not shots["shots"] or any(
        not isinstance(row, dict) or not {"shotId", "startPts", "endPtsExclusive"} <= row.keys()
        for row in shots["shots"]
    ):
        raise EvidenceError("Shot report requires complete shot IDs and PTS intervals.")
    validate_geometry_annotations(annotations, shots, hashes["video"], hashes["shots"])
    if geometry.get("pointOrder") != list(GEOMETRY_POINTS) or geometry.get("stateEncoding") != STATE_CODES:
        raise EvidenceError("Geometry point/state encoding differs from the supported contract.")
    if hands.get("arrays") != "hands.npz" or geometry.get("arrays") != "geometry.npz":
        raise EvidenceError("Observation report must reference its local canonical array filename.")
    for name, key in (("hands", "handArrays"), ("geometry", "geometryArrays")):
        if documents[name].get("arraysSha256") is not None and documents[name]["arraysSha256"] != hashes[key]:
            raise EvidenceError(f"{name} array hash differs from its report.")
    count = shots.get("frameCount")
    if type(count) is not int or count <= 0 or any(documents[name].get("frameCount") != count for name in ("hands", "geometry")):
        raise EvidenceError("Observation frame counts must match the shot inspection.")
    rows = shots.get("shots")
    if not isinstance(rows, list) or not rows or shots.get("shotCount") != len(rows):
        raise EvidenceError("Shot inspection requires a nonempty, complete shot list.")
    if any(documents[name].get("shotCount") != len(rows) for name in ("hands", "geometry")):
        raise EvidenceError("Observation shot counts disagree.")
    time_base = shots.get("timeBase")
    if not isinstance(time_base, list) or len(time_base) != 2 or any(type(value) is not int or value <= 0 for value in time_base):
        raise EvidenceError("PTS time base must be an explicit positive rational pair.")
    if any(type(shots.get(name)) is not int or shots[name] <= 0 for name in ("width", "height")):
        raise EvidenceError("Source image dimensions must be positive integers.")
    hand = _read_arrays(paths["handArrays"], {
        "pts": (np.dtype("int64"), (count,)), "shot_id": (np.dtype("int32"), (count,)),
        "hand_count": (np.dtype("int8"), (count,)),
        "image_landmarks": (np.dtype("float32"), (count, 2, 21, 3)),
        "handedness": (np.dtype("int8"), (count, 2)),
        "handedness_score": (np.dtype("float32"), (count, 2)),
        "detection_source": (np.dtype("int8"), (count,)),
    })
    geo = _read_arrays(paths["geometryArrays"], {
        "pts": (np.dtype("int64"), (count,)), "shot_id": (np.dtype("int32"), (count,)),
        "coordinates": (np.dtype("float32"), (count, 6, 2)),
        "confidence": (np.dtype("float32"), (count, 6)),
        "source": (np.dtype("int8"), (count, 6)), "state": (np.dtype("int8"), (count,)),
    })
    if not np.array_equal(hand["pts"], geo["pts"]) or not np.array_equal(hand["shot_id"], geo["shot_id"]):
        raise EvidenceError("Hand and geometry PTS/shot arrays must match exactly; no interpolation is permitted.")
    if np.any(np.diff(hand["pts"]) <= 0) or hand["pts"][0] != shots.get("firstPts") or hand["pts"][-1] != shots.get("lastPts"):
        raise EvidenceError("PTS must increase strictly and match shot-inspection endpoints.")
    for index, row in enumerate(rows):
        if row.get("shotId") != index or type(row.get("startPts")) is not int or type(row.get("endPtsExclusive")) is not int or row["startPts"] >= row["endPtsExclusive"]:
            raise EvidenceError("Shots require ordered integer IDs and valid integer PTS bounds.")
        if index and rows[index - 1]["endPtsExclusive"] != row["startPts"]:
            raise EvidenceError("Shot intervals must be contiguous and nonoverlapping.")
        expected = (hand["pts"] >= row["startPts"]) & (hand["pts"] < row["endPtsExclusive"])
        if not expected.any() or not np.array_equal(expected, hand["shot_id"] == index):
            raise EvidenceError(f"Observation shot IDs violate shot {index}'s PTS interval.")
    if (hand["hand_count"] < 0).any() or (hand["hand_count"] > 2).any():
        raise EvidenceError("Hand counts must be zero, one or two.")
    observed = np.isfinite(hand["image_landmarks"][..., :2]).all(-1).any(-1)
    if not np.array_equal(observed.sum(-1), hand["hand_count"]):
        raise EvidenceError("Hand counts disagree with available landmark observations.")
    for values, upper, label in (
        (geo["confidence"], 1, "geometry confidence"), (hand["handedness_score"], 1, "anatomical handedness score"),
    ):
        if not np.isfinite(values).all() or (values < 0).any() or (values > upper).any():
            raise EvidenceError(f"Invalid {label} range.")
    if not np.isin(geo["state"], list(STATE_CODES.values())).all() or not np.isin(geo["source"], [0, 1, 2]).all():
        raise EvidenceError("Unknown geometry state or tracking source.")
    if not np.isin(hand["handedness"], [-1, 0, 1]).all() or not np.isin(hand["detection_source"], [0, 1, 2]).all():
        raise EvidenceError("Unknown hand observation encoding.")
    if not np.isfinite(geo["coordinates"][geo["confidence"] > 0]).all():
        raise EvidenceError("Available geometry coordinates must be finite.")
    return {"paths": paths, "hashes": hashes, "documents": documents, "hand": hand, "geometry": geo}


def _point(value, confidence=1.):
    return {"x": float(value[0]), "y": float(value[1]), "confidence": float(confidence)}


def _region_score(anchors, bounds):
    left, right, low, high = bounds
    scores = []
    for along, across in anchors:
        dx = max(left - along, 0., along - right) / .2
        dy = max(low - across, 0., across - high) / 1.5
        scores.append(math.exp(-.5 * (dx * dx + dy * dy)))
    return float(np.dot(scores, [.15, .35, .5]))


def frame_features(landmarks, coordinates, confidence, state, image_size, config=RoleConfig()):
    features = {
        "coordinates": np.full((2, 21, 2), np.nan, np.float32),
        "anchors": np.full((2, 3, 2), np.nan, np.float32),
        "scores": np.zeros((2, 2), np.float32),
        "geometryConfidence": 0., "bodyConfidence": 0.,
        "available": np.zeros(2, bool),
        "imageCenters": np.full((2, 2), np.nan, np.float32),
        "imageAspect": np.asarray(image_size, dtype=np.float64) / max(image_size),
        "pairRoles": np.zeros(2, np.int8),
        "reason": np.full(2, REASON_CODES["geometry_unavailable"], np.int8),
    }
    present = np.isfinite(landmarks[..., :2]).all(-1).any(-1)
    features["reason"][~present] = REASON_CODES["hand_absent"]
    performance_states = (STATE_CODES["trackable"], STATE_CODES["guitar_partial"])
    screen_order = config.plucking_screen_side != "geometry"
    if state not in (*performance_states, STATE_CODES["calibration_unstable"], STATE_CODES["unknown"]):
        return features
    if screen_order:
        for hand in range(2):
            points = landmarks[hand, :, :2]
            valid = np.isfinite(points).all(-1) & ((points >= 0) & (points <= 1)).all(-1)
            if valid.sum() < 3:
                if present[hand]:
                    features["reason"][hand] = REASON_CODES["landmarks_unavailable"]
                continue
            features["imageCenters"][hand] = np.mean(points[valid], axis=0)
            features["available"][hand] = True
        if features["available"].all():
            if same_hand_landmarks(landmarks[0], landmarks[1], image_size):
                # Duplicate observations must not move both existing identities
                # onto one physical hand and corrupt the following single frame.
                features["available"][:] = False
                features["reason"][:] = REASON_CODES["competing_hands"]
                return features
            centers = features["imageCenters"][:, 0]
            if centers[0] != centers[1]:
                order = np.argsort(centers)
                features["pairRoles"][order] = (2, 1) if config.plucking_screen_side == "left" else (1, 2)
    if state not in performance_states:
        return features
    if not np.isfinite(coordinates[:4]).all() or float(min(confidence[:4])) < config.minimum_geometry_confidence:
        return features
    pixels = coordinates * np.asarray(image_size, dtype=np.float64)
    points = {name: _point(pixels[i], confidence[i]) for i, name in enumerate(GEOMETRY_POINTS[:4])}
    if np.linalg.norm(pixels[0] - pixels[1]) < config.minimum_neck_pixels:
        return features
    try:
        frame = GuitarCoordinateFrame.from_landmarks(points)
    except EvidenceError:
        return features
    if frame.fretboard_width < config.minimum_board_pixels:
        return features
    features["geometryConfidence"] = frame.confidence
    body = []
    body_confidence = []
    for index in (4, 5):
        if confidence[index] >= config.minimum_geometry_confidence and np.isfinite(pixels[index]).all():
            transformed = frame.transform(_point(pixels[index], confidence[index]))
            if transformed["alongNeck"] < -.05:
                body.append((transformed["alongNeck"], transformed["acrossFretboard"]))
                body_confidence.append(transformed["confidence"])
    # Without a visible reviewed body anchor, do not invent plucking evidence.
    if body:
        body = np.asarray(body)
        body_region = (float(body[:, 0].min()) - .35, min(-.08, float(body[:, 0].max()) + .35),
                       float(body[:, 1].min()) - 2., float(body[:, 1].max()) + 2.)
        features["bodyConfidence"] = min(body_confidence)
    for hand in range(2):
        if not present[hand]:
            continue
        available = np.isfinite(landmarks[hand, :, :2]).all(-1)
        available &= ((landmarks[hand, :, :2] >= 0) & (landmarks[hand, :, :2] <= 1)).all(-1)
        if not available[0] or sum(available[list(PALM)]) < 3 or sum(available[list(TIPS)]) < 2:
            features["reason"][hand] = REASON_CODES["landmarks_unavailable"]
            continue
        for index in np.flatnonzero(available):
            transformed = frame.transform(_point(landmarks[hand, index, :2] * image_size))
            features["coordinates"][hand, index] = transformed["alongNeck"], transformed["acrossFretboard"]
        values = features["coordinates"][hand]
        anchors = np.stack((
            values[0], np.median(values[list(PALM)][available[list(PALM)]], axis=0),
            np.median(values[list(TIPS)][available[list(TIPS)]], axis=0),
        ))
        features["anchors"][hand] = anchors
        if not screen_order:
            features["available"][hand] = True
        features["scores"][hand, 0] = _region_score(anchors, (-.08, 1.15, -1.2, 1.2))
        if len(body):
            features["scores"][hand, 1] = _region_score(anchors, body_region)
        features["reason"][hand] = REASON_CODES["outside_role_regions"]
    return features


class RoleTracker:
    def __init__(self, config=RoleConfig()):
        self.config = config
        self.shot = None
        self.tracks = {}
        self.next_id = 0
        self.swap_count = 0

    def update(self, features, seconds, shot):
        if shot != self.shot:
            self.tracks = {}
            self.shot = shot
        self.tracks = {key: value for key, value in self.tracks.items() if seconds - value["time"] <= self.config.maximum_gap_seconds}
        hands = list(np.flatnonzero(features["available"]))
        screen_order = self.config.plucking_screen_side != "geometry"
        anchors = {
            hand: features["imageCenters"][hand] * features["imageAspect"] if screen_order else np.median(features["anchors"][hand], axis=0)
            for hand in hands
        }
        possibilities = []
        for assignment in itertools.product([-1, *self.tracks], repeat=len(hands)):
            existing = [value for value in assignment if value != -1]
            if len(set(existing)) != len(existing):
                continue
            cost = 0.
            for hand, key in zip(hands, assignment):
                if key == -1:
                    cost += .5
                    continue
                track = self.tracks[key]
                pair_role = features["pairRoles"][hand]
                if pair_role and track["role"] and pair_role != track["role"]:
                    cost = math.inf
                    break
                delta = (anchors[hand] - track["anchor"]) * ([1., 1.] if screen_order else [1., .15])
                distance = float(np.linalg.norm(delta))
                if distance > .25 + 1.5 * (seconds - track["time"]):
                    cost = math.inf
                    break
                cost += distance
            possibilities.append((cost, assignment))
        assignment = min(possibilities, key=lambda item: (item[0], item[1]))[1]
        roles = np.zeros(2, np.int8)
        certainty = np.zeros(2, np.float32)
        track_ids = np.full(2, -1, np.int32)
        reason = features["reason"].copy()
        candidates = {}
        for hand in hands:
            if features["pairRoles"][hand]:
                candidates[hand] = int(features["pairRoles"][hand])
                continue
            if not np.isfinite(features["anchors"][hand]).all():
                continue
            scores = features["scores"][hand]
            best = int(np.argmax(scores))
            if scores[best] < self.config.minimum_role_score:
                reason[hand] = REASON_CODES[
                    "body_geometry_unavailable"
                    if not features["bodyConfidence"] and features["anchors"][hand, 1, 0] < 0
                    else "outside_role_regions"
                ]
            elif scores[best] - scores[1 - best] < self.config.ambiguity_margin:
                reason[hand] = REASON_CODES["ambiguous_role"]
            else:
                candidates[hand] = best + 1
        if len(candidates) == 2 and len(set(candidates.values())) == 1:
            role = next(iter(candidates.values()))
            ordered = sorted(candidates, key=lambda hand: (-features["scores"][hand, role - 1], hand))
            if features["scores"][ordered[0], role - 1] - features["scores"][ordered[1], role - 1] < self.config.ambiguity_margin:
                for hand in ordered:
                    reason[hand] = REASON_CODES["competing_hands"]
                candidates.clear()
            else:
                reason[ordered[1]] = REASON_CODES["competing_hands"]
                del candidates[ordered[1]]
        for hand, key in zip(hands, assignment):
            if key == -1:
                key = self.next_id
                self.next_id += 1
                self.tracks[key] = {"role": 0, "pending": 0, "since": seconds}
            track = self.tracks[key]
            track_ids[hand] = key
            track.update(anchor=anchors[hand], time=seconds)
            candidate = candidates.get(hand, 0)
            if features["pairRoles"][hand]:
                track["screen_role"] = candidate
            elif candidate and track.get("screen_role", candidate) != candidate:
                reason[hand] = REASON_CODES["ambiguous_role"]
                track["pending"] = 0
                continue
            if not candidate:
                track["pending"] = 0
                continue
            if candidate != track["role"]:
                if track["pending"] != candidate:
                    track.update(pending=candidate, since=seconds)
                if not features["pairRoles"][hand] and seconds - track["since"] + 1e-9 < self.config.confirmation_seconds:
                    reason[hand] = REASON_CODES["temporal_confirmation"]
                    continue
                if track["role"]:
                    self.swap_count += 1
                track.update(role=candidate, pending=0)
            else:
                track["pending"] = 0
            roles[hand] = candidate
            scores = features["scores"][hand]
            geometry_confidence = features["geometryConfidence"] if candidate == 1 else features["bodyConfidence"]
            certainty[hand] = 1. if features["pairRoles"][hand] else geometry_confidence * float(scores[candidate - 1]) * float(scores.max() - scores.min())
            reason[hand] = REASON_CODES["assigned"]
        return roles, certainty, track_ids, reason


def assign_role_arrays(hand, geo, shots, config=RoleConfig()):
    count = len(hand["pts"])
    output = {
        "pts": hand["pts"].copy(), "shot_id": hand["shot_id"].copy(),
        "role": np.zeros((count, 2), np.int8), "role_confidence": np.zeros((count, 2), np.float32),
        "assignment_source": np.zeros((count, 2), np.int8),
        "track_id": np.full((count, 2), -1, np.int32), "reason": np.zeros((count, 2), np.int8),
        "guitar_landmarks": np.full((count, 2, 21, 2), np.nan, np.float32),
        "anchors": np.full((count, 2, 3, 2), np.nan, np.float32),
        "role_scores": np.zeros((count, 2, 2), np.float32),
        "geometry_confidence": np.zeros(count, np.float32),
        "body_confidence": np.zeros(count, np.float32),
        "anatomical_handedness": hand["handedness"].copy(),
        "detection_source": hand["detection_source"].copy(),
    }
    tracker = RoleTracker(config)
    time_base = Fraction(*shots["timeBase"])
    for i, pts in enumerate(hand["pts"]):
        features = frame_features(hand["image_landmarks"][i], geo["coordinates"][i], geo["confidence"][i],
                                  int(geo["state"][i]), (shots["width"], shots["height"]), config)
        output["guitar_landmarks"][i] = features["coordinates"]
        output["anchors"][i] = features["anchors"]
        output["role_scores"][i] = features["scores"]
        output["geometry_confidence"][i] = features["geometryConfidence"]
        output["body_confidence"][i] = features["bodyConfidence"]
        role, confidence, tracks, reason = tracker.update(features, float(int(pts) * time_base), int(hand["shot_id"][i]))
        output["role"][i], output["role_confidence"][i], output["track_id"][i], output["reason"][i] = role, confidence, tracks, reason
        output["assignment_source"][i] = np.where(role > 0, np.where(features["pairRoles"] > 0, 2, 1), 0)
    output["role_available"] = output["role"] != 0
    return output, tracker.swap_count


def _summary(arrays, hand, geo):
    role = arrays["role"]
    any_role = (role > 0).any(-1)
    detected = hand["hand_count"] > 0
    visible = np.isin(geo["state"], [STATE_CODES["trackable"], STATE_CODES["guitar_partial"], STATE_CODES["calibration_unstable"]])
    one_hand = hand["hand_count"] == 1
    previous_roles = {}
    swaps = 0
    for roles, tracks in zip(arrays["role"], arrays["track_id"]):
        for value, track in zip(roles, tracks):
            if value and track >= 0:
                swaps += int(track in previous_roles and previous_roles[track] != value)
                previous_roles[track] = value
    return {
        "frames": len(role), "framesWithAnyRole": int(any_role.sum()),
        "framesWithFretting": int((role == 1).any(-1).sum()),
        "framesWithPlucking": int((role == 2).any(-1).sum()),
        "framesWithBothRoles": int(((role == 1).any(-1) & (role == 2).any(-1)).sum()),
        "anyRoleCoverage": float(any_role.mean()) if len(role) else None,
        "guitarVisibleFrames": int(visible.sum()),
        "visibleAnyRoleCoverage": float(any_role[visible].mean()) if visible.any() else None,
        "oneHandFrames": int(one_hand.sum()),
        "oneHandFramesWithRole": int((one_hand & any_role).sum()),
        "detectedHandFramesWithoutRole": int((detected & ~any_role).sum()),
        "unknownDetectedHandObservations": int((np.isfinite(hand["image_landmarks"][..., :2]).all(-1).any(-1) & (role == 0)).sum()),
        "screenOrderHandAssignments": int((arrays["assignment_source"] == 2).sum()),
        "withinShotRoleSwapCount": swaps,
        "trackIdentityCount": int(len(np.unique(arrays["track_id"][arrays["track_id"] >= 0]))),
    }


def assign_hand_roles(video_path, shots_path, hands_path, geometry_path, annotations_path, output_directory, config=RoleConfig()):
    if not isinstance(config, RoleConfig):
        raise EvidenceError("Role assignment configuration must be RoleConfig.")
    inputs = load_role_inputs(video_path, shots_path, hands_path, geometry_path, annotations_path)
    shots = inputs["documents"]["shots"]
    arrays, swaps = assign_role_arrays(inputs["hand"], inputs["geometry"], shots, config)
    rows = []
    for shot in shots["shots"]:
        selected = arrays["shot_id"] == shot["shotId"]
        rows.append({
            "shotId": shot["shotId"], "startPts": shot["startPts"], "endPtsExclusive": shot["endPtsExclusive"],
            **_summary({key: value[selected] for key, value in arrays.items()},
                       {key: value[selected] for key, value in inputs["hand"].items()},
                       {key: value[selected] for key, value in inputs["geometry"].items()}),
        })
    output = _safe_directory(output_directory, "hand-role observations")
    complete = False
    try:
        path = output / "roles.npz"
        np.savez_compressed(path, **arrays)
        if any(sha256(path) != inputs["hashes"][name] for name, path in inputs["paths"].items()):
            raise EvidenceError("An input changed during role assignment.")
        report = {
            "schemaVersion": 1, "kind": "guitar-relative-hand-roles", "visibility": "private",
            "inputSha256": inputs["hashes"], "timeBase": shots["timeBase"],
            "geometryPreparationMethod": inputs["documents"]["annotations"].get("preparationMethod", "reviewed-annotations"),
            "geometryReviewComplete": inputs["documents"]["annotations"].get("reviewComplete") is True,
            "frameCount": len(arrays["pts"]),
            "roleEncoding": ROLE_CODES, "reasonEncoding": REASON_CODES, "anchorOrder": list(ANCHOR_NAMES),
            "coordinateAxes": ["alongNeck", "acrossFretboard"],
            "coordinatePolicy": "Aspect-correct source pixels; neck/body at zero, nutward positive; no world-landmark or anatomical-role inference.",
            "roleScorePolicy": "Geometric assignments use uncalibrated proximity/separation; reviewed screen-order pairs score 1 conditional on that explicit orientation, not measured accuracy or contact probability. Geometry scores remain separate.",
            "screenOrderPolicy": "When two distinct hands are detected, order by mean X of all finite in-frame XY landmarks. Apply the explicit plucking-screen-side rule immediately, independently of guitar calibration. Single hands, duplicate detections and non-performance/transition frames are not forced into a pair.",
            "assignmentSourceEncoding": ASSIGNMENT_SOURCES,
            "anatomicalHandednessUsedForRole": False, "sameTakeConfirmed": False,
            "trackingResetCount": len(rows), "withinShotRoleSwapCount": swaps,
            "config": asdict(config), **_summary(arrays, inputs["hand"], inputs["geometry"]),
            "implementationSha256": {name: sha256(Path(__file__).with_name(name)) for name in ("hand_roles.py", "hand_tracking.py", "core.py", "geometry.py")},
            "runtime": {"numpy": np.__version__, "opencv": cv2.__version__, "av": av.__version__},
            "shotCount": len(rows), "shots": rows, "arrays": path.name, "arraysSha256": sha256(path),
            "legacyInputArrayHashes": "Upstream reports do not necessarily carry array hashes; exact observed array digests are bound here after schema/PTS validation.",
            "reviewRequired": True, "trainingPerformed": False, "transcriptionModified": False,
        }
        report_path = output / "roles.json"
        _publish(report_path, report)
        complete = True
        return report_path, report
    finally:
        if not complete:
            _cleanup(output)


def load_role_observations(inputs, roles_path):
    roles_path = Path(roles_path)
    report = json.loads(roles_path.read_text(encoding="utf-8"))
    array_path = roles_path.with_name("roles.npz")
    if report.get("kind") != "guitar-relative-hand-roles" or type(report.get("schemaVersion")) is not int or report["schemaVersion"] != 1 or report.get("inputSha256") != inputs["hashes"]:
        raise EvidenceError("Hand-role report is not bound to these observation inputs.")
    if report.get("arrays") != "roles.npz" or report.get("arraysSha256") != sha256(array_path):
        raise EvidenceError("Role arrays differ from their report.")
    count = len(inputs["hand"]["pts"])
    if report.get("roleEncoding") != ROLE_CODES or report.get("reasonEncoding") != REASON_CODES or report.get("frameCount") != count or report.get("timeBase") != inputs["documents"]["shots"]["timeBase"]:
        raise EvidenceError("Role encoding, frame count or time base differs from its inputs.")
    arrays = _read_arrays(array_path, {
        "pts": (np.dtype("int64"), (count,)), "shot_id": (np.dtype("int32"), (count,)),
        "role": (np.dtype("int8"), (count, 2)), "role_confidence": (np.dtype("float32"), (count, 2)),
        "track_id": (np.dtype("int32"), (count, 2)), "reason": (np.dtype("int8"), (count, 2)),
        "role_available": (np.dtype("bool"), (count, 2)), "guitar_landmarks": (np.dtype("float32"), (count, 2, 21, 2)),
        "anchors": (np.dtype("float32"), (count, 2, 3, 2)), "role_scores": (np.dtype("float32"), (count, 2, 2)),
        "geometry_confidence": (np.dtype("float32"), (count,)), "body_confidence": (np.dtype("float32"), (count,)),
        "anatomical_handedness": (np.dtype("int8"), (count, 2)), "detection_source": (np.dtype("int8"), (count,)),
    })
    if "assignmentSourceEncoding" in report:
        if report["assignmentSourceEncoding"] != ASSIGNMENT_SOURCES:
            raise EvidenceError("Unknown role assignment source encoding.")
        arrays.update(_read_arrays(array_path, {"assignment_source": (np.dtype("int8"), (count, 2))}))
        if not np.isin(arrays["assignment_source"], list(ASSIGNMENT_SOURCES.values())).all() or not np.array_equal(arrays["assignment_source"] > 0, arrays["role_available"]):
            raise EvidenceError("Role assignment sources disagree with availability.")
    else:
        arrays["assignment_source"] = arrays["role_available"].astype(np.int8)
    if not np.array_equal(arrays["pts"], inputs["hand"]["pts"]) or not np.array_equal(arrays["shot_id"], inputs["hand"]["shot_id"]):
        raise EvidenceError("Role review PTS/shot arrays disagree.")
    if not np.isin(arrays["role"], list(ROLE_CODES.values())).all() or not np.isin(arrays["reason"], list(REASON_CODES.values())).all():
        raise EvidenceError("Role review contains unknown role/reason codes.")
    if not np.array_equal(arrays["role_available"], arrays["role"] > 0) or (arrays["track_id"] < -1).any():
        raise EvidenceError("Role availability or track identity encoding is inconsistent.")
    for name in ("role_confidence", "role_scores", "geometry_confidence", "body_confidence"):
        values = arrays[name]
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise EvidenceError(f"Invalid {name} in role review arrays.")
    if np.any(arrays["role_confidence"][~arrays["role_available"]] != 0):
        raise EvidenceError("Unavailable roles cannot carry assignment confidence.")
    for code in (1, 2):
        if np.any((arrays["role"] == code).sum(-1) > 1):
            raise EvidenceError("A role is assigned to two hands in the same frame.")
    return report, arrays
