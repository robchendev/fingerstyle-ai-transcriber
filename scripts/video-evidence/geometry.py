"""Geometry schema validation and scene-cut masks for numerical hand inputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from uuid import uuid4

import numpy as np

from core import EvidenceError, GuitarCoordinateFrame, private_output


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
AUTOMATIC_PREPARATION_METHOD = "automatic-geometry-v1"


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


def geometry_row_complete(row):
    if not isinstance(row, dict):
        return False
    geometry = row.get("geometry")
    if row.get("state") in ("trackable", "guitar_partial"):
        if not isinstance(geometry, dict) or not geometry or not set(geometry) <= set(GEOMETRY_POINTS):
            return False
        if row["state"] == "trackable" and set(geometry) != set(GEOMETRY_POINTS):
            return False
        try:
            for name, point in geometry.items():
                _point(point, name)
            if set(REQUIRED_FRAME_POINTS) <= geometry.keys():
                GuitarCoordinateFrame.from_landmarks(geometry)
        except EvidenceError:
            return False
        return True
    return row.get("state") in ("guitar_absent", "transition", "calibration_unstable") and geometry is None


def annotations_ready(document):
    """Readiness is not a claim of manual review or usable coordinate coverage."""
    if not isinstance(document, dict):
        return False
    if document.get("reviewComplete") is True:
        return True
    if (
        document.get("preparationComplete") is not True
        or document.get("preparationMethod") != AUTOMATIC_PREPARATION_METHOD
        or document.get("reviewComplete") is not False
        or any(not isinstance(document.get(name), str) or not document[name] for name in ("videoSha256", "shotsSha256"))
    ):
        return False
    rows = document.get("shots")
    if not isinstance(rows, list) or not rows or not all(geometry_row_complete(row) for row in rows):
        return False
    ids = [row.get("shotId") for row in rows]
    if any(type(value) is not int or value < 0 for value in ids) or len(set(ids)) != len(ids):
        return False
    if any(any(type(row.get(name)) is not int for name in ("startPts", "endPtsExclusive", "keyframePts"))
           or not isinstance(row.get("keyframeImage"), str) or not row["keyframeImage"] for row in rows):
        return False
    if any(right["startPts"] < left["endPtsExclusive"] or right["shotId"] <= left["shotId"]
           for left, right in zip(rows, rows[1:])):
        return False
    try:
        validate_geometry_annotations(document, {"shots": rows}, document.get("videoSha256"), document.get("shotsSha256"))
    except (EvidenceError, KeyError, TypeError):
        return False
    summary = document.get("proposalSummary")
    if not isinstance(summary, dict):
        return False
    accepted = summary.get("autoAcceptedShotIds")
    unavailable = summary.get("unavailableShotIds")
    review = summary.get("reviewRequiredShotIds")
    if any(not isinstance(values, list) or any(type(value) is not int for value in values) or len(set(values)) != len(values) for values in (accepted, unavailable, review)):
        return False
    reviewed = document.get("reviewedShotIds", [])
    if not isinstance(reviewed, list) or any(type(value) is not int for value in reviewed):
        return False
    visible_ids = {row["shotId"] for row in rows if row["state"] in ("trackable", "guitar_partial")}
    masked_ids = set(ids) - visible_ids
    return (
        set(accepted) == visible_ids - set(reviewed)
        and set(unavailable) == masked_ids
        and set(review) <= masked_ids - set(reviewed)
        and set(reviewed) <= set(ids)
        and all(row["state"] != "unknown" for row in rows)
    )


def automatic_cut_intervals(shots):
    review = shots.get("automaticReview")
    if review is None:
        return []
    if not isinstance(review, dict) or review.get("method") != "conservative-cut-boundaries-v1":
        raise EvidenceError("Unsupported automatic cut uncertainty method.")
    intervals = review.get("uncertainIntervals")
    if not isinstance(intervals, list):
        raise EvidenceError("Automatic cut uncertainty requires an interval list.")
    for interval in intervals:
        if (
            not isinstance(interval, dict)
            or any(type(interval.get(name)) is not int for name in ("startPts", "endPtsExclusive"))
            or interval["startPts"] >= interval["endPtsExclusive"]
            or not isinstance(interval.get("reason"), str) or not interval["reason"]
        ):
            raise EvidenceError("Automatic cut uncertainty requires integer source-PTS bounds and a reason.")
    return intervals


def cut_uncertainty_mask(pts, intervals):
    values = np.asarray(pts, dtype=np.int64)
    masked = np.zeros(values.shape, dtype=bool)
    for interval in intervals:
        masked |= (values >= interval["startPts"]) & (values < interval["endPtsExclusive"])
    return masked


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
        required_fields = {
            "shotId",
            "startPts",
            "endPtsExclusive",
            "keyframePts",
            "keyframeImage",
            "state",
            "geometry",
            "reviewNote",
        }
        if not isinstance(row, dict) or not required_fields <= row.keys() or row.keys() - required_fields - {"additionalKeyframes"}:
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
        if state in ("guitar_absent", "transition", "unknown") or (
            state == "calibration_unstable" and geometry is None
        ):
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
        additional = row.get("additionalKeyframes", [])
        if not isinstance(additional, list):
            raise EvidenceError("Additional geometry keyframes must be a list.")
        normalized_seeds = []
        seed_pts = {row["keyframePts"]}
        for seed in additional:
            if not isinstance(seed, dict) or set(seed) != {"keyframePts", "keyframeImage", "geometry", "reviewNote"}:
                raise EvidenceError("Additional geometry keyframe has an invalid schema.")
            pts = seed["keyframePts"]
            if type(pts) is not int or pts in seed_pts or not row["startPts"] <= pts < row["endPtsExclusive"]:
                raise EvidenceError("Additional geometry keyframes require unique in-shot PTS.")
            if state not in ("trackable", "guitar_partial") or not isinstance(seed["geometry"], dict) or not seed["geometry"] or not set(seed["geometry"]) <= set(GEOMETRY_POINTS):
                raise EvidenceError("Additional keyframes require explicit guitar landmarks in a guitar-visible shot.")
            if set(REQUIRED_FRAME_POINTS) <= seed["geometry"].keys():
                GuitarCoordinateFrame.from_landmarks(seed["geometry"])
            seed_pts.add(pts)
            normalized_seeds.append({**seed, "geometry": {name: _point(point, f"shot {row['shotId']} additional {name}") for name, point in seed["geometry"].items()}})
        normalized.append({
            **row,
            "geometry": points,
            "additionalKeyframes": normalized_seeds,
        })
    return normalized
