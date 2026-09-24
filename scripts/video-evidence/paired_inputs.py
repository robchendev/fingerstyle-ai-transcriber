"""Package observed guitar-relative and independent hand motion, without labels."""

from dataclasses import replace
from fractions import Fraction
import itertools
import json
from pathlib import Path
import sys

import numpy as np

from core import EvidenceError, GuitarCoordinateFrame, sha256
from geometry import GEOMETRY_POINTS, STATE_CODES, _cleanup, _publish, _safe_directory, automatic_cut_intervals, cut_uncertainty_mask
from hand_motion import load_audio_clock, validate_clips
from hand_roles import PALM, load_role_inputs, load_role_observations
from hand_tracking import same_hand_landmarks


_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from scripts.video_features import FEATURE_LAYOUT, INPUT_REPRESENTATION, MINIMUM_PALM_PIXELS, SCHEMA_VERSION, STRUCTURED_DIM, VIEW_ORDER
from scripts.fretboard_features import (
    FEATURE_LAYOUT as FRETBOARD_FEATURE_LAYOUT,
    INPUT_REPRESENTATION as FRETBOARD_INPUT_REPRESENTATION,
    SCHEMA_VERSION as FRETBOARD_SCHEMA_VERSION,
    STRUCTURED_DIM as FRETBOARD_STRUCTURED_DIM,
    FINGERTIP_LANDMARKS,
)
from scripts.fretboard_geometry import FretboardGeometry, FretboardGeometryError

MAXIMUM_GAP_SECONDS = .09


def _source_path(value):
    path = Path(value).absolute()
    if path.resolve() != path or not path.is_file() or path.stat().st_nlink != 1 or any(
        part.is_symlink() or getattr(part.stat(), "st_file_attributes", 0) & 0x400
        for part in (path, *path.parents)
    ):
        raise EvidenceError(f"Paired inputs require independent, unaliased source files: {path}")
    return path


def _valid_points(points):
    return np.isfinite(points).all(-1) & ((points >= 0) & (points <= 1)).all(-1)


def _point(xy, confidence=1.):
    return {"x": float(xy[0]), "y": float(xy[1]), "confidence": float(confidence)}


def _coordinate_frame(coordinates, confidence, state, size):
    valid = _valid_points(coordinates) & (confidence >= .5)
    if state not in (STATE_CODES["trackable"], STATE_CODES["guitar_partial"]) or not valid[:4].all():
        return None, valid
    pixels = coordinates * size
    try:
        frame = GuitarCoordinateFrame.from_landmarks({
            name: _point(pixels[i], confidence[i]) for i, name in enumerate(GEOMETRY_POINTS[:4])
        })
    except EvidenceError:
        return None, valid
    if frame.neck_length < 24 or frame.fretboard_width < 4:
        return None, valid
    # Edge midpoints can drift from the joint; keep joint=0 and nut=1 exactly.
    frame = replace(frame, origin_x=float(pixels[1, 0]), origin_y=float(pixels[1, 1]))
    return frame, valid


def _palm_observation(points, size):
    valid = _valid_points(points)
    mcps = np.asarray(PALM[1:])[valid[list(PALM[1:])]]
    if not valid[0] or len(mcps) < 3:
        return None
    pixels = points.astype(np.float64) * size
    scale = float(np.median(np.linalg.norm(pixels[mcps] - pixels[0], axis=-1)))
    if scale < MINIMUM_PALM_PIXELS:
        return None
    return {"pixels": pixels, "valid": valid, "scale": scale, "center": np.median(pixels[[0, *mcps]], axis=0)}


class _PalmTracker:
    """Conservative adjacent-frame source-pixel matching, never role inference."""

    def __init__(self):
        self.previous = {}
        self.next_id = 0

    def update(self, observations, boundary):
        if boundary:
            self.previous = {}
        slots = sorted(observations, key=lambda slot: (*observations[slot]["center"], slot))
        candidates = []
        for assignment in itertools.product([-1, *self.previous], repeat=len(slots)):
            assigned = [key for key in assignment if key >= 0]
            if len(assigned) != len(set(assigned)):
                continue
            cost = 0.
            for slot, key in zip(slots, assignment):
                if key < 0:
                    cost += 1.5
                    continue
                current, prior = observations[slot], self.previous[key]
                common = np.asarray(PALM)[current["valid"][list(PALM)] & prior["valid"][list(PALM)]]
                ratio = current["scale"] / prior["scale"]
                if len(common) < 3 or not .5 <= ratio <= 2:
                    cost = np.inf
                    break
                distance = np.median(np.linalg.norm(current["pixels"][common] - prior["pixels"][common], axis=-1))
                distance /= max(current["scale"], prior["scale"])
                if distance >= 1.5:
                    cost = np.inf
                    break
                cost += float(distance)
            candidates.append((cost, assignment))
        candidates.sort()
        best_cost, assignment = candidates[0]
        uncertain = {
            index for cost, alternate in candidates[1:] if cost - best_cost < .15
            for index, (left, right) in enumerate(zip(assignment, alternate)) if left != right
        }
        result = {}
        for index, (slot, key) in enumerate(zip(slots, assignment)):
            if key < 0 or index in uncertain:
                key = self.next_id
                self.next_id += 1
            result[slot] = key
        self.previous = {key: observations[slot] for slot, key in result.items()}
        return result


def _fretboard_observations(inputs, fretboard_path):
    report_path = _source_path(fretboard_path)
    arrays_path = _source_path(report_path.with_name("fretboard.npz"))
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise EvidenceError(f"Invalid fretboard report: {error}") from error
    if (
        not isinstance(report, dict)
        or report.get("schemaVersion") != 1
        or report.get("kind") != "six-point-fretboard-observations"
        or report.get("videoSha256") != inputs["hashes"]["video"]
        or report.get("shotsSha256") != inputs["hashes"]["shots"]
        or report.get("timeBase") != inputs["documents"]["shots"]["timeBase"]
        or report.get("arrays") != "fretboard.npz"
        or report.get("arraysSha256") != sha256(arrays_path)
        or report.get("sourceEncoding") != {"unavailable": 0, "detector": 1, "optical_flow": 2}
    ):
        raise EvidenceError("Fretboard observations do not match the source video, shots, arrays, or schema.")
    count = len(inputs["hand"]["pts"])
    if report.get("frameCount") != count or report.get("shotCount") != len(inputs["documents"]["shots"]["shots"]):
        raise EvidenceError("Fretboard observation counts differ from the source observations.")
    shapes = {
        "pts": (np.dtype("int64"), (count,)),
        "shot_id": (np.dtype("int32"), (count,)),
        "keypoints": (np.dtype("float32"), (count, 3, 2, 2)),
        "available": (np.dtype("bool"), (count, 3, 2)),
        "confidence": (np.dtype("float32"), (count,)),
        "source": (np.dtype("int8"), (count,)),
        "age_seconds": (np.dtype("float32"), (count,)),
        "flow_error": (np.dtype("float32"), (count,)),
        "detector_anchor": (np.dtype("bool"), (count,)),
    }
    try:
        with np.load(arrays_path, allow_pickle=False) as source:
            if set(source.files) != set(shapes):
                raise EvidenceError("Fretboard arrays must contain exactly the tracked observation contract.")
            arrays = {name: source[name].copy() for name in shapes}
    except (OSError, ValueError, KeyError) as error:
        raise EvidenceError(f"Malformed fretboard observation arrays: {error}") from error
    for name, (dtype, shape) in shapes.items():
        value = arrays[name]
        if value.dtype != dtype or value.shape != shape:
            raise EvidenceError(f"Invalid fretboard {name} dtype or shape.")
    if not np.array_equal(arrays["pts"], inputs["hand"]["pts"]) or not np.array_equal(arrays["shot_id"], inputs["hand"]["shot_id"]):
        raise EvidenceError("Fretboard and hand PTS/shot arrays must match exactly; no interpolation is permitted.")
    if (
        np.isinf(arrays["keypoints"]).any()
        or not np.isfinite(arrays["keypoints"][arrays["available"]]).all()
        or not np.isfinite(arrays["confidence"]).all()
        or not np.isfinite(arrays["age_seconds"]).all()
        or not np.isfinite(arrays["flow_error"]).all()
        or (arrays["confidence"] < 0).any()
        or (arrays["confidence"] > 1).any()
        or (arrays["age_seconds"] < 0).any()
        or (arrays["flow_error"] < 0).any()
        or not np.isin(arrays["source"], [0, 1, 2]).all()
        or np.any(arrays["detector_anchor"] != (arrays["source"] == 1))
        or np.any((arrays["source"] == 0) & arrays["available"].any(axis=(1, 2)))
    ):
        raise EvidenceError("Invalid fretboard availability, quality, source, or detector-anchor values.")
    return report, arrays, report_path, arrays_path


def _fretboard_geometry(arrays, index, size):
    try:
        return FretboardGeometry.from_keypoints(
            arrays["keypoints"][index].astype(np.float64) * size,
            arrays["available"][index],
            tuple(int(value) for value in size),
        )
    except FretboardGeometryError:
        return None


def structured_observations(inputs, roles, clips, fretboard=None):
    hand, geometry, shots = inputs["hand"], inputs["geometry"], inputs["documents"]["shots"]
    # Audio trims can end between video frames; the producer also enforces the exact audio-clock bounds.
    selected, clip_ids = validate_clips(clips, hand["pts"], shots, allow_partial_end=True)
    indices = np.flatnonzero(selected)
    count = len(indices)
    size = np.asarray([shots["width"], shots["height"]], np.float64)
    time_base = Fraction(*shots["timeBase"])
    dimension = FRETBOARD_STRUCTURED_DIM if fretboard is not None else STRUCTURED_DIM
    output = {
        "pts": hand["pts"][indices].copy(),
        "structured": np.zeros((count, len(VIEW_ORDER), dimension), np.float32),
        "structured_available": np.zeros((count, len(VIEW_ORDER), dimension), bool),
        "segment_id": np.full((count, len(VIEW_ORDER)), -1, np.int64),
    }
    previous = [None] * len(VIEW_ORDER)
    tracker, anonymous_views = _PalmTracker(), {}
    blocked = cut_uncertainty_mask(hand["pts"], automatic_cut_intervals(shots))
    blocked |= np.isin(geometry["state"], [STATE_CODES["guitar_absent"], STATE_CODES["transition"]])
    next_segment = 0
    for local, i in enumerate(indices):
        if blocked[i]:
            previous = [None] * len(VIEW_ORDER)
            tracker.previous = {}
            anonymous_views = {}
            continue
        frame, anchor_valid = _coordinate_frame(
            geometry["coordinates"][i], geometry["confidence"][i], geometry["state"][i], size,
        )
        fretboard_geometry = _fretboard_geometry(fretboard, i, size) if fretboard is not None else None
        earlier = int(indices[local - 1]) if local else -1
        boundary = earlier != i - 1 or earlier < 0
        if not boundary:
            elapsed = float((int(hand["pts"][i]) - int(hand["pts"][earlier])) * time_base)
            boundary = (
                not 0 < elapsed <= MAXIMUM_GAP_SECONDS or blocked[earlier]
                or clip_ids[i] != clip_ids[earlier] or hand["shot_id"][i] != hand["shot_id"][earlier]
                or hand["detection_source"][i] != hand["detection_source"][earlier]
            )
        observations = {}
        for slot in range(2):
            palm = _palm_observation(hand["image_landmarks"][i, slot, :, :2], size)
            if palm is not None:
                observations[slot] = palm
        slots = list(range(2))
        if len(observations) == 2 and same_hand_landmarks(
            hand["image_landmarks"][i, 0], hand["image_landmarks"][i, 1], size,
        ):
            keep = min(slots, key=lambda slot: (roles["role"][i, slot] == 0, -observations[slot]["valid"].sum(), slot))
            slots = [keep]
            observations = {keep: observations[keep]}
            boundary = True
        identities = tracker.update(observations, boundary)
        anonymous_views = {
            identities[slot]: anonymous_views[identities[slot]] for slot in observations
            if roles["role"][i, slot] == 0 and identities[slot] in anonymous_views and not boundary
        }
        for slot in sorted(observations, key=lambda slot: (*observations[slot]["center"], slot)):
            if roles["role"][i, slot] == 0 and identities[slot] not in anonymous_views:
                anonymous_views[identities[slot]] = next(view for view in (2, 3) if view not in anonymous_views.values())
        assigned = set()
        for slot in slots:
            code = int(roles["role"][i, slot])
            if code not in (0, 1, 2) or (code and roles["track_id"][i, slot] < 0):
                raise EvidenceError("Paired inputs require unique roles and valid assigned hand tracks.")
            if not code and slot not in observations:
                continue
            role = code - 1 if code else anonymous_views[identities[slot]]
            if role in assigned:
                raise EvidenceError("Paired inputs require unique roles and valid hand tracks.")
            points = hand["image_landmarks"][i, slot, :, :2]
            valid = _valid_points(points)
            if not valid.any():
                continue
            assigned.add(role)
            values = output["structured"][local, role]
            masks = output["structured_available"][local, role]
            if code and frame is not None:
                for point in np.flatnonzero(valid):
                    transformed = frame.transform(_point(points[point] * size))
                    values[point * 2:point * 2 + 2] = transformed["alongNeck"], transformed["acrossFretboard"]
                    masks[point * 2:point * 2 + 2] = True
                for anchor in np.flatnonzero(anchor_valid):
                    transformed = frame.transform(_point(geometry["coordinates"][i, anchor] * size))
                    values[84 + anchor * 2:86 + anchor * 2] = transformed["alongNeck"], transformed["acrossFretboard"]
                    masks[84 + anchor * 2:86 + anchor * 2] = True
                values[96:98] = frame.neck_length / np.linalg.norm(size), frame.fretboard_width / np.linalg.norm(size)
                masks[96:98] = True
            palm = observations.get(slot)
            if palm is not None:
                local_points = (palm["pixels"][valid] - palm["pixels"][0]) / palm["scale"]
                values[98:140].reshape(21, 2)[valid] = local_points
                masks[98:140].reshape(21, 2)[valid] = True
                if valid[9]:
                    direction = palm["pixels"][9] - palm["pixels"][0]
                    norm = np.linalg.norm(direction)
                    if norm >= MINIMUM_PALM_PIXELS:
                        values[184:186] = direction / norm
                        masks[184:186] = True
            fretboard_positions = None
            fretboard_source = None
            if fretboard_geometry is not None:
                fretboard_source = (int(fretboard["shot_id"][i]), int(fretboard["source"][i]))
                values[229:233] = (
                    fretboard["confidence"][i], fretboard["age_seconds"][i],
                    fretboard["flow_error"][i], fretboard["detector_anchor"][i],
                )
                masks[229:233] = True
                fretboard_positions = np.zeros((len(FINGERTIP_LANDMARKS), 2), np.float32)
                fretboard_position_mask = np.zeros((len(FINGERTIP_LANDMARKS), 2), bool)
                for tip_index, point in enumerate(FINGERTIP_LANDMARKS):
                    if not valid[point]:
                        continue
                    try:
                        coordinate = fretboard_geometry.coordinate(points[point].astype(np.float64) * size)
                    except FretboardGeometryError:
                        continue
                    offset = 194 + tip_index * 2
                    values[offset:offset + 2] = coordinate.scale_position, coordinate.string_position
                    masks[offset:offset + 2] = True
                    fretboard_positions[tip_index] = coordinate.scale_position, coordinate.string_position
                    fretboard_position_mask[tip_index] = True
                    if coordinate.fret_position is not None:
                        values[204 + tip_index] = coordinate.fret_position
                        masks[204 + tip_index] = True
                    values[209 + tip_index] = coordinate.nearest_string_distance
                    masks[209 + tip_index] = True
                    values[214 + tip_index] = coordinate.inside_string_span
                    masks[214 + tip_index] = True
            key = (
                int(clip_ids[i]), int(hand["shot_id"][i]), int(roles["track_id"][i, slot]) if code else -1,
                int(hand["detection_source"][i]), identities.get(slot), frame is not None if code else False, tuple(valid),
            )
            prior = previous[role]
            continuous = not boundary and prior is not None and prior[0] == i - 1 and prior[2] == key
            elapsed = float((int(hand["pts"][i]) - int(hand["pts"][prior[0]])) * time_base) if continuous else 0.
            continuous = continuous and 0 < elapsed <= MAXIMUM_GAP_SECONDS
            if continuous:
                before = prior[1]
                output["segment_id"][local, role] = output["segment_id"][before, role]
                velocity_mask = masks[:42] & output["structured_available"][before, role, :42]
                values[42:84][velocity_mask] = (
                    values[:42][velocity_mask] - output["structured"][before, role, :42][velocity_mask]
                ) / elapsed
                masks[42:84] = velocity_mask
                local_mask = masks[98:140] & output["structured_available"][before, role, 98:140]
                values[140:182][local_mask] = (
                    values[98:140][local_mask] - output["structured"][before, role, 98:140][local_mask]
                ) / elapsed
                masks[140:182] = local_mask
                if palm is not None and prior[3] is not None:
                    values[182:184] = (palm["pixels"][0] - prior[3]["pixels"][0]) / prior[3]["scale"] / elapsed
                    masks[182:184] = True
                if (
                    fretboard_positions is not None
                    and prior[4] is not None
                    and prior[5] == fretboard_source
                    and not bool(fretboard["detector_anchor"][i])
                ):
                    velocity_mask = fretboard_position_mask & prior[4][1]
                    velocities = (fretboard_positions - prior[4][0]) / elapsed
                    values[219:229].reshape(-1, 2)[velocity_mask] = velocities[velocity_mask]
                    masks[219:229].reshape(-1, 2)[velocity_mask] = True
            else:
                output["segment_id"][local, role] = next_segment
                next_segment += 1
            previous[role] = (
                i, local, key, palm,
                None if fretboard_positions is None else (fretboard_positions, fretboard_position_mask),
                fretboard_source,
            )
    usable = output["structured_available"].any(-1)
    output["segment_id"][~usable] = -1
    next_segment = 0
    for role in range(len(VIEW_ORDER)):
        previous = None
        for i in np.flatnonzero(usable[:, role]):
            original = int(output["segment_id"][i, role])
            if previous is None or i != previous[0] + 1 or original != previous[1]:
                segment = next_segment
                next_segment += 1
            output["segment_id"][i, role] = segment
            previous = i, original
    return output, indices


def _prepare_paired_inputs(video_path, shots_path, hands_path, geometry_path, annotations_path,
                           roles_path, audio_path, alignment_path, output_directory, clips, *,
                           pair_id=None, fretboard_path=None):
    if pair_id is not None and (not isinstance(pair_id, str) or not pair_id.strip()):
        raise EvidenceError("Pair ID must be a nonempty string.")
    for path in (video_path, shots_path, hands_path, geometry_path, annotations_path, roles_path, audio_path, alignment_path):
        _source_path(path)
    inputs = load_role_inputs(video_path, shots_path, hands_path, geometry_path, annotations_path)
    role_report, roles = load_role_observations(inputs, roles_path)
    paths = {**inputs["paths"], "roles": Path(roles_path).absolute(), "roleArrays": Path(roles_path).absolute().with_name("roles.npz"),
             "trimmedAudio": Path(audio_path).absolute(), "alignment": Path(alignment_path).absolute()}
    clock, duration, alignment, audio_hashes = load_audio_clock(inputs["hashes"]["video"], audio_path, alignment_path)
    hashes = {**inputs["hashes"], **audio_hashes, "roles": sha256(roles_path), "roleArrays": role_report["arraysSha256"]}
    fretboard = None
    if fretboard_path is not None:
        fretboard_report, fretboard, report_path, array_path = _fretboard_observations(inputs, fretboard_path)
        paths.update(fretboard=report_path, fretboardArrays=array_path)
        hashes.update(fretboard=sha256(report_path), fretboardArrays=fretboard_report["arraysSha256"])
    for path in paths.values():
        _source_path(path)
    if len(set(paths.values())) != len(paths):
        raise EvidenceError("Paired-input source paths must not alias one another.")
    arrays, indices = (
        structured_observations(inputs, roles, clips, fretboard=fretboard)
        if fretboard is not None
        else structured_observations(inputs, roles, clips)
    )
    time_base = Fraction(*inputs["documents"]["shots"]["timeBase"])
    for clip in clips:
        if clock.map_pts(clip["startPts"], time_base) < 0 or Fraction(clock.map_pts(clip["endPtsExclusive"], time_base), clock.sample_rate) > duration:
            raise EvidenceError("Paired clip falls outside the aligned trimmed audio.")
    arrays["audio_seconds"] = np.asarray([clock.map_pts(int(pts), time_base) / clock.sample_rate for pts in arrays["pts"]], np.float64)
    arrays["technique_available"] = np.ones(len(indices), bool)
    output = _safe_directory(output_directory, "paired video inputs")
    complete = False
    try:
        if any(sha256(_source_path(path)) != hashes[name] for name, path in paths.items()):
            raise EvidenceError("A bound source changed during paired-input preparation.")
        np.savez_compressed(output / "inputs.npz", **arrays)
        schema_version = FRETBOARD_SCHEMA_VERSION if fretboard is not None else SCHEMA_VERSION
        representation = FRETBOARD_INPUT_REPRESENTATION if fretboard is not None else INPUT_REPRESENTATION
        dimension = FRETBOARD_STRUCTURED_DIM if fretboard is not None else STRUCTURED_DIM
        layout = FRETBOARD_FEATURE_LAYOUT if fretboard is not None else FEATURE_LAYOUT
        report = {
            "kind": "paired-video-inputs", "schemaVersion": schema_version, "visibility": "private",
            "inputRepresentation": representation,
            **({"id": pair_id} if pair_id is not None else {}),
            "audioSha256": hashes["trimmedAudio"], "videoSha256": hashes["video"],
            "timeBase": inputs["documents"]["shots"]["timeBase"],
            "featureDimension": dimension,
            "arraysPath": "inputs.npz", "arraysSha256": sha256(output / "inputs.npz"),
            "inputSha256": hashes, "inputPaths": {name: str(path) for name, path in paths.items()},
            "frameCount": len(indices), "viewOrder": VIEW_ORDER, "featureLayout": layout,
            "coordinatePolicy": "Source-pixel aspect correction; neckBody origin=(0,0), nut=(1,0); perpendicular fretboard-width cross axis.",
            "handCoordinatePolicy": "Observed wrist-centered source-pixel XY; median wrist-to-valid-MCP(5,9,13,17) distance, at least three MCPs and eight source pixels. Wrist velocity uses prior scale and includes camera motion, not guitar contact.",
            "handIdentityPolicy": "Adjacent source-pixel palm matching; ambiguous identity, missing hand, clip, shot, cut, detector change or >90ms gap resets motion. Unknown playing roles remain anonymous; anatomical handedness is not used.",
            "coarseCoordinatePolicy": "Reserved schema slots 186:194 remain zero and unavailable.",
            **({
                "fretboardCoordinatePolicy": "Six-point projective scale and continuous string coordinates; unavailable or invalid geometry is zero and masked.",
                "fretboardMotionPolicy": "Backward velocity requires the same hand, role, shot, geometry source, and available predecessor within 90ms; detector anchors reset motion.",
                "fretboardEvidencePolicy": "Geometry and board-region evidence only; no contact or pressure is inferred.",
            } if fretboard is not None else {}),
            "coverage": {
                "framesWithGeometry": int(arrays["structured_available"][..., :98].any(axis=(1, 2)).sum()),
                "framesWithIndependentHand": int(arrays["structured_available"][..., 98:140].any(axis=(1, 2)).sum()),
                "geometryHandObservations": int(arrays["structured_available"][..., :98].any(-1).sum()),
                "independentHandObservations": int(arrays["structured_available"][..., 98:140].any(-1).sum()),
                "unassignedHandObservations": int(arrays["structured_available"][:, 2:, 98:140].any(-1).sum()),
                "coarseHandObservations": int(arrays["structured_available"][:, :2, 186:188].all(-1).sum()),
            },
            "clock": {"sampleRate": clock.sample_rate, "offsetSamples": clock.offset_samples, "rate": [1, 1]},
            "clips": [{"startPts": row["startPts"], "endPtsExclusive": row["endPtsExclusive"]} for row in sorted(clips, key=lambda value: value["startPts"])],
            "correspondenceIntervals": [],
            "maximumGapSeconds": MAXIMUM_GAP_SECONDS,
        }
        _publish(output / "inputs.json", report)
        complete = True
        return output / "inputs.json", report
    finally:
        if not complete:
            _cleanup(output)


def prepare_paired_inputs(video_path, shots_path, hands_path, geometry_path, annotations_path,
                          roles_path, audio_path, alignment_path, output_directory, clips, *,
                          pair_id=None):
    return _prepare_paired_inputs(
        video_path, shots_path, hands_path, geometry_path, annotations_path,
        roles_path, audio_path, alignment_path, output_directory, clips, pair_id=pair_id,
    )


def prepare_paired_inputs_v5(video_path, shots_path, hands_path, geometry_path, annotations_path,
                             roles_path, fretboard_path, audio_path, alignment_path,
                             output_directory, clips, *, pair_id=None):
    return _prepare_paired_inputs(
        video_path, shots_path, hands_path, geometry_path, annotations_path,
        roles_path, audio_path, alignment_path, output_directory, clips,
        pair_id=pair_id, fretboard_path=fretboard_path,
    )
