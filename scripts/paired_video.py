"""Strict private paired-input adapter for the existing release and audio loader."""

import argparse
from fractions import Fraction
import json
from pathlib import Path, PureWindowsPath
import re
import stat

import numpy as np
import soundfile
import torch

from .dataset_io import ROOT, sha256
from .dataset_release import _regular_file, validate_release
from .transcriber_audio import HarnessError
from .video_features import FEATURE_LAYOUT, INPUT_REPRESENTATION, SCHEMA_VERSION, STRUCTURED_DIM, VELOCITY_SLICES, VIEW_ORDER
from .fretboard_features import (
    FEATURE_LAYOUT as FRETBOARD_FEATURE_LAYOUT,
    INPUT_REPRESENTATION as FRETBOARD_INPUT_REPRESENTATION,
    SCHEMA_VERSION as FRETBOARD_SCHEMA_VERSION,
    STRUCTURED_DIM as FRETBOARD_STRUCTURED_DIM,
)


VIDEO_FIELDS = frozenset({"structured", "structured_available", "technique_available", "segment_id", "frame_indices"})
MAXIMUM_FRAME_DISTANCE = .045
_ARRAY_NAMES = {"audio_seconds", "pts", "technique_available", "segment_id", "structured", "structured_available"}
_INPUT_NAMES = {"video", "shots", "hands", "geometry", "annotations", "handArrays", "geometryArrays", "roles", "roleArrays", "trimmedAudio", "alignment"}
_FRETBOARD_INPUT_NAMES = _INPUT_NAMES | {"fretboard", "fretboardArrays"}
_VELOCITY_OBSERVATIONS = ((0, 42), (98, 140), (98, 100), (186, 188))
_CONTRACTS = {
    SCHEMA_VERSION: {
        "schemaVersion": SCHEMA_VERSION, "inputRepresentation": INPUT_REPRESENTATION,
        "featureDimension": STRUCTURED_DIM, "featureLayout": FEATURE_LAYOUT,
        "inputNames": _INPUT_NAMES, "velocitySlices": VELOCITY_SLICES,
        "velocityObservations": _VELOCITY_OBSERVATIONS,
    },
    FRETBOARD_SCHEMA_VERSION: {
        "schemaVersion": FRETBOARD_SCHEMA_VERSION, "inputRepresentation": FRETBOARD_INPUT_REPRESENTATION,
        "featureDimension": FRETBOARD_STRUCTURED_DIM, "featureLayout": FRETBOARD_FEATURE_LAYOUT,
        "inputNames": _FRETBOARD_INPUT_NAMES, "velocitySlices": (*VELOCITY_SLICES, (219, 229)),
        "velocityObservations": (*_VELOCITY_OBSERVATIONS, (194, 204)),
    },
}


def _digest(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise HarnessError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _file(path):
    try:
        return _regular_file(path)
    except (ValueError, OSError) as error:
        raise HarnessError(f"Invalid paired-video asset: {error}") from error


def _read(path):
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise HarnessError(f"Invalid paired-video JSON {path}: {error}") from error
    if not isinstance(document, dict):
        raise HarnessError("Paired-video JSON must be an object.")
    return document


def _relative(root, value, *, filename=False):
    if not isinstance(value, str):
        raise HarnessError("Paired-video paths must be relative strings.")
    parts = PureWindowsPath(value)
    if parts.drive or parts.root or not parts.parts or ".." in parts.parts or ":" in value or (filename and len(parts.parts) != 1):
        raise HarnessError("Paired-video relative paths must not escape their declared root.")
    return _file(Path(root).joinpath(*parts.parts))


def _read_index(path):
    document = _read(path)
    if document.get("kind") != "paired-video-index" or document.get("pathBasis") != "root":
        raise HarnessError("Numeric paired-video indexes require a supported schema and root-relative paths.")
    _contract(document, "index")
    _digest(document.get("manifestSha256"), "Release identity")
    return document


def _contract(document, label):
    if type(document.get("schemaVersion")) is not int or document["schemaVersion"] not in _CONTRACTS:
        raise HarnessError(f"Numeric paired-video {label} requires schemaVersion 4 or 5; regenerate schema 3/2/RGB data in a new directory.")
    contract = _CONTRACTS[document["schemaVersion"]]
    if {"images", "imageSize", "image_size", "available"} & document.keys():
        raise HarnessError(f"Paired-video {label} cannot contain obsolete RGB/image fields.")
    if (
        document.get("inputRepresentation") != contract["inputRepresentation"]
        or document.get("featureDimension") != contract["featureDimension"]
    ):
        raise HarnessError(f"Paired-video {label} schema, representation, and feature dimension must identify one supported numeric contract.")
    return contract


def index_manifest(index_path, *, root=ROOT, manifest_path=None, default_manifest=None):
    """Resolve the release by its bound hash, never by an unchecked default."""
    root = Path(root).resolve()
    path = Path(index_path)
    path = _file(path if path.is_absolute() else root / path)
    if not path.is_relative_to(root):
        raise HarnessError("Paired-video index must stay inside its private root.")
    document = _read_index(path)
    if manifest_path is not None:
        candidate = Path(manifest_path)
        candidate = _file(candidate if candidate.is_absolute() else root / candidate)
    elif "manifestPath" in document:
        candidate = _relative(root, document["manifestPath"])
    elif default_manifest is not None:
        candidate = Path(default_manifest)
        candidate = candidate if candidate.is_absolute() else root / candidate
        if not candidate.is_file():
            raise HarnessError("This index has no release path; supply --manifest for its hash-bound release.")
        candidate = _file(candidate)
    else:
        raise HarnessError("This index has no release path; supply --manifest for its hash-bound release.")
    if not candidate.is_relative_to(root):
        raise HarnessError("The release manifest must stay inside the private root.")
    if sha256(candidate) != document["manifestSha256"]:
        raise HarnessError("The selected release does not match the paired index manifest hash. Supply --manifest for the index's exact release; no default release was substituted.")
    return candidate


def _stat(path):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1 or getattr(value, "st_file_attributes", 0) & 0x400:
        raise HarnessError(f"Expected an independent regular paired-video file without aliases: {path}")
    return value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_ino, value.st_nlink


def _guard_parents(paths):
    return tuple(sorted({parent for path in paths for parent in path.parents}, key=lambda path: len(path.parts)))


def _check_guards(guards, parents):
    try:
        # Paths were canonicalized on load; lstat catches later aliases without
        # resolving every ancestor again for every file in every window.
        for parent in parents:
            value = parent.lstat()
            if not stat.S_ISDIR(value.st_mode) or getattr(value, "st_file_attributes", 0) & 0x400:
                raise HarnessError(f"A paired-video directory changed or became an alias: {parent}")
        for path, expected in guards.items():
            if _stat(path) != expected:
                raise HarnessError(f"A bound paired-video file changed during use: {path}")
    except OSError as error:
        raise HarnessError(f"A bound paired-video path changed or became unavailable: {error}") from error


def _times(values):
    times = np.asarray(values, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or (times < 0).any() or (np.diff(times) <= 0).any():
        raise HarnessError("Video windows require increasing finite absolute audio times.")
    return times


def _clock_seconds(pts, time_base, clock):
    sample = int(pts) * time_base * clock["sampleRate"] + clock["offsetSamples"]
    magnitude = abs(sample)
    rounded = (2 * magnitude.numerator + magnitude.denominator) // (2 * magnitude.denominator)
    return (rounded if sample >= 0 else -rounded) / clock["sampleRate"]


def empty_video(length, feature_dimension=STRUCTURED_DIM):
    if feature_dimension not in (STRUCTURED_DIM, FRETBOARD_STRUCTURED_DIM):
        raise HarnessError("Empty paired video requires a supported feature dimension.")
    return {
        "technique_available": torch.zeros(1, dtype=torch.bool),
        "segment_id": torch.full((1, len(VIEW_ORDER)), -1, dtype=torch.long),
        "structured": torch.zeros((1, len(VIEW_ORDER), feature_dimension), dtype=torch.float32),
        "structured_available": torch.zeros((1, len(VIEW_ORDER), feature_dimension), dtype=torch.bool),
        "frame_indices": torch.full((length,), -1, dtype=torch.long),
    }


def _validate_features(values, masks, segments, contract):
    count = len(values)
    dimension = contract["featureDimension"]
    for name, value, dtype, shape in (
        ("structured", values, np.float32, (count, len(VIEW_ORDER), dimension)),
        ("structured_available", masks, np.bool_, (count, len(VIEW_ORDER), dimension)),
        ("segment_id", segments, np.int64, (count, len(VIEW_ORDER))),
    ):
        if value.dtype != dtype or value.shape != shape:
            raise HarnessError(f"Invalid paired-video {name} dtype or shape for schema {contract['schemaVersion']}.")
    if not count or not np.isfinite(values).all():
        raise HarnessError("Paired-video features must be nonempty and finite.")
    if np.any(values[~masks] != 0):
        raise HarnessError("Unavailable paired-video inputs must be explicitly zero.")
    if not np.array_equal(segments >= 0, masks.any(-1)) or (segments < -1).any():
        raise HarnessError("Paired-video segment availability is inconsistent.")
    for start, stop in ((0, 42), (42, 84), (84, 96), (98, 140), (140, 182), (182, 184), (184, 186), (186, 188), (188, 190), (190, 192)):
        paired = masks[..., start:stop].reshape(count, len(VIEW_ORDER), -1, 2)
        if np.any(paired[..., 0] != paired[..., 1]):
            raise HarnessError("Paired-video coordinate availability must mask both XY axes together.")
    for start, expected in ((84, [1., 0.]), (86, [0., 0.]), (98, [0., 0.])):
        if not np.allclose(values[..., start:start + 2][masks[..., start]], expected, rtol=0, atol=1e-5):
            label = "local wrist zero" if start == 98 else "neck/body joint zero and nut one"
            raise HarnessError(f"Paired-video coordinates must use {label}.")
    orientation = values[..., 184:186][masks[..., 184]]
    if not np.allclose(np.linalg.norm(orientation, axis=-1), 1., rtol=0, atol=1e-5):
        raise HarnessError("Available paired-video palm orientation must be a unit vector.")
    if masks[:, 2:, 186:].any():
        if contract["schemaVersion"] == SCHEMA_VERSION:
            raise HarnessError("Coarse instrument context is available only for known playing roles, not anonymous views.")
        if masks[:, 2:, 186:194].any():
            raise HarnessError("Coarse instrument context is available only for known playing roles, not anonymous views.")
    axis = values[..., 190:192][masks[..., 190]]
    if not np.allclose(np.linalg.norm(axis, axis=-1), 1., rtol=0, atol=1e-5):
        raise HarnessError("Available paired-video coarse axis must be a unit vector.")
    sign = values[..., 193][masks[..., 193]]
    if not np.isin(sign, (-1., 1.)).all():
        raise HarnessError("Available paired-video coarse axis sign must be +1 or -1; unknown signs must be masked.")
    local = masks[..., 98:140].any(-1) | masks[..., 184:186].any(-1)
    if np.any(local & ~masks[..., 98:100].all(-1)):
        raise HarnessError("Independent hand observations require an available local wrist.")
    for (start, stop), (position_start, position_stop) in zip(
        contract["velocitySlices"], contract["velocityObservations"], strict=True,
    ):
        velocity = masks[..., start:stop]
        positions = masks[..., position_start:position_stop]
        predecessor = positions[1:] & positions[:-1] & (segments[1:] == segments[:-1])[..., None]
        if velocity[0].any() or np.any(velocity[1:] & ~predecessor):
            raise HarnessError("Paired-video motion cannot bridge unavailable points or segments; segment starts cannot have motion.")
    if contract["schemaVersion"] == FRETBOARD_SCHEMA_VERSION:
        for start, stop in ((194, 204), (219, 229)):
            paired = masks[..., start:stop].reshape(count, len(VIEW_ORDER), -1, 2)
            if np.any(paired[..., 0] != paired[..., 1]):
                raise HarnessError("Fretboard coordinate availability must mask both axes together.")
        confidence = values[..., 229][masks[..., 229]]
        age = values[..., 230][masks[..., 230]]
        flow_error = values[..., 231][masks[..., 231]]
        detector_anchor = values[..., 232][masks[..., 232]]
        inside = values[..., 214:219][masks[..., 214:219]]
        positions = values[..., 194:204].reshape(count, len(VIEW_ORDER), 5, 2)
        position_masks = masks[..., 194:204].reshape(count, len(VIEW_ORDER), 5, 2)[..., 0]
        scale = positions[..., 0]
        string = positions[..., 1]
        fret_masks = masks[..., 204:209]
        nearest = values[..., 209:214]
        nearest_masks = masks[..., 209:214]
        inside_masks = masks[..., 214:219]
        quality_masks = masks[..., 229:233]
        if (
            ((confidence < 0) | (confidence > 1)).any()
            or (age < 0).any()
            or (flow_error < 0).any()
            or not np.isin(detector_anchor, (0., 1.)).all()
            or not np.isin(inside, (0., 1.)).all()
            or (nearest[nearest_masks] < 0).any()
        ):
            raise HarnessError("Invalid fretboard geometry quality or board-span feature values.")
        expected_fret = position_masks & (scale >= 0) & (scale < 1)
        expected_inside = ((string >= 0) & (string <= 5)).astype(np.float32)
        nearest_string = np.clip(np.round(string), 0, 5)
        if (
            np.any(fret_masks != expected_fret)
            or np.any(nearest_masks != position_masks)
            or np.any(inside_masks != position_masks)
            or np.any(values[..., 214:219][position_masks] != expected_inside[position_masks])
            or not np.allclose(nearest[position_masks], np.abs(string - nearest_string)[position_masks], rtol=0, atol=1e-5)
            or np.any(quality_masks != quality_masks[..., :1])
            or np.any(position_masks & ~quality_masks[..., :1])
        ):
            raise HarnessError("Fretboard derived feature masks or values disagree with projected coordinates.")
    for segment in np.unique(segments[segments >= 0]):
        rows, views = np.where(segments == segment)
        if len(set(views)) != 1 or (np.diff(rows) != 1).any():
            raise HarnessError("Paired-video segments cannot bridge views or missing frames.")


def validate_video_tensors(video, length):
    """Validate one unbatched model input before padding, without coercion."""
    if set(video) != VIDEO_FIELDS:
        raise HarnessError("Paired batches require numeric video fields; RGB inputs are unsupported.")
    if any(not isinstance(value, torch.Tensor) for value in video.values()):
        raise HarnessError("Paired video inputs must be tensors.")
    count = len(video["structured"])
    dimension = video["structured"].shape[-1] if video["structured"].ndim == 3 else None
    contract = next((value for value in _CONTRACTS.values() if value["featureDimension"] == dimension), None)
    if contract is None:
        raise HarnessError("Paired-video tensors require the schema-4 D194 or schema-5 D233 feature dimension.")
    for name, dtype, shape in (
        ("structured", torch.float32, (count, len(VIEW_ORDER), dimension)),
        ("structured_available", torch.bool, (count, len(VIEW_ORDER), dimension)),
        ("segment_id", torch.long, (count, len(VIEW_ORDER))),
        ("technique_available", torch.bool, (count,)),
        ("frame_indices", torch.long, (length,)),
    ):
        if video[name].dtype != dtype or tuple(video[name].shape) != shape:
            raise HarnessError(f"Invalid paired-video {name} dtype or shape for D{dimension}.")
    _validate_features(
        *(video[name].detach().cpu().numpy() for name in ("structured", "structured_available", "segment_id")),
        contract,
    )
    indices = video["frame_indices"]
    if torch.any((indices < -1) | (indices >= count)):
        raise HarnessError("Paired-video frame indices are outside the retained observations.")
    selected = indices[indices >= 0]
    if len(selected) and not video["structured_available"][selected].flatten(1).any(-1).all():
        raise HarnessError("Paired-video frame indices cannot reference unavailable frames.")


class _Bundle:
    def __init__(self, report_path, audio_sha256):
        self.path = _file(report_path)
        report_hash = sha256(self.path)
        self.report = report = _read(self.path)
        if report.get("kind") != "paired-video-inputs":
            raise HarnessError("Paired-video input report kind differs.")
        self.contract = contract = _contract(report, "inputs")
        self.schema_version = contract["schemaVersion"]
        self.feature_dimension = contract["featureDimension"]
        if report.get("audioSha256") != _digest(audio_sha256, "Expected audio identity"):
            raise HarnessError("Paired-video audio hash does not match the recording.")
        _digest(report.get("videoSha256"), "Video identity")
        if report.get("viewOrder") != VIEW_ORDER or report.get("featureLayout") != contract["featureLayout"]:
            raise HarnessError("Paired-video structured feature layout or view order differs.")
        if report.get("maximumGapSeconds") != .09:
            raise HarnessError("Paired-video continuity must use the supported 90ms maximum gap.")
        count = report.get("frameCount")
        if type(count) is not int or count <= 0:
            raise HarnessError("Paired-video frameCount must be positive.")
        time_base = report.get("timeBase")
        if not isinstance(time_base, list) or len(time_base) != 2 or any(type(value) is not int or value <= 0 for value in time_base):
            raise HarnessError("Paired-video timeBase must be a positive rational pair.")
        self.time_base = Fraction(*time_base)
        arrays_path = _relative(self.path.parent, report.get("arraysPath"), filename=True)
        bindings = {self.path: report_hash, arrays_path: _digest(report.get("arraysSha256"), "Array identity")}
        hashes, paths = report.get("inputSha256"), report.get("inputPaths")
        if not isinstance(hashes, dict) or set(hashes) != contract["inputNames"] or not isinstance(paths, dict) or set(paths) != set(hashes):
            raise HarnessError("Paired-video source paths and input hashes must be complete.")
        for name, value in paths.items():
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise HarnessError("Paired-video inputPaths must be explicit absolute source paths.")
            path = _file(value)
            if path in bindings:
                raise HarnessError("Paired-video sources must not alias other bound assets.")
            bindings[path] = _digest(hashes[name], f"Input {name} identity")
        if hashes["video"] != report["videoSha256"] or hashes["trimmedAudio"] != audio_sha256:
            raise HarnessError("Paired-video source and report identities disagree.")
        before = {path: _stat(path) for path in bindings}
        if any(sha256(path) != expected for path, expected in bindings.items()):
            raise HarnessError("A paired-video asset hash does not match its report.")
        try:
            with np.load(arrays_path, allow_pickle=False) as archive:
                if {"images", "available"} & set(archive.files):
                    raise HarnessError("RGB/image arrays are unsupported; regenerate structure-only paired inputs.")
                if set(archive.files) != _ARRAY_NAMES or len(archive.files) != len(_ARRAY_NAMES):
                    raise HarnessError("Paired-video arrays must contain exactly the input feature contract.")
                arrays = {name: archive[name] for name in _ARRAY_NAMES}
        except (ValueError, KeyError, OSError) as error:
            raise HarnessError(f"Malformed paired-video arrays: {error}") from error
        for name, dtype, shape in (
            ("audio_seconds", np.float64, (count,)), ("pts", np.int64, (count,)),
            ("technique_available", np.bool_, (count,)), ("segment_id", np.int64, (count, len(VIEW_ORDER))),
            ("structured", np.float32, (count, len(VIEW_ORDER), self.feature_dimension)),
            ("structured_available", np.bool_, (count, len(VIEW_ORDER), self.feature_dimension)),
        ):
            value = arrays[name]
            if value.dtype != dtype or value.shape != shape or (value.dtype.kind == "f" and not np.isfinite(value).all()):
                raise HarnessError(f"Invalid paired-video {name} dtype, shape or finite values.")
        self.arrays = arrays
        self.usable = arrays["structured_available"].any(-1)
        self.times = arrays["audio_seconds"]
        _times(self.times)
        if (np.diff(arrays["pts"]) <= 0).any():
            raise HarnessError("Paired-video PTS must increase strictly.")
        _validate_features(arrays["structured"], arrays["structured_available"], arrays["segment_id"], contract)
        if arrays["structured_available"][..., 186:194].any():
            raise HarnessError("Reserved coarse-context feature slots must remain masked.")
        segments = arrays["segment_id"]
        ids = np.unique(segments[segments >= 0])
        if not np.array_equal(ids, np.arange(len(ids))):
            raise HarnessError("Paired-video segment IDs must be unique contiguous integers.")
        for segment in ids:
            rows, roles = np.where(segments == segment)
            if len(set(roles)) != 1 or (np.diff(rows) != 1).any() or (np.diff(self.times[rows]) > .09 + 1e-9).any():
                raise HarnessError("Paired-video segments cannot bridge roles or frame gaps.")
        self._validate_clock(paths)
        if any(_stat(path) != before[path] for path in bindings):
            raise HarnessError("A paired-video asset changed while loading.")
        self._guards = before
        self._parents = _guard_parents(before)
        self.check_unchanged()
        self.identity = {
            "kind": "paired-video-inputs", "schemaVersion": self.schema_version,
            "inputRepresentation": contract["inputRepresentation"],
            "featureDimension": self.feature_dimension, "bundleSha256": report_hash,
            "arraysSha256": report["arraysSha256"], "audioSha256": audio_sha256,
            "videoSha256": report["videoSha256"], "inputSha256": dict(hashes),
        }

    def _validate_clock(self, paths):
        report, arrays = self.report, self.arrays
        clock = report.get("clock")
        if not isinstance(clock, dict) or set(clock) != {"sampleRate", "offsetSamples", "rate"} or type(clock["sampleRate"]) is not int or clock["sampleRate"] <= 0 or type(clock["offsetSamples"]) is not int or clock["rate"] != [1, 1] or any(type(value) is not int for value in clock["rate"]):
            raise HarnessError("Paired-video clock must be an explicit unit-rate sample clock.")
        audio = soundfile.info(paths["trimmedAudio"])
        if audio.samplerate != clock["sampleRate"] or self.times[-1] >= audio.frames / audio.samplerate:
            raise HarnessError("Paired-video clock or frame extent differs from the actual trimmed audio.")
        alignment = _read(Path(paths["alignment"]))
        if alignment.get("kind") != "video-to-trimmed-audio-alignment" or type(alignment.get("schemaVersion")) is not int or alignment["schemaVersion"] != 1 or alignment.get("status") != "supported" or alignment.get("videoSha256") != report["videoSha256"] or alignment.get("trimmedAudioSha256") != report["audioSha256"] or alignment.get("rate") != [1, 1] or any(type(value) is not int for value in alignment["rate"]):
            raise HarnessError("Paired-video alignment does not bind the expected source clock.")
        offset = alignment.get("videoStartSecondsForTrimmedAudioZero")
        if type(offset) not in (float, int) or not np.isfinite(offset) or round(-Fraction(str(offset)) * clock["sampleRate"]) != clock["offsetSamples"]:
            raise HarnessError("Paired-video clock differs from the source alignment.")
        expected = [_clock_seconds(pts, self.time_base, clock) for pts in arrays["pts"]]
        if not np.allclose(expected, self.times, rtol=0, atol=1e-10):
            raise HarnessError("Paired-video audio times disagree with exact source PTS.")
        clips = report.get("clips")
        if not isinstance(clips, list) or not clips:
            raise HarnessError("Paired-video inputs require bounded source clips.")
        selected = np.zeros(len(self.times), bool)
        self.clip_intervals = []
        previous_end = None
        for clip in clips:
            if not isinstance(clip, dict) or set(clip) != {"startPts", "endPtsExclusive"}:
                raise HarnessError("Invalid paired-video clip fields.")
            start, end = clip["startPts"], clip["endPtsExclusive"]
            if type(start) is not int or type(end) is not int or start >= end or (previous_end is not None and start < previous_end):
                raise HarnessError("Paired-video clips must be ordered, nonoverlapping source intervals.")
            mask = (arrays["pts"] >= start) & (arrays["pts"] < end)
            if not mask.any() or arrays["pts"][mask][0] != start:
                raise HarnessError("Paired-video clip start must be retained source PTS.")
            selected |= mask
            previous_end = end
            self.clip_intervals.append((_clock_seconds(start, self.time_base, clock), _clock_seconds(end, self.time_base, clock)))
            if self.clip_intervals[-1][0] < 0 or self.clip_intervals[-1][1] > audio.frames / audio.samplerate:
                raise HarnessError("Paired-video clip extends outside the actual trimmed audio.")
            first = int(np.flatnonzero(mask)[0])
            if any(arrays["structured_available"][first, :, start:stop].any() for start, stop in self.contract["velocitySlices"]):
                raise HarnessError("Paired-video clip starts cannot carry pre-clip motion.")
            if first and np.any((arrays["segment_id"][first] >= 0) & (arrays["segment_id"][first] == arrays["segment_id"][first - 1])):
                raise HarnessError("Paired-video segments cannot cross a selected clip boundary.")
        if not selected.all():
            raise HarnessError("Paired-video arrays contain frames outside the selected clips.")
        with np.load(paths["handArrays"], allow_pickle=False) as source:
            source_pts = source["pts"]
            native = np.zeros(len(source_pts), bool)
            for clip in clips:
                native |= (source_pts >= clip["startPts"]) & (source_pts < clip["endPtsExclusive"])
            if not np.array_equal(source_pts[native], arrays["pts"]):
                raise HarnessError("Paired-video input must retain exact native observation cadence.")
        if self.schema_version == FRETBOARD_SCHEMA_VERSION:
            self._validate_fretboard_source(paths)
        intervals = report.get("correspondenceIntervals")
        if intervals != []:
            raise HarnessError("External score-video correspondence overrides are not supported.")
        expected_technique = np.ones(len(self.times), bool)
        if not np.array_equal(expected_technique, arrays["technique_available"]):
            raise HarnessError("Paired-video technique mask must preserve the local hand observations.")

    def _validate_fretboard_source(self, paths):
        report = _read(Path(paths["fretboard"]))
        array_path = Path(paths["fretboardArrays"])
        count = report.get("frameCount")
        if (
            report.get("kind") != "six-point-fretboard-observations"
            or report.get("schemaVersion") != 1
            or report.get("videoSha256") != self.report["videoSha256"]
            or report.get("shotsSha256") != self.report["inputSha256"]["shots"]
            or report.get("timeBase") != self.report["timeBase"]
            or report.get("arrays") != array_path.name
            or report.get("arraysSha256") != self.report["inputSha256"]["fretboardArrays"]
            or report.get("sourceEncoding") != {"unavailable": 0, "detector": 1, "optical_flow": 2}
            or array_path.parent != Path(paths["fretboard"]).parent
            or type(count) is not int
            or count <= 0
        ):
            raise HarnessError("Schema-5 paired inputs require their source-bound fretboard report and arrays.")
        names = {
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
            with np.load(array_path, allow_pickle=False) as source:
                if set(source.files) != set(names):
                    raise HarnessError("Source fretboard arrays differ from the tracked observation contract.")
                fretboard = {name: source[name] for name in names}
        except (OSError, ValueError, KeyError) as error:
            raise HarnessError(f"Malformed source fretboard arrays: {error}") from error
        for name, (dtype, shape) in names.items():
            if fretboard[name].dtype != dtype or fretboard[name].shape != shape:
                raise HarnessError(f"Invalid source fretboard {name} dtype or shape.")
        if (
            np.isinf(fretboard["keypoints"]).any()
            or not np.isfinite(fretboard["keypoints"][fretboard["available"]]).all()
            or not np.isfinite(fretboard["confidence"]).all()
            or not np.isfinite(fretboard["age_seconds"]).all()
            or not np.isfinite(fretboard["flow_error"]).all()
            or (fretboard["confidence"] < 0).any()
            or (fretboard["confidence"] > 1).any()
            or (fretboard["age_seconds"] < 0).any()
            or (fretboard["flow_error"] < 0).any()
            or np.any((fretboard["source"] == 0) & fretboard["available"].any(axis=(1, 2)))
        ):
            raise HarnessError("Invalid source fretboard geometry availability or quality values.")
        native = np.searchsorted(fretboard["pts"], self.arrays["pts"])
        if (
            (native >= len(fretboard["pts"])).any()
            or not np.array_equal(fretboard["pts"][native], self.arrays["pts"])
            or not np.isin(fretboard["source"], [0, 1, 2]).all()
            or np.any(fretboard["detector_anchor"] != (fretboard["source"] == 1))
        ):
            raise HarnessError("Paired inputs do not preserve the bound fretboard observation cadence or source encoding.")
        quality = np.stack((
            fretboard["confidence"][native], fretboard["age_seconds"][native],
            fretboard["flow_error"][native], fretboard["detector_anchor"][native],
        ), axis=-1).astype(np.float32)
        quality_values = self.arrays["structured"][..., 229:233]
        quality_masks = self.arrays["structured_available"][..., 229:233]
        if np.any(quality_masks & (quality_values != quality[:, None, :])):
            raise HarnessError("Paired-input geometry quality differs from its bound fretboard source.")
        velocity = self.arrays["structured_available"][..., 219:229].any(axis=(1, 2))
        for row in np.flatnonzero(velocity):
            current, previous = native[row], native[row - 1] if row else -1
            if (
                row == 0
                or current != previous + 1
                or fretboard["shot_id"][current] != fretboard["shot_id"][previous]
                or fretboard["source"][current] != fretboard["source"][previous]
                or fretboard["detector_anchor"][current]
            ):
                raise HarnessError("Fretboard motion cannot bridge geometry source changes or detector anchors.")

    def check_unchanged(self):
        _check_guards(self._guards, self._parents)

    def _frame_indices(self, times):
        right = np.searchsorted(self.times, times).clip(0, len(self.times) - 1)
        left = (right - 1).clip(0)
        indices = np.where(abs(self.times[left] - times) <= abs(self.times[right] - times), left, right)
        valid = abs(self.times[indices] - times) <= MAXIMUM_FRAME_DISTANCE
        valid &= self.usable[indices].any(-1)
        in_clip = np.zeros(len(times), bool)
        for start, end in self.clip_intervals:
            in_clip |= (times >= start) & (times < end)
        valid &= in_clip
        interior = (self.times[left] < times - 1e-10) & (times < self.times[right] - 1e-10)
        continuous = np.all(self.arrays["segment_id"][left] == self.arrays["segment_id"][right], axis=-1)
        valid &= ~interior | continuous
        return np.where(valid, indices, -1)

    def has_usable(self, start, stop):
        self.check_unchanged()
        return bool(((self.times >= start - MAXIMUM_FRAME_DISTANCE) & (self.times < stop + MAXIMUM_FRAME_DISTANCE) & self.usable.any(-1)).any())

    def availability_at(self, clip_times):
        """Return structured role coverage and independent technique correspondence."""
        self.check_unchanged()
        times = _times(clip_times)
        indices = self._frame_indices(times)
        valid = indices >= 0
        available = np.zeros((len(times), len(VIEW_ORDER)), bool)
        techniques = np.zeros(len(times), bool)
        available[valid] = self.usable[indices[valid]]
        techniques[valid] = self.arrays["technique_available"][indices[valid]]
        return available, techniques

    def feature_availability_at(self, clip_times):
        """Separate calibrated geometry, independent hands and coarse context."""
        self.check_unchanged()
        indices = self._frame_indices(_times(clip_times))
        valid = indices >= 0
        result = []
        for start, stop in ((0, 98), (98, 140), (186, 194)):
            available = np.zeros((len(indices), len(VIEW_ORDER)), bool)
            available[valid] = self.arrays["structured_available"][indices[valid], :, start:stop].any(-1)
            result.append(available)
        return tuple(result)

    def window(self, clip_times):
        self.check_unchanged()
        times = _times(clip_times)
        if not len(times):
            return empty_video(0, self.feature_dimension)
        mapping = self._frame_indices(times)
        if not (mapping >= 0).any():
            return empty_video(len(times), self.feature_dimension)
        indices = np.flatnonzero((self.times >= times[0] - MAXIMUM_FRAME_DISTANCE) & (self.times <= times[-1] + MAXIMUM_FRAME_DISTANCE))
        result = {name: torch.from_numpy(self.arrays[name][indices].copy()) for name in ("technique_available", "segment_id", "structured", "structured_available")}
        for start, stop in self.contract["velocitySlices"]:
            result["structured"][0, :, start:stop] = 0
            result["structured_available"][0, :, start:stop] = False
        result["frame_indices"] = torch.from_numpy(np.where(mapping >= 0, mapping - indices[0], -1)).long()
        return result


def load_inference_video(bundle_report_path, audio_sha256):
    return _Bundle(bundle_report_path, audio_sha256)


class PairedVideoIndex:
    """All relative bundle paths are resolved against the explicit private root."""

    def __init__(self, index_path, manifest_sha256, records, *, root=ROOT):
        self.root = Path(root).resolve()
        self.path = _file(index_path)
        if not self.path.is_relative_to(self.root):
            raise HarnessError("Paired-video index must stay inside its private root.")
        self._guard = _stat(self.path)
        index_hash = sha256(self.path)
        document = _read_index(self.path)
        self.contract = contract = _contract(document, "index")
        self.schema_version = contract["schemaVersion"]
        self.feature_dimension = contract["featureDimension"]
        if document.get("manifestSha256") != _digest(manifest_sha256, "Release identity"):
            raise HarnessError("Paired-video index belongs to a different release manifest.")
        self.records = {}
        groups = {}
        for entry, _ in records:
            identifier, group, split = entry["id"], entry["groupId"], entry["split"]
            if identifier in self.records or split not in ("train", "validation") or (group in groups and groups[group] != split):
                raise HarnessError("Paired-video index requires original grouped full-release membership.")
            self.records[identifier] = entry
            groups[group] = split
        rows = document.get("records")
        if not isinstance(rows, list) or not rows:
            raise HarnessError("Paired-video index requires at least one bound recording.")
        self.bundles = {}
        self._guards = {self.path: self._guard}
        identities = []
        seen_paths = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"id", "split", "groupId", "audioSha256", "bundlePath", "bundleSha256"}:
                raise HarnessError("Invalid paired-video index recording fields.")
            identifier = row["id"]
            if not isinstance(identifier, str) or identifier not in self.records or identifier in self.bundles:
                raise HarnessError("Paired-video index has an unknown or duplicate recording ID.")
            record = self.records[identifier]
            if any(row[key] != record[key] for key in ("split", "groupId", "audioSha256")):
                raise HarnessError("Paired-video recording split, group or audio hash differs from the release.")
            path = _relative(self.root, row["bundlePath"])
            if path in seen_paths:
                raise HarnessError("Paired-video records must not alias bundle paths.")
            seen_paths.add(path)
            if sha256(path) != _digest(row["bundleSha256"], "Bundle identity"):
                raise HarnessError("Paired-video bundle hash differs from its index.")
            bundle = _Bundle(path, row["audioSha256"])
            if bundle.schema_version != self.schema_version or bundle.feature_dimension != self.feature_dimension:
                raise HarnessError("A paired-video index may contain only one schema and feature dimension.")
            if bundle.report.get("id") != identifier:
                raise HarnessError("Paired-video bundle ID differs from its index.")
            if bundle.times[-1] >= record["sampleCount"] / record["sampleRate"]:
                raise HarnessError("Paired-video frame extends beyond its release audio.")
            self.bundles[identifier] = bundle
            for asset, guard in bundle._guards.items():
                if asset in self._guards and self._guards[asset] != guard:
                    raise HarnessError(f"A shared paired-video asset changed while loading: {asset}")
                self._guards[asset] = guard
            identities.append({key: row[key] for key in ("id", "split", "groupId", "audioSha256", "bundleSha256")})
        self.identity = {
            "kind": "paired-video-index", "schemaVersion": self.schema_version, "indexSha256": index_hash,
            "manifestSha256": manifest_sha256, "inputRepresentation": contract["inputRepresentation"],
            "featureDimension": self.feature_dimension,
            "records": sorted(identities, key=lambda row: row["id"]),
        }
        self._parents = _guard_parents(self._guards)
        self.check_unchanged()

    def check_unchanged(self):
        _check_guards(self._guards, self._parents)

    def _record(self, identifier):
        if identifier not in self.records:
            raise HarnessError(f"Unknown paired-video recording ID: {identifier}")
        return self.bundles.get(identifier)

    def window(self, identifier, clip_times):
        bundle = self._record(identifier)
        self.check_unchanged()
        return bundle.window(clip_times) if bundle is not None else empty_video(
            len(_times(clip_times)), self.feature_dimension,
        )

    def has_usable(self, identifier, start, stop):
        bundle = self._record(identifier)
        return bundle is not None and bundle.has_usable(start, stop)

    def availability_at(self, identifier, clip_times):
        self.check_unchanged()
        bundle = self._record(identifier)
        if bundle is None:
            return np.zeros((len(_times(clip_times)), len(VIEW_ORDER)), bool), np.zeros(len(clip_times), bool)
        return bundle.availability_at(clip_times)


def build_index(manifest_path, bundle_paths, output_path, *, root=ROOT):
    root = Path(root).resolve()
    manifest_path = _file(manifest_path)
    if not manifest_path.is_relative_to(root):
        raise HarnessError("The release manifest must stay inside the private root.")
    _, records, bindings = validate_release(manifest_path)
    rows = {row["id"]: row for row, _ in records}
    result, seen = [], set()
    index_contract = None
    for value in bundle_paths:
        path = _file(value)
        if not path.is_relative_to(root):
            raise HarnessError("Bundle paths must stay inside the private root.")
        report = _read(path)
        identifier = report.get("id")
        if not isinstance(identifier, str) or identifier not in rows or identifier in seen:
            raise HarnessError("Index bundles require unique IDs from the original full release.")
        seen.add(identifier)
        row = rows[identifier]
        bundle = _Bundle(path, row["audioSha256"])
        if index_contract is None:
            index_contract = bundle.contract
        elif (
            bundle.schema_version != index_contract["schemaVersion"]
            or bundle.feature_dimension != index_contract["featureDimension"]
        ):
            raise HarnessError("A paired-video index may contain only one schema and feature dimension.")
        result.append({**{key: row[key] for key in ("id", "split", "groupId", "audioSha256")},
                       "bundlePath": str(path.relative_to(root)), "bundleSha256": bundle.identity["bundleSha256"]})
    if not result:
        raise HarnessError("Provide at least one paired input bundle.")
    output = Path(output_path).absolute()
    if output.resolve() != output or not output.is_relative_to(root) or output.exists():
        raise HarnessError("Index output must be a new unaliased file inside the private root.")
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "kind": "paired-video-index", "schemaVersion": index_contract["schemaVersion"],
        "pathBasis": "root", "visibility": "private",
        "manifestSha256": bindings[manifest_path],
        "inputRepresentation": index_contract["inputRepresentation"],
        "featureDimension": index_contract["featureDimension"],
        "manifestPath": str(manifest_path.relative_to(root)),
        "records": sorted(result, key=lambda row: row["id"]),
    }
    created = False
    try:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            created = True
            stream.write(json.dumps(document, indent=2, allow_nan=False) + "\n")
        PairedVideoIndex(output, bindings[manifest_path], records, root=root)
        if any(sha256(path) != digest for path, digest in bindings.items()):
            raise HarnessError("A release input changed during paired-index construction.")
    except (ValueError, OSError):
        if created:
            output.unlink(missing_ok=True)
        raise
    return output, document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index")
    index.add_argument("--manifest", required=True)
    index.add_argument("--bundle", action="append", required=True)
    index.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        path, report = build_index(args.manifest, args.bundle, args.output)
    except (ValueError, OSError) as error:
        parser.exit(2, f"paired-video: {error}\n")
    print(json.dumps({"index": str(path), "records": len(report["records"]), "inputRepresentation": report["inputRepresentation"], "featureDimension": report["featureDimension"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
