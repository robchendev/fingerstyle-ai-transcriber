"""Six-point projective fretboard geometry."""

from dataclasses import dataclass
import math

import numpy as np


ANCHOR_NAMES = ("nut", "fret12", "bridge")
STRING_NAMES = ("string6", "string1")
KEYPOINT_NAMES = tuple(
    f"{anchor}_{string_name}"
    for anchor in ANCHOR_NAMES
    for string_name in STRING_NAMES
)
CANONICAL_ANCHORS = np.asarray((0., .5, 1.), dtype=np.float64)
STRING_COUNT = 6


class FretboardGeometryError(ValueError):
    pass


def _points(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise FretboardGeometryError(f"{name} must have shape {shape}.")
    return result


def _homography(source, target):
    rows = []
    for (x, y), (u, v) in zip(source, target, strict=True):
        rows.extend((
            [x, y, 1, 0, 0, 0, -u * x, -u * y, -u],
            [0, 0, 0, x, y, 1, -v * x, -v * y, -v],
        ))
    _, _, right = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    matrix = right[-1].reshape(3, 3)
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise FretboardGeometryError("Fretboard anchors produce a degenerate transform.")
    return matrix / matrix[2, 2]


def _transform(matrix, points):
    values = np.asarray(points, dtype=np.float64)
    flat = values.reshape(-1, 2)
    homogeneous = np.column_stack((flat, np.ones(len(flat)))) @ matrix.T
    if (np.abs(homogeneous[:, 2]) < 1e-12).any():
        raise FretboardGeometryError("Fretboard transform maps a point to infinity.")
    return (homogeneous[:, :2] / homogeneous[:, 2, None]).reshape(values.shape)


def fret_scale_position(fret):
    if isinstance(fret, bool) or not isinstance(fret, (int, float)) or not math.isfinite(fret) or fret < 0:
        raise FretboardGeometryError("Fret must be finite and nonnegative.")
    return 1 - 2 ** (-float(fret) / 12)


def continuous_fret(scale_position):
    if not math.isfinite(scale_position) or not 0 <= scale_position < 1:
        return None
    return -12 * math.log2(1 - scale_position)


@dataclass(frozen=True)
class FretboardCoordinate:
    scale_position: float
    string_position: float
    fret_position: float | None
    nearest_string: int
    nearest_string_distance: float
    inside_string_span: bool


@dataclass(frozen=True)
class FretboardGeometry:
    image_to_canonical: np.ndarray
    canonical_to_image: np.ndarray
    available_anchors: tuple[str, ...]
    reprojection_error: float
    image_size: tuple[int, int]

    @classmethod
    def from_keypoints(cls, keypoints, available, image_size):
        points = _points(keypoints, (3, 2, 2), "fretboard keypoints")
        flags = np.asarray(available)
        if flags.shape != (3, 2) or flags.dtype != np.bool_:
            raise FretboardGeometryError("Fretboard availability must be a boolean (3, 2) array.")
        if (
            not isinstance(image_size, (tuple, list))
            or len(image_size) != 2
            or any(type(value) is not int or value <= 0 for value in image_size)
        ):
            raise FretboardGeometryError("image_size must be positive integer (width, height).")
        if np.isinf(points).any() or not np.isfinite(points[flags]).all():
            raise FretboardGeometryError("Available fretboard keypoints must be finite.")
        complete = flags.all(axis=1)
        if int(complete.sum()) < 2:
            raise FretboardGeometryError("At least two complete cross-string anchors are required.")
        source = points[complete].reshape(-1, 2)
        target = np.asarray([
            (CANONICAL_ANCHORS[anchor], string)
            for anchor in np.flatnonzero(complete)
            for string in (0., 1.)
        ])
        matrix = _homography(source, target)
        inverse = np.linalg.inv(matrix)
        projected = _transform(matrix, source)
        error = float(np.sqrt(np.mean(np.square(projected - target))))
        names = tuple(name for name, present in zip(ANCHOR_NAMES, complete, strict=True) if present)
        return cls(matrix, inverse, names, error, tuple(image_size))

    def coordinate(self, point):
        x, y = _transform(self.image_to_canonical, _points(point, (2,), "image point"))
        string_position = float(y * (STRING_COUNT - 1))
        nearest = int(np.clip(round(string_position), 0, STRING_COUNT - 1))
        return FretboardCoordinate(
            scale_position=float(x),
            string_position=string_position,
            fret_position=continuous_fret(float(x)),
            nearest_string=nearest,
            nearest_string_distance=abs(string_position - nearest),
            inside_string_span=0 <= y <= 1,
        )

    def string_paths(self, samples=64):
        if type(samples) is not int or samples < 2:
            raise FretboardGeometryError("String paths require at least two samples.")
        scale = np.linspace(0., 1., samples)
        canonical = np.stack((
            np.broadcast_to(scale, (STRING_COUNT, samples)),
            np.broadcast_to(np.linspace(0., 1., STRING_COUNT)[:, None], (STRING_COUNT, samples)),
        ), axis=-1)
        return _transform(self.canonical_to_image, canonical)

    def fret_lines(self, maximum_fret=24):
        if type(maximum_fret) is not int or maximum_fret < 0:
            raise FretboardGeometryError("maximum_fret must be a nonnegative integer.")
        canonical = np.asarray([
            ((fret_scale_position(fret), 0.), (fret_scale_position(fret), 1.))
            for fret in range(maximum_fret + 1)
        ])
        return _transform(self.canonical_to_image, canonical)
