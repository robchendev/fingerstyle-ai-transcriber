import unittest

import numpy as np

from scripts.fretboard_features import (
    APPENDED_FEATURE_LAYOUT, INPUT_REPRESENTATION, SCHEMA_VERSION, STRUCTURED_DIM,
)
from scripts.fretboard_geometry import (
    FretboardGeometry, FretboardGeometryError, continuous_fret,
    fret_scale_position,
)


def anchors():
    return np.asarray([
        [[900., 180.], [900., 240.]],
        [[575., 265.], [575., 325.]],
        [[250., 350.], [250., 410.]],
    ])


class FretboardGeometryTests(unittest.TestCase):
    def test_projective_coordinates_recover_all_anchor_semantics(self):
        geometry = FretboardGeometry.from_keypoints(
            anchors(), np.ones((3, 2), dtype=bool), (1280, 720),
        )
        for anchor, scale in enumerate((0., .5, 1.)):
            for string in (0, 1):
                result = geometry.coordinate(anchors()[anchor, string])
                self.assertAlmostEqual(result.scale_position, scale, places=5)
                self.assertAlmostEqual(result.string_position, string * 5, places=5)
        self.assertEqual(geometry.available_anchors, ("nut", "fret12", "bridge"))
        self.assertLess(geometry.reprojection_error, 1e-8)

    def test_two_complete_pairs_support_partial_shots(self):
        points = anchors()
        points[2] = np.nan
        available = np.ones((3, 2), dtype=bool)
        available[2] = False
        geometry = FretboardGeometry.from_keypoints(points, available, (1280, 720))
        self.assertEqual(geometry.available_anchors, ("nut", "fret12"))
        self.assertEqual(geometry.string_paths().shape, (6, 64, 2))
        self.assertEqual(geometry.fret_lines(24).shape, (25, 2, 2))

    def test_incomplete_or_degenerate_anchors_fail_closed(self):
        for points, available in (
            (anchors(), np.asarray([[True, True], [True, False], [False, False]])),
            (np.zeros((3, 2, 2)), np.ones((3, 2), dtype=bool)),
        ):
            with self.assertRaises(FretboardGeometryError):
                FretboardGeometry.from_keypoints(points, available, (1280, 720))

    def test_equal_tempered_fret_positions_round_trip(self):
        for fret in (0, 1, 5, 12, 17, 24, 36):
            self.assertAlmostEqual(continuous_fret(fret_scale_position(fret)), fret)
        self.assertIsNone(continuous_fret(1.))

    def test_schema_v5_appends_without_reinterpreting_d194(self):
        self.assertEqual(SCHEMA_VERSION, 5)
        self.assertEqual(INPUT_REPRESENTATION, "guitar-hand-fretboard-233-v1")
        self.assertEqual(STRUCTURED_DIM, 233)
        covered = [
            index
            for value in APPENDED_FEATURE_LAYOUT.values()
            for index in range(value["start"], value["stop"])
        ]
        self.assertEqual(covered, list(range(194, 233)))


if __name__ == "__main__":
    unittest.main()
