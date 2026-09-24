import unittest

import numpy as np

from scripts.fretboard_detector import (
    FretboardDetectorConfig, select_detection,
)


def points():
    return np.asarray([
        [[900., 180.], [900., 240.]],
        [[575., 265.], [575., 325.]],
        [[250., 350.], [250., 410.]],
    ])


class FretboardDetectorTests(unittest.TestCase):
    def test_selects_valid_six_point_geometry(self):
        config = FretboardDetectorConfig()
        selected = select_detection(
            points()[None], np.ones((1, 3, 2)), [.8],
            (1280, 720), config, 1920,
        )
        self.assertEqual(selected.score, .8)
        self.assertTrue(selected.available.all())
        self.assertEqual(selected.image_size, 1920)

    def test_partial_pairs_require_two_complete_anchors(self):
        scores = np.ones((1, 3, 2))
        scores[0, 2] = 0
        selected = select_detection(
            points()[None], scores, [.8], (1280, 720),
            FretboardDetectorConfig(), 1920,
        )
        self.assertIsNotNone(selected)
        scores[0, 1, 1] = 0
        self.assertIsNone(select_detection(
            points()[None], scores, [.8], (1280, 720),
            FretboardDetectorConfig(), 1920,
        ))

    def test_config_rejects_invalid_multiscale_settings(self):
        for options in (
            {"image_sizes": ()}, {"image_sizes": (1920, 1920)},
            {"proposal_score": .5, "acceptance_score": .25},
            {"keypoint_score": 2}, {"device": ""},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                FretboardDetectorConfig(**options)


if __name__ == "__main__":
    unittest.main()
