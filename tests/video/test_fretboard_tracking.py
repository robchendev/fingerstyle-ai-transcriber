import unittest

import cv2
import numpy as np

from fretboard_tracking import TrackingConfig, track_keypoints


class FretboardTrackingTests(unittest.TestCase):
    def test_optical_flow_tracks_translation_at_native_coordinates(self):
        previous = np.zeros((240, 320), np.uint8)
        points = np.asarray([
            [[220, 80], [220, 120]],
            [[160, 90], [160, 140]],
            [[80, 105], [80, 170]],
        ], np.float32)
        for point in points.reshape(-1, 2):
            cv2.circle(previous, tuple(point.astype(int)), 5, 255, -1)
        matrix = np.float32([[1, 0, 7], [0, 1, 4]])
        current = cv2.warpAffine(previous, matrix, (320, 240))
        result = track_keypoints(
            previous, current, points, np.ones((3, 2), bool),
            np.ones(2, np.float32), np.ones(2, np.float32),
            (320, 240), TrackingConfig(flow_error_pixels=1.),
        )
        self.assertIsNotNone(result)
        tracked, available, _, error = result
        np.testing.assert_allclose(tracked, points + [7, 4], atol=.6)
        self.assertTrue(available.all())
        self.assertLess(error, .01)


if __name__ == "__main__":
    unittest.main()
