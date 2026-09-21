from fractions import Fraction
import unittest

import cv2
import numpy as np

from hand_motion import normalize_audio_asset
from core import (
    ClockMapping,
    EvidenceError,
    GuitarCoordinateFrame,
)
from geometry import (
    GEOMETRY_POINTS,
    validate_geometry_annotations,
)
from shot_inspector import ShotConfig, _analysis_frame, cut_score, dissolve_measurement
from audio_sync import _normalized_valid_correlation
from hand_tracking import (
    merge_observations,
    observation_near_wrist,
    pose_hand_crops,
    prefer_guided_detection,
    remap_image_landmarks,
    same_hand_landmarks,
)


def point(x, y, confidence=1.0):
    return {"x": x, "y": y, "confidence": confidence}


def geometry(scale=1.0, offset_x=0.0, offset_y=0.0):
    return {
        "nut": point(offset_x + 10 * scale, offset_y),
        "neckBody": point(offset_x, offset_y),
        "fretboardUpper": point(offset_x, offset_y - scale),
        "fretboardLower": point(offset_x, offset_y + scale),
    }


class VideoEvidenceTests(unittest.TestCase):
    def test_clock_mapping_uses_integer_pts_and_rational_time_base(self):
        mapping = ClockMapping(sample_rate=48000, offset_samples=-24000, rate=Fraction(1001, 1000))
        self.assertEqual(mapping.map_pts(1000, Fraction(1, 1000)), 24048)

    def test_guitar_coordinates_follow_camera_translation_and_scale(self):
        first = GuitarCoordinateFrame.from_landmarks(geometry())
        moved = GuitarCoordinateFrame.from_landmarks(geometry(scale=3, offset_x=50, offset_y=20))
        a = first.transform(point(5, 0.5))
        b = moved.transform(point(65, 21.5))
        self.assertAlmostEqual(a["alongNeck"], b["alongNeck"])
        self.assertAlmostEqual(a["acrossFretboard"], b["acrossFretboard"])

    def test_retained_audio_full_source_is_not_missing_trim_provenance(self):
        asset = {"sourceSha256": "1" * 64, "sha256": "2" * 64,
                 "sampleRate": 48000, "channels": 2, "sampleCount": 12345, "appliedRangeSeconds": None}
        self.assertEqual(normalize_audio_asset(asset)["sourceSampleBounds"], {"startSample": 0, "stopSampleExclusive": 12345})
        for changed in ({k: v for k, v in asset.items() if k != "appliedRangeSeconds"},
                        {**asset, "appliedRangeSeconds": [1., 2.]},
                        {**asset, "sourceSampleBounds": {"startSample": False, "stopSampleExclusive": 12345}},
                        {**asset, "sourceSampleBounds": {"startSample": 1, "stopSampleExclusive": 12346}}):
            with self.assertRaises(EvidenceError):
                normalize_audio_asset(changed)

    def test_cut_score_separates_similar_and_distinct_frames(self):
        black = np.zeros((120, 160, 3), dtype=np.uint8)
        near = black.copy()
        near[:, :5] = 5
        white = np.full_like(black, 255)
        base = _analysis_frame(black, 80)
        self.assertLess(cut_score(base, _analysis_frame(near, 80)), .05)
        self.assertGreater(cut_score(base, _analysis_frame(white, 80)), .3)

    def test_dissolve_measurement_accepts_blend_but_rejects_motion(self):
        start = np.zeros((120, 160, 3), dtype=np.uint8)
        start[:, :80] = 255
        end = np.zeros_like(start)
        end[:, 80:] = 255
        blend = ((start.astype(np.uint16) + end.astype(np.uint16)) // 2).astype(np.uint8)
        moved = np.roll(start, 40, axis=1)
        blend_measurement = dissolve_measurement(
            _analysis_frame(start, 80),
            _analysis_frame(blend, 80),
            _analysis_frame(end, 80),
        )
        motion_measurement = dissolve_measurement(
            _analysis_frame(start, 80),
            _analysis_frame(moved, 80),
            _analysis_frame(end, 80),
        )
        self.assertAlmostEqual(blend_measurement["blendAlpha"], .5, places=1)
        self.assertLess(blend_measurement["residualRatio"], .1)
        self.assertGreater(motion_measurement["residualRatio"], .49)

    def test_shot_config_rejects_invalid_threshold(self):
        with self.assertRaises(EvidenceError):
            ShotConfig(cut_threshold=1.1)
        with self.assertRaises(EvidenceError):
            ShotConfig(candidate_cut_threshold=.3, cut_threshold=.28)

    def test_crop_landmarks_remap_to_full_frame(self):
        points = np.array([[0, 0, 0], [1, 1, .5]], dtype=np.float32)
        remapped = remap_image_landmarks(points, (100, 50, 500, 450), 1000, 500)
        np.testing.assert_allclose(remapped[0], [.1, .1, 0])
        np.testing.assert_allclose(remapped[1], [.5, .9, .2])

    def test_rotated_crop_landmarks_remap_to_full_frame(self):
        crop = (100, 50, 500, 450)
        affine = cv2.getRotationMatrix2D((200, 200), 45, 1)
        original_crop_point = np.array([[[100, 160]]], dtype=np.float32)
        rotated = cv2.transform(original_crop_point, affine)[0, 0] / 400
        remapped = remap_image_landmarks(
            [(rotated[0], rotated[1], 0)],
            crop,
            1000,
            500,
            cv2.invertAffineTransform(affine),
        )
        np.testing.assert_allclose(remapped[0, :2], [.2, .42], atol=1e-6)

    def test_pose_hand_crops_center_beyond_each_wrist(self):
        class Landmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.visibility = 1.0
                self.presence = 1.0

        landmarks = [Landmark(.5, .5) for _ in range(17)]
        landmarks[11], landmarks[12] = Landmark(.4, .3), Landmark(.6, .3)
        landmarks[13], landmarks[15] = Landmark(.4, .4), Landmark(.3, .5)
        landmarks[14], landmarks[16] = Landmark(.6, .4), Landmark(.7, .5)
        crops = pose_hand_crops(landmarks, 1000, 800)
        self.assertEqual(len(crops), 2)
        self.assertLess((crops[0]["crop"][0] + crops[0]["crop"][2]) / 2, 300)
        self.assertGreater((crops[1]["crop"][0] + crops[1]["crop"][2]) / 2, 700)

    def test_merge_observations_deduplicates_same_wrist(self):
        def observation(wrists):
            image = np.full((2, 21, 3), np.nan, dtype=np.float32)
            world = np.full_like(image, np.nan)
            handedness = np.full(2, -1, dtype=np.int8)
            scores = np.zeros(2, dtype=np.float32)
            for index, wrist in enumerate(wrists):
                image[index] = wrist
                image[index, [5, 9, 13, 17], 1] += .05
                scores[index] = .8 + index * .1
            return len(wrists), image, world, handedness, scores

        merged = merge_observations(
            observation([(.2, .3, 0)]),
            observation([(.21, .31, 0), (.8, .7, 0)]),
        )
        self.assertEqual(merged[0], 2)

    def test_close_distinct_small_hands_are_not_merged_by_frame_radius(self):
        left = np.zeros((21, 3), np.float32)
        left[:] = (.36, .63, 0)
        left[[5, 9, 13, 17], 1] += .012
        right = left.copy()
        right[:, 0] -= .07
        right[:, 1] -= .02
        self.assertLess(np.linalg.norm(left[0, :2] - right[0, :2]), .08)
        self.assertFalse(same_hand_landmarks(left, right, (3840, 2160)))
        duplicate = left.copy()
        duplicate[:, :2] += .001
        self.assertTrue(same_hand_landmarks(left, duplicate, (3840, 2160)))
        for factor in (.5, 2):
            self.assertFalse(same_hand_landmarks(left * factor, right * factor, (3840, 2160)))
            self.assertTrue(same_hand_landmarks(left * factor, duplicate * factor, (3840, 2160)))

    def test_wrist_proximity_without_palm_agreement_is_not_duplicate_evidence(self):
        left = np.zeros((21, 3), np.float32)
        left[[5, 9, 13, 17], 0] = .1
        right = left.copy()
        right[[5, 9, 13, 17], 0] = -.1
        self.assertFalse(same_hand_landmarks(left, right))

    def test_guided_observation_must_match_pose_wrist(self):
        image = np.full((2, 21, 3), np.nan, dtype=np.float32)
        world = np.full_like(image, np.nan)
        handedness = np.full(2, -1, dtype=np.int8)
        scores = np.zeros(2, dtype=np.float32)
        image[0] = (.21, .31, 0)
        image[1] = (.7, .8, 0)
        scores[:] = (.7, .9)
        selected = observation_near_wrist(
            (2, image, world, handedness, scores),
            (.2, .3),
            (100, 100, 500, 500),
            1000,
            1000,
        )
        self.assertEqual(selected[0], 1)
        np.testing.assert_allclose(selected[1][0, 0, :2], [.21, .31])
        rejected = observation_near_wrist(
            (1, image, world, handedness, scores),
            (.9, .1),
            (100, 100, 500, 500),
            1000,
            1000,
        )
        self.assertEqual(rejected[0], 0)

    def test_guided_detection_only_replaces_strictly_lower_count(self):
        self.assertTrue(prefer_guided_detection(0, 1))
        self.assertTrue(prefer_guided_detection(1, 2))
        self.assertFalse(prefer_guided_detection(1, 1))
        self.assertFalse(prefer_guided_detection(2, 1))

    def test_geometry_annotations_require_complete_trackable_frame(self):
        shots = {
            "shots": [{
                "shotId": 0,
                "startPts": 0,
                "endPtsExclusive": 100,
            }],
        }
        row = {
            "shotId": 0,
            "startPts": 0,
            "endPtsExclusive": 100,
            "keyframePts": 50,
            "keyframeImage": "keyframes\\shot.jpg",
            "state": "trackable",
            "geometry": {
                "nut": point(.8, .4),
                "neckBody": point(.4, .5),
                "fretboardUpper": point(.4, .48),
                "fretboardLower": point(.4, .52),
                "soundhole": point(.3, .55),
                "bridge": point(.2, .58),
            },
            "reviewNote": None,
        }
        document = {
            "schemaVersion": 1,
            "kind": "guitar-geometry-annotations",
            "visibility": "private",
            "videoSha256": "video",
            "shotsSha256": "shots",
            "pointOrder": list(GEOMETRY_POINTS),
            "coordinateSpace": "normalized_full_frame",
            "shots": [row],
            "reviewComplete": True,
            "trainingPerformed": False,
        }
        normalized = validate_geometry_annotations(document, shots, "video", "shots")
        self.assertEqual(set(normalized[0]["geometry"]), set(GEOMETRY_POINTS))
        row["geometry"] = {"nut": point(.8, .4)}
        with self.assertRaises(EvidenceError):
            validate_geometry_annotations(document, shots, "video", "shots")

    def test_audio_correlation_finds_trimmed_clip_offset(self):
        source = np.random.default_rng(17).normal(size=1000)
        clip = source[230:730] * 2 + 3
        scores = _normalized_valid_correlation(source, clip)
        self.assertEqual(int(np.argmax(scores)), 230)
        self.assertGreater(float(scores[230]), .999)


if __name__ == "__main__":
    unittest.main()
