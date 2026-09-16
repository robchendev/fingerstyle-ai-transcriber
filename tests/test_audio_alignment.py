from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np

from scripts.audio_alignment import (
    AlignmentError,
    Features,
    _cost_block,
    _dtw_row,
    align_features,
    audio_features,
    reference_features,
)


def feature_sequence(pitches, *, hop=0.05, onset=None):
    chroma = np.zeros((len(pitches), 12), dtype=np.float64)
    for index, pitch in enumerate(pitches):
        if pitch is not None:
            chroma[index, pitch % 12] = 1
    return Features(
        np.arange(len(pitches), dtype=np.float64) * hop,
        chroma,
        np.zeros(len(pitches)) if onset is None else np.asarray(onset, dtype=np.float64),
        np.ones(len(pitches)),
    )


def straightforward_dtw(costs, penalty, mode):
    n, m = costs.shape
    total = np.full((n + 1, m + 1), np.inf)
    total[0, 0] = 0
    if mode == "subsequence":
        total[:, 0] = 0
    directions = np.zeros((n, m), dtype=int)
    for i in range(n):
        for j in range(m):
            candidates = (total[i, j], total[i, j + 1] + penalty, total[i + 1, j] + penalty)
            directions[i, j] = np.argmin(candidates)
            total[i + 1, j + 1] = costs[i, j] + min(candidates)
    i = n - 1 if mode == "global" else int(np.argmin(total[1:, m]))
    objective = total[i + 1, m]
    j = m - 1
    path = []
    while i >= 0 and j >= 0:
        path.append((i, j))
        direction = directions[i, j]
        if direction != 2:
            i -= 1
        if direction != 1:
            j -= 1
    return objective, np.array(path[::-1])


class AudioFeatureTests(unittest.TestCase):
    def test_readonly_decoded_pcm_is_not_mutated(self):
        rate = 22050
        times = np.arange(6615, dtype=np.float64) / rate
        tone = (0.5 * np.sin(2 * np.pi * 110 * times)).astype(np.float32)
        payload = np.column_stack((tone, -tone)).tobytes()
        samples = np.frombuffer(payload, dtype=np.float32).reshape(-1, 2)
        self.assertFalse(samples.flags.writeable)
        result = audio_features(samples, rate)
        expected = audio_features(samples.copy(), rate)
        self.assertFalse(samples.flags.writeable)
        self.assertEqual(samples.tobytes(), payload)
        for name in ("times", "chroma", "onset", "activity"):
            np.testing.assert_array_equal(getattr(result, name), getattr(expected, name))

    def test_low_g_and_antiphase_stereo_preserve_tonal_evidence(self):
        rate = 22050
        times = np.arange(rate, dtype=np.float64) / rate
        mono = (0.5 * np.sin(2 * np.pi * 49 * times)).astype(np.float32)[:, None]
        ordinary = audio_features(mono, rate)
        opposite = audio_features(np.column_stack((mono[:, 0], -mono[:, 0])), rate)
        other_channel = audio_features(np.column_stack((np.zeros(len(mono)), mono[:, 0])).astype(np.float32), rate)
        self.assertEqual(int(np.argmax(np.mean(ordinary.chroma[4:-4], axis=0))), 7)
        for result in (ordinary, opposite, other_channel):
            np.testing.assert_allclose(np.linalg.norm(result.chroma, axis=1), 1, atol=1e-12)
            self.assertTrue(np.all(result.activity > 0))
        np.testing.assert_allclose(ordinary.chroma, opposite.chroma, atol=1e-12)
        np.testing.assert_allclose(ordinary.onset, opposite.onset, atol=1e-12)
        np.testing.assert_allclose(ordinary.chroma, other_channel.chroma, atol=1e-12)

    def test_short_supported_audio_times_and_independent_attack_window(self):
        rate = 22050
        times = np.arange(rate, dtype=np.float64) / rate
        samples = (np.sin(2 * np.pi * 220 * times) * (times >= 0.6)).astype(np.float32)[:, None]
        result = audio_features(samples, rate)
        self.assertLessEqual(abs(result.times[np.argmax(result.onset)] - 0.6), 0.05)
        self.assertEqual(result.times.dtype, np.float64)
        self.assertEqual(result.times[0], 0)
        self.assertTrue(np.all(result.times < len(samples) / rate))
        np.testing.assert_allclose(np.diff(result.times), 0.05)
        short = audio_features(samples[int(0.6 * rate):int(0.75 * rate)], rate)
        self.assertGreaterEqual(len(short.times), 2)
        for values in (result.chroma, result.onset, result.activity, short.chroma):
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue(np.all(values >= 0))
        self.assertTrue(np.all(result.onset <= 1))
        self.assertTrue(np.all(result.activity <= 1))

    def test_silent_nonfinite_short_and_bad_audio_inputs_fail(self):
        valid = np.ones((4410, 1), dtype=np.float32)
        invalid = [
            np.zeros_like(valid), valid, valid[:10], np.empty((0, 1), dtype=np.float32),
            valid[:, 0], np.empty((4410, 0), dtype=np.float32),
            np.full_like(valid, np.nan), np.full_like(valid, np.inf), valid.astype(int),
        ]
        for samples in invalid:
            with self.subTest(shape=samples.shape), self.assertRaises(AlignmentError):
                audio_features(samples, 22050)
        for sample_rate in (0, True, 22050.0, 96000):
            with self.subTest(sample_rate=sample_rate), self.assertRaises(AlignmentError):
                audio_features(valid, sample_rate)
        for hop in (0, -0.1, np.nan, True, 0.001):
            with self.subTest(hop=hop), self.assertRaises(AlignmentError):
                audio_features(valid, 22050, hop_seconds=hop)


class ReferenceFeatureTests(unittest.TestCase):
    def test_unknown_end_is_only_transient_and_events_are_unchanged(self):
        events = [
            {"onset": 0.1, "end": None, "pitch": 43, "percussive": False},
            {"onset": 0.5, "end": 0.9, "pitch": None, "percussive": True},
        ]
        before = deepcopy(events)
        result = reference_features(events, 1)
        self.assertTrue(np.any(result.chroma[2:5, 7] > 0))
        self.assertFalse(np.any(result.chroma[6:]))
        self.assertGreater(result.onset[10], 0)
        self.assertLess(result.activity[16], 0.1)
        self.assertEqual(events, before)

    def test_all_events_including_subhop_and_final_events_contribute(self):
        events = [
            {"onset": 0.001, "end": 0.002, "pitch": 60.5, "percussive": False},
            {"onset": 0.2999, "end": None, "pitch": None, "percussive": False},
        ]
        result = reference_features(events, 0.3)
        self.assertGreater(result.chroma[0, 0], 0)
        self.assertGreater(result.chroma[0, 1], 0)
        self.assertGreater(result.onset[-1], 0)
        self.assertTrue(np.all(result.times < 0.3))
        np.testing.assert_allclose(np.diff(result.times), 0.05)

    def test_known_notation_is_a_roll_not_a_decay(self):
        result = reference_features([{"onset": 0, "end": 1, "pitch": 60, "percussive": False}], 1)
        np.testing.assert_allclose(result.chroma[:, 0], 1)
        np.testing.assert_allclose(result.activity, 1)

    def test_invalid_event_is_never_silently_ignored(self):
        event = {"onset": 0.1, "end": 0.5, "pitch": 60, "percussive": False}
        for key, values in {
            "onset": (-1, 1, np.nan, "0", None, True),
            "end": (0.1, 2, np.inf, "0.5", False),
            "pitch": (-1, 128, np.nan, "60", True, 10 ** 1000),
            "percussive": (1, None, "yes", True),
        }.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(AlignmentError):
                    reference_features([event, {**event, key: value}], 1)
        for events in ([], (), [None], [{"onset": 0}]):
            with self.subTest(events=events), self.assertRaises(AlignmentError):
                reference_features(events, 1)
        for duration in (0, -1, 0.09, 901, np.inf, True, "1"):
            with self.subTest(duration=duration), self.assertRaises(AlignmentError):
                reference_features([event], duration)


class FirstAttackTests(unittest.TestCase):
    def test_lead_in_does_not_advance_score_and_late_music_is_retained(self):
        from scripts.audio_alignment import align_first_attack
        reference = feature_sequence([0, 4, 7, 2, 9, 0], onset=[1, 1, 1, 1, 1, 1])
        audio = feature_sequence([None] * 5 + [0, 4, 7, 2, 9, 0], onset=[0] * 5 + [1, 0, 1, 0, 1, 0])
        before = audio.times.copy()
        result = align_first_attack(reference, audio, first_reference_seconds=0)
        self.assertEqual((result["reference_indices"][0], result["audio_indices"][0]), (0, 5))
        self.assertEqual((result["reference_indices"][-1], result["audio_indices"][-1]), (5, 10))
        self.assertEqual(result["fixedFrameAnchors"], [(0, 5)])
        self.assertEqual(result["diagnostics"]["first_attack_audio_seconds"], .25)
        self.assertFalse(result["diagnostics"]["first_attack_is_human_approved"])
        np.testing.assert_array_equal(audio.times, before)

    def test_earliest_compatible_attack_not_loudest_later_attack_or_wrong_pitch_noise(self):
        from scripts.audio_alignment import align_first_attack
        reference = feature_sequence([0, 4, 7, 2, 9], onset=[1, 0, 1, 0, 1])
        audio = feature_sequence([11, None, 0, 4, 7, 2, 9], onset=[1, 0, .25, 0, 1, 0, .8])
        result = align_first_attack(reference, audio, first_reference_seconds=0)
        self.assertEqual(result["audio_indices"][0], 2)
        self.assertEqual(result["diagnostics"]["first_attack_onset_strength"], .25)

    def test_leading_score_rest_is_not_mistaken_for_first_attack_or_deleted(self):
        from scripts.audio_alignment import align_first_attack
        reference = feature_sequence([None, None, 0, 4, 7, 2], onset=[0, 0, 1, 0, 1, 0])
        audio = feature_sequence([None, None, 0, 4, 7, 2], onset=[0, 0, 1, 0, 1, 0])
        result = align_first_attack(reference, audio, first_reference_seconds=.1)
        self.assertEqual(result["reference_indices"][0], 2)
        self.assertEqual(result["audio_indices"][0], 2)
        self.assertLess(result["diagnostics"]["reference_coverage_fraction"], 1)
        self.assertEqual(len(reference.times), 6)

    def test_no_attack_does_not_fall_back_to_file_zero(self):
        from scripts.audio_alignment import align_first_attack
        reference = feature_sequence([0, 4, 7], onset=[1, 0, 1])
        audio = feature_sequence([0, 4, 7], onset=[0, 0, 0])
        with self.assertRaisesRegex(AlignmentError, "No plausible first attack"):
            align_first_attack(reference, audio, first_reference_seconds=0)
        with self.assertRaisesRegex(AlignmentError, "reference onset"):
            align_first_attack(reference, audio, first_reference_seconds=.05)


class DtwTests(unittest.TestCase):
    def assert_complete_path(self, result, n, m, mode):
        r, a = result["reference_indices"], result["audio_indices"]
        self.assertEqual(r.dtype.kind, "i")
        self.assertEqual(a.dtype.kind, "i")
        self.assertEqual(r.shape, a.shape)
        self.assertEqual(r.shape, result["local_costs"].shape)
        self.assertTrue(np.all((0 <= r) & (r < n)))
        self.assertTrue(np.all((0 <= a) & (a < m)))
        steps = np.column_stack((np.diff(r), np.diff(a)))
        self.assertTrue(np.all((0 <= steps) & (steps <= 1)))
        self.assertTrue(np.all(steps.sum(axis=1) >= 1))
        self.assertEqual(a[0], 0)
        self.assertEqual(a[-1], m - 1)
        np.testing.assert_array_equal(np.unique(a), np.arange(m))
        if mode == "global":
            self.assertEqual(r[0], 0)
            self.assertEqual(r[-1], n - 1)
        self.assertTrue(np.isfinite(result["local_costs"]).all())
        self.assertTrue(np.all(result["local_costs"] >= 0))
        self.assertAlmostEqual(result["mean_cost"], np.mean(result["local_costs"]))
        json.dumps(result["diagnostics"], allow_nan=False)
        for value in result["diagnostics"].values():
            for scalar in value if isinstance(value, list) else [value]:
                self.assertIn(type(scalar), (str, int, float, bool, type(None)))

    def test_optimized_rows_match_straightforward_recurrence(self):
        generator = np.random.default_rng(17)
        for mode in ("global", "subsequence"):
            for n, m in ((2, 7), (7, 2), (6, 8), (9, 5)):
                for penalty in (0, 0.08, 1):
                    costs = generator.uniform(0, 1, (n, m))
                    expected, _ = straightforward_dtw(costs, penalty, mode)
                    previous = np.full(m, np.inf)
                    endpoints = []
                    for index, row in enumerate(costs):
                        previous, _ = _dtw_row(row, previous, penalty, index == 0 or mode == "subsequence")
                        endpoints.append(previous[-1])
                    actual = endpoints[-1] if mode == "global" else min(endpoints)
                    self.assertAlmostEqual(actual, expected, places=12)

    def test_full_path_and_objective_match_small_reference(self):
        generator = np.random.default_rng(31)
        for mode in ("global", "subsequence"):
            for n, m in ((2, 7), (7, 2), (7, 9), (9, 5)):
                r = generator.uniform(size=(n, 12))
                a = generator.uniform(size=(m, 12))
                r /= np.linalg.norm(r, axis=1, keepdims=True)
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                reference = Features(np.arange(n) * 0.05, r, generator.uniform(size=n), np.ones(n))
                audio = Features(np.arange(m) * 0.05, a, generator.uniform(size=m), np.ones(m))
                costs = _cost_block(r, a, reference.onset, audio.onset)
                expected, path = straightforward_dtw(costs, 0.08, mode)
                result = align_features(reference, audio, mode=mode)
                self.assert_complete_path(result, n, m, mode)
                np.testing.assert_array_equal(np.column_stack((result["reference_indices"], result["audio_indices"])), path)
                self.assertAlmostEqual(result["diagnostics"]["objective_cost"], expected, places=12)
                np.testing.assert_allclose(result["local_costs"], costs[path[:, 0], path[:, 1]], atol=1e-14)
                non_diagonal = np.count_nonzero(np.any(np.diff(path, axis=0) == 0, axis=1))
                self.assertAlmostEqual(float(result["local_costs"].sum()) + 0.08 * non_diagonal, expected, places=12)

    def test_known_timing_warp_follows_pitch_boundaries(self):
        pitches = [0, 4, 7, 2, 9]
        reference = feature_sequence(np.repeat(pitches, [4, 4, 4, 4, 4]))
        audio = feature_sequence(np.repeat(pitches, [3, 6, 2, 5, 4]))
        result = align_features(reference, audio)
        self.assert_complete_path(result, 20, 20, "global")
        self.assertLess(result["mean_cost"], 1e-12)
        for score_boundary, audio_boundary in zip((4, 8, 12, 16), (3, 9, 11, 16)):
            positions = result["audio_indices"][result["reference_indices"] == score_boundary]
            self.assertEqual(int(positions.min()), audio_boundary)
        repeated = align_features(reference, audio)
        for key in ("reference_indices", "audio_indices", "local_costs"):
            np.testing.assert_array_equal(result[key], repeated[key])
        self.assertEqual(result["diagnostics"], repeated["diagnostics"])

    def test_subsequence_consumes_all_audio_not_all_score(self):
        reference = feature_sequence([11, 11, 0, 4, 7, 2, 9, 10, 10])
        audio = feature_sequence([0, 4, 7, 2, 9])
        result = align_features(reference, audio, mode="subsequence")
        self.assert_complete_path(result, 9, 5, "subsequence")
        np.testing.assert_array_equal(result["reference_indices"], [2, 3, 4, 5, 6])
        np.testing.assert_array_equal(result["audio_indices"], np.arange(5))
        global_result = align_features(reference, audio)
        self.assert_complete_path(global_result, 9, 5, "global")
        self.assertGreater(global_result["mean_cost"], result["mean_cost"])
        reversed_result = align_features(audio, reference, mode="subsequence")
        self.assert_complete_path(reversed_result, 5, 9, "subsequence")
        self.assertGreater(reversed_result["mean_cost"], result["mean_cost"])

    def test_deterministic_ties_and_uninformative_zero_cost(self):
        reference = feature_sequence([0] * 6)
        audio = feature_sequence([0] * 4)
        first = align_features(reference, audio, mode="subsequence", warp_penalty=0)
        second = align_features(reference, audio, mode="subsequence", warp_penalty=0)
        for key in ("reference_indices", "audio_indices", "local_costs"):
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(first["diagnostics"], second["diagnostics"])
        np.testing.assert_array_equal(first["reference_indices"], [0, 0, 0, 0])
        self.assertTrue(first["diagnostics"]["degenerate"])
        self.assertTrue(first["diagnostics"]["uninformative"])
        self.assertTrue(first["diagnostics"]["requires_human_review"])
        self.assertFalse(first["diagnostics"]["cost_is_probability"])

    def test_wrong_pitch_sequence_is_flagged_not_confident(self):
        reference = feature_sequence(np.repeat([0, 4, 7, 0, 7, 4], 4))
        audio = feature_sequence(np.repeat([1, 5, 8, 1, 8, 5], 4))
        result = align_features(reference, audio)
        self.assertGreater(result["mean_cost"], 0.7)
        self.assertTrue(result["diagnostics"]["poor_match"])
        self.assertIn("high_mismatch", result["diagnostics"]["flags"])
        self.assertEqual(result["diagnostics"]["quality"], "flagged")

    def test_stalled_path_is_diagnosed(self):
        reference = feature_sequence([0, 4, 7])
        audio = feature_sequence([0] + [4] * 50 + [7])
        result = align_features(reference, audio)
        self.assert_complete_path(result, 3, 52, "global")
        self.assertTrue(result["diagnostics"]["degenerate"])
        self.assertGreater(result["diagnostics"]["max_reference_stall_seconds"], 2)

    def test_unknown_score_pitch_does_not_require_acoustic_silence(self):
        reference = feature_sequence([None] * 3, onset=[1, 0, 1])
        audible = feature_sequence([0, 4, 7], onset=[1, 0, 1])
        unpitched = feature_sequence([None] * 3, onset=[1, 0, 1])
        first = align_features(reference, audible)
        second = align_features(reference, unpitched)
        self.assertAlmostEqual(first["mean_cost"], second["mean_cost"])
        self.assertGreater(first["mean_cost"], 0)

    def test_onset_only_cues_can_locate_an_audio_subsequence(self):
        reference = feature_sequence([None] * 12, onset=[0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0])
        audio = feature_sequence([None] * 7, onset=[1, 0, 0, 1, 0, 1, 0])
        result = align_features(reference, audio, mode="subsequence")
        np.testing.assert_array_equal(result["reference_indices"], np.arange(2, 9))
        np.testing.assert_array_equal(result["audio_indices"], np.arange(7))
        self.assert_complete_path(result, 12, 7, "subsequence")
        self.assertIsNone(result["diagnostics"]["mean_tonal_mismatch"])
        self.assertTrue(result["diagnostics"]["requires_human_review"])

    def test_invalid_features_and_ranges_fail(self):
        valid = feature_sequence([0, 4, 7])
        invalid = [
            Features(np.array([]), np.empty((0, 12)), np.array([]), np.array([])),
            Features(valid.times, np.ones((3, 11)), valid.onset, valid.activity),
            Features(valid.times, valid.chroma, np.zeros(2), valid.activity),
            Features(valid.times, valid.chroma, valid.onset, np.zeros((3, 1))),
            Features(valid.times + 0.1, valid.chroma, valid.onset, valid.activity),
            Features(np.array([0, 0.05, 0.05]), valid.chroma, valid.onset, valid.activity),
            Features(np.array([0, 0.05, 0.11]), valid.chroma, valid.onset, valid.activity),
            Features(valid.times, -valid.chroma, valid.onset, valid.activity),
            Features(valid.times, valid.chroma, np.full(3, np.nan), valid.activity),
            Features(valid.times, np.full((3, 12), np.inf), valid.onset, valid.activity),
            Features(valid.times, valid.chroma, np.full(3, 1.1), valid.activity),
            Features(valid.times, valid.chroma, valid.onset, np.full(3, -0.1)),
            feature_sequence([None] * 3),
            None,
        ]
        for value in invalid:
            for side in ("reference", "audio"):
                with self.subTest(side=side, value=value), self.assertRaises(AlignmentError):
                    align_features(value if side == "reference" else valid, value if side == "audio" else valid)
        with self.assertRaisesRegex(AlignmentError, "same hop"):
            align_features(valid, feature_sequence([0, 4, 7], hop=0.1))
        for kwargs in (
            {"mode": "partial"}, {"max_cells": 0}, {"max_cells": True}, {"max_cells": 9.0},
            {"max_cells": 120_000_001}, {"warp_penalty": -1}, {"warp_penalty": np.inf},
            {"warp_penalty": 1.1}, {"warp_penalty": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(AlignmentError):
                align_features(valid, valid, **kwargs)

    def test_cell_limit_fails_before_cost_matrix_work(self):
        valid = feature_sequence([0, 4, 7])
        with patch("scripts.audio_alignment._cost_block") as cost:
            with self.assertRaisesRegex(AlignmentError, "9 cells"):
                align_features(valid, valid, max_cells=8)
            cost.assert_not_called()
        self.assert_complete_path(align_features(valid, valid, max_cells=9), 3, 3, "global")


if __name__ == "__main__":
    unittest.main()
