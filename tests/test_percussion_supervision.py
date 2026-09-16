from copy import deepcopy
import unittest

from scripts.percussion_supervision import percussion_annotation_coverage
from scripts.score_alignment import AlignmentInputError
from tests.test_score_alignment import clock_fixture


def coverage_fixture():
    canonical, normalization = clock_fixture()
    canonical.update(
        scoreTimingResolved=True,
        targets={"notes": [], "gestures": [], "rests": []},
        review={"notationSymbols": [], "unresolvedGestures": []},
    )
    candidate = {"denseMapping": [
        {"referenceSeconds": float(t), "clipSeconds": float(t), "scoreQuarter": float(t)}
        for t in (0, 3, 6)
    ]}
    return canonical, candidate, normalization


def unknown_symbol():
    return {
        "id": "p0:m0:v0:b1:n0", "voiceIndex": 0, "onsetQuarter": [1, 1],
        "isAttack": True, "notatedDurationQuarter": [4, 1], "observedDurationQuarter": [4, 1],
        "labelMask": {"attack": True, "notatedDuration": True, "gesture": False, "pitch": False, "fingering": False},
        "sourceSegments": [{
            "writtenBeatId": "m0:v0:b1", "visitIndex": 0, "measureIndex": 0,
            "onsetQuarter": [1, 1], "durationQuarter": [4, 1], "notatedDurationQuarter": [4, 1], "graceMode": None,
        }],
    }


def known_gesture():
    return {
        "id": "hit", "technique": "percussive_hit", "voiceIndex": 0,
        "onsetQuarter": [1, 1], "scoreOnsetKnown": True, "graceMode": None,
        "writtenBeatId": "m0:v0:b1", "visitIndex": 0, "symbolicNoteIds": [],
    }


class PercussionCoverageTests(unittest.TestCase):
    def test_resolved_coverage_includes_known_positive_neighborhoods_without_mutation(self):
        canonical, candidate, normalization = coverage_fixture()
        symbol, gesture = unknown_symbol(), known_gesture()
        symbol["labelMask"]["gesture"] = True
        gesture["symbolicNoteIds"] = [symbol["id"]]
        canonical["review"]["notationSymbols"] = [symbol]
        canonical["targets"]["gestures"] = [gesture]
        before = deepcopy((canonical, candidate, normalization))
        self.assertEqual(percussion_annotation_coverage(canonical, candidate, normalization), [[0., 6.]])
        self.assertEqual((canonical, candidate, normalization), before)

    def test_unclassified_long_symbol_and_source_cluster_censor_attack_not_sustain(self):
        canonical, candidate, normalization = coverage_fixture()
        symbol = unknown_symbol()
        canonical["review"]["notationSymbols"] = [symbol]
        canonical["review"]["unresolvedGestures"] = [{
            "performanceBeatId": "p0:m0:v0:b1", "sourcePerformanceBeatId": "p34:m31:v1:b2",
            "onsetQuarter": [1, 1], "reason": "uninterpreted_dead_note_cluster",
            "symbolicNoteIds": [symbol["id"]], "labelMask": {"gesture": False, "fingering": False},
        }]
        self.assertEqual(percussion_annotation_coverage(canonical, candidate, normalization), [[0., .9], [1.1, 6.]])

    def test_unknown_symbol_timing_or_source_attack_mask_censors_its_measure(self):
        for change in ("unknown-attack", "non-attack", "attack-mask", "grace", "covered-unknown"):
            canonical, candidate, normalization = coverage_fixture()
            symbol = unknown_symbol()
            if change in {"unknown-attack", "covered-unknown"}:
                symbol["isAttack"] = None
            elif change == "non-attack":
                symbol["isAttack"] = False
            elif change == "attack-mask":
                symbol["labelMask"]["attack"] = False
            else:
                symbol["sourceSegments"][0]["graceMode"] = "BeforeBeat"
            if change == "covered-unknown":
                symbol["labelMask"]["gesture"] = True
            canonical["review"]["notationSymbols"] = [symbol]
            with self.subTest(change=change):
                self.assertEqual(percussion_annotation_coverage(canonical, candidate, normalization), [[3.1, 6.]])

    def test_masked_gesture_is_unknown_not_negative_at_a_resolved_attack(self):
        canonical, candidate, normalization = coverage_fixture()
        gesture = known_gesture()
        gesture["labelMask"] = {"gesture": False, "fingering": False}
        canonical["targets"]["gestures"] = [gesture]
        self.assertEqual(percussion_annotation_coverage(canonical, candidate, normalization), [[0., .9], [1.1, 6.]])

    def test_unknown_grace_or_explicit_onset_mask_censors_gesture_measure(self):
        for change in ("unknown", "grace", "onset-mask", "attack-mask"):
            canonical, candidate, normalization = coverage_fixture()
            gesture = known_gesture()
            if change == "unknown":
                gesture["scoreOnsetKnown"] = False
            elif change == "grace":
                gesture["graceMode"] = "BeforeBeat"
            else:
                gesture["labelMask"] = {"gesture": True, change.removesuffix("-mask"): False}
            canonical["targets"]["gestures"] = [gesture]
            with self.subTest(change=change):
                self.assertEqual(percussion_annotation_coverage(canonical, candidate, normalization), [[3.1, 6.]])

    def test_unknown_attack_at_barline_censors_both_surrounding_measures(self):
        canonical, candidate, normalization = coverage_fixture()
        gesture = known_gesture()
        gesture.update(onsetQuarter=[3, 1], visitIndex=1, writtenBeatId="m1:v0:b0", graceMode="BeforeBeat", scoreOnsetKnown=False)
        canonical["targets"]["gestures"] = [gesture]
        self.assertEqual(percussion_annotation_coverage(canonical, candidate, normalization), [])

    def test_unknown_text_uses_normalized_beat_context_not_source_provenance(self):
        canonical, candidate, normalization = coverage_fixture()
        canonical["targets"]["notes"] = [unknown_symbol()]
        canonical["review"]["unresolvedGestures"] = [{
            "performanceBeatId": "p0:m0:v0:b1", "sourcePerformanceBeatId": "p9:m7:v0:b1",
            "onsetQuarter": [1, 1], "reason": "uninterpreted_annotation",
        }]
        self.assertEqual(percussion_annotation_coverage(canonical, candidate, normalization), [[0., .9], [1.1, 6.]])
        canonical["targets"]["notes"][0]["sourceSegments"][0]["graceMode"] = "BeforeBeat"
        self.assertEqual(percussion_annotation_coverage(canonical, candidate, normalization), [[3.1, 6.]])

    def test_explicit_unknown_timing_overrides_regular_linked_symbol(self):
        canonical, candidate, normalization = coverage_fixture()
        symbol = unknown_symbol()
        canonical["review"]["notationSymbols"] = [symbol]
        canonical["review"]["unresolvedGestures"] = [{
            "symbolicNoteIds": [symbol["id"]], "onsetQuarter": [1, 1], "scoreOnsetKnown": False,
        }]
        self.assertEqual(percussion_annotation_coverage(canonical, candidate, normalization), [[3.1, 6.]])

    def test_unlocatable_uncertainty_censors_everything_but_linked_measure_is_bounded(self):
        canonical, candidate, normalization = coverage_fixture()
        canonical["review"]["unresolvedGestures"] = [{"reason": "unknown_source_position"}]
        self.assertEqual(percussion_annotation_coverage(canonical, candidate, normalization), [])
        symbol = unknown_symbol()
        canonical["review"]["notationSymbols"] = [symbol]
        canonical["review"]["unresolvedGestures"][0]["symbolicNoteIds"] = [symbol["id"]]
        self.assertEqual(percussion_annotation_coverage(canonical, candidate, normalization), [[3.1, 6.]])
        canonical["review"]["unresolvedGestures"] = [{"performanceBeatId": "p1:m1:v0:b0"}]
        self.assertEqual(percussion_annotation_coverage(canonical, candidate, normalization), [[0., .9], [1.1, 2.9]])

    def test_partial_mapping_clips_measure_holes_without_extrapolation(self):
        canonical, candidate, normalization = coverage_fixture()
        candidate["denseMapping"] = [
            {"referenceSeconds": float(t), "clipSeconds": 8. + 2 * t, "scoreQuarter": float(t)}
            for t in (1, 3, 5)
        ]
        symbol = unknown_symbol()
        symbol["isAttack"] = None
        canonical["review"]["notationSymbols"] = [symbol]
        self.assertEqual(percussion_annotation_coverage(canonical, candidate, normalization), [[14.1, 18.]])
        symbol["isAttack"], symbol["onsetQuarter"] = True, [0, 1]
        self.assertEqual(percussion_annotation_coverage(canonical, candidate, normalization), [[10., 18.]])

    def test_plateau_has_guarded_hole_and_collapsed_mapping_has_no_coverage(self):
        canonical, candidate, normalization = coverage_fixture()
        candidate["denseMapping"] = [
            {"referenceSeconds": reference, "clipSeconds": clip, "scoreQuarter": reference}
            for reference, clip in ((0., 0.), (1., 1.), (2., 1.), (6., 5.))
        ]
        self.assertEqual(percussion_annotation_coverage(canonical, candidate, normalization), [[0., .9], [1.1, 5.]])
        for point in candidate["denseMapping"]:
            point["clipSeconds"] = 1.
        self.assertEqual(percussion_annotation_coverage(canonical, candidate, normalization), [])

    def test_overlapping_holes_are_sorted_merged_and_bounded(self):
        canonical, candidate, normalization = coverage_fixture()
        canonical["review"]["unresolvedGestures"] = [
            {"onsetQuarter": onset, "scoreOnsetKnown": True}
            for onset in ([6, 1], [1, 1], [21, 20], [0, 1], [4, 1])
        ]
        self.assertEqual(percussion_annotation_coverage(canonical, candidate, normalization), [[.1, .9], [1.05 + .1, 3.9], [4.1, 5.9]])

    def test_nominal_tempo_change_and_candidate_warp_are_used(self):
        canonical, candidate, normalization = coverage_fixture()
        normalization["normalizedTempoEvents"].append({
            "measureIndex": 1, "positionRatio": [0, 1], "offsetQuarter": [0, 1],
            "bpm": 120, "beatUnit": [1, 4], "quarterBpm": [120, 1], "linear": False,
        })
        candidate["denseMapping"] = [
            {"referenceSeconds": 0., "clipSeconds": 2., "scoreQuarter": 0.},
            {"referenceSeconds": 4.5, "clipSeconds": 11., "scoreQuarter": 6.},
        ]
        canonical["review"]["unresolvedGestures"] = [{"onsetQuarter": [4, 1], "scoreOnsetKnown": True}]
        self.assertEqual(percussion_annotation_coverage(canonical, candidate, normalization), [[2., 8.9], [9.1, 11.]])

    def test_nonpercussion_uncertainty_does_not_claim_harmonic_completeness(self):
        canonical, candidate, normalization = coverage_fixture()
        gesture = known_gesture()
        gesture.update(technique="harmonic", scoreOnsetKnown=False, labelMask={"gesture": False})
        canonical["targets"]["gestures"] = [gesture]
        self.assertEqual(percussion_annotation_coverage(canonical, candidate, normalization), [[0., 6.]])

    def test_invalid_mapping_or_score_clock_is_not_silently_accepted(self):
        for field, value in (("clipSeconds", -1.), ("clipSeconds", float("nan")), ("referenceSeconds", 0.), ("referenceSeconds", 7.)):
            canonical, candidate, normalization = coverage_fixture()
            candidate["denseMapping"][1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(AlignmentInputError):
                percussion_annotation_coverage(canonical, candidate, normalization)
        canonical, candidate, normalization = coverage_fixture()
        canonical["scoreTimingResolved"] = False
        with self.assertRaises(AlignmentInputError):
            percussion_annotation_coverage(canonical, candidate, normalization)


if __name__ == "__main__":
    unittest.main()
