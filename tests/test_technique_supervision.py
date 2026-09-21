from types import SimpleNamespace
import unittest

import numpy as np

from scripts.technique_supervision import (
    TECHNIQUE_TYPES,
    canonical_techniques,
    projected_techniques,
    techniques_in_window,
)
from scripts.transcriber_data import encode_targets


def source_note(identifier, string, *, onset=(1, 1), voice=0, beat="m0:v0:b0", techniques=None):
    return {
        "id": identifier,
        "isAttack": True,
        "voiceIndex": voice,
        "string": string,
        "fret": 0,
        "soundingPitchMidi": 40 + string,
        "onsetQuarter": list(onset),
        "notatedDurationQuarter": [1, 1],
        "labelMask": {"attack": True, "pitch": True, "fingering": True, "notatedDuration": True},
        "sourceSegments": [{
            "visitIndex": 0,
            "writtenBeatId": beat,
            "graceMode": None,
            "beatTechniques": techniques or {},
            "harmonic": None,
        }],
    }


class TechniqueSupervisionTests(unittest.TestCase):
    def test_simultaneous_voices_merge_into_complete_strum_membership(self):
        labels = {"targets": {"notes": [
            source_note("upper-a", 1, techniques={"brush": "Down"}),
            source_note("upper-b", 2, techniques={"brush": "Down"}),
            source_note("bass", 6, voice=1, beat="m0:v1:b0"),
        ]}}
        events = canonical_techniques(labels)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["techniques"], ["brush"])
        self.assertEqual(events[0]["directions"], {"brush": "Down"})
        self.assertEqual(events[0]["stringsByTechnique"], {"brush": [1, 2, 6]})
        self.assertTrue(events[0]["membershipComplete"])

    def test_conflicting_direction_is_masked_and_native_types_stay_distinct(self):
        labels = {"targets": {"notes": [
            source_note("a", 1, techniques={"brush": "Down", "rasgueado": "ami_2"}),
            source_note("b", 2, voice=1, beat="m0:v1:b0", techniques={"brush": "Up"}),
        ]}}
        event = canonical_techniques(labels)[0]
        self.assertEqual(event["techniques"], ["brush", "rasgueado"])
        self.assertFalse(event["directionMasks"]["brush"])
        self.assertNotIn("rasgueado", event["directions"])

    def test_projection_window_and_v2_encoding_preserve_weak_member_strings(self):
        labels = {"targets": {
            "notes": [
                source_note("a", 1, techniques={"arpeggio": "Down"}),
                source_note("b", 6, voice=1, beat="m0:v1:b0"),
            ],
            "gestures": [],
        }}
        candidate = {"denseMapping": [
            {"referenceSeconds": 0.0, "clipSeconds": 0.0},
            {"referenceSeconds": 4.0, "clipSeconds": 4.0},
        ]}
        clock = SimpleNamespace(seconds=lambda quarter: float(quarter))
        projected = projected_techniques(labels, candidate, clock)
        local = techniques_in_window(projected, 0, 200, 100)
        self.assertEqual(local[0]["onsetWindowSeconds"], 1.0)
        projected_notes = [{
            "sourceNoteId": note["id"],
            "string": note["string"],
            "fret": note["fret"],
            "soundingPitchMidi": note["soundingPitchMidi"],
            "voiceIndex": note["voiceIndex"],
            "onsetWindowSeconds": 1.0,
            "notatedDurationQuarter": [1, 1],
            "sourceLabelMask": note["labelMask"],
            "supervisionMask": {"onset": True, "pitch": True, "fingering": True, "notatedDuration": True},
        } for note in labels["targets"]["notes"]]
        window = {"targets": {"notes": projected_notes, "gestures": [], "techniques": local}}
        model = SimpleNamespace(architecture_version=2, max_fret=36, max_voices=4)
        targets, masks, _ = encode_targets(window, labels, np.arange(100) * .02, model, negative_onsets_allowed=[True] * 6)
        axis = TECHNIQUE_TYPES.index("arpeggio")
        self.assertEqual(targets["technique"][50, axis], 1)
        self.assertTrue(masks["technique"][50, axis])
        self.assertTrue(masks["technique_strings"][50, axis].all())
        self.assertEqual(targets["technique_strings"][50, axis].tolist(), [1, 0, 0, 0, 0, 1])
        self.assertEqual(targets["technique_direction"][50, axis], 0)
        self.assertTrue(masks["technique_direction"][50, axis])
        self.assertTrue(masks["technique"][50].all())
        self.assertEqual(int(targets["technique"][50].sum()), 1)

    def test_nearby_same_class_strokes_restore_both_positive_masks(self):
        labels = {"targets": {"notes": [
            source_note("a", 1, onset=(1, 1), beat="m0:v0:b0", techniques={"brush": "Down"}),
            source_note("b", 2, onset=(26, 25), beat="m0:v0:b1", techniques={"brush": "Down"}),
        ], "gestures": []}}
        events = [
            {
                **event,
                "onsetWindowSeconds": seconds,
                "supervisionMask": {
                    "onset": True, "technique": True,
                    "direction": event["directionMasks"], "strings": True,
                },
            }
            for event, seconds in zip(canonical_techniques(labels), (.8, .84))
        ]
        notes = [{
            "sourceNoteId": note["id"], "string": note["string"], "fret": note["fret"],
            "soundingPitchMidi": note["soundingPitchMidi"], "voiceIndex": note["voiceIndex"],
            "onsetWindowSeconds": seconds, "notatedDurationQuarter": [1, 1],
            "sourceLabelMask": note["labelMask"],
            "supervisionMask": {"onset": True, "pitch": True, "fingering": True, "notatedDuration": True},
        } for note, seconds in zip(labels["targets"]["notes"], (.8, .84))]
        model = SimpleNamespace(architecture_version=2, max_fret=36, max_voices=4)
        targets, masks, _ = encode_targets(
            {"targets": {"notes": notes, "gestures": [], "techniques": events}},
            labels, np.arange(100) * .02, model, negative_onsets_allowed=[True] * 6,
        )
        axis = TECHNIQUE_TYPES.index("brush")
        self.assertEqual(targets["technique"][[40, 42], axis].tolist(), [1, 1])
        self.assertEqual(masks["technique"][[40, 42], axis].tolist(), [True, True])


if __name__ == "__main__":
    unittest.main()
