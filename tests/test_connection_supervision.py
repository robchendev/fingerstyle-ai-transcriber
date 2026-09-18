from types import SimpleNamespace
import unittest

import numpy as np

from scripts.connection_supervision import CONNECTION_TYPES, canonical_connections, connections_in_window, projected_connections
from scripts.transcriber_data import encode_targets


def note(identifier, onset, pitch, *, string=1, techniques=None, bend=None):
    return {
        "id": identifier, "isAttack": True, "string": string, "fret": 0,
        "basePitchMidi": pitch, "soundingPitchMidi": pitch, "voiceIndex": 0,
        "onsetQuarter": [onset, 1], "notatedDurationQuarter": [1, 1],
        "labelMask": {"attack": True, "pitch": True, "fingering": True, "notatedDuration": True},
        "sourceSegments": [{
            "graceMode": None, "techniques": {
                "hopoDestination": False, "slideFlags": 0, "tapped": False,
                "leftHandTapped": False, "vibrato": None, **(techniques or {}),
            }, "bend": bend,
        }],
    }


class ConnectionSupervisionTests(unittest.TestCase):
    def test_hammer_pull_slide_and_note_techniques_are_relationship_aware(self):
        labels = {"targets": {"notes": [
            note("a", 0, 60, techniques={"slideFlags": 2}),
            note("b", 1, 64),
            note("c", 2, 67, techniques={"hopoDestination": True, "leftHandTapped": True}),
            note("d", 3, 62, techniques={"hopoDestination": True, "vibrato": "Slight"}, bend={
                "OriginOffset": 0.0, "OriginValue": 0.0, "MiddleOffset1": 12.0,
                "MiddleOffset2": 12.0, "MiddleValue": 12.0,
                "DestinationOffset": 99.0, "DestinationValue": 25.0,
            }),
        ]}}
        events = canonical_connections(labels)
        self.assertEqual([event["connection"] for event in events], ["none", "slide_2", "hammer_on", "pull_off"])
        self.assertTrue(events[2]["techniques"]["left_hand_tap"])
        self.assertTrue(events[3]["techniques"]["bend"])
        self.assertTrue(events[3]["techniques"]["vibrato"])
        self.assertEqual(events[3]["bendCurve"], [0.0, 0.0, 12.0, 12.0, 12.0, 99.0, 25.0])
        self.assertEqual(events[3]["priorSourceNoteId"], "c")

    def test_projection_window_and_v3_target_encoding(self):
        labels = {"targets": {
            "notes": [note("a", 0, 60), note("b", 1, 64, techniques={"hopoDestination": True, "tapped": True})],
            "gestures": [],
        }}
        candidate = {"denseMapping": [{"referenceSeconds": 0., "clipSeconds": 0.}, {"referenceSeconds": 4., "clipSeconds": 4.}]}
        projected = projected_connections(labels, candidate, SimpleNamespace(seconds=float))
        local = connections_in_window(projected, 0, 200, 100)
        notes = [{
            "sourceNoteId": source["id"], "string": source["string"], "fret": 0,
            "soundingPitchMidi": source["soundingPitchMidi"], "voiceIndex": 0,
            "onsetWindowSeconds": float(source["onsetQuarter"][0]),
            "notatedDurationQuarter": [1, 1], "sourceLabelMask": source["labelMask"],
            "supervisionMask": {"onset": True, "pitch": True, "fingering": True, "notatedDuration": True},
        } for source in labels["targets"]["notes"]]
        window = {"targets": {"notes": notes, "gestures": [], "techniques": [], "connections": local}}
        model = SimpleNamespace(architecture_version=3, max_fret=36, max_voices=4)
        targets, masks, _ = encode_targets(window, labels, np.arange(100) * .02, model, negative_onsets_allowed=[True] * 6)
        self.assertEqual(targets["connection"][50, 5], CONNECTION_TYPES.index("hammer_on"))
        self.assertTrue(masks["connection"][50, 5])
        self.assertEqual(targets["note_technique"][50, 5].tolist(), [0, 1, 0, 0])
        self.assertTrue(masks["note_technique"][50, 5].all())


if __name__ == "__main__":
    unittest.main()
