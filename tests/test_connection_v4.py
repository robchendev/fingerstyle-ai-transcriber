"""Corrected supervision uses synthetic score provenance, never private media."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from scripts.connection_supervision import (
    CONNECTION_TYPES, GRACE_MODES, RELATION_TYPES, SUPERVISION_VERSION,
    V4_NOTE_TECHNIQUE_TYPES, canonical_connections, connections_in_window,
    projected_connections,
)
from scripts.transcriber_audio import HarnessError
from scripts.transcriber_data import encode_targets
from scripts.transcriber_events import (
    OutputTimeline, checkpoint_event_score, evaluate_events, score_note_attributes, summarize_counts,
)
from scripts.transcriber_model import (
    FingerstyleTranscriber, LOSS_WEIGHTS, ModelConfig, decode_events,
    initialize_v4_from_model, masked_loss,
)
from scripts.transcriber_runtime import _objective, _stats, evaluate_model
from tests.test_connection_supervision import note
from tests.test_transcriber_events import WindowFixture
from tests.test_transcriber_model import choose_category, event_outputs, synthetic_outputs


def source(identifier, onset, fret, *, flags=0, grace=None, voice=0, duration=1, hopo=False):
    value = note(identifier, onset, 64 + fret, techniques={"slideFlags": flags, "hopoDestination": hopo})
    value["fret"], value["voiceIndex"] = fret, voice
    value["notatedDurationQuarter"] = [duration, 1] if grace is None else None
    value["sourceSegments"][0].update(
        graceMode=grace, onsetQuarter=[onset, 1],
        durationQuarter=[duration, 1] if grace is None else None,
        fret=fret, soundingPitchMidi=64 + fret, basePitchMidi=64 + fret,
        tie={"origin": False, "destination": False},
    )
    return value


def labels(*notes):
    return {"targets": {"notes": list(notes), "gestures": []}}


def projected(data):
    return projected_connections(
        data, {"denseMapping": [
            {"referenceSeconds": 0., "clipSeconds": 0.},
            {"referenceSeconds": 8., "clipSeconds": 8.},
        ]}, SimpleNamespace(seconds=float), architecture_version=4,
    )


def encoded(data, *, length=3):
    times = np.arange(length * 50) * .02
    events = projected(data)
    notes = [{
        "sourceNoteId": item["id"], "string": item["string"], "fret": item["fret"],
        "soundingPitchMidi": item["soundingPitchMidi"], "voiceIndex": item["voiceIndex"],
        "onsetWindowSeconds": float(item["onsetQuarter"][0] / item["onsetQuarter"][1]),
        "notatedDurationQuarter": item["notatedDurationQuarter"], "sourceLabelMask": item["labelMask"],
        "supervisionMask": {"onset": True, "pitch": True, "fingering": True, "notatedDuration": True},
    } for item in data["targets"]["notes"] if item["sourceSegments"][0]["graceMode"] is None]
    window = {"targets": {
        "notes": notes, "gestures": [], "techniques": [],
        "connections": connections_in_window(events, 0, length * 100, 100),
    }}
    return encode_targets(window, data, times, ModelConfig(architecture_version=4), negative_onsets_allowed=[True] * 6)


def prediction(event):
    result = {
        "onsetSeconds": event["proposedOnsetClipSeconds"], "string": event["string"],
        "voiceIndex": event["voiceIndex"], "soundingPitchMidi": event["soundingPitchMidi"],
        "fret": event["fret"], "connection": event["connection"],
        "connectionConfidence": 1., "noteTechniques": {key: float(value) for key, value in event["techniques"].items()},
        "grace": deepcopy(event["grace"]), "connectionOrigin": None,
    }
    if event["origin"] is not None and event["connection"] != "none":
        result["connectionOrigin"] = {
            **event["origin"], "onsetSeconds": event["origin"]["proposedOnsetClipSeconds"],
        }
    return result


class CorrectedSupervisionTests(unittest.TestCase):
    def test_terminal_departure_and_combined_slide_flags_stay_on_source(self):
        data = labels(source("a", 0, 3, flags=20), source("b", 1, 5, flags=8), source("c", 2, 7, flags=4))
        events = canonical_connections(data, architecture_version=4)
        self.assertEqual([item["connection"] for item in events], ["none"] * 3)
        self.assertTrue(events[0]["techniques"]["slide_out_down"])
        self.assertTrue(events[0]["techniques"]["slide_in_below"])
        self.assertTrue(events[1]["techniques"]["slide_out_up"])
        self.assertTrue(events[-1]["techniques"]["slide_out_down"])
        target, masks, _ = encoded(data)
        axis = V4_NOTE_TECHNIQUE_TYPES.index("slide_out_down")
        self.assertEqual(target["note_technique"][100, 5, axis], 1)
        self.assertTrue(masks["note_technique"][100, 5, axis])

    def test_grace_legato_slide_is_main_note_anchored_without_an_audio_onset(self):
        data = labels(source("main", 1, 7), source("grace", 1, 3, flags=2, grace="OnBeat"))
        events = canonical_connections(data, architecture_version=4)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["connection"], "none")
        self.assertEqual(event["grace"]["transition"], "slide_2")
        self.assertEqual(event["grace"]["sourceFret"], 3)
        self.assertEqual(event["grace"]["intervalSemitones"], 4)
        self.assertIsNone(event["grace"]["onsetSeconds"])
        self.assertFalse(event["grace"]["timingKnown"])
        target, masks, collisions = encoded(data)
        self.assertEqual(collisions, 0)
        self.assertEqual(target["note_onset"].sum(), 1)
        self.assertEqual(target["grace"][50, 5], 1)
        self.assertEqual(target["grace_transition"][50, 5], RELATION_TYPES.index("slide_2"))
        self.assertEqual(target["grace_mode"][50, 5], GRACE_MODES.index("OnBeat"))
        self.assertEqual(target["grace_fret"][50, 5], 3)
        self.assertTrue(masks["grace_fret"][50, 5])
        self.assertEqual(masks["grace_fret"].sum(), 1)

    def test_relations_use_chronology_same_voice_and_no_gaps(self):
        data = labels(
            source("dest", 1, 7), source("other-voice", 0, 9, voice=1),
            source("origin", 0, 3, flags=2),
        )
        data["targets"]["notes"][1]["onsetQuarter"] = [1, 2]
        events = canonical_connections(data, architecture_version=4)
        destination = next(event for event in events if event["sourceNoteId"] == "dest")
        self.assertEqual(destination["connection"], "slide_2")
        self.assertEqual(destination["priorSourceNoteId"], "origin")
        data["targets"]["notes"][0]["onsetQuarter"] = [8, 1]
        destination = canonical_connections(data, architecture_version=4)[-1]
        self.assertFalse(destination["connectionMask"])

    def test_destination_hopo_and_outgoing_slide_conflict_is_unknown(self):
        data = labels(source("a", 0, 3, flags=2), source("b", 1, 7, hopo=True, flags=12))
        event = canonical_connections(data, architecture_version=4)[1]
        self.assertFalse(event["connectionMask"])
        self.assertFalse(event["techniqueMasks"]["slide_out_down"])
        self.assertFalse(event["techniqueMasks"]["slide_out_up"])
        self.assertTrue(event["techniqueMasks"]["tap"])

    def test_tie_chain_uses_its_final_segment_and_logical_duration(self):
        origin = source("origin", 0, 3, duration=2)
        origin["sourceSegments"].append(deepcopy(source("continuation", 1, 3, flags=2)["sourceSegments"][0]))
        origin["sourceSegments"][1]["tie"]["destination"] = True
        event = canonical_connections(labels(origin, source("dest", 2, 7)), architecture_version=4)[1]
        self.assertTrue(event["connectionMask"])
        self.assertEqual(event["connection"], "slide_2")
        self.assertEqual(event["priorSourceNoteId"], "origin")

    def test_unpitched_and_unknown_notes_are_adjacency_barriers(self):
        dead = source("dead", 1, 0)
        dead["sourceSegments"][0]["techniques"]["dead"] = True
        dead["labelMask"]["pitch"] = False
        data = labels(source("a", 0, 3, flags=2), source("b", 2, 7, hopo=True))
        data["review"] = {"notationSymbols": [dead]}
        event = canonical_connections(data, architecture_version=4)[1]
        self.assertFalse(event["connectionMask"])
        self.assertNotEqual(event["priorSourceNoteId"], "a")

    def test_unknown_grace_attributes_and_multiple_graces_are_masked(self):
        data = labels(source("g", 1, 3, grace="Unknown"), source("main", 1, 7))
        event = canonical_connections(data, architecture_version=4)[0]
        self.assertTrue(event["graceMask"])
        self.assertFalse(event["graceAttributeMasks"]["mode"])
        data["targets"]["notes"].append(source("second-grace", 1, 5, grace="OnBeat"))
        target, masks, _ = encoded(data)
        self.assertFalse(masks["grace"].any())
        self.assertFalse(masks["grace_transition"].any())
        self.assertFalse(masks["connection"].any())
        self.assertEqual(target["note_onset"].sum(), 1)

    def test_cross_voice_or_quantized_frame_collision_masks_all_attributes(self):
        for second in (source("b", 1, 7, voice=1), source("b", 1, 7)):
            data = labels(source("a", 1, 3, flags=4), second)
            target, masks, collisions = encoded(data)
            self.assertEqual(collisions, 1)
            self.assertTrue(masks["note_onset"][50, 5])
            for name in ("connection", "note_technique", "bend_curve", "grace", "grace_fret"):
                self.assertFalse(masks[name].any(), name)

    def test_origin_outside_window_is_censored_without_losing_local_effect(self):
        data = labels(source("a", 0, 3, flags=2), source("b", 1, 7, flags=4))
        event = connections_in_window(projected(data), 50, 250, 100)[0]
        self.assertFalse(event["supervisionMask"]["connection"])
        self.assertTrue(event["supervisionMask"]["techniques"])
        self.assertTrue(event["techniques"]["slide_out_down"])

    def test_unknown_flags_and_flat_audio_mapping_do_not_create_relations(self):
        data = labels(source("a", 0, 3, flags=64), source("b", 1, 7))
        event = canonical_connections(data, architecture_version=4)[1]
        self.assertFalse(event["connectionMask"])
        data["targets"]["notes"][0]["sourceSegments"][0]["techniques"]["slideFlags"] = 2
        events = projected_connections(data, {"denseMapping": [
            {"referenceSeconds": 0., "clipSeconds": 0.},
            {"referenceSeconds": 2., "clipSeconds": 0.},
        ]}, SimpleNamespace(seconds=float), architecture_version=4)
        self.assertFalse(events[1]["connectionMask"])

    def test_origin_quantization_collision_censors_relation_target(self):
        data = labels(source("a", 0, 3, flags=2), source("other", 0, 5, voice=1), source("b", 1, 7))
        data["targets"]["notes"][1]["onsetQuarter"] = [1, 200]
        targets, masks, collisions = encoded(data)
        self.assertEqual(collisions, 1)
        self.assertFalse(masks["connection"][50, 5])

    def test_harmonic_grace_pitch_is_not_treated_as_a_plain_fret_target(self):
        data = labels(source("g", 1, 3, grace="OnBeat"), source("main", 1, 7))
        data["targets"]["notes"][0]["soundingPitchMidi"] += 12
        event = canonical_connections(data, architecture_version=4)[0]
        self.assertTrue(event["graceMask"])
        self.assertFalse(event["graceAttributeMasks"]["fret"])
        self.assertFalse(event["graceAttributeMasks"]["transition"])

    def test_old_v3_supervision_still_uses_original_slide_categories(self):
        data = labels(source("a", 0, 3, flags=4), source("b", 1, 7))
        self.assertEqual(canonical_connections(data)[1]["connection"], "slide_4")
        self.assertEqual(canonical_connections(data, architecture_version=4)[1]["connection"], "none")


class CorrectedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_v4_heads_losses_masks_and_runtime_calibration_are_consistent(self):
        data = labels(source("g", 1, 3, flags=2, grace="OnBeat"), source("a", 1, 7), source("b", 2, 5, flags=4))
        target, masks, _ = encoded(data)
        targets, masks = ({name: value[None] for name, value in group.items()} for group in (target, masks))
        config = ModelConfig(architecture_version=4, hidden_size=8, recurrent_layers=1, dropout=0)
        model = FingerstyleTranscriber(config)
        batch = {
            "features": torch.zeros(1, 150, 96), "conditioning": torch.zeros(1, 150, 12),
            "lengths": torch.tensor([150]), "valid_frames": torch.ones(1, 150, dtype=torch.bool),
            "targets": targets, "masks": masks,
        }
        with torch.no_grad():
            outputs = model(batch["features"], batch["conditioning"], batch["lengths"])
            loss, stats = masked_loss(outputs, targets, masks, batch["valid_frames"])
        self.assertEqual(outputs["connection_logits"].shape, (1, 150, 6, 5))
        self.assertEqual(outputs["note_technique_logits"].shape, (1, 150, 6, 8))
        self.assertEqual(stats["grace_positive"]["count"], 1)
        self.assertEqual(stats["grace_negative"]["count"], 1)
        self.assertEqual(stats["grace_fret"]["count"], 1)
        self.assertAlmostEqual(float(loss), _objective(_stats(stats, LOSS_WEIGHTS), LOSS_WEIGHTS, .02), places=5)
        report = evaluate_model(model, [batch], "cpu")
        self.assertEqual(report["grace_fret_accuracy"]["count"], 1)
        calibration = report["technique_calibration"]["grace"]
        self.assertIsNotNone(calibration["bestThreshold"])
        self.assertEqual(len(calibration["thresholds"]), 4)
        self.assertTrue(all(item["true_positive"] + item["false_negative"] == 1 for item in calibration["thresholds"]))

    def test_v2_v3_transfer_explicitly_resets_incompatible_supervision(self):
        for version in (2, 3):
            config = ModelConfig(architecture_version=version, hidden_size=8, recurrent_layers=1)
            old = FingerstyleTranscriber(config)
            new = FingerstyleTranscriber(replace(config, architecture_version=4))
            original = deepcopy(new.state_dict())
            report = initialize_v4_from_model(new, old)
            self.assertEqual(report["sourceArchitectureVersion"], version)
            self.assertFalse(report["optimizerTransferred"])
            for name, value in new.state_dict().items():
                expected = old.state_dict()[name] if name in report["copiedParameters"] else original[name]
                self.assertTrue(torch.equal(value, expected), name)
            with self.assertRaises(ValueError):
                initialize_v4_from_model(old, new)
            new.load_state_dict(new.state_dict(), strict=True)
            old.load_state_dict(old.state_dict(), strict=True)

    def test_v3_logits_are_not_reinterpreted_and_malformed_v4_is_rejected(self):
        outputs = event_outputs(2, architecture_version=3)
        outputs["note_onset_logits"][1, 5] = 12
        choose_category(outputs, "connection_logits", 1, 5, CONNECTION_TYPES.index("slide_4"))
        result = decode_events(outputs, [0., 1.], tuning=[40, 45, 50, 55, 59, 64], capo=0)
        self.assertEqual(result["notes"][0]["connection"], "slide_4")
        self.assertNotIn("techniqueSchemaVersion", result["notes"][0])
        outputs["grace_logits"] = torch.zeros(2, 6)
        with self.assertRaises(ValueError):
            decode_events(outputs, [0., 1.], tuning=[40, 45, 50, 55, 59, 64], capo=0)

    def test_multi_window_reconstruction_resolves_origin_once(self):
        outputs = event_outputs(150, architecture_version=4)
        for frame, fret in ((40, 3), (90, 7)):
            outputs["note_onset_logits"][frame, 5] = 12
            choose_category(outputs, "fret_logits", frame, 5, fret)
            choose_category(outputs, "pitch_logits", frame, 5, 64 + fret)
        choose_category(outputs, "connection_logits", 90, 5, RELATION_TYPES.index("slide_2"))
        times = np.arange(150) * .02
        timeline = OutputTimeline(times)
        for start, stop in ((0, 70), (50, 150)):
            timeline.add({name: value[start:stop] for name, value in outputs.items()}, times[start:stop], stop * .02)
        result = decode_events(timeline.finish(), torch.tensor(times), tuning=[40, 45, 50, 55, 59, 64], capo=0)
        self.assertEqual(len(result["notes"]), 2)
        destination = result["notes"][1]
        self.assertEqual(destination["connection"], "slide_2")
        self.assertAlmostEqual(destination["connectionOrigin"]["onsetSeconds"], .8)
        self.assertEqual(destination["connectionOrigin"]["soundingPitchMidi"], 67)
        self.assertEqual(destination["techniqueSchemaVersion"], 4)
        self.assertEqual(destination["connectionOriginNoteId"], result["notes"][0]["noteId"])
        self.assertEqual(destination["connectionOrigin"]["noteId"], result["notes"][0]["noteId"])
        repeated = decode_events(timeline.finish(), torch.tensor(times), tuning=[40, 45, 50, 55, 59, 64], capo=0)
        self.assertEqual([note["noteId"] for note in result["notes"]], [note["noteId"] for note in repeated["notes"]])
        retained = deepcopy(result["notes"][1:])
        self.assertNotIn(retained[0]["connectionOriginNoteId"], {note["noteId"] for note in retained})

    def test_decoded_grace_is_not_an_independent_note(self):
        outputs = event_outputs(3, architecture_version=4)
        outputs["note_onset_logits"][1, 5] = 12
        outputs["grace_logits"][1, 5] = 12
        choose_category(outputs, "fret_logits", 1, 5, 7)
        choose_category(outputs, "pitch_logits", 1, 5, 71)
        choose_category(outputs, "grace_fret_logits", 1, 5, 3)
        choose_category(outputs, "grace_mode_logits", 1, 5, GRACE_MODES.index("OnBeat"))
        choose_category(outputs, "grace_transition_logits", 1, 5, RELATION_TYPES.index("slide_2"))
        result = decode_events(outputs, [0., 1., 2.], tuning=[40, 45, 50, 55, 59, 64], capo=0)
        self.assertEqual(len(result["notes"]), 1)
        grace = result["notes"][0]["grace"]
        self.assertEqual(grace["sourceFret"], 3)
        self.assertEqual(grace["intervalSemitones"], 4)
        self.assertEqual(grace["transition"], "slide_2")
        self.assertIsNone(grace["onsetSeconds"])
        self.assertFalse(grace["timingKnown"])
        self.assertEqual(grace["anchorNoteId"], result["notes"][0]["noteId"])

    def test_membership_completed_origin_preserves_parent_and_connection_reference(self):
        outputs = event_outputs(3, architecture_version=4)
        outputs["technique_logits"][1, 0] = 12
        outputs["technique_strings_logits"][1, 0, 5] = 12
        for frame, fret in ((1, 3), (2, 7)):
            choose_category(outputs, "fret_logits", frame, 5, fret)
            choose_category(outputs, "pitch_logits", frame, 5, 64 + fret)
        outputs["note_onset_logits"][2, 5] = 12
        choose_category(outputs, "connection_logits", 2, 5, RELATION_TYPES.index("slide_2"))
        decoded = decode_events(outputs, [0., 1., 2.], tuning=[40, 45, 50, 55, 59, 64], capo=0)
        origin, destination = decoded["notes"]
        self.assertEqual(origin["completionParent"], {
            "technique": decoded["techniques"][0]["technique"], "onsetSeconds": 1.,
        })
        self.assertEqual(destination["connectionOriginNoteId"], origin["noteId"])
        self.assertEqual(destination["connectionOrigin"]["onsetSeconds"], origin["completionParent"]["onsetSeconds"])
        self.assertNotIn("completionParent", destination)


class CorrectedEvaluationTests(unittest.TestCase):
    def test_evaluation_stitches_v4_windows_and_scores_non_none_techniques(self):
        class TimelineModel(torch.nn.Module):
            config = ModelConfig(architecture_version=4)

            def forward(self, features, conditioning, lengths):
                batch, frames, _ = features.shape
                outputs = synthetic_outputs(batch, frames, architecture_version=4)
                for name in ("note_onset_logits", "percussion_logits", "harmonic_logits", "technique_logits", "technique_strings_logits", "note_technique_logits", "grace_logits"):
                    outputs[name].fill_(-12)
                times = features[..., 0]
                outputs["note_onset_logits"][..., 5] = 12 - 100 * torch.minimum((times - 2).abs(), (times - 3).abs())
                for frame in range(frames):
                    for row in range(batch):
                        destination = times[row, frame] > 2.5
                        outputs["pitch_logits"][row, frame, 5, 71 if destination else 67] = 12
                        outputs["fret_logits"][row, frame, 5, 7 if destination else 3] = 12
                        outputs["connection_logits"][row, frame, 5, 4 if destination else 0] = 12
                        if destination:
                            outputs["note_technique_logits"][row, frame, 5, 4] = 12
                return outputs

        dataset = WindowFixture(True)
        data = labels(source("a", 2, 3, flags=2), source("b", 3, 7, flags=4))
        record = dataset.records[0]
        record["data"].labels["targets"] = data["targets"]
        record["connections"] = projected(data)
        report = evaluate_events(TimelineModel(), dataset, "cpu")
        metrics = report["metricsByToleranceSeconds"]["0.1"]
        self.assertEqual(len(report["recordings"][0]["predictions"]["notes"]), 2)
        self.assertEqual(metrics["relation_joint"]["true_positive"], 1)
        self.assertEqual(metrics["note_technique_slide_out_down"]["true_positive"], 1)
        self.assertEqual(checkpoint_event_score(report)["score"], 1.)

    def test_relation_requires_correct_origin_destination_pitch_voice_and_timing(self):
        events = projected(labels(source("a", 1, 3, flags=2), source("b", 2, 7)))
        guessed = [prediction(event) for event in events]
        metrics = score_note_attributes(events, guessed, [[0., 4.]], .1)
        self.assertEqual(metrics["relation_joint"]["true_positive"], 1)
        for field, value in (("soundingPitchMidi", 70), ("voiceIndex", 1), ("onsetSeconds", 1.5)):
            wrong = deepcopy(guessed)
            wrong[1]["connectionOrigin"][field] = value
            metric = score_note_attributes(events, wrong, [[0., 4.]], .1)["relation_joint"]
            self.assertEqual((metric["true_positive"], metric["false_positive"], metric["false_negative"]), (0, 1, 1))
        wrong = deepcopy(guessed)
        wrong[1]["soundingPitchMidi"] += 1
        self.assertEqual(score_note_attributes(events, wrong, [[0., 4.]], .1)["relation_joint"]["true_positive"], 0)

    def test_local_effect_and_grace_attributes_require_correct_main_pitch(self):
        events = projected(labels(source("g", 1, 3, flags=2, grace="OnBeat"), source("a", 1, 7, flags=4)))
        guessed = [prediction(event) for event in events]
        metric = score_note_attributes(events, guessed, [[0., 3.]], .1)
        self.assertEqual(metric["grace_attributes"]["true_positive"], 1)
        self.assertEqual(metric["note_technique_slide_out_down"]["true_positive"], 1)
        guessed[0]["grace"]["sourceFret"] = 4
        metric = score_note_attributes(events, guessed, [[0., 3.]], .1)
        self.assertEqual(metric["grace_presence"]["true_positive"], 1)
        self.assertEqual(metric["grace_attributes"]["true_positive"], 0)
        guessed[0]["soundingPitchMidi"] += 1
        metric = score_note_attributes(events, guessed, [[0., 3.]], .1)
        self.assertEqual(metric["grace_presence"]["true_positive"], 0)
        self.assertEqual(metric["note_technique_slide_out_down"]["true_positive"], 0)

    def test_unknown_attributes_and_outside_origin_are_not_false_positives(self):
        events = projected(labels(source("a", 0, 3, flags=2), source("b", 1, 7, flags=12)))
        guessed = [prediction(event) for event in events]
        metric = score_note_attributes(events, guessed, [[.5, 3.]], .1)
        self.assertEqual(metric["relation_joint"]["unscorable_predictions"], 1)
        self.assertEqual(metric["relation_joint"]["reference_events"], 0)
        self.assertEqual(metric["note_technique_slide_out_down"]["false_positive"], 0)
        self.assertFalse(metric["note_technique_slide_out_down"]["has_negative_coverage"])

    def test_canonical_cross_voice_string_collision_is_not_scored_as_known(self):
        events = projected(labels(source("a", 1, 3, flags=4), source("b", 1, 7, voice=1, flags=4)))
        metrics = score_note_attributes(events, [prediction(event) for event in events], [[0., 3.]], .1)
        self.assertEqual(metrics["note_technique_slide_out_down"]["reference_events"], 0)
        self.assertEqual(metrics["note_technique_slide_out_down"]["unscorable_predictions"], 2)

    def test_hallucinated_effect_is_false_positive_only_in_resolved_attack_coverage(self):
        events = projected(labels(source("a", 1, 3)))
        guessed = prediction(events[0])
        guessed["onsetSeconds"] = 2.
        guessed["noteTechniques"]["slide_out_down"] = 1.
        unknown = score_note_attributes(events, [guessed], [[0., 3.]], .1)
        covered = score_note_attributes(events, [guessed], [[0., 3.]], .1, negative_onsets_allowed=[True] * 6)
        self.assertEqual(unknown["note_technique_slide_out_down"]["unscorable_predictions"], 1)
        self.assertEqual(covered["note_technique_slide_out_down"]["false_positive"], 1)

    def test_checkpoint_score_cannot_hide_zero_non_none_success(self):
        events = projected(labels(source("a", 1, 3, flags=2), source("b", 2, 7)))
        predictions = [prediction(event) for event in events]
        counts = score_note_attributes(events, predictions, [[0., 4.]], .1)
        metrics = {key: summarize_counts(value) for key, value in counts.items()}
        base = {**metrics["relation_joint"], "true_positive": 1000, "reference_events": 1000, "f1": 1.}
        empty = {**base, "true_positive": 0, "reference_events": 0}
        metrics.update(string_fret_pitch_onset=base, wrist_thump=empty, thumb_slap=empty, percussive_hit=empty)
        report = {"techniqueSupervisionVersion": SUPERVISION_VERSION, "metricsByToleranceSeconds": {"0.1": metrics}, "windowVisits": 2}
        self.assertEqual(checkpoint_event_score(report)["score"], 1.)
        metrics["relation_slide_2"]["f1"] = 0.
        self.assertEqual(checkpoint_event_score(report)["score"], 0.)
        metrics["relation_slide_2"]["reference_events"] = 0
        with self.assertRaisesRegex(HarnessError, "non-none"):
            checkpoint_event_score(report)


if __name__ == "__main__":
    unittest.main()
