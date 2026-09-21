from copy import deepcopy
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import reexport_gp as launcher
from scripts.dataset_io import ROOT, publish_json, read_json, sha256
from scripts.transcriber import argument_parser, draft_cli_values


class GPReexportLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="reexport-test-", dir=ROOT / "runs")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "output"
        self.template = self.root / "template.gpt"
        self.template.write_bytes(b"synthetic template")
        options = {"output_directory": str(self.source), "template": str(self.template), "fingering_arranger": None,
                   "symbolic_completer": None, "draft_profile": draft_cli_values(argument_parser().parse_args([
                       "export-gp", "--predictions", "p", "--template", "t", "--full-output", "f",
                       "--single-output", "s", "--report", "r", "--thumb-slap-threshold", ".1",
                       "--strict-note-confidence", "--rhythm-policy", "fingerstyle", "--brush-threshold", ".995"]))}
        identity = {"template": {"path": str(self.template), "sha256": sha256(self.template)}}
        outputs, stages = {}, {}
        for stage, key in (("inference", "predictions"), ("beats", "beatEvidence")):
            p = self.source / f"{key}.json"
            publish_json(p, {"synthetic": True})
            outputs[key] = str(p)
            stages[stage] = {"status": "complete", "outputs": {key: str(p)}, "hashes": {str(p): sha256(p)}}
        summary = {"status": "ready", "outputDirectory": str(self.source), "inputIdentity": identity, "outputs": outputs}
        publish_json(self.source / "transcription.json", summary)
        self.state = {"schemaVersion": 1, "kind": "transcription-pipeline-state", "status": "ready", "options": options,
                      "identity": {"options": deepcopy(options), "inputs": identity}, "stages": stages,
                      "summarySha256": sha256(self.source / "transcription.json")}
        publish_json(self.source / "transcription-state.json", self.state)
        self.args = ["--source-run", str(self.source), "--output-directory", str(self.output),
                     "--note-cutoff", ".85", "--x-cutoff", ".15"]
        self.stdout = self.enterContext(patch("sys.stdout", new=StringIO()))
        self.stderr = self.enterContext(patch("sys.stderr", new=StringIO()))
        self.enterContext(patch("scripts.transcriber.infer", side_effect=AssertionError("No inference")))
        self.enterContext(patch("scripts.transcriber.analyze_beats", side_effect=AssertionError("No beat analysis")))

    def test_export_preserves_profile_and_sources_and_never_replaces_output(self):
        before = {str(p): sha256(p) for p in self.source.iterdir()}
        def export(args):
            self.assertEqual(args.draft_note_threshold, .85)
            self.assertEqual(args.thumb_slap_threshold, .15)
            self.assertEqual(args.brush_threshold, .995)
            self.assertTrue(args.strict_note_confidence)
            self.assertEqual(args.rhythm_policy, "fingerstyle")
            for name in ("full_output", "single_output"):
                Path(getattr(args, name)).write_bytes(b"synthetic GP")
            return {"fullVoices": {"path": args.full_output}, "singleVoice": {"path": args.single_output}}
        with patch.object(launcher.transcriber, "export_gp", side_effect=export) as dispatch:
            self.assertEqual(launcher.main(self.args), 0)
            self.assertEqual(launcher.main(self.args), 1)
            self.assertEqual(dispatch.call_count, 1)
        self.assertEqual(before, {str(p): sha256(p) for p in self.source.iterdir()})
        self.assertFalse(read_json(self.output / "reexport-receipt.json")["inferenceRerun"])

    def test_dry_run_and_invalid_inputs_do_not_dispatch(self):
        with patch.object(launcher.transcriber, "export_gp") as dispatch:
            self.assertEqual(launcher.main([*self.args, "--dry-run"]), 0)
            self.assertFalse(self.output.exists())
            for extra in (["--note-cutoff", ".4"], ["--x-cutoff", ".01"],
                          ["--output-directory", str(self.source)], ["--output-directory", str(self.root)]):
                self.assertEqual(launcher.main([*self.args, *extra]), 1)
            (self.source / "predictions.json").write_text("changed")
            self.assertEqual(launcher.main(self.args), 1)
            dispatch.assert_not_called()

    def test_failure_has_no_success_receipt(self):
        with patch.object(launcher.transcriber, "export_gp", side_effect=RuntimeError("export failed")):
            self.assertEqual(launcher.main(self.args), 1)
        self.assertIn("Re-export failed: export failed", self.stderr.getvalue())
        self.assertNotIn("Re-export completed", self.stdout.getvalue())
        self.assertFalse((self.output / "reexport-receipt.json").exists())
        self.assertFalse((self.output / ".transcription.lock").exists())
