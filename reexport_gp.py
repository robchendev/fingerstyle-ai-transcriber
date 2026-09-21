"""Re-export a completed transcription with new note and thumb-slap cutoffs."""

import argparse
import json
from pathlib import Path
import sys

from scripts import transcriber
from scripts.dataset_io import ROOT, publish_json, read_json, sha256
from scripts.prepare_training_data import regular_path
from scripts.transcriber_audio import HarnessError
from scripts.transcription_pipeline import STATE_NAME, SUMMARY_NAME, _check_hashes, _directory, _lock
from transcribe_video import cutoff


def prepare(args):
    source = regular_path(args.source_run)
    output = regular_path(args.output_directory)
    _directory(argparse.Namespace(data_root=str(ROOT), output_directory=str(output)))
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise HarnessError("Choose a separate re-export output directory, not the source run or its parent/child.")
    if output.exists() and any(output.iterdir()):
        raise HarnessError("Re-export output directory must be new or empty; existing files are not replaced.")
    state_path = regular_path(source / STATE_NAME)
    summary_path = regular_path(source / SUMMARY_NAME)
    state = read_json(state_path)
    if (state.get("schemaVersion") != 1 or state.get("kind") != "transcription-pipeline-state"
            or state.get("status") != "ready" or state.get("options", {}).get("output_directory") != str(source)):
        raise HarnessError("Source must be a completed transcription job.")
    _check_hashes({str(summary_path): state["summarySha256"]})
    summary = read_json(summary_path)
    if (summary.get("status") != "ready" or summary.get("outputDirectory") != str(source)
            or summary.get("inputIdentity") != state["identity"]["inputs"]):
        raise HarnessError("Source summary and saved job identity disagree.")
    options = state["options"]
    if options != state["identity"]["options"]:
        raise HarnessError("Source options differ from the recorded identity.")
    hashes = {str(state_path): sha256(state_path), str(summary_path): state["summarySha256"]}
    paths = {}
    for stage, key in (("inference", "predictions"), ("beats", "beatEvidence")):
        saved = state["stages"][stage]
        path = regular_path(saved["outputs"][key])
        if saved["status"] != "complete" or not path.is_relative_to(source) or summary["outputs"][key] != str(path):
            raise HarnessError(f"Invalid saved {stage} output.")
        hashes[str(path)] = saved["hashes"][str(path)]
        paths[key] = path
    for name in ("template", "fingering_arranger", "symbolic_completer"):
        value = options.get(name)
        if value is not None:
            path = regular_path(value)
            identity = state["identity"]["inputs"][name]
            if identity["path"] != str(path) or path.is_relative_to(output):
                raise HarnessError(f"Unsafe or changed export input: {name}")
            paths[name] = path
            hashes[str(path)] = identity["sha256"]
    _check_hashes(hashes)
    profile = dict(options["draft_profile"])
    percussion_floor = min(.5, *(profile[key] for key in ("draft-percussion-threshold", "thumb-slap-threshold") if profile[key] is not None))
    if args.note_cutoff < .5 or args.x_cutoff < percussion_floor:
        raise HarnessError(f"Saved inference cannot recover missing candidates below note 0.5 or X {percussion_floor:g}; that requires new inference.")
    profile.update({"draft-note-threshold": args.note_cutoff, "thumb-slap-threshold": args.x_cutoff})
    values = [
        "export-gp", "--predictions", str(paths["predictions"]), "--beat-evidence", str(paths["beatEvidence"]),
        "--template", str(paths["template"]), "--full-output", str(output / "transcription.full-voices.gp"),
        "--single-output", str(output / "transcription.single-voice.gp"), "--report", str(output / "gp-output.json"),
    ]
    for key in ("fingering_arranger", "symbolic_completer"):
        if key in paths:
            values.extend(("--" + key.replace("_", "-"), str(paths[key])))
    for key, value in profile.items():
        if isinstance(value, bool):
            if value:
                values.append("--" + key)
        elif value is not None:
            values.extend(("--" + key, str(value)))
    parsed = transcriber.argument_parser().parse_args(values)
    transcriber.draft_cli_values(parsed)
    return output, parsed, values, hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--note-cutoff", type=cutoff, required=True)
    parser.add_argument("--x-cutoff", type=cutoff, required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        transcriber.log_progress("Re-export: checking saved predictions, beat timing and export settings...")
        output, export_args, values, hashes = prepare(args)
        if args.dry_run:
            print(json.dumps({"arguments": values, "inferenceRerun": False}, indent=2), flush=True)
            return 0
        output.mkdir(parents=True, exist_ok=True)
        with _lock(output):
            if any(path.name != ".transcription.lock" for path in output.iterdir()):
                raise HarnessError("Re-export output is no longer empty.")
            _check_hashes(hashes)
            transcriber.log_progress("Re-export: generating GP files only; no model, video or beat-analysis rerun.")
            report = transcriber.export_gp(export_args)
            _check_hashes(hashes)
            publish_json(output / "reexport-receipt.json", {
                "kind": "saved-prediction-gp-reexport", "sourceRun": str(regular_path(args.source_run)),
                "sourceHashes": hashes, "arguments": values, "noteCutoff": args.note_cutoff,
                "xCutoff": args.x_cutoff, "inferenceRerun": False, "trainingPerformed": False,
                "outputs": {key: report[key]["path"] for key in ("fullVoices", "singleVoice")},
            })
        transcriber.log_progress(f"Re-export completed: {output}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"Re-export failed: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
