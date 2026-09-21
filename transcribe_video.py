"""Transcribe a local video using musical settings supplied on the command line."""

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from scripts import transcriber, transcription_pipeline
from scripts.dataset_io import ROOT, publish_json, read_json
from scripts.gp_events import EventExtractionError, TEMPO_BEAT_UNITS
from scripts.prepare_training_data import regular_path
from scripts.transcriber import log_progress
from scripts.transcriber_audio import HarnessError


PROFILE_FLAGS = {"strict-note-confidence", "include-harmonics"}
PROFILE_VALUES = {
    "draft-percussion-threshold", "draft-harmonic-threshold",
    "brush-threshold", "arpeggio-threshold", "pick-stroke-threshold", "rasgueado-threshold",
    "brush-membership-threshold", "arpeggio-membership-threshold",
    "pick-stroke-membership-threshold", "rasgueado-membership-threshold",
    "connection-threshold", "note-technique-threshold", "grace-threshold",
    "chord-tolerance", "same-string-gap", "rhythm-policy",
}


def pitch(value):
    try:
        midi = int(value)
    except ValueError:
        match = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", value)
        if not match:
            raise argparse.ArgumentTypeError("Use a MIDI number or a note with octave, e.g. 38 or D2.") from None
        note, accidental, octave = match.groups()
        midi = 12 * (int(octave) + 1) + {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[note.upper()]
        midi += {"": 0, "#": 1, "b": -1}[accidental]
    if not 0 <= midi <= 127:
        raise argparse.ArgumentTypeError("String pitches must be within MIDI 0..127.")
    return midi


def ratio(value):
    try:
        numerator, denominator = map(int, value.split("/"))
        if numerator <= 0 or denominator <= 0:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError("Use a positive numerator/denominator, e.g. 4/4 or 3/8.") from None
    return [numerator, denominator]


def cutoff(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("Cutoffs must be finite numbers from zero through one.")
    return value


def argument_parser():
    parser = argparse.ArgumentParser(description="Local video to Guitar Pro with explicit fixed tuning, full capo, tempo and meter. No metadata sidecar, trimming, training or automatic GP opening.")
    parser.add_argument("--video", required=True, help="Path to the already-trimmed local performance video.")
    parser.add_argument("--note-cutoff", type=cutoff, required=True)
    parser.add_argument("--x-cutoff", type=cutoff, required=True, help="Thumb-slap X cutoff, separate from other percussion.")
    parser.add_argument("--output-directory", required=True, help="Dedicated job folder under this repository's runs directory.")
    parser.add_argument("--tuning", nargs=6, type=pitch, required=True, metavar="PITCH", help="Six pre-capo pitches in physical string 6-to-1 order, e.g. E2 A2 D3 G3 B3 E4 or 40 45 50 55 59 64. C4 is MIDI 60.")
    parser.add_argument("--capo", type=int, required=True, help="Fixed full capo fret; use 0 for none.")
    parser.add_argument("--bpm", type=float, required=True)
    parser.add_argument("--beat-unit", type=ratio, required=True, help="BPM unit as a fraction of a whole note: 1/4 for quarter notes, 3/8 for dotted quarters.")
    parser.add_argument("--time-signature", type=ratio, required=True, help="Meter, e.g. 4/4 or 6/8.")
    parser.add_argument("--plucking-screen-side", choices=("geometry", "left", "right"), help="Override the hand-side setting in the local preset for this video.")
    parser.add_argument("--settings", default=str(ROOT / "runs" / "transcription-settings.json"), help="Existing local model/template/export preset.")
    parser.add_argument("--dry-run", action="store_true", help="Preview validated arguments without writing files or running conversion.")
    return parser


def prepare(args):
    log_progress("Setup: validating command-line musical settings and local paths...")
    metadata = {
        "openStringMidi": args.tuning, "capoFret": args.capo,
        "tempo": {"bpm": args.bpm, "beatUnit": args.beat_unit},
        "timeSignature": args.time_signature, "tempoChanges": [], "timeSignatureChanges": [],
    }
    transcriber.inference_metadata(metadata)
    if Fraction(*args.beat_unit) not in TEMPO_BEAT_UNITS.values():
        raise HarnessError("GP export supports beat units " + ", ".join(map(str, TEMPO_BEAT_UNITS.values())) + ".")
    video = regular_path(args.video)
    output = regular_path(args.output_directory)
    settings_path = regular_path(args.settings)
    settings = read_json(settings_path)
    required = {"schemaVersion", "checkpoint", "template", "beatCheckpoint", "device", "beatDevice", "pluckingScreenSide", "exportProfile"}
    if not isinstance(settings, dict) or not required <= settings.keys() or settings.keys() - required or type(settings["schemaVersion"]) is not int or settings["schemaVersion"] != 1:
        raise HarnessError("Expected version-1 model settings with checkpoint, template, beatCheckpoint, device, beatDevice, pluckingScreenSide and exportProfile.")
    paths = {"video": video}
    for key, option in (("checkpoint", "checkpoint"), ("template", "template"), ("beatCheckpoint", "beat-checkpoint")):
        if not isinstance(settings[key], str) or not settings[key]:
            raise HarnessError(f"Model setting {key} must be a file path.")
        path = Path(settings[key])
        paths[option] = regular_path(path if path.is_absolute() else ROOT / path)
    for path in paths.values():
        if not path.is_file():
            raise HarnessError(f"Required local input is missing: {path}")
        if path.is_relative_to(output):
            raise HarnessError("Source files must be outside the transcription output directory.")
    # Keep the generated metadata outside the job: the shared pipeline requires immutable external inputs.
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
    metadata_path = regular_path(ROOT / "runs" / "transcription-metadata" / f"{digest}.json")
    if metadata_path.is_relative_to(output):
        raise HarnessError("Output directory must not contain the launcher's private metadata cache.")
    values = ["transcribe", "--data-root", str(ROOT), "--metadata", str(metadata_path), "--output-directory", str(output)]
    for option, path in paths.items():
        values.extend(("--" + option, str(path)))
    for key, option in (("device", "device"), ("beatDevice", "beat-device"), ("pluckingScreenSide", "plucking-screen-side")):
        if not isinstance(settings[key], str) or not settings[key]:
            raise HarnessError(f"Model setting {key} must be a nonempty string.")
        value = (args.plucking_screen_side or settings[key]) if key == "pluckingScreenSide" else settings[key]
        values.extend(("--" + option, value))
    profile = settings["exportProfile"]
    if not isinstance(profile, dict) or profile.keys() - PROFILE_FLAGS - PROFILE_VALUES:
        raise HarnessError("Unsupported preset option in exportProfile; note/X cutoffs and input/output paths must come from the command.")
    for name, value in profile.items():
        if name in PROFILE_FLAGS:
            if type(value) is not bool:
                raise HarnessError(f"Preset option {name} must be a boolean.")
            if value:
                values.append("--" + name)
        elif value is not None:
            if type(value) not in (str, int, float):
                raise HarnessError(f"Invalid preset value for {name}.")
            values.extend(("--" + name, str(value)))
    values.extend(("--draft-note-threshold", str(args.note_cutoff), "--thumb-slap-threshold", str(args.x_cutoff)))
    pipeline_args = transcriber.argument_parser().parse_args(values)
    transcription_pipeline._directory(pipeline_args)
    transcriber.draft_cli_values(pipeline_args)
    log_progress("Setup: musical settings, model settings and required assets validated.")
    return pipeline_args, metadata, values


def main(argv=None):
    args = argument_parser().parse_args(argv)
    try:
        pipeline_args, metadata, values = prepare(args)
        if args.dry_run:
            log_progress("Preview only: no files written or conversion started.")
            print(json.dumps({"metadata": metadata, "arguments": values}, indent=2), flush=True)
            return 0
        metadata_path = regular_path(pipeline_args.metadata)
        if metadata_path.exists():
            if read_json(metadata_path) != metadata:
                raise HarnessError(f"Saved launcher metadata changed; refusing to overwrite: {metadata_path}")
            log_progress(f"Setup: reusing saved command-line metadata: {metadata_path}")
        else:
            publish_json(metadata_path, metadata)
            log_progress(f"Setup: command-line metadata saved automatically: {metadata_path}")
        log_progress(f"Transcription pipeline: starting. Note cutoff={args.note_cutoff}; X cutoff={args.x_cutoff}; output={pipeline_args.output_directory}")
        result = transcription_pipeline.run_transcription(pipeline_args)
        if result["status"] != "ready":
            log_progress(f"Transcription pipeline: {result['status']}; action required.")
            print(json.dumps(result["actions"], indent=2), flush=True)
            return 1
        log_progress("Transcription complete: both GP files saved. No application will be opened.")
        print(f"Single-voice GP: {result['outputs']['singleVoice']}", flush=True)
        print(f"Full-voice GP: {result['outputs']['fullVoices']}", flush=True)
        print(f"Report: {result['report']}", flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, EventExtractionError) as error:
        print(f"Transcription failed: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    raise SystemExit(main())
