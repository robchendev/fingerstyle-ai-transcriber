"""Local video/audio preparation and numerical hand inputs."""

import argparse
import json

from audio_sync import align_soundtrack
from core import EvidenceError
from hand_tracking import track_hands
from hand_roles import RoleConfig, assign_hand_roles
from paired_inputs import prepare_paired_inputs
from models import provision_hand_landmarker, provision_pose_landmarker
from shot_inspector import ShotConfig, inspect_shots, select_shots


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    for name, model in (("provision-hands", "hand_landmarker.task"), ("provision-pose", "pose_landmarker_lite.task")):
        provision = commands.add_parser(name)
        provision.add_argument("--output", default=f"runs\\video-evidence\\models\\{model}")

    inspect = commands.add_parser("inspect-shots")
    inspect.add_argument("--video", required=True)
    inspect.add_argument("--output-directory", required=True)
    inspect.add_argument("--cut-threshold", type=float, default=.28)
    inspect.add_argument("--min-shot-seconds", type=float, default=.20)
    inspect.add_argument("--review")

    selection = commands.add_parser("select-shots")
    selection.add_argument("--video", required=True)
    selection.add_argument("--shots", required=True)
    selection.add_argument("--first-shot", required=True, type=int, help="One-based first source shot.")
    selection.add_argument("--last-shot", required=True, type=int, help="One-based last source shot.")
    selection.add_argument("--output-directory", required=True)

    hands = commands.add_parser("track-hands")
    hands.add_argument("--video", required=True)
    hands.add_argument("--shots", required=True)
    hands.add_argument("--model", default="runs\\video-evidence\\models\\hand_landmarker.task")
    hands.add_argument("--output-directory", required=True)
    hands.add_argument("--maximum-dimension", type=int, default=1280)
    hands.add_argument("--pose-model")

    for name in ("assign-hand-roles", "prepare-paired-inputs"):
        roles = commands.add_parser(name)
        for argument in ("video", "shots", "hands", "geometry", "annotations", "output-directory"):
            roles.add_argument(f"--{argument}", required=True)
        if name == "assign-hand-roles":
            roles.add_argument("--plucking-screen-side", choices=("geometry", "left", "right"), default="geometry",
                               help="Explicit screen orientation; unavailable roles remain anonymous.")
        else:
            for argument in ("roles", "audio", "alignment"):
                roles.add_argument(f"--{argument}", required=True)
            roles.add_argument("--clip", nargs=2, type=int, action="append", required=True, metavar=("START_PTS", "END_PTS"))
            roles.add_argument("--pair-id")

    align = commands.add_parser("align-audio")
    align.add_argument("--video", required=True)
    align.add_argument("--trimmed-audio", required=True)
    align.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command in ("provision-hands", "provision-pose"):
        provision = provision_hand_landmarker if args.command == "provision-hands" else provision_pose_landmarker
        model, receipt = provision(args.output)
        print(json.dumps({"model": str(model), "receipt": str(receipt)}))
    elif args.command == "inspect-shots":
        path, report = inspect_shots(
            args.video, args.output_directory,
            ShotConfig(cut_threshold=args.cut_threshold, min_shot_seconds=args.min_shot_seconds),
            review_path=args.review,
        )
        print(json.dumps({"shots": str(path), "frameCount": report["frameCount"], "shotCount": report["shotCount"]}))
    elif args.command == "select-shots":
        path, report = select_shots(args.video, args.shots, args.first_shot, args.last_shot, args.output_directory)
        print(json.dumps({"shots": str(path), "frameCount": report["frameCount"], "shotCount": report["shotCount"]}))
    elif args.command == "track-hands":
        path, report = track_hands(args.video, args.shots, args.model, args.output_directory,
                                  maximum_dimension=args.maximum_dimension, pose_model_path=args.pose_model)
        print(json.dumps({"hands": str(path), "frameCount": report["frameCount"], "coverage": report["handFrameCoverage"]}))
    elif args.command == "assign-hand-roles":
        path, report = assign_hand_roles(
            args.video, args.shots, args.hands, args.geometry, args.annotations, args.output_directory,
            config=RoleConfig(plucking_screen_side=args.plucking_screen_side),
        )
        print(json.dumps({"roles": str(path), "frameCount": report["frameCount"]}))
    elif args.command == "prepare-paired-inputs":
        clips = [{"label": f"clip-{index + 1}", "startPts": start, "endPtsExclusive": end}
                 for index, (start, end) in enumerate(args.clip)]
        path, report = prepare_paired_inputs(
            args.video, args.shots, args.hands, args.geometry, args.annotations, args.roles,
            args.audio, args.alignment, args.output_directory, clips, pair_id=args.pair_id,
        )
        print(json.dumps({"inputs": str(path), "frameCount": report["frameCount"]}))
    else:
        path, report = align_soundtrack(args.video, args.trimmed_audio, args.output)
        print(json.dumps({"alignment": str(path), "status": report["status"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvidenceError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
