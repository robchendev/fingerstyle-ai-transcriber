"""Bridge validated batch inputs to the sole local preparation/release workflow."""

from pathlib import Path
import sys
from types import SimpleNamespace

from . import prepare_training_data as preparation
from .audio_tools import AcquisitionError
from .dataset_io import ROOT, read_json, sha256
from .dataset_release import release_path, validate_release
from .inspect_gp_files import GpInspectionError
from .score_alignment import AlignmentInputError
from .training_windows import range_sample_bounds, sample_windows


def _inputs(batch, root):
    workspace = preparation.workspace_path(batch.get("workspace", Path(root) / "data"))
    if ("releaseManifest" in batch) == ("releaseVersion" in batch):
        raise ValueError("Supply one existing releaseManifest or one new releaseVersion.")
    if "releaseVersion" in batch:
        preparation.safe_id(batch["releaseVersion"])
    if type(batch.get("acceptConventions", False)) is not bool:
        raise ValueError("acceptConventions must be an explicit boolean.")
    if not batch.get("records"):
        raise ValueError("A canonical batch needs explicitly selected records.")
    identifiers, groups, sources = set(), {}, {}
    for record in batch["records"]:
        identifier = preparation.safe_id(record["id"])
        if identifier.casefold() in identifiers:
            raise ValueError("Batch IDs must be unique, including case.")
        identifiers.add(identifier.casefold())
        group, split = record["groupId"], record["split"]
        if not isinstance(group, str) or not group.strip() or split not in ("train", "validation"):
            raise ValueError(f"{identifier}: explicit group and train/validation split required.")
        if group in groups and groups[group] != split:
            raise ValueError("A relationship group cannot cross train and validation.")
        groups[group] = split
        paths = [preparation.regular_path(record[field]) for field in ("gp", "audio")]
        if paths[0] == paths[1] or paths[0].suffix.lower() != ".gp":
            raise ValueError(f"{identifier}: use distinct original GP and trimmed audio files.")
        for field, path in zip(("gp", "audio"), paths):
            if not path.is_file():
                raise ValueError(f"{identifier}: missing original {field}: {path}")
            digest = sha256(path)
            identity = field, digest
            if identity in sources and sources[identity] != (group, split):
                raise ValueError(f"{identifier}: identical {field} sources have conflicting groups or splits.")
            sources[identity] = group, split
    return workspace


def _manifest_path(batch, workspace):
    if "releaseManifest" in batch:
        return preparation.regular_path(batch["releaseManifest"])
    return preparation.regular_path(workspace / "releases" / batch["releaseVersion"] / "manifest.json")


def _frozen(batch, manifest_path, *, exact=False):
    manifest, released, _ = validate_release(manifest_path)
    available = {entry["id"]: (entry, payload) for entry, payload in released}
    requested = [record["id"] for record in batch["records"]]
    if exact and (set(requested) != set(available) or manifest["version"] != batch["releaseVersion"]):
        raise ValueError("Release version already exists with a different selected scope; choose a NEW version.")
    rows = []
    for record in batch["records"]:
        identifier = record["id"]
        if identifier not in available:
            raise ValueError(f"{identifier}: record is not in the frozen release.")
        entry, payload = available[identifier]
        if any(record[field] != entry[field] for field in ("groupId", "split")):
            raise ValueError(f"{identifier}: group or split differs from the frozen release.")
        gp, source_audio = (preparation.regular_path(record[field]) for field in ("gp", "audio"))
        if sha256(gp) != payload["approval"]["sourceGpSha256"]:
            raise ValueError(f"{identifier}: original GP differs from the approved frozen source.")
        expected_audio = entry["audioSha256"] if source_audio.suffix.lower() == ".flac" else payload.get("sourceAudioSha256")
        if expected_audio is None or sha256(source_audio) != expected_audio:
            raise ValueError(f"{identifier}: original trimmed audio differs from the approved frozen source.")
        rows.append({
            "id": identifier, "groupId": entry["groupId"], "split": entry["split"],
            "gp": str(gp), "sourceAudio": str(source_audio),
            "audio": str(release_path(manifest_path.parent, entry["audioPath"], "audio")),
            "targets": str(release_path(manifest_path.parent, entry["targetsPath"], "targets")),
            "canonical": payload,
        })
    return {"status": "ready", "manifestPath": str(manifest_path), "records": rows, "actions": []}


def _registered(batch, workspace, *, import_missing):
    if import_missing:
        preparation.initialize(workspace)
    document = preparation.load_pairs(workspace)
    pairs = {pair["id"]: pair for pair in document["pairs"]}
    # Check every existing binding before importing any new record.
    for record in batch["records"]:
        pair = pairs.get(record["id"])
        if pair is None:
            if not import_missing:
                raise ValueError(f"{record['id']}: run the batch to import and prepare this source first.")
            continue
        if pair["groupId"] != record["groupId"]:
            raise ValueError(f"{record['id']}: registered group differs; never silently regroup an existing pair.")
        for field, owned in zip(("gp", "audio"), preparation.source_paths(workspace, pair)):
            if sha256(owned) != sha256(preparation.regular_path(record[field])):
                raise ValueError(f"{record['id']}: imported {field} differs from the external source; explicitly revise and invalidate, or use a new workspace.")
    for record in batch["records"]:
        if record["id"] not in pairs:
            pairs[record["id"]] = preparation.add_pair(
                workspace, record["id"], record["groupId"], record["gp"], record["audio"],
                title=record.get("title"),
            )
    return pairs


def _review_args(*, reviewer=None, record=None, ranges=None, anchors=(), exclude_ranges=None,
                 acknowledge_uncertainty=False, percussion_complete=False, accept=False):
    return SimpleNamespace(
        reviewer=reviewer, anchor=list(anchors), clear_anchors=False, approve_range=ranges,
        clear_ranges=False, exclude_range=exclude_ranges, clear_exclusions=False,
        split=record["split"] if accept else None, cue=[], listen=False,
        acknowledge_uncertainty=acknowledge_uncertainty,
        confirm_percussion_completeness=percussion_complete,
        **{option: accept for option in preparation.CONFIRMATIONS},
    )


def _command(workspace, command, *arguments):
    return [sys.executable, "-m", "scripts.prepare_training_data", "--workspace", str(workspace), command, *arguments]


def _action(workspace, record, reason, *, action="review-canonical", command=None, **details):
    identifier = record["id"]
    directory = workspace / "pairs" / identifier
    return {
        "id": identifier, "stage": "canonical-review", "action": action, "reason": reason,
        "command": command or _command(workspace, "review", "--id", identifier),
        "reportPath": str(directory / "review-report.json"), "notationPath": str(directory / "notation.json"),
        "rulesPath": str(directory / "rules.json"), **details,
    }


def _preparation_action(workspace, record, error, ffmpeg_dir):
    reason = str(error)
    action, path = "prepare-canonical", workspace / "pairs" / record["id"] / "rules.json"
    arguments = ["--ids", record["id"]]
    if "--accept-conventions" in reason:
        action = "review-conventions"
        arguments.append("--accept-conventions")
        reason += " Run the suggested command only after confirming that the documented conventions fit this source."
    elif "capo_or_tuning_text_requires_review" in reason:
        action = "review-capo-tuning-text"
        reason += f" Inspect the source GP text, then edit {path}: confirmFixedTuningCapoText may be true only if the mention does not change the fixed tuning/full capo. This cannot bypass active partial capo, missing metadata or automation."
    elif "inconsistent_partial_capo_metadata" in reason:
        action = "review-full-capo-metadata"
        reason += f" Inspect the source metadata, then edit {path}: confirmFullCapoMetadata may be true only for explicitly reviewed full capo with all six partial-capo flags zero. Active partial capo remains unsupported."
    elif isinstance(error, GpInspectionError) or reason.startswith(("Unresolved GP inspection:", "Only one six-string", "Unresolved score timing;")):
        action, path = "review-gp-structure", Path(record["gp"])
        reason += " Select a supported score in a new pair/workspace, or explicitly revise and invalidate the source through the existing workflow. Do not infer missing metadata or overwrite the imported GP."
    if ffmpeg_dir:
        arguments.extend(("--ffmpeg-dir", str(ffmpeg_dir)))
    return _action(
        workspace, record, reason, action=action, path=str(path),
        command=_command(workspace, "prepare", *arguments),
    )


def _review_reasons(report, directory, state, record):
    approval = report["approval"]
    reasons = []
    if any(approval.get(field) is not True for field in preparation.CONFIRMATIONS.values()):
        reasons.append("Inspect normalized.gp, notation evidence and alignment; explicitly accept the score with reviewed ranges.")
    if approval.get("groupId") != record["groupId"] or approval.get("split") != record["split"]:
        reasons.append("Explicitly review the batch's exact group and split.")
    if not report["approvedClipRanges"]:
        reasons.append("Approve explicit audio ranges; candidate bounds are suggestions, not approval.")
    labels = read_json(directory / "canonical.json")
    uncertain = (
        report["timingRisks"] or any(report["uncertainty"].values())
        or any(not all(note["labelMask"].values()) for note in labels["targets"]["notes"])
    )
    if uncertain and approval.get("uncertaintyAcknowledged") is not True:
        reasons.append("Review retained uncertainty/masks and explicitly acknowledge them, or exclude the affected ranges.")
    if report["approvedClipRanges"] and not list(sample_windows(
        range_sample_bounds(report["approvedClipRanges"], state["audio"]["sampleRate"]), state["audio"]["sampleRate"],
    )):
        reasons.append("No approved windows of at least two seconds remain.")
    return reasons


def ensure_canonical(batch, *, root=ROOT):
    """Prepare canonical inputs, retaining existing reviews and frozen releases."""
    workspace = _inputs(batch, root)
    manifest_path = _manifest_path(batch, workspace)
    if "releaseManifest" in batch or manifest_path.is_file():
        return _frozen(batch, manifest_path, exact="releaseManifest" not in batch)
    if manifest_path.parent.exists():
        raise ValueError("Release destination already exists without a manifest; choose a NEW version, never adopt its artifacts.")
    pairs = _registered(batch, workspace, import_missing=True)
    actions, rows, scope = [], [], []
    for record in batch["records"]:
        pair = pairs[record["id"]]
        directory = preparation.regular_path(workspace / "pairs" / pair["id"])
        if (directory / "preparation.json").exists():
            prior = None
            try:
                prior = read_json(preparation.regular_path(directory / "preparation.json"))
                preparation.load_current(workspace, pair)
            except (OSError, ValueError) as error:
                invalidated = isinstance(prior, dict) and prior.get("invalidated")
                actions.append(_action(
                    workspace, record, str(error),
                    action="prepare-canonical" if invalidated else "invalidate-canonical",
                    command=_command(workspace, "prepare", "--ids", pair["id"]) if invalidated else
                    _command(workspace, "invalidate", "--id", pair["id"], "--reason", "REVIEWED_REASON"),
                ))
                continue
        else:
            imported_names = {preparation.RAW_GP_NAME, Path(pair["audioPath"]).name, "rules.json"}
            unexpected = [path.name for path in directory.iterdir() if path.name not in imported_names]
            if unexpected:
                actions.append(_action(
                    workspace, record, f"Unbound preparation artifacts exist ({', '.join(sorted(unexpected))}); inspect them before explicitly preparing. They were not adopted or overwritten.",
                    action="prepare-canonical", command=_command(workspace, "prepare", "--ids", pair["id"]),
                ))
                continue
            try:
                preparation.prepare_pair(
                    workspace, pair, accept_conventions=batch.get("acceptConventions", False),
                    ffmpeg_dir=batch.get("ffmpegDirectory"),
                )
            except (OSError, ValueError, AcquisitionError, GpInspectionError, AlignmentInputError) as error:
                actions.append(_preparation_action(workspace, record, error, batch.get("ffmpegDirectory")))
                continue
        directory, state = preparation.load_current(workspace, pair)
        try:
            report = preparation.review_pair(workspace, pair, _review_args())
        except (OSError, ValueError) as error:
            actions.append(_action(workspace, record, str(error)))
            continue
        review = preparation.load_review(directory, state)
        mapping = review["candidate"]["denseMapping"]
        suggested = [[mapping[0]["clipSeconds"], mapping[-1]["clipSeconds"]]] if mapping else []
        reasons = _review_reasons(report, directory, state, record)
        if reasons:
            notation_review = report["uncertainty"]["unresolvedGestures"] or any(
                beat["referenceOnly"] and beat["text"] for beat in report["notationEvidence"]
            )
            if notation_review:
                reasons.append(
                    f"Inspect the source legend and notation evidence. Record only explicit source-specific meanings in {directory / 'rules.json'}, then invalidate and prepare again; otherwise retain unknown masks and explicitly acknowledge uncertainty. Never infer an unreviewed symbol meaning."
                )
            actions.append(_action(
                workspace, record, " ".join(reasons), suggestedRanges=suggested,
                action="review-notation-rules" if notation_review else "review-canonical",
                **({"path": str(directory / "rules.json")} if notation_review else {}),
                normalizedGpPath=str(directory / "normalized.gp"),
                reviewGuidance="Use the existing review --cue ORDINAL/first-attack/end to audition alignment. Inspect notation.json and normalized.gp; source-specific legend meanings belong in this pair's existing rules.json, followed by explicit invalidation/preparation.",
            ))
        scope.append({
            "id": record["id"], "groupId": record["groupId"], "split": record["split"],
            "sourceGpSha256": report["approval"]["sourceGpSha256"],
            "audioSha256": report["approval"]["audioSha256"],
            "approvedClipRanges": report["approvedClipRanges"],
        })
        rows.append({
            "id": record["id"], "groupId": record["groupId"], "split": record["split"],
            "gp": str(preparation.regular_path(record["gp"])), "sourceAudio": str(preparation.regular_path(record["audio"])),
            "audio": str(directory / preparation.TRIMMED_AUDIO_NAME),
        })
    if not actions:
        validation_groups = list(dict.fromkeys(row["groupId"] for row in batch["records"] if row["split"] == "validation"))
        release_arguments = ["--version", batch["releaseVersion"], "--ids", *[row["id"] for row in batch["records"]]]
        for group in validation_groups:
            release_arguments.extend(("--validation-group", group))
        release_arguments.extend(("--reviewer", "REVIEWER", "--authorize-release"))
        actions.append({
            "stage": "canonical-review", "action": "finalize-release",
            "reason": "All selected pairs are reviewed. Replace REVIEWER with your identity to explicitly authorize this exact release scope; the batch never releases or trains automatically.",
            "releaseVersion": batch["releaseVersion"],
            "selectedScope": scope,
            "validationGroups": validation_groups,
            "command": _command(workspace, "release", *release_arguments),
        })
    return {"status": "needs-review", "records": rows, "actions": actions}


def review_canonical(batch, identifier, *, reviewer=None, ranges=(), anchors=(), exclude_ranges=(),
                     acknowledge_uncertainty=False, percussion_complete=False, accept=False, root=ROOT):
    """Inspect or record source-bound review decisions."""
    workspace = _inputs(batch, root)
    manifest_path = _manifest_path(batch, workspace)
    if "releaseManifest" in batch or manifest_path.exists():
        raise ValueError("Frozen releases are already reviewed and cannot be edited through batch review.")
    records = {record["id"]: record for record in batch["records"]}
    if identifier not in records:
        raise ValueError("Review ID is not in this batch.")
    if type(accept) is not bool or type(acknowledge_uncertainty) is not bool or type(percussion_complete) is not bool:
        raise ValueError("Review decisions must be explicit booleans.")
    if not accept and (ranges or anchors or exclude_ranges or acknowledge_uncertainty or percussion_complete):
        raise ValueError("Recording score decisions requires explicit accept=True.")
    if accept and (not isinstance(reviewer, str) or not reviewer.strip() or not ranges):
        raise ValueError("Explicit score acceptance requires a reviewer and nonempty reviewed ranges.")
    pairs = _registered(batch, workspace, import_missing=False)
    return preparation.review_pair(workspace, pairs[identifier], _review_args(
        reviewer=reviewer, record=records[identifier], ranges=list(ranges) if accept else None,
        anchors=anchors, exclude_ranges=list(exclude_ranges) if exclude_ranges else None,
        acknowledge_uncertainty=acknowledge_uncertainty, percussion_complete=percussion_complete, accept=accept,
    ))


def finalize_canonical(batch, reviewer, *, root=ROOT):
    """Explicit authorization of the explicit batch scope, never training."""
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("Finalizing a canonical release requires a reviewer.")
    workspace = _inputs(batch, root)
    manifest_path = _manifest_path(batch, workspace)
    if "releaseManifest" in batch or manifest_path.is_file():
        return _frozen(batch, manifest_path, exact="releaseManifest" not in batch)
    if manifest_path.parent.exists():
        raise ValueError("Release destination already exists without a manifest; choose a NEW version.")
    pairs = _registered(batch, workspace, import_missing=False)
    groups = list(dict.fromkeys(record["groupId"] for record in batch["records"] if record["split"] == "validation"))
    if {record["split"] for record in batch["records"]} != {"train", "validation"}:
        raise ValueError("A new release requires explicitly reviewed train AND validation relationship groups.")
    for record in batch["records"]:
        pair = pairs[record["id"]]
        directory, state = preparation.load_current(workspace, pair)
        review = preparation.load_review(directory, state)
        if review["approval"].get("split") != record["split"]:
            raise ValueError(f"{record['id']}: reviewed split differs from the batch.")
    preparation.release_dataset(
        workspace, batch["releaseVersion"], groups, [record["id"] for record in batch["records"]],
        reviewer=reviewer, authorize_release=True,
    )
    return _frozen(batch, manifest_path, exact=True)
