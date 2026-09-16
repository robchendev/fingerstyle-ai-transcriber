"""Propose grouped training ranges and coverage without granting approval."""

from collections import Counter, defaultdict
import math
from pathlib import Path

from .dataset_io import read_json, sha256
from .dataset_release import candidate_digest, validate_mapping
from .training_windows import projected_targets, range_sample_bounds, sample_windows, targets_in_window
from .score_alignment import ScoreClock


LOCAL_RISKS = frozenset({"sustained_local_mismatch", "long_score_position_stall", "score_time_compression"})
BOUNDED_RISKS = frozenset({"unmapped_score_prefix"})
COVERAGE_TYPES = ("wrist_thump", "thumb_slap", "percussive_hit", "harmonics")
CONFIRMATIONS = ("authorizedUse", "recordingAndTargetPitchConfirmed", "notationReviewed", "approveExperimentalRangesAndSplit", "groupingConfirmed")
GUARD_SECONDS = .5


def subtract_intervals(ranges, exclusions):
    result = [list(value) for value in ranges]
    for left, right in exclusions:
        if not math.isfinite(left) or not math.isfinite(right) or left >= right:
            raise ValueError("Exclusions require finite increasing bounds.")
        result = [
            part for a, b in result
            for part in ([[a, b]] if right <= a or left >= b else [[a, min(b, left)], [max(a, right), b]])
            if part[0] < part[1]
        ]
    return result


def proposed_ranges(candidate, review, duration):
    mapping = candidate["denseMapping"]
    if not mapping:
        return [], [], ["missing_timing_mapping"], False
    if all(review.get("approval", {}).get(field) is True for field in CONFIRMATIONS):
        ranges = review["approval"]["approvedClipRanges"]
        if not ranges:
            raise ValueError("A reviewed pair has no approved ranges.")
        previous = -1.
        for left, right in ranges:
            if not mapping[0]["clipSeconds"] <= left < right <= mapping[-1]["clipSeconds"] or left < previous:
                raise ValueError("Reviewed ranges are outside the mapping or overlap.")
            previous = right
        return ranges, [], [], True
    triage = candidate.get("triage", {})
    reasons = set(triage.get("reasons", []))
    blockers = sorted(reasons - LOCAL_RISKS - BOUNDED_RISKS)
    exclusions = []
    located = set()
    for passage in triage.get("passages", []):
        reason = passage["reason"]
        if reason not in LOCAL_RISKS:
            continue
        left, right = passage["clipSecondsStart"], passage["clipSecondsEnd"]
        if type(left) not in (int, float) or type(right) not in (int, float) or not math.isfinite(left) or not math.isfinite(right) or not 0 <= left <= right <= duration + 1e-8:
            raise ValueError("A timing-risk passage has invalid clip coordinates.")
        located.add(reason)
        exclusions.append({"reason": reason, "clipSeconds": [left, right], "guardedClipSeconds": [max(0., left - GUARD_SECONDS), min(duration, right + GUARD_SECONDS)]})
    blockers.extend(f"unbounded_{reason}" for reason in sorted((reasons & LOCAL_RISKS) - located))
    if blockers:
        return [], exclusions, blockers, False
    requested = review.get("requestedClipRanges", [])
    ranges = requested or [[mapping[0]["clipSeconds"], mapping[-1]["clipSeconds"]]]
    for left, right in review.get("excludedClipRanges", []):
        exclusions.append({"reason": "human_excluded_range", "clipSeconds": [left, right], "guardedClipSeconds": [max(0., left - GUARD_SECONDS), min(duration, right + GUARD_SECONDS)]})
    for left, right in ranges:
        if not mapping[0]["clipSeconds"] <= left < right <= mapping[-1]["clipSeconds"]:
            raise ValueError("Proposed ranges extend beyond mapped audio; no extrapolation is permitted.")
    return subtract_intervals(ranges, [item["guardedClipSeconds"] for item in exclusions]), exclusions, [], False


def coverage_profile(labels, candidate, normalization, ranges, rate):
    windows = sample_windows(range_sample_bounds(ranges, rate), rate) if ranges else []
    clock = ScoreClock(labels, normalization)
    notes, gestures = projected_targets(labels, candidate, clock)
    selected_notes, selected_gestures = {}, {}
    window_notes = []
    for start, stop in windows:
        projected = targets_in_window(notes, gestures, start, stop, rate)
        attacks = [note for note in projected["notes"] if note["supervisionMask"]["onset"]]
        percussion = [gesture for gesture in projected["gestures"] if gesture["supervisionMask"]["gesture"] and gesture["supervisionMask"]["onset"]]
        selected_notes.update((note["sourceNoteId"], note) for note in attacks)
        selected_gestures.update((gesture["sourceGestureId"], gesture) for gesture in percussion)
        window_notes.append({"startSample": start, "stopSampleExclusive": stop, "noteAttacks": len(attacks), "gestureAttacks": len(percussion)})
    source_notes = {note["id"]: note for note in labels["targets"]["notes"]}
    sample_ranges = []
    for start, stop in sorted(windows):
        if sample_ranges and start <= sample_ranges[-1][1]:
            sample_ranges[-1][1] = max(stop, sample_ranges[-1][1])
        else:
            sample_ranges.append([start, stop])
    counts = Counter(gesture["technique"] for gesture in selected_gestures.values())
    counts["notes"] = len(selected_notes)
    counts["harmonics"] = sum(source_notes[key]["sourceSegments"][0].get("harmonic") is not None for key in selected_notes)
    harmonic_kinds = Counter(source_notes[key]["sourceSegments"][0]["harmonic"]["type"] for key in selected_notes if source_notes[key]["sourceSegments"][0].get("harmonic") is not None)
    return {
        "windows": window_notes, "windowCount": len(windows),
        "sampleRanges": sample_ranges, "uniqueAudioSeconds": sum(stop - start for start, stop in sample_ranges) / rate,
        "audioSecondsIncludingOverlap": sum(stop - start for start, stop in windows) / rate,
        "counts": {key: counts[key] for key in ("notes", *COVERAGE_TYPES)},
        "voices": sorted({note["voiceIndex"] for note in selected_notes.values()}),
        "pitchCounts": {str(key): value for key, value in sorted(Counter(note["soundingPitchMidi"] for note in selected_notes.values() if note["sourceLabelMask"]["pitch"]).items())},
        "fretCounts": {str(key): value for key, value in sorted(Counter(note["fret"] for note in selected_notes.values() if note["sourceLabelMask"]["fingering"]).items())},
        "durationCounts": dict(Counter(f"{note['notatedDurationQuarter'][0]}/{note['notatedDurationQuarter'][1]}" for note in selected_notes.values() if note["sourceLabelMask"]["notatedDuration"] and note["notatedDurationQuarter"] is not None)),
        "harmonicTypes": dict(harmonic_kinds),
    }


def inspect_pair(workspace, pair):
    from .prepare_training_data import load_current, load_review

    directory, state = load_current(workspace, pair)
    review = load_review(directory, state)
    candidate = review["candidate"]
    labels, normalization = read_json(directory / "canonical.json"), read_json(directory / "normalization.json")
    duration = state["audio"]["sampleCount"] / state["audio"]["sampleRate"]
    validate_mapping(candidate["denseMapping"], ScoreClock(labels, normalization), duration)
    ranges, exclusions, blockers, reviewed = proposed_ranges(candidate, review, duration)
    coverage = coverage_profile(labels, candidate, normalization, ranges, state["audio"]["sampleRate"])
    if not coverage["windows"] and not blockers:
        blockers.append("no_windows_after_range_exclusions")
    provided = labels["conditioning"]["providedTiming"]
    initial_tempo = provided["tempo"]
    quarter_bpm = initial_tempo["bpm"] * initial_tempo["beatUnit"][0] / initial_tempo["beatUnit"][1] * 4
    signatures = {tuple(provided["timeSignature"])}
    signatures.update(tuple(item["timeSignature"]) for item in provided["sourceTimeSignatureChanges"])
    bindings = {
        str((directory / name).relative_to(workspace)): sha256(directory / name)
        for name in ("preparation.json", "rules.json", *state["artifacts"])
    }
    review_path = directory / "review.json"
    if review_path.exists():
        bindings[str(review_path.relative_to(workspace))] = sha256(review_path)
    return {
        "id": pair["id"], "title": pair.get("title", pair["id"]), "groupId": pair["groupId"],
        "performerId": pair.get("performerId"), "status": "quarantined" if blockers else "proposed",
        "blockers": blockers, "ranges": ranges, "excludedPassages": exclusions,
        "proposedAudioSeconds": coverage["uniqueAudioSeconds"],
        "audioDurationSeconds": duration, "existingRangeApproval": reviewed,
        "existingSplitConstraint": review.get("approval", {}).get("split"),
        "percussionCompletenessConfirmed": review.get("approval", {}).get("percussionAnnotationsComplete") is True,
        "coverage": coverage, "tuning": labels["conditioning"]["instrument"]["openStringMidi"],
        "capo": labels["conditioning"]["instrument"]["capoFret"], "meters": [list(value) for value in sorted(signatures)],
        "initialQuarterBpm": quarter_bpm,
        "sourceBindings": bindings, "reviewWasPresent": review_path.exists(),
        "sourceGpSha256": state["inputs"]["sourceGpSha256"], "audioSha256": state["audio"]["sha256"],
        "candidateSha256": candidate_digest(candidate),
        "timingReviewStatus": "existing-reviewed-ranges" if reviewed else "proposed-interpolation-not-human-approved",
    }


def _coverage_score(rows, totals):
    score = 0.
    for name in COVERAGE_TYPES:
        events = sum(row["coverage"]["counts"][name] for row in rows)
        sources = sum(row["coverage"]["counts"][name] > 0 for row in rows)
        target = min(20, totals[name])
        score += (min(events, target) / target if target else 0) * 4 + min(sources, 2) * 2
    score += min(len({row["performerId"] for row in rows if row["performerId"]}), 2) * 2
    score += min(len({tuple(row["tuning"]) for row in rows}), 5)
    score += min(len({tuple(meter) for row in rows for meter in row["meters"]}), 3) * 2
    score += min(len({row["capo"] for row in rows}), 4) * .5
    score += min(len({len(row["coverage"]["voices"]) for row in rows}), 2)
    score += min(len({int(pitch) for row in rows for pitch in row["coverage"].get("pitchCounts", {})}), 36) / 18
    score += min(len({int(fret) for row in rows for fret in row["coverage"].get("fretCounts", {})}), 16) / 8
    score += min(len({duration for row in rows for duration in row["coverage"].get("durationCounts", {})}), 8) / 4
    score += min(len({int(row["initialQuarterBpm"] // 30) for row in rows if "initialQuarterBpm" in row}), 3) / 2
    return score


def select_validation(profiles, target_groups=10, explicit_ids=None):
    if type(target_groups) is not int or target_groups < 1:
        raise ValueError("The validation group target must be a positive integer.")
    groups = defaultdict(list)
    for row in profiles:
        if row["status"] == "proposed":
            groups[row["groupId"]].append(row)
    if not groups:
        raise ValueError("No usable groups are available for a proposal.")
    pinned_train, pinned_validation = set(), set()
    for group, rows in groups.items():
        constraints = {row["existingSplitConstraint"] for row in rows if row["existingSplitConstraint"] is not None}
        if len(constraints) > 1 or constraints - {"train", "validation"}:
            raise ValueError("Existing source reviews disagree about a connected group's split.")
        if "train" in constraints:
            pinned_train.add(group)
        if "validation" in constraints:
            pinned_validation.add(group)
    if explicit_ids is not None:
        if not explicit_ids or len(explicit_ids) != len(set(explicit_ids)):
            raise ValueError("Explicit validation IDs must be unique and nonempty.")
        indexed = {row["id"]: row for rows in groups.values() for row in rows}
        if set(explicit_ids) - set(indexed):
            raise ValueError("Explicit validation IDs include unknown or quarantined records.")
        selected_groups = list(dict.fromkeys(indexed[identifier]["groupId"] for identifier in explicit_ids))
        if set(selected_groups) & pinned_train or pinned_validation - set(selected_groups):
            raise ValueError("Validation selection conflicts with existing reviewed split constraints.")
    else:
        selected_groups = sorted(pinned_validation)
        selected = [row for group in selected_groups for row in groups[group]]
        if len(selected_groups) > target_groups:
            raise ValueError("Existing reviewed validation groups exceed the requested target.")
        totals = Counter()
        for rows in groups.values():
            for row in rows:
                totals.update(row["coverage"]["counts"])
        while len(selected_groups) < target_groups:
            options = []
            current_score = _coverage_score(selected, totals)
            for group, rows in groups.items():
                if group in selected_groups or group in pinned_train:
                    continue
                trial = [*selected, *rows]
                # Keep most rare technique examples on the training side.
                if any(sum(row["coverage"]["counts"][name] for row in trial) > max(20, .3 * totals[name]) for name in COVERAGE_TYPES if totals[name]):
                    continue
                improvement = _coverage_score(trial, totals) - current_score
                excluded_fraction = sum(1 - row["proposedAudioSeconds"] / row["audioDurationSeconds"] for row in rows) / len(rows)
                rarity_cost = sum(row["coverage"]["counts"][name] / max(1, totals[name]) for row in rows for name in COVERAGE_TYPES)
                options.append((improvement / len(rows), -excluded_fraction, -rarity_cost, group))
            if not options:
                raise ValueError("Cannot fill the validation target while preserving groups, reviewed splits and training technique coverage; choose explicit IDs or a different target.")
            chosen = max(options)[-1]
            selected_groups.append(chosen)
            selected.extend(groups[chosen])
    validation = [row for group in selected_groups for row in groups[group]]
    training = [row for group, rows in groups.items() if group not in selected_groups for row in rows]
    if not validation or not training:
        raise ValueError("A proposal requires nonempty training and validation groups.")
    return selected_groups


def split_summary(rows):
    counts = Counter()
    source_counts = Counter()
    harmonic_types = Counter()
    for row in rows:
        counts.update(row["coverage"]["counts"])
        source_counts.update(name for name in COVERAGE_TYPES if row["coverage"]["counts"][name] > 0)
        harmonic_types.update(row["coverage"].get("harmonicTypes", {}))
    return {
        "recordings": len(rows), "groups": len({row["groupId"] for row in rows}),
        "windows": sum(row["coverage"]["windowCount"] for row in rows),
        "uniqueAudioSeconds": sum(row["proposedAudioSeconds"] for row in rows),
        "uniqueEventCounts": dict(counts), "recordingsWithTechnique": dict(source_counts),
        "meters": [list(value) for value in sorted({tuple(meter) for row in rows for meter in row["meters"]})],
        "tuningCount": len({tuple(row["tuning"]) for row in rows}),
        "performerCounts": dict(Counter(row["performerId"] or "unspecified" for row in rows)),
        "pitchMidiValues": sorted({int(pitch) for row in rows for pitch in row["coverage"].get("pitchCounts", {})}),
        "fretValues": sorted({int(fret) for row in rows for fret in row["coverage"].get("fretCounts", {})}),
        "voiceIndices": sorted({voice for row in rows for voice in row["coverage"]["voices"]}),
        "notatedDurationValues": sorted({duration for row in rows for duration in row["coverage"].get("durationCounts", {})}),
        "harmonicTypeCounts": dict(harmonic_types),
    }


def render_proposal(document):
    counts = document["splitSummary"]
    lines = [
        "# Expanded training-data proposal", "",
        "**PROPOSAL ONLY: no new training release is active and no training was started.**", "",
        f"Proposed recordings: {sum(value['recordings'] for value in counts.values())}; quarantined: {len(document['quarantined'])}. Existing releases and source files are unchanged.", "",
        "| Split | Recordings | Groups | Windows | Unique minutes |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[f"| {name} | {value['recordings']} | {value['groups']} | {value['windows']} | {value['uniqueAudioSeconds'] / 60:.2f} |" for name, value in counts.items()],
        "", "## Musical coverage", "",
        "Counts are unique resolved source attacks in the proposed windows, not duplicated overlapping-window labels. Unknown symbols/timing remain masked.", "",
        "| Target | Training attacks | Validation attacks | Training recordings | Validation recordings |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[f"| {name} | {counts['train']['uniqueEventCounts'].get(name, 0)} | {counts['validation']['uniqueEventCounts'].get(name, 0)} | {counts['train']['recordingsWithTechnique'].get(name, '-')} | {counts['validation']['recordingsWithTechnique'].get(name, '-')} |" for name in ("notes", *COVERAGE_TYPES)],
        "", "## Validation selection", "",
        "Each relationship group stays entirely in one split. Existing reviewed split assignments are retained; selection aims for class coverage and variation, not a calibrated generalization guarantee.", "",
        "| Recording | ID | Minutes | Wrist | Thumb | Generic | Harmonics |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in document["entries"]:
        if row["split"] == "validation":
            c = row["coverage"]["counts"]
            title = row["title"].replace("|", "/").replace("\n", " ")
            title = title if len(title) <= 80 else title[:77] + "..."
            lines.append(f"| {title} | {row['id']} | {row['proposedAudioSeconds'] / 60:.2f} | {c['wrist_thump']} | {c['thumb_slap']} | {c['percussive_hit']} | {c['harmonics']} |")
    lines.extend([
        "", "## Quarantined recordings", "",
        *[f"- {row['title']} ({row['id']}): {', '.join(row['blockers'])}." for row in document["quarantined"]],
        "", "## Decisions before release", "",
        *[f"- {decision}" for decision in document["requiredDecisions"]],
        "", "Localized timing flags are excluded with a half-second guard. Unbounded/rejected timing is quarantined rather than guessed. Unmapped score/audio tails are not extrapolated. Existing reviewed pilot ranges are retained without enlargement.", "",
        "Low matching cost and the absence of a diagnostic flag are not human timing approval. The proposed interpolation must be explicitly accepted for these experimental ranges before it can become training data. No requirement to repeat completed trims or audition four cues for every file is implied.", "",
        "The accompanying JSON contains every included recording's exact sample windows, ranges, local exclusions, grouping and input hashes. It is not a trainer manifest. Keep both files private.", "",
    ])
    if document["coverageWarnings"]:
        lines.extend(["## Coverage limitations", "", *[f"- {warning}" for warning in document["coverageWarnings"]], ""])
    return "\n".join(lines)


def propose_dataset(workspace, name, *, validation_group_count=10, validation_ids=None, progress=None):
    from .prepare_training_data import load_pairs, regular_path, safe_id, write_json, write_bytes

    workspace = Path(workspace).resolve()
    safe_id(name)
    registry_path = workspace / "pairs.json"
    registry_hash = sha256(registry_path)
    pairs = load_pairs(workspace)["pairs"]
    profiles = []
    implementation = {path.name: sha256(path) for path in Path(__file__).parent.glob("*.py")}
    for index, pair in enumerate(pairs):
        try:
            row = inspect_pair(workspace, pair)
        except (OSError, ValueError) as error:
            row = {"id": pair["id"], "title": pair.get("title", pair["id"]), "groupId": pair["groupId"], "status": "quarantined", "blockers": [f"invalid_preparation: {error}"]}
        profiles.append(row)
        if progress is not None and (index == 0 or (index + 1) % 20 == 0 or index + 1 == len(pairs)):
            progress(f"Proposal: inspected {index + 1}/{len(pairs)} pairs.")
    validation_groups = select_validation(profiles, validation_group_count, validation_ids)
    entries, quarantine = [], []
    for row in profiles:
        if row["status"] != "proposed":
            quarantine.append(row)
        else:
            entries.append({**row, "split": "validation" if row["groupId"] in validation_groups else "train"})
    summaries = {split: split_summary([row for row in entries if row["split"] == split]) for split in ("train", "validation")}
    for field in ("sourceGpSha256", "audioSha256"):
        sources = {}
        for row in entries:
            digest = row.get(field)
            if digest is not None:
                if digest in sources and sources[digest] != row["split"]:
                    raise ValueError("Identical source files cross the proposed split; correct the relationship grouping or choose different validation IDs.")
                sources[digest] = row["split"]
    warnings = []
    for split, summary in summaries.items():
        for technique in COVERAGE_TYPES:
            source_count = summary["recordingsWithTechnique"].get(technique, 0)
            count = summary["uniqueEventCounts"].get(technique, 0)
            if source_count < 2 or count < 20:
                warnings.append(f"{split}: {technique} has only {count} resolved attacks across {source_count} recordings.")
    for kind, count in summaries["train"]["harmonicTypeCounts"].items():
        if not summaries["validation"]["harmonicTypeCounts"].get(kind):
            warnings.append(f"Validation has no {kind} harmonic examples; training contains {count}. This subtype cannot be evaluated from this split.")
    missing_voices = set(summaries["train"]["voiceIndices"]) - set(summaries["validation"]["voiceIndices"])
    if missing_voices:
        warnings.append(f"Validation does not cover source voice indices {sorted(missing_voices)} (zero-based), which occur in training.")
    unreviewed = [row for row in entries if not row["existingRangeApproval"]]
    unconfirmed_percussion = sum(not row["percussionCompletenessConfirmed"] for row in entries)
    document = {
        "schemaVersion": 1, "kind": "training-dataset-proposal", "name": name,
        "trainingReady": False, "visibility": "private", "distributionAuthorized": False,
        "validationGroups": validation_groups, "validationGroupTarget": validation_group_count,
        "selectionPolicy": "Deterministic greedy coverage of four technique types, performer/tuning/meter/voice variation, existing reviewed split constraints and whole-group isolation. Automatic selection limits each validation technique count to max(20,30% of the available total) to retain training examples; explicit IDs can override selection, never source/group approval.",
        "guardSeconds": GUARD_SECONDS, "splitSummary": summaries, "entries": entries, "quarantined": quarantine,
        "coverageWarnings": warnings,
        "requiredDecisions": [
            f"Accept or adjust the exact validation selection ({summaries['validation']['recordings']} recordings / {len(validation_groups)} independent groups), including known arrangement/medley relationships.",
            f"Batch-approve recording/target-pitch correspondence, authorized use and experimental interpolation for the {len(unreviewed)} included recordings without existing range approvals. Reuse the completed notation review and existing normalized targets; this is not a request to retranscribe or retrim them.",
            f"Confirm percussion-annotation completeness for the {unconfirmed_percussion} included recordings without source-bound completeness approval; otherwise their unmarked percussion remains unknown.",
            "Accept the listed localized exclusions and quarantine; any changed source, range, split or rule requires a new bound proposal.",
        ],
        "sourceRegistrySha256": registry_hash, "implementationSha256": implementation,
        "trainingExecution": "human-owner-only", "finalTestSet": False,
    }
    if sha256(registry_path) != registry_hash:
        raise ValueError("Pair registry changed during proposal generation.")
    for row in entries:
        for relative, expected in row["sourceBindings"].items():
            if sha256(workspace / relative) != expected:
                raise ValueError(f"Proposal source changed: {relative}")
        if not row["reviewWasPresent"] and (workspace / "pairs" / row["id"] / "review.json").exists():
            raise ValueError("A new source review appeared during proposal generation.")
    if implementation != {path.name: sha256(path) for path in Path(__file__).parent.glob("*.py")}:
        raise ValueError("Implementation changed during proposal generation.")
    document["proposalSha256"] = candidate_digest(document)
    path = regular_path(workspace / "proposals" / f"{name}.json")
    if path.exists() and read_json(path) != document:
        raise ValueError("A different proposal already uses that name; choose a new proposal name.")
    write_json(path, document)
    write_bytes(path.with_suffix(".md"), render_proposal(document).encode("utf-8"))
    return {
        "proposalPath": str(path), "reviewPath": str(path.with_suffix(".md")), "trainingReady": False,
        "splitSummary": summaries, "quarantinedRecordings": len(quarantine), "coverageWarnings": warnings,
    }
