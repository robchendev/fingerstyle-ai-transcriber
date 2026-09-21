# Local training

Run commands from the repository root with the root Python environment active. Install the root requirements and the separate `scripts\video-evidence\requirements.txt` environment as described in the README. Put FFmpeg and FFprobe on `PATH`; supply a local MediaPipe hand model. `VIDEO_PYTHON` may select the video environment; otherwise the default is `scripts\video-evidence\.venv\Scripts\python.exe`.

Use modern `.gp` scores and already-trimmed local performance videos. No download, browser login or trimming step is provided. Audio is optional: omitted audio is deterministically extracted as native-rate, native-channel PCM-24 FLAC, with decoded timestamps and source hashes retained. An explicit `audio` path must already match the selected performance. Ambiguous streams or discontinuous timestamps require explicit audio and reviewed alignment.

## Prepare and review

Save this generic example as `runs\batch.json`. Choose independent validation groups yourself; related arrangements, shared recordings and identical sources must not cross splits. Paths are repository-relative or absolute. Replace the example files, model path and screen side with your inputs.

```json
{
  "schemaVersion": 1,
  "kind": "paired-preparation-batch",
  "workspace": "data",
  "releaseVersion": "dataset-v1",
  "acceptConventions": true,
  "handModel": "runs\\video-evidence\\models\\hand_landmarker.task",
  "records": [
    {"id": "pair-a", "groupId": "group-a", "split": "train", "gp": "inputs\\pair-a.gp", "video": "inputs\\pair-a-trimmed.mp4", "pluckingScreenSide": "left"},
    {"id": "pair-b", "groupId": "group-b", "split": "validation", "gp": "inputs\\pair-b.gp", "video": "inputs\\pair-b-trimmed.mp4", "pluckingScreenSide": "right"}
  ]
}
```

`acceptConventions` accepts `O` as wrist thump, plain `X` as thumb slap and ghost `X` as percussive hit. Unknown short text remains masked. Review `rules.json` for source-specific notation; changes to imported sources or rules require `invalidate --id ID --reason TEXT`, followed by preparation and review again. Frozen releases remain unchanged.

```powershell
python -m scripts.prepare_training_data batch --manifest runs\batch.json --output-directory runs\video-evidence\batches\dataset-v1
python -m scripts.prepare_training_data batch-status --manifest runs\batch.json --output-directory runs\video-evidence\batches\dataset-v1
python -m scripts.prepare_training_data --workspace data review --id pair-a --cue first-attack --cue end
```

Inspect `next-actions.txt`, each pair's `review-report.json`, `notation.json` and `normalized.gp`. Approve audio-second ranges inside the verified score mapping. The following example bounds and anchors are illustrative: replace them with the actual reviewed times for each recording. Use more anchors or `--exclude-range START:END` for local drift; `--acknowledge-uncertainty` retains the reported masks rather than resolving them.

```powershell
python -m scripts.prepare_training_data batch-review --manifest runs\batch.json --output-directory runs\video-evidence\batches\dataset-v1 --id pair-a --reviewer reviewer --accept-score --range 0.4:16.4 --anchor 1=0.4 --anchor end=16.4 --acknowledge-uncertainty
python -m scripts.prepare_training_data batch-review --manifest runs\batch.json --output-directory runs\video-evidence\batches\dataset-v1 --id pair-b --reviewer reviewer --accept-score --range 0.4:16.4 --anchor 1=0.4 --anchor end=16.4 --acknowledge-uncertainty
python -m scripts.prepare_training_data batch-release --manifest runs\batch.json --output-directory runs\video-evidence\batches\dataset-v1 --reviewer reviewer
python -m scripts.prepare_training_data batch --manifest runs\batch.json --output-directory runs\video-evidence\batches\dataset-v1
```

Release requires approved nonempty train and validation windows. Preparation then builds the paired video index. Resolve reported alignment or shot actions with `batch-review --alignment-offset SECONDS` or `batch-review --accept-shots` before rerunning the batch. `ready` means valid prepared inputs, not musical accuracy or sufficient class coverage; entirely unavailable training-hand evidence is not ready for joint training. The 194-dimensional format retains masked geometry/coarse-context slots; only independent hand evidence is prepared.

## Train from scratch

```powershell
python -m scripts.transcriber config --manifest data\releases\dataset-v1\manifest.json --video-index runs\video-evidence\batches\dataset-v1\paired-index.json --output runs\joint-config.json
python -m scripts.transcriber preflight --config runs\joint-config.json --forward --output runs\joint-preflight.json
python -m scripts.transcriber train --config runs\joint-config.json --run-dir runs\joint-training
```

Edit the generated configuration's device, thread count, batch size and epoch budget before training. Joint training initializes the acoustic, numeric-video and fusion parameters together; there is no acoustic warm-up or frozen refiner. Default whole-video dropout is `0.2`, enabling audio-only inference. Preflight makes no optimizer updates. Training has no default wall-clock limit; optional limits use `--max-hours`.

Current architecture-v4 checkpoint selection requires scorable validation events and at least one covered non-none note technique, relation or anchored-grace event. A notes-only validation set can pass input preflight but cannot complete decoded-event checkpoint selection; do not treat preflight as a coverage guarantee.

Alternatively, `python -m scripts.prepare_training_data batch-train --manifest runs\batch.json --output-directory runs\video-evidence\batches\dataset-v1 --epochs 20 --device cpu --cpu-threads 4` prepares and trains one joint run, pausing on unresolved preparation actions. Rerunning the same command resumes its own checkpoint or reuses a completed result.

For audio-only training, import explicit GP/audio pairs with `init`, `add --id ID --group GROUP --gp FILE --audio FILE`, `prepare --accept-conventions`, and `review`; freeze them with `release --version dataset-v1 --validation-group GROUP --reviewer reviewer --authorize-release`. Then run `python -m scripts.transcriber config --manifest data\releases\dataset-v1\manifest.json --output runs\audio-config.json` without `--video-index`, and use the same preflight/train commands with that configuration.

`models\transcriber.pt` is an inference artifact, **not a training-resume checkpoint**. Start fresh training without it. Resume only an interrupted compatible run using `train --config runs\joint-config.json --run-dir runs\joint-training --resume runs\joint-training\latest.pt`. Source, release and paired-index hashes remain checked on reuse.
