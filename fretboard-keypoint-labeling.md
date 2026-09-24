# Fretboard keypoint labeling

This workflow samples native-resolution frames on Windows, supports annotation
in a local browser on macOS, and exports labels for YOLO training on Windows.
The dataset is stored under `data/`, which is excluded from Git.

## Keypoints

Place these six ordered points:

1. String 6 at the nut
2. String 1 at the nut
3. String 6 at fret 12
4. String 1 at fret 12
5. String 6 at the bridge
6. String 1 at the bridge

The harness derives the nut, fret-12, and bridge centers from each pair.

Use **Visible** when the point can be placed directly and **Occluded** when its
position is identifiable despite an obstruction. Clear a point to mark it
unavailable when it is outside the frame or cannot be identified. Do not
estimate an off-screen nut or bridge.

`Complete` means the entire frame has been reviewed. It does not require all
six points to be available.

## Create the dataset on Windows

From the repository root:

```powershell
.\scripts\video-evidence\.venv\Scripts\python.exe -m scripts.fretboard_annotation init --video-directory runs\video-evidence\sources --frames-per-video 5 --output data\fretboard-keypoints
```

Frames retain their native source resolution. Five frames for each of roughly
210 videos produces about 1,050 candidate images. A diversity-selection stage
will replace uniform sampling before the production labeling pass; use the
current command only for testing the UI.

## Test the UI now on Windows

Create a small five-frame Yoyogi dataset:

```powershell
.\scripts\video-evidence\.venv\Scripts\python.exe -m scripts.fretboard_annotation init --video runs\video-evidence\sources\tab-0001\source.mkv --frames-per-video 5 --output data\fretboard-keypoints-test
```

Start the local browser UI:

```powershell
.\scripts\video-evidence\.venv\Scripts\python.exe -m scripts.fretboard_annotation serve --dataset data\fretboard-keypoints-test
```

The server binds only to `127.0.0.1` and opens
`http://127.0.0.1:8765/`. Stop it with `Ctrl+C`.

Controls:

- Mouse wheel: zoom
- Middle- or right-button drag: pan
- `1` through `6`: select a keypoint
- `V`: place visible points
- `O`: place occluded points
- `X`: clear the selected point
- `[` and `]`: previous and next frame
- **Complete**: mark the frame fully reviewed

## Stop and resume

Every point placement, clear operation, completion change, and note change is
saved to `annotations.json`. Writes use an atomic pending-file replacement.

It is safe to:

1. Leave frames incomplete.
2. Stop the server with `Ctrl+C`.
3. Shut down the computer.
4. Run the same `serve` command later.

The UI resumes from the existing dataset. Do not rerun `init` against the same
output directory; initialization deliberately refuses to overwrite it.

Back up the complete `data\fretboard-keypoints` folder periodically. The
sampled images, manifest, and annotations are all required.

## Annotate on macOS

Copy the repository working tree and the self-contained
`data/fretboard-keypoints` folder to the Mac. The original videos are not
required for annotation because sampled images are included.

Create an annotation environment from the repository root:

```bash
python3.12 -m venv .venv-annotation
source .venv-annotation/bin/activate
python -m pip install opencv-python
```

Start or resume the UI:

```bash
python -m scripts.fretboard_annotation serve --dataset data/fretboard-keypoints
```

Stop it with `Ctrl+C`, then copy the entire annotated dataset folder back to
the Windows repository.

## Export on Windows

After copying the annotated folder back:

```powershell
.\scripts\video-evidence\.venv\Scripts\python.exe -m scripts.fretboard_annotation export --dataset data\fretboard-keypoints
```

This writes YOLO pose labels under `labels/` and a Windows-local `data.yaml`.
Only frames marked complete are exported. Reviewed frames with no available
keypoints are exported as negative examples.

The six-keypoint labels use normalized coordinates, so moving the dataset
between macOS and Windows does not change them. Source images remain at native
resolution; model training can use 4K or a lower `imgsz` without relabeling.

## Expected effort

A production target may reach approximately 800-1,150 diverse images:

- 300-400 initial seed frames
- 400-600 model-assisted corrections
- 100-150 independently reviewed validation frames

Six-point labeling is expected to take roughly 12-24 total hours, spread
across resumable sessions. Repeated angles should be removed by the planned
diversity selector before starting the production pass.

Do not label the full target before testing whether the labels work. Stop after
the first 300-400 diverse frames and train a pilot detector. Add more labels
only for failure modes found on held-out videos.

## Next steps after manual annotation

### 1. Export completed frames on Windows

```powershell
.\scripts\video-evidence\.venv\Scripts\python.exe -m scripts.fretboard_annotation export --dataset data\fretboard-keypoints
```

The command writes `labels/` and `data.yaml`. Incomplete frames remain saved in
`annotations.json` but are not exported. Fully reviewed guitar-absent frames
are exported as negative examples.

### 2. Audit the exported dataset

Before training:

- Confirm all six point definitions use the same physical string ordering.
- Inspect every validation image independently from training images.
- Keep all frames from one source video in one split.
- Check that unavailable off-screen points were not guessed.
- Back up the complete dataset folder.

The current exporter validates coordinate ranges, point order at visible
anchors, and degenerate pairs. A separate visual dataset-audit command is
still to be implemented before production training.

### 3. Train a pilot YOLO pose model on Windows

The images and labels preserve native 4K coordinates. Full 4K training means
using `imgsz=3840`; it will normally require a CUDA GPU and a small batch.
Start with the nano pose architecture and batch size one, then increase the
batch only if GPU memory permits.

The repository does not yet expose a finalized training wrapper or pin an
Ultralytics training dependency. Until that is implemented and validated, do
not treat an ad-hoc training command as the reproducible production workflow.
The intended settings are:

- Task: pose
- Keypoint shape: `[6, 3]`
- Input size: `3840`
- Initial checkpoint: YOLO nano pose, or its architecture trained from scratch
- Batch size: begin at `1`
- Grouped train/validation/test splits from `data.yaml`
- Best checkpoint selected from held-out keypoint quality, not training loss

### 4. Evaluate on held-out videos

Do not judge the model from annotation images or random neighboring frames.
Run it on complete videos that were excluded from training and review:

- Detection coverage on frames where the fretboard is visible
- Nut, fret-12, and bridge endpoint accuracy
- Correct string-6/string-1 orientation
- Close, medium, and wide camera views
- Partial and occluded fretboards
- Scene-cut reacquisition
- Per-frame stability after optical-flow tracking
- Six-string and inferred-fret overlays

The detector is ready for corpus use only when its visual overlays are
trustworthy on held-out videos. A high YOLO confidence score alone is not
sufficient.

### 5. Add hard examples instead of more duplicates

After the pilot:

1. Run it over unlabelled candidate frames.
2. Collect low-confidence detections, rejected geometry, missed fretboards,
   scene-cut failures, and unstable tracks.
3. Cluster similar failures.
4. Label representative failures in the same UI.
5. Retrain and compare against the unchanged held-out set.

Repeat until additional labels no longer improve held-out video behavior.
Model-assisted pre-label import and diversity selection are still to be
implemented; manual annotations remain usable without them.

### 6. Adopt the trained checkpoint

The final local checkpoint must be recorded with:

- SHA-256 digest
- Keypoint order and visibility convention
- Training dataset digest
- Ultralytics and PyTorch versions
- Training configuration
- Held-out evaluation report

Do not overwrite an earlier checkpoint or reinterpret a six-point checkpoint
as the experimental 40-point model.

### 7. Process the video corpus

The production geometry pass will:

- Reset on every scene cut
- Use multiscale YOLO anchors
- Track geometry on intervening frames with optical flow
- Infer fret positions from nut, fret 12, and bridge geometry
- Infer six string paths from the outer-string anchors
- Retain MediaPipe hands on every frame
- Mask unavailable or invalid geometry
- Cache each completed video for safe resume

The six-point detector is not yet wired into the final paired-input producer.
Complete and validate that integration before processing all 8.5 hours.

### 8. Retrain and compare the transcriber

After geometry bundles are complete:

1. Regenerate the paired-video index.
2. Verify geometry and hand coverage for train and validation splits.
3. Train the joint audio/video transcriber from scratch.
4. Select the best checkpoint from held-out decoded-event metrics.
5. Compare audio-only, hands-only, and hands-plus-fretboard models on the same
   unseen songs.

Fretboard geometry is useful only if it improves held-out transcription; do
not assume higher visual coverage automatically improves tablature accuracy.
