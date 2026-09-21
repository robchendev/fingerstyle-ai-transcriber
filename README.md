# Fingerstyle Guitar Transcriber

Local PyTorch training and transcription for fingerstyle guitar. Audio and numeric hand observations produce musical predictions; deterministic rhythm, fingering and notation processing writes editable Guitar Pro `.gp` drafts.

Inputs are trimmed local performance videos. Training also needs the corresponding modern, single-track six-string GP scores. Transcription requires six pre-capo tuning pitches, a fixed full capo, BPM with beat unit, and time signature. Recordings, scores and generated artifacts stay in ignored local directories.

## Setup

Use Python 3.12 and install FFmpeg/FFprobe on `PATH`. Run these commands from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt -r requirements-rhythm.txt
.\.venv\Scripts\python.exe -m pip install --no-deps beat-this==1.1.0
python -m venv scripts\video-evidence\.venv
.\scripts\video-evidence\.venv\Scripts\python.exe -m pip install -r scripts\video-evidence\requirements.txt
.\scripts\video-evidence\.venv\Scripts\python.exe scripts\video-evidence\cli.py provision-hands
New-Item -ItemType Directory -Force .tools\models
Invoke-WebRequest 'https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt' -OutFile '.tools\models\beat-this-final0.ckpt'
```

The included transcription weights are at [`models/transcriber.pt`](models/transcriber.pt). MediaPipe and Beat This! are separate upstream models provisioned above; processing runs locally after setup.

## Transcribe

Supply your own modern GP template to retain its presentation settings. Generated files clear inherited artist, arranger and lyricist attribution fields in the score and stylesheets; the original template is not modified.

```powershell
.\.venv\Scripts\python.exe .\transcribe_video.py --video 'data\inputs\performance.mp4' --checkpoint 'models\transcriber.pt' --template 'data\template.gpt' --beat-checkpoint '.tools\models\beat-this-final0.ckpt' --plucking-screen-side left --note-cutoff 0.8 --x-cutoff 0.3 --output-directory 'runs\transcription' --tuning E2 A2 D3 G3 B3 E4 --capo 0 --bpm 120 --beat-unit 1/4 --time-signature 4/4
```

Tuning is in physical string 6-to-1 order. Set the plucking side to match the actual camera view. The pipeline extracts the soundtrack, detects scene cuts, tracks hands, runs the model and exports full-voice and single-voice GP drafts. Progress and any required alignment review are reported. It does not open Guitar Pro. Repeat unchanged arguments to resume; changed inputs require a new output directory.

Adjust export cutoffs without repeating tracking, inference or beat analysis:

```powershell
.\.venv\Scripts\python.exe .\reexport_gp.py --source-run 'runs\transcription' --note-cutoff 0.85 --x-cutoff 0.35 --output-directory 'runs\reexport'
```

Both launchers support `--dry-run`. Cutoffs filter uncalibrated model scores, not probabilities of correctness. Lowering a cutoff cannot recover candidates discarded during inference.

## Train

See [training with your own videos and scores](docs/TRAINING.md) for preparation, grouped train/validation splits, preflight and training from scratch. Use a training run's `best-events.pt` or `latest.pt` through `--checkpoint`; the included inference artifact is not a training-resume checkpoint.

## Limits

Outputs need musical review and editing. Missing or incorrect notes, fingerings, rhythms and techniques remain possible. Hand tracking supplies posture and motion, not exact fret/string contact. Scene cuts reset tracking; unavailable observations are masked. Experimental guitar calibration is not part of this workflow. The fixed 194-dimensional model interface retains masked reserved slots for checkpoint compatibility.

Partial capos, retuning during a recording and multi-instrument scores are unsupported. Single-voice drafts simplify overlapping parts; the full-voice export preserves the model's separate note durations but is not guaranteed to reproduce editorial voice choices.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py'
.\scripts\video-evidence\.venv\Scripts\python.exe -m unittest discover -s tests\video -t .
```

Video tests use the isolated vision environment. Core tests use synthetic fixtures and the inference/training environment.
