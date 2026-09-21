# Fingerstyle Guitar Transcriber

Turn fingerstyle guitar performances into editable Guitar Pro `.gp` drafts using audio and hand tracking. Run the included PyTorch model locally or train one on your own performances and scores.

Use trimmed local videos. Training also needs matching single-track, six-string scores in modern `.gp` format. For transcription, supply the tuning before applying a capo, capo fret, tempo, beat unit and time signature. Local recordings, scores and outputs are excluded from Git.

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

The included transcription model is at [`models/transcriber.pt`](models/transcriber.pt). Setup also downloads MediaPipe for hand tracking and Beat This! for beat detection. Processing runs locally.

## Transcribe

Supply a GP or GPT template for the page layout. Exported files clear the template's artist, arranger and lyricist credits.

```powershell
.\.venv\Scripts\python.exe .\transcribe_video.py --video 'data\inputs\performance.mp4' --checkpoint 'models\transcriber.pt' --template 'data\template.gpt' --beat-checkpoint '.tools\models\beat-this-final0.ckpt' --plucking-screen-side left --note-cutoff 0.8 --x-cutoff 0.3 --output-directory 'runs\transcription' --tuning E2 A2 D3 G3 B3 E4 --capo 0 --bpm 120 --beat-unit 1/4 --time-signature 4/4
```

List tuning from string 6 to string 1. Set the plucking side to its position on screen, not the player's handedness. The command extracts audio, tracks hands and exports full-voice and single-voice GP drafts. It reports progress and any timing corrections needed. Repeat the same command to resume; changed inputs require a new output directory.

Adjust export cutoffs without rerunning the model or processing the video:

```powershell
.\.venv\Scripts\python.exe .\reexport_gp.py --source-run 'runs\transcription' --note-cutoff 0.85 --x-cutoff 0.35 --output-directory 'runs\reexport'
```

Both commands support `--dry-run` to preview their settings. A cutoff of `0.8` does not mean 80% accuracy. Lowering a re-export cutoff cannot recover notes already discarded by the model.

## Train

See [training with your own videos and scores](docs/TRAINING.md) for data preparation, review and training. To transcribe with your trained model, pass its `best-events.pt` or `latest.pt` to `--checkpoint`. The bundled model cannot resume training.

## Limits

Review and edit the generated notes, fingerings, rhythms and techniques. Hand tracking measures posture and movement, not exact string or fret contact.

Partial capos, retuning during a recording and multi-instrument scores are unsupported. Single-voice drafts simplify overlapping parts. Full-voice drafts retain overlapping note durations, but may group them differently from a manually written score.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py'
.\scripts\video-evidence\.venv\Scripts\python.exe -m unittest discover -s tests\video -t .
```
