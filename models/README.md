# Transcription model

`transcriber.pt` is the bundled joint audio and numeric-hand inference model. It has 1,468,689 parameters and uses architecture 4 with the fixed 194-feature video interface.

The artifact contains tensor weights and audio/model/video configuration only. It does not contain recordings, tablature, dataset manifests, local paths, optimizer state or training history.

The root transcription launcher uses this model by default. Override `--checkpoint` to use a model from your own training run. This inference-only artifact cannot resume training; resume an unchanged training run from its `latest.pt`.

The model produces drafts, not guaranteed correct scores. Tuning, capo, tempo, beat unit and time signature remain required. Guitar-relative geometry and coarse-context feature slots remain masked; hand-local posture and motion provide supporting evidence.

MediaPipe hand tracking and Beat This! beat detection use separately provisioned upstream weights described in the root README. They are not included in `transcriber.pt`.
