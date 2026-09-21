# Transcription model

`transcriber.pt` is the bundled model with 1,468,689 parameters. It transcribes audio with supporting hand-tracking features.

The transcription command uses this model by default. Set `--checkpoint` to use your own trained model. This file cannot resume training; see the [training instructions](../docs/TRAINING.md).

Download the separate hand-tracking and beat-detection models using the [setup instructions](../README.md#setup).
