"""Local Beat This! inference without the package's torchaudio frontend."""

from collections import OrderedDict
from fractions import Fraction
import hashlib
import importlib.metadata
import inspect
import math
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch

from .transcriber_audio import HarnessError


SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 441
N_MELS = 128
F_MIN = 30.0
F_MAX = 11000.0
CHUNK_FRAMES = 1500
BORDER_FRAMES = 6
MODEL_NAME = "Beat This! 1.1 final0"


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _slaney_hz_to_mel(frequencies):
    frequencies = torch.as_tensor(frequencies, dtype=torch.float64)
    linear_spacing = 200 / 3
    minimum_log_hz = 1000.0
    minimum_log_mel = minimum_log_hz / linear_spacing
    log_step = math.log(6.4) / 27
    linear = frequencies / linear_spacing
    logarithmic = minimum_log_mel + torch.log(frequencies.clamp_min(minimum_log_hz) / minimum_log_hz) / log_step
    return torch.where(frequencies >= minimum_log_hz, logarithmic, linear)


def _slaney_mel_to_hz(mels):
    mels = torch.as_tensor(mels, dtype=torch.float64)
    linear_spacing = 200 / 3
    minimum_log_hz = 1000.0
    minimum_log_mel = minimum_log_hz / linear_spacing
    log_step = math.log(6.4) / 27
    linear = mels * linear_spacing
    logarithmic = minimum_log_hz * torch.exp(log_step * (mels - minimum_log_mel))
    return torch.where(mels >= minimum_log_mel, logarithmic, linear)


def mel_filter_bank():
    mel_edges = torch.linspace(float(_slaney_hz_to_mel(F_MIN)), float(_slaney_hz_to_mel(F_MAX)), N_MELS + 2, dtype=torch.float64)
    edges = _slaney_mel_to_hz(mel_edges)
    frequencies = torch.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1, dtype=torch.float64)
    left, center, right = edges[:-2], edges[1:-1], edges[2:]
    rising = (frequencies[:, None] - left) / (center - left)
    falling = (right - frequencies[:, None]) / (right - center)
    return torch.minimum(rising, falling).clamp_min(0).to(torch.float32)


def log_mel_spectrogram(signal):
    if not isinstance(signal, np.ndarray) or signal.ndim != 1 or signal.dtype.kind != "f" or not len(signal) or not np.isfinite(signal).all():
        raise HarnessError("Beat tracking requires a nonempty finite floating mono signal.")
    waveform = torch.from_numpy(np.asarray(signal, dtype=np.float32))
    window = torch.hann_window(N_FFT)
    spectrum = torch.stft(waveform, N_FFT, HOP_LENGTH, window=window, center=True, pad_mode="reflect", return_complex=True)
    magnitude = spectrum.abs() / math.sqrt(N_FFT)
    mel = magnitude.T @ mel_filter_bank()
    result = torch.log1p(1000 * mel)
    if not torch.isfinite(result).all():
        raise HarnessError("Beat-tracking features are nonfinite.")
    return result


def _read_channel(path):
    path = Path(path).resolve()
    with sf.SoundFile(path) as stream:
        if not 0 < len(stream) <= stream.samplerate * 900 or not 1 <= stream.channels <= 8:
            raise HarnessError("Beat tracking accepts nonempty audio up to fifteen minutes and eight channels.")
        source = stream.read(dtype="float32", always_2d=True)
        rate = stream.samplerate
    if not np.isfinite(source).all():
        raise HarnessError("Beat-tracking audio contains nonfinite samples.")
    energies = np.mean(np.square(source, dtype=np.float64), axis=0)
    channel = int(np.argmax(energies))
    mixed = source.mean(axis=1)
    mixed_energy = float(np.mean(np.square(mixed, dtype=np.float64)))
    if mixed_energy >= .25 * energies[channel]:
        signal = mixed
        mode = "channel-mean"
        selected = None
    else:
        signal = source[:, channel]
        mode = "highest-energy-channel"
        selected = channel
    if energies[channel] <= 1e-16:
        raise HarnessError("Beat-tracking audio is silent.")
    if rate != SAMPLE_RATE:
        divisor = math.gcd(rate, SAMPLE_RATE)
        signal = resample_poly(signal, SAMPLE_RATE // divisor, rate // divisor).astype(np.float32)
    return signal, rate, source.shape[1], selected, energies.tolist(), mixed_energy, mode


def _load_model(checkpoint_path, device):
    try:
        from beat_this.model.beat_tracker import BeatThis
    except (ImportError, ModuleNotFoundError) as error:
        raise HarnessError("Beat This! model runtime is not installed; install the rhythm dependencies.") from error
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("hyper_parameters"), dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise HarnessError("Beat This! checkpoint has an unsupported structure.")
    accepted = set(inspect.signature(BeatThis).parameters)
    parameters = {key: value for key, value in checkpoint["hyper_parameters"].items() if key in accepted}
    model = BeatThis(**parameters)
    state = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        if not isinstance(key, str) or not key.startswith("model.") or not isinstance(value, torch.Tensor):
            continue
        state[key.removeprefix("model.").replace("_orig_mod.", "")] = value
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), parameters


def _chunks(features):
    if len(features) <= CHUNK_FRAMES:
        return [(-BORDER_FRAMES, torch.nn.functional.pad(features, (0, 0, BORDER_FRAMES, BORDER_FRAMES)))]
    step = CHUNK_FRAMES - 2 * BORDER_FRAMES
    starts = list(range(-BORDER_FRAMES, len(features) - BORDER_FRAMES, step))
    starts[-1] = len(features) - (CHUNK_FRAMES - BORDER_FRAMES)
    result = []
    for start in starts:
        chunk = features[max(start, 0):min(start + CHUNK_FRAMES, len(features))]
        chunk = torch.nn.functional.pad(chunk, (0, 0, max(0, -start), max(0, min(BORDER_FRAMES, start + CHUNK_FRAMES - len(features)))))
        result.append((start, chunk))
    return result


def _predict(model, features, device):
    beat = torch.full((len(features),), -1000.0)
    downbeat = torch.full((len(features),), -1000.0)
    with torch.inference_mode():
        for start, chunk in reversed(_chunks(features)):
            output = model(chunk.unsqueeze(0).to(device))
            left = BORDER_FRAMES
            right = len(output["beat"][0]) - BORDER_FRAMES
            destination_start = max(0, start + BORDER_FRAMES)
            destination_stop = min(len(features), start + right)
            count = destination_stop - destination_start
            if count <= 0:
                continue
            beat[destination_start:destination_stop] = output["beat"][0, left:left + count].cpu()
            downbeat[destination_start:destination_stop] = output["downbeat"][0, left:left + count].cpu()
    if torch.any(beat <= -999) or torch.any(downbeat <= -999):
        raise HarnessError("Beat This! inference left uncovered frames.")
    return beat, downbeat


def _peak_times(logits):
    peaks = logits.masked_fill(logits != torch.nn.functional.max_pool1d(logits[None], 7, 1, 3)[0], -1000) > 0
    indices = torch.nonzero(peaks).flatten().tolist()
    grouped = []
    for index in indices:
        if grouped and index - grouped[-1][-1] <= 1:
            grouped[-1].append(index)
        else:
            grouped.append([index])
    return np.asarray([sum(group) / len(group) * HOP_LENGTH / SAMPLE_RATE for group in grouped], dtype=np.float64)


def track_beats(audio_path, checkpoint_path, *, device="cpu"):
    audio_path = Path(audio_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise HarnessError(f"Beat This! checkpoint does not exist: {checkpoint_path}")
    audio_hash = _sha256(audio_path)
    checkpoint_hash = _sha256(checkpoint_path)
    signal, source_rate, channels, selected_channel, energies, mixed_energy, channel_mode = _read_channel(audio_path)
    features = log_mel_spectrogram(signal)
    resolved = torch.device(device)
    if resolved.type != "cpu" and resolved.type != "cuda":
        raise HarnessError("Beat tracking supports CPU or CUDA.")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise HarnessError("CUDA was requested for beat tracking but is unavailable.")
    model, parameters = _load_model(checkpoint_path, resolved)
    beat_logits, downbeat_logits = _predict(model, features, resolved)
    beats = _peak_times(beat_logits)
    downbeats = _peak_times(downbeat_logits)
    if not len(beats):
        raise HarnessError("Beat This! did not detect any beats.")
    for index, value in enumerate(downbeats):
        nearest = int(np.argmin(np.abs(beats - value)))
        downbeats[index] = beats[nearest]
    downbeats = np.unique(downbeats)
    if _sha256(audio_path) != audio_hash or _sha256(checkpoint_path) != checkpoint_hash:
        raise HarnessError("Beat-tracking input changed during inference.")
    return {
        "schemaVersion": 1,
        "kind": "audio-beat-evidence",
        "visibility": "private",
        "trainingPerformed": False,
        "model": MODEL_NAME,
        "modelPackageVersion": importlib.metadata.version("beat-this"),
        "checkpointSha256": checkpoint_hash,
        "audioSha256": audio_hash,
        "sourceSampleRate": source_rate,
        "sourceChannels": channels,
        "selectedChannelIndex": selected_channel,
        "channelSelection": channel_mode,
        "channelMeanSquare": energies,
        "downmixMeanSquare": mixed_energy,
        "analysis": {
            "sampleRate": SAMPLE_RATE,
            "fftSize": N_FFT,
            "hopLength": HOP_LENGTH,
            "framesPerSecond": SAMPLE_RATE / HOP_LENGTH,
            "melBands": N_MELS,
            "frequencyRangeHz": [F_MIN, F_MAX],
            "melScale": "Slaney",
            "spectrogramNormalization": "frame_length",
            "power": 1,
            "logMultiplier": 1000,
            "modelParameters": parameters,
        },
        "beatSeconds": beats.tolist(),
        "downbeatSeconds": downbeats.tolist(),
        "beatCount": len(beats),
        "downbeatCount": len(downbeats),
        "policy": "Ordinary multichannel audio is averaged like the pretrained frontend. If downmix energy falls below one quarter of the strongest channel, that channel is used to avoid phase cancellation. Beat/downbeat evidence is not a score or training label.",
    }
