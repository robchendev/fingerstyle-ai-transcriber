"""Small, randomly initialized musical model and reusable event decoding.

Presence scores are uncalibrated, not reliable probabilities. Harmonics remain
positive-only; percussion uses only explicitly masked positive/negative labels.
The sparsity penalty assumes techniques are uncommon, not that unknowns are
negative, and does not supplement classes with confirmed percussion negatives.
This module neither trains a model nor writes notation or downloads weights.
"""

from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import TypedDict

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .technique_supervision import TECHNIQUE_DIRECTIONS, TECHNIQUE_TYPES


PERCUSSION_TYPES = ("wrist_thump", "thumb_slap", "percussive_hit")
HARMONIC_TYPES = ("Natural", "Artificial", "Tap", "Pinch")
HARMONIC_FRETS = (5, 7, 9, 12, 19, 24)
_HARMONIC_OFFSETS = dict(zip(HARMONIC_FRETS, (24, 19, 28, 12, 19, 24)))
PRESENCE_CALIBRATION = "unvalidated-model-scores"

LOSS_WEIGHTS = {
    "note_onset": 1.0,
    "fret": 1.0,
    "pitch": 1.0,
    "voice": 0.25,
    "duration_log": 0.25,
    "harmonic_positive": 0.5,
    "harmonic_kind": 0.25,
    "harmonic_node": 0.25,
    "percussion_positive": 5.0,
    "percussion_negative": 1.0,
    "technique_positive": 4.0,
    "technique_negative": 1.0,
    "technique_direction": 0.5,
    "technique_strings_positive": 2.0,
    "technique_strings_negative": 1.0,
}
LOSS_STAT_KEYS = (
    "note_onset_positive", "note_onset_negative", "fret", "pitch", "voice",
    "duration_log", "harmonic_positive", "harmonic_kind", "harmonic_node",
    "percussion_positive", "percussion_negative", "harmonic_sparsity", "percussion_sparsity",
    "technique_positive", "technique_negative", "technique_direction",
    "technique_strings_positive", "technique_strings_negative",
)
_BASE_HEADS = {
    "note_onset": "note_onset_logits",
    "fret": "fret_logits",
    "pitch": "pitch_logits",
    "voice": "voice_logits",
    "duration_log": "duration_log",
    "harmonic": "harmonic_logits",
    "harmonic_kind": "harmonic_kind_logits",
    "harmonic_node": "harmonic_node_logits",
    "percussion": "percussion_logits",
}
_TECHNIQUE_HEADS = {
    "technique": "technique_logits",
    "technique_direction": "technique_direction_logits",
    "technique_strings": "technique_strings_logits",
}
_HEADS = {**_BASE_HEADS, **_TECHNIQUE_HEADS}
_CATEGORICAL = ("fret", "pitch", "voice", "harmonic_kind", "harmonic_node", "technique_direction")


class LossStat(TypedDict):
    sum: float
    count: int


def _integer(name: str, value: int, minimum: int, maximum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be >= {minimum}" + (f" and <= {maximum}" if maximum is not None else ""))


def _real(name: str, value: float, minimum: float, maximum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be finite and >= {minimum}" + (f" and <= {maximum}" if maximum is not None else ""))


def _tensor(name: str, value: Tensor, shape: tuple[int, ...], *, dtype: torch.dtype | None = None,
            device: torch.device | None = None, floating: bool = False, finite: bool = False) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if floating and not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on {device}")
    if finite and not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class ModelConfig:
    architecture_version: int = 1
    n_mels: int = 96
    conditioning_dim: int = 12
    hidden_size: int = 128
    recurrent_layers: int = 2
    max_fret: int = 36
    max_voices: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in ("architecture_version", "n_mels", "hidden_size", "recurrent_layers", "max_voices"):
            _integer(name, getattr(self, name), 1)
        if self.architecture_version not in (1, 2):
            raise ValueError("architecture_version must be 1 or 2")
        _integer("conditioning_dim", self.conditioning_dim, 12, 12)
        _integer("max_fret", self.max_fret, 0, 127)
        _real("dropout", self.dropout, 0, 1)
        if self.dropout == 1:
            raise ValueError("dropout must be < 1")


class FingerstyleTranscriber(nn.Module):
    """Frequency-ordered convolutional frontend and conditioned bidirectional GRU.

    String axes are physical strings 6 through 1. Conditioning is the caller's
    twelve audio-independent tuning/capo/tempo/beat-unit/meter features, never
    ground-truth voices, attacks, score phase, or legend text.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be ModelConfig")
        self.config = config
        self.conv1 = nn.Conv2d(1, 16, (3, 5), stride=(1, 2), padding=(1, 2))
        self.conv2 = nn.Conv2d(16, 32, (3, 5), stride=(1, 2), padding=(1, 2))
        frequency_bins = (config.n_mels + 3) // 4
        self.projection = nn.Sequential(
            nn.Linear(32 * frequency_bins, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.recurrent = nn.GRU(
            config.hidden_size + config.conditioning_dim, config.hidden_size,
            num_layers=config.recurrent_layers, batch_first=True, bidirectional=True,
            dropout=config.dropout if config.recurrent_layers > 1 else 0.0,
        )
        self.head_shapes = {
            "note_onset_logits": (6,),
            "fret_logits": (6, config.max_fret + 1),
            "pitch_logits": (6, 128),
            "voice_logits": (6, config.max_voices),
            "duration_log": (6,),
            "harmonic_logits": (6,),
            "harmonic_kind_logits": (6, len(HARMONIC_TYPES)),
            "harmonic_node_logits": (6, len(HARMONIC_FRETS)),
            "percussion_logits": (len(PERCUSSION_TYPES),),
        }
        if config.architecture_version >= 2:
            self.head_shapes.update({
                "technique_logits": (len(TECHNIQUE_TYPES),),
                "technique_direction_logits": (len(TECHNIQUE_TYPES), len(TECHNIQUE_DIRECTIONS)),
                "technique_strings_logits": (len(TECHNIQUE_TYPES), 6),
            })
        self.heads = nn.ModuleDict({
            name: nn.Linear(2 * config.hidden_size, math.prod(shape))
            for name, shape in self.head_shapes.items()
        })

    def forward(self, features: Tensor, conditioning: Tensor, lengths: Tensor | None = None) -> dict[str, Tensor]:
        if not isinstance(features, Tensor):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim != 3 or min(features.shape[:2]) < 1:
            raise ValueError("features must have shape (B, T, n_mels), with B and T positive")
        batch, frames, _ = features.shape
        _tensor("features", features, (batch, frames, self.config.n_mels),
                dtype=self.conv1.weight.dtype, device=self.conv1.weight.device, floating=True, finite=True)
        _tensor("conditioning", conditioning, (batch, frames, self.config.conditioning_dim),
                dtype=features.dtype, device=features.device, floating=True, finite=True)
        if lengths is None:
            lengths = torch.full((batch,), frames, dtype=torch.long)
        _tensor("lengths", lengths, (batch,), dtype=torch.long)
        if ((lengths < 1) | (lengths > frames)).any().item():
            raise ValueError("lengths must be between 1 and T")
        valid = torch.arange(frames, device=features.device)[None, :] < lengths.to(features.device)[:, None]
        hidden = features.masked_fill(~valid[:, :, None], 0)[:, None, :, :]
        # Mask after each convolution: bias in padding must not leak back into
        # the last real frame through the next temporal kernel.
        hidden = F.gelu(self.conv1(hidden)).masked_fill(~valid[:, None, :, None], 0)
        hidden = F.gelu(self.conv2(hidden)).masked_fill(~valid[:, None, :, None], 0)
        hidden = self.projection(hidden.permute(0, 2, 1, 3).flatten(2))
        hidden = torch.cat((hidden, conditioning.masked_fill(~valid[:, :, None], 0)), dim=-1)
        packed = pack_padded_sequence(hidden, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.recurrent(packed)
        hidden, _ = pad_packed_sequence(packed, batch_first=True, total_length=frames)
        outputs = {}
        for name, head in self.heads.items():
            values = head(hidden).reshape(batch, frames, *self.head_shapes[name])
            if name == "duration_log":
                values = F.softplus(values)
            mask = valid.reshape(batch, frames, *([1] * len(self.head_shapes[name])))
            outputs[name] = values.masked_fill(~mask, 0)
        return outputs


def _validate_outputs(outputs: Mapping[str, Tensor], *, batched: bool) -> tuple[int, ...]:
    if not isinstance(outputs, Mapping):
        raise TypeError("outputs must be a mapping of head names to tensors")
    heads = dict(_BASE_HEADS)
    present_technique = set(_TECHNIQUE_HEADS.values()) & outputs.keys()
    if present_technique:
        heads.update(_TECHNIQUE_HEADS)
    missing = set(heads.values()) - outputs.keys()
    if missing:
        raise ValueError(f"outputs missing heads: {sorted(missing)}")
    onset = outputs["note_onset_logits"]
    if not isinstance(onset, Tensor):
        raise TypeError("note_onset_logits must be a torch.Tensor")
    rank = 3 if batched else 2
    if onset.ndim != rank or onset.shape[-1] != 6:
        raise ValueError(f"note_onset_logits must have shape {'(B, T, 6)' if batched else '(T, 6)'}")
    prefix = tuple(onset.shape[:-1])
    if batched and min(prefix) < 1:
        raise ValueError("batched outputs must have positive B and T")
    shapes = {
        "note_onset_logits": (6,), "pitch_logits": (6, 128),
        "duration_log": (6,), "harmonic_logits": (6,),
        "harmonic_kind_logits": (6, len(HARMONIC_TYPES)),
        "harmonic_node_logits": (6, len(HARMONIC_FRETS)),
        "percussion_logits": (len(PERCUSSION_TYPES),),
    }
    if present_technique:
        shapes.update({
            "technique_logits": (len(TECHNIQUE_TYPES),),
            "technique_direction_logits": (len(TECHNIQUE_TYPES), len(TECHNIQUE_DIRECTIONS)),
            "technique_strings_logits": (len(TECHNIQUE_TYPES), 6),
        })
    for name in ("fret_logits", "voice_logits"):
        value = outputs[name]
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != rank + 1 or value.shape[-1] < 1:
            raise ValueError(f"{name} must have a nonempty categorical axis")
        shapes[name] = (6, value.shape[-1])
    for name, shape in shapes.items():
        _tensor(name, outputs[name], prefix + shape, floating=True, finite=True,
                dtype=onset.dtype, device=onset.device)
    return prefix


def masked_loss(outputs: Mapping[str, Tensor], targets: Mapping[str, Tensor], masks: Mapping[str, Tensor],
                valid_frames: Tensor, *, sparsity_weight: float = 0.02) -> tuple[Tensor, dict[str, LossStat]]:
    """Return a weighted loss and additive component numerators/denominators.

    Aggregate stats by summing each component's ``sum`` and ``count`` across
    batches, then dividing only nonempty components. ``note_onset`` is the
    average of the nonempty positive and negative means, not their pooled mean.
    Percussion pools weighted BCE sums over weighted observed counts, using
    LOSS_WEIGHTS for positive/negative labels, not separate side means. Other
    supervised means use LOSS_WEIGHTS; both sparsity means use sparsity_weight.
    Duration uses smooth L1 in log1p(quarter-note duration) units.

    Masked-out target values are not inspected (including padded values).
    Harmonic masks may label only positives. Percussion masks may label 0 or 1;
    false masks remain unknown. Priors require positive evidence in this batch:
    percussion on valid frames of classes with no explicit negative labels,
    harmonic at explicitly known note attacks. These are uncalibrated sparsity
    assumptions, not supervised negative observations.
    """
    _real("sparsity_weight", sparsity_weight, 0)
    prefix = _validate_outputs(outputs, batched=True)
    reference = outputs["note_onset_logits"]
    _tensor("valid_frames", valid_frames, prefix, dtype=torch.bool, device=reference.device)
    if not isinstance(targets, Mapping) or not isinstance(masks, Mapping):
        raise TypeError("targets and masks must be mappings")
    active_heads = dict(_BASE_HEADS)
    if "technique_logits" in outputs:
        active_heads.update(_TECHNIQUE_HEADS)
    for label, mapping in (("targets", targets), ("masks", masks)):
        missing = set(active_heads) - mapping.keys()
        if missing:
            raise ValueError(f"{label} missing keys: {sorted(missing)}")
    effective = {}
    for name in active_heads:
        if name == "percussion":
            tail = (len(PERCUSSION_TYPES),)
        elif name in ("technique", "technique_direction"):
            tail = (len(TECHNIQUE_TYPES),)
        elif name == "technique_strings":
            tail = (len(TECHNIQUE_TYPES), 6)
        else:
            tail = (6,)
        shape = prefix + tail
        _tensor(f"targets[{name}]", targets[name], shape, device=reference.device,
                dtype=torch.long if name in _CATEGORICAL else None, floating=name not in _CATEGORICAL)
        _tensor(f"masks[{name}]", masks[name], shape, dtype=torch.bool, device=reference.device)
        effective[name] = masks[name] & valid_frames.reshape(
            *valid_frames.shape, *((1,) * (masks[name].ndim - 2))
        )
        selected = targets[name][effective[name]]
        if not torch.isfinite(selected).all().item():
            raise ValueError(f"supervised {name} targets must be finite")
        if name in _CATEGORICAL:
            classes = outputs[_HEADS[name]].shape[-1]
            if ((selected < 0) | (selected >= classes)).any().item():
                raise ValueError(f"supervised {name} targets must be in [0, {classes})")
        elif name == "duration_log":
            if (selected < 0).any().item():
                raise ValueError("supervised duration_log targets must be nonnegative")
        elif ((selected != 0) & (selected != 1)).any().item():
            raise ValueError(f"supervised {name} targets must be 0 or 1")
        if name == "harmonic" and (selected != 1).any().item():
            raise ValueError(f"{name} has positive-only supervision; masked negative targets are forbidden")

    # Empty slices keep every head connected to a differentiable zero without
    # computing any loss on unknown targets, or overflowing a full-logit sum.
    zero = sum(outputs[name].reshape(-1)[:0].sum() for name in active_heads.values())
    stats: dict[str, LossStat] = {name: {"sum": 0.0, "count": 0} for name in LOSS_STAT_KEYS}

    def term(name: str, values: Tensor) -> Tensor:
        count = values.numel()
        if not count:
            return zero
        numerator = values.sum()
        stats[name] = {"sum": float(numerator.detach().item()), "count": count}
        return numerator / count

    loss = zero
    onset_means = []
    for value, suffix in ((1, "positive"), (0, "negative")):
        mask = effective["note_onset"] & (targets["note_onset"] == value)
        if mask.any().item():
            onset_means.append(term(
                f"note_onset_{suffix}",
                F.binary_cross_entropy_with_logits(reference[mask], targets["note_onset"][mask], reduction="none"),
            ))
    if onset_means:
        loss = loss + LOSS_WEIGHTS["note_onset"] * sum(onset_means) / len(onset_means)
    for name in _CATEGORICAL:
        if name not in active_heads:
            continue
        mask = effective[name]
        if mask.any().item():
            values = F.cross_entropy(outputs[_HEADS[name]][mask], targets[name][mask], reduction="none")
            loss = loss + LOSS_WEIGHTS[name] * term(name, values)
    mask = effective["duration_log"]
    if mask.any().item():
        values = F.smooth_l1_loss(outputs["duration_log"][mask], targets["duration_log"][mask], reduction="none")
        loss = loss + LOSS_WEIGHTS["duration_log"] * term("duration_log", values)
    for name in ("harmonic", "percussion"):
        mask = effective[name]
        if not mask.any().item():
            continue
        predictions = outputs[_HEADS[name]]
        if name == "harmonic":
            values = F.binary_cross_entropy_with_logits(predictions[mask], targets[name][mask], reduction="none")
            loss = loss + LOSS_WEIGHTS["harmonic_positive"] * term("harmonic_positive", values)
            prior_mask = effective["note_onset"] & (targets["note_onset"] == 1)
        else:
            components = []
            for value, suffix in ((1, "positive"), (0, "negative")):
                selected = mask & (targets[name] == value)
                if not selected.any().item():
                    continue
                values = F.binary_cross_entropy_with_logits(
                    predictions[selected], targets[name][selected], reduction="none",
                )
                numerator, count = values.sum(), values.numel()
                stat_name = f"percussion_{suffix}"
                stats[stat_name] = {"sum": float(numerator.detach().item()), "count": count}
                components.append((numerator, count, LOSS_WEIGHTS[stat_name]))
            if len(components) == 1:
                # The class weight cancels; retain the positive-only BCE exactly.
                numerator, count, _ = components[0]
                loss = loss + numerator / count
            else:
                loss = loss + sum(total * weight for total, _, weight in components) / sum(
                    count * weight for _, count, weight in components
                )
            positive_classes = (mask & (targets[name] == 1)).any(dim=(0, 1))
            negative_classes = (mask & (targets[name] == 0)).any(dim=(0, 1))
            prior_classes = positive_classes & ~negative_classes
            prior_mask = valid_frames[:, :, None] & prior_classes[None, None, :]
        loss = loss + sparsity_weight * term(f"{name}_sparsity", predictions[prior_mask].sigmoid())
    if "technique" in active_heads:
        predictions = outputs["technique_logits"]
        components = []
        for value, suffix in ((1, "positive"), (0, "negative")):
            selected = effective["technique"] & (targets["technique"] == value)
            if not selected.any().item():
                continue
            values = F.binary_cross_entropy_with_logits(predictions[selected], targets["technique"][selected], reduction="none")
            numerator, count = values.sum(), values.numel()
            name = f"technique_{suffix}"
            stats[name] = {"sum": float(numerator.detach().item()), "count": count}
            components.append((numerator, count, LOSS_WEIGHTS[name]))
        if components:
            loss = loss + sum(total * weight for total, _, weight in components) / sum(count * weight for _, count, weight in components)
        selected = effective["technique_strings"]
        string_predictions = outputs["technique_strings_logits"]
        string_components = []
        for value, suffix in ((1, "positive"), (0, "negative")):
            mask = selected & (targets["technique_strings"] == value)
            if not mask.any().item():
                continue
            values = F.binary_cross_entropy_with_logits(string_predictions[mask], targets["technique_strings"][mask], reduction="none")
            numerator, count = values.sum(), values.numel()
            name = f"technique_strings_{suffix}"
            stats[name] = {"sum": float(numerator.detach().item()), "count": count}
            string_components.append((numerator, count, LOSS_WEIGHTS[name]))
        if string_components:
            loss = loss + sum(total * weight for total, _, weight in string_components) / sum(count * weight for _, count, weight in string_components)
    return loss, stats


def _temporal_peaks(scores: list[float], seconds: list[float], threshold: float, gap: float) -> list[int]:
    candidates = []
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[stop] == scores[start]:
            stop += 1
        if (scores[start] >= threshold and (start == 0 or scores[start] > scores[start - 1])
                and (stop == len(scores) or scores[start] > scores[stop])):
            candidates.append(start)
        start = stop
    accepted_times: list[float] = []
    accepted = []
    for index in sorted(candidates, key=lambda i: (-scores[i], seconds[i])):
        time = seconds[index]
        position = bisect_left(accepted_times, time)
        if position and time - accepted_times[position - 1] < gap:
            continue
        if position < len(accepted_times) and accepted_times[position] - time < gap:
            continue
        accepted_times.insert(position, time)
        accepted.append(index)
    return sorted(accepted)


@torch.no_grad()
def decode_events(outputs: Mapping[str, Tensor], frame_seconds: Tensor | Sequence[float], *,
                  tuning: Sequence[int], capo: int, onset_threshold: float = 0.5,
                  percussion_threshold: float = 0.5, harmonic_threshold: float = 0.5,
                  technique_threshold: float = 0.5,
                  min_gap_seconds: float = 0.04, max_duration_quarter: float = 64) -> dict:
    """Decode the complete unbatched timeline into hypotheses, never GP output.

    Local maxima collapse equal-score plateaus to their first frame. Greedy
    strongest-first NMS uses actual seconds independently per string/technique;
    equal scores favor earlier attacks. Percussion never supplies fingering.
    Below-threshold harmonic/percussion scores are not confirmed absence.
    """
    for name, value in (("onset_threshold", onset_threshold), ("percussion_threshold", percussion_threshold),
                        ("harmonic_threshold", harmonic_threshold), ("technique_threshold", technique_threshold)):
        _real(name, value, 0, 1)
    _real("min_gap_seconds", min_gap_seconds, 0)
    _real("max_duration_quarter", max_duration_quarter, 0)
    if max_duration_quarter == 0:
        raise ValueError("max_duration_quarter must be positive")
    _integer("capo", capo, 0, 127)
    if isinstance(tuning, (str, bytes)) or not isinstance(tuning, Sequence):
        raise TypeError("tuning must be a sequence of six integer MIDI pitches in physical 6-to-1 order")
    if len(tuning) != 6:
        raise ValueError("tuning must contain six open-string MIDI pitches")
    for value in tuning:
        _integer("tuning pitch", value, 0, 127 - capo)
    (frames,) = _validate_outputs(outputs, batched=False)
    if not isinstance(frame_seconds, Tensor):
        if isinstance(frame_seconds, (str, bytes)) or not isinstance(frame_seconds, Sequence):
            raise TypeError("frame_seconds must be a tensor or a sequence of real numbers")
        for value in frame_seconds:
            _real("frame_seconds entry", value, 0)
        frame_seconds = torch.tensor(frame_seconds, dtype=torch.float64)
    _tensor("frame_seconds", frame_seconds, (frames,), finite=True)
    if frame_seconds.dtype == torch.bool or frame_seconds.is_complex():
        raise TypeError("frame_seconds must contain real numeric seconds")
    if (frame_seconds < 0).any().item() or (frame_seconds[1:] <= frame_seconds[:-1]).any().item():
        raise ValueError("frame_seconds must be nonnegative and strictly increasing")
    if (outputs["duration_log"] < 0).any().item():
        raise ValueError("duration_log predictions must be nonnegative")
    seconds = frame_seconds.detach().cpu().tolist()
    values = {name: value.detach().cpu() for name, value in outputs.items() if name in _HEADS.values()}
    onsets = values["note_onset_logits"].sigmoid()
    harmonics = values["harmonic_logits"].sigmoid()
    percussion = values["percussion_logits"].sigmoid()
    categories = {name: values[_HEADS[name]].argmax(-1) for name in _CATEGORICAL if _HEADS[name] in values}
    notes = []
    duration_limit_log = math.log1p(max_duration_quarter)
    for axis in range(6):
        for frame in _temporal_peaks(onsets[:, axis].tolist(), seconds, onset_threshold, min_gap_seconds):
            fret = int(categories["fret"][frame, axis])
            pitch = int(categories["pitch"][frame, axis])
            base = int(tuning[axis]) + int(capo) + fret
            expected = base
            uncertainty = []
            harmonic = None
            harmonic_score = float(harmonics[frame, axis])
            if harmonic_score >= harmonic_threshold:
                kind = HARMONIC_TYPES[int(categories["harmonic_kind"][frame, axis])]
                node = HARMONIC_FRETS[int(categories["harmonic_node"][frame, axis])]
                harmonic = {"type": kind, "fret": node, "confidence": harmonic_score}
                expected = (int(tuning[axis]) + int(capo) if kind == "Natural" else base) + _HARMONIC_OFFSETS[node]
                uncertainty.append("harmonic_presence_uncalibrated")
            if expected != pitch:
                uncertainty.append("sounding_pitch_harmonic_mismatch" if harmonic else "sounding_pitch_fret_mismatch")
            if expected > 127:
                uncertainty.append("expected_pitch_outside_midi_range")
            duration_log = float(values["duration_log"][frame, axis])
            duration = math.expm1(duration_log) if duration_log <= duration_limit_log else float(max_duration_quarter)
            if duration_log > duration_limit_log:
                uncertainty.append("duration_clipped")
            notes.append({
                "onsetSeconds": float(seconds[frame]),
                "string": 6 - axis,
                "fret": fret,
                "soundingPitchMidi": pitch,
                "fretBasePitchMidi": base,
                "expectedSoundingPitchMidi": expected,
                "voiceIndex": int(categories["voice"][frame, axis]),
                "notatedDurationQuarter": duration,
                "harmonic": harmonic,
                "confidence": float(onsets[frame, axis]),
                "uncertainty": uncertainty,
            })
    gestures = []
    for axis, technique in enumerate(PERCUSSION_TYPES):
        for frame in _temporal_peaks(percussion[:, axis].tolist(), seconds, percussion_threshold, min_gap_seconds):
            gestures.append({
                "technique": technique,
                "onsetSeconds": float(seconds[frame]),
                "confidence": float(percussion[frame, axis]),
                "presenceCalibration": PRESENCE_CALIBRATION,
            })
    techniques = []
    if "technique_logits" in values:
        technique_scores = values["technique_logits"].sigmoid()
        membership = values["technique_strings_logits"].sigmoid()
        directions = categories["technique_direction"]
        for axis, technique in enumerate(TECHNIQUE_TYPES):
            for frame in _temporal_peaks(technique_scores[:, axis].tolist(), seconds, technique_threshold, min_gap_seconds):
                strings = [6 - string_axis for string_axis in range(6) if float(membership[frame, axis, string_axis]) >= .5]
                techniques.append({
                    "technique": technique,
                    "direction": TECHNIQUE_DIRECTIONS[int(directions[frame, axis])],
                    "strings": sorted(strings, reverse=True),
                    "onsetSeconds": float(seconds[frame]),
                    "confidence": float(technique_scores[frame, axis]),
                    "stringMembershipConfidence": {
                        str(6 - string_axis): float(membership[frame, axis, string_axis])
                        for string_axis in range(6)
                    },
                    "presenceCalibration": PRESENCE_CALIBRATION,
                })
    notes.sort(key=lambda note: (note["onsetSeconds"], -note["string"]))
    gestures.sort(key=lambda gesture: (gesture["onsetSeconds"], PERCUSSION_TYPES.index(gesture["technique"])))
    techniques.sort(key=lambda event: (event["onsetSeconds"], TECHNIQUE_TYPES.index(event["technique"])))
    return {
        "notes": notes,
        "percussion": gestures,
        **({"techniques": techniques} if "technique_logits" in values else {}),
        "policy": {
            "output": "musical-event-hypotheses-only; no notation writer",
            "stringAxis": "physical-6-to-1",
            "confidence": "uncalibrated model scores, not reliable probabilities",
            "presenceCalibration": PRESENCE_CALIBRATION,
            "presenceAbsence": "below threshold means not emitted, not a confirmed negative",
            "sparsityPrior": "weak uncommon-technique assumption, not supervised negative labels",
            "peakSelection": "first plateau frame; strongest-first NMS; earlier equal-score peaks win",
            "pitchConstraints": "preserve predicted sounding pitches; report inconsistencies without repair",
            "duration": "notated quarter-note units, not acoustic release or wall-clock duration",
            "percussionFingering": "not inferred from symbolic percussion carriers",
        },
    }
