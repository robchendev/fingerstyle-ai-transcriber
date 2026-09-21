"""Joint audio, optional guitar context and independent numeric hand features.

Known playing roles stay separate; anonymous tracks share an encoder and pool
symmetrically before every musical head. No RGB or inferred role labels enter.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from .transcriber_model import FingerstyleTranscriber, _integer, _real, _tensor
from .video_features import OBSERVATION_SLICES, SCHEMA_VERSION, STRUCTURED_DIM, VELOCITY_SLICES, VIEW_ORDER


_VIDEO_FIELDS = frozenset({
    "technique_available", "segment_id", "structured", "structured_available", "frame_indices",
})


@dataclass(frozen=True)
class VideoConfig:
    hidden_size: int = 64
    temporal_layers: int = 1
    structured_dim: int = STRUCTURED_DIM
    input_schema_version: int = SCHEMA_VERSION
    architecture_version: int = 4
    modality_dropout: float = 0.2

    def __post_init__(self):
        _integer("hidden_size", self.hidden_size, 1)
        _integer("temporal_layers", self.temporal_layers, 1)
        _integer("structured_dim", self.structured_dim, STRUCTURED_DIM, STRUCTURED_DIM)
        _integer("input_schema_version", self.input_schema_version, SCHEMA_VERSION, SCHEMA_VERSION)
        _integer("architecture_version", self.architecture_version, 4, 4)
        _real("modality_dropout", self.modality_dropout, 0, 1)


def _segmented_temporal(values, active, segment_ids, recurrent):
    """Pack independent contiguous runs; neither GRU direction crosses a gap."""
    runs = []
    for batch_index, (flags, segments) in enumerate(zip(
        active.cpu().tolist(), segment_ids.cpu().tolist(),
    )):
        start = None
        for frame in range(len(flags) + 1):
            if start is not None and (
                frame == len(flags) or not flags[frame] or segments[frame] != segments[frame - 1]
            ):
                runs.append((batch_index, start, frame))
                start = None
            if frame < len(flags) and flags[frame] and start is None:
                start = frame
    result = values.new_zeros(*values.shape[:2], 2 * recurrent.hidden_size)
    if runs:
        sequences = [values[batch, start:stop] for batch, start, stop in runs]
        packed, _ = recurrent(pack_sequence(sequences, enforce_sorted=False))
        hidden, _ = pad_packed_sequence(packed, batch_first=True)
        for index, (batch, start, stop) in enumerate(runs):
            result[batch, start:stop] = hidden[index, :stop - start]
    return result


def _technique_structure_mask(structured_available, segment_ids):
    active = structured_available.any(dim=-1)
    observed = torch.zeros_like(active)
    for start, stop in OBSERVATION_SLICES:
        observed |= structured_available[:, :, start:stop].any(dim=-1)
    starts = torch.ones_like(active)
    starts[:, 1:] = ~active[:, :-1] | (segment_ids[:, 1:] != segment_ids[:, :-1])
    positions = torch.arange(active.shape[1], device=active.device).expand(active.shape[0], -1)
    run_start = torch.where(starts, positions, 0).cummax(dim=1).values
    last_observed = torch.where(observed, positions, -1).cummax(dim=1).values
    # A derived velocity cannot start a segment or establish its own missing
    # predecessor. This also excludes leading velocity-only placeholder rows.
    active = active & (last_observed >= run_start)
    continuation = torch.zeros_like(active)
    continuation[:, 1:] = active[:, 1:] & active[:, :-1] & (segment_ids[:, 1:] == segment_ids[:, :-1])
    result = structured_available.clone()
    for start, stop in VELOCITY_SLICES:
        result[:, :, start:stop] &= continuation[:, :, None]
    return result


class _StructuredBranch(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        self.structure_encoder = nn.Sequential(
            nn.Linear(2 * config.structured_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.recurrent = nn.GRU(
            hidden, hidden, num_layers=config.temporal_layers,
            batch_first=True, bidirectional=True,
        )

    def forward(self, structured, structured_available, segment_ids, gate, *, reset_velocity=False):
        structured_available = structured_available & gate[:, :, None]
        if reset_velocity:
            structured_available = _technique_structure_mask(structured_available, segment_ids)
        active = structured_available.any(dim=-1)
        structured = structured.masked_fill(~structured_available, 0)
        encoded_structure = self.structure_encoder(torch.cat((
            structured, structured_available.to(structured.dtype),
        ), dim=-1)).masked_fill(~active[:, :, None], 0)
        return _segmented_temporal(encoded_structure, active, segment_ids, self.recurrent), active


class AudioVideoTranscriber(nn.Module):
    """Trainable audio and role-conditioned video fused before all musical heads.

    Missing views have independent presence masks. Correspondence exclusions
    gate plucking and anonymous tracks, not fretting evidence or audio labels.
    """

    def __init__(self, audio: FingerstyleTranscriber, video_config: VideoConfig):
        super().__init__()
        if not isinstance(audio, FingerstyleTranscriber):
            raise TypeError("audio must be FingerstyleTranscriber")
        if not isinstance(video_config, VideoConfig):
            raise TypeError("video_config must be VideoConfig")
        self.audio = audio
        self.config = audio.config
        self.video_config = video_config
        self.head_shapes = dict(audio.head_shapes)
        self.position_branch = _StructuredBranch(video_config)
        self.technique_branch = _StructuredBranch(video_config)
        self.anonymous_branch = _StructuredBranch(video_config)
        audio_dimensions = 2 * self.config.hidden_size
        self.fusion = nn.Sequential(
            nn.Linear(audio_dimensions + 6 * video_config.hidden_size + 3, audio_dimensions),
            nn.GELU(),
            nn.Linear(audio_dimensions, audio_dimensions),
        )

    def _validate_video(self, video, batch, frames, device):
        if not isinstance(video, Mapping) or set(video) != _VIDEO_FIELDS:
            raise ValueError(f"Numeric video schema {SCHEMA_VERSION} requires exactly {sorted(_VIDEO_FIELDS)}; RGB fields are unsupported")
        structured = video["structured"]
        if not isinstance(structured, Tensor):
            raise TypeError("video.structured must be a torch.Tensor")
        if structured.ndim != 4 or structured.shape[1] < 1:
            raise ValueError(f"video.structured must have shape (B, V, {len(VIEW_ORDER)}, {STRUCTURED_DIM}), V >= 1")
        count = structured.shape[1]
        dimensions = self.video_config.structured_dim
        shapes = {
            "technique_available": ((batch, count), torch.bool),
            "segment_id": ((batch, count, len(VIEW_ORDER)), torch.int64),
            "structured": ((batch, count, len(VIEW_ORDER), dimensions), torch.float32),
            "structured_available": ((batch, count, len(VIEW_ORDER), dimensions), torch.bool),
            "frame_indices": ((batch, frames), torch.int64),
        }
        for name, (shape, dtype) in shapes.items():
            _tensor(f"video.{name}", video[name], shape, dtype=dtype, device=device, finite=True)
        if video["structured_available"][:, :, 2:, 186:].any().item():
            raise ValueError("Coarse instrument context requires known playing roles, not anonymous views")
        available = video["structured_available"].any(dim=-1)
        segments = video["segment_id"]
        if (segments < -1).any().item() or not torch.equal(segments >= 0, available):
            raise ValueError("video.segment_id must be nonnegative exactly where a view has evidence, else -1")
        indices = video["frame_indices"]
        if ((indices < -1) | (indices >= count)).any().item():
            raise ValueError("video.frame_indices must be -1 or a valid video frame index")

    def forward(self, features, conditioning, lengths=None, *, video=None):
        audio_hidden, valid = self.audio.encode(features, conditioning, lengths)
        batch, frames = valid.shape
        visual = audio_hidden.new_zeros(batch, frames, len(VIEW_ORDER), 2 * self.video_config.hidden_size)
        present = torch.zeros(batch, frames, len(VIEW_ORDER), dtype=torch.bool, device=features.device)
        if video is not None:
            self._validate_video(video, batch, frames, features.device)
            indices = video["frame_indices"]
            mapped = (indices >= 0) & valid
            for view, branch in enumerate((
                self.position_branch, self.technique_branch, self.anonymous_branch, self.anonymous_branch,
            )):
                gate = torch.ones_like(video["technique_available"]) if view == 0 else video["technique_available"]
                hidden, active = branch(
                    video["structured"][:, :, view], video["structured_available"][:, :, view],
                    video["segment_id"][:, :, view], gate, reset_velocity=view != 0,
                )
                visual[:, :, view] = hidden.gather(
                    1, indices.clamp_min(0)[:, :, None].expand(-1, -1, hidden.shape[-1]),
                )
                present[:, :, view] = mapped & active.gather(1, indices.clamp_min(0))
            has_video = present.any(dim=(1, 2))
            if self.training and self.video_config.modality_dropout and has_video.any().item():
                dropped = (torch.rand(batch, device=features.device) < self.video_config.modality_dropout) & has_video
                present = present & ~dropped[:, None, None]
        visual = visual.masked_fill(~present[:, :, :, None], 0)
        anonymous_count = present[:, :, 2:].sum(dim=-1, keepdim=True).to(audio_hidden.dtype)
        anonymous = visual[:, :, 2:].sum(dim=2) / anonymous_count.clamp_min(1)
        fused = audio_hidden + self.fusion(torch.cat((
            audio_hidden, visual[:, :, :2].flatten(2), anonymous,
            present[:, :, :2].to(audio_hidden.dtype), anonymous_count,
        ), dim=-1))
        return self.audio.decode_hidden(fused, valid)
