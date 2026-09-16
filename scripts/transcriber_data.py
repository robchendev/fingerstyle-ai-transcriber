"""Load approved private windows and preserve canonical supervision."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import scipy
import soundfile
import torch
from torch.utils.data import Dataset, Sampler

from .canonical_events import fraction
from .dataset_io import ROOT, sha256
from .dataset_release import release_path, validate_release
from .transcriber_audio import HarnessError, audio_features, conditioning_features, read_audio_window


class EpochShuffleSampler(Sampler):
    def __init__(self, data_source, seed=17):
        self.data_source, self.seed, self.epoch = data_source, seed, 0
        if type(seed) is not int or not 0 <= seed < 2 ** 63:
            raise HarnessError("Sampler seed must be a nonnegative 63-bit integer.")

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0 or epoch + self.seed >= 2 ** 63:
            raise HarnessError("Invalid deterministic sampler epoch.")
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self):
        return len(self.data_source)


def collate_windows(items):
    if not items:
        raise HarnessError("Cannot collate an empty batch.")
    lengths = torch.tensor([len(item["features"]) for item in items], dtype=torch.long)
    longest = int(lengths.max())

    def padded(values):
        output = values[0].new_zeros((len(values), longest, *values[0].shape[1:]))
        for index, value in enumerate(values):
            output[index, :len(value)] = value
        return output

    return {
        "features": padded([item["features"] for item in items]),
        "conditioning": padded([item["conditioning"] for item in items]),
        "lengths": lengths,
        "valid_frames": torch.arange(longest)[None, :] < lengths[:, None],
        "targets": {key: padded([item["targets"][key] for item in items]) for key in items[0]["targets"]},
        "masks": {key: padded([item["masks"][key] for item in items]) for key in items[0]["masks"]},
        "metadata": [item["metadata"] for item in items],
    }


def training_conditioning(record, clip_times):
    data, candidate = record["data"], record["candidate"]
    mapping = candidate["denseMapping"]
    audio = np.array([point["clipSeconds"] for point in mapping])
    quarters = np.array([point["scoreQuarter"] for point in mapping])
    if np.any(clip_times < audio[0]) or np.any(clip_times > audio[-1]) or np.any(np.diff(audio) < 0) or not np.isfinite(audio).all():
        raise HarnessError("Training frames extend outside the approved candidate mapping.")
    positions = np.interp(clip_times, audio, quarters)
    visits = data.labels["measureVisits"]
    starts = [float(fraction(visit["onsetQuarter"], "measure start")) for visit in visits]
    tempos = [{
        "position": starts[event["measureIndex"]] + float(fraction(event["offsetQuarter"], "tempo offset")),
        "bpm": event["bpm"], "beatUnit": event["beatUnit"], "linear": event["linear"],
    } for event in data.normalization["normalizedTempoEvents"]]
    timing = data.labels["conditioning"]["providedTiming"]
    meters = [{"position": 0., "timeSignature": timing["timeSignature"]}]
    meters.extend({"position": starts[event["measureIndex"]], "timeSignature": event["timeSignature"]} for event in timing["sourceTimeSignatureChanges"])
    instrument = data.labels["conditioning"]["instrument"]
    return conditioning_features(instrument["openStringMidi"], instrument["capoFret"], tempos, meters, positions)


def encode_targets(window, canonical, frame_times, model_config, *, negative_onsets_allowed):
    from .transcriber_model import HARMONIC_FRETS, HARMONIC_TYPES, PERCUSSION_TYPES

    count = len(frame_times)
    targets = {name: torch.zeros((count, 6), dtype=torch.long if name in {"fret", "pitch", "voice", "harmonic_kind", "harmonic_node"} else torch.float32) for name in ("note_onset", "fret", "pitch", "voice", "duration_log", "harmonic", "harmonic_kind", "harmonic_node")}
    targets["percussion"] = torch.zeros((count, len(PERCUSSION_TYPES)))
    masks = {name: torch.zeros_like(value, dtype=torch.bool) for name, value in targets.items()}
    hop = frame_times[1] - frame_times[0] if count > 1 else .02
    valid_interior = (frame_times >= .5) & (frame_times < frame_times[-1] - .5)
    masks["note_onset"][:] = torch.from_numpy(valid_interior[:, None] & np.array(negative_onsets_allowed)[None, :])
    negative_percussion = window["targets"].get("negativePercussionSupervision", False)
    coverage = window["targets"].get("percussionAnnotationCoverage", [])
    if type(negative_percussion) is not bool or not isinstance(coverage, list) or bool(coverage) != negative_percussion:
        raise HarnessError("Percussion negative labels require explicit nonempty annotation coverage.")
    covered = np.zeros(count, dtype=bool)
    previous_end = -1.
    for left, right in coverage:
        if not np.isfinite([left, right]).all() or not 0 <= left < right <= frame_times[-1] + hop + 1e-9 or left < previous_end:
            raise HarnessError("Invalid percussion annotation coverage.")
        covered |= (frame_times >= left) & (frame_times < right)
        previous_end = right
    masks["percussion"][:] = torch.from_numpy((valid_interior & covered)[:, None])
    source_notes = {note["id"]: note for note in canonical["targets"]["notes"]}
    source_gestures = {gesture["id"]: gesture for gesture in canonical["targets"]["gestures"]}
    events = {}
    radius = max(1, int(np.ceil(.05 / hop)))
    for note in window["targets"]["notes"]:
        if not note["supervisionMask"]["onset"]:
            continue
        onset = note["onsetWindowSeconds"]
        index = int(np.argmin(np.abs(frame_times - onset)))
        string = 6 - note["string"]
        if not 0 <= string < 6 or not 0 <= onset <= frame_times[-1] + hop:
            raise HarnessError("A supervised note lies outside its window or string range.")
        masks["note_onset"][max(0, index - radius):min(count, index + radius + 1), string] = False
        events.setdefault((index, string), []).append(note)
    collisions = 0
    for (index, string), notes in events.items():
        targets["note_onset"][index, string] = 1
        masks["note_onset"][index, string] = True
        if len(notes) != 1:
            collisions += 1
            continue
        note = notes[0]
        source = source_notes[note["sourceNoteId"]]
        if note["sourceLabelMask"] != source["labelMask"]:
            raise HarnessError("Window note masks differ from canonical supervision.")
        for field, source_field, limit in (("fret", "fret", model_config.max_fret + 1), ("pitch", "soundingPitchMidi", 128), ("voice", "voiceIndex", model_config.max_voices)):
            permitted = note["supervisionMask"]["fingering" if field == "fret" else "pitch" if field == "pitch" else "onset"]
            value = note[source_field]
            if permitted:
                if type(value) is not int or not 0 <= value < limit:
                    raise HarnessError(f"Supervised {field} is outside model vocabulary.")
                targets[field][index, string] = value
                masks[field][index, string] = True
        if note["supervisionMask"]["notatedDuration"]:
            duration = float(fraction(note["notatedDurationQuarter"], "note duration"))
            if duration <= 0:
                raise HarnessError("A supervised duration must be positive.")
            targets["duration_log"][index, string] = np.log1p(duration)
            masks["duration_log"][index, string] = True
        harmonic = source["sourceSegments"][0].get("harmonic")
        if harmonic is not None:
            targets["harmonic"][index, string] = 1
            masks["harmonic"][index, string] = True
            if harmonic["type"] in HARMONIC_TYPES:
                targets["harmonic_kind"][index, string] = HARMONIC_TYPES.index(harmonic["type"])
                masks["harmonic_kind"][index, string] = True
            node = float(fraction(harmonic["fret"], "harmonic node"))
            if node in HARMONIC_FRETS:
                targets["harmonic_node"][index, string] = HARMONIC_FRETS.index(node)
                masks["harmonic_node"][index, string] = True
    percussion_events = []
    for gesture in window["targets"]["gestures"]:
        source = source_gestures[gesture["sourceGestureId"]]
        if gesture["technique"] != source["technique"]:
            raise HarnessError("Window gesture differs from its canonical source.")
        if gesture["technique"] not in PERCUSSION_TYPES or not gesture["supervisionMask"]["gesture"] or not gesture["supervisionMask"]["onset"]:
            continue
        index = int(np.argmin(np.abs(frame_times - gesture["onsetWindowSeconds"])))
        category = PERCUSSION_TYPES.index(gesture["technique"])
        masks["percussion"][max(0, index - radius):min(count, index + radius + 1), category] = False
        percussion_events.append((index, category))
    for index, category in percussion_events:
        targets["percussion"][index, category] = 1
        masks["percussion"][index, category] = True
    return targets, masks, collisions


def negative_onset_coverage(labels):
    allowed = [True] * 6
    for note in labels["targets"]["notes"]:
        if note["isAttack"] is not True or not note["labelMask"]["attack"] or note["sourceSegments"][0]["graceMode"] is not None:
            allowed[6 - note["string"]] = False
    return allowed


class TrainingDataset(Dataset):
    @staticmethod
    def _stat(path):
        value = path.stat()
        return value.st_size, value.st_mtime_ns

    def check_unchanged(self):
        for path, original in self._guards.items():
            if self._stat(path) != original:
                raise HarnessError("A bound dataset file changed during use; stop and revalidate the release.")

    def __len__(self):
        return len(self.windows)

    def _features(self, record, window):
        row, data = record["row"], record["data"]
        key = hashlib.sha256(json.dumps({"audio": row["audioSha256"], "start": window["startSample"], "stop": window["stopSampleExclusive"], "config": asdict(self.feature_config), "featureImplementation": sha256(Path(__file__).with_name("transcriber_audio.py")), "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "soundfile": soundfile.__version__}, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{key}.npz" if self.cache_dir is not None else None
        if path is not None and (path.resolve() != path.absolute() or path.is_symlink() or path.exists() and path.stat().st_nlink > 1):
            raise HarnessError("Feature cache files must not alias other assets.")
        if path is not None and path.exists():
            with np.load(path, allow_pickle=False) as cached:
                features = np.array(cached["features"], copy=True)
                times = np.array(cached["times"], copy=True)
            expected_samples = ((window["stopSampleExclusive"] - window["startSample"]) * self.feature_config.sample_rate + row["sampleRate"] - 1) // row["sampleRate"]
            expected_frames = (expected_samples + self.feature_config.hop_length - 1) // self.feature_config.hop_length
            if features.dtype != np.float32 or features.ndim != 2 or features.shape != (expected_frames, self.feature_config.n_mels) or times.shape != (expected_frames,) or not np.isfinite(features).all() or not np.allclose(times, np.arange(expected_frames) * self.feature_config.hop_seconds):
                raise HarnessError("Corrupt feature cache; remove it explicitly before retrying.")
            return torch.from_numpy(features), times
        samples, rate = read_audio_window(data.audio_path, window["startSample"], window["stopSampleExclusive"], sample_rate=row["sampleRate"], channels=row["channels"], sample_count=data.entry["audioAsset"]["sampleCount"])
        features, times = audio_features(samples, rate, self.feature_config)
        if path is not None:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.cache_dir, prefix=".features-", suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    np.savez_compressed(stream, features=features.numpy(), times=times)
                temporary.replace(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return features, times

    def __getitem__(self, index):
        self.check_unchanged()
        record, window = self.windows[index]
        features, times = self._features(record, window)
        clip_times = times + window["startSample"] / record["row"]["sampleRate"]
        conditioning = training_conditioning(record, clip_times)
        targets, masks, collisions = encode_targets(window, record["data"].labels, times, self.model_config, negative_onsets_allowed=record["negativeAllowed"])
        return {
            "features": features, "conditioning": conditioning, "targets": targets, "masks": masks,
            "metadata": {"windowId": window["windowId"], "stringFrameCollisionsMasked": collisions},
        }

    def __init__(self, manifest_path, split, feature_config, model_config, *, root=ROOT, cache_dir=None):
        self.root = Path(root).resolve()
        self.manifest_path = Path(manifest_path).absolute()
        if not self.manifest_path.is_relative_to(self.root):
            raise HarnessError("The release manifest must be inside its declared private data root.")
        if split not in ("train", "validation"):
            raise HarnessError("Training releases contain train and validation splits only.")
        if feature_config.n_mels != model_config.n_mels or model_config.conditioning_dim != 12:
            raise HarnessError("Feature/model dimensions disagree.")
        self.feature_config, self.model_config = feature_config, model_config
        manifest, records, bindings = validate_release(self.manifest_path)
        self.manifest_sha256 = bindings[self.manifest_path]
        self._guards = {path: self._stat(path) for path in bindings}
        self.cache_dir = Path(cache_dir).resolve() if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.records, self.windows = [], []
        for row, payload in records:
            if row["split"] != split:
                continue
            labels = payload["canonical"]
            data = SimpleNamespace(
                labels=labels, normalization=payload["normalization"],
                audio_path=release_path(self.manifest_path.parent, row["audioPath"], "audio"),
                entry={"audioAsset": {"sampleCount": row["sampleCount"]}},
            )
            record = {
                "data": data, "candidate": payload["candidate"], "row": row, "negativeAllowed": negative_onset_coverage(labels),
                "percussionAnnotationsComplete": payload["approval"].get("percussionAnnotationsComplete") is True,
            }
            self.records.append(record)
            self.windows.extend((record, window) for window in payload["windows"])
        if len(self.windows) != manifest["counts"]["windowsBySplit"][split]:
            raise HarnessError("Training release window count changed during loading.")
