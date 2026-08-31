from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class GestureSample:
    path: Path
    label: str
    sequence: np.ndarray
    subject: str | None = None
    split: str | None = None


def resample_sequence(sequence: np.ndarray, frames: int = 64) -> np.ndarray:
    if sequence.ndim != 3 or sequence.shape[1:] != (22, 3):
        raise ValueError(f"expected [T, 22, 3], got {sequence.shape}")
    if len(sequence) < 2:
        raise ValueError("a gesture needs at least two frames")
    source = np.linspace(0.0, 1.0, len(sequence))
    target = np.linspace(0.0, 1.0, frames)
    result = np.empty((frames, 22, 3), dtype=np.float32)
    for node in range(22):
        for channel in range(3):
            result[:, node, channel] = np.interp(target, source, sequence[:, node, channel])
    return result


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Remove global translation and scale while preserving motion."""
    sequence = np.asarray(sequence, dtype=np.float32).copy()
    sequence -= sequence[:, :1, :]
    lengths = []
    from .topology import EDGES
    for source, target in EDGES:
        lengths.append(np.linalg.norm(sequence[:, source] - sequence[:, target], axis=-1))
    scale = max(float(np.mean(lengths)), 1e-6)
    return sequence / scale


def load_sample(path: str | Path, frames: int = 64) -> GestureSample:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    label = str(payload.get("label", "")).strip()
    if not label:
        raise ValueError(f"{path}: missing label")
    sequence = np.asarray(payload.get("sequence"), dtype=np.float32)
    if not np.isfinite(sequence).all():
        raise ValueError(f"{path}: sequence contains NaN or infinity")
    return GestureSample(path, label, resample_sequence(normalize_sequence(sequence), frames), payload.get("subject"), payload.get("split"))


def discover_samples(root: str | Path, frames: int = 64) -> list[GestureSample]:
    paths = sorted(Path(root).rglob("*.json"))
    samples, errors = [], []
    for path in paths:
        try:
            samples.append(load_sample(path, frames))
        except json.JSONDecodeError as error:
            errors.append(f"{path}: invalid JSON ({error})")
        except (ValueError, TypeError) as error:
            errors.append(str(error))
    if errors:
        preview = "\n".join(f"- {item}" for item in errors[:8])
        raise ValueError(f"invalid dataset files:\n{preview}")
    if not samples:
        raise ValueError(f"no labelled gesture JSON files found in {root}")
    return samples


def stratified_split(samples: Iterable[GestureSample], val_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    groups: dict[str, list[GestureSample]] = {}
    for sample in samples:
        groups.setdefault(sample.label, []).append(sample)
    train, val = [], []
    for label, items in sorted(groups.items()):
        items = list(items)
        rng.shuffle(items)
        val_count = max(1, round(len(items) * val_ratio)) if len(items) > 1 else 0
        val.extend(items[:val_count])
        train.extend(items[val_count:])
    rng.shuffle(train); rng.shuffle(val)
    return train, val


class GestureDataset(Dataset):
    def __init__(self, samples: list[GestureSample], labels: list[str], augment: bool = False, augment_level: str = "light", temporal_crop: bool = False):
        self.samples = samples
        self.label_to_index = {label: index for index, label in enumerate(labels)}
        self.augment = augment
        self.augment_level = augment_level
        self.temporal_crop = temporal_crop

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        sequence = sample.sequence.copy()
        # Temporal crop: mimic the live-capture window (24..58 frames) by
        # taking a random contiguous sub-window and resampling back to 64 frames.
        if self.temporal_crop:
            total = sequence.shape[0]
            crop_len = np.random.randint(max(24, int(total * 0.35)), int(total * 0.92) + 1)
            start = np.random.randint(0, total - crop_len + 1)
            sequence = resample_sequence(sequence[start:start + crop_len], total)
        if self.augment:
            angle = np.random.uniform(-0.12, 0.12)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
            sequence[..., :2] = sequence[..., :2] @ rotation.T
            sequence += np.random.normal(0, 0.008, sequence.shape).astype(np.float32)
            if self.augment_level == "strong":
                # Per-sample 3D scale jitter + lateral translation.
                scale = np.random.uniform(0.9, 1.1)
                sequence *= scale
                sequence[..., 0] += np.random.normal(0, 0.01, sequence.shape[0])[:, None]
                sequence[..., 1] += np.random.normal(0, 0.01, sequence.shape[0])[:, None]
        # ST-GCN convention: channels, time, vertices.
        tensor = torch.from_numpy(sequence).permute(2, 0, 1)
        return tensor, self.label_to_index[sample.label]
