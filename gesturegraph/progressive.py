from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .data import normalize_sequence, resample_sequence
from .model import build_model
from .shrec import LABELS_14


DEFAULT_OBSERVATION_RATIOS = (0.25, 0.50, 0.65, 0.80, 1.00)
PROGRESSIVE_EXPERIMENTS = (
    "00_full_sequence",
    "01_mild_temporal_crop",
    "02_causal_prefix",
    "03_gru_evidence",
    "04_class_diffusion",
)
EXTENDED_PROGRESSIVE_EXPERIMENTS = PROGRESSIVE_EXPERIMENTS + (
    "05_gated_class_diffusion",
    "06_reliability_gated_diffusion",
)


@dataclass(frozen=True)
class RawGestureSample:
    index: int
    label: str
    sequence: np.ndarray
    split: str


def load_raw_shrec17_npz(
    root: str | Path,
    split: str,
    classes: int = 14,
) -> list[RawGestureSample]:
    if split not in {"train", "test"}:
        raise ValueError("SHREC split must be train or test")
    if classes != 14:
        raise ValueError("progressive experiments currently require 14 classes")
    path = Path(root) / f"{split}.npz"
    with np.load(path, allow_pickle=False) as archive:
        coordinates = np.asarray(archive["coordinates"], dtype=np.float32)
        offsets = np.asarray(archive["offsets"], dtype=np.int64)
        labels = np.asarray(archive["labels14"], dtype=np.int64)
    if offsets.shape != (len(labels) + 1,) or offsets[0] != 0 or offsets[-1] != len(coordinates):
        raise ValueError(f"{path}: invalid offsets")
    samples = []
    for index, label_value in enumerate(labels):
        start, end = int(offsets[index]), int(offsets[index + 1])
        sequence = coordinates[start:end].copy()
        if sequence.shape[0] < 2 or sequence.shape[1:] != (22, 3):
            raise ValueError(f"{path}#{index}: invalid sequence shape {sequence.shape}")
        samples.append(RawGestureSample(index, LABELS_14[int(label_value) - 1], sequence, split))
    return samples


def stratified_raw_split(
    samples: Iterable[RawGestureSample],
    val_ratio: float,
    seed: int,
) -> tuple[list[RawGestureSample], list[RawGestureSample]]:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[RawGestureSample]] = {}
    for sample in samples:
        groups.setdefault(sample.label, []).append(sample)
    train: list[RawGestureSample] = []
    validation: list[RawGestureSample] = []
    for _, items in sorted(groups.items()):
        items = list(items)
        rng.shuffle(items)
        count = max(1, round(len(items) * val_ratio))
        validation.extend(items[:count])
        train.extend(items[count:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def _augment_coordinates(sequence: np.ndarray) -> np.ndarray:
    sequence = sequence.copy()
    angle = np.random.uniform(-0.12, 0.12)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    sequence[..., :2] = sequence[..., :2] @ rotation.T
    sequence += np.random.normal(0.0, 0.008, sequence.shape).astype(np.float32)
    return sequence


def full_view(sequence: np.ndarray, frames: int = 64, augment: bool = False):
    view = resample_sequence(normalize_sequence(sequence), frames)
    if augment:
        view = _augment_coordinates(view)
    return view, frames, 1.0


def causal_prefix_view(
    sequence: np.ndarray,
    ratio: float,
    frames: int = 64,
    augment: bool = False,
):
    """Build a future-free prefix without stretching it to look complete."""
    if not 0.0 < ratio <= 1.0:
        raise ValueError("observation ratio must be in (0, 1]")
    source_length = max(2, min(len(sequence), math.ceil(len(sequence) * ratio)))
    valid_length = max(2, min(frames, math.ceil(frames * ratio)))
    prefix = normalize_sequence(sequence[:source_length])
    observed = resample_sequence(prefix, valid_length)
    if augment:
        observed = _augment_coordinates(observed)
    padded = np.empty((frames, 22, 3), dtype=np.float32)
    padded[:valid_length] = observed
    padded[valid_length:] = observed[-1]
    return padded, valid_length, float(ratio)


def mild_temporal_crop_view(
    sequence: np.ndarray,
    frames: int = 64,
    augment: bool = False,
):
    """Energy-aware 50-100% random window, used only as a robustness control."""
    length = len(sequence)
    ratio = 0.5 + 0.5 * float(np.random.beta(2.0, 1.5))
    window = max(2, min(length, round(length * ratio)))
    velocity = np.linalg.norm(np.diff(sequence, axis=0), axis=-1).mean(axis=1)
    peak = int(np.argmax(velocity)) + 1
    centered = peak - window // 2
    jitter = int(np.random.uniform(-0.15, 0.15) * window)
    start = int(np.clip(centered + jitter, 0, length - window))
    crop = sequence[start:start + window]
    static_before = int(np.random.randint(0, 5))
    static_after = int(np.random.randint(0, 5))
    if static_before:
        crop = np.concatenate([np.repeat(crop[:1], static_before, axis=0), crop], axis=0)
    if static_after:
        crop = np.concatenate([crop, np.repeat(crop[-1:], static_after, axis=0)], axis=0)
    return full_view(crop, frames, augment)


class SingleViewDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[RawGestureSample],
        labels: Sequence[str],
        mode: str,
        frames: int = 64,
        ratios: Sequence[float] = DEFAULT_OBSERVATION_RATIOS,
        augment: bool = False,
    ):
        if mode not in {"full", "mild", "causal"}:
            raise ValueError(f"unknown single-view mode: {mode}")
        self.samples = list(samples)
        self.label_to_index = {label: index for index, label in enumerate(labels)}
        self.mode = mode
        self.frames = frames
        self.ratios = tuple(float(value) for value in ratios)
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.mode == "full":
            view, valid, ratio = full_view(sample.sequence, self.frames, self.augment)
        elif self.mode == "mild":
            view, valid, ratio = mild_temporal_crop_view(sample.sequence, self.frames, self.augment)
        else:
            ratio = float(np.random.choice(self.ratios))
            view, valid, ratio = causal_prefix_view(sample.sequence, ratio, self.frames, self.augment)
        tensor = torch.from_numpy(view).permute(2, 0, 1)
        return tensor, valid, ratio, self.label_to_index[sample.label]


class PrefixSequenceDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[RawGestureSample],
        labels: Sequence[str],
        frames: int = 64,
        ratios: Sequence[float] = DEFAULT_OBSERVATION_RATIOS,
        augment: bool = False,
    ):
        self.samples = list(samples)
        self.label_to_index = {label: index for index, label in enumerate(labels)}
        self.frames = frames
        self.ratios = tuple(float(value) for value in ratios)
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        views, lengths = [], []
        for ratio in self.ratios:
            view, valid, _ = causal_prefix_view(sample.sequence, ratio, self.frames, self.augment)
            views.append(torch.from_numpy(view).permute(2, 0, 1))
            lengths.append(valid)
        return (
            torch.stack(views),
            torch.tensor(lengths, dtype=torch.long),
            torch.tensor(self.ratios, dtype=torch.float32),
            self.label_to_index[sample.label],
        )


def build_agcrn_encoder(num_classes: int = 14, dropout: float = 0.15) -> nn.Module:
    return build_model(
        "stem_agcrn_control",
        num_classes,
        64,
        dropout,
        "none",
        {"pe_dim": 21, "attention_heads": 4, "adaptive_dim": 10,
         "stem_channels": 32, "semantic_hidden": 64, "spectral_weighting": "none"},
    )


class DirectPrefixModel(nn.Module):
    def __init__(self, num_classes: int = 14, dropout: float = 0.15):
        super().__init__()
        self.encoder = build_agcrn_encoder(num_classes, dropout)
        self.head = nn.Linear(129, num_classes)

    def encode_sequence(self, views, valid_lengths, ratios):
        if views.ndim != 5:
            raise ValueError("views must have shape [B, S, C, T, V]")
        batch, steps = views.shape[:2]
        features = self.encoder.encode(
            views.flatten(0, 1), valid_lengths.flatten(0, 1)
        ).reshape(batch, steps, -1)
        return torch.cat([features, ratios.to(features.dtype).unsqueeze(-1)], dim=-1)

    def forward(self, views, valid_lengths, ratios):
        if views.ndim == 4:
            features = self.encoder.encode(views, valid_lengths)
            return self.head(torch.cat([features, ratios.to(features.dtype).unsqueeze(-1)], dim=-1))
        return self.head(self.encode_sequence(views, valid_lengths, ratios))

    def online_step(self, view, valid_length, ratio, state=None):
        """Evaluate one newly observed prefix; direct models have no recurrent state."""
        del state
        return self.forward(view, valid_length, ratio), None


class GRUEvidenceModel(nn.Module):
    def __init__(self, num_classes: int = 14, dropout: float = 0.15):
        super().__init__()
        self.encoder = build_agcrn_encoder(num_classes, dropout)
        self.gru = nn.GRU(129, 128, batch_first=True)
        self.head = nn.Linear(128, num_classes)

    def forward(self, views, valid_lengths, ratios):
        batch, steps = views.shape[:2]
        features = self.encoder.encode(
            views.flatten(0, 1), valid_lengths.flatten(0, 1)
        ).reshape(batch, steps, -1)
        states, _ = self.gru(torch.cat([features, ratios.to(features.dtype).unsqueeze(-1)], dim=-1))
        return self.head(states)

    def online_step(self, view, valid_length, ratio, state=None):
        """Advance the GRU by one causal observation while preserving its hidden state."""
        features = self.encoder.encode(view, valid_length)
        condition = torch.cat(
            [features, ratio.to(features.dtype).reshape(-1, 1)], dim=-1
        ).unsqueeze(1)
        output, next_state = self.gru(condition, state)
        return self.head(output[:, 0]), next_state


class ClassDiffusionModel(nn.Module):
    """Four-step uniform categorical diffusion conditioned on causal features."""

    def __init__(
        self,
        num_classes: int = 14,
        dropout: float = 0.15,
        betas: Sequence[float] = (0.25, 0.45, 0.65, 0.90),
        inheritance: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.steps = len(betas)
        self.inference_steps = self.steps
        self.inheritance = inheritance
        self.encoder = build_agcrn_encoder(num_classes, dropout)
        self.step_embedding = nn.Embedding(self.steps + 1, 16)
        self.denoiser = nn.Sequential(
            nn.Linear(129 + num_classes + 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        identity = torch.eye(num_classes)
        uniform = torch.full((num_classes, num_classes), 1.0 / num_classes)
        transitions = [
            (1.0 - float(beta)) * identity + float(beta) * uniform for beta in betas
        ]
        cumulative = [identity]
        for transition in transitions:
            cumulative.append(cumulative[-1] @ transition)
        posteriors = []
        for step, transition in enumerate(transitions, start=1):
            previous = cumulative[step - 1]
            current = cumulative[step]
            # [current j, clean i, previous k]
            numerator = previous[:, None, :] * transition.T[None, :, :]
            denominator = current.T[:, :, None].clamp_min(1e-12)
            posterior = numerator.permute(1, 0, 2) / denominator
            posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            posteriors.append(posterior)
        self.register_buffer("transitions", torch.stack(transitions))
        self.register_buffer("cumulative", torch.stack(cumulative))
        self.register_buffer("posteriors", torch.stack(posteriors))

    def _denoise_logits(self, condition, noisy_class, step):
        one_hot = F.one_hot(noisy_class, self.num_classes).to(condition.dtype)
        step_ids = torch.full_like(noisy_class, step)
        embedded = self.step_embedding(step_ids)
        return self.denoiser(torch.cat([condition, one_hot, embedded], dim=-1))

    def encode_sequence(self, views, valid_lengths, ratios):
        batch, updates = views.shape[:2]
        features = self.encoder.encode(
            views.flatten(0, 1), valid_lengths.flatten(0, 1)
        ).reshape(batch, updates, -1)
        return torch.cat([features, ratios.to(features.dtype).unsqueeze(-1)], dim=-1)

    def denoising_loss(self, conditions, targets):
        flat = conditions.flatten(0, 1)
        repeated_targets = targets[:, None].expand(-1, conditions.shape[1]).reshape(-1)
        steps = torch.randint(1, self.steps + 1, (len(flat),), device=flat.device)
        clean = F.one_hot(repeated_targets, self.num_classes).to(flat.dtype)
        probabilities = torch.einsum("bi,bij->bj", clean, self.cumulative[steps])
        noisy = torch.multinomial(probabilities, 1).squeeze(1)
        one_hot = F.one_hot(noisy, self.num_classes).to(flat.dtype)
        embedded = self.step_embedding(steps)
        logits = self.denoiser(torch.cat([flat, one_hot, embedded], dim=-1))
        return F.cross_entropy(logits, repeated_targets)

    def _reverse_distribution(self, condition, initial):
        batch = condition.shape[0]
        distribution = initial
        classes = torch.arange(self.num_classes, device=condition.device)
        requested_steps = int(self.inference_steps)
        if not 1 <= requested_steps <= self.steps:
            raise ValueError(f"inference_steps must be in [1, {self.steps}]")
        for iteration, step in enumerate(range(self.steps, 0, -1), start=1):
            expanded_condition = condition[:, None, :].expand(-1, self.num_classes, -1)
            noisy = classes[None, :].expand(batch, -1)
            logits = self._denoise_logits(
                expanded_condition.reshape(batch * self.num_classes, -1),
                noisy.reshape(-1),
                step,
            ).reshape(batch, self.num_classes, self.num_classes)
            clean_probability = logits.softmax(dim=-1)
            if iteration == requested_steps:
                clean_distribution = torch.einsum(
                    "bj,bji->bi", distribution, clean_probability
                )
                return clean_distribution / clean_distribution.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
            reverse = torch.einsum(
                "bji,jik->bjk", clean_probability, self.posteriors[step - 1]
            )
            distribution = torch.einsum("bj,bjk->bk", distribution, reverse)
            distribution = distribution / distribution.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        raise RuntimeError("reverse diffusion did not produce an output")

    def forward(self, views, valid_lengths, ratios, return_auxiliary=False):
        conditions = self.encode_sequence(views, valid_lengths, ratios)
        uniform = torch.full(
            (views.shape[0], self.num_classes),
            1.0 / self.num_classes,
            device=views.device,
            dtype=conditions.dtype,
        )
        previous = uniform
        outputs = []
        for update in range(conditions.shape[1]):
            initial = (
                uniform if update == 0
                else self.inheritance * previous + (1.0 - self.inheritance) * uniform
            )
            previous = self._reverse_distribution(conditions[:, update], initial)
            outputs.append(previous)
        log_probabilities = torch.stack(outputs, dim=1).clamp_min(1e-8).log()
        if return_auxiliary:
            return log_probabilities, conditions
        return log_probabilities

    def online_step(self, view, valid_length, ratio, state=None):
        """Advance one update using only the current prefix and previous class posterior."""
        features = self.encoder.encode(view, valid_length)
        condition = torch.cat(
            [features, ratio.to(features.dtype).reshape(-1, 1)], dim=-1
        )
        uniform = torch.full(
            (len(view), self.num_classes),
            1.0 / self.num_classes,
            device=view.device,
            dtype=condition.dtype,
        )
        initial = (
            uniform
            if state is None
            else self.inheritance * state + (1.0 - self.inheritance) * uniform
        )
        posterior = self._reverse_distribution(condition, initial)
        return posterior.clamp_min(1e-8).log(), posterior


class GatedClassDiffusionModel(ClassDiffusionModel):
    """Class diffusion with evidence-conditioned inheritance instead of fixed 0.5 mixing."""

    def __init__(self, num_classes: int = 14, dropout: float = 0.15, **kwargs):
        super().__init__(num_classes=num_classes, dropout=dropout, **kwargs)
        self.evidence_head = nn.Linear(129, num_classes)
        gate_features = 2 * num_classes + 4
        self.inheritance_gate = nn.Sequential(
            nn.Linear(gate_features, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.inheritance_gate[-1].weight)
        nn.init.zeros_(self.inheritance_gate[-1].bias)

    @staticmethod
    def _entropy(probabilities):
        return -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)

    def _gated_initial(self, condition, ratio, previous, uniform):
        evidence = self.evidence_head(condition).softmax(dim=-1)
        if previous is None:
            return uniform, evidence, None
        midpoint = 0.5 * (previous + evidence)
        js_divergence = 0.5 * (
            (previous * (previous.clamp_min(1e-8).log() - midpoint.clamp_min(1e-8).log())).sum(
                dim=-1, keepdim=True
            )
            + (evidence * (evidence.clamp_min(1e-8).log() - midpoint.clamp_min(1e-8).log())).sum(
                dim=-1, keepdim=True
            )
        )
        gate_input = torch.cat([
            previous,
            evidence,
            ratio.to(condition.dtype).reshape(-1, 1),
            self._entropy(previous),
            self._entropy(evidence),
            js_divergence,
        ], dim=-1)
        gate = self._gate_value(gate_input)
        return gate * previous + (1.0 - gate) * uniform, evidence, gate

    def _gate_value(self, gate_input):
        return self.inheritance_gate(gate_input).sigmoid()

    def denoising_loss(self, conditions, targets):
        denoising = super().denoising_loss(conditions, targets)
        flat = conditions.flatten(0, 1)
        repeated_targets = targets[:, None].expand(-1, conditions.shape[1]).reshape(-1)
        evidence = F.cross_entropy(self.evidence_head(flat), repeated_targets)
        return denoising + 0.5 * evidence

    def forward(
        self,
        views,
        valid_lengths,
        ratios,
        return_auxiliary=False,
        return_diagnostics=False,
    ):
        conditions = self.encode_sequence(views, valid_lengths, ratios)
        uniform = torch.full(
            (views.shape[0], self.num_classes),
            1.0 / self.num_classes,
            device=views.device,
            dtype=conditions.dtype,
        )
        previous = None
        outputs = []
        evidence_outputs = []
        gates = []
        previous_posteriors = []
        for update in range(conditions.shape[1]):
            initial, evidence, gate = self._gated_initial(
                conditions[:, update], ratios[:, update], previous, uniform
            )
            previous = self._reverse_distribution(conditions[:, update], initial)
            outputs.append(previous)
            evidence_outputs.append(evidence)
            if gate is not None:
                gates.append(gate.squeeze(-1))
                previous_posteriors.append(outputs[-2])
        log_probabilities = torch.stack(outputs, dim=1).clamp_min(1e-8).log()
        if return_diagnostics:
            gate_values = (
                torch.stack(gates, dim=1)
                if gates
                else torch.empty((len(views), 0), device=views.device, dtype=conditions.dtype)
            )
            diagnostics = {
                "gates": gate_values,
                "evidence": torch.stack(evidence_outputs, dim=1),
                "previous_posteriors": (
                    torch.stack(previous_posteriors, dim=1)
                    if previous_posteriors
                    else torch.empty(
                        (len(views), 0, self.num_classes),
                        device=views.device,
                        dtype=conditions.dtype,
                    )
                ),
            }
            if return_auxiliary:
                return log_probabilities, conditions, diagnostics
            return log_probabilities, diagnostics
        if return_auxiliary:
            return log_probabilities, conditions
        return log_probabilities

    def online_step(self, view, valid_length, ratio, state=None):
        features = self.encoder.encode(view, valid_length)
        condition = torch.cat(
            [features, ratio.to(features.dtype).reshape(-1, 1)], dim=-1
        )
        uniform = torch.full(
            (len(view), self.num_classes),
            1.0 / self.num_classes,
            device=view.device,
            dtype=condition.dtype,
        )
        initial, _, _ = self._gated_initial(condition, ratio, state, uniform)
        posterior = self._reverse_distribution(condition, initial)
        return posterior.clamp_min(1e-8).log(), posterior


class ReliabilityGatedClassDiffusionModel(GatedClassDiffusionModel):
    """Bounded gate supervised to estimate whether the previous posterior is reliable."""

    def _gate_value(self, gate_input):
        return 0.05 + 0.90 * self.inheritance_gate(gate_input).sigmoid()

    def reliability_loss(self, diagnostics, targets):
        gates = diagnostics["gates"]
        previous = diagnostics["previous_posteriors"]
        repeated_targets = targets[:, None, None].expand(-1, gates.shape[1], 1)
        target_reliability = previous.gather(-1, repeated_targets).squeeze(-1).detach()
        return F.binary_cross_entropy(gates, target_reliability)


def build_progressive_model(experiment: str, num_classes: int = 14, dropout: float = 0.15):
    if experiment in PROGRESSIVE_EXPERIMENTS[:3]:
        return DirectPrefixModel(num_classes, dropout)
    if experiment == "03_gru_evidence":
        return GRUEvidenceModel(num_classes, dropout)
    if experiment == "04_class_diffusion":
        return ClassDiffusionModel(num_classes, dropout)
    if experiment == "05_gated_class_diffusion":
        return GatedClassDiffusionModel(num_classes, dropout)
    if experiment == "06_reliability_gated_diffusion":
        return ReliabilityGatedClassDiffusionModel(num_classes, dropout)
    raise ValueError(f"unknown progressive experiment: {experiment}")
