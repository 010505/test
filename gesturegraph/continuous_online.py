from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .progressive import RawGestureSample, causal_prefix_view


def dense_observation_ratios(
    frames: int = 64,
    min_frames: int = 16,
    stride: int = 1,
) -> tuple[float, ...]:
    """Return the causal update schedule used by frame-by-frame deployment."""
    if frames < 2:
        raise ValueError("frames must be at least 2")
    if not 2 <= min_frames <= frames:
        raise ValueError("min_frames must be in [2, frames]")
    if stride < 1:
        raise ValueError("stride must be positive")
    observed = list(range(min_frames, frames + 1, stride))
    if observed[-1] != frames:
        observed.append(frames)
    return tuple(value / frames for value in observed)


class DensePrefixSequenceDataset(Dataset):
    """All causal prefixes needed for a continuous online sequence."""

    def __init__(
        self,
        samples: Sequence[RawGestureSample],
        labels: Sequence[str],
        frames: int = 64,
        min_frames: int = 16,
        stride: int = 1,
    ):
        self.samples = list(samples)
        self.label_to_index = {label: index for index, label in enumerate(labels)}
        self.frames = int(frames)
        self.ratios = dense_observation_ratios(frames, min_frames, stride)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        views, lengths = [], []
        for ratio in self.ratios:
            view, valid, _ = causal_prefix_view(sample.sequence, ratio, self.frames, False)
            views.append(torch.from_numpy(view).permute(2, 0, 1))
            lengths.append(valid)
        return (
            torch.stack(views),
            torch.tensor(lengths, dtype=torch.long),
            torch.tensor(self.ratios, dtype=torch.float32),
            self.label_to_index[sample.label],
        )


@torch.inference_mode()
def evaluate_continuous_online(
    model: torch.nn.Module,
    dataset: DensePrefixSequenceDataset,
    device: torch.device,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one reverse NFE per causal update while preserving the posterior state."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probability_batches, target_batches = [], []
    model.eval()
    for views, lengths, ratios, targets in loader:
        views = views.to(device)
        lengths = lengths.to(device)
        ratios = ratios.to(device)
        state = None
        outputs = []
        for update in range(views.shape[1]):
            log_probability, state = model.online_step(
                views[:, update], lengths[:, update], ratios[:, update], state
            )
            outputs.append(log_probability.exp().cpu())
        probability_batches.append(torch.stack(outputs, dim=1))
        target_batches.append(targets)
    return (
        torch.cat(probability_batches).numpy(),
        torch.cat(target_batches).numpy(),
    )


@dataclass(frozen=True)
class EarlyDecisionResult:
    indices: np.ndarray
    predictions: np.ndarray
    accuracy: float
    average_decision_ratio: float


def continuous_decisions(
    probabilities: np.ndarray,
    targets: np.ndarray,
    ratios: Sequence[float],
    confidence: float,
    margin: float,
    stable_updates: int = 2,
    min_decision_ratio: float | None = None,
) -> EarlyDecisionResult:
    """Select the first stable, confident online decision for every sample."""
    probabilities = np.asarray(probabilities)
    targets = np.asarray(targets)
    ratios_array = np.asarray(ratios, dtype=np.float64)
    if probabilities.ndim != 3 or probabilities.shape[1] != len(ratios_array):
        raise ValueError("probabilities must have shape [samples, updates, classes]")
    if stable_updates < 2:
        raise ValueError("stable_updates must be at least 2")
    if min_decision_ratio is None:
        min_decision_ratio = float(ratios_array[0])

    predictions_by_update = probabilities.argmax(axis=-1)
    selected = np.full(len(probabilities), len(ratios_array) - 1, dtype=np.int64)
    for sample_index, sample_probabilities in enumerate(probabilities):
        sample_predictions = predictions_by_update[sample_index]
        for update in range(stable_updates - 1, len(ratios_array)):
            if ratios_array[update] < min_decision_ratio:
                continue
            recent = sample_predictions[update - stable_updates + 1:update + 1]
            order = np.argsort(sample_probabilities[update])[::-1]
            if (
                np.all(recent == recent[-1])
                and sample_probabilities[update, order[0]] >= confidence
                and sample_probabilities[update, order[0]]
                - sample_probabilities[update, order[1]] >= margin
            ):
                selected[sample_index] = update
                break
    predictions = predictions_by_update[np.arange(len(selected)), selected]
    return EarlyDecisionResult(
        indices=selected,
        predictions=predictions,
        accuracy=float(np.mean(predictions == targets)),
        average_decision_ratio=float(np.mean(ratios_array[selected])),
    )


def calibrate_continuous_early_exit(
    probabilities: np.ndarray,
    targets: np.ndarray,
    ratios: Sequence[float],
    accuracy_tolerance: float = 0.01,
) -> dict[str, float | int]:
    """Calibrate dense-update confidence, margin, and stability on validation data."""
    full_accuracy = float(np.mean(probabilities[:, -1].argmax(axis=-1) == targets))
    target_accuracy = full_accuracy - float(accuracy_tolerance)
    candidates = []
    for min_decision_ratio in (0.25, 0.40, 0.50, 0.60):
        for stable_updates in (2, 4, 6, 8):
            for confidence in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98):
                for margin in (0.05, 0.10, 0.20, 0.30, 0.40):
                    result = continuous_decisions(
                        probabilities, targets, ratios,
                        confidence, margin, stable_updates, min_decision_ratio,
                    )
                    if result.accuracy >= target_accuracy:
                        candidates.append(
                            (
                                result.average_decision_ratio,
                                -result.accuracy,
                                min_decision_ratio,
                                stable_updates,
                                confidence,
                                margin,
                            )
                        )
    if candidates:
        _, _, min_decision_ratio, stable_updates, confidence, margin = min(candidates)
    else:
        min_decision_ratio, stable_updates, confidence, margin = 1.0, 2, 1.01, 1.01
    result = continuous_decisions(
        probabilities, targets, ratios, confidence, margin,
        stable_updates, min_decision_ratio,
    )
    return {
        "confidence_threshold": float(confidence),
        "margin_threshold": float(margin),
        "stable_updates": int(stable_updates),
        "min_decision_ratio": float(min_decision_ratio),
        "target_accuracy": float(target_accuracy),
        "accuracy": result.accuracy,
        "average_decision_ratio": result.average_decision_ratio,
        "full_accuracy": full_accuracy,
    }
