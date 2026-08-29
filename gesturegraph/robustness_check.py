from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .regression_check import load_checkpoint
from .shrec import load_shrec17

DEFAULT_EPSILON = 0.03  # normalized coordinate units - roughly 4x the noise the model already sees during training augmentation
DEFAULT_TRIALS = 5


def perturb_sequence(sequence: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    if epsilon < 0:
        raise ValueError("epsilon can't be negative")
    noise = rng.uniform(-epsilon, epsilon, size=sequence.shape).astype(np.float32)
    return (sequence.astype(np.float32) + noise)


def sequence_to_tensor(sequence: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(sequence).permute(2, 0, 1)


def predict_one(model, sequence: np.ndarray, labels: list[str]) -> str:
    tensor = sequence_to_tensor(sequence).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
    return labels[int(logits.argmax(dim=1).item())]


def flip_rate_for_sample(true_label: str, clean_prediction: str, perturbed_predictions: list[str]) -> float | None:
    if clean_prediction != true_label:
        return None
    if not perturbed_predictions:
        raise ValueError("need at least one perturbed prediction to compute a flip rate")
    flips = sum(1 for prediction in perturbed_predictions if prediction != clean_prediction)
    return flips / len(perturbed_predictions)


def aggregate_by_class(records: list[tuple[str, float | None]]) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for label, rate in records:
        if rate is None:
            continue
        totals.setdefault(label, []).append(rate)
    return {label: sum(rates) / len(rates) for label, rates in sorted(totals.items())}


def find_fragile_classes(class_flip_rates: dict[str, float], threshold: float = 0.2) -> list[dict]:
    fragile = [
        {"label": label, "flip_rate": rate}
        for label, rate in class_flip_rates.items()
        if rate > threshold
    ]
    return sorted(fragile, key=lambda row: row["flip_rate"], reverse=True)


def build_report(truth: list[str], clean_predictions: list[str], perturbed_predictions: list[list[str]], threshold: float = 0.2) -> dict:
    if not (len(truth) == len(clean_predictions) == len(perturbed_predictions)):
        raise ValueError("truth, clean_predictions and perturbed_predictions must all be the same length")
    per_sample_rates = [
        flip_rate_for_sample(true_label, clean, perturbed)
        for true_label, clean, perturbed in zip(truth, clean_predictions, perturbed_predictions)
    ]
    class_flip_rates = aggregate_by_class(list(zip(truth, per_sample_rates)))
    evaluated = [rate for rate in per_sample_rates if rate is not None]
    overall_flip_rate = sum(evaluated) / len(evaluated) if evaluated else 0.0
    return {
        "samples_evaluated": len(evaluated),
        "samples_skipped_already_wrong": len(per_sample_rates) - len(evaluated),
        "overall_flip_rate": overall_flip_rate,
        "flip_rate_by_class": class_flip_rates,
        "fragile_classes": find_fragile_classes(class_flip_rates, threshold),
    }


def print_summary(report: dict) -> None:
    print(f"Samples evaluated: {report['samples_evaluated']} (skipped {report['samples_skipped_already_wrong']} the model already got wrong)")
    print(f"Overall flip rate under perturbation: {report['overall_flip_rate']:.1%}")
    if report["fragile_classes"]:
        print("\nFragile classes (flip rate above threshold):")
        for row in report["fragile_classes"]:
            print(f"  {row['label']:14s} {row['flip_rate']:.1%}")
    else:
        print("\nNo class crossed the fragility threshold.")


def run_robustness_check(model, labels: list[str], samples, epsilon: float, trials: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    truth, clean_predictions, perturbed_predictions = [], [], []
    for sample in samples:
        clean_prediction = predict_one(model, sample.sequence, labels)
        trial_predictions = [
            predict_one(model, perturb_sequence(sample.sequence, epsilon, rng), labels)
            for _ in range(trials)
        ]
        truth.append(sample.label)
        clean_predictions.append(clean_prediction)
        perturbed_predictions.append(trial_predictions)
    return build_report(truth, clean_predictions, perturbed_predictions)


def main():
    parser = argparse.ArgumentParser(description="Bounded-noise robustness sanity check")
    parser.add_argument("--data", required=True, help="path to the official SHREC'17 folder")
    parser.add_argument("--checkpoint", required=True, help="path to the checkpoint to test (best.pt)")
    parser.add_argument("--classes", type=int, default=14, choices=[14, 28])
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON, help="max per-coordinate noise, in normalized units")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="perturbed copies to test per sample")
    parser.add_argument("--threshold", type=float, default=0.2, help="flip rate above which a class is flagged fragile")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/robustness_report.json")
    args = parser.parse_args()

    model, checkpoint = load_checkpoint(args.checkpoint)
    samples = load_shrec17(args.data, "test", int(checkpoint["frames"]), args.classes)

    report = run_robustness_check(model, checkpoint["labels"], samples, args.epsilon, args.trials, args.seed)
    report["epsilon"] = args.epsilon
    report["trials_per_sample"] = args.trials
    print_summary(report)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report written to {destination}")


if __name__ == "__main__":
    main()
