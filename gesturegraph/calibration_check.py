from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import GestureDataset
from .regression_check import load_checkpoint
from .shrec import load_shrec17


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def confidences_and_correctness(model, loader) -> tuple[list[float], list[bool]]:
    confidences, corrects = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs).numpy()
            probabilities = softmax(logits)
            predictions = probabilities.argmax(axis=1)
            batch_confidences = probabilities.max(axis=1)
            confidences.extend(float(c) for c in batch_confidences)
            corrects.extend(bool(p == t) for p, t in zip(predictions, targets.numpy()))
    return confidences, corrects


def assign_bins(confidences: list[float], num_bins: int) -> list[int]:
    if num_bins < 1:
        raise ValueError("num_bins must be at least 1")
    indices = []
    for confidence in confidences:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {confidence}")
        index = int(confidence * num_bins)
        indices.append(min(index, num_bins - 1))
    return indices


def bin_statistics(confidences: list[float], corrects: list[bool], num_bins: int) -> list[dict]:
    if not (len(confidences) == len(corrects)):
        raise ValueError("confidences and corrects must be the same length")
    bin_indices = assign_bins(confidences, num_bins)
    stats = []
    for bin_index in range(num_bins):
        members = [i for i, b in enumerate(bin_indices) if b == bin_index]
        if not members:
            stats.append({"bin": bin_index, "count": 0, "avg_confidence": 0.0, "accuracy": 0.0})
            continue
        bin_confidences = [confidences[i] for i in members]
        bin_corrects = [corrects[i] for i in members]
        stats.append({
            "bin": bin_index,
            "count": len(members),
            "avg_confidence": sum(bin_confidences) / len(bin_confidences),
            "accuracy": sum(bin_corrects) / len(bin_corrects),
        })
    return stats


def expected_calibration_error(stats: list[dict], total_count: int) -> float:
    if total_count == 0:
        raise ValueError("total_count must be greater than zero")
    return sum(
        (row["count"] / total_count) * abs(row["accuracy"] - row["avg_confidence"])
        for row in stats
    )


def maximum_calibration_error(stats: list[dict]) -> float:
    populated = [row for row in stats if row["count"] > 0]
    if not populated:
        return 0.0
    return max(abs(row["accuracy"] - row["avg_confidence"]) for row in populated)


def build_report(confidences: list[float], corrects: list[bool], num_bins: int) -> dict:
    stats = bin_statistics(confidences, corrects, num_bins)
    return {
        "num_bins": num_bins,
        "sample_count": len(confidences),
        "overall_accuracy": sum(corrects) / len(corrects) if corrects else 0.0,
        "overall_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "expected_calibration_error": expected_calibration_error(stats, len(confidences)),
        "maximum_calibration_error": maximum_calibration_error(stats),
        "bins": stats,
    }


def print_summary(report: dict) -> None:
    print(f"Samples: {report['sample_count']}")
    print(f"Overall accuracy: {report['overall_accuracy']:.1%}")
    print(f"Overall confidence: {report['overall_confidence']:.1%}")
    print(f"Expected Calibration Error: {report['expected_calibration_error']:.1%}")
    print(f"Maximum Calibration Error: {report['maximum_calibration_error']:.1%}")
    for row in report["bins"]:
        if row["count"]:
            print(f"  bin {row['bin']:2d} n={row['count']:4d} conf={row['avg_confidence']:.1%} acc={row['accuracy']:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Confidence calibration check")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--classes", type=int, default=14, choices=[14, 28])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--output", default="runs/calibration_report.json")
    args = parser.parse_args()

    model, checkpoint = load_checkpoint(args.checkpoint)
    samples = load_shrec17(args.data, "test", int(checkpoint["frames"]), args.classes)
    loader = DataLoader(GestureDataset(samples, checkpoint["labels"]), batch_size=args.batch_size)

    confidences, corrects = confidences_and_correctness(model, loader)
    report = build_report(confidences, corrects, args.num_bins)
    print_summary(report)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report written to {destination}")


if __name__ == "__main__":
    main()
