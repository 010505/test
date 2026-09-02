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


def predictions_by_subject(model, samples, labels: list[str], batch_size: int) -> list[tuple[str, bool]]:
    dataset = GestureDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    index_to_label = {index: label for index, label in enumerate(labels)}
    predictions = []
    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs)
            batch_predictions = logits.argmax(dim=1).tolist()
            predictions.extend(index_to_label[p] for p in batch_predictions)
    if len(predictions) != len(samples):
        raise ValueError("prediction count does not match sample count")
    records = []
    for sample, prediction in zip(samples, predictions):
        if sample.subject is None:
            raise ValueError(f"{sample.path}: sample has no subject recorded")
        records.append((sample.subject, prediction == sample.label))
    return records


def per_subject_accuracy(records: list[tuple[str, bool]]) -> dict[str, float]:
    correct: dict[str, int] = {}
    total: dict[str, int] = {}
    for subject, is_correct in records:
        total[subject] = total.get(subject, 0) + 1
        correct[subject] = correct.get(subject, 0) + (1 if is_correct else 0)
    return {subject: correct[subject] / total[subject] for subject in sorted(total)}


def accuracy_spread(per_subject: dict[str, float]) -> dict[str, float]:
    if not per_subject:
        raise ValueError("need at least one subject to compute spread")
    values = np.array(list(per_subject.values()))
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def find_biased_subjects(per_subject: dict[str, float]) -> list[dict]:
    if len(per_subject) < 4:
        return []
    values = np.array(list(per_subject.values()))
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    median = float(np.median(values))
    flagged = [
        {"subject": subject, "accuracy": accuracy, "deviation_from_median": accuracy - median}
        for subject, accuracy in per_subject.items()
        if accuracy < lower_bound
    ]
    return sorted(flagged, key=lambda row: row["accuracy"])


def build_report(records: list[tuple[str, bool]]) -> dict:
    per_subject = per_subject_accuracy(records)
    return {
        "subject_count": len(per_subject),
        "per_subject_accuracy": per_subject,
        "spread": accuracy_spread(per_subject),
        "biased_subjects": find_biased_subjects(per_subject),
    }


def print_summary(report: dict) -> None:
    spread = report["spread"]
    print(f"Subjects evaluated: {report['subject_count']}")
    print(f"Accuracy range: {spread['min']:.1%} - {spread['max']:.1%} (mean {spread['mean']:.1%}, std {spread['std']:.1%})")
    if report["biased_subjects"]:
        print("Subjects flagged as statistical outliers (Tukey lower fence):")
        for row in report["biased_subjects"]:
            print(f"  subject {row['subject']}: {row['accuracy']:.1%} ({row['deviation_from_median']:+.1%} vs median)")
    else:
        print("No subject falls outside the expected accuracy spread")


def main():
    parser = argparse.ArgumentParser(description="Per-subject generalization / bias check")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--classes", type=int, default=14, choices=[14, 28])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="runs/subject_bias_report.json")
    args = parser.parse_args()

    model, checkpoint = load_checkpoint(args.checkpoint)
    samples = load_shrec17(args.data, "test", int(checkpoint["frames"]), args.classes)

    records = predictions_by_subject(model, samples, checkpoint["labels"], args.batch_size)
    report = build_report(records)
    print_summary(report)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report written to {destination}")


if __name__ == "__main__":
    main()
