from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import GestureDataset, GestureSample
from .model import build_model
from .shrec import load_shrec17

FLIP_REGRESSION = "regression" 
FLIP_FIXED = "fixed"           
FLIP_STABLE_CORRECT = "stable_correct"
FLIP_STABLE_WRONG = "stable_wrong"


def load_checkpoint(path: str | Path, device: str = "cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    for key in ("model_state", "labels", "frames", "model_name"):
        if key not in checkpoint:
            raise ValueError(f"{path}: checkpoint is missing '{key}', can't rebuild the model")
    model = build_model(
        checkpoint["model_name"],
        len(checkpoint["labels"]),
        int(checkpoint["frames"]),
        float(checkpoint.get("dropout", 0.15)),
        checkpoint.get("ablation", "none"),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def predict_labels(model, samples: list[GestureSample], labels: list[str], batch_size: int = 32) -> list[str]:
    index_to_label = {index: label for index, label in enumerate(labels)}
    dataset = GestureDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    with torch.no_grad():
        for inputs, _ in loader:
            logits = model(inputs)
            predictions.extend(index_to_label[i] for i in logits.argmax(dim=1).tolist())
    return predictions


def classify_flips(truth: list[str], old_predictions: list[str], new_predictions: list[str]) -> list[dict]:
    if not (len(truth) == len(old_predictions) == len(new_predictions)):
        raise ValueError("truth, old_predictions and new_predictions must all be the same length")
    flips = []
    for index, (true_label, old_pred, new_pred) in enumerate(zip(truth, old_predictions, new_predictions)):
        old_correct = old_pred == true_label
        new_correct = new_pred == true_label
        if old_correct and not new_correct:
            category = FLIP_REGRESSION
        elif not old_correct and new_correct:
            category = FLIP_FIXED
        elif old_correct and new_correct:
            category = FLIP_STABLE_CORRECT
        else:
            category = FLIP_STABLE_WRONG
        flips.append({
            "index": index,
            "truth": true_label,
            "old_prediction": old_pred,
            "new_prediction": new_pred,
            "category": category,
        })
    return flips


def per_class_regression_counts(flips: list[dict]) -> dict[str, int]:
    counts = Counter(flip["truth"] for flip in flips if flip["category"] == FLIP_REGRESSION)
    return dict(sorted(counts.items()))


def per_class_accuracy(truth: list[str], predictions: list[str]) -> dict[str, float]:
    correct = Counter()
    total = Counter()
    for true_label, prediction in zip(truth, predictions):
        total[true_label] += 1
        if prediction == true_label:
            correct[true_label] += 1
    return {label: correct[label] / total[label] for label in sorted(total)}


def find_hidden_regressions(old_class_accuracy: dict[str, float], new_class_accuracy: dict[str, float]) -> list[dict]:
    hidden = []
    for label in sorted(old_class_accuracy):
        old_accuracy = old_class_accuracy[label]
        new_accuracy = new_class_accuracy.get(label, 0.0)
        if new_accuracy < old_accuracy:
            hidden.append({
                "label": label,
                "old_accuracy": old_accuracy,
                "new_accuracy": new_accuracy,
                "drop": old_accuracy - new_accuracy,
            })
    return sorted(hidden, key=lambda row: row["drop"], reverse=True)


def build_report(truth: list[str], old_predictions: list[str], new_predictions: list[str]) -> dict:
    flips = classify_flips(truth, old_predictions, new_predictions)
    tally = Counter(flip["category"] for flip in flips)
    old_accuracy = tally[FLIP_STABLE_CORRECT] / len(flips) + tally[FLIP_REGRESSION] / len(flips)
    new_accuracy = tally[FLIP_STABLE_CORRECT] / len(flips) + tally[FLIP_FIXED] / len(flips)
    old_class_accuracy = per_class_accuracy(truth, old_predictions)
    new_class_accuracy = per_class_accuracy(truth, new_predictions)
    hidden_regressions = find_hidden_regressions(old_class_accuracy, new_class_accuracy)
    return {
        "total_samples": len(flips),
        "old_overall_accuracy": old_accuracy,
        "new_overall_accuracy": new_accuracy,
        "overall_direction": "improved" if new_accuracy > old_accuracy else "regressed" if new_accuracy < old_accuracy else "unchanged",
        "flip_counts": dict(tally),
        "regressions_by_class": per_class_regression_counts(flips),
        "old_accuracy_by_class": old_class_accuracy,
        "new_accuracy_by_class": new_class_accuracy,
        "hidden_regressions": hidden_regressions,
        "regressed_samples": [flip for flip in flips if flip["category"] == FLIP_REGRESSION],
    }


def print_summary(report: dict) -> None:
    print(f"Old model accuracy: {report['old_overall_accuracy']:.1%}")
    print(f"New model accuracy: {report['new_overall_accuracy']:.1%} ({report['overall_direction']})")
    print(f"Flips: {report['flip_counts']}")
    if report["hidden_regressions"]:
        print("\nHidden regressions (worse per-class despite overall improvement):")
        for row in report["hidden_regressions"]:
            print(f"  {row['label']:14s} {row['old_accuracy']:.1%} -> {row['new_accuracy']:.1%}  (-{row['drop']:.1%})")
    else:
        print("\nNo hidden regressions: every class is at least as good as before.")
    if report["regressions_by_class"]:
        print("\nRegressed sample counts by class:")
        for label, count in report["regressions_by_class"].items():
            print(f"  {label:14s} {count}")


def main():
    parser = argparse.ArgumentParser(description="Compare two checkpoints for backward-compatibility regressions")
    parser.add_argument("--data", required=True, help="path to the official SHREC'17 folder")
    parser.add_argument("--old", required=True, help="path to the earlier checkpoint (best.pt)")
    parser.add_argument("--new", required=True, help="path to the newer checkpoint (best.pt)")
    parser.add_argument("--classes", type=int, default=14, choices=[14, 28])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="runs/regression_report.json")
    args = parser.parse_args()

    old_model, old_checkpoint = load_checkpoint(args.old)
    new_model, new_checkpoint = load_checkpoint(args.new)

    old_samples = load_shrec17(args.data, "test", int(old_checkpoint["frames"]), args.classes)
    new_samples = load_shrec17(args.data, "test", int(new_checkpoint["frames"]), args.classes)
    truth = [sample.label for sample in old_samples]
    if truth != [sample.label for sample in new_samples]:
        raise ValueError("old and new samples are not in the same order - can't compare them sample-by-sample")

    old_predictions = predict_labels(old_model, old_samples, old_checkpoint["labels"], args.batch_size)
    new_predictions = predict_labels(new_model, new_samples, new_checkpoint["labels"], args.batch_size)

    report = build_report(truth, old_predictions, new_predictions)
    print_summary(report)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report written to {destination}")


if __name__ == "__main__":
    main()
