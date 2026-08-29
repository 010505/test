from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .analyze import JOINT_GROUPS, accuracy
from .data import GestureDataset
from .regression_check import load_checkpoint
from .shrec import load_shrec17


def compute_importance_profile(model, loader) -> dict[str, float]:
    baseline = accuracy(model, loader)
    profile = {}
    for name, nodes in JOINT_GROUPS.items():
        masked = accuracy(model, loader, nodes)
        profile[name] = baseline - masked
    return profile


def rank_groups(drops: dict[str, float]) -> list[str]:
    return sorted(drops, key=lambda name: drops[name], reverse=True)


def spearman_correlation(ranking_a: list[str], ranking_b: list[str]) -> float:
    if sorted(ranking_a) != sorted(ranking_b):
        raise ValueError("both rankings must contain the same set of labels")
    position_a = {label: index for index, label in enumerate(ranking_a)}
    position_b = {label: index for index, label in enumerate(ranking_b)}
    n = len(ranking_a)
    if n < 2:
        return 1.0
    squared_diffs = sum((position_a[label] - position_b[label]) ** 2 for label in ranking_a)
    return 1 - (6 * squared_diffs) / (n * (n ** 2 - 1))


def compare_importance_profiles(old_drops: dict[str, float], new_drops: dict[str, float]) -> dict:
    if set(old_drops) != set(new_drops):
        raise ValueError("both profiles must cover the same joint groups")
    old_ranking = rank_groups(old_drops)
    new_ranking = rank_groups(new_drops)
    return {
        "old_ranking": old_ranking,
        "new_ranking": new_ranking,
        "old_top_group": old_ranking[0],
        "new_top_group": new_ranking[0],
        "top_group_changed": old_ranking[0] != new_ranking[0],
        "rank_correlation": spearman_correlation(old_ranking, new_ranking),
    }


def flag_explanation_regression(comparison: dict, correlation_threshold: float) -> bool:
    return comparison["top_group_changed"] or comparison["rank_correlation"] < correlation_threshold


def build_report(old_drops: dict[str, float], new_drops: dict[str, float], correlation_threshold: float) -> dict:
    comparison = compare_importance_profiles(old_drops, new_drops)
    return {
        "old_importance": old_drops,
        "new_importance": new_drops,
        "comparison": comparison,
        "correlation_threshold": correlation_threshold,
        "explanation_regression": flag_explanation_regression(comparison, correlation_threshold),
    }


def print_summary(report: dict) -> None:
    comparison = report["comparison"]
    print(f"Old top group: {comparison['old_top_group']}")
    print(f"New top group: {comparison['new_top_group']}")
    print(f"Rank correlation: {comparison['rank_correlation']:.2f}")
    if report["explanation_regression"]:
        print("WARNING: explanation regression detected - the model shifted which joints it relies on")
    else:
        print("Reasoning stayed consistent between checkpoints")


def main():
    parser = argparse.ArgumentParser(description="Compare joint-group reliance between two checkpoints")
    parser.add_argument("--data", required=True)
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--classes", type=int, default=14, choices=[14, 28])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--correlation-threshold", type=float, default=0.5)
    parser.add_argument("--output", default="runs/explanation_consistency_report.json")
    args = parser.parse_args()

    old_model, old_checkpoint = load_checkpoint(args.old)
    new_model, new_checkpoint = load_checkpoint(args.new)

    old_samples = load_shrec17(args.data, "test", int(old_checkpoint["frames"]), args.classes)
    new_samples = load_shrec17(args.data, "test", int(new_checkpoint["frames"]), args.classes)

    old_loader = DataLoader(GestureDataset(old_samples, old_checkpoint["labels"]), batch_size=args.batch_size)
    new_loader = DataLoader(GestureDataset(new_samples, new_checkpoint["labels"]), batch_size=args.batch_size)

    old_drops = compute_importance_profile(old_model, old_loader)
    new_drops = compute_importance_profile(new_model, new_loader)

    report = build_report(old_drops, new_drops, args.correlation_threshold)
    print_summary(report)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report written to {destination}")


if __name__ == "__main__":
    main()
