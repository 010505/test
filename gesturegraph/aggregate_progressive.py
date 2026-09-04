from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .progressive import PROGRESSIVE_EXPERIMENTS


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1))}


def main():
    parser = argparse.ArgumentParser(description="Aggregate progressive three-seed experiments")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", default="runs/progressive_aggregate")
    args = parser.parse_args()
    rows = {experiment: [] for experiment in PROGRESSIVE_EXPERIMENTS}
    for root in args.runs:
        for experiment in PROGRESSIVE_EXPERIMENTS:
            path = Path(root) / experiment / "model.json"
            rows[experiment].append(json.loads(path.read_text(encoding="utf-8")))
    aggregate = {}
    for experiment, experiments in rows.items():
        ratios = list(experiments[0]["official_test"]["ratio_accuracy"])
        aggregate[experiment] = {
            "seeds": [row["seed"] for row in experiments],
            "parameters": experiments[0]["parameters"],
            "ratio_accuracy": {
                ratio: stats([row["official_test"]["ratio_accuracy"][ratio] for row in experiments])
                for ratio in ratios
            },
            "prefix_auc": stats([row["official_test"]["prefix_auc"] for row in experiments]),
            "full_accuracy": stats([row["official_test"]["full_accuracy"] for row in experiments]),
            "decision_accuracy": stats([row["official_test_early_exit"]["accuracy"] for row in experiments]),
            "average_decision_ratio": stats([
                row["official_test_early_exit"]["average_decision_ratio"] for row in experiments
            ]),
            "latency_ms_per_sample_update": stats([
                row["latency_ms_per_sample_update"] for row in experiments
            ]),
        }
    comparisons = {}
    for reference in ("00_full_sequence", "02_causal_prefix", "03_gru_evidence"):
        comparisons[f"04_class_diffusion_minus_{reference}"] = {}
        for metric, path in (
            ("prefix_auc_pp", ("official_test", "prefix_auc")),
            ("full_accuracy_pp", ("official_test", "full_accuracy")),
            ("decision_accuracy_pp", ("official_test_early_exit", "accuracy")),
            ("average_decision_ratio_pp", ("official_test_early_exit", "average_decision_ratio")),
        ):
            values = [
                100.0 * (diffusion[path[0]][path[1]] - baseline[path[0]][path[1]])
                for diffusion, baseline in zip(rows["04_class_diffusion"], rows[reference])
            ]
            comparisons[f"04_class_diffusion_minus_{reference}"][metric] = {
                "per_seed": values,
                **stats(values),
            }
    aggregate["paired_comparisons"] = comparisons
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    lines = [
        "# Progressive early-recognition benchmark",
        "",
        "| Experiment | 25% | 50% | 65% | 80% | 100% | Prefix AUC | ADR | Decision accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment in PROGRESSIVE_EXPERIMENTS:
        row = aggregate[experiment]
        ratio_values = [row["ratio_accuracy"][key]["mean"] for key in row["ratio_accuracy"]]
        lines.append(
            f"| {experiment} | " + " | ".join(f"{value:.2%}" for value in ratio_values) +
            f" | {row['prefix_auc']['mean']:.2%} +/- {row['prefix_auc']['std']:.2%}" +
            f" | {row['average_decision_ratio']['mean']:.3f}" +
            f" | {row['decision_accuracy']['mean']:.2%} |"
        )
    report = "\n".join(lines) + "\n"
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
