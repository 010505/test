from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CONTROL = "00_stgcn_control"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate repeated GestureGraph improvement runs")
    parser.add_argument("--runs", nargs="+", required=True, help="Run roots containing summary.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--control", default=CONTROL, help="Experiment name used for paired deltas")
    args = parser.parse_args()

    roots = [Path(path) for path in args.runs]
    run_rows = [json.loads((root / "summary.json").read_text(encoding="utf-8")) for root in roots]
    by_run = [{row["experiment"]: row for row in rows} for rows in run_rows]
    experiment_order = [row["experiment"] for row in run_rows[0]]
    if any(list(run) != experiment_order for run in by_run):
        raise ValueError("all runs must contain the same experiments in the same order")
    controls = np.asarray(
        [run[args.control]["official_test_accuracy"] for run in by_run],
        dtype=np.float64,
    )

    aggregate = []
    for experiment in experiment_order:
        rows = [run[experiment] for run in by_run]
        invariant_keys = ("architecture", "dataset", "frames", "epochs", "batch_size", "learning_rate", "model_config")
        for key in invariant_keys:
            if any(row[key] != rows[0][key] for row in rows[1:]):
                raise ValueError(f"{experiment}: inconsistent {key}")
        test = np.asarray([row["official_test_accuracy"] for row in rows], dtype=np.float64)
        validation = np.asarray([row["best_validation_accuracy"] for row in rows], dtype=np.float64)
        delta = (test - controls) * 100.0
        aggregate.append({
            "experiment": experiment,
            "architecture": rows[0]["architecture"],
            "parameters": rows[0]["parameters"],
            "seeds": [row["seed"] for row in rows],
            "official_test_accuracy": [float(value) for value in test],
            "test_mean": float(test.mean()),
            "test_std": float(test.std(ddof=1)) if len(test) > 1 else 0.0,
            "validation_mean": float(validation.mean()),
            "validation_std": float(validation.std(ddof=1)) if len(validation) > 1 else 0.0,
            "paired_delta_pp": [float(value) for value in delta],
            "paired_delta_mean_pp": float(delta.mean()),
            "paired_delta_std_pp": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
        })

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    with (output / "aggregate.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "experiment", "architecture", "parameters", "seeds",
                "test_mean", "test_std", "paired_delta_mean_pp", "paired_delta_std_pp",
            ],
        )
        writer.writeheader()
        for row in aggregate:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    lines = [
        "# Three-seed GestureGraph comparison",
        "",
        "Mean ± sample standard deviation across seeds. Delta is paired against the",
        f"`{args.control}` control trained with the same seed and validation split.",
        "",
        f"| Experiment | Parameters | Official test | Paired delta vs {args.control} |",
        "|---|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['experiment']} | {row['parameters']:,} | "
            f"{row['test_mean']:.2%} ± {row['test_std']:.2%} | "
            f"{row['paired_delta_mean_pp']:+.2f} ± {row['paired_delta_std_pp']:.2f} pp |"
        )
    lines.extend(["", "Seeds: " + ", ".join(map(str, aggregate[0]["seeds"])) + ".", ""])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
