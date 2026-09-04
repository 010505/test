from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def stats(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate pure-QKV three-seed results")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--biased-aggregate", required=True)
    args = parser.parse_args()
    roots = [Path(path) for path in args.runs]
    runs = [
        {row["experiment"]: row for row in json.loads((root / "summary.json").read_text(encoding="utf-8"))}
        for root in roots
    ]
    control = np.asarray([run["00_pure_qkv_control"]["official_test_accuracy"] for run in runs])
    semantic = np.asarray([run["01_semantic_pure_qkv"]["official_test_accuracy"] for run in runs])
    biased = {
        row["family"]: row
        for row in json.loads(Path(args.biased_aggregate).read_text(encoding="utf-8"))
    }["qkv"]
    delta = (semantic - control) * 100.0
    result = {
        "seeds": [run["00_pure_qkv_control"]["seed"] for run in runs],
        "pure_control_accuracy": control.tolist(),
        "pure_semantic_accuracy": semantic.tolist(),
        "pure_control": stats(control),
        "pure_semantic": stats(semantic),
        "paired_se_delta_pp": delta.tolist(),
        "paired_se_delta_mean_pp": float(delta.mean()),
        "paired_se_delta_std_pp": float(delta.std(ddof=1)),
        "pure_minus_biased_pp": {
            "without_se": float((control.mean() - biased["control_mean"]) * 100.0),
            "with_se": float((semantic.mean() - biased["semantic_mean"]) * 100.0),
        },
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "aggregate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Pure QKV versus physical-bias QKV",
        "",
        "| Variant | No semantic SE | With semantic SE | Paired SE delta |",
        "|---|---:|---:|---:|",
        f"| Pure QKV | {result['pure_control']['mean']:.2%} +/- {result['pure_control']['std']:.2%} | "
        f"{result['pure_semantic']['mean']:.2%} +/- {result['pure_semantic']['std']:.2%} | "
        f"{result['paired_se_delta_mean_pp']:+.2f} +/- {result['paired_se_delta_std_pp']:.2f} pp |",
        f"| Physical-bias QKV | {biased['control_mean']:.2%} +/- {biased['control_std']:.2%} | "
        f"{biased['semantic_mean']:.2%} +/- {biased['semantic_std']:.2%} | "
        f"{biased['paired_delta_mean_pp']:+.2f} +/- {biased['paired_delta_std_pp']:.2f} pp |",
        "",
        f"Removing the physical bias changes the mean by "
        f"{result['pure_minus_biased_pp']['without_se']:+.2f} pp without SE and "
        f"{result['pure_minus_biased_pp']['with_se']:+.2f} pp with SE.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
