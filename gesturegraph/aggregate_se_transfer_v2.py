from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PAIRS = [
    ("stable_qkv", "00_stable_qkv_control", "01_semantic_stable_qkv"),
    ("gwnet_gated", "02_gwnet_gated_control", "03_semantic_gwnet_gated"),
    ("agcrn_dynamic", "04_agcrn_dynamic_control", "05_semantic_agcrn_dynamic"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate stable-QKV and gated GWN/AGCRN runs")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    roots = [Path(path) for path in args.runs]
    runs = [
        {row["experiment"]: row for row in json.loads((root / "summary.json").read_text(encoding="utf-8"))}
        for root in roots
    ]
    rows = []
    for family, control_name, semantic_name in PAIRS:
        control = np.asarray([run[control_name]["official_test_accuracy"] for run in runs])
        semantic = np.asarray([run[semantic_name]["official_test_accuracy"] for run in runs])
        delta = (semantic - control) * 100.0
        rows.append({
            "family": family,
            "control": control_name,
            "semantic": semantic_name,
            "seeds": [run[control_name]["seed"] for run in runs],
            "control_accuracy": control.tolist(),
            "semantic_accuracy": semantic.tolist(),
            "control_mean": float(control.mean()),
            "control_std": float(control.std(ddof=1)),
            "semantic_mean": float(semantic.mean()),
            "semantic_std": float(semantic.std(ddof=1)),
            "paired_delta_pp": delta.tolist(),
            "paired_delta_mean_pp": float(delta.mean()),
            "paired_delta_std_pp": float(delta.std(ddof=1)),
        })
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "aggregate.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "# Stable QKV and gated GWN/AGCRN: three-seed paired comparison",
        "",
        "| Spatial operator | No semantic SE | With semantic SE | Paired SE delta |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['family']} | {row['control_mean']:.2%} +/- {row['control_std']:.2%} | "
            f"{row['semantic_mean']:.2%} +/- {row['semantic_std']:.2%} | "
            f"{row['paired_delta_mean_pp']:+.2f} +/- {row['paired_delta_std_pp']:.2f} pp |"
        )
    lines.append("")
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
