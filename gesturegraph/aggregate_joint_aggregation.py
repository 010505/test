from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PAIRS = [
    ("gwnet_joint", "02_stem_gwnet_control", "03_stem_semantic_gwnet"),
    ("agcrn_joint", "04_stem_agcrn_control", "05_stem_semantic_agcrn"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate joint-support GWN/AGCRN rechecks")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gated-aggregate")
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
            "paired_se_delta_pp": delta.tolist(),
            "paired_se_delta_mean_pp": float(delta.mean()),
            "paired_se_delta_std_pp": float(delta.std(ddof=1)),
        })
    by_family = {row["family"]: row for row in rows}
    gwn, agcrn = by_family["gwnet_joint"], by_family["agcrn_joint"]
    comparison = {
        "agcrn_minus_gwn_without_se_pp": (agcrn["control_mean"] - gwn["control_mean"]) * 100.0,
        "agcrn_minus_gwn_with_se_pp": (agcrn["semantic_mean"] - gwn["semantic_mean"]) * 100.0,
    }
    if args.gated_aggregate:
        gated = {
            row["family"]: row
            for row in json.loads(Path(args.gated_aggregate).read_text(encoding="utf-8"))
        }
        comparison["joint_minus_gated_pp"] = {
            "gwn_without_se": (gwn["control_mean"] - gated["gwnet_gated"]["control_mean"]) * 100.0,
            "gwn_with_se": (gwn["semantic_mean"] - gated["gwnet_gated"]["semantic_mean"]) * 100.0,
            "agcrn_without_se": (agcrn["control_mean"] - gated["agcrn_dynamic"]["control_mean"]) * 100.0,
            "agcrn_with_se": (agcrn["semantic_mean"] - gated["agcrn_dynamic"]["semantic_mean"]) * 100.0,
        }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "aggregate.json").write_text(
        json.dumps({"families": rows, "comparison": comparison}, indent=2), encoding="utf-8"
    )
    lines = [
        "# Joint-support aggregation recheck",
        "",
        "| Operator | No semantic SE | With semantic SE | Paired SE delta |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['family']} | {row['control_mean']:.2%} +/- {row['control_std']:.2%} | "
            f"{row['semantic_mean']:.2%} +/- {row['semantic_std']:.2%} | "
            f"{row['paired_se_delta_mean_pp']:+.2f} +/- {row['paired_se_delta_std_pp']:.2f} pp |"
        )
    lines.extend([
        "",
        f"AGCRN minus matched GWN is {comparison['agcrn_minus_gwn_without_se_pp']:+.2f} pp "
        f"without SE and {comparison['agcrn_minus_gwn_with_se_pp']:+.2f} pp with SE.",
    ])
    if "joint_minus_gated_pp" in comparison:
        values = comparison["joint_minus_gated_pp"]
        lines.extend([
            "",
            "| Joint aggregation minus gated v2 | No semantic SE | With semantic SE |",
            "|---|---:|---:|",
            f"| GWN | {values['gwn_without_se']:+.2f} pp | {values['gwn_with_se']:+.2f} pp |",
            f"| AGCRN | {values['agcrn_without_se']:+.2f} pp | {values['agcrn_with_se']:+.2f} pp |",
        ])
    lines.append("")
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
