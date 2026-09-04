from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


QKV_EXPERIMENTS = ("00_stable_qkv_control", "01_semantic_stable_qkv")
GRAPH_EXPERIMENTS = (
    "02_gwnet_gated_control",
    "03_semantic_gwnet_gated",
    "04_agcrn_dynamic_control",
    "05_semantic_agcrn_dynamic",
)


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose v2 attention and graph fusion parameters")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--previous-aggregate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    roots = [Path(path) for path in args.runs]
    aggregate = {
        row["family"]: row
        for row in json.loads(Path(args.aggregate).read_text(encoding="utf-8"))
    }

    diagnostics: dict[str, object] = {"qkv": {}, "branch_gates": {}}
    for experiment in QKV_EXPERIMENTS:
        temperatures: list[float] = []
        residual_scales: list[float] = []
        for root in roots:
            state = torch.load(root / experiment / "best.pt", map_location="cpu", weights_only=True)["model_state"]
            for name, value in state.items():
                if name.endswith("log_temperature"):
                    temperatures.extend(value.exp().tolist())
                elif name.endswith("residual_scale"):
                    residual_scales.append(float(value))
        diagnostics["qkv"][experiment] = {
            "temperatures": describe(temperatures),
            "residual_scales": describe(residual_scales),
        }

    for experiment in GRAPH_EXPERIMENTS:
        gates: list[float] = []
        for root in roots:
            archive = np.load(root / experiment / "adjacency_matrices.npz")
            gates.extend(archive["dynamic_branch_gates"].tolist())
        diagnostics["branch_gates"][experiment] = describe(gates)

    gwn = aggregate["gwnet_gated"]
    agcrn = aggregate["agcrn_dynamic"]
    diagnostics["agcrn_minus_gwn_pp"] = {
        "without_semantic_se": (agcrn["control_mean"] - gwn["control_mean"]) * 100.0,
        "with_semantic_se": (agcrn["semantic_mean"] - gwn["semantic_mean"]) * 100.0,
    }

    if args.previous_aggregate:
        previous = {
            row["family"]: row
            for row in json.loads(Path(args.previous_aggregate).read_text(encoding="utf-8"))
        }
        diagnostics["qkv_vs_previous"] = {
            "control_mean_delta_pp": (
                aggregate["stable_qkv"]["control_mean"] - previous["qkv"]["control_mean"]
            ) * 100.0,
            "control_std_ratio": (
                aggregate["stable_qkv"]["control_std"] / previous["qkv"]["control_std"]
            ),
            "semantic_mean_delta_pp": (
                aggregate["stable_qkv"]["semantic_mean"] - previous["qkv"]["semantic_mean"]
            ) * 100.0,
            "semantic_std_ratio": (
                aggregate["stable_qkv"]["semantic_std"] / previous["qkv"]["semantic_std"]
            ),
        }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    lines = [
        "# V2 learned-parameter diagnostics",
        "",
        f"- AGCRN dynamic factorisation minus matched GWN: "
        f"{diagnostics['agcrn_minus_gwn_pp']['without_semantic_se']:+.2f} pp without SE and "
        f"{diagnostics['agcrn_minus_gwn_pp']['with_semantic_se']:+.2f} pp with SE.",
    ]
    for experiment, values in diagnostics["branch_gates"].items():
        lines.append(
            f"- `{experiment}` dynamic gate: {values['mean']:.4f} +/- {values['std']:.4f}, "
            f"range {values['minimum']:.4f}-{values['maximum']:.4f}."
        )
    if "qkv_vs_previous" in diagnostics:
        comparison = diagnostics["qkv_vs_previous"]
        lines.extend([
            f"- Stable QKV control mean change: {comparison['control_mean_delta_pp']:+.2f} pp; "
            f"std is {comparison['control_std_ratio']:.2f}x the previous QKV control.",
            f"- Stable semantic-QKV mean change: {comparison['semantic_mean_delta_pp']:+.2f} pp; "
            f"std is {comparison['semantic_std_ratio']:.2f}x the previous semantic QKV.",
        ])
    (output / "DIAGNOSTICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
