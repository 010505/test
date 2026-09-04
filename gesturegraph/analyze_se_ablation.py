from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import build_model


CONTROL = "00_stem_control"
GATED = "05_stem_residual_mlp_gated_se_k21"
MLP_EXPERIMENTS = {"04_stem_residual_mlp_se_k21", GATED}


def class_accuracy(path: Path) -> tuple[list[str], np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray(payload["matrix"], dtype=np.float64)
    return payload["labels"], np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse trained semantic-SE ablations")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    roots = [Path(path) for path in args.runs]
    rows_by_run = [
        {row["experiment"]: row for row in json.loads((root / "summary.json").read_text(encoding="utf-8"))}
        for root in roots
    ]
    experiment_order = list(rows_by_run[0])
    labels: list[str] | None = None
    class_rows = []
    for experiment in experiment_order:
        scores = []
        controls = []
        for root in roots:
            current_labels, current = class_accuracy(root / experiment / "test_confusion.json")
            control_labels, control = class_accuracy(root / CONTROL / "test_confusion.json")
            if current_labels != control_labels:
                raise ValueError("class label order differs")
            labels = current_labels
            scores.append(current)
            controls.append(control)
        score_array = np.stack(scores)
        control_array = np.stack(controls)
        class_rows.append({
            "experiment": experiment,
            "mean_accuracy": score_array.mean(axis=0).tolist(),
            "std_accuracy": score_array.std(axis=0, ddof=1).tolist(),
            "paired_delta_pp": ((score_array - control_array) * 100.0).mean(axis=0).tolist(),
        })

    gate_runs = []
    scale_runs = []
    for root, rows in zip(roots, rows_by_run):
        for experiment in MLP_EXPERIMENTS:
            checkpoint = torch.load(root / experiment / "best.pt", map_location="cpu", weights_only=True)
            model = build_model(
                checkpoint["model_name"],
                len(checkpoint["labels"]),
                checkpoint["frames"],
                checkpoint["dropout"],
                checkpoint["ablation"],
                checkpoint["model_config"],
            )
            model.load_state_dict(checkpoint["model_state"])
            scale_runs.append({
                "seed": rows[experiment]["seed"],
                "experiment": experiment,
                "position_scale": [float(block.spatial.position_scale.detach()) for block in model.blocks],
                "semantic_scale": [float(block.spatial.semantic_scale.detach()) for block in model.blocks],
            })
            if experiment == GATED:
                multipliers = [
                    (block.spatial.spectral_weights() / block.spatial.spectral_eigenvalues).detach().numpy()
                    for block in model.blocks
                ]
                gate_runs.append({
                    "seed": rows[experiment]["seed"],
                    "multipliers_by_layer": np.stack(multipliers).tolist(),
                })

    gate_array = np.stack([row["multipliers_by_layer"] for row in gate_runs])
    analysis = {
        "labels": labels,
        "per_class": class_rows,
        "learned_scales": scale_runs,
        "gate": {
            "seeds": [row["seed"] for row in gate_runs],
            "multipliers_by_seed_layer_frequency": gate_array.tolist(),
            "mean_by_layer_frequency": gate_array.mean(axis=0).tolist(),
            "mean": float(gate_array.mean()),
            "std": float(gate_array.std(ddof=1)),
            "min": float(gate_array.min()),
            "max": float(gate_array.max()),
        },
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "se_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    assert labels is not None
    lines = [
        "# Semantic SE per-class and learned-gate analysis",
        "",
        "Mean paired class-accuracy delta against the stem-only control across seeds 42/43/44.",
        "",
        "| Class | Linear K21 | Residual MLP K21 | MLP + gate K21 |",
        "|---|---:|---:|---:|",
    ]
    by_experiment = {row["experiment"]: row for row in class_rows}
    selected = [
        by_experiment["03_stem_linear_se_k21"],
        by_experiment["04_stem_residual_mlp_se_k21"],
        by_experiment[GATED],
    ]
    for index, label in enumerate(labels):
        lines.append(
            f"| {label} | " + " | ".join(
                f"{row['paired_delta_pp'][index]:+.2f} pp" for row in selected
            ) + " |"
        )
    lines.extend([
        "",
        "## Learned spectral gate",
        "",
        f"Across 3 seeds × 4 layers × 21 frequencies, multiplicative gates have "
        f"mean {analysis['gate']['mean']:.4f}, std {analysis['gate']['std']:.4f}, "
        f"range [{analysis['gate']['min']:.4f}, {analysis['gate']['max']:.4f}].",
        "A value of 1.0 is the fixed-eigenvalue initialisation.",
        "",
    ])
    (output / "SE_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
