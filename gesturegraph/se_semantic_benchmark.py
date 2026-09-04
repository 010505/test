from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from .improvement_benchmark import run_and_log


EXPERIMENTS = [
    ("00_stem_control", "stem_stgcn", None),
    ("01_stem_linear_se_k8", "stem_linear_se", 8),
    ("02_stem_linear_se_k16", "stem_linear_se", 16),
    ("03_stem_linear_se_k21", "stem_linear_se", 21),
    ("04_stem_residual_mlp_se_k21", "stem_residual_mlp_se", 21),
    ("05_stem_residual_mlp_gated_se_k21", "stem_residual_mlp_gated_se", 21),
    (
        "06_stem_residual_mlp_learnable_values_se_k21",
        "stem_residual_mlp_learnable_values_se",
        21,
    ),
]


def write_report(root: Path, rows: list[dict]) -> None:
    control = next((row for row in rows if row["experiment"] == "00_stem_control"), None)
    control_test = control.get("official_test_accuracy") if control else None
    lines = [
        "# Stem-first semantic spectral encoding ablation",
        "",
        "All models first lift XYZ from 3 to 32 channels. The stem-only model is",
        "the matched control. Checkpoints are selected by validation accuracy.",
        "",
        "| Experiment | Backbone | PE dim | Parameters | Best validation | Official test | Delta vs stem |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        test = row.get("official_test_accuracy")
        pe_dim = "-" if row["architecture"] == "stem_stgcn" else row["model_config"]["pe_dim"]
        delta = (test - control_test) * 100 if test is not None and control_test is not None else None
        lines.append(
            f"| {row['experiment']} | {row['architecture']} | {pe_dim} | {row['parameters']:,} | "
            f"{row['best_validation_accuracy']:.2%} | "
            f"{test:.2%} | {delta:+.2f} pp |"
            if test is not None and delta is not None
            else f"| {row['experiment']} | {row['architecture']} | {pe_dim} | {row['parameters']:,} | "
            f"{row['best_validation_accuracy']:.2%} | n/a | n/a |"
        )
    lines.extend([
        "",
        "Definitions:",
        "",
        "- `stem_linear_se`: fixed eigenvalue-weighted Laplacian SE with a direct linear branch.",
        "- `stem_residual_mlp_se`: the linear branch plus a learnable residual ReLU MLP.",
        "- `stem_residual_mlp_gated_se`: residual MLP plus positive bounded per-layer spectral gates.",
        "- `stem_residual_mlp_learnable_values_se`: fixed eigenvectors plus directly optimized per-layer spectral values initialised from the Laplacian eigenvalues.",
        "",
    ])
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stem-first semantic SE ablations")
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset", default="shrec17_npz", choices=["shrec17", "shrec17_npz"])
    parser.add_argument("--output", default="runs/se_semantic_ablation")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--stem-channels", type=int, default=32)
    parser.add_argument("--semantic-hidden", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", nargs="*", choices=[name for name, _, _ in EXPERIMENTS])
    args = parser.parse_args()

    selected = [item for item in EXPERIMENTS if not args.only or item[0] in args.only]
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    provenance_path = Path(args.data) / "provenance.json"
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": str(Path(args.data).resolve()),
        "dataset": args.dataset,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "frames": args.frames,
        "seed": args.seed,
        "device": args.device,
        "stem_channels": args.stem_channels,
        "semantic_hidden": args.semantic_hidden,
        "experiments": [
            {"name": name, "architecture": model, "pe_dim": pe_dim}
            for name, model, pe_dim in selected
        ],
        "dataset_provenance": json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.exists() else None,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows: list[dict] = []
    for experiment, model, pe_dim in selected:
        destination = root / experiment
        metadata_path = destination / "model.json"
        if metadata_path.exists() and args.resume:
            print(f"\n=== {experiment}: reuse completed result ===", flush=True)
        else:
            if destination.exists() and any(destination.iterdir()):
                raise FileExistsError(
                    f"{destination} already contains files; use a new output root or --resume"
                )
            destination.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, "-m", "gesturegraph.train",
                "--dataset", args.dataset,
                "--data", args.data,
                "--output", str(destination),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--frames", str(args.frames),
                "--seed", str(args.seed),
                "--device", args.device,
                "--model", model,
                "--pe-dim", str(pe_dim or 8),
                "--stem-channels", str(args.stem_channels),
                "--semantic-hidden", str(args.semantic_hidden),
            ]
            print(f"\n=== {experiment} | {model} ===", flush=True)
            (destination / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
            run_and_log(command, destination / "train.log")
        rows.append({"experiment": experiment, **json.loads(metadata_path.read_text(encoding="utf-8"))})
        (root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        write_report(root, rows)

    print("\nStem-first semantic SE summary")
    for row in rows:
        print(
            f"{row['experiment']:40s} val={row['best_validation_accuracy']:.2%} "
            f"test={row['official_test_accuracy']:.2%}"
        )


if __name__ == "__main__":
    main()
