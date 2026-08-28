from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENTS = [
    ("01_spectral_pe", "spectral_pe_stgcn"),
    ("02_qkv_spatial_attention", "spectral_pe_qkv"),
    ("03_gwnet_adaptive_support", "gwnet_adaptive_support"),
    ("04_agcrn_factorized_adjacency", "agcrn_factorized_adjacency"),
]
CONTROL = ("00_stgcn_control", "stgcn")


def run_and_log(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def load_baseline(path: Path) -> dict:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return next((row for row in rows if row.get("experiment") == "stgcn_full"), {})


def write_report(root: Path, baseline: dict, results: list[dict]) -> None:
    baseline_test = baseline.get("official_test_accuracy")
    baseline_parameters = baseline.get("parameters", "n/a")
    baseline_parameters_text = (
        f"{baseline_parameters:,}" if isinstance(baseline_parameters, int) else str(baseline_parameters)
    )
    lines = [
        "# GestureGraph independent backbone experiments",
        "",
        "All experiments use the official SHREC'17 split, the same seed, frame count, optimiser, augmentation, and epoch budget.",
        f"Dataset mode: `{(baseline or results[0]).get('dataset', 'unknown') if (baseline or results) else 'unknown'}`.",
        "",
        "| Experiment | Backbone | Parameters | Best validation | Official test | Delta vs ST-GCN |",
        "|---|---|---:|---:|---:|---:|",
    ]
    if baseline:
        lines.append(
            f"| Baseline | stgcn | {baseline_parameters_text} | "
            f"{baseline['best_validation_accuracy']:.2%} | {baseline_test:.2%} | 0.00 pp |"
        )
    for row in results:
        test = row.get("official_test_accuracy")
        delta = (test - baseline_test) * 100 if test is not None and baseline_test is not None else None
        lines.append(
            f"| {row['experiment']} | {row['architecture']} | {row.get('parameters', 'n/a'):,} | "
            f"{row['best_validation_accuracy']:.2%} | "
            f"{test:.2%} | {delta:+.2f} pp |" if test is not None and delta is not None else
            f"| {row['experiment']} | {row['architecture']} | {row.get('parameters', 'n/a'):,} | "
            f"{row['best_validation_accuracy']:.2%} | n/a | n/a |"
        )
    lines.extend([
        "",
        "Experiment 3 stores the fixed physical and full learned adaptive supports in ",
        "`03_gwnet_adaptive_support/adjacency_matrices.npz` for reproducible visualisation.",
        "",
    ])
    if (baseline or results) and (baseline or results[0]).get("dataset") == "shrec17_npz":
        lines.extend([
            "This run uses the DD-Net copy of the official 1960/840 skeleton split.",
            "DD-Net median-filtered every coordinate channel before serialisation; the",
            "same-run ST-GCN control is therefore the only valid baseline for deltas.",
            "",
        ])

    comparison_rows = ([baseline] if baseline else []) + results
    confusion_rows = []
    for row in comparison_rows:
        experiment = row.get("experiment")
        confusion_path = root / str(experiment) / "test_confusion.json" if experiment else None
        if confusion_path and confusion_path.exists():
            payload = json.loads(confusion_path.read_text(encoding="utf-8"))
            matrix = payload["matrix"]
            accuracy = [
                matrix[index][index] / max(sum(matrix[index]), 1)
                for index in range(len(matrix))
            ]
            confusion_rows.append((experiment, payload["labels"], accuracy))
    if confusion_rows:
        lines.extend([
            "## Per-class official-test accuracy",
            "",
            "| Class | " + " | ".join(name for name, _, _ in confusion_rows) + " |",
            "|---|" + "|".join("---:" for _ in confusion_rows) + "|",
        ])
        labels = confusion_rows[0][1]
        for index, label in enumerate(labels):
            lines.append(
                f"| {label} | "
                + " | ".join(f"{accuracy[index]:.1%}" for _, _, accuracy in confusion_rows)
                + " |"
            )
        lines.append("")
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run four independent GestureGraph backbone improvements")
    parser.add_argument("--data", required=True, help="HandGestureDataset_SHREC2017 directory")
    parser.add_argument("--dataset", default="shrec17", choices=["shrec17", "shrec17_npz"])
    parser.add_argument("--output", default="runs/improvement_backbones")
    parser.add_argument("--baseline-summary", default="runs/shrec17_benchmark/summary.json")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--pe-dim", type=int, default=8)
    parser.add_argument("--attention-heads", "--gat-heads", dest="attention_heads", type=int, default=4)
    parser.add_argument("--adaptive-dim", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="Reuse completed experiment directories")
    parser.add_argument("--include-control", action="store_true", help="Retrain ST-GCN on the identical data and protocol")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=[name for name, _ in EXPERIMENTS],
        help="Run only selected experiment directory names",
    )
    args = parser.parse_args()

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    selected = [item for item in EXPERIMENTS if not args.only or item[0] in args.only]
    if args.include_control:
        selected = [CONTROL, *selected]
    dataset_provenance_path = Path(args.data) / "provenance.json"
    dataset_provenance = (
        json.loads(dataset_provenance_path.read_text(encoding="utf-8"))
        if dataset_provenance_path.exists()
        else None
    )
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
        "pe_dim": args.pe_dim,
        "attention_heads": args.attention_heads,
        "adaptive_dim": args.adaptive_dim,
        "experiments": [{"name": name, "architecture": model} for name, model in selected],
        "method_references": {
            "spatial_qkv": "Per-frame full-joint scaled dot-product attention after fixed eigenvalue-weighted Laplacian spatial encoding; no temporal position encoding",
            "adaptive_adjacency": "Graph WaveNet: two node embeddings followed by ReLU and row softmax",
            "factorized_weights": "AGCRN AVWGCN: node embeddings combine a shared weight pool into node-specific W",
        },
        "dataset_provenance": dataset_provenance,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    results = []
    baseline = load_baseline(Path(args.baseline_summary))
    for experiment, model in selected:
        destination = root / experiment
        metadata_path = destination / "model.json"
        if metadata_path.exists() and args.resume:
            print(f"\n=== {experiment}: reuse completed result ===", flush=True)
        else:
            if destination.exists() and any(destination.iterdir()):
                raise FileExistsError(
                    f"{destination} already contains files; use a new output root or --resume for completed runs"
                )
            destination.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "gesturegraph.train",
                "--dataset",
                args.dataset,
                "--data",
                args.data,
                "--output",
                str(destination),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--frames",
                str(args.frames),
                "--seed",
                str(args.seed),
                "--device",
                args.device,
                "--model",
                model,
                "--pe-dim",
                str(args.pe_dim),
                "--attention-heads",
                str(args.attention_heads),
                "--adaptive-dim",
                str(args.adaptive_dim),
            ]
            print(f"\n=== {experiment} | {model} ===", flush=True)
            (destination / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
            run_and_log(command, destination / "train.log")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        results.append({"experiment": experiment, **metadata})
        (root / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        same_run_control = next((row for row in results if row["experiment"] == CONTROL[0]), None)
        comparison_baseline = same_run_control or baseline
        comparison_results = [row for row in results if row["experiment"] != CONTROL[0]]
        write_report(root, comparison_baseline, comparison_results)

    print("\nIndependent-backbone summary")
    for row in results:
        test = row.get("official_test_accuracy")
        print(
            f"{row['experiment']:34s} val={row['best_validation_accuracy']:.2%} "
            f"test={test:.2%}" if test is not None else
            f"{row['experiment']:34s} val={row['best_validation_accuracy']:.2%} test=n/a"
        )


if __name__ == "__main__":
    main()
