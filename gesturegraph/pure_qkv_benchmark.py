from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from .improvement_benchmark import run_and_log


EXPERIMENTS = [
    ("00_pure_qkv_control", "stem_pure_qkv_control", False),
    ("01_semantic_pure_qkv", "stem_semantic_pure_qkv", True),
]


def write_report(root: Path, rows: list[dict]) -> None:
    by_name = {row["experiment"]: row for row in rows}
    control = by_name.get("00_pure_qkv_control")
    lines = [
        "# Pure QKV attention without physical-adjacency bias",
        "",
        "| Experiment | Parameters | Best validation | Official test | Paired SE delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for experiment, _, _ in EXPERIMENTS:
        if experiment not in by_name:
            continue
        row = by_name[experiment]
        delta = (
            (row["official_test_accuracy"] - control["official_test_accuracy"]) * 100
            if control else 0.0
        )
        lines.append(
            f"| {experiment} | {row['parameters']:,} | {row['best_validation_accuracy']:.2%} | "
            f"{row['official_test_accuracy']:.2%} | {delta:+.2f} pp |"
        )
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pure QKV with and without semantic SE")
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset", default="shrec17_npz", choices=["shrec17", "shrec17_npz"])
    parser.add_argument("--output", default="runs/pure_qkv")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--pe-dim", type=int, default=21)
    parser.add_argument("--stem-channels", type=int, default=32)
    parser.add_argument("--semantic-hidden", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
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
        "stem_channels": args.stem_channels,
        "semantic_hidden": args.semantic_hidden,
        "attention_heads": args.attention_heads,
        "physical_attention_bias": False,
        "experiments": [
            {"name": name, "architecture": model, "semantic_se": uses_se}
            for name, model, uses_se in EXPERIMENTS
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = []
    for experiment, model, _ in EXPERIMENTS:
        destination = root / experiment
        metadata_path = destination / "model.json"
        if metadata_path.exists() and args.resume:
            print(f"\n=== {experiment}: reuse completed result ===", flush=True)
        else:
            if destination.exists() and any(destination.iterdir()):
                raise FileExistsError(f"{destination} contains files; use a new root or --resume")
            destination.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, "-m", "gesturegraph.train",
                "--dataset", args.dataset, "--data", args.data,
                "--output", str(destination), "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size), "--frames", str(args.frames),
                "--seed", str(args.seed), "--device", args.device, "--model", model,
                "--pe-dim", str(args.pe_dim), "--stem-channels", str(args.stem_channels),
                "--semantic-hidden", str(args.semantic_hidden),
                "--attention-heads", str(args.attention_heads),
            ]
            print(f"\n=== {experiment} | {model} ===", flush=True)
            (destination / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
            run_and_log(command, destination / "train.log")
        rows.append({"experiment": experiment, **json.loads(metadata_path.read_text(encoding="utf-8"))})
        (root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        write_report(root, rows)


if __name__ == "__main__":
    main()
