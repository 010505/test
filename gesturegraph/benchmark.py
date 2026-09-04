from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = [
    ("stgcn_full", "stgcn", "none"),
    ("mlp_baseline", "mlp", "none"),
    ("stgcn_no_graph", "stgcn", "no_graph"),
    ("stgcn_single_frame", "stgcn", "single_frame"),
]


def main():
    parser = argparse.ArgumentParser(description="Run the Project A baseline and ablation matrix")
    parser.add_argument("--data", required=True, help="HandGestureDataset_SHREC2017 directory")
    parser.add_argument("--output", default="runs/shrec17_benchmark")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    results = []
    for experiment, model, ablation in EXPERIMENTS:
        destination = root / experiment
        command = [
            sys.executable, "-m", "gesturegraph.train", "--dataset", "shrec17",
            "--data", args.data, "--output", str(destination), "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size), "--device", args.device,
            "--model", model, "--ablation", ablation,
        ]
        print(f"\n=== {experiment} ===", flush=True)
        subprocess.run(command, check=True)
        metadata = json.loads((destination / "model.json").read_text(encoding="utf-8"))
        results.append({"experiment": experiment, **metadata})
        (root / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nExperiment summary")
    for row in results:
        print(f"{row['experiment']:22s} val={row['best_validation_accuracy']:.1%} test={row['official_test_accuracy']:.1%}")


if __name__ == "__main__":
    main()
