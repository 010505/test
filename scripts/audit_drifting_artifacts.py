from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gesturegraph.drifting import ONE_STEP_EXPERIMENTS
from gesturegraph.drifting_benchmark import BASELINE_EXPERIMENTS


def main():
    parser = argparse.ArgumentParser(description="Audit one-step drifting experiment artifacts")
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    experiments = BASELINE_EXPERIMENTS + ONE_STEP_EXPERIMENTS
    reference_targets = None
    for root_name in args.runs:
        root = Path(root_name)
        print(root)
        for experiment in experiments:
            directory = root / experiment
            result = json.loads((directory / "model.json").read_text(encoding="utf-8"))
            predictions = np.load(directory / "test_predictions.npz")
            probabilities = predictions["probabilities"]
            targets = predictions["targets"]
            if probabilities.shape != (840, 5, 14):
                raise ValueError(f"unexpected prediction shape: {directory}: {probabilities.shape}")
            if not np.isfinite(probabilities).all():
                raise ValueError(f"non-finite probabilities: {directory}")
            if not np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-5):
                raise ValueError(f"unnormalized probabilities: {directory}")
            if reference_targets is None:
                reference_targets = targets
            elif not np.array_equal(reference_targets, targets):
                raise ValueError(f"test target ordering differs: {directory}")
            suffix = ""
            checkpoint_path = directory / "best.pt"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                if any(
                    "teacher" in key or "bank" in key
                    for key in checkpoint["model_state"]
                ):
                    raise ValueError(f"training-only state leaked into checkpoint: {directory}")
                suffix = f" best_epoch={checkpoint['best_epoch']}"
            print(
                f"  {experiment}: {probabilities.shape} "
                f"AUC={result['official_test']['prefix_auc']:.4f}{suffix}"
            )
    print("artifact audit: OK")


if __name__ == "__main__":
    main()

