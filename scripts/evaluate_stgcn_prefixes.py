from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gesturegraph.model import build_model
from gesturegraph.progressive import (
    DEFAULT_OBSERVATION_RATIOS,
    PrefixSequenceDataset,
    load_raw_shrec17_npz,
)
from gesturegraph.progressive_benchmark import prefix_auc

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the original ST-GCN checkpoint on causal observation prefixes."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "runs" / "shrec17_benchmark" / "stgcn_full" / "best.pt"),
    )
    parser.add_argument(
        "--data",
        default=str(ROOT / "data" / "shrec17_ddnet_npz"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "runs" / "stgcn_prefix_evaluation" / "metrics.json"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    device = choose_device(args.device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    labels = checkpoint["labels"]
    frames = int(checkpoint.get("frames", 64))
    model = build_model(
        checkpoint.get("model_name", "stgcn"),
        len(labels),
        frames,
        float(checkpoint.get("dropout", 0.15)),
        checkpoint.get("ablation", "none"),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    ratios = tuple(float(value) for value in DEFAULT_OBSERVATION_RATIOS)
    samples = load_raw_shrec17_npz(args.data, "test", classes=14)
    dataset = PrefixSequenceDataset(
        samples,
        labels,
        frames=frames,
        ratios=ratios,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    correct = np.zeros(len(ratios), dtype=np.int64)
    total = 0
    with torch.inference_mode():
        for views, _lengths, _progress, targets in loader:
            batch, steps = views.shape[:2]
            logits = model(views.flatten(0, 1).to(device)).reshape(batch, steps, -1)
            predictions = logits.argmax(dim=-1).cpu()
            correct += (predictions == targets[:, None]).sum(dim=0).numpy()
            total += batch

    accuracies = correct.astype(np.float64) / total
    result = {
        "method": "Original ST-GCN",
        "checkpoint": str(checkpoint_path),
        "checkpoint_best_validation_accuracy": float(checkpoint["best_accuracy"]),
        "data": str(Path(args.data).resolve()),
        "split": "test",
        "samples": total,
        "observation_ratios": list(ratios),
        "protocol": (
            "Causal prefixes are resampled only within the observed segment, padded by "
            "repeating the last observed pose to 64 frames, and evaluated without augmentation."
        ),
        "run_count": 1,
        "uncertainty_note": "Single released checkpoint; no cross-seed standard deviation is available.",
        "ratio_accuracy": {
            f"{ratio:.2f}": {"mean": float(accuracy), "std": 0.0}
            for ratio, accuracy in zip(ratios, accuracies)
        },
        "prefix_auc": {
            "mean": prefix_auc(np.asarray(ratios), accuracies),
            "std": 0.0,
        },
        "full_accuracy": float(accuracies[-1]),
        "device": str(device),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
