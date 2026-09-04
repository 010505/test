from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import GestureDataset
from .model import build_model
from .shrec import load_shrec17, load_shrec17_npz

JOINT_GROUPS = {
    "wrist_palm": [0, 1],
    "thumb": [2, 3, 4, 5],
    "index": [6, 7, 8, 9],
    "middle": [10, 11, 12, 13],
    "ring": [14, 15, 16, 17],
    "little": [18, 19, 20, 21],
}


def accuracy(model, loader, masked_nodes=None):
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            if masked_nodes:
                inputs[:, :, :, masked_nodes] = 0
            predictions = model(inputs).argmax(dim=1)
            correct += (predictions == targets).sum().item(); total += len(targets)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Finger masking interpretation on the official SHREC test split")
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset", default="shrec17", choices=["shrec17", "shrec17_npz"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="runs/joint_ablation.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    labels = checkpoint["labels"]; frames = int(checkpoint.get("frames", 64))
    model = build_model(checkpoint.get("model_name", "stgcn"), len(labels), frames, float(checkpoint.get("dropout", .15)), checkpoint.get("ablation", "none"))
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    loader_fn = load_shrec17_npz if args.dataset == "shrec17_npz" else load_shrec17
    samples = loader_fn(args.data, "test", frames, 14)
    loader = DataLoader(GestureDataset(samples, labels), batch_size=args.batch_size)
    baseline = accuracy(model, loader)
    results = {"baseline_accuracy": baseline, "groups": {}}
    for name, nodes in JOINT_GROUPS.items():
        masked = accuracy(model, loader, nodes)
        results["groups"][name] = {"masked_accuracy": masked, "accuracy_drop": baseline - masked, "nodes": nodes}
        print(f"{name:12s} masked={masked:.1%} drop={baseline-masked:+.1%}")
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
