from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gesturegraph.progressive import (
    EXTENDED_PROGRESSIVE_EXPERIMENTS,
    PROGRESSIVE_EXPERIMENTS,
    build_progressive_model,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--expected-epochs", type=int, default=40)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXTENDED_PROGRESSIVE_EXPERIMENTS,
        default=list(PROGRESSIVE_EXPERIMENTS),
    )
    args = parser.parse_args()
    checkpoints = 0
    for root_value in args.runs:
        root = Path(root_value)
        for experiment in args.experiments:
            directory = root / experiment
            history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
            metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
            checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
            if len(history) != args.expected_epochs or metadata["epochs"] != args.expected_epochs:
                raise AssertionError(f"{directory}: incomplete history")
            model = build_progressive_model(experiment, len(checkpoint["labels"]))
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            if experiment in PROGRESSIVE_EXPERIMENTS[:3]:
                logits = model(torch.zeros(1, 3, 64, 22), torch.tensor([64]), torch.tensor([1.0]))
                expected = (1, 14)
            else:
                logits = model(
                    torch.zeros(1, 5, 3, 64, 22),
                    torch.tensor([[16, 32, 42, 52, 64]]),
                    torch.tensor([[0.25, 0.50, 0.65, 0.80, 1.00]]),
                )
                expected = (1, 5, 14)
            if tuple(logits.shape) != expected or not torch.isfinite(logits).all():
                raise AssertionError(f"{directory}: invalid reload output")
            checkpoints += 1
    print(json.dumps({"run_roots": len(args.runs), "checkpoints": checkpoints, "status": "ok"}))


if __name__ == "__main__":
    main()
