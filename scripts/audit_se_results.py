from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gesturegraph.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit semantic-SE training artifacts")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--expected-epochs", type=int, default=40)
    parser.add_argument("--expected-experiments", type=int, default=6)
    args = parser.parse_args()

    checkpoints = 0
    for root_text in args.runs:
        root = Path(root_text)
        rows = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        if len(rows) != args.expected_experiments:
            raise ValueError(
                f"{root}: expected {args.expected_experiments} experiments, got {len(rows)}"
            )
        for row in rows:
            directory = root / row["experiment"]
            required = ["best.pt", "history.json", "model.json", "test_confusion.json", "command.json", "train.log"]
            missing = [name for name in required if not (directory / name).exists()]
            if missing:
                raise FileNotFoundError(f"{directory}: missing {missing}")
            history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
            if len(history) != args.expected_epochs:
                raise ValueError(f"{directory}: expected {args.expected_epochs} epochs, got {len(history)}")
            checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
            model = build_model(
                checkpoint["model_name"],
                len(checkpoint["labels"]),
                checkpoint["frames"],
                checkpoint["dropout"],
                checkpoint["ablation"],
                checkpoint["model_config"],
            ).eval()
            model.load_state_dict(checkpoint["model_state"])
            with torch.no_grad():
                logits = model(torch.zeros(1, 3, checkpoint["frames"], 22))
            if logits.shape != (1, 14) or not torch.isfinite(logits).all():
                raise ValueError(f"{directory}: invalid logits")
            checkpoints += 1
    print(json.dumps({"run_roots": len(args.runs), "checkpoints": checkpoints, "status": "ok"}))


if __name__ == "__main__":
    main()
