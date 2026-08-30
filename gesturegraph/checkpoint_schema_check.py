from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import build_model

REQUIRED_KEYS = ("model_state", "labels", "frames", "best_accuracy", "model_name", "ablation", "dropout")


def missing_keys(checkpoint: dict) -> list[str]:
    return [key for key in REQUIRED_KEYS if key not in checkpoint]


def validate_labels(labels) -> list[str]:
    errors = []
    if not isinstance(labels, list):
        errors.append(f"labels must be a list, got {type(labels).__name__}")
        return errors
    if not labels:
        errors.append("labels list is empty")
    if not all(isinstance(label, str) for label in labels):
        errors.append("labels must all be strings")
    if len(set(labels)) != len(labels):
        errors.append("labels contain duplicates")
    return errors


def validate_frames(frames) -> list[str]:
    if not isinstance(frames, int) or isinstance(frames, bool):
        return [f"frames must be an int, got {type(frames).__name__}"]
    if frames < 2:
        return ["frames must be at least 2"]
    return []


def validate_best_accuracy(best_accuracy) -> list[str]:
    if not isinstance(best_accuracy, (int, float)) or isinstance(best_accuracy, bool):
        return [f"best_accuracy must be numeric, got {type(best_accuracy).__name__}"]
    if not 0.0 <= float(best_accuracy) <= 1.0:
        return [f"best_accuracy {best_accuracy} is outside [0, 1]"]
    return []


def validate_dropout(dropout) -> list[str]:
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
        return [f"dropout must be numeric, got {type(dropout).__name__}"]
    if not 0.0 <= float(dropout) < 1.0:
        return [f"dropout {dropout} is outside [0, 1)"]
    return []


def validate_model_state(model_state) -> list[str]:
    if not isinstance(model_state, dict):
        return [f"model_state must be a dict, got {type(model_state).__name__}"]
    if not model_state:
        return ["model_state is empty"]
    errors = []
    for name, value in model_state.items():
        if not isinstance(value, torch.Tensor):
            errors.append(f"model_state[{name!r}] is not a tensor")
            continue
        if not torch.isfinite(value).all():
            errors.append(f"model_state[{name!r}] contains NaN or infinite values")
    return errors


def validate_model_name_buildable(model_name, num_classes: int, frames: int, dropout: float, ablation) -> list[str]:
    if not isinstance(model_name, str) or not model_name:
        return [f"model_name must be a non-empty string, got {model_name!r}"]
    try:
        build_model(model_name, num_classes, frames, dropout, ablation if isinstance(ablation, str) else "none")
    except ValueError as error:
        return [f"model_name {model_name!r} is not buildable: {error}"]
    return []


def validate_checkpoint(checkpoint: dict) -> dict:
    missing = missing_keys(checkpoint)
    if missing:
        return {"valid": False, "errors": [f"missing required key: {key}" for key in missing]}

    errors = []
    errors.extend(validate_labels(checkpoint["labels"]))
    errors.extend(validate_frames(checkpoint["frames"]))
    errors.extend(validate_best_accuracy(checkpoint["best_accuracy"]))
    errors.extend(validate_dropout(checkpoint["dropout"]))
    errors.extend(validate_model_state(checkpoint["model_state"]))

    num_classes = len(checkpoint["labels"]) if isinstance(checkpoint["labels"], list) and checkpoint["labels"] else 1
    frames = checkpoint["frames"] if isinstance(checkpoint["frames"], int) and checkpoint["frames"] >= 2 else 64
    dropout = checkpoint["dropout"] if isinstance(checkpoint["dropout"], (int, float)) else 0.15
    errors.extend(validate_model_name_buildable(checkpoint["model_name"], num_classes, frames, dropout, checkpoint["ablation"]))

    return {"valid": len(errors) == 0, "errors": errors}


def scan_directory(root: str | Path) -> dict[str, dict]:
    root = Path(root)
    results = {}
    for path in sorted(root.rglob("*.pt")):
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            results[str(path)] = {"valid": False, "errors": [f"failed to load checkpoint: {error}"]}
            continue
        if not isinstance(checkpoint, dict):
            results[str(path)] = {"valid": False, "errors": [f"checkpoint is not a dict, got {type(checkpoint).__name__}"]}
            continue
        results[str(path)] = validate_checkpoint(checkpoint)
    return results


def print_summary(results: dict[str, dict]) -> None:
    valid_count = sum(1 for report in results.values() if report["valid"])
    print(f"{valid_count}/{len(results)} checkpoints valid")
    for path, report in results.items():
        if not report["valid"]:
            print(f"\n{path}")
            for error in report["errors"]:
                print(f"  - {error}")


def main():
    parser = argparse.ArgumentParser(description="Validate checkpoint files against the schema every check script depends on")
    parser.add_argument("--runs", required=True, help="directory to scan recursively for .pt files")
    parser.add_argument("--output", default="runs/checkpoint_schema_report.json")
    args = parser.parse_args()

    results = scan_directory(args.runs)
    print_summary(results)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull report written to {destination}")


if __name__ == "__main__":
    main()
