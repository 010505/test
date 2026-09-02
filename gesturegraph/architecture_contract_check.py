from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import build_model

KNOWN_ARCHITECTURES = (
    "stgcn",
    "mlp",
    "spectral_pe_stgcn",
    "spectral_pe_qkv",
    "spectral_pe_qkv_stable",
    "gwnet_adaptive_support",
    "agcrn_factorized_adjacency",
    "velocity_agcrn",
    "gated_agcrn",
    "velocity_gated_agcrn",
)


def check_output_shape(model, num_classes: int, frames: int) -> list[str]:
    inputs = torch.randn(2, 3, frames, 22)
    output = model(inputs)
    if tuple(output.shape) != (2, num_classes):
        return [f"expected output shape (2, {num_classes}), got {tuple(output.shape)}"]
    return []


def check_deterministic_eval(model, frames: int) -> list[str]:
    model.eval()
    inputs = torch.randn(2, 3, frames, 22)
    with torch.no_grad():
        first = model(inputs)
        second = model(inputs)
    if not torch.allclose(first, second):
        return ["model is not deterministic in eval mode"]
    return []


def check_gradient_flow(model, num_classes: int, frames: int) -> list[str]:
    model.train()
    inputs = torch.randn(2, 3, frames, 22)
    targets = torch.randint(0, num_classes, (2,))
    loss = torch.nn.functional.cross_entropy(model(inputs), targets)
    loss.backward()
    errors = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            errors.append(f"parameter {name!r} received no gradient")
        elif parameter.grad.abs().sum().item() == 0:
            errors.append(f"parameter {name!r} received an all-zero gradient")
    return errors


def check_batch_independence(model, frames: int) -> list[str]:
    model.eval()
    torch.manual_seed(0)
    inputs = torch.randn(4, 3, frames, 22)
    with torch.no_grad():
        batched = model(inputs)
        single = model(inputs[2:3])
    if not torch.allclose(batched[2:3], single, atol=1e-5):
        return ["batching changed the prediction for a fixed sample - batchnorm or graph handling may be leaking across samples"]
    return []


def run_contract(model, num_classes: int, frames: int) -> list[str]:
    errors = []
    errors.extend(check_output_shape(model, num_classes, frames))
    errors.extend(check_deterministic_eval(model, frames))
    errors.extend(check_gradient_flow(model, num_classes, frames))
    errors.extend(check_batch_independence(model, frames))
    return errors


def check_architecture(name: str, num_classes: int = 14, frames: int = 64, dropout: float = 0.15) -> dict:
    try:
        model = build_model(name, num_classes, frames, dropout, "none")
    except ValueError:
        return {"name": name, "status": "pending", "errors": []}
    errors = run_contract(model, num_classes, frames)
    return {"name": name, "status": "pass" if not errors else "fail", "errors": errors}


def check_all_known_architectures(names: tuple[str, ...] = KNOWN_ARCHITECTURES, num_classes: int = 14, frames: int = 64, dropout: float = 0.15) -> list[dict]:
    return [check_architecture(name, num_classes, frames, dropout) for name in names]


def print_summary(results: list[dict]) -> None:
    for row in results:
        marker = {"pass": "PASS", "fail": "FAIL", "pending": "PENDING"}[row["status"]]
        print(f"{marker:8s} {row['name']}")
        for error in row["errors"]:
            print(f"           - {error}")


def main():
    parser = argparse.ArgumentParser(description="Contract check for current and planned model backbones")
    parser.add_argument("--num-classes", type=int, default=14)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--output", default="runs/architecture_contract_report.json")
    args = parser.parse_args()

    results = check_all_known_architectures(KNOWN_ARCHITECTURES, args.num_classes, args.frames, args.dropout)
    print_summary(results)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull report written to {destination}")


if __name__ == "__main__":
    main()
