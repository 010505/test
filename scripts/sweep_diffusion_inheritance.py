from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gesturegraph.progressive import (
    PrefixSequenceDataset,
    build_progressive_model,
    load_raw_shrec17_npz,
    stratified_raw_split,
)
from gesturegraph.progressive_benchmark import (
    calibrate_early_exit,
    choose_device,
    decision_metrics,
    error_recovery_metrics,
    evaluate_prefixes,
    set_reproducible,
)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Inference-only sweep of diffusion inheritance")
    parser.add_argument("--data", default="data/shrec17_ddnet_npz")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--values", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--output", default="runs/diffusion_inheritance_sweep")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    if any(value < 0.0 or value > 1.0 for value in args.values):
        raise ValueError("inheritance values must be in [0, 1]")
    device = choose_device(args.device)
    official_train = load_raw_shrec17_npz(args.data, "train")
    official_test = load_raw_shrec17_npz(args.data, "test")
    rows = {f"{value:.2f}": [] for value in args.values}

    for run_name in args.runs:
        run = Path(run_name)
        checkpoint = torch.load(
            run / "04_class_diffusion" / "best.pt", map_location=device, weights_only=True
        )
        seed = int(checkpoint["seed"])
        labels = list(checkpoint["labels"])
        ratios = tuple(float(value) for value in checkpoint["observation_ratios"])
        _, validation_samples = stratified_raw_split(official_train, 0.2, seed)
        validation_loader = DataLoader(
            PrefixSequenceDataset(validation_samples, labels, checkpoint["frames"], ratios),
            batch_size=args.eval_batch_size,
        )
        test_loader = DataLoader(
            PrefixSequenceDataset(official_test, labels, checkpoint["frames"], ratios),
            batch_size=args.eval_batch_size,
        )
        for inheritance in args.values:
            set_reproducible(seed)
            model = build_progressive_model("04_class_diffusion", len(labels)).to(device)
            model.load_state_dict(checkpoint["model_state"])
            model.inheritance = float(inheritance)
            validation = evaluate_prefixes(model, validation_loader, device, ratios, collect=True)
            calibration = calibrate_early_exit(validation, ratios)
            test = evaluate_prefixes(model, test_loader, device, ratios, collect=True)
            early_exit = decision_metrics(
                test["probabilities"], test["targets"], ratios,
                calibration["confidence_threshold"], calibration["margin_threshold"],
            )
            recovery = error_recovery_metrics(test["probabilities"], test["targets"], ratios)
            row = {
                "seed": seed,
                "inheritance": float(inheritance),
                "validation_prefix_auc": validation["prefix_auc"],
                "official_prefix_auc": test["prefix_auc"],
                "official_full_accuracy": test["full_accuracy"],
                "official_early_exit": early_exit,
                "official_error_recovery": recovery,
                "calibration": calibration,
            }
            rows[f"{inheritance:.2f}"].append(row)
            print(
                f"seed {seed} gamma {inheritance:.2f} | AUC {test['prefix_auc']:.2%} "
                f"| full {test['full_accuracy']:.2%} | ADR {early_exit['average_decision_ratio']:.3f} "
                f"| recovery {recovery['final_recovery_rate']:.2%}"
            )

    aggregate = {}
    for key, experiments in rows.items():
        aggregate[key] = {
            "inheritance": float(key),
            "prefix_auc": stats([row["official_prefix_auc"] for row in experiments]),
            "full_accuracy": stats([row["official_full_accuracy"] for row in experiments]),
            "decision_accuracy": stats([
                row["official_early_exit"]["accuracy"] for row in experiments
            ]),
            "average_decision_ratio": stats([
                row["official_early_exit"]["average_decision_ratio"] for row in experiments
            ]),
            "final_recovery_rate": stats([
                row["official_error_recovery"]["final_recovery_rate"] for row in experiments
            ]),
            "initially_correct_final_retention": stats([
                row["official_error_recovery"]["initially_correct_final_retention"]
                for row in experiments
            ]),
        }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "type": "inference_only_post_training_sensitivity_sweep",
            "checkpoint_training_inheritance": 0.5,
            "values": args.values,
            "seeds": [int(Path(run).name.replace("progressive_seed", "")) for run in args.runs],
            "official_test_samples": len(official_test),
            "device": str(device),
        },
        "per_seed": rows,
        "aggregate": aggregate,
    }
    (output / "analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Diffusion inheritance sensitivity sweep",
        "",
        "This is an inference-only sensitivity analysis over checkpoints trained with gamma=0.50; "
        "it is a screening experiment, not a replacement for independently trained models.",
        "",
        "| Gamma | Prefix AUC | Full accuracy | Decision accuracy | ADR | Final recovery | Correct retention |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in rows:
        row = aggregate[key]
        lines.append(
            f"| {float(key):.2f} | {row['prefix_auc']['mean']:.2%} +/- {row['prefix_auc']['std']:.2%}"
            f" | {row['full_accuracy']['mean']:.2%}"
            f" | {row['decision_accuracy']['mean']:.2%}"
            f" | {row['average_decision_ratio']['mean']:.3f}"
            f" | {row['final_recovery_rate']['mean']:.2%}"
            f" | {row['initially_correct_final_retention']['mean']:.2%} |"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output / 'analysis.json'} and {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
