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
    measure_online_latency,
    set_reproducible,
)


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Inference-only reverse-step sweep")
    parser.add_argument("--data", default="data/shrec17_ddnet_npz")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--output", default="runs/diffusion_step_sweep")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--latency-iterations", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()
    if any(step < 1 or step > 4 for step in args.steps):
        raise ValueError("steps must be in [1, 4]")

    device = choose_device(args.device)
    official_train = load_raw_shrec17_npz(args.data, "train")
    official_test = load_raw_shrec17_npz(args.data, "test")
    rows = {str(step): [] for step in args.steps}
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
        for reverse_steps in args.steps:
            set_reproducible(seed)
            model = build_progressive_model("04_class_diffusion", len(labels)).to(device)
            model.load_state_dict(checkpoint["model_state"])
            model.inference_steps = reverse_steps
            validation = evaluate_prefixes(model, validation_loader, device, ratios, collect=True)
            calibration = calibrate_early_exit(validation, ratios)
            test = evaluate_prefixes(model, test_loader, device, ratios, collect=True)
            early_exit = decision_metrics(
                test["probabilities"], test["targets"], ratios,
                calibration["confidence_threshold"], calibration["margin_threshold"],
            )
            recovery = error_recovery_metrics(test["probabilities"], test["targets"], ratios)
            latency = measure_online_latency(
                model, test_loader, device, iterations=args.latency_iterations
            )
            rows[str(reverse_steps)].append({
                "seed": seed,
                "reverse_steps": reverse_steps,
                "validation_prefix_auc": validation["prefix_auc"],
                "official_prefix_auc": test["prefix_auc"],
                "official_full_accuracy": test["full_accuracy"],
                "official_early_exit": early_exit,
                "official_error_recovery": recovery,
                "online_latency": latency,
                "calibration": calibration,
            })
            print(
                f"seed {seed} steps {reverse_steps} | AUC {test['prefix_auc']:.2%} "
                f"| full {test['full_accuracy']:.2%} | {latency['mean_update_ms']:.3f} ms/update"
            )

    aggregate = {}
    for key, experiments in rows.items():
        aggregate[key] = {
            "reverse_steps": int(key),
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
            "correct_retention": stats([
                row["official_error_recovery"]["initially_correct_final_retention"]
                for row in experiments
            ]),
            "online_mean_update_ms": stats([
                row["online_latency"]["mean_update_ms"] for row in experiments
            ]),
        }
    full_latency = aggregate["4"]["online_mean_update_ms"]["mean"]
    for row in aggregate.values():
        row["latency_reduction_vs_four_steps"] = (
            1.0 - row["online_mean_update_ms"]["mean"] / full_latency
        )

    payload = {
        "protocol": {
            "type": "inference_only_reverse_step_truncation",
            "seeds": [int(Path(run).name.replace("progressive_seed", "")) for run in args.runs],
            "official_test_samples": len(official_test),
            "device": str(device),
        },
        "per_seed": rows,
        "aggregate": aggregate,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Class-diffusion reverse-step sweep",
        "",
        "Each truncated path returns the denoiser's exact clean-class marginal at its final "
        "executed reverse step. No checkpoint is retrained or modified.",
        "",
        "| Reverse steps | Prefix AUC | Full accuracy | Decision accuracy | ADR | Final recovery | Online/update | Latency reduction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in rows:
        row = aggregate[key]
        lines.append(
            f"| {key} | {row['prefix_auc']['mean']:.2%} +/- {row['prefix_auc']['std']:.2%}"
            f" | {row['full_accuracy']['mean']:.2%}"
            f" | {row['decision_accuracy']['mean']:.2%}"
            f" | {row['average_decision_ratio']['mean']:.3f}"
            f" | {row['final_recovery_rate']['mean']:.2%}"
            f" | {row['online_mean_update_ms']['mean']:.3f} ms"
            f" | {row['latency_reduction_vs_four_steps']:.1%} |"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output / 'analysis.json'} and {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
