from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from gesturegraph.progressive import (
    GatedClassDiffusionModel,
    PrefixSequenceDataset,
    build_progressive_model,
    load_raw_shrec17_npz,
)
from gesturegraph.progressive_benchmark import (
    choose_device,
    decision_metrics,
    error_recovery_metrics,
    evaluate_prefixes,
    measure_online_latency,
)


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def evaluate_gated(model, loader, device, ratios):
    model.eval()
    probabilities_all, evidence_all, gates_all, targets_all = [], [], [], []
    with torch.no_grad():
        for views, lengths, progress, targets in loader:
            views = views.to(device)
            lengths = lengths.to(device)
            progress = progress.to(device)
            outputs, diagnostics = model(
                views, lengths, progress, return_diagnostics=True
            )
            probabilities_all.append(outputs.exp().cpu())
            evidence_all.append(diagnostics["evidence"].cpu())
            gates_all.append(diagnostics["gates"].cpu())
            targets_all.append(targets)
    probabilities = torch.cat(probabilities_all).numpy()
    evidence = torch.cat(evidence_all).numpy()
    gates = torch.cat(gates_all).numpy()
    targets = torch.cat(targets_all).numpy()
    predictions = probabilities.argmax(axis=-1)
    accuracies = (predictions == targets[:, None]).mean(axis=0)
    return {
        "probabilities": probabilities,
        "evidence": evidence,
        "gates": gates,
        "targets": targets,
        "ratio_accuracy": {
            f"{ratio:.2f}": float(value) for ratio, value in zip(ratios, accuracies)
        },
        "prefix_auc": float(np.trapezoid(accuracies, ratios) / (ratios[-1] - ratios[0])),
        "full_accuracy": float(accuracies[-1]),
    }


def gate_diagnostics(probabilities, evidence, gates, targets, ratios):
    predictions = probabilities.argmax(axis=-1)
    evidence_predictions = evidence.argmax(axis=-1)
    updates = {}
    for gate_index, update in enumerate(range(1, len(ratios))):
        values = gates[:, gate_index]
        previous_correct = predictions[:, update - 1] == targets
        agreement = predictions[:, update - 1] == evidence_predictions[:, update]

        def masked_mean(mask):
            return float(values[mask].mean()) if np.any(mask) else None

        updates[f"{ratios[update]:.2f}"] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "q10": float(np.quantile(values, 0.10)),
            "median": float(np.quantile(values, 0.50)),
            "q90": float(np.quantile(values, 0.90)),
            "when_previous_correct": masked_mean(previous_correct),
            "when_previous_wrong": masked_mean(~previous_correct),
            "when_evidence_agrees": masked_mean(agreement),
            "when_evidence_disagrees": masked_mean(~agreement),
        }
    return updates


def common_initial_errors(baseline, gated, targets):
    baseline_correct = baseline.argmax(axis=-1) == targets[:, None]
    gated_correct = gated.argmax(axis=-1) == targets[:, None]
    common = ~baseline_correct[:, 0] & ~gated_correct[:, 0]
    count = int(common.sum())

    def rate(values):
        return float(values[common].mean()) if count else None

    return {
        "samples": count,
        "baseline_final_recovery": rate(baseline_correct[:, -1]),
        "gated_final_recovery": rate(gated_correct[:, -1]),
        "baseline_only_recovers": rate(baseline_correct[:, -1] & ~gated_correct[:, -1]),
        "gated_only_recovers": rate(~baseline_correct[:, -1] & gated_correct[:, -1]),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare fixed and learned-gated class diffusion")
    parser.add_argument("--data", default="data/shrec17_ddnet_npz")
    parser.add_argument("--baseline-runs", nargs="+", required=True)
    parser.add_argument("--gated-runs", nargs="+", required=True)
    parser.add_argument(
        "--candidate-experiment",
        default="05_gated_class_diffusion",
        choices=["05_gated_class_diffusion", "06_reliability_gated_diffusion"],
    )
    parser.add_argument("--output", default="runs/gated_diffusion_aggregate")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--latency-iterations", type=int, default=50)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()
    if len(args.baseline_runs) != len(args.gated_runs):
        raise ValueError("baseline and gated run lists must have equal lengths")

    device = choose_device(args.device)
    test_samples = load_raw_shrec17_npz(args.data, "test")
    per_seed = []
    for baseline_name, gated_name in zip(args.baseline_runs, args.gated_runs):
        baseline_root, gated_root = Path(baseline_name), Path(gated_name)
        baseline_checkpoint = torch.load(
            baseline_root / "04_class_diffusion" / "best.pt",
            map_location=device,
            weights_only=True,
        )
        gated_checkpoint = torch.load(
            gated_root / args.candidate_experiment / "best.pt",
            map_location=device,
            weights_only=True,
        )
        seed = int(gated_checkpoint["seed"])
        if seed != int(baseline_checkpoint["seed"]):
            raise ValueError("paired runs must use the same seed")
        labels = list(gated_checkpoint["labels"])
        ratios = tuple(float(value) for value in gated_checkpoint["observation_ratios"])
        loader = DataLoader(
            PrefixSequenceDataset(test_samples, labels, gated_checkpoint["frames"], ratios),
            batch_size=args.eval_batch_size,
        )

        baseline_model = build_progressive_model("04_class_diffusion", len(labels)).to(device)
        baseline_model.load_state_dict(baseline_checkpoint["model_state"])
        baseline_eval = evaluate_prefixes(
            baseline_model, loader, device, ratios, collect=True
        )
        baseline_probabilities = baseline_eval.pop("probabilities")
        targets = baseline_eval.pop("targets")

        gated_model = build_progressive_model(args.candidate_experiment, len(labels)).to(device)
        gated_model.load_state_dict(gated_checkpoint["model_state"])
        if not isinstance(gated_model, GatedClassDiffusionModel):
            raise TypeError("expected GatedClassDiffusionModel")
        gated_eval = evaluate_gated(gated_model, loader, device, ratios)
        gated_probabilities = gated_eval.pop("probabilities")
        evidence = gated_eval.pop("evidence")
        gates = gated_eval.pop("gates")
        gated_targets = gated_eval.pop("targets")
        if not np.array_equal(targets, gated_targets):
            raise ValueError("paired predictions are not aligned")

        baseline_result = json.loads(
            (baseline_root / "04_class_diffusion" / "model.json").read_text(encoding="utf-8")
        )
        gated_result = json.loads(
            (gated_root / args.candidate_experiment / "model.json").read_text(encoding="utf-8")
        )
        baseline_calibration = baseline_result["early_exit_calibration"]
        gated_calibration = gated_result["early_exit_calibration"]
        baseline_decision = decision_metrics(
            baseline_probabilities, targets, ratios,
            baseline_calibration["confidence_threshold"],
            baseline_calibration["margin_threshold"],
        )
        gated_decision = decision_metrics(
            gated_probabilities, targets, ratios,
            gated_calibration["confidence_threshold"],
            gated_calibration["margin_threshold"],
        )
        row = {
            "seed": seed,
            "baseline": {
                "official_test": baseline_eval,
                "early_exit": baseline_decision,
                "recovery": error_recovery_metrics(
                    baseline_probabilities, targets, ratios
                ),
                "online_latency": measure_online_latency(
                    baseline_model, loader, device, args.latency_iterations
                ),
            },
            "gated": {
                "official_test": gated_eval,
                "early_exit": gated_decision,
                "recovery": error_recovery_metrics(
                    gated_probabilities, targets, ratios
                ),
                "online_latency": measure_online_latency(
                    gated_model, loader, device, args.latency_iterations
                ),
                "gate_diagnostics": gate_diagnostics(
                    gated_probabilities, evidence, gates, targets, ratios
                ),
            },
            "common_initial_errors": common_initial_errors(
                baseline_probabilities, gated_probabilities, targets
            ),
        }
        per_seed.append(row)
        print(
            f"seed {seed}: baseline AUC {baseline_eval['prefix_auc']:.2%}, "
            f"gated {gated_eval['prefix_auc']:.2%}"
        )

    def model_aggregate(name):
        rows = [row[name] for row in per_seed]
        return {
            "prefix_auc": stats([row["official_test"]["prefix_auc"] for row in rows]),
            "full_accuracy": stats([row["official_test"]["full_accuracy"] for row in rows]),
            "decision_accuracy": stats([row["early_exit"]["accuracy"] for row in rows]),
            "average_decision_ratio": stats([
                row["early_exit"]["average_decision_ratio"] for row in rows
            ]),
            "final_recovery_rate": stats([
                row["recovery"]["final_recovery_rate"] for row in rows
            ]),
            "correct_retention": stats([
                row["recovery"]["initially_correct_final_retention"] for row in rows
            ]),
            "online_mean_update_ms": stats([
                row["online_latency"]["mean_update_ms"] for row in rows
            ]),
        }

    aggregate = {
        "baseline": model_aggregate("baseline"),
        "gated": model_aggregate("gated"),
    }
    for metric, path in {
        "prefix_auc_pp": ("official_test", "prefix_auc"),
        "full_accuracy_pp": ("official_test", "full_accuracy"),
        "decision_accuracy_pp": ("early_exit", "accuracy"),
        "average_decision_ratio_pp": ("early_exit", "average_decision_ratio"),
        "final_recovery_pp": ("recovery", "final_recovery_rate"),
        "correct_retention_pp": ("recovery", "initially_correct_final_retention"),
    }.items():
        differences = [
            100.0 * (row["gated"][path[0]][path[1]] - row["baseline"][path[0]][path[1]])
            for row in per_seed
        ]
        aggregate[metric] = {"per_seed": differences, **stats(differences)}
    common_rows = [row["common_initial_errors"] for row in per_seed]
    aggregate["common_initial_errors"] = {
        "samples": stats([row["samples"] for row in common_rows]),
        "baseline_final_recovery": stats([
            row["baseline_final_recovery"] for row in common_rows
        ]),
        "gated_final_recovery": stats([
            row["gated_final_recovery"] for row in common_rows
        ]),
        "gated_minus_baseline_pp": stats([
            100.0 * (row["gated_final_recovery"] - row["baseline_final_recovery"])
            for row in common_rows
        ]),
    }
    aggregate["gate_means"] = {
        f"{ratio:.2f}": stats([
            row["gated"]["gate_diagnostics"][f"{ratio:.2f}"]["mean"]
            for row in per_seed
        ])
        for ratio in ratios[1:]
    }

    payload = {
        "protocol": {
            "candidate_experiment": args.candidate_experiment,
            "seeds": [row["seed"] for row in per_seed],
            "epochs": int(gated_checkpoint["epochs"]),
            "official_test_samples": len(test_samples),
            "ratios": ratios,
            "device": str(device),
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    b, g, common = aggregate["baseline"], aggregate["gated"], aggregate["common_initial_errors"]
    lines = [
        f"# {args.candidate_experiment} experiment",
        "",
        "Formal 40-epoch, three-seed comparison on the 840-sample official test set.",
        "",
        "| Model | Prefix AUC | Full accuracy | Decision accuracy | ADR | Final recovery | Correct retention | Online/update |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Fixed gamma=0.50 | {b['prefix_auc']['mean']:.2%} +/- {b['prefix_auc']['std']:.2%} | {b['full_accuracy']['mean']:.2%} | {b['decision_accuracy']['mean']:.2%} | {b['average_decision_ratio']['mean']:.3f} | {b['final_recovery_rate']['mean']:.2%} | {b['correct_retention']['mean']:.2%} | {b['online_mean_update_ms']['mean']:.3f} ms |",
        f"| Learned gate | {g['prefix_auc']['mean']:.2%} +/- {g['prefix_auc']['std']:.2%} | {g['full_accuracy']['mean']:.2%} | {g['decision_accuracy']['mean']:.2%} | {g['average_decision_ratio']['mean']:.3f} | {g['final_recovery_rate']['mean']:.2%} | {g['correct_retention']['mean']:.2%} | {g['online_mean_update_ms']['mean']:.3f} ms |",
        "",
        f"Paired AUC difference: {aggregate['prefix_auc_pp']['mean']:+.2f} +/- {aggregate['prefix_auc_pp']['std']:.2f} pp.",
        f"Common-initial-error final recovery: baseline {common['baseline_final_recovery']['mean']:.2%}, gated {common['gated_final_recovery']['mean']:.2%}, difference {common['gated_minus_baseline_pp']['mean']:+.2f} +/- {common['gated_minus_baseline_pp']['std']:.2f} pp.",
        "",
        "Mean learned inheritance gates (higher means more trust in the previous posterior):",
        "",
    ]
    lines.extend(
        f"- {ratio}: {values['mean']:.3f} +/- {values['std']:.3f}"
        for ratio, values in aggregate["gate_means"].items()
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output / 'analysis.json'} and {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
