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
)
from gesturegraph.progressive_benchmark import (
    choose_device,
    decision_metrics,
    error_recovery_metrics,
    evaluate_prefixes,
    measure_online_latency,
    set_reproducible,
)


DEFAULT_EXPERIMENTS = ("02_causal_prefix", "03_gru_evidence", "04_class_diffusion")


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def common_initial_error_metrics(first, second, targets, ratios):
    first_correct = first.argmax(axis=-1) == targets[:, None]
    second_correct = second.argmax(axis=-1) == targets[:, None]
    common_wrong = ~first_correct[:, 0] & ~second_correct[:, 0]
    count = int(common_wrong.sum())

    def rate(mask):
        return float(mask[common_wrong].mean()) if count else None

    return {
        "common_initial_wrong_samples": count,
        "first_model_final_recovery": rate(first_correct[:, -1]),
        "second_model_final_recovery": rate(second_correct[:, -1]),
        "both_recover": rate(first_correct[:, -1] & second_correct[:, -1]),
        "first_only_recovers": rate(first_correct[:, -1] & ~second_correct[:, -1]),
        "second_only_recovers": rate(~first_correct[:, -1] & second_correct[:, -1]),
        "neither_recovers": rate(~first_correct[:, -1] & ~second_correct[:, -1]),
        "point_recovery": {
            f"{ratio:.2f}": {
                "first_model": rate(first_correct[:, update]),
                "second_model": rate(second_correct[:, update]),
            }
            for update, ratio in enumerate(ratios[1:], start=1)
        },
    }


def aggregate_rows(per_seed, experiments, ratios):
    aggregate = {}
    for experiment in experiments:
        rows = [row[experiment] for row in per_seed]
        aggregate[experiment] = {
            "initial_wrong_samples": stats([
                row["error_recovery"]["initial_wrong_samples"] for row in rows
            ]),
            "final_recovery_rate": stats([
                row["error_recovery"]["final_recovery_rate"] for row in rows
            ]),
            "initially_correct_final_retention": stats([
                row["error_recovery"]["initially_correct_final_retention"] for row in rows
            ]),
            "point_recovery": {
                f"{ratio:.2f}": stats([
                    row["error_recovery"]["point_recovery"][f"{ratio:.2f}"] for row in rows
                ])
                for ratio in ratios[1:]
            },
            "ever_recovered_by_ratio": {
                f"{ratio:.2f}": stats([
                    row["error_recovery"]["ever_recovered_by_ratio"][f"{ratio:.2f}"]
                    for row in rows
                ])
                for ratio in ratios[1:]
            },
            "stable_recovery_from_ratio": {
                f"{ratio:.2f}": stats([
                    row["error_recovery"]["stable_recovery_from_ratio"][f"{ratio:.2f}"]
                    for row in rows
                ])
                for ratio in ratios[1:]
            },
            "online_sequence_ms": stats([
                row["online_latency"]["sequence_ms"] for row in rows
            ]),
            "online_mean_update_ms": stats([
                row["online_latency"]["mean_update_ms"] for row in rows
            ]),
            "average_decision_updates": stats([
                row["early_exit"]["average_decision_updates"] for row in rows
            ]),
            "estimated_early_exit_compute_ms": stats([
                row["online_latency"]["mean_update_ms"]
                * row["early_exit"]["average_decision_updates"]
                for row in rows
            ]),
            "batched_throughput_ms_per_sample_update": stats([
                row["batched_throughput_ms_per_sample_update"] for row in rows
            ]),
        }

    paired = [row["gru_vs_diffusion_common_initial_errors"] for row in per_seed]
    aggregate["gru_vs_diffusion_common_initial_errors"] = {
        "common_initial_wrong_samples": stats([
            row["common_initial_wrong_samples"] for row in paired
        ]),
        "gru_final_recovery": stats([
            row["first_model_final_recovery"] for row in paired
        ]),
        "diffusion_final_recovery": stats([
            row["second_model_final_recovery"] for row in paired
        ]),
        "diffusion_minus_gru_pp": stats([
            100.0 * (row["second_model_final_recovery"] - row["first_model_final_recovery"])
            for row in paired
        ]),
        "gru_only_recovers": stats([row["first_only_recovers"] for row in paired]),
        "diffusion_only_recovers": stats([row["second_only_recovers"] for row in paired]),
    }
    causal_latency = aggregate["02_causal_prefix"]["online_mean_update_ms"]["mean"]
    gru_latency = aggregate["03_gru_evidence"]["online_mean_update_ms"]["mean"]
    diffusion_latency = aggregate["04_class_diffusion"]["online_mean_update_ms"]["mean"]
    aggregate["online_latency_ratios"] = {
        "gru_over_causal": gru_latency / causal_latency,
        "diffusion_over_causal": diffusion_latency / causal_latency,
        "diffusion_over_gru": diffusion_latency / gru_latency,
    }
    return aggregate


def write_report(output, payload, experiments, ratios):
    aggregate = payload["aggregate"]
    names = {
        "02_causal_prefix": "Causal prefix",
        "03_gru_evidence": "GRU evidence",
        "04_class_diffusion": "Class diffusion",
    }
    lines = [
        "# Early-error recovery and online-latency analysis",
        "",
        "All recovery rates are conditioned on an incorrect 25% prefix prediction. "
        "Results are mean +/- sample standard deviation over seeds 42, 43 and 44 on "
        "the 840-sample official test set.",
        "",
        "## Recovery after an incorrect 25% prediction",
        "",
        "| Model | Initial errors | Correct at 50% | Correct at 65% | Correct at 80% | Final recovery | Initially-correct retention |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment in experiments:
        row = aggregate[experiment]
        values = [row["point_recovery"][f"{ratio:.2f}"]["mean"] for ratio in ratios[1:-1]]
        lines.append(
            f"| {names[experiment]} | {row['initial_wrong_samples']['mean']:.1f} | "
            + " | ".join(f"{value:.2%}" for value in values)
            + f" | {row['final_recovery_rate']['mean']:.2%} +/- {row['final_recovery_rate']['std']:.2%}"
            + f" | {row['initially_correct_final_retention']['mean']:.2%} |"
        )
    paired = aggregate["gru_vs_diffusion_common_initial_errors"]
    lines += [
        "",
        "## Paired common-error subset",
        "",
        "For each seed, this subset contains exactly the official-test samples that both GRU "
        "and diffusion classify incorrectly at 25%, avoiding a comparison over different errors.",
        "",
        f"- Common initial errors: {paired['common_initial_wrong_samples']['mean']:.1f} +/- {paired['common_initial_wrong_samples']['std']:.1f} samples.",
        f"- GRU final recovery: {paired['gru_final_recovery']['mean']:.2%} +/- {paired['gru_final_recovery']['std']:.2%}.",
        f"- Diffusion final recovery: {paired['diffusion_final_recovery']['mean']:.2%} +/- {paired['diffusion_final_recovery']['std']:.2%}.",
        f"- Paired difference: {paired['diffusion_minus_gru_pp']['mean']:+.2f} +/- {paired['diffusion_minus_gru_pp']['std']:.2f} percentage points.",
        "",
        "## Latency",
        "",
        "The online benchmark uses batch size 1 and advances all five causal updates while "
        "preserving recurrent/posterior state. The throughput column retains the previous "
        "batch-8 offline metric and should not be interpreted as user-facing response time.",
        "",
        "| Model | Online mean/update | Online five updates | Avg. updates to decision | Estimated decision compute | Batch-8 throughput/sample/update |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for experiment in experiments:
        row = aggregate[experiment]
        lines.append(
            f"| {names[experiment]} | {row['online_mean_update_ms']['mean']:.3f} +/- {row['online_mean_update_ms']['std']:.3f} ms"
            f" | {row['online_sequence_ms']['mean']:.3f} ms"
            f" | {row['average_decision_updates']['mean']:.2f}"
            f" | {row['estimated_early_exit_compute_ms']['mean']:.3f} ms"
            f" | {row['batched_throughput_ms_per_sample_update']['mean']:.3f} ms |"
        )
    ratios_row = aggregate["online_latency_ratios"]
    lines += [
        "",
        f"Diffusion is {ratios_row['diffusion_over_gru']:.2f}x the GRU online update latency "
        f"and {ratios_row['diffusion_over_causal']:.2f}x the direct causal model latency.",
        "",
        "These observational recovery results test whether a model escapes early errors. "
        "They do not alone prove that Markov state structure is the causal mechanism.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze early-error recovery and online latency")
    parser.add_argument("--data", default="data/shrec17_ddnet_npz")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--experiments", nargs="+", default=list(DEFAULT_EXPERIMENTS))
    parser.add_argument("--output", default="runs/progressive_recovery")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--latency-iterations", type=int, default=20)
    args = parser.parse_args()

    device = choose_device(args.device)
    test_samples = load_raw_shrec17_npz(args.data, "test")
    per_seed = []
    ratios = None
    for run in args.runs:
        run = Path(run)
        seed_rows = {}
        probabilities_by_experiment = {}
        targets = None
        seed = None
        for experiment in args.experiments:
            checkpoint = torch.load(run / experiment / "best.pt", map_location=device, weights_only=True)
            seed = int(checkpoint["seed"])
            set_reproducible(seed)
            labels = list(checkpoint["labels"])
            ratios = tuple(float(value) for value in checkpoint["observation_ratios"])
            dataset = PrefixSequenceDataset(test_samples, labels, checkpoint["frames"], ratios)
            loader = DataLoader(dataset, batch_size=args.eval_batch_size)
            model = build_progressive_model(experiment, len(labels)).to(device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            evaluation = evaluate_prefixes(model, loader, device, ratios, collect=True)
            probabilities = evaluation.pop("probabilities")
            targets = evaluation.pop("targets")
            recovery = error_recovery_metrics(probabilities, targets, ratios)
            online_latency = measure_online_latency(
                model, loader, device, iterations=args.latency_iterations
            )
            previous_result = json.loads((run / experiment / "model.json").read_text(encoding="utf-8"))
            calibration = previous_result["early_exit_calibration"]
            early_exit = decision_metrics(
                probabilities,
                targets,
                ratios,
                calibration["confidence_threshold"],
                calibration["margin_threshold"],
            )
            seed_rows[experiment] = {
                "official_test": evaluation,
                "error_recovery": recovery,
                "online_latency": online_latency,
                "early_exit": early_exit,
                "batched_throughput_ms_per_sample_update": previous_result[
                    "latency_ms_per_sample_update"
                ],
            }
            probabilities_by_experiment[experiment] = probabilities
            print(
                f"seed {seed} {experiment}: final recovery "
                f"{recovery['final_recovery_rate']:.2%}, online "
                f"{online_latency['mean_update_ms']:.3f} ms/update"
            )
        seed_rows["seed"] = seed
        seed_rows["gru_vs_diffusion_common_initial_errors"] = common_initial_error_metrics(
            probabilities_by_experiment["03_gru_evidence"],
            probabilities_by_experiment["04_class_diffusion"],
            targets,
            ratios,
        )
        per_seed.append(seed_rows)

    payload = {
        "protocol": {
            "official_test_samples": len(test_samples),
            "seeds": [row["seed"] for row in per_seed],
            "observation_ratios": ratios,
            "device": str(device),
            "latency_batch_size": 1,
        },
        "per_seed": per_seed,
        "aggregate": aggregate_rows(per_seed, args.experiments, ratios),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(output, payload, args.experiments, ratios)
    print(f"Wrote {output / 'analysis.json'} and {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
