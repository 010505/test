from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .drifting import ONE_STEP_EXPERIMENTS
from .drifting_benchmark import BASELINE_EXPERIMENTS


ALL_EXPERIMENTS = BASELINE_EXPERIMENTS + ONE_STEP_EXPERIMENTS


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate one-step drifting experiments")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", default="runs/drifting_aggregate")
    args = parser.parse_args()

    rows = {experiment: [] for experiment in ALL_EXPERIMENTS}
    for root_name in args.runs:
        root = Path(root_name)
        for experiment in ALL_EXPERIMENTS:
            path = root / experiment / "model.json"
            rows[experiment].append(json.loads(path.read_text(encoding="utf-8")))
    seeds = [row["seed"] for row in rows[ALL_EXPERIMENTS[0]]]
    if len(set(seeds)) != len(seeds):
        raise ValueError("runs must contain distinct seeds")
    for experiment in ALL_EXPERIMENTS:
        if [row["seed"] for row in rows[experiment]] != seeds:
            raise ValueError(f"seed ordering differs for {experiment}")

    aggregate = {}
    for experiment, experiments in rows.items():
        ratio_keys = list(experiments[0]["official_test"]["ratio_accuracy"])
        aggregate[experiment] = {
            "seeds": seeds,
            "parameters": experiments[0]["parameters"],
            "ratio_accuracy": {
                ratio: stats([
                    row["official_test"]["ratio_accuracy"][ratio] for row in experiments
                ]) for ratio in ratio_keys
            },
            "prefix_auc": stats([row["official_test"]["prefix_auc"] for row in experiments]),
            "full_accuracy": stats([
                row["official_test"]["full_accuracy"] for row in experiments
            ]),
            "decision_accuracy": stats([
                row["official_test_early_exit"]["accuracy"] for row in experiments
            ]),
            "average_decision_ratio": stats([
                row["official_test_early_exit"]["average_decision_ratio"]
                for row in experiments
            ]),
            "final_recovery_rate": stats([
                row["official_test_error_recovery"]["final_recovery_rate"]
                for row in experiments
            ]),
            "correct_retention": stats([
                row["official_test_error_recovery"]["initially_correct_final_retention"]
                for row in experiments
            ]),
            "online_mean_update_ms": stats([
                row["online_latency"]["mean_update_ms"] for row in experiments
            ]),
        }

    reference = aggregate["04_four_step_teacher"]
    for experiment in ALL_EXPERIMENTS[1:]:
        row = aggregate[experiment]
        row["auc_delta_vs_four_step_pp"] = stats([
            100.0 * (
                candidate["official_test"]["prefix_auc"]
                - baseline["official_test"]["prefix_auc"]
            )
            for candidate, baseline in zip(
                rows[experiment], rows["04_four_step_teacher"]
            )
        ])
        row["latency_reduction_vs_four_step"] = (
            1.0 - row["online_mean_update_ms"]["mean"]
            / reference["online_mean_update_ms"]["mean"]
        )

    paired_comparisons = {}
    for name, candidate_name, baseline_name in (
        ("direct_minus_truncation", "07_one_step_direct", "04_one_step_truncation"),
        ("distillation_minus_direct", "08_one_step_distilled", "07_one_step_direct"),
        ("drift_minus_distillation", "09_one_step_conditional_drift", "08_one_step_distilled"),
        ("drift_minus_four_step", "09_one_step_conditional_drift", "04_four_step_teacher"),
    ):
        paired_comparisons[name] = {}
        for metric, section, key in (
            ("prefix_auc_pp", "official_test", "prefix_auc"),
            ("full_accuracy_pp", "official_test", "full_accuracy"),
            ("decision_accuracy_pp", "official_test_early_exit", "accuracy"),
            ("final_recovery_pp", "official_test_error_recovery", "final_recovery_rate"),
        ):
            values = [
                100.0 * (candidate[section][key] - baseline[section][key])
                for candidate, baseline in zip(rows[candidate_name], rows[baseline_name])
            ]
            paired_comparisons[name][metric] = {
                "per_seed": values,
                **stats(values),
            }

    payload = {
        "protocol": {
            "epochs": rows["07_one_step_direct"][0]["epochs"],
            "seeds": seeds,
            "official_test_samples": 840,
            "student_initialization": rows["07_one_step_direct"][0][
                "training_objective"
            ]["initialized_from_teacher"],
        },
        "aggregate": aggregate,
        "paired_comparisons": paired_comparisons,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "aggregate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    names = {
        "04_four_step_teacher": "Four-step teacher",
        "04_one_step_truncation": "One-step truncation",
        "07_one_step_direct": "One-step direct training",
        "08_one_step_distilled": "One-step teacher distillation",
        "09_one_step_conditional_drift": "Distillation + conditional CatDrift",
    }
    lines = [
        "# Training-time drifting for one-step class inference",
        "",
        "Formal 40-epoch, three-seed comparison on the 840-sample official test set.",
        "The trainable students are independent checkpoints and use one reverse NFE at inference.",
        "",
        "Protocol: each student is initialized from its same-seed four-step teacher; the verified",
        "AGCRN encoder and BatchNorm statistics are frozen, leaving 22,366 trainable parameters",
        "in the one-step class head out of 1,010,376 total. AdamW uses lr=1e-4, effective batch",
        "size 32, cosine decay, 40 epochs, and validation Prefix AUC checkpoint selection with",
        "epoch 0 retained as a candidate. Loss weights are 0.5 denoising, 0.1 distillation, and",
        "0.05 conditional drifting. Drifting uses radii 0.2/0.5/1.0, strength 0.25, context",
        "weight 0.25, and a 64-entry memory bank per class/observation bucket.",
        "",
        "| Model | Prefix AUC | Full accuracy | Decision accuracy | ADR | Final recovery | Online/update | Latency reduction | Delta vs 4-step |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment in ALL_EXPERIMENTS:
        row = aggregate[experiment]
        delta = "--" if experiment == "04_four_step_teacher" else (
            f"{row['auc_delta_vs_four_step_pp']['mean']:+.2f} pp"
        )
        latency_reduction = "--" if experiment == "04_four_step_teacher" else (
            f"{row['latency_reduction_vs_four_step']:.1%}"
        )
        lines.append(
            f"| {names[experiment]}"
            f" | {row['prefix_auc']['mean']:.2%} +/- {row['prefix_auc']['std']:.2%}"
            f" | {row['full_accuracy']['mean']:.2%}"
            f" | {row['decision_accuracy']['mean']:.2%}"
            f" | {row['average_decision_ratio']['mean']:.3f}"
            f" | {row['final_recovery_rate']['mean']:.2%}"
            f" | {row['online_mean_update_ms']['mean']:.3f} ms"
            f" | {latency_reduction}"
            f" | {delta} |"
        )
    lines.extend([
        "",
        "Paired ablations (mean +/- sample standard deviation):",
        "",
        f"- Direct training vs truncation: {paired_comparisons['direct_minus_truncation']['prefix_auc_pp']['mean']:+.2f} +/- {paired_comparisons['direct_minus_truncation']['prefix_auc_pp']['std']:.2f} pp AUC.",
        f"- Distillation vs direct training: {paired_comparisons['distillation_minus_direct']['prefix_auc_pp']['mean']:+.2f} +/- {paired_comparisons['distillation_minus_direct']['prefix_auc_pp']['std']:.2f} pp AUC.",
        f"- Conditional CatDrift vs distillation: {paired_comparisons['drift_minus_distillation']['prefix_auc_pp']['mean']:+.2f} +/- {paired_comparisons['drift_minus_distillation']['prefix_auc_pp']['std']:.2f} pp AUC.",
        "",
        "Drifting implementation: four-step teacher posteriors are positives, detached one-step",
        "student posteriors are negatives, and the training-only memory bank is separated by",
        "true class and observation ratio. The normalized attraction-minus-repulsion field tilts",
        "student logits before softmax; the resulting frozen target is optimized by KL divergence.",
        "No teacher, bank, or iterative drift computation is used at inference.",
        "",
        "Primary references: [Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770),",
        "[official Drifting code](https://github.com/lambertae/drifting), and",
        "[Categorical Drifting Models](https://openreview.net/pdf?id=tEBiaXboS6).",
        "",
        "Reproduction command (replace SEED in both paths):",
        "",
        "```bash",
        "python -m gesturegraph.drifting_benchmark --teacher runs/progressive_seedSEED/04_class_diffusion/best.pt --output runs/drifting_seedSEED --epochs 40 --lr 0.0001 --init-from-teacher --freeze-encoder --device cuda",
        "```",
        "",
    ])
    cpu_latency_path = output / "cpu_latency.json"
    if cpu_latency_path.exists():
        cpu_latency = json.loads(cpu_latency_path.read_text(encoding="utf-8"))["aggregate"]
        lines.extend([
            "## CPU latency audit",
            "",
            "Single-thread, batch-one online inference; three repeats of 200 five-update sequences",
            "per checkpoint, reporting the per-seed median and then the three-seed mean.",
            "",
            "| Model | CPU online/update | Reduction vs 4-step |",
            "|---|---:|---:|",
        ])
        for experiment in ALL_EXPERIMENTS:
            row = cpu_latency[experiment]
            reduction = "--" if experiment == "04_four_step_teacher" else (
                f"{row['reduction_vs_four_step']:.1%}"
            )
            lines.append(
                f"| {names[experiment]} | {row['mean']:.3f} +/- {row['std']:.3f} ms"
                f" | {reduction} |"
            )
        lines.extend([
            "",
            "The conservative CPU reduction is smaller because the unchanged AGCRN encoder",
            "dominates total update time. All three trained students have identical one-step",
            "inference graphs, so their small latency differences are measurement noise.",
            "",
        ])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
