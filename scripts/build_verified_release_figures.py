from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEEDS = (42, 43, 44)


def _summary(run_root: Path) -> dict[str, dict]:
    rows = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    return {row["experiment"]: row for row in rows}


def _stats(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": [float(value) for value in array],
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
    }


def collect_backbones(root: Path) -> dict[str, dict[str, object]]:
    legacy = [_summary(root / f"backbones_seed{seed}") for seed in SEEDS]
    pure = [_summary(root / f"pure_qkv_seed{seed}") for seed in SEEDS]
    concat = [_summary(root / f"concat_seed{seed}") for seed in SEEDS]
    gated = [_summary(root / f"gated_seed{seed}") for seed in SEEDS]
    definitions = [
        ("ST-GCN", legacy, "00_stgcn_control"),
        ("Attention (X+SE, pure QKV)", pure, "01_semantic_pure_qkv"),
        ("Graph WaveNet (gate)", gated, "02_gwnet_gated_control"),
        ("Graph WaveNet (concat)", concat, "02_stem_gwnet_control"),
        ("AGCRN (gate)", gated, "04_agcrn_dynamic_control"),
        ("AGCRN (concat)", concat, "04_stem_agcrn_control"),
    ]
    return {
        name: _stats([run[experiment]["official_test_accuracy"] for run in runs])
        for name, runs, experiment in definitions
    }


def plot_backbones(results: dict[str, dict[str, object]], output: Path) -> None:
    labels = list(results)
    means = np.asarray([results[label]["mean"] for label in labels]) * 100.0
    stds = np.asarray([results[label]["std"] for label in labels]) * 100.0
    colors = ["#5B6573", "#009E73", "#56B4E9", "#0072B2", "#E69F00", "#D55E00"]
    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, edgecolor="#222222", linewidth=1.1)
    baseline = means[0]
    ax.axhline(baseline, color="#5B6573", linewidth=1.5, linestyle="--", alpha=0.8)
    for index, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        delta = mean - baseline
        delta_text = "baseline" if index == 0 else f"{delta:+.2f} pp"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.7,
            f"{mean:.2f}% +/- {std:.2f}\n{delta_text}",
            ha="center",
            va="bottom",
            fontsize=16,
        )
    ax.set_ylabel("Official Test Accuracy (%)", fontsize=19)
    ax.set_xticks(x, [label.replace(" (", "\n(") for label in labels], fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(60, max(86, float(np.max(means + stds)) + 6))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, facecolor="white", pil_kwargs={"quality": 96})
    plt.close(fig)


def plot_progressive(progressive_path: Path, drifting_path: Path, output: Path) -> dict[str, dict]:
    progressive = json.loads(progressive_path.read_text(encoding="utf-8"))
    drifting = json.loads(drifting_path.read_text(encoding="utf-8"))["aggregate"]
    series = [
        ("Original AGCRN", progressive["00_full_sequence"], "#CC79A7", "P", "--"),
        ("AGCRN + GRU", progressive["03_gru_evidence"], "#0072B2", "o", "--"),
        ("AGCRN + 4-step diffusion", progressive["04_class_diffusion"], "#E69F00", "s", ":"),
        ("AGCRN + 1-step distilled", drifting["08_one_step_distilled"], "#009E73", "D", "-"),
    ]
    ratios = np.asarray([25, 50, 65, 80, 100], dtype=np.float64)
    exported: dict[str, dict] = {}
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    for name, row, color, marker, linestyle in series:
        means = np.asarray([row["ratio_accuracy"][f"{ratio / 100:.2f}"]["mean"] for ratio in ratios]) * 100
        stds = np.asarray([row["ratio_accuracy"][f"{ratio / 100:.2f}"]["std"] for ratio in ratios]) * 100
        auc = float(row["prefix_auc"]["mean"] * 100)
        exported[name] = {"accuracy": means.tolist(), "std": stds.tolist(), "auc": auc}
        ax.plot(ratios, means, color=color, marker=marker, linestyle=linestyle,
                linewidth=2.8, markersize=9, label=f"{name} (AUC={auc:.2f}%)")
        ax.fill_between(ratios, means - stds, means + stds, color=color, alpha=0.13, linewidth=0)
    ax.set_xlabel("Observation Ratio (%)", fontsize=19)
    ax.set_ylabel("Recognition Accuracy (%)", fontsize=19)
    ax.set_xticks(ratios)
    ax.tick_params(labelsize=16)
    ax.set_ylim(50, 85)
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, facecolor="white", pil_kwargs={"quality": 96})
    plt.close(fig)
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description="Build figures from verified three-seed release results")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--progressive", type=Path)
    parser.add_argument("--drifting", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backbone_results = collect_backbones(args.root)
    args.output.mkdir(parents=True, exist_ok=True)
    plot_backbones(backbone_results, args.output / "verified_backbone_comparison.jpg")
    report: dict[str, object] = {
        "protocol": {
            "seeds": list(SEEDS),
            "epochs": 40,
            "dataset": "SHREC'17 official 14-class split, DD-Net numeric NPZ",
            "test_samples": 840,
        },
        "backbone_results": backbone_results,
    }
    if args.progressive and args.drifting:
        report["progressive_results"] = plot_progressive(
            args.progressive,
            args.drifting,
            args.output / "verified_progressive_auc.jpg",
        )
    (args.output / "verified_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
