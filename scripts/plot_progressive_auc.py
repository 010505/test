from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROGRESSIVE = ROOT / "runs" / "progressive_aggregate" / "aggregate.json"
DRIFTING = ROOT / "runs" / "drifting_aggregate" / "aggregate.json"
STGCN_PREFIX = ROOT / "runs" / "stgcn_prefix_evaluation" / "metrics.json"
OUTPUT = ROOT / "deliverables" / "figures" / "progressive_auc"


def load_series():
    progressive = json.loads(PROGRESSIVE.read_text(encoding="utf-8"))
    drifting = json.loads(DRIFTING.read_text(encoding="utf-8"))["aggregate"]
    stgcn = json.loads(STGCN_PREFIX.read_text(encoding="utf-8"))
    return [
        ("Original ST-GCN", stgcn),
        ("Original AGCRN", progressive["00_full_sequence"]),
        ("AGCRN + GRU Evidence", progressive["03_gru_evidence"]),
        ("AGCRN + Four-step Diffusion", progressive["04_class_diffusion"]),
        ("AGCRN + One-step Distilled", drifting["08_one_step_distilled"]),
    ]


def main():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
        "axes.linewidth": 1.25,
        "lines.linewidth": 2.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    ratios = np.array([25, 50, 65, 80, 100], dtype=float)
    styles = [
        {"color": "#4D4D4D", "marker": "^", "linestyle": "-."},
        {"color": "#CC79A7", "marker": "P", "linestyle": "--"},
        {"color": "#0072B2", "marker": "o", "linestyle": "--"},
        {"color": "#E69F00", "marker": "s", "linestyle": ":"},
        {"color": "#009E73", "marker": "D", "linestyle": "-"},
    ]

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    series = load_series()
    all_lower_bounds = []
    for (name, row), style in zip(series, styles):
        ratio_rows = row["ratio_accuracy"]
        means = np.array([ratio_rows[f"{ratio / 100:.2f}"]["mean"] for ratio in ratios]) * 100
        stds = np.array([ratio_rows[f"{ratio / 100:.2f}"]["std"] for ratio in ratios]) * 100
        auc = row["prefix_auc"]["mean"] * 100
        all_lower_bounds.extend(means - stds)
        suffix = ", single checkpoint" if row.get("run_count") == 1 else ""
        ax.plot(
            ratios,
            means,
            label=f"{name}  (AUC = {auc:.2f}%{suffix})",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.9,
            **style,
        )
        if np.any(stds > 0):
            ax.fill_between(
                ratios,
                means - stds,
                means + stds,
                color=style["color"],
                alpha=0.14,
                linewidth=0,
            )

    ax.set_xlabel("Observation Ratio (%)", labelpad=10)
    ax.set_ylabel("Recognition Accuracy (%)", labelpad=10)
    ax.set_xticks(ratios)
    # Reserve a clean right-hand gutter for non-overlapping endpoint labels.
    ax.set_xlim(22, 112)
    lower_limit = min(50, 5 * np.floor((min(all_lower_bounds) - 2) / 5))
    ax.set_ylim(lower_limit, 85)
    ax.set_yticks(np.arange(lower_limit, 86, 5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=6, width=1.2)
    ax.legend(loc="lower right", frameon=False, handlelength=3.3)

    # Stagger close endpoint values and connect each label to its curve.
    endpoint_y = {
        "Original ST-GCN": 70.2,
        "Original AGCRN": 80.6,
        "AGCRN + GRU Evidence": 76.4,
        "AGCRN + Four-step Diffusion": 78.5,
        "AGCRN + One-step Distilled": 82.5,
    }
    for (name, row), style in zip(series, styles):
        final_value = row["ratio_accuracy"]["1.00"]["mean"] * 100
        ax.annotate(
            f"{final_value:.2f}%",
            xy=(100, final_value),
            xytext=(102.0, endpoint_y[name]),
            textcoords="data",
            color=style["color"],
            fontsize=14,
            va="center",
            fontweight="bold" if "One-step" in name else "normal",
            arrowprops={
                "arrowstyle": "-",
                "color": style["color"],
                "linewidth": 1.1,
                "shrinkA": 3,
                "shrinkB": 4,
            },
            clip_on=False,
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "progressive_accuracy_auc_comparison"
    fig.savefig(stem.with_suffix(".jpg"), dpi=300, bbox_inches="tight",
                facecolor="white", pil_kwargs={"quality": 96})
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(stem.with_suffix(".jpg"))


if __name__ == "__main__":
    main()
