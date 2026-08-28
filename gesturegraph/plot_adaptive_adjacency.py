from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gesturegraph-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


NODE_LABELS = [
    "wrist", "palm",
    "thumb-1", "thumb-2", "thumb-3", "thumb-tip",
    "index-1", "index-2", "index-3", "index-tip",
    "middle-1", "middle-2", "middle-3", "middle-tip",
    "ring-1", "ring-2", "ring-3", "ring-tip",
    "little-1", "little-2", "little-3", "little-tip",
]


def validate_matrix(name: str, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (22, 22):
        raise ValueError(f"{name} must have shape (22, 22), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def add_group_boundaries(axis, colour: str = "white") -> None:
    for boundary in (1.5, 5.5, 9.5, 13.5, 17.5):
        axis.axhline(boundary, color=colour, linewidth=0.35, alpha=0.85)
        axis.axvline(boundary, color=colour, linewidth=0.35, alpha=0.85)


def matrix_panel(figure, axis, matrix, title, cmap, vmin, vmax, colourbar_label, boundary_colour="white"):
    ticks = np.arange(22)
    edges = np.arange(23) - 0.5
    image = axis.pcolor(
        edges, edges, matrix, cmap=cmap, vmin=vmin, vmax=vmax,
        linewidth=0, rasterized=False,
    )
    axis.set_title(title)
    axis.set_xlabel("Target joint index")
    axis.set_ylabel("Source joint index")
    axis.set_xlim(-0.5, 21.5)
    axis.set_ylim(21.5, -0.5)
    axis.set_aspect("equal")
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_xticklabels(ticks, rotation=90)
    axis.set_yticklabels(ticks)
    axis.tick_params(length=1.5, pad=1)
    add_group_boundaries(axis, boundary_colour)
    colour_bar = figure.colorbar(image, ax=axis, shrink=0.76, pad=0.025)
    if colour_bar.solids is not None:
        colour_bar.solids.set_rasterized(False)
    colour_bar.set_label(colourbar_label)
    colour_bar.ax.tick_params(labelsize=6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot fixed and learned Graph WaveNet supports")
    parser.add_argument("--input", required=True, help="adjacency_matrices.npz from experiment 3")
    parser.add_argument("--output", required=True, help="Output basename without extension")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    archive = np.load(args.input)
    physical = validate_matrix("physical", archive["physical"])
    adaptive = validate_matrix("adaptive_support", archive["adaptive_support"])
    if np.any(adaptive < 0) or not np.allclose(adaptive.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("adaptive_support must be non-negative and row-stochastic")

    uniform = np.full_like(adaptive, 1.0 / adaptive.shape[1])
    adaptive_deviation = adaptive - uniform
    support_difference = adaptive - physical
    shared_max = float(max(physical.max(), adaptive.max()))
    deviation_limit = float(max(np.abs(adaptive_deviation).max(), 1e-8))
    difference_limit = float(max(np.abs(support_difference).max(), 1e-8))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })
    figure, axes = plt.subplots(1, 4, figsize=(13.4, 3.75), constrained_layout=True)
    matrix_panel(
        figure, axes[0], physical, "(a) Fixed physical support $A_{phys}$",
        "viridis", 0.0, shared_max, "Support weight",
    )
    matrix_panel(
        figure, axes[1], adaptive, "(b) Learned support $A_{adp}$\n(shared scale with a)",
        "viridis", 0.0, shared_max, "Support weight",
    )
    matrix_panel(
        figure, axes[2], adaptive_deviation,
        "(c) Learned structure around uniform $1/22$",
        "RdBu_r", -deviation_limit, deviation_limit, "Adaptive weight - 1/22", "black",
    )
    matrix_panel(
        figure, axes[3], support_difference,
        "(d) Descriptive difference $A_{adp}-A_{phys}$",
        "RdBu_r", -difference_limit, difference_limit, "Support-weight difference", "black",
    )
    figure.suptitle("GestureGraph experiment 3: fixed and learned Graph WaveNet supports", fontsize=10)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    preview_path = output.with_name(output.name + "_preview").with_suffix(".png")
    figure.savefig(preview_path, dpi=150, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    with Image.open(output.with_suffix(".png")) as image:
        ImageOps.grayscale(image).save(output.with_name(output.name + "_grayscale").with_suffix(".png"))

    physical_mask = physical > 0
    connections = []
    nonphysical_connections = []
    for source in range(22):
        for target in range(22):
            row = {
                "source": source,
                "source_name": NODE_LABELS[source],
                "target": target,
                "target_name": NODE_LABELS[target],
                "adaptive_weight": float(adaptive[source, target]),
                "is_physical_or_self": bool(physical_mask[source, target]),
            }
            connections.append(row)
            if not physical_mask[source, target]:
                nonphysical_connections.append(row)
    connections.sort(key=lambda row: row["adaptive_weight"], reverse=True)
    nonphysical_connections.sort(key=lambda row: row["adaptive_weight"], reverse=True)

    row_entropy = -(adaptive * np.log(np.clip(adaptive, 1e-12, None))).sum(axis=1)
    metadata = {
        "input": str(Path(args.input).resolve()),
        "matrix_shape": list(adaptive.shape),
        "physical": {
            "minimum": float(physical.min()),
            "maximum": float(physical.max()),
            "nonzero": int(np.count_nonzero(physical)),
        },
        "adaptive": {
            "minimum": float(adaptive.min()),
            "maximum": float(adaptive.max()),
            "nonzero": int(np.count_nonzero(adaptive)),
            "row_sum_minimum": float(adaptive.sum(axis=1).min()),
            "row_sum_maximum": float(adaptive.sum(axis=1).max()),
            "mean_row_entropy_nats": float(row_entropy.mean()),
            "maximum_row_entropy_nats": float(np.log(adaptive.shape[1])),
            "mean_absolute_asymmetry": float(np.mean(np.abs(adaptive - adaptive.T))),
        },
        "node_labels": NODE_LABELS,
        "top_connections": connections[:args.top_k],
        "top_nonphysical_connections": nonphysical_connections[:args.top_k],
        "shared_colour_range_for_panels_a_b": {"minimum": 0.0, "maximum": shared_max},
        "note": "The model consumes A_phys and A_adp as separate supports; panel d is descriptive, not an adjacency used by the network.",
    }
    output.with_name(output.name + "_metadata").with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
