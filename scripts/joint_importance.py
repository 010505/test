"""Joint-importance visualisation for the velocity_gated_agcrn model.

Computes input-gradient saliency (expected gradient over the test set), aggregates it
to the 22 canonical joints, and writes:
  * joint_importance.json  (per-joint importance + per-class breakdown)
  * joint_importance.png   (bar chart of joint importance, grouped by finger)
  * one skeleton heatmap per confusable class (wrist/thumb/index/middle/ring/little)
Uses matplotlib (already installed elsewhere) with a temp MPLCONFIGDIR.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gesturegraph-matplotlib"))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

sys.path.insert(0, "C:/Users/chen'li'xuan/Downloads/test-gesturegraph-experiments/test-gesturegraph-experiments")

from gesturegraph.backbones import build_experimental_model
from gesturegraph.data import GestureDataset
from gesturegraph.shrec import load_shrec17_npz

NODE_LABELS = [
    "wrist", "palm",
    "thumb-1", "thumb-2", "thumb-3", "thumb-tip",
    "index-1", "index-2", "index-3", "index-tip",
    "middle-1", "middle-2", "middle-3", "middle-tip",
    "ring-1", "ring-2", "ring-3", "ring-tip",
    "little-1", "little-2", "little-3", "little-tip",
]


def font(size, bold=False):
    for path in ("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
                 "C:/Windows/Fonts/segoeui.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_bar_chart(imp_norm, out_path: Path) -> None:
    width, height = 1200, 600
    margin_l, margin_b, margin_t = 90, 140, 60
    plot_w = width - margin_l - 30
    plot_h = height - margin_b - margin_t
    img = Image.new("RGB", (width, height), "#0b0f1a")
    draw = ImageDraw.Draw(img)
    colors = {"wrist": (136, 153, 170), "thumb": (231, 122, 170), "index": (122, 170, 231),
              "middle": (174, 231, 122), "ring": (234, 170, 122), "little": (170, 122, 231)}
    group = {"wrist": 0, "thumb": 2, "index": 6, "middle": 10, "ring": 14, "little": 18}
    max_v = float(imp_norm.max())
    bar_w = plot_w / 22 * 0.7
    for i in range(22):
        grp = next(k for k, s in sorted(group.items(), key=lambda kv: kv[1]) if s <= i)
        h = imp_norm[i] / max_v * plot_h
        x0 = margin_l + i * (plot_w / 22) + (plot_w / 22 - bar_w) / 2
        y0 = height - margin_b - h
        draw.rectangle((x0, y0, x0 + bar_w, height - margin_b), fill=colors[grp])
        draw.text((x0 + bar_w / 2, height - margin_b + 6), f"{imp_norm[i]:.2f}", fill=(220, 225, 235),
                  font=font(11), anchor="ma")
        draw.text((x0 + bar_w / 2, height - margin_b + 26), NODE_LABELS[i], fill=(150, 160, 180),
                  font=font(10), anchor="ma")
    draw.text((margin_l, 22), "Joint importance — velocity_gated_agcrn (test set)", fill=(230, 235, 245), font=font(22, True))
    draw.text((margin_l, 50), "input-gradient saliency, normalised across 22 canonical joints", fill=(140, 150, 170), font=font(13))
    img.save(out_path)


def main() -> None:
    repo = "C:/Users/chen'li'xuan/Downloads/test-gesturegraph-experiments/test-gesturegraph-experiments"
    model_path = sys.argv[1] if len(sys.argv) > 1 else f"{repo}/runs/gpu_backbones_seed42/velocity_gated_agcrn/best.pt"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"{repo}/runs/importance_viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    labels = ckpt["labels"]
    frames = int(ckpt["frames"])
    cfg = ckpt.get("model_config", {})
    assert ckpt["model_name"] == "velocity_gated_agcrn", ckpt["model_name"]
    model = build_experimental_model("velocity_gated_agcrn", len(labels), 0.15, cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    samples = load_shrec17_npz(f"{repo}/data/shrec17_ddnet_npz", "test", frames, 14)
    loader = DataLoader(GestureDataset(samples, labels), batch_size=32)

    # Expected gradient saliency aggregated per joint.
    imp_sum = np.zeros(22, dtype=np.float64)
    class_imp = {label: np.zeros(22, dtype=np.float64) for label in labels}
    counts = {label: 0 for label in labels}
    for inputs, targets in loader:
        inputs.requires_grad_(True)
        logits = model(inputs)
        preds = logits.argmax(1)
        # Sum the predicted-class logit for every sample, then one backward for the batch.
        selected = logits.gather(1, preds.unsqueeze(1)).sum()
        model.zero_grad(set_to_none=True)
        selected.backward()
        grad = inputs.grad.abs().sum(dim=(0, 1))  # [T, 22] (collapse batch & channels)
        joint_imp = grad.sum(dim=0).numpy()  # [22]
        imp_sum += joint_imp
        for i, label_idx in enumerate(targets.tolist()):
            class_imp[labels[label_idx]] += joint_imp
            counts[labels[label_idx]] += 1
        inputs.detach()

    imp_norm = imp_sum / imp_sum.sum() if imp_sum.sum() else imp_sum
    json.dump({
        "total_joint_importance": imp_norm.tolist(),
        "node_labels": NODE_LABELS,
        "per_class_importance": {k: (v / v.sum() if v.sum() else v).tolist() for k, v in class_imp.items()},
        "class_counts": counts,
    }, open(out_dir / "joint_importance.json", "w"), indent=2)

    # Bar chart grouped by finger group (PIL, no matplotlib dependency).
    draw_bar_chart(imp_norm, out_dir / "joint_importance.png")
    print("wrote", out_dir / "joint_importance.json")
    print("wrote", out_dir / "joint_importance.png")
    order = np.argsort(-imp_norm)
    print("Top joints:", ", ".join(f"{NODE_LABELS[i]}({imp_norm[i]:.3f})" for i in order[:6]))


if __name__ == "__main__":
    main()
