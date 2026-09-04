from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from .data import GestureDataset
from .model import build_model
from .shrec import load_shrec17
from .topology import EDGES
from .visualize import font, project


def render_reference(sample, label: str, confidence: float, destination: Path):
    width, height = 420, 300
    projected = project(sample.sequence)
    lower = projected.min(axis=(0, 1))
    upper = projected.max(axis=(0, 1))
    center = (lower + upper) / 2
    scale = min((width - 44) / max(float(upper[0] - lower[0]), 1e-6), (height - 82) / max(float(upper[1] - lower[1]), 1e-6))
    frames = []
    for frame_index in range(0, len(projected), 2):
        joints = projected[frame_index]
        image = Image.new("RGB", (width, height), "#050712")
        draw = ImageDraw.Draw(image)
        for y in range(height):
            shade = int(7 + y / height * 10)
            draw.line((0, y, width, y), fill=(shade, shade + 3, shade + 12))
        xy = np.column_stack((width / 2 + (joints[:, 0] - center[0]) * scale, height * .52 - (joints[:, 1] - center[1]) * scale))
        for source, target in EDGES:
            draw.line((*xy[source], *xy[target]), fill="#48f7e3", width=3)
        for index, (x, y) in enumerate(xy):
            radius = 5 if index in (0, 1, 5, 9, 13, 17, 21) else 3
            color = "#ff4fbd" if index in (0, 1) else "#eafffb"
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        draw.text((18, 16), label.upper().replace("_", " "), font=font(17, True), fill="#eafffb")
        draw.text((18, height - 28), f"SHREC'17 REFERENCE  {confidence:.0%}", font=font(12, True), fill="#8798bd")
        draw.text((width - 64, 18), f"{frame_index + 1:02d}/64", font=font(11, True), fill="#8798bd")
        frames.append(image)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(destination, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True)


def main():
    parser = argparse.ArgumentParser(description="Generate one official SHREC reference animation per 14-class label")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="assets/references")
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    labels = checkpoint["labels"]
    frames = int(checkpoint.get("frames", 64))
    model = build_model(checkpoint.get("model_name", "stgcn"), len(labels), frames, float(checkpoint.get("dropout", .15)), checkpoint.get("ablation", "none"))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    samples = load_shrec17(args.data, "test", frames, 14)
    loader = DataLoader(GestureDataset(samples, labels), batch_size=64)
    probabilities = []
    with torch.no_grad():
        for inputs, _ in loader:
            probabilities.append(model(inputs).softmax(1).numpy())
    probabilities = np.concatenate(probabilities)
    truth = np.asarray([labels.index(sample.label) for sample in samples])
    predicted = probabilities.argmax(1)

    output = Path(args.output)
    metadata = {}
    for label_index, label in enumerate(labels):
        candidates = np.flatnonzero((truth == label_index) & (predicted == label_index))
        if not len(candidates):
            candidates = np.flatnonzero(truth == label_index)
        index = int(candidates[np.argmax(probabilities[candidates, label_index])])
        confidence = float(probabilities[index, label_index])
        destination = output / f"{label}.gif"
        render_reference(samples[index], label, confidence, destination)
        metadata[label] = {"file": str(destination), "confidence": confidence, "source": str(samples[index].path)}
        print(label, f"{confidence:.1%}", destination)
    (output / "references.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
