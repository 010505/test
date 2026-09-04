from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from .data import GestureDataset
from .model import build_model
from .shrec import load_shrec17
from .topology import EDGES


def font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def project(sequence):
    points = sequence.reshape(-1, 3)
    _, _, axes = np.linalg.svd(points - points.mean(0), full_matrices=False)
    projected = (points - points.mean(0)) @ axes[:2].T
    projected = projected.reshape(len(sequence), 22, 2)
    span = max(float(np.ptp(projected[..., 0])), float(np.ptp(projected[..., 1])), 1e-6)
    return projected / span


def make_gif(sample, truth, prediction, confidence, output):
    width = height = 620
    projected = project(sample.sequence)
    frames = []
    for frame_index, joints in enumerate(projected):
        image = Image.new("RGB", (width, height), "#050712")
        draw = ImageDraw.Draw(image)
        for y in range(height):
            shade = int(7 + y / height * 10)
            draw.line((0, y, width, y), fill=(shade, shade + 3, shade + 12))
        xy = np.column_stack((width / 2 + joints[:, 0] * 470, height * .56 - joints[:, 1] * 470))
        for source, target in EDGES:
            draw.line((*xy[source], *xy[target]), fill="#48f7e3", width=4)
        for index, (x, y) in enumerate(xy):
            radius = 7 if index in (0, 1, 5, 9, 13, 17, 21) else 5
            color = "#ff4fbd" if index in (0, 1) else "#eafffb"
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
        status = "CORRECT" if truth == prediction else "MISCLASSIFIED"
        status_color = "#48f7e3" if truth == prediction else "#ff4fbd"
        draw.text((28, 24), "GESTUREGRAPH / SHREC'17", font=font(17, True), fill="#8798bd")
        draw.text((28, 52), status, font=font(28, True), fill=status_color)
        draw.text((28, height-76), f"TRUE  {truth.upper().replace('_', ' ')}", font=font(19, True), fill="#eafffb")
        draw.text((28, height-48), f"PRED  {prediction.upper().replace('_', ' ')}  {confidence:.1%}", font=font(17), fill="#aebcdb")
        draw.text((width-110, 28), f"{frame_index+1:02d}/64", font=font(16, True), fill="#8798bd")
        frames.append(image)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=55, loop=0, optimize=True)


def main():
    parser = argparse.ArgumentParser(description="Render confident correct and incorrect SHREC predictions")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="runs/shrec17_benchmark")
    args = parser.parse_args()
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    labels = checkpoint["labels"]; frames = int(checkpoint.get("frames", 64))
    model = build_model(checkpoint.get("model_name", "stgcn"), len(labels), frames, float(checkpoint.get("dropout", .15)), checkpoint.get("ablation", "none"))
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    samples = load_shrec17(args.data, "test", frames, 14)
    loader = DataLoader(GestureDataset(samples, labels), batch_size=32)
    probabilities = []
    with torch.no_grad():
        for inputs, _ in loader:
            probabilities.append(model(inputs).softmax(1).numpy())
    probabilities = np.concatenate(probabilities)
    truth = np.asarray([labels.index(sample.label) for sample in samples])
    predicted = probabilities.argmax(1); confidence = probabilities.max(1)
    correct_indices = np.flatnonzero(predicted == truth)
    incorrect_indices = np.flatnonzero(predicted != truth)
    chosen = {
        "correct": int(correct_indices[np.argmax(confidence[correct_indices])]),
        "incorrect": int(incorrect_indices[np.argmax(confidence[incorrect_indices])]),
    }
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for kind, index in chosen.items():
        true_label, predicted_label = labels[truth[index]], labels[predicted[index]]
        destination = output / f"{kind}_classification.gif"
        make_gif(samples[index], true_label, predicted_label, float(confidence[index]), destination)
        metadata[kind] = {"file": str(destination), "true": true_label, "predicted": predicted_label, "confidence": float(confidence[index]), "source": str(samples[index].path)}
        print(kind, true_label, "->", predicted_label, f"{confidence[index]:.1%}")
    (output / "visualizations.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
