from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gesturegraph.data import normalize_sequence, resample_sequence
from gesturegraph.progressive import (
    PrefixSequenceDataset,
    build_progressive_model,
    load_raw_shrec17_npz,
)
from gesturegraph.topology import EDGES


LABEL_EN = {
    "expand": "Expand",
    "grab": "Grab",
    "pinch": "Pinch",
    "rotation_ccw": "Counterclockwise rotation",
    "rotation_cw": "Clockwise rotation",
    "shake": "Shake",
    "swipe_down": "Swipe down",
    "swipe_left": "Swipe left",
    "swipe_plus": "Plus-shaped swipe",
    "swipe_right": "Swipe right",
    "swipe_up": "Swipe up",
    "swipe_v": "V-shaped swipe",
    "swipe_x": "X-shaped swipe",
    "tap": "Tap",
}

PALETTE = {
    "paper": "#F5F7FA",
    "ink": "#17212B",
    "muted": "#647382",
    "line": "#D6DEE5",
    "panel": "#FFFFFF",
    "skeleton_bg": "#07111F",
    "cyan": "#4DE3D2",
    "blue": "#0072B2",
    "teal": "#009E73",
    "orange": "#D55E00",
    "purple": "#8A62B8",
    "joint": "#F4FAFF",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def project(sequence: np.ndarray) -> np.ndarray:
    points = sequence.reshape(-1, 3)
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projected = (centered @ axes[:2].T).reshape(len(sequence), 22, 2)
    span = max(float(np.ptp(projected[..., 0])), float(np.ptp(projected[..., 1])), 1e-6)
    return projected / span


def decision_update(
    probabilities: np.ndarray,
    confidence: float,
    margin: float,
) -> int:
    selected = len(probabilities) - 1
    for update in range(1, len(probabilities)):
        current = probabilities[update]
        previous = probabilities[update - 1]
        order = np.argsort(current)[::-1]
        stable = int(order[0]) == int(previous.argmax())
        separated = float(current[order[0]] - current[order[1]]) >= margin
        if stable and float(current[order[0]]) >= confidence and separated:
            selected = update
            break
    return selected


def choose_recovery_sample(
    probabilities: np.ndarray,
    targets: np.ndarray,
    confidence: float,
    margin: float,
) -> tuple[int, int, int]:
    predictions = probabilities.argmax(axis=-1)
    candidates = []
    for index, target in enumerate(targets):
        if predictions[index, 0] == target or predictions[index, -1] != target:
            continue
        recovery = next(
            (step for step in range(1, predictions.shape[1]) if predictions[index, step] == target),
            predictions.shape[1] - 1,
        )
        selected = decision_update(probabilities[index], confidence, margin)
        if selected >= predictions.shape[1] - 1 or predictions[index, selected] != target:
            continue
        remains_correct = bool(np.all(predictions[index, recovery:] == target))
        decision_probability = float(probabilities[index, selected, target])
        final_probability = float(probabilities[index, -1, target])
        score = (
            1 if remains_correct else 0,
            -selected,
            -recovery,
            decision_probability,
            final_probability,
        )
        candidates.append((score, index, recovery, selected))
    if not candidates:
        raise RuntimeError("No early-error recovery sample with a correct early decision was found")
    _, index, recovery, selected = max(candidates, key=lambda item: item[0])
    return index, recovery, selected


def evaluate(
    model: torch.nn.Module,
    dataset: PrefixSequenceDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_probabilities, all_targets = [], []
    model.eval()
    with torch.inference_mode():
        for views, lengths, ratios, targets in loader:
            outputs = model(views.to(device), lengths.to(device), ratios.to(device))
            all_probabilities.append(outputs.exp().cpu().numpy())
            all_targets.append(targets.numpy())
    return np.concatenate(all_probabilities), np.concatenate(all_targets)


def rounded_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    outline: str | None = None,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=width)


def draw_skeleton(
    draw: ImageDraw.ImageDraw,
    joints: np.ndarray,
    box: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    xy = np.column_stack(
        (
            left + width * 0.50 + joints[:, 0] * width * 0.82,
            top + height * 0.53 - joints[:, 1] * height * 0.82,
        )
    )
    for source, target in EDGES:
        draw.line((*xy[source], *xy[target]), fill=PALETTE["cyan"], width=5)
    for index, (x, y) in enumerate(xy):
        radius = 8 if index in (0, 1, 5, 9, 13, 17, 21) else 6
        color = PALETTE["purple"] if index in (0, 1) else PALETTE["joint"]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def label_text(label: str) -> str:
    return LABEL_EN.get(label, label.replace("_", " ").title())


def render_frame(
    projected: np.ndarray,
    frame_index: int,
    probabilities: np.ndarray,
    labels: list[str],
    truth_index: int,
    ratios: tuple[float, ...],
    recovery_update: int,
    selected_update: int,
    confidence_threshold: float,
    margin_threshold: float,
    checkpoint_seed: int,
) -> Image.Image:
    canvas = Image.new("RGB", (1400, 820), PALETTE["paper"])
    draw = ImageDraw.Draw(canvas)
    progress = (frame_index + 1) / len(projected)
    available = [index for index, ratio in enumerate(ratios) if progress + 1e-9 >= ratio]
    update = available[-1] if available else -1
    truth = labels[truth_index]

    draw.text((42, 24), "Progressive Gesture Recognition: Error Recovery and Early Decision", font=font(31, True), fill=PALETTE["ink"])
    draw.text(
        (42, 64),
        f"Best validation checkpoint · seed {checkpoint_seed} · four-step class diffusion",
        font=font(17),
        fill=PALETTE["muted"],
    )

    rounded_rectangle(draw, (36, 100, 748, 778), PALETTE["skeleton_bg"])
    draw.text((62, 124), f"MOTION SKELETON  ·  FRAME {frame_index + 1:02d}/64", font=font(21, True), fill=PALETTE["joint"])
    draw.text((62, 158), f"Ground truth: {label_text(truth)}", font=font(19), fill=PALETTE["joint"])
    draw_skeleton(draw, projected[frame_index], (76, 194, 710, 700))
    draw.line((76, 720, 708, 720), fill="#30445B", width=8)
    draw.line((76, 720, 76 + int(632 * progress), 720), fill=PALETTE["cyan"], width=8)
    draw.text((76, 738), f"Observed: {progress:.0%}", font=font(18, True), fill=PALETTE["joint"])

    rounded_rectangle(draw, (774, 100, 1364, 778), PALETTE["panel"], PALETTE["line"], 2)
    draw.text((806, 128), "Class Posterior and Decision State", font=font(24, True), fill=PALETTE["ink"])

    if update < 0:
        draw.text((806, 188), "WAITING FOR FIRST UPDATE (25%)", font=font(21, True), fill=PALETTE["muted"])
        draw.text((806, 232), "No future frames are used.", font=font(18), fill=PALETTE["muted"])
        current = np.full(len(labels), 1.0 / len(labels), dtype=np.float32)
        prediction = None
        status = "OBSERVING"
        status_color = PALETTE["muted"]
    else:
        current = probabilities[update]
        prediction = int(current.argmax())
        predicted_label = labels[prediction]
        if update == 0 and prediction != truth_index:
            status, status_color = "EARLY ERROR", PALETTE["orange"]
        elif update == recovery_update:
            status, status_color = "RECOVERED", PALETTE["teal"]
        elif update >= selected_update:
            status, status_color = f"EARLY DECISION AT {ratios[selected_update]:.0%}", PALETTE["blue"]
        elif prediction == truth_index:
            status, status_color = "CORRECT · CONTINUE", PALETTE["teal"]
        else:
            status, status_color = "OBSERVING", PALETTE["orange"]
        draw.text((806, 180), status, font=font(22, True), fill=status_color)
        draw.text(
            (806, 218),
            f"Current prediction: {label_text(predicted_label)}",
            font=font(18, True),
            fill=PALETTE["ink"],
        )

    order = np.argsort(current)[::-1][:3]
    draw.text((806, 270), "Top-3 Class Probabilities", font=font(19, True), fill=PALETTE["ink"])
    for rank, class_index in enumerate(order):
        value = float(current[class_index])
        label = labels[int(class_index)]
        y = 314 + rank * 82
        is_prediction = prediction is not None and int(class_index) == prediction
        is_truth = int(class_index) == truth_index
        if is_prediction and not is_truth:
            color = PALETTE["orange"]
        elif is_truth:
            color = PALETTE["teal"]
        elif is_prediction:
            color = PALETTE["blue"]
        else:
            color = "#9AA8B4"
        draw.text((806, y), f"{rank + 1}. {label_text(label)}", font=font(16, is_prediction), fill=PALETTE["ink"])
        draw.rounded_rectangle((806, y + 32, 1300, y + 52), radius=10, fill="#E6EBEF")
        draw.rounded_rectangle((806, y + 32, 806 + max(3, int(494 * value)), y + 52), radius=10, fill=color)
        draw.text((1306, y + 27), f"{value * 100:6.3f}%", font=font(14, True), fill=PALETTE["ink"])

    draw.text((806, 566), "Five-Stage Prediction Trajectory", font=font(19, True), fill=PALETTE["ink"])
    x_positions = np.linspace(830, 1310, len(ratios)).astype(int)
    draw.line((x_positions[0], 624, x_positions[-1], 624), fill=PALETTE["line"], width=4)
    stage_predictions = probabilities.argmax(axis=-1)
    for stage, (x, ratio) in enumerate(zip(x_positions, ratios)):
        reached = stage <= update
        correct = int(stage_predictions[stage]) == truth_index
        fill = PALETTE["teal"] if correct else PALETTE["orange"]
        if not reached:
            fill = "#D7DEE4"
        radius = 13 if stage == selected_update else 10
        draw.ellipse((x - radius, 624 - radius, x + radius, 624 + radius), fill=fill, outline=PALETTE["ink"] if stage == selected_update else None, width=3)
        draw.text((x - 19, 646), f"{ratio:.0%}", font=font(14, stage == selected_update), fill=PALETTE["ink"])
        short_label = labels[int(stage_predictions[stage])].replace("rotation_", "rot_").replace("swipe_", "sw_")
        result_text = ("OK " if correct else "ERR ") + short_label
        draw.text((x - 38, 674), result_text, font=font(12), fill=fill if reached else PALETTE["muted"])

    draw.text(
        (806, 728),
        f"Decision rule: same class twice; conf. ≥ {confidence_threshold:.2f}; Top-2 margin ≥ {margin_threshold:.2f}",
        font=font(13),
        fill=PALETTE["muted"],
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a progressive class-diffusion recovery GIF")
    parser.add_argument("--data", default="data/shrec17_ddnet_npz")
    parser.add_argument("--checkpoint", default="runs/progressive_seed43/04_class_diffusion/best.pt")
    parser.add_argument("--model-json", default="runs/progressive_seed43/04_class_diffusion/model.json")
    parser.add_argument("--output", default="runs/progressive_visualization")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    labels = list(checkpoint["labels"])
    ratios = tuple(float(value) for value in checkpoint["observation_ratios"])
    model = build_progressive_model(checkpoint["experiment"], len(labels))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    samples = load_raw_shrec17_npz(args.data, "test", 14)
    dataset = PrefixSequenceDataset(samples, labels, frames=int(checkpoint["frames"]), ratios=ratios)
    probabilities, targets = evaluate(model, dataset, device, args.batch_size)

    model_result = json.loads(Path(args.model_json).read_text(encoding="utf-8"))
    calibration = model_result["early_exit_calibration"]
    confidence = float(calibration["confidence_threshold"])
    margin = float(calibration["margin_threshold"])
    sample_index, recovery_update, selected_update = choose_recovery_sample(
        probabilities, targets, confidence, margin
    )

    display_sequence = resample_sequence(normalize_sequence(samples[sample_index].sequence), 64)
    projected = project(display_sequence)
    sample_probabilities = probabilities[sample_index]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    gif_path = output / "progressive_class_diffusion_recovery.gif"
    mobile_gif_path = output / "progressive_class_diffusion_recovery_mobile.gif"
    preview_path = output / "progressive_class_diffusion_recovery_preview.png"
    metadata_path = output / "progressive_class_diffusion_recovery.json"

    frames = [
        render_frame(
            projected,
            frame_index,
            sample_probabilities,
            labels,
            int(targets[sample_index]),
            ratios,
            recovery_update,
            selected_update,
            confidence,
            margin,
            int(checkpoint["seed"]),
        )
        for frame_index in range(64)
    ]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=True,
    )
    mobile_frames = [
        frame.resize((840, 492), Image.Resampling.LANCZOS).convert(
            "P", palette=Image.Palette.ADAPTIVE, colors=128
        )
        for frame in frames[::2]
    ]
    mobile_frames[0].save(
        mobile_gif_path,
        save_all=True,
        append_images=mobile_frames[1:],
        duration=180,
        loop=0,
        optimize=True,
    )
    frames[min(63, math.ceil(ratios[selected_update] * 64) - 1)].save(preview_path)

    predictions = sample_probabilities.argmax(axis=-1)
    metadata = {
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_seed": int(checkpoint["seed"]),
        "best_validation_prefix_auc": float(checkpoint["best_validation_score"]),
        "sample_test_index": int(sample_index),
        "sample_npz_index": int(samples[sample_index].index),
        "truth": labels[int(targets[sample_index])],
        "ratios": list(ratios),
        "predictions": [labels[int(value)] for value in predictions],
        "top3": [
            [
                {"label": labels[int(class_index)], "probability": float(stage[class_index])}
                for class_index in np.argsort(stage)[::-1][:3]
            ]
            for stage in sample_probabilities
        ],
        "recovery_ratio": ratios[recovery_update],
        "decision_ratio": ratios[selected_update],
        "confidence_threshold": confidence,
        "margin_threshold": margin,
        "device": str(device),
        "gif": str(gif_path),
        "mobile_gif": str(mobile_gif_path),
        "preview": str(preview_path),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
