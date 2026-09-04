from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gesturegraph.continuous_online import (
    DensePrefixSequenceDataset,
    calibrate_continuous_early_exit,
    continuous_decisions,
    evaluate_continuous_online,
)
from gesturegraph.data import normalize_sequence, resample_sequence
from gesturegraph.drifting import build_one_step_model
from gesturegraph.progressive import load_raw_shrec17_npz, stratified_raw_split
from scripts.render_progressive_diffusion_gif import (
    PALETTE,
    draw_skeleton,
    font,
    label_text,
    project,
    rounded_rectangle,
)


def choose_sample(
    probabilities: np.ndarray,
    targets: np.ndarray,
    decision_indices: np.ndarray,
) -> tuple[int, int]:
    predictions = probabilities.argmax(axis=-1)
    candidates = []
    for sample_index, target in enumerate(targets):
        correct = predictions[sample_index] == int(target)
        if correct[0] or not correct[-1]:
            continue
        recovered = np.flatnonzero(correct)
        if not len(recovered):
            continue
        recovery_index = int(recovered[0])
        decision_index = int(decision_indices[sample_index])
        if decision_index >= probabilities.shape[1] - 1:
            continue
        if predictions[sample_index, decision_index] != int(target):
            continue
        if not correct[decision_index:].all():
            continue
        decision_probability = float(probabilities[sample_index, decision_index, target])
        final_probability = float(probabilities[sample_index, -1, target])
        candidates.append(
            (
                decision_index,
                recovery_index,
                -decision_probability,
                -final_probability,
                sample_index,
            )
        )
    if not candidates:
        raise RuntimeError("no stable online error-recovery sample was found")
    *_, sample_index = min(candidates)
    correct = predictions[sample_index] == int(targets[sample_index])
    return int(sample_index), int(np.flatnonzero(correct)[0])


def measure_online_step_latency(
    model: torch.nn.Module,
    dataset: DensePrefixSequenceDataset,
    device: torch.device,
    iterations: int = 30,
) -> float:
    views, lengths, ratios, _ = dataset[0]
    views = views.to(device)
    lengths = lengths.to(device)
    ratios = ratios.to(device)

    def run() -> None:
        state = None
        for update in range(len(ratios)):
            _, state = model.online_step(
                views[update:update + 1],
                lengths[update:update + 1],
                ratios[update:update + 1],
                state,
            )

    with torch.inference_mode():
        for _ in range(3):
            run()
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            run()
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / (iterations * len(ratios))


def render_frame(
    projected: np.ndarray,
    frame_index: int,
    probabilities: np.ndarray,
    ratios: tuple[float, ...],
    labels: list[str],
    truth_index: int,
    recovery_index: int,
    decision_index: int,
    calibration: dict,
    checkpoint_seed: int,
    latency_ms: float,
) -> Image.Image:
    canvas = Image.new("RGB", (1400, 820), PALETTE["paper"])
    draw = ImageDraw.Draw(canvas)
    progress = (frame_index + 1) / 64
    update = frame_index - 15 if frame_index >= 15 else -1
    truth = labels[truth_index]

    draw.text(
        (42, 24),
        "Continuous Online Class Diffusion: Frame-by-Frame Markov Decisions",
        font=font(30, True),
        fill=PALETTE["ink"],
    )
    draw.text(
        (42, 64),
        f"Best one-step student · seed {checkpoint_seed} · one reverse NFE per frame",
        font=font(17),
        fill=PALETTE["muted"],
    )

    rounded_rectangle(draw, (36, 100, 748, 778), PALETTE["skeleton_bg"])
    draw.text(
        (62, 124), f"LIVE SKELETON  ·  FRAME {frame_index + 1:02d}/64",
        font=font(21, True), fill=PALETTE["joint"],
    )
    draw.text((62, 158), f"Ground truth: {label_text(truth)}", font=font(19), fill=PALETTE["joint"])
    draw_skeleton(draw, projected[frame_index], (76, 194, 710, 680))
    draw.line((76, 704, 708, 704), fill="#30445B", width=8)
    draw.line((76, 704, 76 + int(632 * progress), 704), fill=PALETTE["cyan"], width=8)
    draw.text((76, 724), f"Observed: {progress:.0%}", font=font(18, True), fill=PALETTE["joint"])
    if update >= 0:
        draw.text(
            (360, 724),
            f"Online update {update + 1:02d}/{len(ratios)} · {latency_ms:.2f} ms/update",
            font=font(15), fill=PALETTE["joint"],
        )

    rounded_rectangle(draw, (774, 100, 1364, 778), PALETTE["panel"], PALETTE["line"], 2)
    draw.text((806, 128), "Posterior State and Real-Time Decision", font=font(23, True), fill=PALETTE["ink"])

    if update < 0:
        current = np.full(len(labels), 1.0 / len(labels), dtype=np.float32)
        prediction = None
        draw.text((806, 180), "WARM-UP · FIRST UPDATE AT FRAME 16", font=font(20, True), fill=PALETTE["muted"])
        draw.text((806, 218), "State: uniform prior · no future frames used", font=font(16), fill=PALETTE["muted"])
    else:
        current = probabilities[update]
        prediction = int(current.argmax())
        if update < recovery_index and prediction != truth_index:
            status, color = "EARLY ERROR · MARKOV STATE CAN RECOVER", PALETTE["orange"]
        elif update < decision_index:
            status, color = "RECOVERED · ACCUMULATING STABILITY", PALETTE["teal"]
        else:
            status, color = f"EARLY DECISION AT FRAME {decision_index + 16}", PALETTE["blue"]
        draw.text((806, 180), status, font=font(19, True), fill=color)
        draw.text(
            (806, 218), f"Current prediction: {label_text(labels[prediction])}",
            font=font(17, True), fill=PALETTE["ink"],
        )

    order = np.argsort(current)[::-1][:3]
    draw.text((806, 260), "Live Top-3 Probabilities", font=font(18, True), fill=PALETTE["ink"])
    for rank, class_index in enumerate(order):
        value = float(current[class_index])
        y = 298 + rank * 68
        is_prediction = prediction is not None and int(class_index) == prediction
        is_truth = int(class_index) == truth_index
        color = PALETTE["teal"] if is_truth else PALETTE["orange"] if is_prediction else "#9AA8B4"
        draw.text(
            (806, y), f"{rank + 1}. {label_text(labels[int(class_index)])}",
            font=font(15, is_prediction), fill=PALETTE["ink"],
        )
        draw.rounded_rectangle((806, y + 28, 1288, y + 46), radius=9, fill="#E6EBEF")
        draw.rounded_rectangle(
            (806, y + 28, 806 + max(3, int(482 * value)), y + 46),
            radius=9, fill=color,
        )
        draw.text((1296, y + 24), f"{value * 100:6.2f}%", font=font(13, True), fill=PALETTE["ink"])

    chart_left, chart_top, chart_right, chart_bottom = 806, 540, 1328, 690
    draw.text((806, 500), "Continuous Posterior Trajectory", font=font(18, True), fill=PALETTE["ink"])
    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=PALETTE["line"], width=2)
    draw.line((chart_left, chart_top, chart_left, chart_bottom), fill=PALETTE["line"], width=2)
    for value in (0.0, 0.5, 1.0):
        y = chart_bottom - int(value * (chart_bottom - chart_top))
        draw.line((chart_left, y, chart_right, y), fill="#EEF1F4", width=1)
        draw.text((776, y - 7), f"{value:.1f}", font=font(11), fill=PALETTE["muted"])

    initial_wrong = int(probabilities[0].argmax())
    mean_competitor = probabilities.mean(axis=0)
    competitor_order = np.argsort(mean_competitor)[::-1]
    third = next(int(value) for value in competitor_order if value not in {truth_index, initial_wrong})
    curve_classes = (truth_index, initial_wrong, third)
    curve_colors = (PALETTE["teal"], PALETTE["orange"], PALETTE["purple"])
    if update >= 0:
        xs = np.linspace(chart_left, chart_right, len(ratios))
        for class_index, color in zip(curve_classes, curve_colors):
            points = []
            for dense_update in range(update + 1):
                y = chart_bottom - probabilities[dense_update, class_index] * (chart_bottom - chart_top)
                points.append((float(xs[dense_update]), float(y)))
            if len(points) > 1:
                draw.line(points, fill=color, width=4)
            elif points:
                x, y = points[0]
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        cursor_x = float(xs[update])
        draw.line((cursor_x, chart_top, cursor_x, chart_bottom), fill=PALETTE["ink"], width=2)
        if update >= decision_index:
            decision_x = float(xs[decision_index])
            draw.line((decision_x, chart_top, decision_x, chart_bottom), fill=PALETTE["blue"], width=3)

    legend_y = 704
    legend_x = chart_left
    for class_index, color in zip(curve_classes, curve_colors):
        draw.line((legend_x, legend_y + 6, legend_x + 18, legend_y + 6), fill=color, width=4)
        draw.text((legend_x + 24, legend_y), label_text(labels[class_index]), font=font(11), fill=PALETTE["ink"])
        legend_x += 165
    draw.text(
        (806, 742),
        f"Stop rule: {calibration['stable_updates']} stable updates; conf. ≥ {calibration['confidence_threshold']:.2f}; "
        f"margin ≥ {calibration['margin_threshold']:.2f}; start ≥ {calibration['min_decision_ratio']:.0%}",
        font=font(12), fill=PALETTE["muted"],
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Render dense one-step online class diffusion")
    parser.add_argument("--data", default="data/shrec17_ddnet_npz")
    parser.add_argument("--checkpoint", default="runs/drifting_seed43/08_one_step_distilled/best.pt")
    parser.add_argument("--output", default="runs/continuous_online_seed43")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--min-frames", type=int, default=16)
    args = parser.parse_args()

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("experiment") != "08_one_step_distilled":
        raise ValueError("continuous deployment requires an 08_one_step_distilled checkpoint")
    labels = list(checkpoint["labels"])
    frames = int(checkpoint["frames"])
    model = build_one_step_model(len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    original_stage_updates = len(checkpoint["observation_ratios"]) - 1
    dense_transitions = frames - args.min_frames
    frame_inheritance = float(model.inheritance) ** (
        original_stage_updates / dense_transitions
    )
    model.inheritance = frame_inheritance
    model.eval()

    official_train = load_raw_shrec17_npz(args.data, "train")
    _, validation_samples = stratified_raw_split(official_train, 0.2, int(checkpoint["seed"]))
    test_samples = load_raw_shrec17_npz(args.data, "test")
    validation_dataset = DensePrefixSequenceDataset(
        validation_samples, labels, frames, args.min_frames, 1
    )
    test_dataset = DensePrefixSequenceDataset(test_samples, labels, frames, args.min_frames, 1)
    ratios = validation_dataset.ratios

    print(f"Evaluating {len(validation_samples)} validation samples over {len(ratios)} online updates...", flush=True)
    validation_probabilities, validation_targets = evaluate_continuous_online(
        model, validation_dataset, device, args.batch_size
    )
    calibration = calibrate_continuous_early_exit(
        validation_probabilities, validation_targets, ratios
    )
    print(f"Calibrated continuous stop rule: {calibration}", flush=True)
    print(f"Evaluating {len(test_samples)} official test samples...", flush=True)
    test_probabilities, test_targets = evaluate_continuous_online(
        model, test_dataset, device, args.batch_size
    )
    decisions = continuous_decisions(
        test_probabilities,
        test_targets,
        ratios,
        float(calibration["confidence_threshold"]),
        float(calibration["margin_threshold"]),
        int(calibration["stable_updates"]),
        float(calibration["min_decision_ratio"]),
    )
    sample_index, recovery_index = choose_sample(
        test_probabilities, test_targets, decisions.indices
    )
    decision_index = int(decisions.indices[sample_index])
    latency_ms = measure_online_step_latency(model, test_dataset, device)

    display_sequence = resample_sequence(normalize_sequence(test_samples[sample_index].sequence), 64)
    projected = project(display_sequence)
    sample_probabilities = test_probabilities[sample_index]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    gif_path = output / "continuous_online_class_diffusion.gif"
    mobile_path = output / "continuous_online_class_diffusion_mobile.gif"
    preview_path = output / "continuous_online_class_diffusion_preview.png"
    metadata_path = output / "continuous_online_class_diffusion.json"
    predictions_path = output / "continuous_test_predictions.npz"

    rendered = [
        render_frame(
            projected, frame_index, sample_probabilities, ratios, labels,
            int(test_targets[sample_index]), recovery_index, decision_index,
            calibration, int(checkpoint["seed"]), latency_ms,
        )
        for frame_index in range(64)
    ]
    rendered[0].save(
        gif_path, save_all=True, append_images=rendered[1:],
        duration=100, loop=0, optimize=True,
    )
    mobile_frames = [
        frame.resize((840, 492), Image.Resampling.LANCZOS).convert(
            "P", palette=Image.Palette.ADAPTIVE, colors=128
        )
        for frame in rendered[::2]
    ]
    mobile_frames[0].save(
        mobile_path, save_all=True, append_images=mobile_frames[1:],
        duration=190, loop=0, optimize=True,
    )
    preview_frame = min(63, decision_index + 15)
    rendered[preview_frame].save(preview_path)
    np.savez_compressed(
        predictions_path,
        probabilities=test_probabilities,
        targets=test_targets,
        ratios=np.asarray(ratios, dtype=np.float32),
    )

    test_full_accuracy = float(
        np.mean(test_probabilities[:, -1].argmax(axis=-1) == test_targets)
    )
    ratio_accuracy = np.mean(
        test_probabilities.argmax(axis=-1) == test_targets[:, None], axis=0
    )
    prefix_auc = float(np.trapezoid(ratio_accuracy, ratios) / (ratios[-1] - ratios[0]))
    truth_index = int(test_targets[sample_index])
    predictions = sample_probabilities.argmax(axis=-1)
    metadata = {
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_seed": int(checkpoint["seed"]),
        "checkpoint_validation_prefix_auc_five_stage": float(checkpoint["best_validation_score"]),
        "deployment": {
            "reverse_nfe_per_update": 1,
            "updates": len(ratios),
            "first_frame": args.min_frames,
            "last_frame": frames,
            "stride_frames": 1,
            "latency_ms_per_batch_one_update": latency_ms,
            "stage_inheritance": float(checkpoint.get("inheritance", 0.5)),
            "frame_inheritance": frame_inheritance,
            "inheritance_scaling": "gamma_frame^(48) = gamma_stage^(4)",
        },
        "continuous_validation_calibration": calibration,
        "continuous_official_test": {
            "prefix_auc": prefix_auc,
            "full_accuracy": test_full_accuracy,
            "early_decision_accuracy": decisions.accuracy,
            "average_decision_ratio": decisions.average_decision_ratio,
            "average_decision_frame": decisions.average_decision_ratio * frames,
        },
        "sample": {
            "test_index": int(sample_index),
            "npz_index": int(test_samples[sample_index].index),
            "truth": labels[truth_index],
            "initial_prediction": labels[int(predictions[0])],
            "recovery_frame": int(recovery_index + args.min_frames),
            "decision_frame": int(decision_index + args.min_frames),
            "final_prediction": labels[int(predictions[-1])],
            "ratios": list(ratios),
            "predictions": [labels[int(value)] for value in predictions],
            "probabilities": sample_probabilities.tolist(),
        },
        "device": str(device),
        "gif": str(gif_path),
        "mobile_gif": str(mobile_path),
        "preview": str(preview_path),
        "predictions_npz": str(predictions_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
