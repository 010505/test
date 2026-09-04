from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gesturegraph.continuous_online import continuous_decisions
from gesturegraph.data import normalize_sequence, resample_sequence
from gesturegraph.progressive import load_raw_shrec17_npz
from scripts.render_continuous_online_diffusion_gif import render_frame
from scripts.render_progressive_diffusion_gif import project


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a delayed-recovery online alternative")
    parser.add_argument("--data", default="data/shrec17_ddnet_npz")
    parser.add_argument("--checkpoint", default="runs/drifting_seed43/08_one_step_distilled/best.pt")
    parser.add_argument("--source", default="runs/continuous_online_seed43")
    parser.add_argument("--sample-index", type=int, default=354)
    parser.add_argument("--name", default="continuous_online_class_diffusion_alternative_1")
    args = parser.parse_args()

    source = Path(args.source)
    base_metadata = json.loads(
        (source / "continuous_online_class_diffusion.json").read_text(encoding="utf-8")
    )
    cached = np.load(source / "continuous_test_predictions.npz")
    probabilities = cached["probabilities"]
    targets = cached["targets"]
    ratios = tuple(float(value) for value in cached["ratios"])
    calibration = base_metadata["continuous_validation_calibration"]
    decisions = continuous_decisions(
        probabilities,
        targets,
        ratios,
        float(calibration["confidence_threshold"]),
        float(calibration["margin_threshold"]),
        int(calibration["stable_updates"]),
        float(calibration["min_decision_ratio"]),
    )

    sample_index = int(args.sample_index)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    labels = list(checkpoint["labels"])
    samples = load_raw_shrec17_npz(args.data, "test")
    sample_probabilities = probabilities[sample_index]
    truth_index = int(targets[sample_index])
    predictions = sample_probabilities.argmax(axis=-1)
    correct = predictions == truth_index
    recovered = np.flatnonzero(correct)
    if correct[0] or not len(recovered) or not correct[-1]:
        raise ValueError("sample must start wrong and finish correct")
    recovery_index = int(recovered[0])
    decision_index = int(decisions.indices[sample_index])
    if decision_index >= len(ratios) - 1 or predictions[decision_index] != truth_index:
        raise ValueError("sample must make a correct early decision")
    if not correct[decision_index:].all():
        raise ValueError("sample must remain correct after the early decision")

    sequence = resample_sequence(normalize_sequence(samples[sample_index].sequence), 64)
    projected = project(sequence)
    latency_ms = float(
        base_metadata["deployment"]["latency_ms_per_batch_one_update"]
    )
    rendered = [
        render_frame(
            projected,
            frame_index,
            sample_probabilities,
            ratios,
            labels,
            truth_index,
            recovery_index,
            decision_index,
            calibration,
            int(checkpoint["seed"]),
            latency_ms,
        )
        for frame_index in range(64)
    ]

    gif_path = source / f"{args.name}.gif"
    mobile_path = source / f"{args.name}_mobile.gif"
    preview_path = source / f"{args.name}_preview.png"
    metadata_path = source / f"{args.name}.json"
    rendered[0].save(
        gif_path,
        save_all=True,
        append_images=rendered[1:],
        duration=100,
        loop=0,
        optimize=True,
    )
    mobile_frames = [
        frame.resize((840, 492), Image.Resampling.LANCZOS).convert(
            "P", palette=Image.Palette.ADAPTIVE, colors=128
        )
        for frame in rendered[::2]
    ]
    mobile_frames[0].save(
        mobile_path,
        save_all=True,
        append_images=mobile_frames[1:],
        duration=190,
        loop=0,
        optimize=True,
    )
    rendered[decision_index + 15].save(preview_path)

    transition_runs = []
    run_start = 0
    for update in range(1, len(predictions) + 1):
        if update == len(predictions) or predictions[update] != predictions[run_start]:
            transition_runs.append(
                {
                    "start_frame": run_start + 16,
                    "end_frame": update - 1 + 16,
                    "prediction": labels[int(predictions[run_start])],
                }
            )
            run_start = update
    metadata = {
        "source_checkpoint": args.checkpoint,
        "sample_test_index": sample_index,
        "sample_npz_index": int(samples[sample_index].index),
        "truth": labels[truth_index],
        "prediction_runs": transition_runs,
        "recovery_frame": recovery_index + 16,
        "decision_frame": decision_index + 16,
        "final_prediction": labels[int(predictions[-1])],
        "stop_rule": calibration,
        "gif": str(gif_path),
        "mobile_gif": str(mobile_path),
        "preview": str(preview_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
