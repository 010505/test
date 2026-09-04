from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gesturegraph.drifting import ONE_STEP_EXPERIMENTS, build_one_step_model
from gesturegraph.progressive import PrefixSequenceDataset, build_progressive_model, load_raw_shrec17_npz
from gesturegraph.progressive_benchmark import measure_online_latency


EXPERIMENTS = (
    "04_four_step_teacher",
    "04_one_step_truncation",
) + ONE_STEP_EXPERIMENTS


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Stable CPU latency audit for drifting models")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--data", default="data/shrec17_ddnet_npz")
    parser.add_argument("--output", default="runs/drifting_aggregate/cpu_latency.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cpu")
    test_samples = load_raw_shrec17_npz(args.data, "test")
    per_seed = []
    for run_name in args.runs:
        root = Path(run_name)
        teacher_checkpoint = torch.load(
            root.parent / root.name.replace("drifting_", "progressive_")
            / "04_class_diffusion" / "best.pt",
            map_location=device,
            weights_only=True,
        )
        labels = list(teacher_checkpoint["labels"])
        ratios = tuple(float(value) for value in teacher_checkpoint["observation_ratios"])
        loader = DataLoader(
            PrefixSequenceDataset(test_samples, labels, teacher_checkpoint["frames"], ratios),
            batch_size=8,
        )
        models = {}
        for experiment, steps in (("04_four_step_teacher", 4), ("04_one_step_truncation", 1)):
            model = build_progressive_model("04_class_diffusion", len(labels))
            model.load_state_dict(teacher_checkpoint["model_state"])
            model.inference_steps = steps
            models[experiment] = model.eval()
        for experiment in ONE_STEP_EXPERIMENTS:
            checkpoint = torch.load(root / experiment / "best.pt", map_location=device, weights_only=True)
            model = build_one_step_model(len(labels))
            model.load_state_dict(checkpoint["model_state"])
            models[experiment] = model.eval()

        row = {"seed": int(teacher_checkpoint["seed"]), "measurements": {}}
        for experiment, model in models.items():
            values = [
                measure_online_latency(model, loader, device, iterations=args.iterations)[
                    "mean_update_ms"
                ]
                for _ in range(args.repeats)
            ]
            row["measurements"][experiment] = {
                "repeats_ms": values,
                "median_ms": float(np.median(values)),
            }
            print(
                f"seed {row['seed']} {experiment}: "
                f"{row['measurements'][experiment]['median_ms']:.3f} ms/update"
            )
        per_seed.append(row)

    aggregate = {}
    baseline = None
    for experiment in EXPERIMENTS:
        medians = [row["measurements"][experiment]["median_ms"] for row in per_seed]
        aggregate[experiment] = stats(medians)
        if experiment == "04_four_step_teacher":
            baseline = aggregate[experiment]["mean"]
    for experiment in EXPERIMENTS[1:]:
        aggregate[experiment]["reduction_vs_four_step"] = (
            1.0 - aggregate[experiment]["mean"] / baseline
        )
    payload = {
        "protocol": {
            "device": "cpu",
            "torch_threads": 1,
            "batch_size": 1,
            "repeats": args.repeats,
            "sequences_per_repeat": args.iterations,
            "updates_per_sequence": 5,
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

