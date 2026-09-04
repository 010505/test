from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from gesturegraph.model import build_model
from gesturegraph.shrec import load_shrec17_npz


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(
        checkpoint["model_name"],
        len(checkpoint["labels"]),
        int(checkpoint["frames"]),
        float(checkpoint.get("dropout", 0.15)),
        checkpoint.get("ablation", "none"),
        checkpoint.get("model_config"),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


def tensors_from_samples(samples, labels: list[str]):
    label_to_index = {label: index for index, label in enumerate(labels)}
    coordinates = np.stack([sample.sequence for sample in samples]).astype(np.float32)
    inputs = torch.from_numpy(coordinates).permute(0, 3, 1, 2).contiguous()
    targets = torch.tensor([label_to_index[sample.label] for sample in samples], dtype=torch.long)
    return inputs, targets


def macro_metrics(targets: torch.Tensor, predictions: torch.Tensor, classes: int) -> tuple[float, float]:
    recalls, f1s = [], []
    for index in range(classes):
        true_mask = targets == index
        predicted_mask = predictions == index
        tp = int((true_mask & predicted_mask).sum())
        fn = int((true_mask & ~predicted_mask).sum())
        fp = int((~true_mask & predicted_mask).sum())
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
    return float(np.mean(recalls)), float(np.mean(f1s))


def calibration_metrics(confidences: torch.Tensor, correct: torch.Tensor, bins: int = 10) -> dict:
    ece = 0.0
    mce = 0.0
    rows = []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = (confidences >= lower) & (confidences < upper if index < bins - 1 else confidences <= upper)
        count = int(mask.sum())
        if count:
            confidence = float(confidences[mask].mean())
            accuracy = float(correct[mask].float().mean())
            gap = abs(accuracy - confidence)
            ece += count / len(confidences) * gap
            mce = max(mce, gap)
        else:
            confidence = accuracy = 0.0
        rows.append({"bin": index, "count": count, "mean_confidence": confidence, "accuracy": accuracy})
    return {
        "mean_confidence": float(confidences.mean()),
        "ece_10": ece,
        "mce_10": mce,
        "bins": rows,
    }


@torch.inference_mode()
def evaluate_checkpoint(
    model,
    labels: list[str],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    batch_size: int,
    epsilon: float,
    trials: int,
    noise_seed: int,
) -> dict:
    rng = np.random.default_rng(noise_seed)
    clean_predictions, confidences = [], []
    perturbed_correct = 0
    total_perturbed = 0
    flip_counts = torch.zeros(len(inputs), dtype=torch.long)

    for start in range(0, len(inputs), batch_size):
        end = min(start + batch_size, len(inputs))
        batch = inputs[start:end].to(device)
        batch_targets = targets[start:end].to(device)
        probabilities = model(batch).softmax(dim=1)
        confidence, prediction = probabilities.max(dim=1)
        clean_predictions.append(prediction.cpu())
        confidences.append(confidence.cpu())

        for _ in range(trials):
            noise = rng.uniform(-epsilon, epsilon, size=tuple(batch.shape)).astype(np.float32)
            perturbed = batch + torch.from_numpy(noise).to(device)
            perturbed_prediction = model(perturbed).argmax(dim=1)
            flip_counts[start:end] += (perturbed_prediction != prediction).cpu()
            perturbed_correct += int((perturbed_prediction == batch_targets).sum())
            total_perturbed += len(batch_targets)

    predictions = torch.cat(clean_predictions)
    confidences = torch.cat(confidences)
    correct = predictions == targets
    clean_accuracy = float(correct.float().mean())
    macro_recall, macro_f1 = macro_metrics(targets, predictions, len(labels))
    eligible = correct
    eligible_flip_rate = float(flip_counts[eligible].sum() / (int(eligible.sum()) * trials))

    class_flip_rates = {}
    for index, label in enumerate(labels):
        mask = eligible & (targets == index)
        class_flip_rates[label] = (
            float(flip_counts[mask].sum() / (int(mask.sum()) * trials)) if int(mask.sum()) else None
        )

    return {
        "sample_count": len(inputs),
        "clean_accuracy": clean_accuracy,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "perturbation": {
            "epsilon": epsilon,
            "trials_per_sample": trials,
            "perturbed_accuracy": perturbed_correct / total_perturbed,
            "flip_rate_among_clean_correct": eligible_flip_rate,
            "clean_correct_samples_evaluated": int(eligible.sum()),
            "flip_rate_by_class": class_flip_rates,
        },
        "calibration": calibration_metrics(confidences, correct),
    }


@torch.inference_mode()
def benchmark_forward(model, inputs: torch.Tensor, device: torch.device, batch_size: int) -> dict:
    if len(inputs) < batch_size:
        repeats = int(np.ceil(batch_size / len(inputs)))
        inputs = inputs.repeat(repeats, 1, 1, 1)
    batch = inputs[:batch_size].to(device)
    warmup = 50 if batch_size == 1 else 20
    iterations = 300 if batch_size == 1 else 100
    for _ in range(warmup):
        model(batch)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            model(batch)
        end.record()
        torch.cuda.synchronize(device)
        elapsed_ms = start.elapsed_time(end)
    else:
        start_time = time.perf_counter()
        for _ in range(iterations):
            model(batch)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    latency_ms = elapsed_ms / iterations
    return {
        "batch_size": batch_size,
        "latency_ms_per_batch": latency_ms,
        "latency_ms_per_sample": latency_ms / batch_size,
        "throughput_samples_per_second": batch_size * 1000.0 / latency_ms,
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
    }


def model_efficiency(model, checkpoint_path: Path, inputs: torch.Tensor, device: torch.device) -> dict:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "parameters": parameters,
        "trainable_parameters": trainable,
        "parameter_memory_mib_fp32": parameters * 4 / (1024**2),
        "checkpoint_size_mib": checkpoint_path.stat().st_size / (1024**2),
        "batch_1": benchmark_forward(model, inputs, device, 1),
        "batch_32": benchmark_forward(model, inputs, device, 32),
    }


def mean_std(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "sample_std": float(array.std(ddof=1))}


def aggregate_effectiveness(rows: list[dict]) -> dict:
    metric_paths = {
        "clean_accuracy": lambda row: row["clean_accuracy"],
        "macro_f1": lambda row: row["macro_f1"],
        "perturbed_accuracy": lambda row: row["perturbation"]["perturbed_accuracy"],
        "flip_rate": lambda row: row["perturbation"]["flip_rate_among_clean_correct"],
        "mean_confidence": lambda row: row["calibration"]["mean_confidence"],
        "ece_10": lambda row: row["calibration"]["ece_10"],
    }
    return {name: mean_std([getter(row) for row in rows]) for name, getter in metric_paths.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate final no-SE concat AGCRN checkpoints")
    parser.add_argument("--data", type=Path, default=Path("data/shrec17_ddnet_npz"))
    parser.add_argument(
        "--agcrn-checkpoints",
        type=Path,
        nargs="+",
        default=[
            Path(f"runs/joint_aggregation_recheck_seed{seed}/04_stem_agcrn_control/best.pt")
            for seed in (42, 43, 44)
        ],
    )
    parser.add_argument(
        "--stgcn-checkpoints",
        type=Path,
        nargs="+",
        default=[
            Path(f"runs/se_semantic_ablation_seed{seed}/00_stem_control/best.pt")
            for seed in (42, 43, 44)
        ],
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("runs/agcrn_concat_effectiveness_20260902/report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    first_model, first_checkpoint = load_checkpoint(args.agcrn_checkpoints[0], device)
    samples = load_shrec17_npz(args.data, "test", int(first_checkpoint["frames"]), 14)
    inputs, targets = tensors_from_samples(samples, first_checkpoint["labels"])

    seed_results = []
    for index, checkpoint_path in enumerate(args.agcrn_checkpoints):
        if index == 0:
            model, checkpoint = first_model, first_checkpoint
        else:
            model, checkpoint = load_checkpoint(checkpoint_path, device)
        result = evaluate_checkpoint(
            model,
            checkpoint["labels"],
            inputs,
            targets,
            device,
            args.batch_size,
            args.epsilon,
            args.trials,
            20260902,
        )
        result["checkpoint"] = str(checkpoint_path)
        result["seed"] = int(checkpoint_path.parts[-3].split("seed")[-1])
        seed_results.append(result)

    aggregate = aggregate_effectiveness(seed_results)

    agcrn_efficiency = model_efficiency(first_model, args.agcrn_checkpoints[0], inputs, device)
    stgcn_seed_results = []
    baseline_model = baseline_checkpoint = baseline_inputs = None
    for index, checkpoint_path in enumerate(args.stgcn_checkpoints):
        model, checkpoint = load_checkpoint(checkpoint_path, device)
        model_inputs, model_targets = tensors_from_samples(samples, checkpoint["labels"])
        result = evaluate_checkpoint(
            model,
            checkpoint["labels"],
            model_inputs,
            model_targets,
            device,
            args.batch_size,
            args.epsilon,
            args.trials,
            20260902,
        )
        result["checkpoint"] = str(checkpoint_path)
        result["seed"] = int(checkpoint_path.parts[-3].split("seed")[-1])
        stgcn_seed_results.append(result)
        if index == 0:
            baseline_model, baseline_checkpoint, baseline_inputs = model, checkpoint, model_inputs
    stgcn_aggregate = aggregate_effectiveness(stgcn_seed_results)
    baseline_efficiency = model_efficiency(
        baseline_model, args.stgcn_checkpoints[0], baseline_inputs, device
    )

    report = {
        "model": "AGCRN no-SE concat joint-support",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dataset": "SHREC'17 official 14-class test split via numeric NPZ",
        "seed_results": seed_results,
        "aggregate": aggregate,
        "matched_stgcn_seed_results": stgcn_seed_results,
        "matched_stgcn_aggregate": stgcn_aggregate,
        "efficiency": {
            "agcrn_seed42": agcrn_efficiency,
            "matched_stem_stgcn_seed42": baseline_efficiency,
        },
        "notes": [
            "Perturbation is uniform coordinate noise in [-epsilon, epsilon] after standard sequence normalization.",
            "Flip rate follows the teammate protocol and is computed only for samples classified correctly before perturbation.",
            "Latency measures model forward only and excludes file loading and preprocessing.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "aggregate": aggregate,
                "matched_stgcn_aggregate": stgcn_aggregate,
                "efficiency": report["efficiency"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
