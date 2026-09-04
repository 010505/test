from __future__ import annotations

import argparse
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .progressive import (
    DEFAULT_OBSERVATION_RATIOS,
    EXTENDED_PROGRESSIVE_EXPERIMENTS,
    PROGRESSIVE_EXPERIMENTS,
    ClassDiffusionModel,
    ReliabilityGatedClassDiffusionModel,
    PrefixSequenceDataset,
    SingleViewDataset,
    build_progressive_model,
    load_raw_shrec17_npz,
    stratified_raw_split,
)


def set_reproducible(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(requested: str):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prefix_auc(ratios, accuracies):
    return float(np.trapezoid(accuracies, ratios) / (ratios[-1] - ratios[0]))


def evaluate_prefixes(model, loader, device, ratios, collect=False):
    model.eval()
    correct = np.zeros(len(ratios), dtype=np.int64)
    total = 0
    loss_sum = 0.0
    probabilities_all = []
    targets_all = []
    with torch.no_grad():
        for views, lengths, progress, targets in loader:
            views = views.to(device)
            lengths = lengths.to(device)
            progress = progress.to(device)
            targets = targets.to(device)
            outputs = model(views, lengths, progress)
            if isinstance(model, ClassDiffusionModel):
                loss = F.nll_loss(
                    outputs.reshape(-1, outputs.shape[-1]),
                    targets[:, None].expand(-1, outputs.shape[1]).reshape(-1),
                )
                probabilities = outputs.exp()
            else:
                loss = F.cross_entropy(
                    outputs.reshape(-1, outputs.shape[-1]),
                    targets[:, None].expand(-1, outputs.shape[1]).reshape(-1),
                )
                probabilities = outputs.softmax(dim=-1)
            predictions = probabilities.argmax(dim=-1)
            correct += (predictions == targets[:, None]).sum(dim=0).cpu().numpy()
            total += len(targets)
            loss_sum += float(loss) * len(targets)
            if collect:
                probabilities_all.append(probabilities.cpu())
                targets_all.append(targets.cpu())
    accuracies = correct / max(total, 1)
    result = {
        "loss": loss_sum / max(total, 1),
        "ratio_accuracy": {f"{ratio:.2f}": float(value) for ratio, value in zip(ratios, accuracies)},
        "prefix_auc": prefix_auc(np.asarray(ratios), accuracies),
        "full_accuracy": float(accuracies[-1]),
    }
    if collect:
        result["probabilities"] = torch.cat(probabilities_all).numpy()
        result["targets"] = torch.cat(targets_all).numpy()
    return result


def decision_metrics(probabilities, targets, ratios, confidence, margin):
    decisions = []
    decision_updates = []
    predictions = []
    for sample_probabilities in probabilities:
        selected = len(ratios) - 1
        for update in range(1, len(ratios)):
            current = sample_probabilities[update]
            previous = sample_probabilities[update - 1]
            order = np.argsort(current)[::-1]
            stable = int(order[0]) == int(previous.argmax())
            if stable and current[order[0]] >= confidence and current[order[0]] - current[order[1]] >= margin:
                selected = update
                break
        decisions.append(ratios[selected])
        decision_updates.append(selected + 1)
        predictions.append(int(sample_probabilities[selected].argmax()))
    return {
        "accuracy": float(np.mean(np.asarray(predictions) == targets)),
        "average_decision_ratio": float(np.mean(decisions)),
        "average_decision_updates": float(np.mean(decision_updates)),
    }


def error_recovery_metrics(probabilities, targets, ratios):
    """Measure whether samples misclassified at the first prefix later recover.

    This is an observational trajectory metric.  It supports claims about error
    correction, but it does not by itself prove which internal state caused the
    recovery.
    """
    probabilities = np.asarray(probabilities)
    targets = np.asarray(targets)
    ratios = np.asarray(ratios, dtype=np.float64)
    if probabilities.ndim != 3 or probabilities.shape[1] != len(ratios):
        raise ValueError("probabilities must have shape [samples, ratios, classes]")
    if probabilities.shape[0] != len(targets):
        raise ValueError("targets and probabilities must contain the same samples")
    predictions = probabilities.argmax(axis=-1)
    correct = predictions == targets[:, None]
    initial_wrong = ~correct[:, 0]
    initial_correct = correct[:, 0]
    wrong_count = int(initial_wrong.sum())
    correct_count = int(initial_correct.sum())

    def conditional_rate(mask, condition):
        denominator = int(condition.sum())
        return float(np.logical_and(mask, condition).sum() / denominator) if denominator else None

    point_recovery = {}
    ever_recovered = {}
    stable_recovery = {}
    for update, ratio in enumerate(ratios[1:], start=1):
        key = f"{ratio:.2f}"
        point_recovery[key] = conditional_rate(correct[:, update], initial_wrong)
        ever_recovered[key] = conditional_rate(correct[:, 1:update + 1].any(axis=1), initial_wrong)
        stable_recovery[key] = conditional_rate(correct[:, update:].all(axis=1), initial_wrong)

    transitions = {}
    for update in range(1, len(ratios)):
        key = f"{ratios[update - 1]:.2f}->{ratios[update]:.2f}"
        previous_wrong = ~correct[:, update - 1]
        previous_correct = correct[:, update - 1]
        transitions[key] = {
            "wrong_to_correct": conditional_rate(correct[:, update], previous_wrong),
            "correct_to_wrong": conditional_rate(~correct[:, update], previous_correct),
            "previous_wrong_samples": int(previous_wrong.sum()),
            "previous_correct_samples": int(previous_correct.sum()),
        }

    first_recovery_update = np.full(len(targets), -1, dtype=np.int64)
    for update in range(1, len(ratios)):
        newly_recovered = initial_wrong & (first_recovery_update < 0) & correct[:, update]
        first_recovery_update[newly_recovered] = update
    first_recovery_distribution = {
        f"{ratio:.2f}": (
            float((first_recovery_update[initial_wrong] == update).mean()) if wrong_count else None
        )
        for update, ratio in enumerate(ratios[1:], start=1)
    }
    first_recovery_distribution["never"] = (
        float((first_recovery_update[initial_wrong] < 0).mean()) if wrong_count else None
    )

    return {
        "definition": "conditioned on an incorrect prediction at the first observation ratio",
        "initial_ratio": float(ratios[0]),
        "initial_wrong_samples": wrong_count,
        "initial_correct_samples": correct_count,
        "point_recovery": point_recovery,
        "ever_recovered_by_ratio": ever_recovered,
        "stable_recovery_from_ratio": stable_recovery,
        "first_recovery_distribution": first_recovery_distribution,
        "final_recovery_rate": conditional_rate(correct[:, -1], initial_wrong),
        "initially_correct_final_retention": conditional_rate(correct[:, -1], initial_correct),
        "transitions": transitions,
    }


def calibrate_early_exit(validation, ratios):
    probabilities = validation["probabilities"]
    targets = validation["targets"]
    target_accuracy = validation["full_accuracy"] - 0.01
    candidates = []
    for confidence in np.arange(0.35, 0.91, 0.05):
        for margin in np.arange(0.05, 0.61, 0.05):
            metrics = decision_metrics(probabilities, targets, ratios, confidence, margin)
            if metrics["accuracy"] >= target_accuracy:
                candidates.append((metrics["average_decision_ratio"], -metrics["accuracy"], confidence, margin))
    if not candidates:
        confidence, margin = 1.01, 1.01
    else:
        _, _, confidence, margin = min(candidates)
    metrics = decision_metrics(probabilities, targets, ratios, confidence, margin)
    return {
        "confidence_threshold": float(confidence),
        "margin_threshold": float(margin),
        "target_accuracy": float(target_accuracy),
        **metrics,
    }


def measure_latency(model, loader, device):
    views, lengths, progress, _ = next(iter(loader))
    views = views[: min(8, len(views))].to(device)
    lengths = lengths[: len(views)].to(device)
    progress = progress[: len(views)].to(device)
    with torch.no_grad():
        for _ in range(3):
            model(views, lengths, progress)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        iterations = 10
        for _ in range(iterations):
            model(views, lengths, progress)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return elapsed * 1000.0 / (iterations * len(views) * views.shape[1])


def measure_online_latency(model, loader, device, iterations=20):
    """Measure a true batch-one stream of causal updates with persistent state."""
    views, lengths, progress, _ = next(iter(loader))
    views = views[:1].to(device)
    lengths = lengths[:1].to(device)
    progress = progress[:1].to(device)

    def run_sequence():
        state = None
        output = None
        for update in range(views.shape[1]):
            output, state = model.online_step(
                views[:, update], lengths[:, update], progress[:, update], state
            )
        return output

    with torch.no_grad():
        for _ in range(3):
            run_sequence()
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            run_sequence()
        if device.type == "cuda":
            torch.cuda.synchronize()
    sequence_ms = (time.perf_counter() - started) * 1000.0 / iterations
    return {
        "batch_size": 1,
        "updates": int(views.shape[1]),
        "sequence_ms": sequence_ms,
        "mean_update_ms": sequence_ms / views.shape[1],
    }


def train_experiment(args, experiment, train_samples, val_samples, test_samples, labels, device):
    output = Path(args.output) / experiment
    output.mkdir(parents=True, exist_ok=True)
    ratios = tuple(args.observation_ratios)
    sequence_training = experiment in {
        "03_gru_evidence", "04_class_diffusion", "05_gated_class_diffusion",
        "06_reliability_gated_diffusion",
    }
    if sequence_training:
        train_dataset = PrefixSequenceDataset(train_samples, labels, args.frames, ratios, augment=True)
        batch_size = args.sequence_batch_size
        accumulation = max(1, math.ceil(args.batch_size / batch_size))
    else:
        mode = {
            "00_full_sequence": "full",
            "01_mild_temporal_crop": "mild",
            "02_causal_prefix": "causal",
        }[experiment]
        train_dataset = SingleViewDataset(train_samples, labels, mode, args.frames, ratios, augment=True)
        batch_size = args.batch_size
        accumulation = 1
    validation_dataset = PrefixSequenceDataset(val_samples, labels, args.frames, ratios)
    test_dataset = PrefixSequenceDataset(test_samples, labels, args.frames, ratios)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=sequence_training,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=args.eval_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size)

    model = build_progressive_model(experiment, len(labels), args.dropout).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"\n=== {experiment} | parameters {parameters:,} | batch {batch_size} x accum {accumulation} ===")
    history = []
    selection_name = (
        "validation_full_accuracy"
        if experiment in {"00_full_sequence", "01_mild_temporal_crop"}
        else "validation_prefix_auc"
    )
    best_score = -1.0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch_index, (views, lengths, progress, targets) in enumerate(train_loader):
            views = views.to(device)
            lengths = lengths.to(device)
            progress = progress.to(device)
            targets = targets.to(device)
            if sequence_training:
                if isinstance(model, ReliabilityGatedClassDiffusionModel):
                    outputs = model(
                        views,
                        lengths,
                        progress,
                        return_auxiliary=True,
                        return_diagnostics=True,
                    )
                else:
                    outputs = model(views, lengths, progress, return_auxiliary=True) if isinstance(model, ClassDiffusionModel) else model(views, lengths, progress)
                if isinstance(model, ClassDiffusionModel):
                    if isinstance(model, ReliabilityGatedClassDiffusionModel):
                        log_probabilities, conditions, diagnostics = outputs
                    else:
                        log_probabilities, conditions = outputs
                    primary = F.nll_loss(
                        log_probabilities.reshape(-1, len(labels)),
                        targets[:, None].expand(-1, len(ratios)).reshape(-1),
                    )
                    auxiliary = model.denoising_loss(conditions, targets)
                    if isinstance(model, ReliabilityGatedClassDiffusionModel):
                        auxiliary = auxiliary + 0.5 * model.reliability_loss(
                            diagnostics, targets
                        )
                    loss = primary + 0.5 * auxiliary
                else:
                    loss = F.cross_entropy(
                        outputs.reshape(-1, len(labels)),
                        targets[:, None].expand(-1, len(ratios)).reshape(-1),
                    )
            else:
                outputs = model(views, lengths, progress)
                loss = F.cross_entropy(outputs, targets)
            (loss / accumulation).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach()) * len(targets)
            seen += len(targets)
        scheduler.step()
        validation = evaluate_prefixes(model, validation_loader, device, ratios)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "val_prefix_auc": validation["prefix_auc"],
            "val_full_accuracy": validation["full_accuracy"],
            "val_ratio_accuracy": validation["ratio_accuracy"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | loss {row['train_loss']:.4f} | "
            f"val AUC {row['val_prefix_auc']:.1%} | full {row['val_full_accuracy']:.1%}"
        )
        selection_score = (
            validation["full_accuracy"]
            if selection_name == "validation_full_accuracy"
            else validation["prefix_auc"]
        )
        if selection_score >= best_score:
            best_score = selection_score
            torch.save({
                "model_state": model.state_dict(),
                "experiment": experiment,
                "labels": labels,
                "frames": args.frames,
                "observation_ratios": ratios,
                "selection_metric": selection_name,
                "best_validation_score": best_score,
                "seed": args.seed,
                "epochs": args.epochs,
                "parameters": parameters,
            }, output / "best.pt")

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    validation = evaluate_prefixes(model, validation_loader, device, ratios, collect=True)
    calibration = calibrate_early_exit(validation, ratios)
    test = evaluate_prefixes(model, test_loader, device, ratios, collect=True)
    test_decision = decision_metrics(
        test["probabilities"], test["targets"], ratios,
        calibration["confidence_threshold"], calibration["margin_threshold"],
    )
    recovery = error_recovery_metrics(test["probabilities"], test["targets"], ratios)
    latency = measure_latency(model, test_loader, device)
    online_latency = measure_online_latency(model, test_loader, device)
    test.pop("probabilities")
    test.pop("targets")
    validation.pop("probabilities")
    validation.pop("targets")
    result = {
        "experiment": experiment,
        "seed": args.seed,
        "epochs": args.epochs,
        "effective_batch_size": args.batch_size,
        "parameters": parameters,
        "selection_metric": selection_name,
        "validation": validation,
        "official_test": test,
        "early_exit_calibration": calibration,
        "official_test_early_exit": test_decision,
        "official_test_error_recovery": recovery,
        "latency_ms_per_sample_update": latency,
        "online_latency": online_latency,
        "device": str(device),
        "python_version": platform.python_version(),
    }
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "model.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"Official | AUC {test['prefix_auc']:.1%} | full {test['full_accuracy']:.1%} | "
        f"ADR {test_decision['average_decision_ratio']:.3f} | decision {test_decision['accuracy']:.1%}"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Progressive causal-prefix and class-diffusion benchmark")
    parser.add_argument("--data", default="data/shrec17_ddnet_npz")
    parser.add_argument("--output", default="runs/progressive_seed42")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXTENDED_PROGRESSIVE_EXPERIMENTS,
        default=list(PROGRESSIVE_EXPERIMENTS),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--observation-ratios", type=float, nargs="+", default=list(DEFAULT_OBSERVATION_RATIOS))
    args = parser.parse_args()

    set_reproducible(args.seed)
    ratios = tuple(args.observation_ratios)
    if sorted(ratios) != list(ratios) or ratios[-1] != 1.0 or any(value <= 0 or value > 1 for value in ratios):
        raise ValueError("observation ratios must be sorted, positive and end at 1.0")
    official_train = load_raw_shrec17_npz(args.data, "train")
    test_samples = load_raw_shrec17_npz(args.data, "test")
    train_samples, val_samples = stratified_raw_split(official_train, args.val_ratio, args.seed)
    labels = sorted({sample.label for sample in official_train + test_samples})
    device = choose_device(args.device)
    print(f"Split: {len(train_samples)} train / {len(val_samples)} validation / {len(test_samples)} official test")
    print(f"Ratios: {ratios} | device: {device}")
    results = []
    for experiment in args.experiments:
        set_reproducible(args.seed)
        results.append(train_experiment(
            args, experiment, train_samples, val_samples, test_samples, labels, device
        ))
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
