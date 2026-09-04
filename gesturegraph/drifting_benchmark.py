from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .drifting import (
    ONE_STEP_EXPERIMENTS,
    ConditionalDriftMemoryBank,
    build_one_step_model,
    conditional_categorical_drift_target,
    distillation_loss,
)
from .progressive import (
    DEFAULT_OBSERVATION_RATIOS,
    PrefixSequenceDataset,
    build_progressive_model,
    load_raw_shrec17_npz,
    stratified_raw_split,
)
from .progressive_benchmark import (
    calibrate_early_exit,
    choose_device,
    decision_metrics,
    error_recovery_metrics,
    evaluate_prefixes,
    measure_latency,
    measure_online_latency,
    set_reproducible,
)


BASELINE_EXPERIMENTS = (
    "04_four_step_teacher",
    "04_one_step_truncation",
)


def _load_teacher(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("experiment") != "04_class_diffusion":
        raise ValueError(f"teacher must be a 04_class_diffusion checkpoint: {path}")
    labels = list(checkpoint["labels"])
    model = build_progressive_model("04_class_diffusion", len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint, labels


def _evaluate_and_save(
    model,
    validation_loader,
    test_loader,
    device,
    ratios,
    output: Path,
    experiment: str,
    seed: int,
    epochs: int,
    parameters: int,
    latency_iterations: int,
    extra: dict | None = None,
):
    validation = evaluate_prefixes(model, validation_loader, device, ratios, collect=True)
    calibration = calibrate_early_exit(validation, ratios)
    test = evaluate_prefixes(model, test_loader, device, ratios, collect=True)
    test_decision = decision_metrics(
        test["probabilities"], test["targets"], ratios,
        calibration["confidence_threshold"], calibration["margin_threshold"],
    )
    recovery = error_recovery_metrics(test["probabilities"], test["targets"], ratios)
    latency = measure_latency(model, test_loader, device)
    online_latency = measure_online_latency(
        model, test_loader, device, iterations=latency_iterations
    )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "test_predictions.npz",
        probabilities=test["probabilities"],
        targets=test["targets"],
        ratios=np.asarray(ratios, dtype=np.float32),
    )
    test.pop("probabilities")
    test.pop("targets")
    validation.pop("probabilities")
    validation.pop("targets")
    result = {
        "experiment": experiment,
        "seed": int(seed),
        "epochs": int(epochs),
        "parameters": int(parameters),
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
    if extra:
        result.update(extra)
    (output / "model.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def train_student(
    args,
    experiment,
    teacher,
    train_loader,
    validation_loader,
    test_loader,
    labels,
    ratios,
    device,
):
    output = Path(args.output) / experiment
    output.mkdir(parents=True, exist_ok=True)
    model = build_one_step_model(len(labels), args.dropout).to(device)
    if args.init_from_teacher:
        model.load_state_dict(teacher.state_dict())
    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable_parameters)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    accumulation = max(1, math.ceil(args.batch_size / args.sequence_batch_size))
    bank = ConditionalDriftMemoryBank(args.bank_capacity)
    model.eval()
    initial_validation = evaluate_prefixes(model, validation_loader, device, ratios)
    best_score = initial_validation["prefix_auc"]
    history = []
    print(
        f"\n=== {experiment} | parameters {parameters:,} | "
        f"batch {args.sequence_batch_size} x accum {accumulation} ==="
    )
    torch.save({
        "model_state": model.state_dict(),
        "experiment": experiment,
        "labels": labels,
        "frames": args.frames,
        "observation_ratios": ratios,
        "selection_metric": "validation_prefix_auc",
        "best_validation_score": best_score,
        "best_epoch": 0,
        "seed": args.seed,
        "epochs": args.epochs,
        "parameters": parameters,
        "trainable_parameters": trainable_parameter_count,
        "one_step": True,
        "initialized_from_teacher": bool(args.init_from_teacher),
        "encoder_frozen": bool(args.freeze_encoder),
    }, output / "best.pt")
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_encoder:
            # Frozen BatchNorm statistics are part of the pretrained encoder.
            # Calling model.train() must not silently rewrite them.
            model.encoder.eval()
        sums = {"loss": 0.0, "primary": 0.0, "denoising": 0.0, "kd": 0.0, "drift": 0.0}
        seen = 0
        for batch_index, (views, lengths, progress, targets) in enumerate(train_loader):
            views = views.to(device)
            lengths = lengths.to(device)
            progress = progress.to(device)
            targets = targets.to(device)
            student_log, conditions = model(
                views, lengths, progress, return_auxiliary=True
            )
            repeated_targets = targets[:, None].expand(-1, len(ratios)).reshape(-1)
            primary = F.nll_loss(student_log.reshape(-1, len(labels)), repeated_targets)
            denoising = model.denoising_loss(conditions, targets)
            kd = torch.zeros((), device=device)
            drift = torch.zeros((), device=device)
            if experiment != "07_one_step_direct":
                with torch.no_grad():
                    teacher_log = teacher(views, lengths, progress)
                kd = distillation_loss(student_log, teacher_log, args.temperature)
                if experiment == "09_one_step_conditional_drift":
                    drift_target = conditional_categorical_drift_target(
                        student_log,
                        teacher_log.exp(),
                        conditions.detach(),
                        targets,
                        bank,
                        radii=tuple(args.drift_radii),
                        drift_strength=args.drift_strength,
                        context_weight=args.context_weight,
                    )
                    drift = F.kl_div(
                        student_log,
                        drift_target,
                        reduction="batchmean",
                    ) / student_log.shape[1]
            loss = primary + args.denoising_weight * denoising
            if experiment != "07_one_step_direct":
                loss = loss + args.distillation_weight * kd
            if experiment == "09_one_step_conditional_drift":
                loss = loss + args.drift_weight * drift

            (loss / accumulation).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            batch_count = len(targets)
            for key, value in (
                ("loss", loss), ("primary", primary), ("denoising", denoising),
                ("kd", kd), ("drift", drift),
            ):
                sums[key] += float(value.detach()) * batch_count
            seen += batch_count
        scheduler.step()
        validation = evaluate_prefixes(model, validation_loader, device, ratios)
        row = {
            "epoch": epoch,
            "train_loss": sums["loss"] / seen,
            "train_primary": sums["primary"] / seen,
            "train_denoising": sums["denoising"] / seen,
            "train_distillation": sums["kd"] / seen,
            "train_drift": sums["drift"] / seen,
            "val_prefix_auc": validation["prefix_auc"],
            "val_full_accuracy": validation["full_accuracy"],
            "val_ratio_accuracy": validation["ratio_accuracy"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | loss {row['train_loss']:.4f} | "
            f"val AUC {row['val_prefix_auc']:.1%} | full {row['val_full_accuracy']:.1%}"
        )
        if validation["prefix_auc"] >= best_score:
            best_score = validation["prefix_auc"]
            torch.save({
                "model_state": model.state_dict(),
                "experiment": experiment,
                "labels": labels,
                "frames": args.frames,
                "observation_ratios": ratios,
                "selection_metric": "validation_prefix_auc",
                "best_validation_score": best_score,
                "best_epoch": epoch,
                "seed": args.seed,
                "epochs": args.epochs,
                "parameters": parameters,
                "trainable_parameters": trainable_parameter_count,
                "one_step": True,
                "initialized_from_teacher": bool(args.init_from_teacher),
                "encoder_frozen": bool(args.freeze_encoder),
            }, output / "best.pt")

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    result = _evaluate_and_save(
        model, validation_loader, test_loader, device, ratios, output,
        experiment, args.seed, args.epochs, parameters, args.latency_iterations,
        extra={
            "training_objective": {
                "trainable_parameters": trainable_parameter_count,
                "denoising_weight": args.denoising_weight,
                "distillation_weight": (
                    args.distillation_weight if experiment != "07_one_step_direct" else 0.0
                ),
                "drift_weight": (
                    args.drift_weight if experiment == "09_one_step_conditional_drift" else 0.0
                ),
                "temperature": args.temperature,
                "drift_radii": list(args.drift_radii),
                "drift_strength": args.drift_strength,
                "context_weight": args.context_weight,
                "bank_capacity_per_class_ratio": args.bank_capacity,
                "initialized_from_teacher": bool(args.init_from_teacher),
                "encoder_frozen": bool(args.freeze_encoder),
            }
        },
    )
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(
        f"Official | AUC {result['official_test']['prefix_auc']:.2%} | "
        f"full {result['official_test']['full_accuracy']:.2%} | "
        f"ADR {result['official_test_early_exit']['average_decision_ratio']:.3f} | "
        f"{result['online_latency']['mean_update_ms']:.3f} ms/update"
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="One-step distillation and conditional categorical-drifting benchmark"
    )
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--data", default="data/shrec17_ddnet_npz")
    parser.add_argument("--output", default="runs/drifting_seed42")
    parser.add_argument("--experiments", nargs="+", choices=ONE_STEP_EXPERIMENTS,
                        default=list(ONE_STEP_EXPERIMENTS))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--denoising-weight", type=float, default=0.5)
    parser.add_argument("--distillation-weight", type=float, default=0.1)
    parser.add_argument("--drift-weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--drift-radii", type=float, nargs="+", default=[0.2, 0.5, 1.0])
    parser.add_argument("--drift-strength", type=float, default=0.25)
    parser.add_argument("--context-weight", type=float, default=0.25)
    parser.add_argument("--bank-capacity", type=int, default=64)
    parser.add_argument("--latency-iterations", type=int, default=100)
    parser.add_argument("--init-from-teacher", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--observation-ratios", type=float, nargs="+",
                        default=list(DEFAULT_OBSERVATION_RATIOS))
    args = parser.parse_args()

    device = choose_device(args.device)
    teacher, teacher_checkpoint, labels = _load_teacher(Path(args.teacher), device)
    teacher_seed = int(teacher_checkpoint["seed"])
    if args.seed is None:
        args.seed = teacher_seed
    if args.seed != teacher_seed:
        raise ValueError("student seed must match the teacher checkpoint seed")
    ratios = tuple(float(value) for value in args.observation_ratios)
    teacher_ratios = tuple(float(value) for value in teacher_checkpoint["observation_ratios"])
    if ratios != teacher_ratios:
        raise ValueError("student observation ratios must match the teacher checkpoint")
    if args.frames != int(teacher_checkpoint["frames"]):
        raise ValueError("student frame count must match the teacher checkpoint")

    set_reproducible(args.seed)
    official_train = load_raw_shrec17_npz(args.data, "train")
    test_samples = load_raw_shrec17_npz(args.data, "test")
    train_samples, val_samples = stratified_raw_split(
        official_train, args.val_ratio, args.seed
    )
    train_loader = DataLoader(
        PrefixSequenceDataset(train_samples, labels, args.frames, ratios, augment=True),
        batch_size=args.sequence_batch_size,
        shuffle=True,
        drop_last=True,
    )
    validation_loader = DataLoader(
        PrefixSequenceDataset(val_samples, labels, args.frames, ratios),
        batch_size=args.eval_batch_size,
    )
    test_loader = DataLoader(
        PrefixSequenceDataset(test_samples, labels, args.frames, ratios),
        batch_size=args.eval_batch_size,
    )
    print(
        f"Split: {len(train_samples)} train / {len(val_samples)} validation / "
        f"{len(test_samples)} official test | seed {args.seed} | device {device}"
    )

    results = []
    for experiment, steps in zip(BASELINE_EXPERIMENTS, (4, 1)):
        teacher.inference_steps = steps
        result = _evaluate_and_save(
            teacher, validation_loader, test_loader, device, ratios,
            Path(args.output) / experiment, experiment, args.seed,
            int(teacher_checkpoint["epochs"]),
            int(teacher_checkpoint["parameters"]), args.latency_iterations,
            extra={"source_checkpoint": str(Path(args.teacher).resolve()),
                   "inference_steps": steps, "training_performed": False},
        )
        results.append(result)
        print(
            f"Baseline {steps} step | AUC {result['official_test']['prefix_auc']:.2%} | "
            f"full {result['official_test']['full_accuracy']:.2%} | "
            f"{result['online_latency']['mean_update_ms']:.3f} ms/update"
        )
    teacher.inference_steps = teacher.steps

    for experiment in args.experiments:
        set_reproducible(args.seed)
        results.append(train_student(
            args, experiment, teacher, train_loader, validation_loader,
            test_loader, labels, ratios, device,
        ))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
