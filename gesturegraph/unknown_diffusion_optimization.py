from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .drifting import OneStepClassDiffusionModel, distillation_loss
from .progressive import (
    DEFAULT_OBSERVATION_RATIOS,
    ClassDiffusionModel,
    load_raw_shrec17_npz,
    stratified_raw_split,
)
from .progressive_benchmark import prefix_auc, set_reproducible
from .unknown_diffusion_benchmark import (
    TEST_UNKNOWN_MODES,
    TRAIN_UNKNOWN_MODES,
    UNKNOWN_LABEL,
    encode_samples,
    expand_teacher_to_unknown,
    load_base_teacher,
    make_unknown_samples,
    metric_auc,
    stage_metrics,
    trainable_denoiser_parameters,
)


def forward_conditions_with_no_retention(
    model: ClassDiffusionModel,
    conditions: torch.Tensor,
    no_retention: float = 1.0,
) -> torch.Tensor:
    """Run Markov updates while optionally weakening only the inherited No mass."""
    if not 0.0 <= no_retention <= 1.0:
        raise ValueError("no_retention must be in [0, 1]")
    uniform = torch.full(
        (len(conditions), model.num_classes),
        1.0 / model.num_classes,
        device=conditions.device,
        dtype=conditions.dtype,
    )
    previous = uniform
    outputs = []
    for update in range(conditions.shape[1]):
        if update == 0:
            initial = uniform
        else:
            inherited = previous
            if model.num_classes == 15 and no_retention != 1.0:
                inherited = previous.clone()
                inherited[:, -1] = inherited[:, -1] * no_retention
                inherited = inherited / inherited.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            initial = model.inheritance * inherited + (1.0 - model.inheritance) * uniform
        previous = model._reverse_distribution(conditions[:, update], initial)
        outputs.append(previous)
    return torch.stack(outputs, dim=1).clamp_min(1e-8).log()


def closed_set_preservation_losses(
    log_probabilities: torch.Tensor,
    old_teacher_log_probabilities: torch.Tensor,
    targets: torch.Tensor,
    margin: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Protect the old 14-way boundary and separate known/No in both directions."""
    if log_probabilities.shape[-1] != 15 or old_teacher_log_probabilities.shape[-1] != 14:
        raise ValueError("preservation requires 15-state student and 14-state teacher")
    zero = log_probabilities.sum() * 0.0
    known_mask = targets < 14
    unknown_mask = ~known_mask
    closed_kd = zero
    known_margin = zero
    unknown_margin = zero
    if known_mask.any():
        known_log = log_probabilities[known_mask]
        old_log = old_teacher_log_probabilities[known_mask]
        closed_kd = distillation_loss(
            F.log_softmax(known_log[..., :14], dim=-1), old_log, temperature
        )
        known_targets = targets[known_mask, None, None].expand(-1, known_log.shape[1], 1)
        correct = known_log[..., :14].gather(-1, known_targets).squeeze(-1)
        known_margin = F.relu(float(margin) + known_log[..., 14] - correct).mean()
    if unknown_mask.any():
        unknown_log = log_probabilities[unknown_mask]
        best_known = unknown_log[..., :14].max(dim=-1).values
        unknown_margin = F.relu(
            float(margin) + best_known - unknown_log[..., 14]
        ).mean()
    return closed_kd, known_margin, unknown_margin


@torch.no_grad()
def predict_conditions(
    model: ClassDiffusionModel,
    conditions: torch.Tensor,
    batch_size: int,
    device: torch.device,
    no_retention: float = 1.0,
    no_biases: Sequence[float] | None = None,
) -> np.ndarray:
    model.eval()
    outputs = []
    bias_tensor = None
    if no_biases is not None:
        if len(no_biases) != conditions.shape[1]:
            raise ValueError("one No bias is required per observation")
        bias_tensor = torch.as_tensor(no_biases, dtype=torch.float32, device=device)
    for start in range(0, len(conditions), batch_size):
        batch = conditions[start:start + batch_size].to(device)
        log_probabilities = forward_conditions_with_no_retention(
            model, batch, no_retention
        )
        if bias_tensor is not None:
            adjusted = log_probabilities.clone()
            adjusted[..., 14] += bias_tensor[None, :]
            log_probabilities = F.log_softmax(adjusted, dim=-1)
        outputs.append(log_probabilities.exp().cpu())
    return torch.cat(outputs).numpy()


def evaluate_open_set(
    model: ClassDiffusionModel,
    known: tuple[torch.Tensor, torch.Tensor],
    unknown_by_type: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ratios: Sequence[float],
    batch_size: int,
    device: torch.device,
    no_retention: float = 1.0,
    no_biases: Sequence[float] | None = None,
) -> dict:
    known_conditions, known_targets = known
    known_probabilities = predict_conditions(
        model, known_conditions, batch_size, device, no_retention, no_biases
    )
    pooled_unknown = []
    per_type = {}
    for mode, (conditions, _) in unknown_by_type.items():
        probabilities = predict_conditions(
            model, conditions, batch_size, device, no_retention, no_biases
        )
        pooled_unknown.append(probabilities)
        rows = stage_metrics(known_probabilities, known_targets.numpy(), probabilities, ratios)
        per_type[mode] = {
            "stage_metrics": [asdict(row) for row in rows],
            "unknown_recall_auc": metric_auc(rows, "unknown_recall"),
            "final_unknown_recall": rows[-1].unknown_recall,
        }
    rows = stage_metrics(
        known_probabilities,
        known_targets.numpy(),
        np.concatenate(pooled_unknown, axis=0),
        ratios,
    )
    return {
        "stage_metrics": [asdict(row) for row in rows],
        "known_accuracy_auc": metric_auc(rows, "known_accuracy"),
        "known_false_no_auc": metric_auc(rows, "known_false_no_rate"),
        "unknown_recall_auc": metric_auc(rows, "unknown_recall"),
        "balanced_accuracy_auc": metric_auc(rows, "balanced_accuracy"),
        "final_known_accuracy": rows[-1].known_accuracy,
        "final_known_false_no_rate": rows[-1].known_false_no_rate,
        "final_unknown_recall": rows[-1].unknown_recall,
        "per_type": per_type,
    }


@torch.no_grad()
def old_teacher_predictions(
    model: ClassDiffusionModel,
    conditions: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs = []
    model.eval()
    for start in range(0, len(conditions), batch_size):
        outputs.append(
            forward_conditions_with_no_retention(
                model, conditions[start:start + batch_size].to(device)
            ).cpu()
        )
    return torch.cat(outputs)


def old_known_metrics(
    old_log_probabilities: torch.Tensor,
    targets: torch.Tensor,
    ratios: Sequence[float],
) -> dict:
    predictions = old_log_probabilities.argmax(dim=-1).numpy()
    target_array = targets.numpy()
    stage_accuracy = [
        float(np.mean(predictions[:, update] == target_array))
        for update in range(len(ratios))
    ]
    return {
        "stage_accuracy": stage_accuracy,
        "known_accuracy_auc": prefix_auc(
            np.asarray(ratios, dtype=np.float64), np.asarray(stage_accuracy)
        ),
        "final_known_accuracy": stage_accuracy[-1],
    }


def constrained_score(metrics: dict, old_known_auc: float, args) -> float:
    known_shortfall = max(
        0.0,
        old_known_auc - args.max_known_drop - metrics["known_accuracy_auc"],
    )
    false_no_excess = max(
        0.0, metrics["known_false_no_auc"] - args.max_false_no
    )
    return float(
        metrics["balanced_accuracy_auc"]
        - args.constraint_penalty * (known_shortfall + false_no_excess)
    )


def save_checkpoint(
    model: ClassDiffusionModel,
    destination: Path,
    checkpoint: dict,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), **checkpoint}, destination)


def train_protected_model(
    model: ClassDiffusionModel,
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    validation_known: tuple[torch.Tensor, torch.Tensor],
    validation_unknown: dict[str, tuple[torch.Tensor, torch.Tensor]],
    old_validation: dict,
    labels: Sequence[str],
    ratios: Sequence[float],
    args,
    output: Path,
    device: torch.device,
    experiment: str,
    teacher_15: ClassDiffusionModel | None = None,
) -> tuple[ClassDiffusionModel, list[dict]]:
    model = model.to(device)
    if teacher_15 is not None:
        teacher_15.eval()
    optimizer = torch.optim.AdamW(
        trainable_denoiser_parameters(model), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loader = DataLoader(TensorDataset(*train_data), batch_size=args.batch_size, shuffle=True)
    history = []
    best_score = -float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "primary": 0.0,
            "denoising": 0.0,
            "closed_kd": 0.0,
            "known_margin": 0.0,
            "unknown_margin": 0.0,
            "student_kd": 0.0,
        }
        seen = 0
        for conditions, targets, old_log in loader:
            conditions = conditions.to(device)
            targets = targets.to(device)
            old_log = old_log.to(device)
            log_probabilities = forward_conditions_with_no_retention(model, conditions)
            repeated = targets[:, None].expand(-1, conditions.shape[1]).reshape(-1)
            primary = F.nll_loss(log_probabilities.reshape(-1, 15), repeated)
            denoising = model.denoising_loss(conditions, targets)
            closed_kd, known_margin, unknown_margin = closed_set_preservation_losses(
                log_probabilities, old_log, targets, args.margin, args.temperature
            )
            student_kd = log_probabilities.sum() * 0.0
            if teacher_15 is not None:
                with torch.no_grad():
                    teacher_log = forward_conditions_with_no_retention(
                        teacher_15, conditions
                    )
                student_kd = distillation_loss(
                    log_probabilities, teacher_log, args.temperature
                )
            loss = (
                primary
                + args.denoising_weight * denoising
                + args.closed_kd_weight * closed_kd
                + args.known_margin_weight * known_margin
                + args.unknown_margin_weight * unknown_margin
                + args.student_kd_weight * student_kd
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            for name, value in (
                ("loss", loss),
                ("primary", primary),
                ("denoising", denoising),
                ("closed_kd", closed_kd),
                ("known_margin", known_margin),
                ("unknown_margin", unknown_margin),
                ("student_kd", student_kd),
            ):
                totals[name] += float(value.detach()) * len(targets)
            seen += len(targets)
        scheduler.step()
        validation = evaluate_open_set(
            model,
            validation_known,
            validation_unknown,
            ratios,
            args.eval_batch_size,
            device,
        )
        score = constrained_score(validation, old_validation["known_accuracy_auc"], args)
        row = {
            "epoch": epoch,
            **{f"train_{name}": value / seen for name, value in totals.items()},
            "selection_score": score,
            "validation": validation,
        }
        history.append(row)
        print(
            f"{experiment} epoch {epoch:03d} | loss {row['train_loss']:.4f} | "
            f"score {score:.2%} | known {validation['known_accuracy_auc']:.2%} | "
            f"false No {validation['known_false_no_auc']:.2%} | "
            f"No {validation['unknown_recall_auc']:.2%}"
        )
        if score >= best_score:
            best_score = score
            save_checkpoint(
                model,
                output / "best.pt",
                {
                    "experiment": experiment,
                    "labels": list(labels),
                    "frames": args.frames,
                    "observation_ratios": list(ratios),
                    "seed": args.seed,
                    "epochs": epoch,
                    "selection_metric": "constrained_validation_open_set_score",
                    "best_validation_score": score,
                    "unknown_label": UNKNOWN_LABEL,
                    "unknown_available_at_every_observation": True,
                    "closed_kd_weight": args.closed_kd_weight,
                    "known_margin_weight": args.known_margin_weight,
                    "unknown_margin_weight": args.unknown_margin_weight,
                    "margin": args.margin,
                },
            )
    best = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model_state"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model, history


def collect_validation_arrays(
    model: ClassDiffusionModel,
    known: tuple[torch.Tensor, torch.Tensor],
    unknown_by_type: dict[str, tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    device: torch.device,
    no_retention: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    known_probabilities = predict_conditions(
        model, known[0], batch_size, device, no_retention
    )
    unknown_probabilities = np.concatenate([
        predict_conditions(model, data[0], batch_size, device, no_retention)
        for data in unknown_by_type.values()
    ])
    return known_probabilities, known[1].numpy(), unknown_probabilities


def select_no_retention(
    model: ClassDiffusionModel,
    validation_known: tuple[torch.Tensor, torch.Tensor],
    validation_unknown: dict[str, tuple[torch.Tensor, torch.Tensor]],
    old_validation: dict,
    ratios: Sequence[float],
    args,
    device: torch.device,
) -> tuple[float, list[dict]]:
    rows = []
    for retention in args.no_retention_candidates:
        metrics = evaluate_open_set(
            model,
            validation_known,
            validation_unknown,
            ratios,
            args.eval_batch_size,
            device,
            retention,
        )
        score = constrained_score(metrics, old_validation["known_accuracy_auc"], args)
        rows.append({"no_retention": retention, "selection_score": score, "metrics": metrics})
    best = max(rows, key=lambda row: (row["selection_score"], -row["no_retention"]))
    return float(best["no_retention"]), rows


def calibrate_no_biases(
    known_probabilities: np.ndarray,
    known_targets: np.ndarray,
    unknown_probabilities: np.ndarray,
    old_stage_accuracy: Sequence[float],
    max_known_drop: float,
    max_false_no: float,
    bias_grid: Sequence[float],
) -> tuple[list[float], list[dict]]:
    biases = []
    diagnostics = []
    for update, old_accuracy in enumerate(old_stage_accuracy):
        candidates = []
        for bias in bias_grid:
            known_log = np.log(np.clip(known_probabilities[:, update], 1e-8, 1.0))
            unknown_log = np.log(np.clip(unknown_probabilities[:, update], 1e-8, 1.0))
            known_log[:, 14] += float(bias)
            unknown_log[:, 14] += float(bias)
            known_prediction = known_log.argmax(axis=-1)
            unknown_prediction = unknown_log.argmax(axis=-1)
            known_accuracy = float(np.mean(known_prediction == known_targets))
            false_no = float(np.mean(known_prediction == 14))
            unknown_recall = float(np.mean(unknown_prediction == 14))
            feasible = (
                known_accuracy >= old_accuracy - max_known_drop
                and false_no <= max_false_no
            )
            penalty = (
                max(0.0, old_accuracy - max_known_drop - known_accuracy)
                + max(0.0, false_no - max_false_no)
            )
            candidates.append({
                "bias": float(bias),
                "known_accuracy": known_accuracy,
                "known_false_no_rate": false_no,
                "unknown_recall": unknown_recall,
                "balanced_accuracy": 0.5 * (known_accuracy + unknown_recall),
                "feasible": feasible,
                "fallback_score": 0.5 * (known_accuracy + unknown_recall) - 2.0 * penalty,
            })
        feasible = [candidate for candidate in candidates if candidate["feasible"]]
        if feasible:
            selected = max(
                feasible,
                key=lambda row: (row["unknown_recall"], row["balanced_accuracy"], -abs(row["bias"])),
            )
        else:
            selected = max(candidates, key=lambda row: row["fallback_score"])
        biases.append(selected["bias"])
        diagnostics.append(selected)
    return biases, diagnostics


def write_report(output: Path, summary: dict) -> None:
    lines = [
        "# Closed-set-preserving per-step No experiment",
        "",
        "Every observation still selects from the same 15-state posterior. B adds old",
        "14-class preservation and bidirectional margins; C weakens inherited No mass;",
        "D calibrates one No logit bias per observation on validation data.",
        "",
        "| Variant | Known AUC | False-No AUC | No AUC | Balanced AUC | Final known | Final false No | Final No recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("A_current", "B_protected", "C_weak_no", "D_calibrated"):
        row = summary[name]
        lines.append(
            f"| {name} | {row['known_accuracy_auc']:.2%} | {row['known_false_no_auc']:.2%} | "
            f"{row['unknown_recall_auc']:.2%} | {row['balanced_accuracy_auc']:.2%} | "
            f"{row['final_known_accuracy']:.2%} | {row['final_known_false_no_rate']:.2%} | "
            f"{row['final_unknown_recall']:.2%} |"
        )
    lines.extend([
        "",
        f"Selected No retention: `{summary['selected_no_retention']}`.",
        f"Selected stage biases: `{summary['selected_no_biases']}`.",
        "",
        "D remains a 15-way argmax at every step; calibration does not add a rejection head.",
    ])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preserve 14-class decisions while adding per-step No")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--data", default="data/shrec17_ddnet_npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--encoder-batch-size", type=int, default=16)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--unknown-train-ratio", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--denoising-weight", type=float, default=0.5)
    parser.add_argument("--student-kd-weight", type=float, default=0.1)
    parser.add_argument("--closed-kd-weight", type=float, default=2.0)
    parser.add_argument("--known-margin-weight", type=float, default=1.0)
    parser.add_argument("--unknown-margin-weight", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--max-known-drop", type=float, default=0.02)
    parser.add_argument("--max-false-no", type=float, default=0.05)
    parser.add_argument("--constraint-penalty", type=float, default=2.0)
    parser.add_argument("--no-retention-candidates", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--bias-min", type=float, default=-6.0)
    parser.add_argument("--bias-max", type=float, default=2.0)
    parser.add_argument("--bias-step", type=float, default=0.05)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--observation-ratios", type=float, nargs="+", default=list(DEFAULT_OBSERVATION_RATIOS))
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    base_teacher, base_checkpoint = load_base_teacher(Path(args.teacher), device)
    teacher_seed = int(base_checkpoint["seed"])
    args.seed = teacher_seed if args.seed is None else args.seed
    if args.seed != teacher_seed:
        raise ValueError("experiment seed must match source teacher")
    if not 0.0 < args.unknown_train_ratio <= 1.0:
        raise ValueError("unknown-train-ratio must be in (0, 1]")
    if args.bias_step <= 0 or args.bias_min > args.bias_max:
        raise ValueError("invalid bias grid")
    if any(not 0.0 <= value <= 1.0 for value in args.no_retention_candidates):
        raise ValueError("No retention candidates must be in [0, 1]")
    set_reproducible(args.seed)
    random.seed(args.seed)
    ratios = tuple(float(value) for value in args.observation_ratios)
    labels = list(base_checkpoint["labels"]) + [UNKNOWN_LABEL]

    official_train = load_raw_shrec17_npz(args.data, "train")
    known_test_samples = load_raw_shrec17_npz(args.data, "test")
    known_train, known_validation_samples = stratified_raw_split(
        official_train, args.val_ratio, args.seed
    )
    unknown_pool = make_unknown_samples(known_train, TRAIN_UNKNOWN_MODES, args.seed + 101)
    unknown_count = max(1, round(len(known_train) * args.unknown_train_ratio))
    selection_rng = np.random.default_rng(args.seed + 102)
    selected = selection_rng.choice(len(unknown_pool), size=unknown_count, replace=False)
    unknown_train = [unknown_pool[int(index)] for index in selected]
    unknown_validation_samples = {
        mode: make_unknown_samples(known_validation_samples, (mode,), args.seed + 200 + index)
        for index, mode in enumerate(TRAIN_UNKNOWN_MODES)
    }
    unknown_test_samples = {
        mode: make_unknown_samples(known_test_samples, (mode,), args.seed + 300 + index)
        for index, mode in enumerate(TEST_UNKNOWN_MODES)
    }

    print(
        f"Seed {args.seed} | device {device} | known train/val/test "
        f"{len(known_train)}/{len(known_validation_samples)}/{len(known_test_samples)} | "
        f"Unknown train {len(unknown_train)}"
    )
    train_conditions, train_targets = encode_samples(
        base_teacher,
        list(known_train) + unknown_train,
        labels,
        args.frames,
        ratios,
        args.encoder_batch_size,
        device,
    )
    validation_known = encode_samples(
        base_teacher, known_validation_samples, labels, args.frames, ratios,
        args.encoder_batch_size, device,
    )
    validation_unknown = {
        mode: encode_samples(
            base_teacher, samples, labels, args.frames, ratios,
            args.encoder_batch_size, device,
        )
        for mode, samples in unknown_validation_samples.items()
    }
    test_known = encode_samples(
        base_teacher, known_test_samples, labels, args.frames, ratios,
        args.encoder_batch_size, device,
    )
    test_unknown = {
        mode: encode_samples(
            base_teacher, samples, labels, args.frames, ratios,
            args.encoder_batch_size, device,
        )
        for mode, samples in unknown_test_samples.items()
    }
    train_old_log = old_teacher_predictions(
        base_teacher, train_conditions, args.eval_batch_size, device
    )
    validation_old_log = old_teacher_predictions(
        base_teacher, validation_known[0], args.eval_batch_size, device
    )
    test_old_log = old_teacher_predictions(
        base_teacher, test_known[0], args.eval_batch_size, device
    )
    old_validation = old_known_metrics(validation_old_log, validation_known[1], ratios)
    old_test = old_known_metrics(test_old_log, test_known[1], ratios)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protected_teacher, _ = train_protected_model(
        expand_teacher_to_unknown(base_teacher),
        (train_conditions, train_targets, train_old_log),
        validation_known,
        validation_unknown,
        old_validation,
        labels,
        ratios,
        args,
        output / "01_protected_teacher",
        device,
        "15_state_protected_four_step_teacher",
    )
    protected_student = OneStepClassDiffusionModel(num_classes=15, dropout=args.dropout)
    protected_student.load_state_dict(protected_teacher.state_dict())
    protected_student, _ = train_protected_model(
        protected_student,
        (train_conditions, train_targets, train_old_log),
        validation_known,
        validation_unknown,
        old_validation,
        labels,
        ratios,
        args,
        output / "02_protected_one_step",
        device,
        "15_state_protected_one_step_student",
        protected_teacher,
    )

    selected_retention, retention_sweep = select_no_retention(
        protected_student,
        validation_known,
        validation_unknown,
        old_validation,
        ratios,
        args,
        device,
    )
    known_probabilities, known_targets, unknown_probabilities = collect_validation_arrays(
        protected_student,
        validation_known,
        validation_unknown,
        args.eval_batch_size,
        device,
        selected_retention,
    )
    bias_grid = np.arange(
        args.bias_min, args.bias_max + args.bias_step * 0.5, args.bias_step
    )
    selected_biases, bias_diagnostics = calibrate_no_biases(
        known_probabilities,
        known_targets,
        unknown_probabilities,
        old_validation["stage_accuracy"],
        args.max_known_drop,
        args.max_false_no,
        bias_grid,
    )

    common_variant_checkpoint = {
        "labels": labels,
        "frames": args.frames,
        "observation_ratios": list(ratios),
        "seed": args.seed,
        "epochs": int(torch.load(
            output / "02_protected_one_step" / "best.pt",
            map_location="cpu",
            weights_only=True,
        )["epochs"]),
        "unknown_label": UNKNOWN_LABEL,
        "unknown_available_at_every_observation": True,
        "source_checkpoint": "02_protected_one_step/best.pt",
    }
    save_checkpoint(
        protected_student,
        output / "03_weak_no_inheritance" / "best.pt",
        {
            **common_variant_checkpoint,
            "experiment": "15_state_weak_no_inheritance",
            "no_retention": selected_retention,
            "no_biases": [0.0] * len(ratios),
        },
    )
    save_checkpoint(
        protected_student,
        output / "04_stage_calibrated" / "best.pt",
        {
            **common_variant_checkpoint,
            "experiment": "15_state_stage_calibrated_no",
            "no_retention": selected_retention,
            "no_biases": selected_biases,
        },
    )

    baseline_summary = json.loads(Path(args.baseline_summary).read_text(encoding="utf-8"))
    summary = {
        "seed": args.seed,
        "epochs": args.epochs,
        "unknown_train_ratio": args.unknown_train_ratio,
        "labels": labels,
        "unknown_available_at_every_observation": True,
        "old_14_validation": old_validation,
        "old_14_test": old_test,
        "selected_no_retention": selected_retention,
        "retention_sweep": retention_sweep,
        "selected_no_biases": selected_biases,
        "bias_diagnostics": bias_diagnostics,
        "A_current": baseline_summary["one_step_distilled"],
        "B_protected": evaluate_open_set(
            protected_student, test_known, test_unknown, ratios,
            args.eval_batch_size, device,
        ),
        "C_weak_no": evaluate_open_set(
            protected_student, test_known, test_unknown, ratios,
            args.eval_batch_size, device, selected_retention,
        ),
        "D_calibrated": evaluate_open_set(
            protected_student, test_known, test_unknown, ratios,
            args.eval_batch_size, device, selected_retention, selected_biases,
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "03_no_retention_sweep.json").write_text(
        json.dumps(retention_sweep, indent=2), encoding="utf-8"
    )
    (output / "04_stage_bias_calibration.json").write_text(
        json.dumps({"biases": selected_biases, "diagnostics": bias_diagnostics}, indent=2),
        encoding="utf-8",
    )
    write_report(output, summary)
    print(f"Wrote {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
