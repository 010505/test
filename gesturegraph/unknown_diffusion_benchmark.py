from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .drifting import (
    ConditionalDriftMemoryBank,
    OneStepClassDiffusionModel,
    conditional_categorical_drift_target,
    distillation_loss,
)
from .progressive import (
    DEFAULT_OBSERVATION_RATIOS,
    ClassDiffusionModel,
    PrefixSequenceDataset,
    RawGestureSample,
    build_progressive_model,
    load_raw_shrec17_npz,
    stratified_raw_split,
)
from .progressive_benchmark import prefix_auc, set_reproducible


UNKNOWN_LABEL = "unknown"
TRAIN_UNKNOWN_MODES = ("frozen", "shuffled", "noise", "splice")
TEST_UNKNOWN_MODES = TRAIN_UNKNOWN_MODES + ("reversed", "joint_permuted")


def make_unknown_sequence(
    sequence: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    partner: np.ndarray | None = None,
) -> np.ndarray:
    """Create a deterministic proxy-Unknown sequence from a known gesture."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if mode == "frozen":
        frame = int(rng.integers(0, len(sequence)))
        return np.repeat(sequence[frame:frame + 1], len(sequence), axis=0)
    if mode == "shuffled":
        order = rng.permutation(len(sequence))
        if np.array_equal(order, np.arange(len(sequence))):
            order = order[::-1]
        return sequence[order].copy()
    if mode == "noise":
        mean = sequence.mean(axis=(0, 1), keepdims=True)
        std = sequence.std(axis=(0, 1), keepdims=True).clip(min=1e-3)
        return rng.normal(mean, std, size=sequence.shape).astype(np.float32)
    if mode == "splice":
        if partner is None:
            raise ValueError("splice mode requires a partner sequence")
        split = max(1, len(sequence) // 2)
        partner_indices = np.linspace(0, len(partner) - 1, len(sequence) - split).round().astype(int)
        return np.concatenate([sequence[:split], partner[partner_indices]], axis=0).astype(np.float32)
    if mode == "reversed":
        return sequence[::-1].copy()
    if mode == "joint_permuted":
        order = np.arange(22)
        order[1:] = rng.permutation(order[1:])
        return sequence[:, order].copy()
    raise ValueError(f"unknown proxy mode: {mode}")


def make_unknown_samples(
    samples: Sequence[RawGestureSample],
    modes: Sequence[str],
    seed: int,
    one_per_source: bool = True,
) -> list[RawGestureSample]:
    if not modes:
        raise ValueError("at least one Unknown mode is required")
    rng = np.random.default_rng(seed)
    by_other_label = {
        sample.index: [candidate for candidate in samples if candidate.label != sample.label]
        for sample in samples
    }
    output: list[RawGestureSample] = []
    for position, sample in enumerate(samples):
        selected_modes = (modes[int(rng.integers(0, len(modes)))],) if one_per_source else modes
        for mode_index, mode in enumerate(selected_modes):
            partner = None
            if mode == "splice":
                candidates = by_other_label[sample.index]
                if not candidates:
                    raise ValueError("splice mode requires at least two known labels")
                partner = candidates[int(rng.integers(0, len(candidates)))].sequence
            transformed = make_unknown_sequence(sample.sequence, mode, rng, partner)
            output.append(RawGestureSample(
                index=-(1 + position * max(1, len(modes)) + mode_index),
                label=UNKNOWN_LABEL,
                sequence=transformed,
                split=f"{sample.split}:unknown:{mode}",
            ))
    return output


def expand_teacher_to_unknown(base: ClassDiffusionModel) -> ClassDiffusionModel:
    """Expand a trained 14-state teacher to 15 states while preserving known weights."""
    if base.num_classes != 14:
        raise ValueError("the expansion source must be a 14-class teacher")
    expanded = ClassDiffusionModel(num_classes=15)
    old_state = base.state_dict()
    new_state = expanded.state_dict()
    for key, value in old_state.items():
        if key in new_state and value.shape == new_state[key].shape:
            new_state[key] = value.detach().clone()

    new_state["encoder.classifier.weight"][:14] = old_state["encoder.classifier.weight"]
    new_state["encoder.classifier.bias"][:14] = old_state["encoder.classifier.bias"]

    old_first = old_state["denoiser.0.weight"]
    new_first = new_state["denoiser.0.weight"]
    new_first[:, :129] = old_first[:, :129]
    new_first[:, 129:143] = old_first[:, 129:143]
    new_first[:, 144:] = old_first[:, 143:]
    new_state["denoiser.0.weight"] = new_first
    new_state["denoiser.3.weight"][:14] = old_state["denoiser.3.weight"]
    new_state["denoiser.3.bias"][:14] = old_state["denoiser.3.bias"]
    expanded.load_state_dict(new_state)
    return expanded


def forward_conditions(model: ClassDiffusionModel, conditions: torch.Tensor) -> torch.Tensor:
    """Run the existing Markov update at every observation using cached causal features."""
    uniform = torch.full(
        (len(conditions), model.num_classes),
        1.0 / model.num_classes,
        device=conditions.device,
        dtype=conditions.dtype,
    )
    previous = uniform
    outputs = []
    for update in range(conditions.shape[1]):
        initial = (
            uniform
            if update == 0
            else model.inheritance * previous + (1.0 - model.inheritance) * uniform
        )
        previous = model._reverse_distribution(conditions[:, update], initial)
        outputs.append(previous)
    return torch.stack(outputs, dim=1).clamp_min(1e-8).log()


@torch.no_grad()
def encode_samples(
    encoder_model: ClassDiffusionModel,
    samples: Sequence[RawGestureSample],
    labels: Sequence[str],
    frames: int,
    ratios: Sequence[float],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = PrefixSequenceDataset(samples, labels, frames, ratios, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    conditions, targets = [], []
    encoder_model.eval()
    for views, lengths, progress, target in loader:
        views = views.to(device)
        lengths = lengths.to(device)
        progress = progress.to(device)
        conditions.append(encoder_model.encode_sequence(views, lengths, progress).cpu())
        targets.append(target)
    return torch.cat(conditions), torch.cat(targets)


def trainable_denoiser_parameters(model: ClassDiffusionModel) -> list[torch.nn.Parameter]:
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


@dataclass
class StageMetrics:
    ratio: float
    known_accuracy: float
    known_false_no_rate: float
    unknown_recall: float
    balanced_accuracy: float


@torch.no_grad()
def predict_conditions(
    model: ClassDiffusionModel,
    conditions: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(conditions), batch_size):
        batch = conditions[start:start + batch_size].to(device)
        outputs.append(forward_conditions(model, batch).exp().cpu())
    return torch.cat(outputs).numpy()


def stage_metrics(
    known_probabilities: np.ndarray,
    known_targets: np.ndarray,
    unknown_probabilities: np.ndarray,
    ratios: Sequence[float],
    unknown_index: int = 14,
) -> list[StageMetrics]:
    rows = []
    known_predictions = known_probabilities.argmax(axis=-1)
    unknown_predictions = unknown_probabilities.argmax(axis=-1)
    for update, ratio in enumerate(ratios):
        known_accuracy = float(np.mean(known_predictions[:, update] == known_targets))
        false_no = float(np.mean(known_predictions[:, update] == unknown_index))
        unknown_recall = float(np.mean(unknown_predictions[:, update] == unknown_index))
        rows.append(StageMetrics(
            float(ratio), known_accuracy, false_no, unknown_recall,
            0.5 * (known_accuracy + unknown_recall),
        ))
    return rows


def metric_auc(rows: Sequence[StageMetrics], field: str) -> float:
    values = np.asarray([getattr(row, field) for row in rows], dtype=np.float64)
    ratios = np.asarray([row.ratio for row in rows], dtype=np.float64)
    return prefix_auc(ratios, values)


def evaluate_open_set(
    model: ClassDiffusionModel,
    known: tuple[torch.Tensor, torch.Tensor],
    unknown_by_type: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ratios: Sequence[float],
    batch_size: int,
    device: torch.device,
) -> dict:
    known_conditions, known_targets = known
    known_probabilities = predict_conditions(model, known_conditions, batch_size, device)
    per_type = {}
    pooled_unknown = []
    for mode, (conditions, _) in unknown_by_type.items():
        probabilities = predict_conditions(model, conditions, batch_size, device)
        pooled_unknown.append(probabilities)
        rows = stage_metrics(known_probabilities, known_targets.numpy(), probabilities, ratios)
        per_type[mode] = {
            "stage_metrics": [asdict(row) for row in rows],
            "unknown_recall_auc": metric_auc(rows, "unknown_recall"),
            "final_unknown_recall": rows[-1].unknown_recall,
        }
    pooled = np.concatenate(pooled_unknown, axis=0)
    rows = stage_metrics(known_probabilities, known_targets.numpy(), pooled, ratios)
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


def save_checkpoint(
    model: ClassDiffusionModel,
    destination: Path,
    labels: Sequence[str],
    ratios: Sequence[float],
    seed: int,
    epoch: int,
    score: float,
    experiment: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "experiment": experiment,
        "labels": list(labels),
        "frames": 64,
        "observation_ratios": list(ratios),
        "seed": int(seed),
        "epochs": int(epoch),
        "selection_metric": "validation_balanced_prefix_auc",
        "best_validation_score": float(score),
        "unknown_label": UNKNOWN_LABEL,
        "unknown_available_at_every_observation": True,
    }, destination)


def train_teacher(
    model: ClassDiffusionModel,
    train_data: tuple[torch.Tensor, torch.Tensor],
    validation_known: tuple[torch.Tensor, torch.Tensor],
    validation_unknown: dict[str, tuple[torch.Tensor, torch.Tensor]],
    labels: Sequence[str],
    ratios: Sequence[float],
    args,
    output: Path,
    device: torch.device,
) -> tuple[ClassDiffusionModel, list[dict]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        trainable_denoiser_parameters(model), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loader = DataLoader(TensorDataset(*train_data), batch_size=args.batch_size, shuffle=True)
    history, best_score = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for conditions, targets in loader:
            conditions = conditions.to(device)
            targets = targets.to(device)
            log_probabilities = forward_conditions(model, conditions)
            repeated = targets[:, None].expand(-1, conditions.shape[1]).reshape(-1)
            primary = F.nll_loss(log_probabilities.reshape(-1, len(labels)), repeated)
            denoising = model.denoising_loss(conditions, targets)
            loss = primary + 0.5 * denoising
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
            seen += len(targets)
        scheduler.step()
        validation = evaluate_open_set(
            model, validation_known, validation_unknown, ratios,
            args.eval_batch_size, device,
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "validation_balanced_prefix_auc": validation["balanced_accuracy_auc"],
            "validation_known_accuracy_auc": validation["known_accuracy_auc"],
            "validation_unknown_recall_auc": validation["unknown_recall_auc"],
        }
        history.append(row)
        print(
            f"Teacher epoch {epoch:03d} | loss {row['train_loss']:.4f} | "
            f"balanced AUC {row['validation_balanced_prefix_auc']:.2%} | "
            f"known {row['validation_known_accuracy_auc']:.2%} | "
            f"No {row['validation_unknown_recall_auc']:.2%}"
        )
        if validation["balanced_accuracy_auc"] >= best_score:
            best_score = validation["balanced_accuracy_auc"]
            save_checkpoint(
                model, output / "best.pt", labels, ratios, args.seed, epoch,
                best_score, "15_state_four_step_teacher",
            )
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model, history


def train_student(
    teacher: ClassDiffusionModel,
    train_data: tuple[torch.Tensor, torch.Tensor],
    validation_known: tuple[torch.Tensor, torch.Tensor],
    validation_unknown: dict[str, tuple[torch.Tensor, torch.Tensor]],
    labels: Sequence[str],
    ratios: Sequence[float],
    args,
    output: Path,
    device: torch.device,
    use_catdrift: bool,
) -> tuple[OneStepClassDiffusionModel, list[dict]]:
    student = OneStepClassDiffusionModel(num_classes=len(labels), dropout=args.dropout).to(device)
    student.load_state_dict(teacher.state_dict())
    teacher.eval()
    optimizer = torch.optim.AdamW(
        trainable_denoiser_parameters(student), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loader = DataLoader(TensorDataset(*train_data), batch_size=args.batch_size, shuffle=True)
    history, best_score = [], -1.0
    bank = ConditionalDriftMemoryBank(args.bank_capacity)
    experiment = "15_state_one_step_catdrift" if use_catdrift else "15_state_one_step_distilled"
    for epoch in range(1, args.epochs + 1):
        student.train()
        totals = {"loss": 0.0, "primary": 0.0, "denoising": 0.0, "kd": 0.0, "drift": 0.0}
        seen = 0
        for conditions, targets in loader:
            conditions = conditions.to(device)
            targets = targets.to(device)
            student_log = forward_conditions(student, conditions)
            with torch.no_grad():
                teacher_log = forward_conditions(teacher, conditions)
            repeated = targets[:, None].expand(-1, conditions.shape[1]).reshape(-1)
            primary = F.nll_loss(student_log.reshape(-1, len(labels)), repeated)
            denoising = student.denoising_loss(conditions, targets)
            kd = distillation_loss(student_log, teacher_log, args.temperature)
            drift = torch.zeros((), device=device)
            if use_catdrift:
                drift_target = conditional_categorical_drift_target(
                    student_log, teacher_log.exp(), conditions.detach(), targets, bank,
                    radii=tuple(args.drift_radii),
                    drift_strength=args.drift_strength,
                    context_weight=args.context_weight,
                )
                drift = F.kl_div(student_log, drift_target, reduction="batchmean") / student_log.shape[1]
            loss = primary + 0.5 * denoising + 0.1 * kd
            if use_catdrift:
                loss = loss + 0.05 * drift
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            for key, value in (
                ("loss", loss), ("primary", primary), ("denoising", denoising),
                ("kd", kd), ("drift", drift),
            ):
                totals[key] += float(value.detach()) * len(targets)
            seen += len(targets)
        scheduler.step()
        validation = evaluate_open_set(
            student, validation_known, validation_unknown, ratios,
            args.eval_batch_size, device,
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value / seen for key, value in totals.items()},
            "validation_balanced_prefix_auc": validation["balanced_accuracy_auc"],
            "validation_known_accuracy_auc": validation["known_accuracy_auc"],
            "validation_unknown_recall_auc": validation["unknown_recall_auc"],
        }
        history.append(row)
        print(
            f"{experiment} epoch {epoch:03d} | loss {row['train_loss']:.4f} | "
            f"balanced AUC {row['validation_balanced_prefix_auc']:.2%} | "
            f"known {row['validation_known_accuracy_auc']:.2%} | "
            f"No {row['validation_unknown_recall_auc']:.2%}"
        )
        if validation["balanced_accuracy_auc"] >= best_score:
            best_score = validation["balanced_accuracy_auc"]
            save_checkpoint(
                student, output / "best.pt", labels, ratios, args.seed, epoch,
                best_score, experiment,
            )
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    student.load_state_dict(checkpoint["model_state"])
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return student, history


def load_base_teacher(path: Path, device: torch.device) -> tuple[ClassDiffusionModel, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_progressive_model("04_class_diffusion", len(checkpoint["labels"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def write_report(output: Path, summary: dict) -> None:
    lines = [
        "# 15-state Unknown class-diffusion experiment",
        "",
        "Every 25/50/65/80/100% observation emits a 15-state posterior. Unknown is",
        "a normal candidate at every Markov update and is not absorbing, so later",
        "evidence may recover an early No decision into a known gesture.",
        "",
        "| Model | Known AUC | False-No AUC | Unknown AUC | Balanced AUC | Final known | Final No recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("teacher", "one_step_distilled", "one_step_catdrift"):
        row = summary[name]
        lines.append(
            f"| {name} | {row['known_accuracy_auc']:.2%} | {row['known_false_no_auc']:.2%} | "
            f"{row['unknown_recall_auc']:.2%} | {row['balanced_accuracy_auc']:.2%} | "
            f"{row['final_known_accuracy']:.2%} | {row['final_unknown_recall']:.2%} |"
        )
    lines.extend(["", "## Per-observation one-step CatDrift", ""])
    for row in summary["one_step_catdrift"]["stage_metrics"]:
        lines.append(
            f"- {row['ratio']:.0%}: known {row['known_accuracy']:.2%}, "
            f"false No {row['known_false_no_rate']:.2%}, No recall {row['unknown_recall']:.2%}."
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="15-state Markov class diffusion with per-step No")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--data", default="data/shrec17_ddnet_npz")
    parser.add_argument("--output", default="runs/unknown_diffusion_seed42")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--bank-capacity", type=int, default=64)
    parser.add_argument("--drift-radii", type=float, nargs="+", default=[0.2, 0.5, 1.0])
    parser.add_argument("--drift-strength", type=float, default=0.25)
    parser.add_argument("--context-weight", type=float, default=0.25)
    parser.add_argument(
        "--unknown-train-ratio",
        type=float,
        default=1.0,
        help="number of proxy-Unknown training samples per known training sample",
    )
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
        raise ValueError("experiment seed must match the source teacher")
    ratios = tuple(float(value) for value in args.observation_ratios)
    labels = list(base_checkpoint["labels"]) + [UNKNOWN_LABEL]
    set_reproducible(args.seed)
    random.seed(args.seed)

    official_train = load_raw_shrec17_npz(args.data, "train")
    known_test = load_raw_shrec17_npz(args.data, "test")
    known_train, known_validation = stratified_raw_split(official_train, args.val_ratio, args.seed)
    if not 0.0 < args.unknown_train_ratio <= 1.0:
        raise ValueError("unknown-train-ratio must be in (0, 1]")
    unknown_pool = make_unknown_samples(known_train, TRAIN_UNKNOWN_MODES, args.seed + 101)
    unknown_count = max(1, round(len(known_train) * args.unknown_train_ratio))
    selection_rng = np.random.default_rng(args.seed + 102)
    selected = selection_rng.choice(len(unknown_pool), size=unknown_count, replace=False)
    unknown_train = [unknown_pool[int(index)] for index in selected]
    unknown_validation_by_type = {
        mode: make_unknown_samples(known_validation, (mode,), args.seed + 200 + index)
        for index, mode in enumerate(TRAIN_UNKNOWN_MODES)
    }
    unknown_test_by_type = {
        mode: make_unknown_samples(known_test, (mode,), args.seed + 300 + index)
        for index, mode in enumerate(TEST_UNKNOWN_MODES)
    }

    print(
        f"Seed {args.seed} | device {device} | known train/val/test "
        f"{len(known_train)}/{len(known_validation)}/{len(known_test)} | "
        f"Unknown train {len(unknown_train)}"
    )
    encode_labels = list(base_checkpoint["labels"]) + [UNKNOWN_LABEL]
    # The 14-state encoder supplies the same 128-D representation; its classifier
    # is not used by encode_sequence. Unknown is learned entirely in the existing
    # 15-state diffusion/search space.
    train_conditions = encode_samples(
        base_teacher, list(known_train) + unknown_train, encode_labels,
        args.frames, ratios, args.encoder_batch_size, device,
    )
    validation_known = encode_samples(
        base_teacher, known_validation, encode_labels, args.frames, ratios,
        args.encoder_batch_size, device,
    )
    validation_unknown = {
        mode: encode_samples(
            base_teacher, samples, encode_labels, args.frames, ratios,
            args.encoder_batch_size, device,
        )
        for mode, samples in unknown_validation_by_type.items()
    }
    test_known = encode_samples(
        base_teacher, known_test, encode_labels, args.frames, ratios,
        args.encoder_batch_size, device,
    )
    test_unknown = {
        mode: encode_samples(
            base_teacher, samples, encode_labels, args.frames, ratios,
            args.encoder_batch_size, device,
        )
        for mode, samples in unknown_test_by_type.items()
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    teacher, _ = train_teacher(
        expand_teacher_to_unknown(base_teacher), train_conditions,
        validation_known, validation_unknown, labels, ratios, args,
        output / "00_four_step_teacher_15", device,
    )
    distilled, _ = train_student(
        teacher, train_conditions, validation_known, validation_unknown,
        labels, ratios, args, output / "01_one_step_distilled_15", device, False,
    )
    catdrift, _ = train_student(
        teacher, train_conditions, validation_known, validation_unknown,
        labels, ratios, args, output / "02_one_step_catdrift_15", device, True,
    )

    summary = {
        "seed": args.seed,
        "epochs": args.epochs,
        "labels": labels,
        "unknown_label": UNKNOWN_LABEL,
        "unknown_available_at_every_observation": True,
        "unknown_train_ratio": args.unknown_train_ratio,
        "train_unknown_modes": list(TRAIN_UNKNOWN_MODES),
        "test_unknown_modes": list(TEST_UNKNOWN_MODES),
        "teacher": evaluate_open_set(
            teacher, test_known, test_unknown, ratios, args.eval_batch_size, device,
        ),
        "one_step_distilled": evaluate_open_set(
            distilled, test_known, test_unknown, ratios, args.eval_batch_size, device,
        ),
        "one_step_catdrift": evaluate_open_set(
            catdrift, test_known, test_unknown, ratios, args.eval_batch_size, device,
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(output, summary)
    print(f"Wrote {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
