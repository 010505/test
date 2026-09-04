from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .drifting import build_one_step_model
from .progressive import (
    DEFAULT_OBSERVATION_RATIOS,
    load_raw_shrec17_npz,
    stratified_raw_split,
)
from .progressive_benchmark import prefix_auc, set_reproducible
from .unknown_diffusion_benchmark import (
    TEST_UNKNOWN_MODES,
    TRAIN_UNKNOWN_MODES,
    UNKNOWN_LABEL,
    encode_samples,
    make_unknown_samples,
    metric_auc,
    stage_metrics,
)
from .unknown_diffusion_optimization import forward_conditions_with_no_retention


def load_evidence_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("experiment") != "08_one_step_distilled":
        raise ValueError("evidence checkpoint must be 08_one_step_distilled")
    labels = list(checkpoint["labels"])
    if len(labels) != 14:
        raise ValueError("hierarchical evidence model must contain 14 known classes")
    model = build_one_step_model(14).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint, labels


@torch.no_grad()
def cached_evidence(
    model,
    conditions: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs = []
    model.eval()
    for start in range(0, len(conditions), batch_size):
        log_probabilities = forward_conditions_with_no_retention(
            model, conditions[start:start + batch_size].to(device)
        )
        outputs.append(log_probabilities.exp().cpu())
    return torch.cat(outputs)


def _balanced_split(members: Sequence[int], prototypes: np.ndarray) -> tuple[list[int], list[int]]:
    members = list(members)
    if len(members) < 2:
        return members, []
    values = prototypes[members]
    centered = values - values.mean(axis=0, keepdims=True)
    if np.allclose(centered, 0.0):
        ordered = sorted(members)
    else:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
        nonzero = np.flatnonzero(np.abs(direction) > 1e-8)
        if len(nonzero) and direction[nonzero[0]] < 0:
            direction = -direction
        projection = centered @ direction
        ordered = [member for _, member in sorted(zip(projection.tolist(), members))]
    middle = len(ordered) // 2
    return ordered[:middle], ordered[middle:]


def build_balanced_tree_levels(prototypes: np.ndarray) -> list[list[list[int]]]:
    """Build root plus four nested partitions, ending in fourteen singleton leaves."""
    if prototypes.shape[0] != 14:
        raise ValueError("exactly fourteen class prototypes are required")
    levels: list[list[list[int]]] = [[list(range(14))]]
    while any(len(node) > 1 for node in levels[-1]):
        next_level = []
        for node in levels[-1]:
            if len(node) == 1:
                next_level.append(node)
                continue
            left, right = _balanced_split(node, prototypes)
            next_level.extend([left, right])
        levels.append(next_level)
    if len(levels) != 5 or len(levels[-1]) != 14:
        raise RuntimeError("balanced fourteen-class tree must have four decision depths")
    return levels


def level_masks(levels: Sequence[Sequence[Sequence[int]]]) -> list[torch.Tensor]:
    output = []
    for level in levels:
        masks = torch.zeros((len(level), 14), dtype=torch.bool)
        for index, members in enumerate(level):
            masks[index, list(members)] = True
        output.append(masks)
    return output


def choose_group(
    evidence: torch.Tensor,
    active_mask: torch.Tensor,
    next_masks: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """Choose a child if supported, otherwise a same-depth complement branch."""
    masks = next_masks.to(evidence.device)
    scores = evidence @ masks.to(evidence.dtype).T
    subset = (masks[None, :, :] <= active_mask[:, None, :]).all(dim=-1)
    allowed = torch.where(support[:, None], subset, ~subset)
    empty = ~allowed.any(dim=-1)
    if empty.any():
        allowed[empty] = True
    scores = scores.masked_fill(~allowed, -1.0)
    selected = scores.argmax(dim=-1)
    return masks[selected]


class HierarchicalVerificationSearch(nn.Module):
    """Verification/backtracking controller over a fixed fourteen-class tree."""

    def __init__(
        self,
        masks: Sequence[torch.Tensor],
        hidden: int = 96,
        dropout: float = 0.15,
        known_inheritance: float = 0.5,
    ):
        super().__init__()
        if len(masks) != 5:
            raise ValueError("root plus four hierarchy levels are required")
        self.known_inheritance = float(known_inheritance)
        self.level_names = []
        for depth, value in enumerate(masks):
            name = f"tree_level_{depth}"
            self.register_buffer(name, value.to(torch.bool))
            self.level_names.append(name)
        # condition, newly arrived feature delta, 14-way evidence, active mask,
        # support mass, evidence maximum, normalized entropy, active fraction,
        # and current support probability.
        input_dim = 129 + 129 + 14 + 14 + 5
        self.support_head = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.knownness_head = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _features(
        self,
        condition: torch.Tensor,
        delta: torch.Tensor,
        evidence: torch.Tensor,
        active: torch.Tensor,
        support_probability: torch.Tensor,
    ) -> torch.Tensor:
        support_mass = (evidence * active.to(evidence.dtype)).sum(dim=-1, keepdim=True)
        maximum = evidence.max(dim=-1, keepdim=True).values
        entropy = -(
            evidence.clamp_min(1e-8) * evidence.clamp_min(1e-8).log()
        ).sum(dim=-1, keepdim=True) / np.log(14.0)
        fraction = active.to(evidence.dtype).mean(dim=-1, keepdim=True)
        return torch.cat([
            condition,
            delta,
            evidence,
            active.to(evidence.dtype),
            support_mass,
            maximum,
            entropy,
            fraction,
            support_probability.unsqueeze(-1),
        ], dim=-1)

    def forward(
        self,
        conditions: torch.Tensor,
        evidence: torch.Tensor,
        allow_backtracking: bool = True,
        known_blend_weights: Sequence[float] | torch.Tensor | None = None,
        no_biases: Sequence[float] | torch.Tensor | None = None,
    ):
        if conditions.shape[:2] != evidence.shape[:2] or evidence.shape[-1] != 14:
            raise ValueError("conditions and evidence must align as [B, S, 129/14]")
        batch, stages = conditions.shape[:2]
        blend_weights = (
            torch.zeros(stages, dtype=conditions.dtype, device=conditions.device)
            if known_blend_weights is None
            else torch.as_tensor(
                known_blend_weights, dtype=conditions.dtype, device=conditions.device
            )
        )
        no_bias = (
            torch.zeros(stages, dtype=conditions.dtype, device=conditions.device)
            if no_biases is None
            else torch.as_tensor(no_biases, dtype=conditions.dtype, device=conditions.device)
        )
        if blend_weights.shape != (stages,) or no_bias.shape != (stages,):
            raise ValueError("one blend weight and one No bias are required per stage")
        if bool(((blend_weights < 0.0) | (blend_weights > 1.0)).any()):
            raise ValueError("known blend weights must be in [0, 1]")
        root = getattr(self, self.level_names[0]).to(conditions.device)
        active = root[0].expand(batch, -1)
        previous_condition = torch.zeros_like(conditions[:, 0])
        known_state = None
        class_outputs = []
        known_logits = []
        support_logits = []
        active_before = []
        active_after = []
        candidate_sizes = []
        for stage in range(stages):
            current_evidence = evidence[:, stage]
            condition = conditions[:, stage]
            delta = condition - previous_condition
            active_before.append(active)
            if stage == 0:
                support_logit = torch.full(
                    (batch,), 12.0, dtype=condition.dtype, device=condition.device
                )
                support_probability = support_logit.sigmoid()
                next_masks = getattr(self, self.level_names[1])
                active = choose_group(
                    current_evidence, active, next_masks, torch.ones(batch, dtype=torch.bool, device=condition.device)
                )
            else:
                preliminary = self._features(
                    condition,
                    delta,
                    current_evidence,
                    active,
                    torch.full((batch,), 0.5, dtype=condition.dtype, device=condition.device),
                )
                support_logit = self.support_head(preliminary).squeeze(-1)
                support_probability = support_logit.sigmoid()
                next_depth = min(stage + 1, len(self.level_names) - 1)
                next_masks = getattr(self, self.level_names[next_depth])
                branch_support = (
                    support_probability >= 0.5
                    if allow_backtracking
                    else torch.ones(batch, dtype=torch.bool, device=condition.device)
                )
                active = choose_group(
                    current_evidence,
                    active,
                    next_masks,
                    branch_support,
                )
            features = self._features(
                condition, delta, current_evidence, active, support_probability
            )
            current_known = self.knownness_head(features).squeeze(-1)
            known_state = (
                current_known
                if known_state is None
                else self.known_inheritance * known_state
                + (1.0 - self.known_inheritance) * current_known
            )
            masked = current_evidence * active.to(current_evidence.dtype)
            masked = masked + 1e-6 * current_evidence * (~active).to(current_evidence.dtype)
            masked = masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            # Candidate search is an early-decision aid. A stage-dependent blend
            # can hand the conditional known-class distribution back to the
            # frozen 14-class evidence head; weight 1.0 exactly preserves its
            # class ordering at the full observation.
            known_distribution = (
                (1.0 - blend_weights[stage]) * masked
                + blend_weights[stage] * current_evidence
            )
            known_distribution = known_distribution / known_distribution.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            # A positive bias favours residual No; a negative bias favours the
            # known set. It never changes ordering among the fourteen actions.
            known_probability = (known_state - no_bias[stage]).sigmoid()
            output = torch.cat([
                known_probability[:, None] * known_distribution,
                (1.0 - known_probability)[:, None],
            ], dim=-1)
            class_outputs.append(output)
            known_logits.append(known_state)
            support_logits.append(support_logit)
            active_after.append(active)
            candidate_sizes.append(active.sum(dim=-1))
            previous_condition = condition
        return torch.stack(class_outputs, dim=1), {
            "known_logits": torch.stack(known_logits, dim=1),
            "support_logits": torch.stack(support_logits, dim=1),
            "active_before": torch.stack(active_before, dim=1),
            "active_after": torch.stack(active_after, dim=1),
            "candidate_sizes": torch.stack(candidate_sizes, dim=1),
        }


def controller_losses(
    diagnostics: dict,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    known_target = (targets < 14).to(diagnostics["known_logits"].dtype)
    known_target = known_target[:, None].expand_as(diagnostics["known_logits"])
    knownness = F.binary_cross_entropy_with_logits(
        diagnostics["known_logits"], known_target
    )
    active = diagnostics["active_before"][:, 1:]
    known = targets < 14
    safe_targets = targets.clamp_max(13)
    membership = active.gather(
        -1, safe_targets[:, None, None].expand(-1, active.shape[1], 1)
    ).squeeze(-1)
    support_target = (membership & known[:, None]).to(
        diagnostics["support_logits"].dtype
    )
    support = F.binary_cross_entropy_with_logits(
        diagnostics["support_logits"][:, 1:], support_target
    )
    return knownness, support


@torch.no_grad()
def predict_controller(
    model: HierarchicalVerificationSearch,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int,
    device: torch.device,
    allow_backtracking: bool = True,
    known_blend_weights: Sequence[float] | torch.Tensor | None = None,
    no_biases: Sequence[float] | torch.Tensor | None = None,
) -> tuple[np.ndarray, dict]:
    conditions, evidence, targets = data
    model.eval()
    probabilities = []
    diagnostics = {"active_before": [], "active_after": [], "candidate_sizes": [], "support_logits": []}
    for start in range(0, len(conditions), batch_size):
        output, diagnostic = model(
            conditions[start:start + batch_size].to(device),
            evidence[start:start + batch_size].to(device),
            allow_backtracking=allow_backtracking,
            known_blend_weights=known_blend_weights,
            no_biases=no_biases,
        )
        probabilities.append(output.cpu())
        for key in diagnostics:
            diagnostics[key].append(diagnostic[key].cpu())
    return torch.cat(probabilities).numpy(), {
        key: torch.cat(value).numpy() for key, value in diagnostics.items()
    }


def evaluate_controller(
    model: HierarchicalVerificationSearch,
    known: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    unknown_by_type: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ratios: Sequence[float],
    batch_size: int,
    device: torch.device,
    allow_backtracking: bool = True,
    known_blend_weights: Sequence[float] | torch.Tensor | None = None,
    no_biases: Sequence[float] | torch.Tensor | None = None,
) -> dict:
    known_probabilities, known_diagnostics = predict_controller(
        model, known, batch_size, device, allow_backtracking,
        known_blend_weights, no_biases,
    )
    targets = known[2].numpy()
    pooled_unknown = []
    per_type = {}
    for mode, data in unknown_by_type.items():
        probabilities, _ = predict_controller(
            model, data, batch_size, device, allow_backtracking,
            known_blend_weights, no_biases,
        )
        pooled_unknown.append(probabilities)
        rows = stage_metrics(known_probabilities, targets, probabilities, ratios)
        per_type[mode] = {
            "stage_metrics": [asdict(row) for row in rows],
            "unknown_recall_auc": metric_auc(rows, "unknown_recall"),
            "final_unknown_recall": rows[-1].unknown_recall,
        }
    rows = stage_metrics(
        known_probabilities, targets, np.concatenate(pooled_unknown), ratios
    )
    conditional_predictions = known_probabilities[..., :14].argmax(axis=-1)
    conditional_accuracy = [
        float(np.mean(conditional_predictions[:, stage] == targets))
        for stage in range(len(ratios))
    ]
    active_after = known_diagnostics["active_after"]
    initial_wrong = ~active_after[:, 0, :][np.arange(len(targets)), targets]
    recovered = active_after[np.arange(len(targets)), :, targets]
    initial_count = int(initial_wrong.sum())
    recovery = [
        float(recovered[initial_wrong, stage].mean()) if initial_count else 1.0
        for stage in range(len(ratios))
    ]
    support_prediction = known_diagnostics["support_logits"][:, 1:] >= 0.0
    active_before = known_diagnostics["active_before"][:, 1:]
    support_target = active_before[
        np.arange(len(targets))[:, None], np.arange(len(ratios) - 1)[None, :], targets[:, None]
    ]
    return {
        "stage_metrics": [asdict(row) for row in rows],
        "known_accuracy_auc": metric_auc(rows, "known_accuracy"),
        "known_false_no_auc": metric_auc(rows, "known_false_no_rate"),
        "unknown_recall_auc": metric_auc(rows, "unknown_recall"),
        "balanced_accuracy_auc": metric_auc(rows, "balanced_accuracy"),
        "final_known_accuracy": rows[-1].known_accuracy,
        "final_known_false_no_rate": rows[-1].known_false_no_rate,
        "final_unknown_recall": rows[-1].unknown_recall,
        "conditional_known_stage_accuracy": conditional_accuracy,
        "conditional_known_accuracy_auc": prefix_auc(
            np.asarray(ratios), np.asarray(conditional_accuracy)
        ),
        "final_conditional_known_accuracy": conditional_accuracy[-1],
        "candidate_size_by_stage": known_diagnostics["candidate_sizes"].mean(axis=0).tolist(),
        "initial_wrong_group_samples": initial_count,
        "wrong_group_recovery_by_stage": recovery,
        "final_wrong_group_recovery": recovery[-1],
        "support_gate_accuracy": float(np.mean(support_prediction == support_target)),
        "per_type": per_type,
    }


def old_known_metrics(evidence: torch.Tensor, targets: torch.Tensor, ratios: Sequence[float]) -> dict:
    predictions = evidence.argmax(dim=-1).numpy()
    target_array = targets.numpy()
    accuracy = [
        float(np.mean(predictions[:, stage] == target_array))
        for stage in range(len(ratios))
    ]
    return {
        "stage_accuracy": accuracy,
        "known_accuracy_auc": prefix_auc(np.asarray(ratios), np.asarray(accuracy)),
        "final_known_accuracy": accuracy[-1],
    }


def selection_score(metrics: dict, old_metrics: dict, args) -> float:
    known_shortfall = max(
        0.0,
        old_metrics["known_accuracy_auc"] - args.max_known_drop - metrics["known_accuracy_auc"],
    )
    false_no_excess = max(0.0, metrics["known_false_no_auc"] - args.max_false_no)
    return float(
        metrics["balanced_accuracy_auc"]
        - args.constraint_penalty * (known_shortfall + false_no_excess)
    )


def write_report(output: Path, summary: dict) -> None:
    row = summary["hierarchical_verification"]
    no_backtracking = summary["hierarchical_no_backtracking"]
    base = summary["baseline_15_state"]
    lines = [
        "# Hierarchical verification Markov search",
        "",
        "The model keeps fourteen known leaves, verifies the previous active set,",
        "switches to complement branches when contradicted, and emits No as residual",
        "known-set mass rather than as a fifteenth action prototype.",
        "",
        "| Model | Known AUC | False-No AUC | No AUC | Balanced AUC | Final known | Final false No | Final No recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Existing 15-state | {base['known_accuracy_auc']:.2%} | {base['known_false_no_auc']:.2%} | {base['unknown_recall_auc']:.2%} | {base['balanced_accuracy_auc']:.2%} | {base['final_known_accuracy']:.2%} | {base['final_known_false_no_rate']:.2%} | {base['final_unknown_recall']:.2%} |",
        f"| Hierarchy without backtracking | {no_backtracking['known_accuracy_auc']:.2%} | {no_backtracking['known_false_no_auc']:.2%} | {no_backtracking['unknown_recall_auc']:.2%} | {no_backtracking['balanced_accuracy_auc']:.2%} | {no_backtracking['final_known_accuracy']:.2%} | {no_backtracking['final_known_false_no_rate']:.2%} | {no_backtracking['final_unknown_recall']:.2%} |",
        f"| Hierarchical verification | {row['known_accuracy_auc']:.2%} | {row['known_false_no_auc']:.2%} | {row['unknown_recall_auc']:.2%} | {row['balanced_accuracy_auc']:.2%} | {row['final_known_accuracy']:.2%} | {row['final_known_false_no_rate']:.2%} | {row['final_unknown_recall']:.2%} |",
        "",
        f"Candidate sizes: `{row['candidate_size_by_stage']}`.",
        f"Initial wrong-group samples: `{row['initial_wrong_group_samples']}`.",
        f"Final recovery from an initially wrong group: `{row['final_wrong_group_recovery']:.2%}`.",
        f"Support-gate accuracy: `{row['support_gate_accuracy']:.2%}`.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verification/backtracking Markov candidate search")
    parser.add_argument("--evidence-checkpoint", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--data", default="data/shrec17_ddnet_npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--encoder-batch-size", type=int, default=16)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--unknown-train-ratio", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--known-inheritance", type=float, default=0.5)
    parser.add_argument("--support-weight", type=float, default=1.0)
    parser.add_argument("--max-known-drop", type=float, default=0.02)
    parser.add_argument("--max-false-no", type=float, default=0.05)
    parser.add_argument("--constraint-penalty", type=float, default=2.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--observation-ratios", type=float, nargs="+", default=list(DEFAULT_OBSERVATION_RATIOS))
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    evidence_model, evidence_checkpoint, known_labels = load_evidence_model(
        Path(args.evidence_checkpoint), device
    )
    evidence_seed = int(evidence_checkpoint["seed"])
    args.seed = evidence_seed if args.seed is None else args.seed
    if args.seed != evidence_seed:
        raise ValueError("experiment seed must match evidence checkpoint")
    if not 0.0 < args.unknown_train_ratio <= 1.0:
        raise ValueError("unknown-train-ratio must be in (0, 1]")
    set_reproducible(args.seed)
    random.seed(args.seed)
    ratios = tuple(float(value) for value in args.observation_ratios)
    labels = known_labels + [UNKNOWN_LABEL]

    official_train = load_raw_shrec17_npz(args.data, "train")
    known_test_samples = load_raw_shrec17_npz(args.data, "test")
    known_train, known_validation_samples = stratified_raw_split(
        official_train, args.val_ratio, args.seed
    )
    unknown_pool = make_unknown_samples(known_train, TRAIN_UNKNOWN_MODES, args.seed + 101)
    unknown_count = max(1, round(len(known_train) * args.unknown_train_ratio))
    rng = np.random.default_rng(args.seed + 102)
    selected = rng.choice(len(unknown_pool), size=unknown_count, replace=False)
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

    def prepare(samples):
        conditions, targets = encode_samples(
            evidence_model, samples, labels, args.frames, ratios,
            args.encoder_batch_size, device,
        )
        evidence = cached_evidence(
            evidence_model, conditions, args.eval_batch_size, device
        )
        return conditions, evidence, targets

    train_data = prepare(list(known_train) + unknown_train)
    validation_known = prepare(known_validation_samples)
    validation_unknown = {
        mode: prepare(samples) for mode, samples in unknown_validation_samples.items()
    }
    test_known = prepare(known_test_samples)
    test_unknown = {
        mode: prepare(samples) for mode, samples in unknown_test_samples.items()
    }

    known_train_count = len(known_train)
    train_conditions, _, train_targets = train_data
    prototypes = np.zeros((14, 128), dtype=np.float64)
    for label in range(14):
        selected_features = train_conditions[:known_train_count, -1, :128][
            train_targets[:known_train_count] == label
        ]
        prototypes[label] = selected_features.mean(dim=0).numpy()
    levels = build_balanced_tree_levels(prototypes)
    masks = level_masks(levels)
    model = HierarchicalVerificationSearch(
        masks, dropout=args.dropout, known_inheritance=args.known_inheritance
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loader = DataLoader(TensorDataset(*train_data), batch_size=args.batch_size, shuffle=True)
    old_validation = old_known_metrics(validation_known[1], validation_known[2], ratios)
    history = []
    best_score = -float("inf")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = {"loss": 0.0, "knownness": 0.0, "support": 0.0}
        seen = 0
        for conditions, evidence, targets in loader:
            conditions = conditions.to(device)
            evidence = evidence.to(device)
            targets = targets.to(device)
            _, diagnostics = model(conditions, evidence)
            knownness, support = controller_losses(diagnostics, targets)
            loss = knownness + args.support_weight * support
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total["loss"] += float(loss.detach()) * len(targets)
            total["knownness"] += float(knownness.detach()) * len(targets)
            total["support"] += float(support.detach()) * len(targets)
            seen += len(targets)
        scheduler.step()
        validation = evaluate_controller(
            model, validation_known, validation_unknown, ratios,
            args.eval_batch_size, device,
        )
        score = selection_score(validation, old_validation, args)
        row = {
            "epoch": epoch,
            **{f"train_{name}": value / seen for name, value in total.items()},
            "selection_score": score,
            "validation": validation,
        }
        history.append(row)
        print(
            f"Hierarchy epoch {epoch:03d} | loss {row['train_loss']:.4f} | "
            f"score {score:.2%} | known {validation['known_accuracy_auc']:.2%} | "
            f"false No {validation['known_false_no_auc']:.2%} | "
            f"No {validation['unknown_recall_auc']:.2%} | "
            f"recover {validation['final_wrong_group_recovery']:.2%}"
        )
        if score >= best_score:
            best_score = score
            torch.save({
                "model_state": model.state_dict(),
                "experiment": "hierarchical_verification_markov_search",
                "labels": labels,
                "known_labels": known_labels,
                "frames": args.frames,
                "observation_ratios": list(ratios),
                "seed": args.seed,
                "epochs": epoch,
                "tree_levels": levels,
                "selection_metric": "constrained_validation_open_set_score",
                "best_validation_score": score,
                "unknown_is_residual_mass": True,
                "unknown_is_peer_class": False,
            }, output / "best.pt")

    best = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model_state"])
    test_metrics = evaluate_controller(
        model, test_known, test_unknown, ratios, args.eval_batch_size, device
    )
    baseline = json.loads(Path(args.baseline_summary).read_text(encoding="utf-8"))[
        "one_step_distilled"
    ]
    summary = {
        "seed": args.seed,
        "epochs": args.epochs,
        "tree_levels": levels,
        "old_14_test": old_known_metrics(test_known[1], test_known[2], ratios),
        "baseline_15_state": baseline,
        "hierarchical_no_backtracking": evaluate_controller(
            model,
            test_known,
            test_unknown,
            ratios,
            args.eval_batch_size,
            device,
            allow_backtracking=False,
        ),
        "hierarchical_verification": test_metrics,
    }
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(output, summary)
    print(f"Wrote {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
