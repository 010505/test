from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .hierarchical_markov_search import (
    HierarchicalVerificationSearch,
    choose_group,
    level_masks,
    load_evidence_model,
)
from .progressive import load_raw_shrec17_npz, stratified_raw_split
from .progressive_benchmark import prefix_auc, set_reproducible
from .reverse_relabel_evaluation import (
    SEMANTIC_REVERSE_MAPPING,
    reverse_with_source_labels,
)
from .temporal_pe_order_optimization import (
    TemporalPEOrderKnownness,
    encode_temporal_samples,
)


@torch.no_grad()
def cache_temporal(temporal_model, data, batch_size: int, device: torch.device):
    frames, lengths = data[3], data[4]
    temporal_model.eval()
    pooled = []
    for start in range(0, len(frames), batch_size):
        current, _ = temporal_model.encode_temporal(
            frames[start:start + batch_size].to(device),
            lengths[start:start + batch_size].to(device),
        )
        pooled.append(current.cpu())
    return torch.cat(pooled)


class TemporalCandidateMembership(nn.Module):
    """Temporal residual for P(true class in current candidate set).

    The module never predicts whether a sequence is reversed. It only adjusts
    the candidate-support logit of a frozen hierarchical Markov controller.
    """

    def __init__(
        self,
        base: HierarchicalVerificationSearch,
        temporal_dim: int = 64,
        hidden: int = 96,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        # temporal state, evidence, active-set mask, base support logit
        input_dim = temporal_dim + 14 + 14 + 1
        self.support_residual = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.support_residual[-1].weight)
        nn.init.zeros_(self.support_residual[-1].bias)

    def forward(
        self,
        conditions: torch.Tensor,
        evidence: torch.Tensor,
        temporal: torch.Tensor,
        known_blend_weights=(0.0, 0.25, 0.5, 0.75, 1.0),
    ):
        if temporal.shape[:2] != evidence.shape[:2] or temporal.shape[-1] != 64:
            raise ValueError("temporal features must align as [B, S, 64]")
        batch, stages = conditions.shape[:2]
        blend = torch.as_tensor(
            known_blend_weights, dtype=conditions.dtype, device=conditions.device
        )
        if blend.shape != (stages,):
            raise ValueError("one known blend weight is required per stage")
        root = getattr(self.base, self.base.level_names[0]).to(conditions.device)
        active = root[0].expand(batch, -1)
        previous_condition = torch.zeros_like(conditions[:, 0])
        known_state = None
        outputs, support_logits, residuals = [], [], []
        active_before, active_after, candidate_sizes = [], [], []
        for stage in range(stages):
            current_condition = conditions[:, stage]
            current_evidence = evidence[:, stage]
            delta = current_condition - previous_condition
            active_before.append(active)
            if stage == 0:
                base_logit = torch.full(
                    (batch,), 12.0,
                    dtype=current_condition.dtype,
                    device=current_condition.device,
                )
                residual = torch.zeros_like(base_logit)
                support_logit = base_logit
                active = choose_group(
                    current_evidence,
                    active,
                    getattr(self.base, self.base.level_names[1]),
                    torch.ones(batch, dtype=torch.bool, device=conditions.device),
                )
            else:
                preliminary = self.base._features(
                    current_condition,
                    delta,
                    current_evidence,
                    active,
                    torch.full(
                        (batch,), 0.5,
                        dtype=current_condition.dtype,
                        device=current_condition.device,
                    ),
                )
                base_logit = self.base.support_head(preliminary).squeeze(-1)
                residual_input = torch.cat([
                    temporal[:, stage],
                    current_evidence,
                    active.to(current_evidence.dtype),
                    base_logit[:, None],
                ], dim=-1)
                residual = self.support_residual(residual_input).squeeze(-1)
                support_logit = base_logit + residual
                next_depth = min(stage + 1, len(self.base.level_names) - 1)
                active = choose_group(
                    current_evidence,
                    active,
                    getattr(self.base, self.base.level_names[next_depth]),
                    support_logit >= 0.0,
                )
            support_probability = support_logit.sigmoid()
            features = self.base._features(
                current_condition,
                delta,
                current_evidence,
                active,
                support_probability,
            )
            current_known = self.base.knownness_head(features).squeeze(-1)
            known_state = (
                current_known if known_state is None else
                self.base.known_inheritance * known_state
                + (1.0 - self.base.known_inheritance) * current_known
            )
            masked = current_evidence * active.to(current_evidence.dtype)
            masked = masked + 1e-6 * current_evidence * (~active).to(current_evidence.dtype)
            masked = masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            known_distribution = (
                (1.0 - blend[stage]) * masked
                + blend[stage] * current_evidence
            )
            known_distribution = known_distribution / known_distribution.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            known_probability = known_state.sigmoid()
            outputs.append(torch.cat([
                known_probability[:, None] * known_distribution,
                (1.0 - known_probability)[:, None],
            ], dim=-1))
            support_logits.append(support_logit)
            residuals.append(residual)
            active_after.append(active)
            candidate_sizes.append(active.sum(dim=-1))
            previous_condition = current_condition
        return torch.stack(outputs, dim=1), {
            "support_logits": torch.stack(support_logits, dim=1),
            "support_residuals": torch.stack(residuals, dim=1),
            "active_before": torch.stack(active_before, dim=1),
            "active_after": torch.stack(active_after, dim=1),
            "candidate_sizes": torch.stack(candidate_sizes, dim=1),
        }


def support_loss(diagnostics: dict, targets: torch.Tensor) -> torch.Tensor:
    active = diagnostics["active_before"][:, 1:]
    membership = active.gather(
        -1, targets[:, None, None].expand(-1, active.shape[1], 1)
    ).squeeze(-1)
    return F.binary_cross_entropy_with_logits(
        diagnostics["support_logits"][:, 1:], membership.to(torch.float32)
    )


@torch.no_grad()
def predict(model, data, batch_size: int, device: torch.device):
    conditions, evidence, targets, temporal = data
    model.eval()
    probabilities, diagnostics = [], {
        "support_logits": [], "active_before": [], "active_after": [],
        "candidate_sizes": [], "support_residuals": [],
    }
    for start in range(0, len(conditions), batch_size):
        output, current = model(
            conditions[start:start + batch_size].to(device),
            evidence[start:start + batch_size].to(device),
            temporal[start:start + batch_size].to(device),
        )
        probabilities.append(output.cpu())
        for key in diagnostics:
            diagnostics[key].append(current[key].cpu())
    return torch.cat(probabilities).numpy(), {
        key: torch.cat(value).numpy() for key, value in diagnostics.items()
    }


def candidate_metrics(probabilities, diagnostics, targets, ratios):
    targets = np.asarray(targets)
    conditional = probabilities[..., :14]
    conditional /= conditional.sum(axis=-1, keepdims=True).clip(1e-12)
    prediction = conditional.argmax(axis=-1)
    stage_accuracy = [
        float(np.mean(prediction[:, stage] == targets))
        for stage in range(prediction.shape[1])
    ]
    active_after = diagnostics["active_after"]
    contains = active_after[
        np.arange(len(targets))[:, None],
        np.arange(active_after.shape[1])[None, :],
        targets[:, None],
    ]
    active_before = diagnostics["active_before"][:, 1:]
    membership = active_before[
        np.arange(len(targets))[:, None],
        np.arange(active_before.shape[1])[None, :],
        targets[:, None],
    ]
    support_prediction = diagnostics["support_logits"][:, 1:] >= 0.0
    initial_wrong = ~contains[:, 0]
    return {
        "stage_accuracy": stage_accuracy,
        "accuracy_auc": prefix_auc(np.asarray(ratios), np.asarray(stage_accuracy)),
        "final_accuracy": stage_accuracy[-1],
        "candidate_contains_target": contains.mean(axis=0).tolist(),
        "candidate_contains_target_auc": prefix_auc(
            np.asarray(ratios), contains.mean(axis=0)
        ),
        "final_recovery_from_initial_wrong": float(
            contains[initial_wrong, -1].mean() if initial_wrong.any() else 1.0
        ),
        "support_accuracy": float(np.mean(support_prediction == membership)),
        "candidate_size": diagnostics["candidate_sizes"].mean(axis=0).tolist(),
        "mean_absolute_support_residual": float(
            np.mean(np.abs(diagnostics["support_residuals"][:, 1:]))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal candidate-membership experiment")
    parser.add_argument("--hierarchy-checkpoint", required=True)
    parser.add_argument("--evidence-checkpoint", required=True)
    parser.add_argument("--temporal-checkpoint", required=True)
    parser.add_argument("--data", default="data/shrec17_ddnet_npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--encoder-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--forward-stability-weight", type=float, default=1.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    evidence_model, evidence_checkpoint, labels = load_evidence_model(
        Path(args.evidence_checkpoint), device
    )
    hierarchy_checkpoint = torch.load(
        args.hierarchy_checkpoint, map_location=device, weights_only=True
    )
    temporal_checkpoint = torch.load(
        args.temporal_checkpoint, map_location=device, weights_only=True
    )
    seed = int(evidence_checkpoint["seed"])
    if any(seed != int(checkpoint["seed"]) for checkpoint in (
        hierarchy_checkpoint, temporal_checkpoint
    )):
        raise ValueError("all checkpoints must share the same seed")
    set_reproducible(seed)
    ratios = tuple(float(value) for value in evidence_checkpoint["observation_ratios"])
    label_to_index = {label: index for index, label in enumerate(labels)}
    reverse_map = torch.tensor([
        label_to_index[SEMANTIC_REVERSE_MAPPING[label]] for label in labels
    ], dtype=torch.long)

    masks = level_masks(hierarchy_checkpoint["tree_levels"])
    base = HierarchicalVerificationSearch(masks).to(device)
    base.load_state_dict(hierarchy_checkpoint["model_state"], strict=True)
    base.eval()
    temporal_model = TemporalPEOrderKnownness(
        order_rejection_strength=float(
            temporal_checkpoint.get("order_rejection_strength", 0.0)
        )
    ).to(device)
    temporal_model.load_state_dict(temporal_checkpoint["model_state"], strict=True)
    temporal_model.eval()
    for parameter in temporal_model.parameters():
        parameter.requires_grad_(False)

    official_train = load_raw_shrec17_npz(args.data, "train")
    test_samples = load_raw_shrec17_npz(args.data, "test")
    train_samples, validation_samples = stratified_raw_split(official_train, 0.2, seed)

    def prepare(samples):
        raw = encode_temporal_samples(
            evidence_model, samples, labels, 64, ratios,
            args.encoder_batch_size, args.eval_batch_size, device,
        )
        temporal = cache_temporal(
            temporal_model, raw, args.eval_batch_size, device
        )
        return raw[0], raw[1], raw[2], temporal

    train_forward = prepare(train_samples)
    train_reverse = prepare(reverse_with_source_labels(train_samples))
    validation_forward = prepare(validation_samples)
    validation_reverse = prepare(reverse_with_source_labels(validation_samples))
    test_forward = prepare(test_samples)
    test_reverse = prepare(reverse_with_source_labels(test_samples))

    mapped_train = reverse_map[train_reverse[2]]
    mapped_validation = reverse_map[validation_reverse[2]].numpy()
    mapped_test = reverse_map[test_reverse[2]].numpy()
    is_forward = torch.cat([
        torch.ones(len(train_forward[0]), dtype=torch.float32),
        torch.zeros(len(train_reverse[0]), dtype=torch.float32),
    ])
    training = (
        torch.cat([train_forward[0], train_reverse[0]]),
        torch.cat([train_forward[1], train_reverse[1]]),
        torch.cat([train_forward[2], mapped_train]),
        torch.cat([train_forward[3], train_reverse[3]]),
        is_forward,
    )
    loader = DataLoader(TensorDataset(*training), batch_size=args.batch_size, shuffle=True)
    model = TemporalCandidateMembership(base).to(device)
    optimizer = torch.optim.AdamW(
        model.support_residual.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    initial_forward_probability, initial_forward_diagnostics = predict(
        model, validation_forward, args.eval_batch_size, device
    )
    initial_reverse_probability, initial_reverse_diagnostics = predict(
        model, validation_reverse, args.eval_batch_size, device
    )
    baseline_validation_forward = candidate_metrics(
        initial_forward_probability, initial_forward_diagnostics,
        validation_forward[2].numpy(), ratios,
    )
    baseline_validation_reverse = candidate_metrics(
        initial_reverse_probability, initial_reverse_diagnostics,
        mapped_validation, ratios,
    )

    def selection_score(forward, reverse):
        candidate_score = 0.5 * (
            forward["candidate_contains_target_auc"]
            + reverse["candidate_contains_target_auc"]
        )
        support_score = 0.5 * (
            forward["support_accuracy"] + reverse["support_accuracy"]
        )
        forward_shortfall = max(
            0.0,
            baseline_validation_forward["candidate_contains_target_auc"]
            - 0.005 - forward["candidate_contains_target_auc"],
        )
        return 0.9 * candidate_score + 0.1 * support_score - 5.0 * forward_shortfall

    history = []
    best = {
        "epoch": 0,
        "score": selection_score(
            baseline_validation_forward, baseline_validation_reverse
        ),
        "model_state": {
            key: value.detach().cpu().clone()
            for key, value in model.support_residual.state_dict().items()
        },
    }
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.base.eval()
        total, seen = 0.0, 0
        for conditions, evidence, targets, temporal, forward_mask in loader:
            conditions, evidence = conditions.to(device), evidence.to(device)
            targets, temporal = targets.to(device), temporal.to(device)
            forward_mask = forward_mask.to(device) > 0.5
            _, diagnostics = model(conditions, evidence, temporal)
            classification = support_loss(diagnostics, targets)
            stability = (
                diagnostics["support_residuals"][forward_mask, 1:].square().mean()
                if forward_mask.any() else classification.new_zeros(())
            )
            loss = classification + args.forward_stability_weight * stability
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(targets)
            seen += len(targets)
        scheduler.step()
        forward_probability, forward_diagnostics = predict(
            model, validation_forward, args.eval_batch_size, device
        )
        reverse_probability, reverse_diagnostics = predict(
            model, validation_reverse, args.eval_batch_size, device
        )
        forward = candidate_metrics(
            forward_probability, forward_diagnostics,
            validation_forward[2].numpy(), ratios,
        )
        reverse = candidate_metrics(
            reverse_probability, reverse_diagnostics, mapped_validation, ratios,
        )
        score = selection_score(forward, reverse)
        row = {
            "epoch": epoch, "loss": total / seen, "score": score,
            "forward": forward, "reverse": reverse,
        }
        history.append(row)
        print(
            f"Candidate epoch {epoch:03d} | loss {row['loss']:.4f} | "
            f"support F/R {forward['support_accuracy']:.2%}/"
            f"{reverse['support_accuracy']:.2%} | target-in-set F/R "
            f"{forward['candidate_contains_target'][-1]:.2%}/"
            f"{reverse['candidate_contains_target'][-1]:.2%}"
        )
        if score >= best["score"]:
            best = {
                "epoch": epoch, "score": score,
                "model_state": {
                    key: value.detach().cpu().clone()
                    for key, value in model.support_residual.state_dict().items()
                },
            }

    model.support_residual.load_state_dict(best["model_state"], strict=True)

    def evaluate(data, targets):
        probabilities, diagnostics = predict(model, data, args.eval_batch_size, device)
        return candidate_metrics(probabilities, diagnostics, targets, ratios)

    # An all-zero temporal residual is the frozen candidate verifier baseline.
    trained_state = {
        key: value.detach().cpu().clone()
        for key, value in model.support_residual.state_dict().items()
    }
    for parameter in model.support_residual.parameters():
        parameter.data.zero_()
    baseline_forward = evaluate(test_forward, test_forward[2].numpy())
    baseline_reverse = evaluate(test_reverse, mapped_test)
    model.support_residual.load_state_dict(trained_state, strict=True)
    temporal_forward = evaluate(test_forward, test_forward[2].numpy())
    temporal_reverse = evaluate(test_reverse, mapped_test)

    summary = {
        "seed": seed,
        "selected_epoch": best["epoch"],
        "semantic_reverse_mapping": SEMANTIC_REVERSE_MAPPING,
        "uses_reverse_gate": False,
        "decision_target": "P(true_class_in_current_candidate_set)",
        "baseline_forward": baseline_forward,
        "baseline_reversed": baseline_reverse,
        "temporal_forward": temporal_forward,
        "temporal_reversed": temporal_reverse,
        "full_observation_uses_original_evidence_exactly": True,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment": "temporal_candidate_membership",
        "model_state": trained_state,
        "seed": seed,
        "epochs": best["epoch"],
        "uses_reverse_gate": False,
        "decision_target": "P(true_class_in_current_candidate_set)",
        "semantic_reverse_mapping": SEMANTIC_REVERSE_MAPPING,
    }, output / "best.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Temporal candidate-membership experiment", "",
        "The model does not predict forward versus reversed order. It predicts",
        "whether the true class remains inside the current candidate set and",
        "uses contradiction to backtrack into the complement.", "",
        "| Variant | Forward support | Reversed support | Forward target in final set | Reversed target in final set | Forward AUC | Reversed AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Frozen verifier | {baseline_forward['support_accuracy']:.2%} | {baseline_reverse['support_accuracy']:.2%} | {baseline_forward['candidate_contains_target'][-1]:.2%} | {baseline_reverse['candidate_contains_target'][-1]:.2%} | {baseline_forward['accuracy_auc']:.2%} | {baseline_reverse['accuracy_auc']:.2%} |",
        f"| Temporal PE membership | {temporal_forward['support_accuracy']:.2%} | {temporal_reverse['support_accuracy']:.2%} | {temporal_forward['candidate_contains_target'][-1]:.2%} | {temporal_reverse['candidate_contains_target'][-1]:.2%} | {temporal_forward['accuracy_auc']:.2%} | {temporal_reverse['accuracy_auc']:.2%} |",
        "", f"Selected epoch: `{best['epoch']}`.",
        "Full-observation known-class output is exactly the frozen original posterior.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
