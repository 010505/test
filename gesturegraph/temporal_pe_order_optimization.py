from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .endpoint_knownness_optimization import (
    combine_probabilities,
    conditional_action_probabilities,
    evaluate_arrays,
)
from .hierarchical_endpoint_consistency import endpoint_blend_weights
from .hierarchical_markov_search import (
    HierarchicalVerificationSearch,
    cached_evidence,
    level_masks,
    load_evidence_model,
    old_known_metrics,
)
from .progressive import PrefixSequenceDataset, load_raw_shrec17_npz, stratified_raw_split
from .progressive_benchmark import set_reproducible
from .unknown_diffusion_benchmark import (
    TEST_UNKNOWN_MODES,
    TRAIN_UNKNOWN_MODES,
    UNKNOWN_LABEL,
    make_unknown_samples,
)
from .unknown_diffusion_optimization import calibrate_no_biases


def normalized_temporal_encoding(
    lengths: torch.Tensor,
    time_steps: int,
    channels: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Continuous sinusoidal PE on each valid prefix, plus a padding mask."""
    if channels % 2:
        raise ValueError("temporal encoding channels must be even")
    if lengths.ndim != 1:
        raise ValueError("lengths must have shape [N]")
    device = lengths.device
    position = torch.arange(time_steps, device=device, dtype=dtype)[None, :]
    denominator = (lengths.to(dtype) - 1.0).clamp_min(1.0)[:, None]
    normalized = position / denominator
    frequencies = torch.pow(
        torch.tensor(2.0, device=device, dtype=dtype),
        torch.arange(channels // 2, device=device, dtype=dtype)
        / max(1, channels // 8),
    )
    angles = 2.0 * torch.pi * normalized[..., None] * frequencies[None, None, :]
    encoding = torch.cat([angles.sin(), angles.cos()], dim=-1)
    mask = position < lengths[:, None]
    return encoding * mask[..., None], mask


def reverse_valid_frames(frames: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Reverse only valid temporal positions and preserve right padding."""
    if frames.ndim != 3 or lengths.shape != (len(frames),):
        raise ValueError("frames/lengths must have shapes [N,T,C] and [N]")
    output = frames.clone()
    for index, length in enumerate(lengths.tolist()):
        output[index, :length] = frames[index, :length].flip(0)
    return output


class TemporalPEOrderKnownness(nn.Module):
    """Knownness head with explicit temporal PE and an order auxiliary task."""

    def __init__(
        self,
        frame_channels: int = 128,
        temporal_channels: int = 64,
        hidden: int = 192,
        dropout: float = 0.2,
        order_rejection_strength: float = 0.0,
    ):
        super().__init__()
        self.temporal_channels = temporal_channels
        self.order_rejection_strength = float(order_rejection_strength)
        self.frame_projection = nn.Linear(frame_channels, temporal_channels)
        self.pe_scale = nn.Parameter(torch.tensor(1.0))
        self.temporal = nn.Sequential(
            nn.Conv1d(temporal_channels, temporal_channels, 3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(temporal_channels, temporal_channels, 3, padding=1),
            nn.GELU(),
        )
        self.temporal_attention = nn.Linear(temporal_channels, 1)
        self.order_head = nn.Sequential(
            nn.LayerNorm(temporal_channels),
            nn.Linear(temporal_channels, temporal_channels // 2),
            nn.GELU(),
            nn.Linear(temporal_channels // 2, 1),
        )
        # Same causal evidence statistics as the previous dedicated detector,
        # plus the order-aware temporal representation and order score.
        causal_dim = 3 * 129 + 3 * 14 + 4
        self.knownness = nn.Sequential(
            nn.LayerNorm(causal_dim + temporal_channels + 1),
            nn.Linear(causal_dim + temporal_channels + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    @staticmethod
    def causal_features(conditions: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        previous_condition = torch.cat(
            [torch.zeros_like(conditions[:, :1]), conditions[:, :-1]], dim=1
        )
        previous_evidence = torch.cat(
            [torch.zeros_like(evidence[:, :1]), evidence[:, :-1]], dim=1
        )
        condition_delta = conditions - previous_condition
        evidence_delta = evidence - previous_evidence
        count = torch.arange(
            1, conditions.shape[1] + 1,
            device=conditions.device,
            dtype=conditions.dtype,
        )[None, :, None]
        condition_mean = conditions.cumsum(dim=1) / count
        evidence_mean = evidence.cumsum(dim=1) / count
        maximum = evidence.max(dim=-1, keepdim=True).values
        entropy = -(
            evidence.clamp_min(1e-8) * evidence.clamp_min(1e-8).log()
        ).sum(dim=-1, keepdim=True) / np.log(14.0)
        evidence_change = evidence_delta.abs().sum(dim=-1, keepdim=True)
        condition_change = condition_delta[..., :128].square().mean(
            dim=-1, keepdim=True
        ).sqrt()
        return torch.cat([
            conditions,
            condition_delta,
            condition_mean,
            evidence,
            evidence_delta,
            evidence_mean,
            maximum,
            entropy,
            evidence_change,
            condition_change,
        ], dim=-1)

    def encode_temporal(
        self,
        frames: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if frames.ndim != 4 or lengths.shape != frames.shape[:2]:
            raise ValueError("frames/lengths must have shapes [B,S,T,C] and [B,S]")
        batch, stages, time_steps, channels = frames.shape
        flat_frames = frames.reshape(batch * stages, time_steps, channels)
        flat_lengths = lengths.reshape(-1)
        projected = self.frame_projection(flat_frames)
        encoding, mask = normalized_temporal_encoding(
            flat_lengths, time_steps, self.temporal_channels, projected.dtype
        )
        temporal = self.temporal(
            (projected + self.pe_scale * encoding).transpose(1, 2)
        ).transpose(1, 2)
        scores = self.temporal_attention(temporal).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1)
        pooled = torch.einsum("nt,ntc->nc", attention, temporal)
        order_logits = self.order_head(pooled).squeeze(-1)
        return (
            pooled.reshape(batch, stages, -1),
            order_logits.reshape(batch, stages),
        )

    def forward(self, conditions, evidence, frames, lengths):
        temporal, order_logits = self.encode_temporal(frames, lengths)
        features = self.causal_features(conditions, evidence)
        known_logits = self.knownness(torch.cat([
            features, temporal, order_logits[..., None]
        ], dim=-1)).squeeze(-1)
        # Couple the auxiliary task to rejection monotonically. Correct-order
        # logits near +infinity add almost zero; low order confidence can only
        # reduce knownness and therefore cannot make an invalid order look more
        # known. This is the missing bridge between learning order and using it.
        known_logits = known_logits + self.order_rejection_strength * F.logsigmoid(
            order_logits
        )
        return known_logits, order_logits

    def reversed_order_logits(self, frames, lengths):
        batch, stages, time_steps, channels = frames.shape
        reversed_frames = reverse_valid_frames(
            frames.reshape(batch * stages, time_steps, channels),
            lengths.reshape(-1),
        ).reshape_as(frames)
        _, logits = self.encode_temporal(reversed_frames, lengths)
        return logits


@torch.no_grad()
def encode_temporal_samples(
    encoder_model,
    samples,
    labels,
    frames: int,
    ratios,
    batch_size: int,
    evidence_batch_size: int,
    device: torch.device,
):
    dataset = PrefixSequenceDataset(samples, labels, frames, ratios, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    conditions = []
    frame_features = []
    output_lengths = []
    targets = []
    encoder_model.eval()
    encoder = encoder_model.encoder
    if not hasattr(encoder, "forward_feature_map"):
        raise TypeError("encoder must expose forward_feature_map")
    for views, lengths, progress, target in loader:
        batch, stages = views.shape[:2]
        flat_views = views.flatten(0, 1).to(device)
        flat_lengths = lengths.flatten(0, 1).to(device)
        feature_map = encoder.forward_feature_map(flat_views)
        temporal = feature_map.mean(dim=-1).transpose(1, 2)
        temporal_lengths = torch.div(
            flat_lengths + 3, 4, rounding_mode="floor"
        ).clamp(min=1, max=temporal.shape[1])
        mask = torch.arange(temporal.shape[1], device=device)[None, :]
        mask = mask < temporal_lengths[:, None]
        pooled = (temporal * mask[..., None]).sum(dim=1)
        pooled = pooled / temporal_lengths.to(temporal.dtype)[:, None]
        condition = torch.cat([
            pooled.reshape(batch, stages, -1),
            progress.to(device=device, dtype=pooled.dtype)[..., None],
        ], dim=-1)
        conditions.append(condition.cpu())
        frame_features.append(temporal.reshape(batch, stages, temporal.shape[1], -1).cpu())
        output_lengths.append(temporal_lengths.reshape(batch, stages).cpu())
        targets.append(target)
    conditions = torch.cat(conditions)
    evidence = cached_evidence(
        encoder_model, conditions, evidence_batch_size, device
    )
    return (
        conditions,
        evidence,
        torch.cat(targets),
        torch.cat(frame_features),
        torch.cat(output_lengths),
    )


@torch.no_grad()
def detector_logits(model, data, batch_size: int, device: torch.device):
    conditions, evidence, _, frames, lengths = data
    model.eval()
    outputs = []
    for start in range(0, len(conditions), batch_size):
        logits, _ = model(
            conditions[start:start + batch_size].to(device),
            evidence[start:start + batch_size].to(device),
            frames[start:start + batch_size].to(device),
            lengths[start:start + batch_size].to(device),
        )
        outputs.append(logits.cpu())
    return torch.cat(outputs).numpy()


def write_report(output: Path, summary: dict) -> None:
    baseline = summary["dedicated_without_temporal_pe_order"]
    proposed = summary["temporal_pe_order"]
    lines = [
        "# Temporal PE plus order auxiliary loss",
        "",
        "The original action backbone and endpoint distribution are frozen.",
        "A parallel knownness branch consumes pre-pooling temporal feature maps,",
        "adds normalized sinusoidal time PE, and learns forward-versus-reversed",
        "order as an auxiliary task.",
        "",
        "| Model | Known AUC | False-No AUC | No AUC | Balanced AUC | Final joint known | Final action-only | Final No |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Dedicated knownness baseline | {baseline['known_accuracy_auc']:.2%} | {baseline['known_false_no_auc']:.2%} | {baseline['unknown_recall_auc']:.2%} | {baseline['balanced_accuracy_auc']:.2%} | {baseline['final_known_accuracy']:.2%} | {baseline['final_conditional_known_accuracy']:.2%} | {baseline['final_unknown_recall']:.2%} |",
        f"| Temporal PE + order loss | {proposed['known_accuracy_auc']:.2%} | {proposed['known_false_no_auc']:.2%} | {proposed['unknown_recall_auc']:.2%} | {proposed['balanced_accuracy_auc']:.2%} | {proposed['final_known_accuracy']:.2%} | {proposed['final_conditional_known_accuracy']:.2%} | {proposed['final_unknown_recall']:.2%} |",
        "",
        f"Selected epoch: `{summary['selected_epoch']}`.",
        f"Selected blend exponent: `{summary['selected_exponent']}`.",
        f"Final original prediction agreement: `{summary['final_original_prediction_agreement']:.2%}`.",
        f"Order accuracy on held-out known prefixes: `{summary['validation_order_accuracy']:.2%}`.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal PE and order-loss OOD experiment")
    parser.add_argument("--hierarchy-checkpoint", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--evidence-checkpoint", required=True)
    parser.add_argument("--data", default="data/shrec17_ddnet_npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--encoder-batch-size", type=int, default=16)
    parser.add_argument("--unknown-train-ratio", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--order-weight", type=float, default=0.2)
    parser.add_argument("--order-rejection-strength", type=float, default=0.0)
    parser.add_argument("--reverse-known-weight", type=float, default=0.2)
    parser.add_argument("--max-false-no", type=float, default=0.05)
    parser.add_argument("--blend-exponents", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--bias-min", type=float, default=-10.0)
    parser.add_argument("--bias-max", type=float, default=2.0)
    parser.add_argument("--bias-step", type=float, default=0.25)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    hierarchy_checkpoint = torch.load(
        args.hierarchy_checkpoint, map_location=device, weights_only=True
    )
    baseline_summary = json.loads(Path(args.baseline_summary).read_text(encoding="utf-8"))
    evidence_model, evidence_checkpoint, known_labels = load_evidence_model(
        Path(args.evidence_checkpoint), device
    )
    seed = int(evidence_checkpoint["seed"])
    if seed != int(hierarchy_checkpoint["seed"]) or seed != int(baseline_summary["seed"]):
        raise ValueError("all source artifacts must use the same seed")
    ratios = tuple(float(value) for value in hierarchy_checkpoint["observation_ratios"])
    labels = known_labels + [UNKNOWN_LABEL]
    set_reproducible(seed)
    random.seed(seed)

    hierarchy = HierarchicalVerificationSearch(
        level_masks(hierarchy_checkpoint["tree_levels"])
    ).to(device)
    hierarchy.load_state_dict(hierarchy_checkpoint["model_state"], strict=True)
    hierarchy.eval()
    for parameter in hierarchy.parameters():
        parameter.requires_grad_(False)

    official_train = load_raw_shrec17_npz(args.data, "train")
    test_samples = load_raw_shrec17_npz(args.data, "test")
    known_train, validation_samples = stratified_raw_split(official_train, 0.2, seed)
    unknown_pool = make_unknown_samples(known_train, TRAIN_UNKNOWN_MODES, seed + 101)
    unknown_count = max(1, round(len(known_train) * args.unknown_train_ratio))
    rng = np.random.default_rng(seed + 102)
    selected_unknown = rng.choice(len(unknown_pool), size=unknown_count, replace=False)
    unknown_train = [unknown_pool[int(index)] for index in selected_unknown]
    # Order supervision must match deployment: reverse raw coordinates first,
    # then run the frozen nonlinear backbone. Reversing an already-computed TCN
    # feature map is not equivalent and is kept out of the formal experiment.
    reversed_order_train = make_unknown_samples(
        known_train, ("reversed",), seed + 401
    )
    validation_unknown_samples = {
        mode: make_unknown_samples(validation_samples, (mode,), seed + 200 + index)
        for index, mode in enumerate(TRAIN_UNKNOWN_MODES)
    }
    test_unknown_samples = {
        mode: make_unknown_samples(test_samples, (mode,), seed + 300 + index)
        for index, mode in enumerate(TEST_UNKNOWN_MODES)
    }

    def prepare(samples):
        return encode_temporal_samples(
            evidence_model, samples, labels, 64, ratios,
            args.encoder_batch_size, args.eval_batch_size, device,
        )

    train_data = prepare(list(known_train) + unknown_train)
    reversed_order_data = prepare(reversed_order_train)
    validation_known = prepare(validation_samples)
    validation_unknown = {
        mode: prepare(samples) for mode, samples in validation_unknown_samples.items()
    }
    test_known = prepare(test_samples)
    test_unknown = {
        mode: prepare(samples) for mode, samples in test_unknown_samples.items()
    }
    old_validation = old_known_metrics(
        validation_known[1], validation_known[2], ratios
    )

    action_validation = {}
    action_test = {}
    for exponent in args.blend_exponents:
        exponent = float(exponent)
        weights = endpoint_blend_weights(ratios, exponent)
        action_validation[exponent] = {
            "known": conditional_action_probabilities(
                hierarchy, validation_known[:3], args.eval_batch_size, device, weights
            ),
            "unknown": {
                mode: conditional_action_probabilities(
                    hierarchy, data[:3], args.eval_batch_size, device, weights
                ) for mode, data in validation_unknown.items()
            },
        }
        action_test[exponent] = {
            "known": conditional_action_probabilities(
                hierarchy, test_known[:3], args.eval_batch_size, device, weights
            ),
            "unknown": {
                mode: conditional_action_probabilities(
                    hierarchy, data[:3], args.eval_batch_size, device, weights
                ) for mode, data in test_unknown.items()
            },
        }

    model = TemporalPEOrderKnownness(
        dropout=args.dropout,
        order_rejection_strength=args.order_rejection_strength,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loader = DataLoader(TensorDataset(*train_data), batch_size=args.batch_size, shuffle=True)
    known_count = len(known_train)
    order_loader = DataLoader(
        TensorDataset(
            train_data[3][:known_count],
            train_data[4][:known_count],
            reversed_order_data[0],
            reversed_order_data[1],
            reversed_order_data[3],
            reversed_order_data[4],
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    stage_weight = torch.tensor([0.5, 0.75, 1.0, 1.25, 2.0], device=device)
    bias_grid = np.arange(
        args.bias_min, args.bias_max + 0.5 * args.bias_step, args.bias_step
    ).tolist()
    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "knownness": 0.0,
            "order": 0.0,
            "reverse_known": 0.0,
        }
        seen = 0
        order_iterator = iter(order_loader)
        for conditions, evidence, targets, frames, lengths in loader:
            conditions = conditions.to(device)
            evidence = evidence.to(device)
            targets = targets.to(device)
            frames = frames.to(device)
            lengths = lengths.to(device)
            known_target = (targets < 14).to(conditions.dtype)
            known_logits, forward_order = model(
                conditions, evidence, frames, lengths
            )
            known_loss = F.binary_cross_entropy_with_logits(
                known_logits,
                known_target[:, None].expand_as(known_logits),
                reduction="none",
            )
            known_loss = (known_loss * stage_weight[None, :]).mean()
            try:
                (
                    order_frames,
                    order_lengths,
                    reversed_conditions,
                    reversed_evidence,
                    reversed_frames,
                    reversed_lengths,
                ) = next(order_iterator)
            except StopIteration:
                order_iterator = iter(order_loader)
                (
                    order_frames,
                    order_lengths,
                    reversed_conditions,
                    reversed_evidence,
                    reversed_frames,
                    reversed_lengths,
                ) = next(order_iterator)
            order_frames = order_frames.to(device)
            order_lengths = order_lengths.to(device)
            reversed_frames = reversed_frames.to(device)
            reversed_lengths = reversed_lengths.to(device)
            reversed_conditions = reversed_conditions.to(device)
            reversed_evidence = reversed_evidence.to(device)
            _, positive_order = model.encode_temporal(order_frames, order_lengths)
            _, negative_order = model.encode_temporal(
                reversed_frames, reversed_lengths
            )
            order_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(
                    positive_order, torch.ones_like(positive_order)
                )
                + F.binary_cross_entropy_with_logits(
                    negative_order, torch.zeros_like(negative_order)
                )
            )
            reversed_known_logits, _ = model(
                reversed_conditions,
                reversed_evidence,
                reversed_frames,
                reversed_lengths,
            )
            reverse_known_loss = F.binary_cross_entropy_with_logits(
                reversed_known_logits, torch.ones_like(reversed_known_logits)
            )
            loss = (
                known_loss
                + args.order_weight * order_loss
                + args.reverse_known_weight * reverse_known_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.detach()) * len(targets)
            totals["knownness"] += float(known_loss.detach()) * len(targets)
            totals["order"] += float(order_loss.detach()) * len(targets)
            totals["reverse_known"] += float(reverse_known_loss.detach()) * len(targets)
            seen += len(targets)
        scheduler.step()

        validation_known_logits = detector_logits(
            model, validation_known, args.eval_batch_size, device
        )
        validation_unknown_logits = {
            mode: detector_logits(model, data, args.eval_batch_size, device)
            for mode, data in validation_unknown.items()
        }
        epoch_candidates = []
        for exponent in args.blend_exponents:
            exponent = float(exponent)
            action = action_validation[exponent]
            known_uncalibrated = combine_probabilities(
                action["known"], validation_known_logits
            )
            unknown_uncalibrated = np.concatenate([
                combine_probabilities(action["unknown"][mode], validation_unknown_logits[mode])
                for mode in validation_unknown
            ])
            biases, calibration = calibrate_no_biases(
                known_uncalibrated,
                validation_known[2].numpy(),
                unknown_uncalibrated,
                old_validation["stage_accuracy"],
                max_known_drop=0.0,
                max_false_no=args.max_false_no,
                bias_grid=bias_grid,
            )
            known_probabilities = combine_probabilities(
                action["known"], validation_known_logits, biases
            )
            unknown_probabilities = {
                mode: combine_probabilities(
                    action["unknown"][mode], validation_unknown_logits[mode], biases
                ) for mode in validation_unknown
            }
            metrics = evaluate_arrays(
                known_probabilities,
                validation_known[2].numpy(),
                unknown_probabilities,
                ratios,
            )
            shortfall = max(
                0.0,
                old_validation["final_known_accuracy"] - metrics["final_known_accuracy"],
            )
            score = metrics["balanced_accuracy_auc"] - 10.0 * shortfall
            epoch_candidates.append({
                "exponent": exponent,
                "biases": biases,
                "calibration": calibration,
                "metrics": metrics,
                "score": float(score),
            })
        selected = max(
            epoch_candidates,
            key=lambda row: (row["score"], row["metrics"]["unknown_recall_auc"]),
        )
        history.append({
            "epoch": epoch,
            **{f"train_{name}": value / seen for name, value in totals.items()},
            "selected": selected,
        })
        print(
            f"Temporal PE epoch {epoch:03d} | loss {totals['loss'] / seen:.4f} | "
            f"order {totals['order'] / seen:.4f} | score {selected['score']:.2%} | "
            f"known {selected['metrics']['known_accuracy_auc']:.2%} | "
            f"No {selected['metrics']['unknown_recall_auc']:.2%} | final "
            f"{selected['metrics']['final_known_accuracy']:.2%}/"
            f"{old_validation['final_known_accuracy']:.2%}"
        )
        if best is None or selected["score"] >= best["score"]:
            best = {
                **selected,
                "epoch": epoch,
                "model_state": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
            }

    model.load_state_dict(best["model_state"], strict=True)
    test_known_logits = detector_logits(model, test_known, args.eval_batch_size, device)
    test_unknown_logits = {
        mode: detector_logits(model, data, args.eval_batch_size, device)
        for mode, data in test_unknown.items()
    }
    action = action_test[best["exponent"]]
    known_probabilities = combine_probabilities(
        action["known"], test_known_logits, best["biases"]
    )
    unknown_probabilities = {
        mode: combine_probabilities(
            action["unknown"][mode], test_unknown_logits[mode], best["biases"]
        ) for mode in test_unknown
    }
    test_metrics = evaluate_arrays(
        known_probabilities, test_known[2].numpy(), unknown_probabilities, ratios
    )
    agreement = float(np.mean(
        known_probabilities[:, -1, :14].argmax(axis=-1)
        == test_known[1][:, -1].argmax(dim=-1).numpy()
    ))
    with torch.no_grad():
        frames = validation_known[3].to(device)
        lengths = validation_known[4].to(device)
        _, forward_order = model(
            validation_known[0].to(device),
            validation_known[1].to(device),
            frames,
            lengths,
        )
        reversed_order = model.reversed_order_logits(frames, lengths)
        order_accuracy = 0.5 * (
            float((forward_order >= 0).float().mean())
            + float((reversed_order < 0).float().mean())
        )
    summary = {
        "seed": seed,
        "dedicated_without_temporal_pe_order": baseline_summary[
            "endpoint_dedicated_knownness"
        ],
        "temporal_pe_order": test_metrics,
        "selected_epoch": best["epoch"],
        "selected_exponent": best["exponent"],
        "selected_blend_weights": endpoint_blend_weights(ratios, best["exponent"]),
        "selected_no_biases": best["biases"],
        "final_original_prediction_agreement": agreement,
        "validation_order_accuracy": order_accuracy,
        "order_weight": args.order_weight,
        "order_rejection_strength": args.order_rejection_strength,
        "reverse_known_weight": args.reverse_known_weight,
        "train_unknown_ratio": args.unknown_train_ratio,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment": "temporal_pe_order_knownness",
        "model_state": best["model_state"],
        "seed": seed,
        "epochs": best["epoch"],
        "known_blend_weights": summary["selected_blend_weights"],
        "no_biases": best["biases"],
        "order_weight": args.order_weight,
        "order_rejection_strength": args.order_rejection_strength,
        "reverse_known_weight": args.reverse_known_weight,
        "temporal_pe": "normalized_continuous_sinusoidal",
        "order_auxiliary": "forward_vs_reversed_valid_prefix",
        "order_negative_generation": "raw_coordinates_reversed_before_frozen_backbone",
        "reversed_semantics": "known_action_not_unknown",
        "full_observation_preserves_original_known_distribution": True,
        "unknown_is_residual_mass": True,
        "unknown_is_peer_class": False,
    }, output / "best.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(output, summary)
    print(f"Wrote {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
