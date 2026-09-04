from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .hierarchical_markov_search import load_evidence_model, old_known_metrics
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


class ReverseAwareActionAdapter(nn.Module):
    """Gated residual correction over a frozen original 14-class posterior."""

    def __init__(self, hidden: int = 192, dropout: float = 0.2):
        super().__init__()
        input_dim = 129 + 14 + 64 + 1
        self.delta_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 14),
        )
        self.reverse_gate = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, conditions, evidence, temporal, order_logits):
        features = torch.cat([
            conditions, evidence, temporal, order_logits[..., None]
        ], dim=-1)
        delta = self.delta_head(features)
        gate_logits = self.reverse_gate(features).squeeze(-1)
        gate = gate_logits.sigmoid()
        logits = evidence.clamp_min(1e-8).log() + gate[..., None] * delta
        return logits, gate_logits


@torch.no_grad()
def cache_temporal(temporal_model, data, batch_size: int, device: torch.device):
    frames, lengths = data[3], data[4]
    temporal_model.eval()
    pooled, order = [], []
    for start in range(0, len(frames), batch_size):
        current, current_order = temporal_model.encode_temporal(
            frames[start:start + batch_size].to(device),
            lengths[start:start + batch_size].to(device),
        )
        pooled.append(current.cpu())
        order.append(current_order.cpu())
    return torch.cat(pooled), torch.cat(order)


@torch.no_grad()
def predict_adapter(model, data, batch_size: int, device: torch.device):
    conditions, evidence, _, temporal, order = data
    model.eval()
    adapted, gates = [], []
    for start in range(0, len(conditions), batch_size):
        logits, gate_logits = model(
            conditions[start:start + batch_size].to(device),
            evidence[start:start + batch_size].to(device),
            temporal[start:start + batch_size].to(device),
            order[start:start + batch_size].to(device),
        )
        adapted.append(logits.softmax(dim=-1).cpu())
        gates.append(gate_logits.sigmoid().cpu())
    return torch.cat(adapted).numpy(), torch.cat(gates).numpy()


def threshold_predictions(
    original: np.ndarray,
    adapted: np.ndarray,
    gates: np.ndarray,
    threshold: float,
) -> np.ndarray:
    use_adapter = gates >= float(threshold)
    return np.where(use_adapter[..., None], adapted, original)


def stage_accuracy(probabilities: np.ndarray, targets: np.ndarray) -> list[float]:
    predictions = probabilities.argmax(axis=-1)
    return [
        float(np.mean(predictions[:, stage] == targets))
        for stage in range(predictions.shape[1])
    ]


def write_report(output: Path, summary: dict) -> None:
    old = summary["original_forward"]
    reverse_old = summary["original_reversed"]
    forward = summary["adapted_forward"]
    reverse = summary["adapted_reversed"]
    lines = [
        "# Reverse-aware action adapter",
        "",
        "Reversed gestures use semantic 14-class labels rather than No. The",
        "adapter applies a residual correction only when its reverse gate exceeds",
        "a validation-selected threshold; otherwise it returns the original",
        "posterior exactly.",
        "",
        "| Model | Forward AUC | Forward final | Reversed AUC | Reversed final |",
        "|---|---:|---:|---:|---:|",
        f"| Original frozen model | {old['accuracy_auc']:.2%} | {old['final_accuracy']:.2%} | {reverse_old['accuracy_auc']:.2%} | {reverse_old['final_accuracy']:.2%} |",
        f"| Reverse-aware adapter | {forward['accuracy_auc']:.2%} | {forward['final_accuracy']:.2%} | {reverse['accuracy_auc']:.2%} | {reverse['final_accuracy']:.2%} |",
        "",
        f"Selected epoch: `{summary['selected_epoch']}`.",
        f"Selected reverse threshold: `{summary['selected_threshold']}`.",
        f"Forward adapter activation: `{forward['adapter_activation']:.2%}`.",
        f"Reversed adapter activation: `{reverse['adapter_activation']:.2%}`.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train semantic reversed-action correction")
    parser.add_argument("--evidence-checkpoint", required=True)
    parser.add_argument("--temporal-checkpoint", required=True)
    parser.add_argument("--data", default="data/shrec17_ddnet_npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--encoder-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--kd-weight", type=float, default=0.5)
    parser.add_argument("--gate-weight", type=float, default=0.2)
    parser.add_argument("--max-forward-drop", type=float, default=0.005)
    parser.add_argument(
        "--gate-thresholds", type=float, nargs="+",
        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    evidence_model, evidence_checkpoint, labels = load_evidence_model(
        Path(args.evidence_checkpoint), device
    )
    temporal_checkpoint = torch.load(
        args.temporal_checkpoint, map_location=device, weights_only=True
    )
    seed = int(evidence_checkpoint["seed"])
    if seed != int(temporal_checkpoint["seed"]):
        raise ValueError("evidence and temporal checkpoints must share the seed")
    ratios = tuple(float(value) for value in evidence_checkpoint["observation_ratios"])
    label_to_index = {label: index for index, label in enumerate(labels)}
    reverse_map = torch.tensor([
        label_to_index[SEMANTIC_REVERSE_MAPPING[label]] for label in labels
    ], dtype=torch.long)
    set_reproducible(seed)

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
        temporal, order = cache_temporal(
            temporal_model, raw, args.eval_batch_size, device
        )
        return raw[0], raw[1], raw[2], temporal, order

    train_forward = prepare(train_samples)
    train_reversed = prepare(reverse_with_source_labels(train_samples))
    validation_forward = prepare(validation_samples)
    validation_reversed = prepare(reverse_with_source_labels(validation_samples))
    test_forward = prepare(test_samples)
    test_reversed = prepare(reverse_with_source_labels(test_samples))

    mapped_train = reverse_map[train_reversed[2]]
    mapped_validation = reverse_map[validation_reversed[2]].numpy()
    mapped_test = reverse_map[test_reversed[2]].numpy()
    is_reverse = torch.cat([
        torch.zeros(len(train_forward[0]), dtype=torch.float32),
        torch.ones(len(train_reversed[0]), dtype=torch.float32),
    ])
    training = (
        torch.cat([train_forward[0], train_reversed[0]]),
        torch.cat([train_forward[1], train_reversed[1]]),
        torch.cat([train_forward[2], mapped_train]),
        torch.cat([train_forward[3], train_reversed[3]]),
        torch.cat([train_forward[4], train_reversed[4]]),
        is_reverse,
    )
    loader = DataLoader(TensorDataset(*training), batch_size=args.batch_size, shuffle=True)
    model = ReverseAwareActionAdapter(dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    old_validation = old_known_metrics(
        validation_forward[1], validation_forward[2], ratios
    )
    stage_weight = torch.tensor([0.5, 0.75, 1.0, 1.25, 2.0], device=device)
    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = {"loss": 0.0, "classification": 0.0, "kd": 0.0, "gate": 0.0}
        seen = 0
        for conditions, evidence, targets, temporal, order, reverse_target in loader:
            conditions = conditions.to(device)
            evidence = evidence.to(device)
            targets = targets.to(device)
            temporal = temporal.to(device)
            order = order.to(device)
            reverse_target = reverse_target.to(device)
            logits, gate_logits = model(conditions, evidence, temporal, order)
            repeated = targets[:, None].expand(-1, logits.shape[1])
            classification = F.cross_entropy(
                logits.flatten(0, 1), repeated.reshape(-1), reduction="none"
            ).reshape_as(gate_logits)
            classification = (classification * stage_weight[None, :]).mean()
            forward_mask = reverse_target < 0.5
            if forward_mask.any():
                kd = F.kl_div(
                    F.log_softmax(logits[forward_mask], dim=-1),
                    evidence[forward_mask],
                    reduction="batchmean",
                ) / logits.shape[1]
            else:
                kd = classification.new_zeros(())
            gate = F.binary_cross_entropy_with_logits(
                gate_logits,
                reverse_target[:, None].expand_as(gate_logits),
            )
            loss = classification + args.kd_weight * kd + args.gate_weight * gate
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            for name, value in (
                ("loss", loss), ("classification", classification),
                ("kd", kd), ("gate", gate),
            ):
                total[name] += float(value.detach()) * len(targets)
            seen += len(targets)
        scheduler.step()

        forward_adapted, forward_gates = predict_adapter(
            model, validation_forward, args.eval_batch_size, device
        )
        reverse_adapted, reverse_gates = predict_adapter(
            model, validation_reversed, args.eval_batch_size, device
        )
        candidates = []
        for threshold in args.gate_thresholds:
            forward_probability = threshold_predictions(
                validation_forward[1].numpy(), forward_adapted,
                forward_gates, threshold,
            )
            reverse_probability = threshold_predictions(
                validation_reversed[1].numpy(), reverse_adapted,
                reverse_gates, threshold,
            )
            forward_accuracy = stage_accuracy(
                forward_probability, validation_forward[2].numpy()
            )
            reverse_accuracy = stage_accuracy(reverse_probability, mapped_validation)
            forward_auc = prefix_auc(np.asarray(ratios), np.asarray(forward_accuracy))
            reverse_auc = prefix_auc(np.asarray(ratios), np.asarray(reverse_accuracy))
            shortfall = max(
                0.0,
                old_validation["final_known_accuracy"] - args.max_forward_drop
                - forward_accuracy[-1],
            )
            score = 0.5 * (forward_auc + reverse_auc) - 10.0 * shortfall
            candidates.append({
                "threshold": float(threshold),
                "forward_stage_accuracy": forward_accuracy,
                "reverse_stage_accuracy": reverse_accuracy,
                "forward_auc": forward_auc,
                "reverse_auc": reverse_auc,
                "score": float(score),
            })
        selected = max(candidates, key=lambda row: (row["score"], row["reverse_auc"]))
        row = {
            "epoch": epoch,
            **{f"train_{name}": value / seen for name, value in total.items()},
            "selected": selected,
        }
        history.append(row)
        print(
            f"Reverse adapter epoch {epoch:03d} | loss {row['train_loss']:.4f} | "
            f"score {selected['score']:.2%} | forward "
            f"{selected['forward_stage_accuracy'][-1]:.2%}/"
            f"{old_validation['final_known_accuracy']:.2%} | reversed "
            f"{selected['reverse_stage_accuracy'][-1]:.2%}"
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
    forward_adapted, forward_gates = predict_adapter(
        model, test_forward, args.eval_batch_size, device
    )
    reverse_adapted, reverse_gates = predict_adapter(
        model, test_reversed, args.eval_batch_size, device
    )
    forward_probability = threshold_predictions(
        test_forward[1].numpy(), forward_adapted, forward_gates, best["threshold"]
    )
    reverse_probability = threshold_predictions(
        test_reversed[1].numpy(), reverse_adapted, reverse_gates, best["threshold"]
    )
    original_forward_accuracy = stage_accuracy(
        test_forward[1].numpy(), test_forward[2].numpy()
    )
    original_reverse_accuracy = stage_accuracy(
        test_reversed[1].numpy(), mapped_test
    )
    adapted_forward_accuracy = stage_accuracy(
        forward_probability, test_forward[2].numpy()
    )
    adapted_reverse_accuracy = stage_accuracy(reverse_probability, mapped_test)

    def metric(stage, gates, threshold):
        return {
            "stage_accuracy": stage,
            "accuracy_auc": prefix_auc(np.asarray(ratios), np.asarray(stage)),
            "final_accuracy": stage[-1],
            "adapter_activation": float(np.mean(gates[:, -1] >= threshold)),
        }

    summary = {
        "seed": seed,
        "semantic_reverse_mapping": SEMANTIC_REVERSE_MAPPING,
        "original_forward": metric(
            original_forward_accuracy, np.zeros_like(forward_gates), 1.0
        ),
        "original_reversed": metric(
            original_reverse_accuracy, np.zeros_like(reverse_gates), 1.0
        ),
        "adapted_forward": metric(
            adapted_forward_accuracy, forward_gates, best["threshold"]
        ),
        "adapted_reversed": metric(
            adapted_reverse_accuracy, reverse_gates, best["threshold"]
        ),
        "selected_epoch": best["epoch"],
        "selected_threshold": best["threshold"],
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment": "reverse_aware_action_adapter",
        "model_state": best["model_state"],
        "seed": seed,
        "epochs": best["epoch"],
        "reverse_threshold": best["threshold"],
        "semantic_reverse_mapping": SEMANTIC_REVERSE_MAPPING,
        "original_posterior_is_exact_when_adapter_inactive": True,
    }, output / "best.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(output, summary)
    print(f"Wrote {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
