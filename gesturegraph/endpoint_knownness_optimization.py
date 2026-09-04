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

from .hierarchical_endpoint_consistency import endpoint_blend_weights
from .hierarchical_markov_search import (
    HierarchicalVerificationSearch,
    cached_evidence,
    level_masks,
    load_evidence_model,
    old_known_metrics,
    predict_controller,
)
from .progressive import load_raw_shrec17_npz, stratified_raw_split
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
from .unknown_diffusion_optimization import calibrate_no_biases


class TemporalKnownnessDetector(nn.Module):
    """Causal binary knownness head separated from action candidate search."""

    def __init__(self, hidden: int = 192, dropout: float = 0.2):
        super().__init__()
        # Current, first difference and cumulative mean for conditions/evidence,
        # plus confidence, entropy and change magnitudes.
        input_dim = 3 * 129 + 3 * 14 + 4
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
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
            1, conditions.shape[1] + 1, device=conditions.device, dtype=conditions.dtype
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

    def forward(self, conditions: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        return self.network(self.causal_features(conditions, evidence)).squeeze(-1)


@torch.no_grad()
def detector_logits(model, data, batch_size: int, device: torch.device) -> np.ndarray:
    conditions, evidence, _ = data
    model.eval()
    outputs = []
    for start in range(0, len(conditions), batch_size):
        outputs.append(model(
            conditions[start:start + batch_size].to(device),
            evidence[start:start + batch_size].to(device),
        ).cpu())
    return torch.cat(outputs).numpy()


@torch.no_grad()
def conditional_action_probabilities(
    hierarchy,
    data,
    batch_size: int,
    device: torch.device,
    weights: Sequence[float],
) -> np.ndarray:
    probabilities, _ = predict_controller(
        hierarchy,
        data,
        batch_size,
        device,
        known_blend_weights=weights,
    )
    known = probabilities[..., :14]
    return known / np.clip(known.sum(axis=-1, keepdims=True), 1e-12, None)


def combine_probabilities(
    conditional: np.ndarray,
    known_logits: np.ndarray,
    no_biases: Sequence[float] | None = None,
) -> np.ndarray:
    bias = np.zeros(known_logits.shape[1], dtype=np.float32)
    if no_biases is not None:
        bias = np.asarray(no_biases, dtype=np.float32)
    known_probability = 1.0 / (1.0 + np.exp(-(known_logits - bias[None, :])))
    return np.concatenate([
        known_probability[..., None] * conditional,
        (1.0 - known_probability)[..., None],
    ], axis=-1)


def evaluate_arrays(known_probabilities, targets, unknown_by_type, ratios) -> dict:
    pooled = []
    per_type = {}
    for mode, probabilities in unknown_by_type.items():
        pooled.append(probabilities)
        rows = stage_metrics(known_probabilities, targets, probabilities, ratios)
        per_type[mode] = {
            "stage_metrics": [asdict(row) for row in rows],
            "unknown_recall_auc": metric_auc(rows, "unknown_recall"),
            "final_unknown_recall": rows[-1].unknown_recall,
        }
    rows = stage_metrics(known_probabilities, targets, np.concatenate(pooled), ratios)
    conditional_predictions = known_probabilities[..., :14].argmax(axis=-1)
    conditional_accuracy = [
        float(np.mean(conditional_predictions[:, stage] == targets))
        for stage in range(len(ratios))
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
        "per_type": per_type,
    }


def write_report(output: Path, summary: dict) -> None:
    old = summary["old_14_test"]
    endpoint = summary["endpoint_shared_knownness"]
    dedicated = summary["endpoint_dedicated_knownness"]
    lines = [
        "# Dedicated causal knownness optimization",
        "",
        "The fourteen-class endpoint is unchanged. This experiment replaces only",
        "the shared hierarchy knownness output with a dedicated causal detector",
        "using current evidence, first differences, and cumulative statistics.",
        "",
        "| Model | Known AUC | False-No AUC | No AUC | Balanced AUC | Final known | Final conditional | Final No |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Original 14-class | {old['known_accuracy_auc']:.2%} | n/a | n/a | n/a | {old['final_known_accuracy']:.2%} | {old['final_known_accuracy']:.2%} | n/a |",
        f"| Endpoint + shared knownness | {endpoint['known_accuracy_auc']:.2%} | {endpoint['known_false_no_auc']:.2%} | {endpoint['unknown_recall_auc']:.2%} | {endpoint['balanced_accuracy_auc']:.2%} | {endpoint['final_known_accuracy']:.2%} | {endpoint['final_conditional_known_accuracy']:.2%} | {endpoint['final_unknown_recall']:.2%} |",
        f"| Endpoint + dedicated knownness | {dedicated['known_accuracy_auc']:.2%} | {dedicated['known_false_no_auc']:.2%} | {dedicated['unknown_recall_auc']:.2%} | {dedicated['balanced_accuracy_auc']:.2%} | {dedicated['final_known_accuracy']:.2%} | {dedicated['final_conditional_known_accuracy']:.2%} | {dedicated['final_unknown_recall']:.2%} |",
        "",
        f"Selected epoch: `{summary['selected_epoch']}`.",
        f"Selected blend exponent: `{summary['selected_exponent']}`.",
        f"Selected No biases: `{summary['selected_no_biases']}`.",
        f"Final original-model prediction agreement: `{summary['final_original_prediction_agreement']:.2%}`.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a dedicated causal knownness head")
    parser.add_argument("--hierarchy-checkpoint", required=True)
    parser.add_argument("--endpoint-summary", required=True)
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
    parser.add_argument("--max-false-no", type=float, default=0.05)
    parser.add_argument("--blend-exponents", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--bias-min", type=float, default=-8.0)
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
    endpoint_summary = json.loads(Path(args.endpoint_summary).read_text(encoding="utf-8"))
    evidence_model, evidence_checkpoint, known_labels = load_evidence_model(
        Path(args.evidence_checkpoint), device
    )
    seed = int(evidence_checkpoint["seed"])
    if seed != int(hierarchy_checkpoint["seed"]) or seed != int(endpoint_summary["seed"]):
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
    validation_unknown_samples = {
        mode: make_unknown_samples(validation_samples, (mode,), seed + 200 + index)
        for index, mode in enumerate(TRAIN_UNKNOWN_MODES)
    }
    test_unknown_samples = {
        mode: make_unknown_samples(test_samples, (mode,), seed + 300 + index)
        for index, mode in enumerate(TEST_UNKNOWN_MODES)
    }

    def prepare(samples):
        conditions, targets = encode_samples(
            evidence_model, samples, labels, 64, ratios,
            args.encoder_batch_size, device,
        )
        evidence = cached_evidence(
            evidence_model, conditions, args.eval_batch_size, device
        )
        return conditions, evidence, targets

    train_data = prepare(list(known_train) + unknown_train)
    validation_known = prepare(validation_samples)
    validation_unknown = {
        mode: prepare(samples) for mode, samples in validation_unknown_samples.items()
    }
    test_known = prepare(test_samples)
    test_unknown = {
        mode: prepare(samples) for mode, samples in test_unknown_samples.items()
    }
    old_validation = old_known_metrics(validation_known[1], validation_known[2], ratios)
    old_test = old_known_metrics(test_known[1], test_known[2], ratios)

    action_validation = {}
    action_test = {}
    for exponent in args.blend_exponents:
        weights = endpoint_blend_weights(ratios, exponent)
        action_validation[float(exponent)] = {
            "known": conditional_action_probabilities(
                hierarchy, validation_known, args.eval_batch_size, device, weights
            ),
            "unknown": {
                mode: conditional_action_probabilities(
                    hierarchy, data, args.eval_batch_size, device, weights
                ) for mode, data in validation_unknown.items()
            },
        }
        action_test[float(exponent)] = {
            "known": conditional_action_probabilities(
                hierarchy, test_known, args.eval_batch_size, device, weights
            ),
            "unknown": {
                mode: conditional_action_probabilities(
                    hierarchy, data, args.eval_batch_size, device, weights
                ) for mode, data in test_unknown.items()
            },
        }

    detector = TemporalKnownnessDetector(dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(detector.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loader = DataLoader(TensorDataset(*train_data), batch_size=args.batch_size, shuffle=True)
    stage_weight = torch.tensor([0.5, 0.75, 1.0, 1.25, 2.0], device=device)
    bias_grid = np.arange(
        args.bias_min, args.bias_max + 0.5 * args.bias_step, args.bias_step
    ).tolist()
    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        detector.train()
        total_loss = 0.0
        seen = 0
        for conditions, evidence, targets in loader:
            conditions = conditions.to(device)
            evidence = evidence.to(device)
            target = (targets.to(device) < 14).to(conditions.dtype)
            logits = detector(conditions, evidence)
            loss_per_stage = F.binary_cross_entropy_with_logits(
                logits, target[:, None].expand_as(logits), reduction="none"
            )
            loss = (loss_per_stage * stage_weight[None, :]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
            seen += len(targets)
        scheduler.step()

        validation_known_logits = detector_logits(
            detector, validation_known, args.eval_batch_size, device
        )
        validation_unknown_logits = {
            mode: detector_logits(detector, data, args.eval_batch_size, device)
            for mode, data in validation_unknown.items()
        }
        pooled_unknown_logits = np.concatenate(list(validation_unknown_logits.values()))
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
            final_shortfall = max(
                0.0,
                old_validation["final_known_accuracy"] - metrics["final_known_accuracy"],
            )
            score = metrics["balanced_accuracy_auc"] - 10.0 * final_shortfall
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
            "train_loss": total_loss / seen,
            "selected": selected,
        })
        print(
            f"Knownness epoch {epoch:03d} | loss {total_loss / seen:.4f} | "
            f"score {selected['score']:.2%} | known "
            f"{selected['metrics']['known_accuracy_auc']:.2%} | No "
            f"{selected['metrics']['unknown_recall_auc']:.2%} | final "
            f"{selected['metrics']['final_known_accuracy']:.2%}/"
            f"{old_validation['final_known_accuracy']:.2%}"
        )
        if best is None or selected["score"] >= best["score"]:
            best = {
                **selected,
                "epoch": epoch,
                "model_state": {
                    key: value.detach().cpu().clone()
                    for key, value in detector.state_dict().items()
                },
            }

    detector.load_state_dict(best["model_state"], strict=True)
    test_known_logits = detector_logits(detector, test_known, args.eval_batch_size, device)
    test_unknown_logits = {
        mode: detector_logits(detector, data, args.eval_batch_size, device)
        for mode, data in test_unknown.items()
    }
    test_action = action_test[best["exponent"]]
    dedicated_known = combine_probabilities(
        test_action["known"], test_known_logits, best["biases"]
    )
    dedicated_unknown = {
        mode: combine_probabilities(
            test_action["unknown"][mode], test_unknown_logits[mode], best["biases"]
        ) for mode in test_unknown
    }
    dedicated = evaluate_arrays(
        dedicated_known, test_known[2].numpy(), dedicated_unknown, ratios
    )
    agreement = float(np.mean(
        dedicated_known[:, -1, :14].argmax(axis=-1)
        == test_known[1][:, -1].argmax(dim=-1).numpy()
    ))
    summary = {
        "seed": seed,
        "old_14_test": old_test,
        "endpoint_shared_knownness": endpoint_summary["endpoint_consistent"],
        "endpoint_dedicated_knownness": dedicated,
        "selected_epoch": best["epoch"],
        "selected_exponent": best["exponent"],
        "selected_blend_weights": endpoint_blend_weights(ratios, best["exponent"]),
        "selected_no_biases": best["biases"],
        "final_original_prediction_agreement": agreement,
        "train_unknown_ratio": args.unknown_train_ratio,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment": "endpoint_dedicated_causal_knownness",
        "model_state": best["model_state"],
        "seed": seed,
        "epochs": best["epoch"],
        "known_blend_weights": summary["selected_blend_weights"],
        "no_biases": best["biases"],
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
