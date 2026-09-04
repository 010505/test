from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .hierarchical_markov_search import (
    HierarchicalVerificationSearch,
    cached_evidence,
    evaluate_controller,
    level_masks,
    load_evidence_model,
    old_known_metrics,
    predict_controller,
)
from .progressive import DEFAULT_OBSERVATION_RATIOS, load_raw_shrec17_npz, stratified_raw_split
from .progressive_benchmark import set_reproducible
from .unknown_diffusion_benchmark import (
    TEST_UNKNOWN_MODES,
    TRAIN_UNKNOWN_MODES,
    UNKNOWN_LABEL,
    encode_samples,
    make_unknown_samples,
)
from .unknown_diffusion_optimization import calibrate_no_biases


def endpoint_blend_weights(ratios: Sequence[float], exponent: float) -> list[float]:
    """Return a monotonic handoff schedule whose final weight is exactly one."""
    values = np.asarray(ratios, dtype=np.float64)
    if len(values) < 2 or not np.all(np.diff(values) > 0):
        raise ValueError("observation ratios must be strictly increasing")
    progress = (values - values[0]) / (values[-1] - values[0])
    weights = np.power(np.clip(progress, 0.0, 1.0), float(exponent))
    weights[0] = 0.0
    weights[-1] = 1.0
    return weights.tolist()


@torch.no_grad()
def collect_probabilities(
    model: HierarchicalVerificationSearch,
    known,
    unknown_by_type,
    batch_size: int,
    device: torch.device,
    blend_weights: Sequence[float],
):
    known_probabilities, _ = predict_controller(
        model,
        known,
        batch_size,
        device,
        known_blend_weights=blend_weights,
    )
    unknown_probabilities = []
    for data in unknown_by_type.values():
        probabilities, _ = predict_controller(
            model,
            data,
            batch_size,
            device,
            known_blend_weights=blend_weights,
        )
        unknown_probabilities.append(probabilities)
    return known_probabilities, np.concatenate(unknown_probabilities, axis=0)


@torch.no_grad()
def final_evidence_agreement(
    model: HierarchicalVerificationSearch,
    known,
    batch_size: int,
    device: torch.device,
    blend_weights: Sequence[float],
) -> float:
    probabilities, _ = predict_controller(
        model,
        known,
        batch_size,
        device,
        known_blend_weights=blend_weights,
    )
    endpoint_prediction = probabilities[:, -1, :14].argmax(axis=-1)
    evidence_prediction = known[1][:, -1].argmax(dim=-1).numpy()
    return float(np.mean(endpoint_prediction == evidence_prediction))


def write_report(output: Path, summary: dict) -> None:
    old = summary["old_14_test"]
    current = summary["hierarchical_current"]
    endpoint = summary["endpoint_consistent"]
    lines = [
        "# Endpoint-consistent hierarchical Markov search",
        "",
        "Candidate search controls early decisions. Its conditional known-class",
        "distribution is progressively blended back into the frozen fourteen-class",
        "evidence distribution, with blend weight exactly 1.0 at 100% frames.",
        "No remains a separate residual knownness decision and is calibrated only",
        "on validation data.",
        "",
        f"Selected handoff exponent: `{summary['selected_exponent']}`.",
        f"Selected knownness inheritance: `{summary['selected_known_inheritance']}`.",
        f"Blend weights: `{summary['selected_blend_weights']}`.",
        f"No biases: `{summary['selected_no_biases']}`.",
        "",
        "| Model | Known AUC | False-No AUC | No AUC | Balanced AUC | Final known | Final conditional known | Final No |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Original 14-class | {old['known_accuracy_auc']:.2%} | n/a | n/a | n/a | {old['final_known_accuracy']:.2%} | {old['final_known_accuracy']:.2%} | n/a |",
        f"| Current hierarchy | {current['known_accuracy_auc']:.2%} | {current['known_false_no_auc']:.2%} | {current['unknown_recall_auc']:.2%} | {current['balanced_accuracy_auc']:.2%} | {current['final_known_accuracy']:.2%} | {current['final_conditional_known_accuracy']:.2%} | {current['final_unknown_recall']:.2%} |",
        f"| Endpoint-consistent + calibrated No | {endpoint['known_accuracy_auc']:.2%} | {endpoint['known_false_no_auc']:.2%} | {endpoint['unknown_recall_auc']:.2%} | {endpoint['balanced_accuracy_auc']:.2%} | {endpoint['final_known_accuracy']:.2%} | {endpoint['final_conditional_known_accuracy']:.2%} | {endpoint['final_unknown_recall']:.2%} |",
        "",
        f"Final known-class agreement with the original model: `{summary['final_original_prediction_agreement']:.2%}`.",
        f"Validation-selected checkpoint epoch: `{summary['hierarchy_selected_epoch']}`.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore the original 14-class endpoint")
    parser.add_argument("--hierarchy-checkpoint", required=True)
    parser.add_argument("--evidence-checkpoint", required=True)
    parser.add_argument("--data", default="data/shrec17_ddnet_npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--encoder-batch-size", type=int, default=16)
    parser.add_argument("--max-false-no", type=float, default=0.05)
    parser.add_argument("--blend-exponents", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument(
        "--known-inheritances", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75]
    )
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
    evidence_model, evidence_checkpoint, known_labels = load_evidence_model(
        Path(args.evidence_checkpoint), device
    )
    seed = int(evidence_checkpoint["seed"])
    if seed != int(hierarchy_checkpoint["seed"]):
        raise ValueError("hierarchy and evidence checkpoints must use the same seed")
    ratios = tuple(
        float(value)
        for value in hierarchy_checkpoint.get(
            "observation_ratios", DEFAULT_OBSERVATION_RATIOS
        )
    )
    labels = known_labels + [UNKNOWN_LABEL]
    set_reproducible(seed)

    model = HierarchicalVerificationSearch(
        level_masks(hierarchy_checkpoint["tree_levels"])
    ).to(device)
    model.load_state_dict(hierarchy_checkpoint["model_state"], strict=True)
    model.eval()

    official_train = load_raw_shrec17_npz(args.data, "train")
    test_samples = load_raw_shrec17_npz(args.data, "test")
    _, validation_samples = stratified_raw_split(official_train, args.val_ratio, seed)
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
            evidence_model,
            samples,
            labels,
            args.frames,
            ratios,
            args.encoder_batch_size,
            device,
        )
        evidence = cached_evidence(
            evidence_model, conditions, args.eval_batch_size, device
        )
        return conditions, evidence, targets

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
    old_test = old_known_metrics(test_known[1], test_known[2], ratios)
    bias_grid = np.arange(
        args.bias_min, args.bias_max + 0.5 * args.bias_step, args.bias_step
    ).tolist()

    candidates = []
    for inheritance in args.known_inheritances:
        if not 0.0 <= inheritance <= 1.0:
            raise ValueError("known inheritance must be in [0, 1]")
        model.known_inheritance = float(inheritance)
        for exponent in args.blend_exponents:
            weights = endpoint_blend_weights(ratios, exponent)
            known_probabilities, unknown_probabilities = collect_probabilities(
                model,
                validation_known,
                validation_unknown,
                args.eval_batch_size,
                device,
                weights,
            )
            biases, calibration = calibrate_no_biases(
                known_probabilities,
                validation_known[2].numpy(),
                unknown_probabilities,
                old_validation["stage_accuracy"],
                max_known_drop=0.0,
                max_false_no=args.max_false_no,
                bias_grid=bias_grid,
            )
            metrics = evaluate_controller(
                model,
                validation_known,
                validation_unknown,
                ratios,
                args.eval_batch_size,
                device,
                known_blend_weights=weights,
                no_biases=biases,
            )
            final_shortfall = max(
                0.0,
                old_validation["final_known_accuracy"] - metrics["final_known_accuracy"],
            )
            score = metrics["balanced_accuracy_auc"] - 10.0 * final_shortfall
            candidates.append({
                "exponent": float(exponent),
                "known_inheritance": float(inheritance),
                "weights": weights,
                "biases": biases,
                "calibration": calibration,
                "validation": metrics,
                "selection_score": float(score),
            })
            print(
                f"Inheritance {inheritance:g} | exponent {exponent:g} | "
                f"score {score:.2%} | known {metrics['known_accuracy_auc']:.2%} | "
                f"final {metrics['final_known_accuracy']:.2%}/"
                f"{old_validation['final_known_accuracy']:.2%} original | "
                f"No {metrics['unknown_recall_auc']:.2%}"
            )

    selected = max(
        candidates,
        key=lambda row: (
            row["selection_score"],
            row["validation"]["unknown_recall_auc"],
            -row["exponent"],
        ),
    )
    model.known_inheritance = 0.5
    current = evaluate_controller(
        model,
        test_known,
        test_unknown,
        ratios,
        args.eval_batch_size,
        device,
    )
    model.known_inheritance = selected["known_inheritance"]
    endpoint = evaluate_controller(
        model,
        test_known,
        test_unknown,
        ratios,
        args.eval_batch_size,
        device,
        known_blend_weights=selected["weights"],
        no_biases=selected["biases"],
    )
    agreement = final_evidence_agreement(
        model,
        test_known,
        args.eval_batch_size,
        device,
        selected["weights"],
    )
    summary = {
        "seed": seed,
        "old_14_test": old_test,
        "hierarchical_current": current,
        "endpoint_consistent": endpoint,
        "selected_exponent": selected["exponent"],
        "selected_known_inheritance": selected["known_inheritance"],
        "selected_blend_weights": selected["weights"],
        "selected_no_biases": selected["biases"],
        "final_original_prediction_agreement": agreement,
        "hierarchy_selected_epoch": int(hierarchy_checkpoint["epochs"]),
        "validation_candidates": candidates,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment": "hierarchical_endpoint_consistency",
        "hierarchy_model_state": hierarchy_checkpoint["model_state"],
        "tree_levels": hierarchy_checkpoint["tree_levels"],
        "known_labels": known_labels,
        "labels": labels,
        "seed": seed,
        "frames": args.frames,
        "observation_ratios": list(ratios),
        "known_blend_weights": selected["weights"],
        "no_biases": selected["biases"],
        "known_inheritance": selected["known_inheritance"],
        "full_observation_preserves_original_known_distribution": True,
        "unknown_is_residual_mass": True,
        "unknown_is_peer_class": False,
    }, output / "best.pt")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_report(output, summary)
    print(f"Wrote {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
