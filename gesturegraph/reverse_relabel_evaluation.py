from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .hierarchical_markov_search import cached_evidence, load_evidence_model
from .progressive import RawGestureSample, load_raw_shrec17_npz
from .progressive_benchmark import prefix_auc, set_reproducible
from .unknown_diffusion_benchmark import encode_samples


OOD_MODES_WITHOUT_REVERSED = (
    "frozen", "shuffled", "noise", "splice", "joint_permuted"
)

# Semantic protocol for the SHREC'17 14-label vocabulary. Directional pairs
# swap; path-shape and oscillatory gestures retain their semantic label. Pinch
# is treated as the same interaction performed in the opposite temporal
# direction because the vocabulary contains no separate "unpinch" class.
SEMANTIC_REVERSE_MAPPING = {
    "grab": "expand",
    "expand": "grab",
    "pinch": "pinch",
    "rotation_cw": "rotation_ccw",
    "rotation_ccw": "rotation_cw",
    "tap": "tap",
    "swipe_right": "swipe_left",
    "swipe_left": "swipe_right",
    "swipe_up": "swipe_down",
    "swipe_down": "swipe_up",
    "swipe_x": "swipe_x",
    "swipe_v": "swipe_v",
    "swipe_plus": "swipe_plus",
    "shake": "shake",
}


def reverse_with_source_labels(samples):
    return [
        RawGestureSample(
            index=sample.index,
            label=sample.label,
            sequence=sample.sequence[::-1].copy(),
            split=f"{sample.split}:reversed_relabelled",
        )
        for sample in samples
    ]


def best_involutive_mapping(counts: np.ndarray) -> list[int]:
    """Maximum-support label map constrained by M(M(y)) = y."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape[0] != counts.shape[1]:
        raise ValueError("counts must be square")
    classes = counts.shape[0]

    @lru_cache(maxsize=None)
    def solve(mask: int):
        if mask == 0:
            return 0.0, ()
        first = (mask & -mask).bit_length() - 1
        remaining = mask & ~(1 << first)
        score, pairs = solve(remaining)
        best = (score + counts[first, first], ((first, first),) + pairs)
        candidates = remaining
        while candidates:
            second = (candidates & -candidates).bit_length() - 1
            next_mask = remaining & ~(1 << second)
            score, pairs = solve(next_mask)
            candidate = (
                score + counts[first, second] + counts[second, first],
                ((first, second), (second, first)) + pairs,
            )
            if candidate[0] > best[0]:
                best = candidate
            candidates &= ~(1 << second)
        return best

    _, pairs = solve((1 << classes) - 1)
    mapping = list(range(classes))
    for source, target in pairs:
        mapping[source] = target
    return mapping


def corrected_ood_metrics(metrics: dict, ratios) -> dict:
    stage_rows = []
    for stage, ratio in enumerate(ratios):
        known_accuracy = metrics["stage_metrics"][stage]["known_accuracy"]
        false_no = metrics["stage_metrics"][stage]["known_false_no_rate"]
        unknown_recall = float(np.mean([
            metrics["per_type"][mode]["stage_metrics"][stage]["unknown_recall"]
            for mode in OOD_MODES_WITHOUT_REVERSED
        ]))
        stage_rows.append({
            "ratio": float(ratio),
            "known_accuracy": known_accuracy,
            "known_false_no_rate": false_no,
            "unknown_recall": unknown_recall,
            "balanced_accuracy": 0.5 * (known_accuracy + unknown_recall),
        })

    def auc(name):
        return prefix_auc(
            np.asarray(ratios),
            np.asarray([row[name] for row in stage_rows]),
        )

    return {
        "stage_metrics": stage_rows,
        "known_accuracy_auc": auc("known_accuracy"),
        "known_false_no_auc": auc("known_false_no_rate"),
        "unknown_recall_auc": auc("unknown_recall"),
        "balanced_accuracy_auc": auc("balanced_accuracy"),
        "final_known_accuracy": stage_rows[-1]["known_accuracy"],
        "final_known_false_no_rate": stage_rows[-1]["known_false_no_rate"],
        "final_unknown_recall": stage_rows[-1]["unknown_recall"],
        "included_unknown_modes": list(OOD_MODES_WITHOUT_REVERSED),
    }


def write_report(output: Path, summary: dict) -> None:
    labels = summary["labels"]
    mapping = summary["reverse_mapping"]
    baseline = summary["corrected_baseline_ood"]
    proposed = summary["corrected_temporal_ood"]
    reverse = summary["reverse_action_evaluation"]
    lines = [
        "# Reversed-action relabelling evaluation",
        "",
        "Reversed sequences are evaluated as labelled known actions, not as No.",
        "The primary mapping follows the vocabulary semantics and is involutive.",
        "A model-derived mapping from official-training predictions is retained",
        "only as a confusion diagnostic; it is never used as ground truth.",
        "",
        "## Learned mapping",
        "",
        "| Source | Reversed label | Train prediction agreement | Test accuracy |",
        "|---|---|---:|---:|",
    ]
    for index, label in enumerate(labels):
        row = reverse["per_class"][label]
        lines.append(
            f"| {label} | {mapping[label]} | {row['training_mapping_support']:.2%} | "
            f"{row['test_accuracy']:.2%} |"
        )
    lines.extend([
        "",
        "## Reversed labelled-action accuracy",
        "",
        f"Stage accuracy: `{reverse['stage_accuracy']}`.",
        f"Final accuracy: `{reverse['final_accuracy']:.2%}`.",
        f"Final false rejection as No: `{reverse['final_false_no_rate']:.2%}`.",
        "",
        "## Corrected OOD result (reversed excluded)",
        "",
        "| Model | Known AUC | False-No AUC | No AUC | Balanced AUC | Final known | Final No |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Dedicated knownness | {baseline['known_accuracy_auc']:.2%} | {baseline['known_false_no_auc']:.2%} | {baseline['unknown_recall_auc']:.2%} | {baseline['balanced_accuracy_auc']:.2%} | {baseline['final_known_accuracy']:.2%} | {baseline['final_unknown_recall']:.2%} |",
        f"| Temporal PE + order semantics | {proposed['known_accuracy_auc']:.2%} | {proposed['known_false_no_auc']:.2%} | {proposed['unknown_recall_auc']:.2%} | {proposed['balanced_accuracy_auc']:.2%} | {proposed['final_known_accuracy']:.2%} | {proposed['final_unknown_recall']:.2%} |",
    ])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel reversed gestures as known actions")
    parser.add_argument("--evidence-checkpoint", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--temporal-summary", required=True)
    parser.add_argument("--data", default="data/shrec17_ddnet_npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--encoder-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    model, checkpoint, labels = load_evidence_model(
        Path(args.evidence_checkpoint), device
    )
    seed = int(checkpoint["seed"])
    ratios = tuple(float(value) for value in checkpoint["observation_ratios"])
    set_reproducible(seed)
    train = reverse_with_source_labels(load_raw_shrec17_npz(args.data, "train"))
    test = reverse_with_source_labels(load_raw_shrec17_npz(args.data, "test"))

    def predict(samples):
        conditions, targets = encode_samples(
            model, samples, labels, args.frames, ratios,
            args.encoder_batch_size, device,
        )
        probabilities = cached_evidence(
            model, conditions, args.eval_batch_size, device
        )
        return probabilities.argmax(dim=-1).numpy(), targets.numpy()

    train_predictions, train_targets = predict(train)
    test_predictions, test_targets = predict(test)
    counts = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for source, prediction in zip(train_targets, train_predictions[:, -1]):
        counts[source, prediction] += 1
    learned_mapping = np.asarray(best_involutive_mapping(counts), dtype=np.int64)
    label_to_index = {label: index for index, label in enumerate(labels)}
    if set(SEMANTIC_REVERSE_MAPPING) != set(labels):
        raise ValueError("semantic reverse mapping must cover checkpoint labels")
    mapping = np.asarray([
        label_to_index[SEMANTIC_REVERSE_MAPPING[label]] for label in labels
    ], dtype=np.int64)
    if not np.all(mapping[mapping] == np.arange(len(labels))):
        raise ValueError("semantic reverse mapping must be involutive")
    mapped_test_targets = mapping[test_targets]
    stage_accuracy = [
        float(np.mean(test_predictions[:, stage] == mapped_test_targets))
        for stage in range(len(ratios))
    ]

    baseline_summary = json.loads(Path(args.baseline_summary).read_text(encoding="utf-8"))
    temporal_summary = json.loads(Path(args.temporal_summary).read_text(encoding="utf-8"))
    baseline_metrics = baseline_summary["endpoint_dedicated_knownness"]
    temporal_metrics = temporal_summary["temporal_pe_order"]
    per_class = {}
    for source, label in enumerate(labels):
        selected = test_targets == source
        per_class[label] = {
            "mapped_label": labels[mapping[source]],
            "training_mapping_support": float(
                counts[source, mapping[source]] / max(1, counts[source].sum())
            ),
            "test_accuracy": float(np.mean(
                test_predictions[selected, -1] == mapping[source]
            )),
            "samples": int(selected.sum()),
        }
    summary = {
        "seed": seed,
        "labels": labels,
        "reverse_mapping": {
            label: labels[mapping[index]] for index, label in enumerate(labels)
        },
        "diagnostic_model_derived_mapping": {
            label: labels[learned_mapping[index]] for index, label in enumerate(labels)
        },
        "mapping_source": "SHREC17_label_semantics",
        "mapping_is_involutive": bool(np.all(mapping[mapping] == np.arange(len(labels)))),
        "reverse_action_evaluation": {
            "stage_accuracy": stage_accuracy,
            "accuracy_auc": prefix_auc(np.asarray(ratios), np.asarray(stage_accuracy)),
            "final_accuracy": stage_accuracy[-1],
            "final_false_no_rate": temporal_metrics["per_type"]["reversed"]
            ["final_unknown_recall"],
            "per_class": per_class,
            "train_confusion_counts": counts.tolist(),
        },
        "corrected_baseline_ood": corrected_ood_metrics(baseline_metrics, ratios),
        "corrected_temporal_ood": corrected_ood_metrics(temporal_metrics, ratios),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(output, summary)
    print(f"Wrote {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
