from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .regression_check import load_checkpoint
from .shrec import load_shrec17


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def confidence_from_logits(logits: np.ndarray) -> float:
    return float(softmax(logits).max())


def sequence_to_tensor(sequence: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(sequence.astype(np.float32)).permute(2, 0, 1)


def predict_confidence(model, sequence: np.ndarray) -> float:
    tensor = sequence_to_tensor(sequence).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor).squeeze(0).numpy()
    return confidence_from_logits(logits)


def make_noise_sequence(reference_sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mean = reference_sequence.mean(axis=(0, 1))
    std = reference_sequence.std(axis=(0, 1))
    return rng.normal(mean, std, size=reference_sequence.shape).astype(np.float32)


def make_frozen_sequence(reference_sequence: np.ndarray) -> np.ndarray:
    frames = reference_sequence.shape[0]
    return np.repeat(reference_sequence[:1], frames, axis=0).astype(np.float32)


def make_shuffled_sequence(reference_sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(reference_sequence.shape[0])
    return reference_sequence[order].astype(np.float32)


def classify_confidence(confidence: float, threshold: float) -> str:
    return "rejected" if confidence < threshold else "falsely_confident"


def confidence_threshold_from_id(id_confidences: list[float], percentile: float) -> float:
    if not id_confidences:
        raise ValueError("need at least one id confidence to derive a threshold")
    return float(np.percentile(id_confidences, percentile))


def summarize_by_type(records: list[tuple[str, float]], threshold: float) -> dict[str, dict]:
    grouped: dict[str, list[float]] = {}
    for kind, confidence in records:
        grouped.setdefault(kind, []).append(confidence)
    summary = {}
    for kind, confidences in sorted(grouped.items()):
        rejected = sum(1 for c in confidences if classify_confidence(c, threshold) == "rejected")
        summary[kind] = {
            "mean_confidence": sum(confidences) / len(confidences),
            "rejection_rate": rejected / len(confidences),
            "count": len(confidences),
        }
    return summary


def compare_id_vs_ood(id_confidences: list[float], ood_confidences: list[float]) -> dict:
    if not id_confidences or not ood_confidences:
        raise ValueError("need at least one id and one ood confidence to compare")
    id_mean = sum(id_confidences) / len(id_confidences)
    ood_mean = sum(ood_confidences) / len(ood_confidences)
    return {
        "id_mean_confidence": id_mean,
        "ood_mean_confidence": ood_mean,
        "margin": id_mean - ood_mean,
        "ood_more_confident_than_id": ood_mean > id_mean,
    }


def build_report(id_confidences: list[float], ood_records: list[tuple[str, float]], percentile: float) -> dict:
    threshold = confidence_threshold_from_id(id_confidences, percentile)
    ood_confidences = [confidence for _, confidence in ood_records]
    return {
        "threshold": threshold,
        "threshold_percentile": percentile,
        "id_mean_confidence": sum(id_confidences) / len(id_confidences),
        "comparison": compare_id_vs_ood(id_confidences, ood_confidences),
        "by_type": summarize_by_type(ood_records, threshold),
    }


def print_summary(report: dict) -> None:
    print(f"ID mean confidence: {report['id_mean_confidence']:.1%}")
    print(f"Threshold (percentile {report['threshold_percentile']}): {report['threshold']:.1%}")
    comparison = report["comparison"]
    print(f"OOD mean confidence: {comparison['ood_mean_confidence']:.1%} (margin {comparison['margin']:+.1%})")
    if comparison["ood_more_confident_than_id"]:
        print("WARNING: model is on average more confident on garbage than on real gestures")
    for kind, stats in report["by_type"].items():
        print(f"  {kind:10s} mean={stats['mean_confidence']:.1%} rejection_rate={stats['rejection_rate']:.1%}")


def run_ood_check(model, samples, percentile: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    id_confidences = [predict_confidence(model, sample.sequence) for sample in samples]
    ood_records = []
    for sample in samples:
        ood_records.append(("noise", predict_confidence(model, make_noise_sequence(sample.sequence, rng))))
        ood_records.append(("frozen", predict_confidence(model, make_frozen_sequence(sample.sequence))))
        ood_records.append(("shuffled", predict_confidence(model, make_shuffled_sequence(sample.sequence, rng))))
    return build_report(id_confidences, ood_records, percentile)


def main():
    parser = argparse.ArgumentParser(description="Out-of-distribution confidence check")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--classes", type=int, default=14, choices=[14, 28])
    parser.add_argument("--percentile", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/ood_report.json")
    args = parser.parse_args()

    model, checkpoint = load_checkpoint(args.checkpoint)
    samples = load_shrec17(args.data, "test", int(checkpoint["frames"]), args.classes)

    report = run_ood_check(model, samples, args.percentile, args.seed)
    print_summary(report)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report written to {destination}")


if __name__ == "__main__":
    main()
