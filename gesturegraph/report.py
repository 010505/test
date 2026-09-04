from __future__ import annotations

import argparse
import json
from pathlib import Path


def percentage(value):
    return "—" if value is None else f"{value:.2%}"


def main():
    parser = argparse.ArgumentParser(description="Create a concise Markdown experiment report")
    parser.add_argument("--benchmark", default="runs/shrec17_benchmark/summary.json")
    parser.add_argument("--joint-ablation", default="runs/shrec17_benchmark/joint_ablation.json")
    parser.add_argument("--output", default="runs/shrec17_benchmark/REPORT.md")
    args = parser.parse_args()
    rows = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    lines = ["# SHREC'17 GestureGraph Experiment Report", "", "## Model comparison", "", "| Experiment | Validation | Official test |", "|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['experiment']} | {percentage(row['best_validation_accuracy'])} | {percentage(row['official_test_accuracy'])} |")
    by_name = {row["experiment"]: row for row in rows}
    if "stgcn_full" in by_name:
        full = by_name["stgcn_full"]["official_test_accuracy"]
        lines += ["", "## Ablation deltas", ""]
        for key, description in (("mlp_baseline", "Graph structure vs. flattened MLP"), ("stgcn_no_graph", "Remove spatial neighbours"), ("stgcn_single_frame", "Remove motion by repeating the middle frame")):
            if key in by_name and full is not None:
                score = by_name[key]["official_test_accuracy"]
                lines.append(f"- {description}: {percentage(score)} ({score-full:+.2%} vs full ST-GCN).")
    confusion_path = Path(args.benchmark).parent / "stgcn_full" / "test_confusion.json"
    if confusion_path.exists():
        confusion = json.loads(confusion_path.read_text(encoding="utf-8"))
        labels, matrix = confusion["labels"], confusion["matrix"]
        pairs = sorted(((matrix[i][j], labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels)) if i != j), reverse=True)[:5]
        lines += ["", "## Largest official-test confusions", ""]
        for count, truth, predicted in pairs:
            lines.append(f"- `{truth}` predicted as `{predicted}`: {count} sequences.")
    joint_path = Path(args.joint_ablation)
    if joint_path.exists():
        joint = json.loads(joint_path.read_text(encoding="utf-8"))
        lines += ["", "## Joint-group masking", "", "| Masked group | Accuracy | Drop |", "|---|---:|---:|"]
        for name, result in joint["groups"].items():
            lines.append(f"| {name} | {percentage(result['masked_accuracy'])} | {result['accuracy_drop']:+.2%} |")
    lines += ["", "## Interpretation checklist", "", "- Discuss whether ST-GCN beats the MLP baseline.", "- Separate the effects of spatial graph edges and temporal motion.", "- Inspect the largest confusion-matrix cells instead of reporting accuracy alone.", "- Include one correctly and one incorrectly classified moving-skeleton example.", "- Report failed hypotheses and data limitations honestly."]
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
