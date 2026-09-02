from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

NOISE_FILENAME_PATTERN = re.compile(r"^noise_(?P<strength>[a-zA-Z]+)_seed(?P<seed>\d+)\.json$")
BASELINE_STRENGTH = "control"


def parse_strength_and_seed(filename: str) -> tuple[str, int]:
    match = NOISE_FILENAME_PATTERN.match(filename)
    if not match:
        raise ValueError(f"filename {filename!r} does not match the expected noise report naming pattern")
    return match.group("strength"), int(match.group("seed"))


def load_reports(directory: str | Path) -> list[tuple[str, str, int, dict]]:
    directory = Path(directory)
    reports = []
    for path in sorted(directory.glob("noise_*_seed*.json")):
        strength, seed = parse_strength_and_seed(path.name)
        report = json.loads(path.read_text(encoding="utf-8"))
        reports.append((path.name, strength, seed, report))
    return reports


def group_by_strength(reports: list[tuple[str, str, int, dict]]) -> dict[str, list[tuple[str, str, int, dict]]]:
    grouped: dict[str, list[tuple[str, str, int, dict]]] = {}
    for entry in reports:
        grouped.setdefault(entry[1], []).append(entry)
    return grouped


def average_flip_rate_by_class(strength_reports: list[tuple[str, str, int, dict]]) -> dict[str, float]:
    per_class: dict[str, list[float]] = {}
    for _, _, _, report in strength_reports:
        for label, rate in report.get("flip_rate_by_class", {}).items():
            per_class.setdefault(label, []).append(rate)
    return {label: sum(rates) / len(rates) for label, rates in sorted(per_class.items())}


def compare_to_baseline(strength_average: dict[str, float], baseline_average: dict[str, float]) -> dict[str, float]:
    shared_labels = set(strength_average) & set(baseline_average)
    return {label: strength_average[label] - baseline_average[label] for label in sorted(shared_labels)}


def find_worse_than_baseline_classes(comparison: dict[str, float]) -> list[dict]:
    worse = [{"label": label, "delta": delta} for label, delta in comparison.items() if delta > 0]
    return sorted(worse, key=lambda row: row["delta"], reverse=True)


def build_meta_report(reports: list[tuple[str, str, int, dict]]) -> dict:
    if not reports:
        raise ValueError("no noise reports were provided")
    grouped = group_by_strength(reports)
    if BASELINE_STRENGTH not in grouped:
        raise ValueError(f"no baseline ({BASELINE_STRENGTH!r}) reports found among the provided files")
    baseline_average = average_flip_rate_by_class(grouped[BASELINE_STRENGTH])

    per_strength = {}
    for strength, strength_reports in sorted(grouped.items()):
        if strength == BASELINE_STRENGTH:
            continue
        strength_average = average_flip_rate_by_class(strength_reports)
        comparison = compare_to_baseline(strength_average, baseline_average)
        per_strength[strength] = {
            "average_flip_rate_by_class": strength_average,
            "delta_vs_baseline": comparison,
            "worse_than_baseline": find_worse_than_baseline_classes(comparison),
        }

    return {
        "total_runs": len(reports),
        "baseline_average_flip_rate_by_class": baseline_average,
        "per_strength": per_strength,
    }


def print_summary(meta_report: dict) -> None:
    print(f"Runs aggregated: {meta_report['total_runs']}")
    for strength, data in meta_report["per_strength"].items():
        worse = data["worse_than_baseline"]
        if worse:
            print(f"\n{strength} is noisier than baseline on: {[row['label'] for row in worse]}")
        else:
            print(f"\n{strength} is not worse than baseline on any class")


def main():
    parser = argparse.ArgumentParser(description="Aggregate noise-robustness reports and compare augmentation strengths to baseline")
    parser.add_argument("--reports", required=True, help="directory containing noise_*_seed*.json files")
    parser.add_argument("--output", default="runs/noise_fragility_meta_report.json")
    args = parser.parse_args()

    reports = load_reports(args.reports)
    meta_report = build_meta_report(reports)
    print_summary(meta_report)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(meta_report, indent=2), encoding="utf-8")
    print(f"\nFull report written to {destination}")


if __name__ == "__main__":
    main()
