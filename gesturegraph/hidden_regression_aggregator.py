from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REGRESSION_FILENAME_PATTERN = re.compile(r"^regression_(?P<strength>[a-zA-Z]+)_vs_control_seed(?P<seed>\d+)\.json$")


def parse_strength_and_seed(filename: str) -> tuple[str, int]:
    match = REGRESSION_FILENAME_PATTERN.match(filename)
    if not match:
        raise ValueError(f"filename {filename!r} does not match the expected regression report naming pattern")
    return match.group("strength"), int(match.group("seed"))


def load_reports(directory: str | Path) -> list[tuple[str, str, int, dict]]:
    directory = Path(directory)
    reports = []
    for path in sorted(directory.glob("regression_*_vs_control_seed*.json")):
        strength, seed = parse_strength_and_seed(path.name)
        report = json.loads(path.read_text(encoding="utf-8"))
        reports.append((path.name, strength, seed, report))
    return reports


def aggregate_hidden_regressions(reports: list[tuple[str, str, int, dict]]) -> dict[str, dict]:
    per_class: dict[str, list[float]] = {}
    for _, _, _, report in reports:
        for entry in report.get("hidden_regressions", []):
            per_class.setdefault(entry["label"], []).append(entry["drop"])
    total_runs = len(reports)
    aggregated = {}
    for label, drops in sorted(per_class.items()):
        aggregated[label] = {
            "count": len(drops),
            "total_runs": total_runs,
            "fraction_of_runs": len(drops) / total_runs if total_runs else 0.0,
            "average_drop": sum(drops) / len(drops),
            "max_drop": max(drops),
        }
    return aggregated


def find_systematically_fragile_classes(aggregated: dict[str, dict]) -> list[dict]:
    fragile = [
        {"label": label, **stats}
        for label, stats in aggregated.items()
        if stats["count"] > stats["total_runs"] / 2
    ]
    return sorted(fragile, key=lambda row: (row["count"], row["average_drop"]), reverse=True)


def group_by_strength(reports: list[tuple[str, str, int, dict]]) -> dict[str, list[tuple[str, str, int, dict]]]:
    grouped: dict[str, list[tuple[str, str, int, dict]]] = {}
    for entry in reports:
        grouped.setdefault(entry[1], []).append(entry)
    return grouped


def per_strength_fragility(reports: list[tuple[str, str, int, dict]]) -> dict[str, list[dict]]:
    grouped = group_by_strength(reports)
    result = {}
    for strength, strength_reports in sorted(grouped.items()):
        aggregated = aggregate_hidden_regressions(strength_reports)
        result[strength] = find_systematically_fragile_classes(aggregated)
    return result


def build_meta_report(reports: list[tuple[str, str, int, dict]]) -> dict:
    if not reports:
        raise ValueError("no regression reports were provided")
    overall = aggregate_hidden_regressions(reports)
    return {
        "total_runs": len(reports),
        "strengths_covered": sorted({entry[1] for entry in reports}),
        "overall_per_class": overall,
        "systematically_fragile_classes": find_systematically_fragile_classes(overall),
        "per_strength_fragile_classes": per_strength_fragility(reports),
    }


def print_summary(meta_report: dict) -> None:
    print(f"Runs aggregated: {meta_report['total_runs']} across strengths {meta_report['strengths_covered']}")
    if meta_report["systematically_fragile_classes"]:
        print("\nClasses flagged as hidden regressions in a majority of all runs:")
        for row in meta_report["systematically_fragile_classes"]:
            print(f"  {row['label']:14s} {row['count']}/{row['total_runs']} runs, avg drop {row['average_drop']:.1%}")
    else:
        print("\nNo class regresses in a majority of runs.")
    for strength, rows in meta_report["per_strength_fragile_classes"].items():
        if rows:
            print(f"\n{strength}: consistently regresses {[row['label'] for row in rows]}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate hidden regressions across multiple regression_check.py reports")
    parser.add_argument("--reports", required=True, help="directory containing regression_*_vs_control_seed*.json files")
    parser.add_argument("--output", default="runs/hidden_regression_meta_report.json")
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
