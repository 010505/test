import json
import tempfile
import unittest
from pathlib import Path

from gesturegraph.hidden_regression_aggregator import (
    aggregate_hidden_regressions,
    build_meta_report,
    find_systematically_fragile_classes,
    group_by_strength,
    load_reports,
    parse_strength_and_seed,
    per_strength_fragility,
)


def make_report(hidden_labels_and_drops):
    return {"hidden_regressions": [{"label": label, "old_accuracy": 0.8, "new_accuracy": 0.8 - drop, "drop": drop} for label, drop in hidden_labels_and_drops]}


class ParseStrengthAndSeedTests(unittest.TestCase):
    def test_parses_a_well_formed_filename(self):
        self.assertEqual(parse_strength_and_seed("regression_mild_vs_control_seed43.json"), ("mild", 43))

    def test_rejects_a_filename_that_does_not_match(self):
        with self.assertRaises(ValueError):
            parse_strength_and_seed("not_a_regression_report.json")


class AggregateHiddenRegressionsTests(unittest.TestCase):
    def test_counts_and_averages_drops_per_class(self):
        reports = [
            ("a.json", "mild", 42, make_report([("pinch", 0.10), ("swipe_up", 0.05)])),
            ("b.json", "mild", 43, make_report([("pinch", 0.20)])),
            ("c.json", "mild", 44, make_report([])),
        ]
        aggregated = aggregate_hidden_regressions(reports)
        self.assertEqual(aggregated["pinch"]["count"], 2)
        self.assertEqual(aggregated["pinch"]["total_runs"], 3)
        self.assertAlmostEqual(aggregated["pinch"]["average_drop"], 0.15)
        self.assertEqual(aggregated["swipe_up"]["count"], 1)

    def test_class_absent_from_every_report_is_not_included(self):
        reports = [("a.json", "mild", 42, make_report([("pinch", 0.1)]))]
        aggregated = aggregate_hidden_regressions(reports)
        self.assertNotIn("grab", aggregated)


class FindSystematicallyFragileClassesTests(unittest.TestCase):
    def test_flags_classes_in_a_strict_majority_of_runs(self):
        reports = [
            ("a.json", "mild", 42, make_report([("pinch", 0.1)])),
            ("b.json", "mild", 43, make_report([("pinch", 0.1)])),
            ("c.json", "mild", 44, make_report([])),
        ]
        aggregated = aggregate_hidden_regressions(reports)
        fragile = find_systematically_fragile_classes(aggregated)
        self.assertEqual([row["label"] for row in fragile], ["pinch"])

    def test_a_class_in_exactly_half_the_runs_is_not_flagged(self):
        reports = [
            ("a.json", "mild", 42, make_report([("pinch", 0.1)])),
            ("b.json", "mild", 43, make_report([])),
        ]
        aggregated = aggregate_hidden_regressions(reports)
        self.assertEqual(find_systematically_fragile_classes(aggregated), [])


class GroupByStrengthTests(unittest.TestCase):
    def test_groups_reports_by_their_strength(self):
        reports = [
            ("a.json", "mild", 42, make_report([])),
            ("b.json", "energy", 42, make_report([])),
            ("c.json", "mild", 43, make_report([])),
        ]
        grouped = group_by_strength(reports)
        self.assertEqual(len(grouped["mild"]), 2)
        self.assertEqual(len(grouped["energy"]), 1)


class PerStrengthFragilityTests(unittest.TestCase):
    def test_each_strength_is_evaluated_against_its_own_run_count(self):
        reports = [
            ("a.json", "mild", 42, make_report([("pinch", 0.1)])),
            ("b.json", "mild", 43, make_report([("pinch", 0.1)])),
            ("c.json", "energy", 42, make_report([("pinch", 0.1)])),
            ("d.json", "energy", 43, make_report([])),
            ("e.json", "energy", 44, make_report([])),
        ]
        result = per_strength_fragility(reports)
        self.assertEqual([row["label"] for row in result["mild"]], ["pinch"])
        self.assertEqual(result["energy"], [])


class LoadReportsAndBuildMetaReportTests(unittest.TestCase):
    def test_loads_real_files_from_disk_and_builds_a_meta_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "regression_mild_vs_control_seed42.json").write_text(json.dumps(make_report([("pinch", 0.1)])), encoding="utf-8")
            (path / "regression_mild_vs_control_seed43.json").write_text(json.dumps(make_report([("pinch", 0.1)])), encoding="utf-8")
            (path / "regression_energy_vs_control_seed42.json").write_text(json.dumps(make_report([])), encoding="utf-8")
            reports = load_reports(path)
            self.assertEqual(len(reports), 3)
            meta_report = build_meta_report(reports)
            self.assertEqual(meta_report["total_runs"], 3)
            self.assertIn("pinch", [row["label"] for row in meta_report["systematically_fragile_classes"]])

    def test_empty_report_list_raises(self):
        with self.assertRaises(ValueError):
            build_meta_report([])


if __name__ == "__main__":
    unittest.main()
