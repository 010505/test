import json
import tempfile
import unittest
from pathlib import Path

from gesturegraph.noise_fragility_aggregator import (
    average_flip_rate_by_class,
    build_meta_report,
    compare_to_baseline,
    find_worse_than_baseline_classes,
    group_by_strength,
    load_reports,
    parse_strength_and_seed,
)


def make_report(flip_rate_by_class):
    return {"flip_rate_by_class": flip_rate_by_class}


class ParseStrengthAndSeedTests(unittest.TestCase):
    def test_parses_a_well_formed_filename(self):
        self.assertEqual(parse_strength_and_seed("noise_mild_seed43.json"), ("mild", 43))

    def test_parses_the_control_baseline_filename(self):
        self.assertEqual(parse_strength_and_seed("noise_control_seed42.json"), ("control", 42))

    def test_rejects_a_filename_that_does_not_match(self):
        with self.assertRaises(ValueError):
            parse_strength_and_seed("some_other_report.json")


class GroupByStrengthTests(unittest.TestCase):
    def test_groups_by_strength(self):
        reports = [
            ("a.json", "control", 42, make_report({})),
            ("b.json", "mild", 42, make_report({})),
            ("c.json", "mild", 43, make_report({})),
        ]
        grouped = group_by_strength(reports)
        self.assertEqual(len(grouped["mild"]), 2)
        self.assertEqual(len(grouped["control"]), 1)


class AverageFlipRateByClassTests(unittest.TestCase):
    def test_averages_across_reports(self):
        reports = [
            ("a.json", "mild", 42, make_report({"pinch": 0.10, "grab": 0.02})),
            ("b.json", "mild", 43, make_report({"pinch": 0.20})),
        ]
        averages = average_flip_rate_by_class(reports)
        self.assertAlmostEqual(averages["pinch"], 0.15)
        self.assertAlmostEqual(averages["grab"], 0.02)


class CompareToBaselineTests(unittest.TestCase):
    def test_computes_delta_only_for_shared_labels(self):
        strength_average = {"pinch": 0.20, "grab": 0.05}
        baseline_average = {"pinch": 0.10, "swipe": 0.03}
        comparison = compare_to_baseline(strength_average, baseline_average)
        self.assertAlmostEqual(comparison["pinch"], 0.10)
        self.assertNotIn("grab", comparison)
        self.assertNotIn("swipe", comparison)


class FindWorseThanBaselineClassesTests(unittest.TestCase):
    def test_only_positive_deltas_are_flagged_worst_first(self):
        comparison = {"pinch": 0.10, "grab": -0.02, "swipe": 0.05}
        worse = find_worse_than_baseline_classes(comparison)
        self.assertEqual([row["label"] for row in worse], ["pinch", "swipe"])

    def test_no_regressions_gives_empty_list(self):
        self.assertEqual(find_worse_than_baseline_classes({"pinch": -0.01}), [])


class BuildMetaReportTests(unittest.TestCase):
    def test_realistic_case_matching_the_teammate_report_pattern(self):
        reports = [
            ("noise_control_seed42.json", "control", 42, make_report({"pinch": 0.049, "grab": 0.02})),
            ("noise_control_seed43.json", "control", 43, make_report({"pinch": 0.038, "grab": 0.02})),
            ("noise_mild_seed42.json", "mild", 42, make_report({"pinch": 0.052, "grab": 0.02})),
            ("noise_mild_seed43.json", "mild", 43, make_report({"pinch": 0.049, "grab": 0.02})),
        ]
        meta_report = build_meta_report(reports)
        self.assertIn("mild", meta_report["per_strength"])
        self.assertIn("pinch", [row["label"] for row in meta_report["per_strength"]["mild"]["worse_than_baseline"]])
        self.assertNotIn("grab", [row["label"] for row in meta_report["per_strength"]["mild"]["worse_than_baseline"]])

    def test_missing_baseline_raises(self):
        reports = [("noise_mild_seed42.json", "mild", 42, make_report({"pinch": 0.1}))]
        with self.assertRaises(ValueError):
            build_meta_report(reports)

    def test_empty_report_list_raises(self):
        with self.assertRaises(ValueError):
            build_meta_report([])


class LoadReportsFromDiskTests(unittest.TestCase):
    def test_loads_real_files_and_parses_their_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "noise_control_seed42.json").write_text(json.dumps(make_report({"pinch": 0.04})), encoding="utf-8")
            (path / "noise_mild_seed42.json").write_text(json.dumps(make_report({"pinch": 0.05})), encoding="utf-8")
            reports = load_reports(path)
            self.assertEqual(len(reports), 2)
            strengths = {entry[1] for entry in reports}
            self.assertEqual(strengths, {"control", "mild"})


if __name__ == "__main__":
    unittest.main()
