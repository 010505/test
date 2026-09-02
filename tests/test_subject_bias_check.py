import unittest

from gesturegraph.subject_bias_check import (
    accuracy_spread,
    build_report,
    find_biased_subjects,
    per_subject_accuracy,
)


class PerSubjectAccuracyTests(unittest.TestCase):
    def test_computes_accuracy_independently_per_subject(self):
        records = [("1", True), ("1", False), ("2", True), ("2", True)]
        accuracy = per_subject_accuracy(records)
        self.assertAlmostEqual(accuracy["1"], 0.5)
        self.assertAlmostEqual(accuracy["2"], 1.0)


class AccuracySpreadTests(unittest.TestCase):
    def test_reports_min_max_mean_std(self):
        per_subject = {"1": 0.9, "2": 0.7, "3": 0.8}
        spread = accuracy_spread(per_subject)
        self.assertAlmostEqual(spread["min"], 0.7)
        self.assertAlmostEqual(spread["max"], 0.9)
        self.assertAlmostEqual(spread["mean"], 0.8)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            accuracy_spread({})


class FindBiasedSubjectsTests(unittest.TestCase):
    def test_flags_a_subject_far_below_the_rest(self):
        per_subject = {"1": 0.85, "2": 0.88, "3": 0.90, "4": 0.87, "5": 0.30}
        biased = find_biased_subjects(per_subject)
        self.assertEqual(len(biased), 1)
        self.assertEqual(biased[0]["subject"], "5")

    def test_no_flag_when_all_subjects_are_close(self):
        per_subject = {"1": 0.85, "2": 0.88, "3": 0.90, "4": 0.87}
        self.assertEqual(find_biased_subjects(per_subject), [])

    def test_too_few_subjects_returns_empty_without_error(self):
        per_subject = {"1": 0.9, "2": 0.1}
        self.assertEqual(find_biased_subjects(per_subject), [])


class BuildReportTests(unittest.TestCase):
    def test_report_matches_a_realistic_uneven_dataset(self):
        records = (
            [("1", True)] * 9 + [("1", False)] * 1
            + [("2", True)] * 8 + [("2", False)] * 2
            + [("3", True)] * 9 + [("3", False)] * 1
            + [("4", True)] * 2 + [("4", False)] * 8
        )
        report = build_report(records)
        self.assertEqual(report["subject_count"], 4)
        self.assertEqual(report["biased_subjects"][0]["subject"], "4")


if __name__ == "__main__":
    unittest.main()
