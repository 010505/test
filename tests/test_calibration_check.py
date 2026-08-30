import numpy as np
import unittest

from gesturegraph.calibration_check import (
    assign_bins,
    bin_statistics,
    build_report,
    expected_calibration_error,
    maximum_calibration_error,
    softmax,
)


class SoftmaxTests(unittest.TestCase):
    def test_rows_sum_to_one(self):
        logits = np.array([[1.0, 2.0, 0.5], [0.1, 0.1, 0.1]])
        probabilities = softmax(logits)
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))


class AssignBinsTests(unittest.TestCase):
    def test_confidence_one_lands_in_the_last_bin(self):
        self.assertEqual(assign_bins([1.0], num_bins=10), [9])

    def test_confidence_zero_lands_in_the_first_bin(self):
        self.assertEqual(assign_bins([0.0], num_bins=10), [0])

    def test_mid_value_lands_in_the_expected_bin(self):
        self.assertEqual(assign_bins([0.55], num_bins=10), [5])

    def test_out_of_range_confidence_raises(self):
        with self.assertRaises(ValueError):
            assign_bins([1.5], num_bins=10)

    def test_zero_bins_raises(self):
        with self.assertRaises(ValueError):
            assign_bins([0.5], num_bins=0)


class BinStatisticsTests(unittest.TestCase):
    def test_groups_samples_into_the_correct_bins(self):
        confidences = [0.05, 0.15, 0.85, 0.95]
        corrects = [True, False, True, True]
        stats = bin_statistics(confidences, corrects, num_bins=10)
        first_bin = stats[0]
        self.assertEqual(first_bin["count"], 1)
        self.assertAlmostEqual(first_bin["accuracy"], 1.0)
        last_bin = stats[9]
        self.assertEqual(last_bin["count"], 1)
        self.assertAlmostEqual(last_bin["avg_confidence"], 0.95)

    def test_empty_bins_report_zero_count(self):
        stats = bin_statistics([0.05], [True], num_bins=10)
        self.assertEqual(stats[5]["count"], 0)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            bin_statistics([0.5, 0.6], [True], num_bins=10)


class ExpectedCalibrationErrorTests(unittest.TestCase):
    def test_perfectly_calibrated_model_has_zero_ece(self):
        confidences = [0.9] * 10
        corrects = [True] * 9 + [False]  # 90% confidence, 90% accuracy
        stats = bin_statistics(confidences, corrects, num_bins=10)
        ece = expected_calibration_error(stats, total_count=10)
        self.assertAlmostEqual(ece, 0.0, places=2)

    def test_overconfident_model_has_positive_ece(self):
        confidences = [0.95] * 10
        corrects = [True] * 5 + [False] * 5  # 95% confidence, 50% accuracy
        stats = bin_statistics(confidences, corrects, num_bins=10)
        ece = expected_calibration_error(stats, total_count=10)
        self.assertAlmostEqual(ece, 0.45, places=2)

    def test_zero_total_count_raises(self):
        with self.assertRaises(ValueError):
            expected_calibration_error([], total_count=0)


class MaximumCalibrationErrorTests(unittest.TestCase):
    def test_matches_the_worst_bin(self):
        stats = [
            {"bin": 0, "count": 5, "avg_confidence": 0.1, "accuracy": 0.1},
            {"bin": 1, "count": 5, "avg_confidence": 0.9, "accuracy": 0.4},
        ]
        self.assertAlmostEqual(maximum_calibration_error(stats), 0.5)

    def test_empty_bins_give_zero(self):
        stats = [{"bin": 0, "count": 0, "avg_confidence": 0.0, "accuracy": 0.0}]
        self.assertEqual(maximum_calibration_error(stats), 0.0)


class BuildReportTests(unittest.TestCase):
    def test_report_flags_a_realistic_overconfidence_case(self):
        confidences = [0.98, 0.97, 0.96, 0.95]
        corrects = [True, False, False, True]
        report = build_report(confidences, corrects, num_bins=10)
        self.assertEqual(report["sample_count"], 4)
        self.assertAlmostEqual(report["overall_accuracy"], 0.5)
        self.assertGreater(report["expected_calibration_error"], 0.3)


if __name__ == "__main__":
    unittest.main()
