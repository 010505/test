import numpy as np
import unittest

from gesturegraph.robustness_check import (
    aggregate_by_class,
    build_report,
    find_fragile_classes,
    flip_rate_for_sample,
    perturb_sequence,
)


class PerturbSequenceTests(unittest.TestCase):
    def test_perturbation_stays_within_the_epsilon_budget(self):
        rng = np.random.default_rng(1)
        sequence = np.zeros((64, 22, 3), dtype=np.float32)
        perturbed = perturb_sequence(sequence, epsilon=0.03, rng=rng)
        self.assertLessEqual(np.abs(perturbed - sequence).max(), 0.03)

    def test_shape_and_dtype_are_preserved(self):
        rng = np.random.default_rng(1)
        sequence = np.random.default_rng(2).normal(size=(64, 22, 3)).astype(np.float32)
        perturbed = perturb_sequence(sequence, epsilon=0.02, rng=rng)
        self.assertEqual(perturbed.shape, sequence.shape)
        self.assertEqual(perturbed.dtype, np.float32)

    def test_zero_epsilon_returns_the_sequence_unchanged(self):
        rng = np.random.default_rng(1)
        sequence = np.random.default_rng(3).normal(size=(64, 22, 3)).astype(np.float32)
        perturbed = perturb_sequence(sequence, epsilon=0.0, rng=rng)
        self.assertTrue(np.array_equal(perturbed, sequence))

    def test_negative_epsilon_is_rejected(self):
        rng = np.random.default_rng(1)
        with self.assertRaises(ValueError):
            perturb_sequence(np.zeros((64, 22, 3), dtype=np.float32), epsilon=-0.01, rng=rng)


class FlipRateForSampleTests(unittest.TestCase):
    def test_already_wrong_sample_returns_none(self):
        rate = flip_rate_for_sample("grab", "tap", ["tap", "tap", "grab"])
        self.assertIsNone(rate)

    def test_counts_only_predictions_that_differ_from_the_clean_one(self):
        rate = flip_rate_for_sample("grab", "grab", ["grab", "tap", "grab", "swipe", "grab"])
        self.assertAlmostEqual(rate, 2 / 5)

    def test_fully_stable_prediction_gives_zero(self):
        rate = flip_rate_for_sample("grab", "grab", ["grab"] * 5)
        self.assertEqual(rate, 0.0)

    def test_empty_trial_list_raises(self):
        with self.assertRaises(ValueError):
            flip_rate_for_sample("grab", "grab", [])


class AggregateByClassTests(unittest.TestCase):
    def test_none_entries_are_excluded_from_the_average(self):
        records = [("grab", 0.4), ("grab", None), ("grab", 0.2), ("tap", 0.0)]
        averages = aggregate_by_class(records)
        self.assertAlmostEqual(averages["grab"], 0.3)
        self.assertAlmostEqual(averages["tap"], 0.0)

    def test_a_class_with_only_none_entries_is_left_out_entirely(self):
        records = [("grab", None), ("grab", None)]
        self.assertEqual(aggregate_by_class(records), {})


class FindFragileClassesTests(unittest.TestCase):
    def test_flags_classes_strictly_above_the_threshold_worst_first(self):
        rates = {"grab": 0.05, "tap": 0.45, "swipe": 0.25, "rotate": 0.20}
        fragile = find_fragile_classes(rates, threshold=0.2)
        labels_in_order = [row["label"] for row in fragile]
        self.assertEqual(labels_in_order, ["tap", "swipe"])

    def test_no_class_over_threshold_returns_empty_list(self):
        rates = {"grab": 0.01, "tap": 0.02}
        self.assertEqual(find_fragile_classes(rates, threshold=0.2), [])


class BuildReportTests(unittest.TestCase):
    def test_end_to_end_report_on_a_small_synthetic_case(self):
        truth = ["grab", "grab", "tap", "tap"]
        clean_predictions = ["grab", "tap", "tap", "tap"]
        perturbed_predictions = [
            ["grab", "grab", "tap"],
            ["tap", "swipe", "tap"],
            ["tap", "tap", "tap"],
            ["grab", "tap", "tap"],
        ]
        report = build_report(truth, clean_predictions, perturbed_predictions, threshold=0.2)
        self.assertEqual(report["samples_evaluated"], 3)
        self.assertEqual(report["samples_skipped_already_wrong"], 1)
        self.assertAlmostEqual(report["flip_rate_by_class"]["grab"], 1 / 3)
        self.assertAlmostEqual(report["flip_rate_by_class"]["tap"], (0 + 1 / 3) / 2)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            build_report(["grab"], ["grab", "tap"], [["grab"]])


if __name__ == "__main__":
    unittest.main()
