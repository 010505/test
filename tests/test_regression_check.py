import unittest

from gesturegraph.regression_check import (
    FLIP_FIXED,
    FLIP_REGRESSION,
    FLIP_STABLE_CORRECT,
    FLIP_STABLE_WRONG,
    build_report,
    classify_flips,
    find_hidden_regressions,
    per_class_accuracy,
    per_class_regression_counts,
)


class ClassifyFlipsTests(unittest.TestCase):
    def test_all_four_flip_categories_are_detected(self):
        truth = ["grab", "grab", "grab", "grab"]
        old_predictions = ["grab", "tap", "grab", "tap"]   # correct, wrong, correct, wrong
        new_predictions = ["grab", "tap", "tap", "grab"]   # correct, wrong, wrong,   correct
        flips = classify_flips(truth, old_predictions, new_predictions)
        categories = [flip["category"] for flip in flips]
        self.assertEqual(categories, [FLIP_STABLE_CORRECT, FLIP_STABLE_WRONG, FLIP_REGRESSION, FLIP_FIXED])

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            classify_flips(["grab"], ["grab", "tap"], ["grab"])

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(classify_flips([], [], []), [])


class PerClassRegressionCountsTests(unittest.TestCase):
    def test_counts_only_regressions_grouped_by_truth_label(self):
        truth = ["grab", "grab", "tap", "tap", "tap"]
        old_predictions = ["grab", "grab", "tap", "tap", "tap"]  # all correct
        new_predictions = ["grab", "tap", "tap", "grab", "tap"]  # grab->tap regresses, one tap->grab regresses
        flips = classify_flips(truth, old_predictions, new_predictions)
        counts = per_class_regression_counts(flips)
        self.assertEqual(counts, {"grab": 1, "tap": 1})

    def test_no_regressions_gives_empty_dict(self):
        truth = ["grab", "tap"]
        flips = classify_flips(truth, truth, truth)  # perfect predictions both times
        self.assertEqual(per_class_regression_counts(flips), {})


class PerClassAccuracyTests(unittest.TestCase):
    def test_accuracy_is_computed_per_label_not_globally(self):
        truth = ["grab", "grab", "tap", "tap"]
        predictions = ["grab", "tap", "tap", "tap"]  # grab: 1/2, tap: 2/2
        accuracy = per_class_accuracy(truth, predictions)
        self.assertAlmostEqual(accuracy["grab"], 0.5)
        self.assertAlmostEqual(accuracy["tap"], 1.0)


class HiddenRegressionTests(unittest.TestCase):
    def test_flags_a_class_that_got_worse_even_if_others_improved(self):
        old_accuracy = {"grab": 0.60, "tap": 0.90, "swipe": 0.70}
        new_accuracy = {"grab": 0.95, "tap": 0.70, "swipe": 0.70}  # grab way up, tap quietly down
        hidden = find_hidden_regressions(old_accuracy, new_accuracy)
        self.assertEqual(len(hidden), 1)
        self.assertEqual(hidden[0]["label"], "tap")
        self.assertAlmostEqual(hidden[0]["drop"], 0.20)

    def test_no_regressions_when_every_class_holds_or_improves(self):
        old_accuracy = {"grab": 0.60, "tap": 0.90}
        new_accuracy = {"grab": 0.70, "tap": 0.90}
        self.assertEqual(find_hidden_regressions(old_accuracy, new_accuracy), [])

    def test_a_class_missing_from_the_new_model_counts_as_zero_accuracy(self):
        # if the new label set somehow dropped a class entirely, that should
        # read as a full regression, not silently pass
        old_accuracy = {"grab": 0.80}
        new_accuracy = {}
        hidden = find_hidden_regressions(old_accuracy, new_accuracy)
        self.assertEqual(hidden[0]["new_accuracy"], 0.0)


class BuildReportTests(unittest.TestCase):
    def test_report_catches_a_realistic_hidden_regression_scenario(self):
        # old model: decent and even across two classes
        # new model: big win on "grab" masks a real loss on "tap"
        truth = ["grab"] * 10 + ["tap"] * 10
        old_predictions = (["grab"] * 6 + ["tap"] * 4) + (["tap"] * 9 + ["grab"] * 1)
        new_predictions = (["grab"] * 10) + (["tap"] * 5 + ["grab"] * 5)

        report = build_report(truth, old_predictions, new_predictions)

        self.assertAlmostEqual(report["old_overall_accuracy"], 15 / 20)
        self.assertAlmostEqual(report["new_overall_accuracy"], 15 / 20)
        # overall accuracy is identical, so a naive check would say "no change" -
        # the per-class breakdown should still catch the swap
        self.assertEqual(report["overall_direction"], "unchanged")
        self.assertEqual(len(report["hidden_regressions"]), 1)
        self.assertEqual(report["hidden_regressions"][0]["label"], "tap")

    def test_regressed_samples_list_matches_the_regression_flip_count(self):
        truth = ["grab", "grab", "grab"]
        old_predictions = ["grab", "grab", "grab"]
        new_predictions = ["grab", "tap", "tap"]
        report = build_report(truth, old_predictions, new_predictions)
        self.assertEqual(report["flip_counts"].get("regression"), 2)
        self.assertEqual(len(report["regressed_samples"]), 2)


if __name__ == "__main__":
    unittest.main()
