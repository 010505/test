import unittest

from gesturegraph.explanation_consistency_check import (
    build_report,
    compare_importance_profiles,
    flag_explanation_regression,
    rank_groups,
    spearman_correlation,
)


class RankGroupsTests(unittest.TestCase):
    def test_orders_by_descending_drop(self):
        drops = {"thumb": 0.1, "index": 0.4, "wrist_palm": 0.05, "little": 0.2}
        self.assertEqual(rank_groups(drops), ["index", "little", "thumb", "wrist_palm"])


class SpearmanCorrelationTests(unittest.TestCase):
    def test_identical_rankings_give_correlation_one(self):
        ranking = ["index", "thumb", "wrist_palm", "little"]
        self.assertAlmostEqual(spearman_correlation(ranking, ranking), 1.0)

    def test_fully_reversed_rankings_give_negative_correlation(self):
        forward = ["a", "b", "c", "d"]
        reversed_ranking = ["d", "c", "b", "a"]
        self.assertAlmostEqual(spearman_correlation(forward, reversed_ranking), -1.0)

    def test_mismatched_label_sets_raise(self):
        with self.assertRaises(ValueError):
            spearman_correlation(["a", "b"], ["a", "c"])

    def test_single_group_returns_one(self):
        self.assertEqual(spearman_correlation(["a"], ["a"]), 1.0)


class CompareImportanceProfilesTests(unittest.TestCase):
    def test_detects_a_top_group_swap(self):
        old_drops = {"thumb": 0.4, "index": 0.1, "middle": 0.05, "ring": 0.02, "little": 0.01, "wrist_palm": 0.0}
        new_drops = {"thumb": 0.05, "index": 0.45, "middle": 0.05, "ring": 0.02, "little": 0.01, "wrist_palm": 0.0}
        comparison = compare_importance_profiles(old_drops, new_drops)
        self.assertEqual(comparison["old_top_group"], "thumb")
        self.assertEqual(comparison["new_top_group"], "index")
        self.assertTrue(comparison["top_group_changed"])

    def test_identical_profiles_show_no_change(self):
        drops = {"thumb": 0.3, "index": 0.2, "middle": 0.1, "ring": 0.05, "little": 0.02, "wrist_palm": 0.0}
        comparison = compare_importance_profiles(drops, dict(drops))
        self.assertFalse(comparison["top_group_changed"])
        self.assertAlmostEqual(comparison["rank_correlation"], 1.0)

    def test_mismatched_group_sets_raise(self):
        with self.assertRaises(ValueError):
            compare_importance_profiles({"thumb": 0.1}, {"index": 0.1})


class FlagExplanationRegressionTests(unittest.TestCase):
    def test_flags_when_top_group_changed_even_if_correlation_is_high(self):
        comparison = {"top_group_changed": True, "rank_correlation": 0.9}
        self.assertTrue(flag_explanation_regression(comparison, correlation_threshold=0.5))

    def test_flags_when_correlation_drops_below_threshold(self):
        comparison = {"top_group_changed": False, "rank_correlation": 0.2}
        self.assertTrue(flag_explanation_regression(comparison, correlation_threshold=0.5))

    def test_no_flag_when_stable_and_correlation_is_high(self):
        comparison = {"top_group_changed": False, "rank_correlation": 0.9}
        self.assertFalse(flag_explanation_regression(comparison, correlation_threshold=0.5))


class BuildReportTests(unittest.TestCase):
    def test_report_matches_a_realistic_swap_scenario(self):
        old_drops = {"thumb": 0.35, "index": 0.30, "middle": 0.10, "ring": 0.05, "little": 0.02, "wrist_palm": 0.0}
        new_drops = {"thumb": 0.08, "index": 0.32, "middle": 0.30, "ring": 0.05, "little": 0.02, "wrist_palm": 0.0}
        report = build_report(old_drops, new_drops, correlation_threshold=0.5)
        self.assertTrue(report["explanation_regression"])
        self.assertEqual(report["comparison"]["new_top_group"], "index")


if __name__ == "__main__":
    unittest.main()
