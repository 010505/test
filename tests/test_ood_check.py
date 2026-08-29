import numpy as np
import unittest

from gesturegraph.ood_check import (
    build_report,
    classify_confidence,
    compare_id_vs_ood,
    confidence_from_logits,
    confidence_threshold_from_id,
    make_frozen_sequence,
    make_noise_sequence,
    make_shuffled_sequence,
    softmax,
    summarize_by_type,
)


class SoftmaxTests(unittest.TestCase):
    def test_output_sums_to_one(self):
        logits = np.array([2.0, 1.0, 0.1])
        probabilities = softmax(logits)
        self.assertAlmostEqual(probabilities.sum(), 1.0, places=6)

    def test_largest_logit_gets_largest_probability(self):
        logits = np.array([0.5, 3.0, -1.0])
        probabilities = softmax(logits)
        self.assertEqual(int(np.argmax(probabilities)), 1)

    def test_confidence_from_logits_matches_softmax_max(self):
        logits = np.array([0.1, 0.2, 5.0])
        self.assertAlmostEqual(confidence_from_logits(logits), float(softmax(logits).max()))


class SyntheticSequenceTests(unittest.TestCase):
    def test_frozen_sequence_repeats_the_first_frame(self):
        sequence = np.random.default_rng(1).normal(size=(64, 22, 3)).astype(np.float32)
        frozen = make_frozen_sequence(sequence)
        for frame in frozen:
            self.assertTrue(np.array_equal(frame, sequence[0]))

    def test_shuffled_sequence_keeps_the_same_frames_in_a_different_order(self):
        rng = np.random.default_rng(2)
        sequence = np.random.default_rng(1).normal(size=(64, 22, 3)).astype(np.float32)
        shuffled = make_shuffled_sequence(sequence, rng)
        self.assertEqual(shuffled.shape, sequence.shape)
        original_frames = {sequence[i].tobytes() for i in range(sequence.shape[0])}
        shuffled_frames = {shuffled[i].tobytes() for i in range(shuffled.shape[0])}
        self.assertEqual(original_frames, shuffled_frames)

    def test_noise_sequence_matches_reference_statistics(self):
        rng = np.random.default_rng(3)
        sequence = np.random.default_rng(1).normal(loc=5.0, scale=2.0, size=(500, 22, 3)).astype(np.float32)
        noise = make_noise_sequence(sequence, rng)
        self.assertEqual(noise.shape, sequence.shape)
        self.assertAlmostEqual(float(noise.mean()), float(sequence.mean()), delta=0.5)


class ClassifyConfidenceTests(unittest.TestCase):
    def test_below_threshold_is_rejected(self):
        self.assertEqual(classify_confidence(0.1, threshold=0.3), "rejected")

    def test_at_or_above_threshold_is_falsely_confident(self):
        self.assertEqual(classify_confidence(0.3, threshold=0.3), "falsely_confident")
        self.assertEqual(classify_confidence(0.9, threshold=0.3), "falsely_confident")


class ConfidenceThresholdTests(unittest.TestCase):
    def test_threshold_matches_the_requested_percentile(self):
        confidences = [0.5, 0.6, 0.7, 0.8, 0.9]
        threshold = confidence_threshold_from_id(confidences, percentile=0.0)
        self.assertAlmostEqual(threshold, 0.5)

    def test_empty_list_raises(self):
        with self.assertRaises(ValueError):
            confidence_threshold_from_id([], percentile=5.0)


class SummarizeByTypeTests(unittest.TestCase):
    def test_groups_and_averages_correctly(self):
        records = [("noise", 0.2), ("noise", 0.4), ("frozen", 0.9)]
        summary = summarize_by_type(records, threshold=0.3)
        self.assertAlmostEqual(summary["noise"]["mean_confidence"], 0.3)
        self.assertAlmostEqual(summary["noise"]["rejection_rate"], 0.5)
        self.assertAlmostEqual(summary["frozen"]["rejection_rate"], 0.0)


class CompareIdVsOodTests(unittest.TestCase):
    def test_flags_when_ood_is_more_confident_than_id(self):
        comparison = compare_id_vs_ood(id_confidences=[0.3, 0.4], ood_confidences=[0.8, 0.9])
        self.assertTrue(comparison["ood_more_confident_than_id"])
        self.assertLess(comparison["margin"], 0)

    def test_no_flag_when_id_is_more_confident(self):
        comparison = compare_id_vs_ood(id_confidences=[0.9, 0.8], ood_confidences=[0.2, 0.3])
        self.assertFalse(comparison["ood_more_confident_than_id"])

    def test_empty_inputs_raise(self):
        with self.assertRaises(ValueError):
            compare_id_vs_ood([], [0.5])
        with self.assertRaises(ValueError):
            compare_id_vs_ood([0.5], [])


class BuildReportTests(unittest.TestCase):
    def test_report_flags_a_realistic_dangerous_case(self):
        id_confidences = [0.9, 0.85, 0.95, 0.88]
        ood_records = [("noise", 0.92), ("noise", 0.9), ("frozen", 0.2), ("shuffled", 0.3)]
        report = build_report(id_confidences, ood_records, percentile=5.0)
        self.assertIn("noise", report["by_type"])
        self.assertGreater(report["by_type"]["noise"]["mean_confidence"], report["threshold"])


if __name__ == "__main__":
    unittest.main()
