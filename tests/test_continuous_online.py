import unittest

import numpy as np

from gesturegraph.continuous_online import (
    continuous_decisions,
    dense_observation_ratios,
)


class ContinuousOnlineTest(unittest.TestCase):
    def test_dense_ratios_cover_every_frame_and_endpoint(self):
        ratios = dense_observation_ratios(64, 16, 1)
        self.assertEqual(len(ratios), 49)
        self.assertEqual(ratios[0], 0.25)
        self.assertEqual(ratios[-1], 1.0)

    def test_dense_ratios_append_endpoint_for_stride(self):
        ratios = dense_observation_ratios(64, 16, 7)
        self.assertEqual(ratios[-1], 1.0)

    def test_decision_requires_consecutive_stability(self):
        probabilities = np.asarray(
            [[[0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.1, 0.9]]],
            dtype=np.float32,
        )
        result = continuous_decisions(
            probabilities,
            np.asarray([1]),
            (0.25, 0.50, 0.75, 1.0),
            confidence=0.7,
            margin=0.2,
            stable_updates=3,
        )
        self.assertEqual(int(result.indices[0]), 3)
        self.assertEqual(int(result.predictions[0]), 1)

    def test_decision_respects_minimum_ratio(self):
        probabilities = np.asarray(
            [[[0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]],
            dtype=np.float32,
        )
        result = continuous_decisions(
            probabilities,
            np.asarray([1]),
            (0.25, 0.50, 0.75, 1.0),
            confidence=0.7,
            margin=0.2,
            stable_updates=2,
            min_decision_ratio=0.75,
        )
        self.assertEqual(int(result.indices[0]), 2)


if __name__ == "__main__":
    unittest.main()
