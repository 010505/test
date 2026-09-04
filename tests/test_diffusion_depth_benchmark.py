import math
import unittest

from gesturegraph.diffusion_depth_benchmark import (
    BASE_BETAS,
    SUPPORTED_DEPTHS,
    resample_betas,
)
from gesturegraph.progressive import ClassDiffusionModel


class DiffusionDepthBenchmarkTests(unittest.TestCase):
    def test_all_schedules_preserve_terminal_noise(self):
        reference = math.prod(1.0 - beta for beta in BASE_BETAS)
        for depth in SUPPORTED_DEPTHS:
            betas = resample_betas(depth)
            self.assertEqual(len(betas), depth)
            self.assertAlmostEqual(
                math.prod(1.0 - beta for beta in betas), reference, places=12
            )

    def test_every_depth_constructs_the_requested_model(self):
        for depth in SUPPORTED_DEPTHS:
            model = ClassDiffusionModel(betas=resample_betas(depth))
            self.assertEqual(model.steps, depth)
            self.assertEqual(model.inference_steps, depth)

    def test_unsupported_depth_is_rejected(self):
        with self.assertRaises(ValueError):
            resample_betas(3)


if __name__ == "__main__":
    unittest.main()
