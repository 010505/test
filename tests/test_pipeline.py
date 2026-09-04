import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gesturegraph.data import GestureSample, resample_sequence, stratified_split
from gesturegraph.model import FlatMLP, HandSTGCN
from gesturegraph.shrec import load_shrec17
from gesturegraph.topology import EDGES, NUM_NODES, normalized_adjacency


class PipelineTests(unittest.TestCase):
    def test_adjacency_is_symmetric_and_finite(self):
        adjacency = normalized_adjacency()
        self.assertEqual(adjacency.shape, (NUM_NODES, NUM_NODES))
        self.assertTrue(np.allclose(adjacency, adjacency.T))
        self.assertTrue(np.isfinite(adjacency).all())
        self.assertIn((0, 2), EDGES)

    def test_resample_produces_training_shape(self):
        sequence = np.random.default_rng(1).normal(size=(19, 22, 3)).astype(np.float32)
        self.assertEqual(resample_sequence(sequence).shape, (64, 22, 3))

    def test_model_forward_shape(self):
        model = HandSTGCN(num_classes=6).eval()
        with torch.no_grad():
            output = model(torch.zeros(2, 3, 64, 22))
        self.assertEqual(tuple(output.shape), (2, 6))
        with torch.no_grad():
            self.assertEqual(tuple(HandSTGCN(6, use_graph=False)(torch.zeros(2, 3, 64, 22)).shape), (2, 6))
            self.assertEqual(tuple(HandSTGCN(6, single_frame=True)(torch.zeros(2, 3, 64, 22)).shape), (2, 6))
            self.assertEqual(tuple(FlatMLP(6)(torch.zeros(2, 3, 64, 22)).shape), (2, 6))

    def test_split_keeps_each_repeated_label_in_validation(self):
        sequence = np.zeros((64, 22, 3), dtype=np.float32)
        samples = [GestureSample(Path(f"{label}-{i}.json"), label, sequence) for label in ("a", "b") for i in range(5)]
        train, val = stratified_split(samples, val_ratio=.2)
        self.assertEqual({sample.label for sample in val}, {"a", "b"})
        self.assertEqual(len(train) + len(val), len(samples))

    def test_official_shrec_layout_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = np.array([[1, 1, 2, 1, 1, 1, 3]])
            np.savetxt(root / "train_gestures.txt", row, fmt="%d")
            target = root / "gesture_1/finger_1/subject_2/essai_1"
            target.mkdir(parents=True)
            np.savetxt(target / "skeletons_world.txt", np.arange(3 * 22 * 3).reshape(3, 66))
            samples = load_shrec17(root, "train")
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].label, "grab")
            self.assertEqual(samples[0].subject, "2")
            self.assertEqual(samples[0].sequence.shape, (64, 22, 3))


if __name__ == "__main__":
    unittest.main()
