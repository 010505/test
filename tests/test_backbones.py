import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from gesturegraph.backbones import (
    LEGACY_EXPERIMENTAL_MODEL_NAMES,
    AGCRNFactorizedSTGCN,
    EigenvalueWeightedSpatial,
    GraphWaveNetAdaptiveAdjacency,
    GraphWaveNetSupportProjection,
    MultiHeadQKVSpatial,
    NodeAdaptiveGraphWaveNetSupportProjection,
)
from gesturegraph.model import build_model
from gesturegraph.topology import NUM_NODES, laplacian_eigenpairs, laplacian_positional_encoding
from gesturegraph.shrec import load_shrec17_npz


class ExperimentalBackboneTests(unittest.TestCase):
    def test_safe_npz_loader_preserves_official_split_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = np.arange(3 * 22 * 3, dtype=np.float32).reshape(3, 22, 3)
            second = np.arange(4 * 22 * 3, dtype=np.float32).reshape(4, 22, 3)
            np.savez(
                root / "train.npz",
                coordinates=np.concatenate([first, second]),
                offsets=np.asarray([0, 3, 7]),
                labels14=np.asarray([1, 14]),
                labels28=np.asarray([1, 28]),
            )
            samples = load_shrec17_npz(root, "train", frames=8, classes=14)
            self.assertEqual([sample.label for sample in samples], ["grab", "shake"])
            self.assertEqual(samples[0].sequence.shape, (8, 22, 3))

    def test_laplacian_encoding_is_deterministic_and_orthonormal(self):
        first = laplacian_positional_encoding(8)
        second = laplacian_positional_encoding(8)
        self.assertEqual(first.shape, (NUM_NODES, 8))
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.allclose(first.T @ first, np.eye(8), atol=1e-5))
        with self.assertRaises(ValueError):
            laplacian_positional_encoding(NUM_NODES)

    def test_laplacian_eigenpairs_align_values_with_encoding_columns(self):
        values, vectors = laplacian_eigenpairs(8)
        self.assertEqual(values.shape, (8,))
        self.assertEqual(vectors.shape, (NUM_NODES, 8))
        self.assertTrue(np.all(values > 0))
        self.assertTrue(np.all(values[:-1] <= values[1:]))
        adjacency = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float64)
        from gesturegraph.topology import EDGES
        for source, target in EDGES:
            adjacency[source, target] = adjacency[target, source] = 1.0
        degree = adjacency.sum(axis=1)
        laplacian = np.eye(NUM_NODES) - np.diag(degree ** -0.5) @ adjacency @ np.diag(degree ** -0.5)
        self.assertTrue(np.allclose(laplacian @ vectors, vectors * values, atol=1e-5))

    def test_qkv_attention_injects_spectral_position_at_every_spatial_layer(self):
        model = build_model(
            "spectral_pe_qkv",
            14,
            model_config={"pe_dim": 8, "attention_heads": 4, "adaptive_dim": 10},
        )
        self.assertEqual(len(model.blocks), 4)
        for block in model.blocks:
            self.assertIsInstance(block.spatial, EigenvalueWeightedSpatial)
            self.assertEqual(block.spatial.spectral_encoding.shape, (NUM_NODES, 8))
            self.assertEqual(block.spatial.spectral_eigenvalues.shape, (8,))
            self.assertEqual(block.spatial.position_projection.in_features, 8)
            self.assertIsInstance(block.spatial.spatial, MultiHeadQKVSpatial)
            self.assertEqual(block.spatial.spatial.heads, 4)
            self.assertEqual(block.spatial.spatial.physical_adjacency.shape, (NUM_NODES, NUM_NODES))
            self.assertFalse(hasattr(block.spatial.spatial, "attention_mask"))
            self.assertFalse(hasattr(block.spatial, "position_scale"))
            self.assertFalse(hasattr(block.spatial, "pre_norm"))

    def test_every_experimental_spatial_layer_uses_eigenvalue_weighted_encoding(self):
        for name in LEGACY_EXPERIMENTAL_MODEL_NAMES:
            with self.subTest(model=name):
                model = build_model(name, 14, model_config={"pe_dim": 8, "attention_heads": 4, "adaptive_dim": 10})
                self.assertEqual(len(model.blocks), 4)
                for block in model.blocks:
                    self.assertIsInstance(block.spatial, EigenvalueWeightedSpatial)
                    self.assertEqual(block.spatial.spectral_eigenvalues.shape, (8,))
                    self.assertEqual(block.spatial.spectral_encoding.shape, (NUM_NODES, 8))

    def test_direct_gwnet_support_is_full_row_stochastic(self):
        torch.manual_seed(5)
        graph = GraphWaveNetAdaptiveAdjacency(10)
        adaptive = graph.adaptive_adjacency()
        self.assertEqual(adaptive.shape, (NUM_NODES, NUM_NODES))
        self.assertTrue(torch.all(adaptive >= 0))
        self.assertTrue(torch.allclose(adaptive.sum(dim=1), torch.ones(NUM_NODES), atol=1e-6))
        self.assertGreater(int(torch.count_nonzero(adaptive)), NUM_NODES)

    def test_all_backbones_forward_backward_and_critical_gradients(self):
        critical_parameters = {
            "spectral_pe_stgcn": "blocks.0.spatial.position_projection.weight",
            "spectral_pe_qkv": "blocks.0.spatial.spatial.query_projection.weight",
            "gwnet_adaptive_support": "adaptive_graph.nodevec1",
            "agcrn_factorized_adjacency": "blocks.0.spatial.spatial.weights_pool",
        }
        for offset, name in enumerate(LEGACY_EXPERIMENTAL_MODEL_NAMES):
            with self.subTest(model=name):
                torch.manual_seed(10 + offset)
                model = build_model(
                    name,
                    14,
                    frames=16,
                    model_config={"pe_dim": 8, "attention_heads": 4, "adaptive_dim": 10},
                )
                inputs = torch.randn(2, 3, 16, NUM_NODES)
                logits = model(inputs)
                self.assertEqual(tuple(logits.shape), (2, 14))
                self.assertTrue(torch.isfinite(logits).all())
                logits.square().mean().backward()
                parameters = dict(model.named_parameters())
                gradient = parameters[critical_parameters[name]].grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_direct_gwnet_backbone_uses_two_order_two_supports(self):
        model = build_model("gwnet_adaptive_support", 14, model_config={"adaptive_dim": 10})
        self.assertEqual(len(model.blocks), 4)
        for block in model.blocks:
            self.assertIsInstance(block.spatial, EigenvalueWeightedSpatial)
            self.assertIsInstance(block.spatial.spatial, GraphWaveNetSupportProjection)
            self.assertEqual(block.spatial.spatial.order, 2)
            self.assertEqual(
                block.spatial.spatial.projection.in_channels,
                block.spatial.spatial.projection.out_channels * 5,
            )

    def test_experiment_four_has_node_specific_factorised_weight_pools(self):
        model = build_model(
            "agcrn_factorized_adjacency",
            14,
            model_config={"adaptive_dim": 10},
        )
        self.assertIsInstance(model, AGCRNFactorizedSTGCN)
        self.assertEqual(len(model.blocks), 4)
        for block in model.blocks:
            self.assertIsInstance(block.spatial, EigenvalueWeightedSpatial)
            self.assertIsInstance(block.spatial.spatial, NodeAdaptiveGraphWaveNetSupportProjection)
            self.assertEqual(block.spatial.spatial.order, 2)
            channels = block.spatial.spatial.bias_pool.shape[1]
            self.assertEqual(tuple(block.spatial.spatial.weights_pool.shape), (10, 5 * channels, channels))
        self.assertIsInstance(model.adaptive_graph, GraphWaveNetAdaptiveAdjacency)
        self.assertEqual(model.adaptive_graph.nodevec1.shape, (NUM_NODES, 10))
        self.assertEqual(model.adaptive_graph.nodevec2.shape, (NUM_NODES, 10))


if __name__ == "__main__":
    unittest.main()
