import torch
import unittest

from gesturegraph.backbones import (
    GraphWaveNetAdaptiveAdjacency,
    MultiHeadQKVSpatial,
    NodeAdaptiveGraphWaveNetSupportProjection,
    StableMultiHeadQKVSpatial,
    TemporalGate,
    _velocity_channels,
)


class TemporalGateTests(unittest.TestCase):
    def test_output_shape_matches_input(self):
        gate = TemporalGate(32)
        x = torch.randn(2, 32, 16, 22)
        self.assertEqual(gate(x).shape, x.shape)

    def test_gate_never_amplifies_the_signal(self):
        gate = TemporalGate(32)
        gate.eval()
        x = torch.randn(2, 32, 16, 22)
        with torch.no_grad():
            out = gate(x)
        self.assertTrue((out.abs() <= x.abs() + 1e-6).all())

    def test_zero_input_gives_zero_output(self):
        gate = TemporalGate(16)
        x = torch.zeros(1, 16, 8, 22)
        self.assertTrue(torch.equal(gate(x), x))


class VelocityChannelsTests(unittest.TestCase):
    def test_first_frame_velocity_is_zero(self):
        x = torch.randn(2, 3, 10, 22)
        combined = _velocity_channels(x)
        self.assertTrue(torch.equal(combined[:, 3:, 0, :], torch.zeros(2, 3, 22)))

    def test_later_frames_match_the_true_difference(self):
        x = torch.randn(2, 3, 10, 22)
        combined = _velocity_channels(x)
        expected = x[:, :, 1:, :] - x[:, :, :-1, :]
        self.assertTrue(torch.allclose(combined[:, 3:, 1:, :], expected))

    def test_position_channels_are_unchanged(self):
        x = torch.randn(2, 3, 10, 22)
        combined = _velocity_channels(x)
        self.assertTrue(torch.equal(combined[:, :3, :, :], x))

    def test_channel_count_doubles(self):
        x = torch.randn(2, 3, 10, 22)
        self.assertEqual(_velocity_channels(x).shape[1], 6)


class GraphWaveNetAdaptiveAdjacencyTests(unittest.TestCase):
    def test_adaptive_support_rows_sum_to_one(self):
        module = GraphWaveNetAdaptiveAdjacency(embedding_dim=6)
        adaptive = module.adaptive_adjacency()
        self.assertTrue(torch.allclose(adaptive.sum(dim=1), torch.ones(adaptive.shape[0]), atol=1e-5))

    def test_adaptive_support_is_non_negative(self):
        module = GraphWaveNetAdaptiveAdjacency(embedding_dim=6)
        adaptive = module.adaptive_adjacency()
        self.assertTrue((adaptive >= 0).all())

    def test_rejects_non_positive_embedding_dim(self):
        with self.assertRaises(ValueError):
            GraphWaveNetAdaptiveAdjacency(embedding_dim=0)

    def test_components_returns_physical_and_adaptive(self):
        module = GraphWaveNetAdaptiveAdjacency(embedding_dim=6)
        components = module.components()
        self.assertIn("physical", components)
        self.assertIn("adaptive_support", components)


class NodeAdaptiveGraphWaveNetSupportProjectionTests(unittest.TestCase):
    def test_missing_adjacency_raises(self):
        module = NodeAdaptiveGraphWaveNetSupportProjection(16, embedding_dim=6)
        x = torch.randn(2, 16, 8, 22)
        with self.assertRaises(ValueError):
            module(x, adjacency=None, node_embeddings=torch.randn(22, 6))

    def test_missing_node_embeddings_raises(self):
        module = NodeAdaptiveGraphWaveNetSupportProjection(16, embedding_dim=6)
        x = torch.randn(2, 16, 8, 22)
        support = torch.eye(22)
        with self.assertRaises(ValueError):
            module(x, adjacency=(support, support), node_embeddings=None)

    def test_output_shape_with_valid_inputs(self):
        module = NodeAdaptiveGraphWaveNetSupportProjection(16, embedding_dim=6)
        x = torch.randn(2, 16, 8, 22)
        support = torch.eye(22)
        node_embeddings = torch.randn(22, 6)
        out = module(x, adjacency=(support, support), node_embeddings=node_embeddings)
        self.assertEqual(out.shape, x.shape)

    def test_rejects_non_positive_order(self):
        with self.assertRaises(ValueError):
            NodeAdaptiveGraphWaveNetSupportProjection(16, embedding_dim=6, order=0)


class QKVAttentionInitializationTests(unittest.TestCase):
    def test_stable_variant_has_a_layernorm_the_original_lacks(self):
        stable = StableMultiHeadQKVSpatial(32, heads=4)
        unstable = MultiHeadQKVSpatial(32, heads=4)
        self.assertTrue(hasattr(stable, "input_norm"))
        self.assertFalse(hasattr(unstable, "input_norm"))

    def test_stable_residual_gain_starts_at_one_not_point_one(self):
        stable = StableMultiHeadQKVSpatial(32, heads=4)
        unstable = MultiHeadQKVSpatial(32, heads=4)
        self.assertAlmostEqual(stable.residual_gain.item(), 1.0)
        self.assertAlmostEqual(unstable.residual_scale.item(), 0.1)

    def test_stable_temperature_clamp_prevents_non_finite_output(self):
        module = StableMultiHeadQKVSpatial(32, heads=4, dropout=0.0)
        with torch.no_grad():
            module.per_head_temperature.fill_(-5.0)
        module.eval()
        x = torch.randn(2, 32, 16, 22)
        with torch.no_grad():
            out = module(x)
        self.assertTrue(torch.isfinite(out).all())

    def test_both_variants_preserve_shape(self):
        x = torch.randn(2, 32, 16, 22)
        stable = StableMultiHeadQKVSpatial(32, heads=4)
        unstable = MultiHeadQKVSpatial(32, heads=4)
        self.assertEqual(stable(x).shape, x.shape)
        self.assertEqual(unstable(x).shape, x.shape)

    def test_both_variants_reject_zero_heads(self):
        with self.assertRaises(ValueError):
            StableMultiHeadQKVSpatial(32, heads=0)
        with self.assertRaises(ValueError):
            MultiHeadQKVSpatial(32, heads=0)

    def test_both_variants_produce_real_gradients_through_every_parameter(self):
        for cls in (StableMultiHeadQKVSpatial, MultiHeadQKVSpatial):
            module = cls(32, heads=4, dropout=0.0)
            x = torch.randn(2, 32, 16, 22, requires_grad=True)
            out = module(x)
            out.sum().backward()
            for name, parameter in module.named_parameters():
                self.assertIsNotNone(parameter.grad, f"{cls.__name__}.{name} got no gradient")


if __name__ == "__main__":
    unittest.main()
