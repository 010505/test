import unittest

import torch
from torch import nn

from gesturegraph.backbones import (
    PURE_QKV_MODEL_NAMES,
    SE_IMPROVEMENT_MODEL_NAMES,
    SE_TRANSFER_MODEL_NAMES,
    SE_TRANSFER_V2_MODEL_NAMES,
    GatedDualSupportProjection,
    GraphWaveNetSupportProjection,
    MultiHeadQKVSpatial,
    NodeAdaptiveGraphWaveNetSupportProjection,
    PureMultiHeadQKVSpatial,
    SemanticSpectralSpatial,
    StableMultiHeadQKVSpatial,
    StemSemanticSESTGCN,
    StemSTGCN,
)
from gesturegraph.model import build_model
from gesturegraph.topology import NUM_NODES


class StemSemanticSETests(unittest.TestCase):
    def test_stem_control_lifts_xyz_before_first_spatial_layer(self):
        model = build_model("stem_stgcn", 14, model_config={"stem_channels": 32})
        self.assertIsInstance(model, StemSTGCN)
        self.assertEqual(model.input_stem.network[0].in_channels, 3)
        self.assertEqual(model.input_stem.network[0].out_channels, 32)
        self.assertEqual(model.blocks[0].temporal[0].in_channels, 32)
        self.assertEqual(model.blocks[0].temporal[0].out_channels, 32)

    def test_linear_se_dimensions_are_not_compressed_to_xyz(self):
        for dimensions in (8, 16, 21):
            with self.subTest(dimensions=dimensions):
                model = build_model(
                    "stem_linear_se",
                    14,
                    model_config={"pe_dim": dimensions, "stem_channels": 32},
                )
                self.assertIsInstance(model, StemSemanticSESTGCN)
                first = model.blocks[0].spatial
                self.assertIsInstance(first, SemanticSpectralSpatial)
                self.assertEqual(first.position_projection.in_features, dimensions)
                self.assertEqual(first.position_projection.out_features, 32)
                self.assertEqual(first.spectral_encoding.shape, (NUM_NODES, dimensions))

    def test_residual_mlp_keeps_linear_branch_and_semantic_branch(self):
        model = build_model(
            "stem_residual_mlp_se",
            14,
            model_config={"pe_dim": 21, "stem_channels": 32, "semantic_hidden": 64},
        )
        first = model.blocks[0].spatial
        self.assertIsInstance(first.position_projection, nn.Linear)
        self.assertIsNotNone(first.semantic_projection)
        self.assertEqual(first.semantic_projection[0].in_features, 21)
        self.assertEqual(first.semantic_projection[0].out_features, 64)
        self.assertAlmostEqual(float(first.semantic_scale.detach()), 0.1, places=6)

    def test_learnable_gate_starts_at_fixed_eigenvalues_and_receives_gradient(self):
        model = build_model(
            "stem_residual_mlp_gated_se",
            14,
            frames=16,
            model_config={"pe_dim": 21, "stem_channels": 32, "semantic_hidden": 64},
        )
        for block in model.blocks:
            spatial = block.spatial
            self.assertTrue(torch.equal(spatial.spectral_weights(), spatial.spectral_eigenvalues))
            self.assertIsNotNone(spatial.spectral_gate_delta)
        logits = model(torch.randn(2, 3, 16, NUM_NODES))
        logits.square().mean().backward()
        gate_gradient = model.blocks[0].spatial.spectral_gate_delta.grad
        self.assertIsNotNone(gate_gradient)
        self.assertTrue(torch.isfinite(gate_gradient).all())
        self.assertGreater(float(gate_gradient.abs().sum()), 0.0)

    def test_direct_learnable_values_keep_eigenvectors_fixed_and_receive_gradient(self):
        model = build_model(
            "stem_residual_mlp_learnable_values_se",
            14,
            frames=16,
            model_config={"pe_dim": 21, "stem_channels": 32, "semantic_hidden": 64},
        )
        fixed_encodings = [block.spatial.spectral_encoding.clone() for block in model.blocks]
        for block in model.blocks:
            spatial = block.spatial
            self.assertTrue(
                torch.equal(spatial.learnable_spectral_values, spatial.spectral_eigenvalues)
            )
            self.assertIsNone(spatial.spectral_gate_delta)
            self.assertFalse(spatial.spectral_encoding.requires_grad)
            self.assertEqual(tuple(spatial.learnable_spectral_values.shape), (21,))

        logits = model(torch.randn(2, 3, 16, NUM_NODES))
        logits.square().mean().backward()
        for block, initial_encoding in zip(model.blocks, fixed_encodings):
            spatial = block.spatial
            gradient = spatial.learnable_spectral_values.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)
            self.assertTrue(torch.equal(spatial.spectral_encoding, initial_encoding))

    def test_all_new_backbones_forward_backward(self):
        for offset, name in enumerate(SE_IMPROVEMENT_MODEL_NAMES):
            with self.subTest(model=name):
                torch.manual_seed(100 + offset)
                model = build_model(
                    name,
                    14,
                    frames=16,
                    model_config={"pe_dim": 21, "stem_channels": 32, "semantic_hidden": 64},
                )
                logits = model(torch.randn(2, 3, 16, NUM_NODES))
                self.assertEqual(tuple(logits.shape), (2, 14))
                self.assertTrue(torch.isfinite(logits).all())
                logits.mean().backward()
                stem_gradient = model.input_stem.network[0].weight.grad
                self.assertIsNotNone(stem_gradient)
                self.assertTrue(torch.isfinite(stem_gradient).all())

    def test_transfer_pairs_differ_only_by_semantic_wrapper(self):
        pairs = [
            ("stem_qkv_control", "stem_semantic_qkv", MultiHeadQKVSpatial),
            ("stem_gwnet_control", "stem_semantic_gwnet", GraphWaveNetSupportProjection),
            ("stem_agcrn_control", "stem_semantic_agcrn", NodeAdaptiveGraphWaveNetSupportProjection),
        ]
        config = {
            "pe_dim": 21,
            "stem_channels": 32,
            "semantic_hidden": 64,
            "attention_heads": 4,
            "adaptive_dim": 10,
        }
        for control_name, semantic_name, operator_type in pairs:
            with self.subTest(pair=semantic_name):
                control = build_model(control_name, 14, model_config=config)
                semantic = build_model(semantic_name, 14, model_config=config)
                self.assertEqual(control.input_stem.network[0].out_channels, 32)
                self.assertEqual(semantic.input_stem.network[0].out_channels, 32)
                self.assertIsInstance(control.blocks[0].spatial, operator_type)
                self.assertIsInstance(semantic.blocks[0].spatial, SemanticSpectralSpatial)
                self.assertIsInstance(semantic.blocks[0].spatial.spatial, operator_type)

    def test_all_transfer_backbones_forward_backward(self):
        config = {
            "pe_dim": 21,
            "stem_channels": 32,
            "semantic_hidden": 64,
            "attention_heads": 4,
            "adaptive_dim": 10,
        }
        for offset, name in enumerate(SE_TRANSFER_MODEL_NAMES):
            with self.subTest(model=name):
                torch.manual_seed(200 + offset)
                model = build_model(name, 14, frames=16, model_config=config)
                logits = model(torch.randn(2, 3, 16, NUM_NODES))
                self.assertEqual(tuple(logits.shape), (2, 14))
                self.assertTrue(torch.isfinite(logits).all())
                logits.square().mean().backward()
                gradient = model.input_stem.network[0].weight.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())

    def test_stable_qkv_uses_pre_norm_positive_temperatures_and_unit_gain(self):
        model = build_model(
            "stem_stable_qkv_control",
            14,
            frames=16,
            model_config={"stem_channels": 32, "attention_heads": 4},
        )
        spatial = model.blocks[0].spatial
        self.assertIsInstance(spatial, StableMultiHeadQKVSpatial)
        self.assertIsInstance(spatial.pre_norm, nn.LayerNorm)
        self.assertTrue(torch.all(spatial.temperatures() > 0))
        self.assertTrue(torch.allclose(spatial.temperatures(), torch.ones(4)))
        self.assertAlmostEqual(float(spatial.residual_scale.detach()), 1.0, places=6)

    def test_gwn_agcrn_v2_differ_only_in_dynamic_weight_factorisation(self):
        config = {"stem_channels": 32, "adaptive_dim": 10}
        gwn = build_model("stem_gwnet_gated_control", 14, frames=16, model_config=config)
        agcrn = build_model("stem_agcrn_dynamic_control", 14, frames=16, model_config=config)
        gwn_operator = gwn.blocks[0].spatial
        agcrn_operator = agcrn.blocks[0].spatial
        self.assertIsInstance(gwn_operator, GatedDualSupportProjection)
        self.assertIsInstance(agcrn_operator, GatedDualSupportProjection)
        self.assertFalse(gwn_operator.factorize_dynamic)
        self.assertTrue(agcrn_operator.factorize_dynamic)
        self.assertIsInstance(gwn_operator.physical_projection, nn.Conv2d)
        self.assertIsInstance(agcrn_operator.physical_projection, nn.Conv2d)
        self.assertIsNone(agcrn_operator.dynamic_projection)
        self.assertAlmostEqual(float(gwn_operator.dynamic_gate().detach()), 0.1, places=6)
        self.assertAlmostEqual(float(agcrn_operator.dynamic_gate().detach()), 0.1, places=6)

    def test_all_v2_backbones_forward_backward(self):
        config = {
            "pe_dim": 21,
            "stem_channels": 32,
            "semantic_hidden": 64,
            "attention_heads": 4,
            "adaptive_dim": 10,
        }
        for offset, name in enumerate(SE_TRANSFER_V2_MODEL_NAMES):
            with self.subTest(model=name):
                torch.manual_seed(300 + offset)
                model = build_model(name, 14, frames=16, model_config=config)
                logits = model(torch.randn(2, 3, 16, NUM_NODES))
                self.assertEqual(tuple(logits.shape), (2, 14))
                self.assertTrue(torch.isfinite(logits).all())
                logits.square().mean().backward()
                gradient = model.input_stem.network[0].weight.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())

    def test_pure_qkv_has_no_physical_adjacency_bias_or_mask(self):
        model = build_model(
            "stem_semantic_pure_qkv",
            14,
            frames=16,
            model_config={
                "pe_dim": 21,
                "stem_channels": 32,
                "semantic_hidden": 64,
                "attention_heads": 4,
            },
        )
        wrapper = model.blocks[0].spatial
        self.assertIsInstance(wrapper, SemanticSpectralSpatial)
        attention = wrapper.spatial
        self.assertIsInstance(attention, PureMultiHeadQKVSpatial)
        self.assertFalse(hasattr(attention, "physical_bias"))
        self.assertFalse(hasattr(attention, "physical_adjacency"))
        weights = attention.attention_weights(torch.randn(2, 32, 4, NUM_NODES))
        self.assertEqual(tuple(weights.shape), (2, 4, 4, NUM_NODES, NUM_NODES))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1))))

    def test_pure_qkv_pair_forward_backward(self):
        config = {
            "pe_dim": 21,
            "stem_channels": 32,
            "semantic_hidden": 64,
            "attention_heads": 4,
        }
        for offset, name in enumerate(PURE_QKV_MODEL_NAMES):
            with self.subTest(model=name):
                torch.manual_seed(400 + offset)
                model = build_model(name, 14, frames=16, model_config=config)
                logits = model(torch.randn(2, 3, 16, NUM_NODES))
                self.assertEqual(tuple(logits.shape), (2, 14))
                self.assertTrue(torch.isfinite(logits).all())
                logits.square().mean().backward()
                gradient = model.input_stem.network[0].weight.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())


if __name__ == "__main__":
    unittest.main()
