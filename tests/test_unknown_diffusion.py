import unittest

import numpy as np
import torch

from gesturegraph.progressive import ClassDiffusionModel, RawGestureSample
from gesturegraph.unknown_diffusion_benchmark import (
    UNKNOWN_LABEL,
    expand_teacher_to_unknown,
    forward_conditions,
    make_unknown_samples,
    make_unknown_sequence,
    stage_metrics,
)
from gesturegraph.unknown_diffusion_optimization import (
    calibrate_no_biases,
    closed_set_preservation_losses,
    forward_conditions_with_no_retention,
)
from gesturegraph.hierarchical_markov_search import (
    HierarchicalVerificationSearch,
    build_balanced_tree_levels,
    choose_group,
    level_masks,
)
from gesturegraph.endpoint_knownness_optimization import (
    TemporalKnownnessDetector,
    combine_probabilities,
)
from gesturegraph.temporal_pe_order_optimization import (
    TemporalPEOrderKnownness,
    normalized_temporal_encoding,
    reverse_valid_frames,
)
from gesturegraph.reverse_relabel_evaluation import best_involutive_mapping
from gesturegraph.reverse_aware_action_optimization import (
    ReverseAwareActionAdapter,
    threshold_predictions,
)
from gesturegraph.temporal_candidate_membership import (
    TemporalCandidateMembership,
    support_loss,
)


class UnknownProxyTests(unittest.TestCase):
    def setUp(self):
        self.sequence = np.random.default_rng(2).normal(size=(20, 22, 3)).astype(np.float32)

    def test_all_proxy_modes_preserve_skeleton_shape(self):
        rng = np.random.default_rng(4)
        for mode in ("frozen", "shuffled", "noise", "splice", "reversed", "joint_permuted"):
            result = make_unknown_sequence(self.sequence, mode, rng, self.sequence[::-1])
            self.assertEqual(result.shape, self.sequence.shape)
            self.assertTrue(np.isfinite(result).all())

    def test_unknown_samples_receive_the_fifteenth_label(self):
        samples = [
            RawGestureSample(0, "grab", self.sequence, "train"),
            RawGestureSample(1, "tap", self.sequence * 2, "train"),
        ]
        unknown = make_unknown_samples(samples, ("splice",), seed=3)
        self.assertEqual(len(unknown), 2)
        self.assertTrue(all(sample.label == UNKNOWN_LABEL for sample in unknown))


class FifteenStateDiffusionTests(unittest.TestCase):
    def test_expansion_preserves_known_output_rows(self):
        torch.manual_seed(3)
        base = ClassDiffusionModel(num_classes=14)
        expanded = expand_teacher_to_unknown(base)
        self.assertEqual(expanded.num_classes, 15)
        self.assertTrue(torch.equal(
            expanded.denoiser[-1].weight[:14], base.denoiser[-1].weight
        ))

    def test_every_observation_has_a_normalized_no_candidate(self):
        model = ClassDiffusionModel(num_classes=15)
        conditions = torch.randn(2, 5, 129)
        probabilities = forward_conditions(model, conditions).exp()
        self.assertEqual(tuple(probabilities.shape), (2, 5, 15))
        self.assertTrue(torch.allclose(probabilities.sum(dim=-1), torch.ones(2, 5), atol=1e-5))
        self.assertTrue(torch.all(probabilities[..., 14] > 0))

    def test_no_is_not_absorbing_and_later_evidence_can_recover_known(self):
        class ScriptedDiffusion(ClassDiffusionModel):
            def __init__(self):
                super().__init__(num_classes=15)
                self.update = 0

            def _reverse_distribution(self, condition, initial):
                output = torch.zeros_like(initial)
                output[:, 14 if self.update == 0 else 3] = 1.0
                self.update += 1
                return output

        model = ScriptedDiffusion()
        probabilities = forward_conditions(model, torch.randn(1, 2, 129)).exp()
        self.assertEqual(probabilities.argmax(dim=-1).tolist(), [[14, 3]])

    def test_stage_metrics_count_no_at_each_observation(self):
        known = np.zeros((2, 2, 15), dtype=np.float32)
        unknown = np.zeros((2, 2, 15), dtype=np.float32)
        known[:, :, 0] = 1.0
        unknown[:, :, 14] = 1.0
        rows = stage_metrics(known, np.zeros(2, dtype=np.int64), unknown, (0.25, 1.0))
        self.assertEqual([row.unknown_recall for row in rows], [1.0, 1.0])
        self.assertEqual([row.known_false_no_rate for row in rows], [0.0, 0.0])


class UnknownOptimizationTests(unittest.TestCase):
    def test_weak_no_inheritance_keeps_every_stage_normalized(self):
        torch.manual_seed(7)
        model = ClassDiffusionModel(num_classes=15)
        model.eval()
        conditions = torch.randn(3, 4, 129)
        full = forward_conditions_with_no_retention(model, conditions, 1.0).exp()
        weak = forward_conditions_with_no_retention(model, conditions, 0.25).exp()
        self.assertTrue(torch.allclose(weak.sum(dim=-1), torch.ones(3, 4), atol=1e-5))
        self.assertTrue(torch.allclose(full[:, 0], weak[:, 0], atol=1e-6))
        self.assertTrue(torch.all(weak[..., 14] > 0))

    def test_bidirectional_margin_rewards_correct_separation(self):
        logits = torch.full((2, 2, 15), -4.0)
        logits[0, :, 3] = 4.0
        logits[1, :, 14] = 4.0
        log_probabilities = logits.log_softmax(dim=-1)
        old_logits = logits[..., :14].log_softmax(dim=-1)
        targets = torch.tensor([3, 14])
        _, known_margin, unknown_margin = closed_set_preservation_losses(
            log_probabilities, old_logits, targets, margin=0.5, temperature=2.0
        )
        self.assertAlmostEqual(float(known_margin), 0.0, places=6)
        self.assertAlmostEqual(float(unknown_margin), 0.0, places=6)

    def test_stage_calibration_can_suppress_false_no_without_removing_no(self):
        known = np.full((2, 1, 15), 1e-4, dtype=np.float32)
        unknown = np.full((2, 1, 15), 1e-4, dtype=np.float32)
        known[:, 0, 0] = 0.40
        known[:, 0, 14] = 0.50
        unknown[:, 0, 0] = 0.10
        unknown[:, 0, 14] = 0.80
        biases, rows = calibrate_no_biases(
            known, np.zeros(2, dtype=np.int64), unknown, [1.0],
            max_known_drop=0.0, max_false_no=0.0, bias_grid=[-2.0, 0.0],
        )
        self.assertEqual(biases, [-2.0])
        self.assertEqual(rows[0]["known_false_no_rate"], 0.0)
        self.assertEqual(rows[0]["unknown_recall"], 1.0)


class HierarchicalVerificationTests(unittest.TestCase):
    def test_balanced_tree_contracts_four_times_to_singletons(self):
        prototypes = np.eye(14, dtype=np.float32)
        levels = build_balanced_tree_levels(prototypes)
        self.assertEqual([len(level) for level in levels], [1, 2, 4, 8, 14])
        self.assertTrue(all(len(node) == 1 for node in levels[-1]))
        for previous, current in zip(levels, levels[1:]):
            self.assertTrue(all(
                any(set(node).issubset(parent) for parent in previous)
                for node in current
            ))

    def test_rejection_switches_from_wrong_ab_branch_and_recovers_c(self):
        evidence = torch.tensor([[0.05, 0.05, 0.80, 0.10]])
        active_ab = torch.tensor([[True, True, False, False]])
        leaves = torch.eye(4, dtype=torch.bool)
        selected = choose_group(
            evidence, active_ab, leaves, torch.tensor([False])
        )
        self.assertEqual(selected.tolist(), [[False, False, True, False]])

    def test_residual_no_is_not_a_peer_leaf_and_outputs_normalize(self):
        levels = build_balanced_tree_levels(np.eye(14, dtype=np.float32))
        model = HierarchicalVerificationSearch(level_masks(levels), dropout=0.0)
        model.eval()
        conditions = torch.randn(3, 5, 129)
        evidence = torch.softmax(torch.randn(3, 5, 14), dim=-1)
        probabilities, diagnostics = model(conditions, evidence)
        self.assertEqual(tuple(probabilities.shape), (3, 5, 15))
        self.assertTrue(torch.allclose(
            probabilities.sum(dim=-1), torch.ones(3, 5), atol=1e-5
        ))
        sizes = diagnostics["candidate_sizes"]
        self.assertTrue(torch.all(sizes[:, 1:] <= sizes[:, :-1]))
        self.assertTrue(torch.all(probabilities[..., 14] > 0))

    def test_full_observation_blend_preserves_original_fourteen_class_order(self):
        levels = build_balanced_tree_levels(np.eye(14, dtype=np.float32))
        model = HierarchicalVerificationSearch(level_masks(levels), dropout=0.0)
        model.eval()
        conditions = torch.randn(4, 5, 129)
        evidence = torch.softmax(torch.randn(4, 5, 14), dim=-1)
        probabilities, _ = model(
            conditions,
            evidence,
            known_blend_weights=[0.0, 0.25, 0.5, 0.75, 1.0],
        )
        conditional = probabilities[:, -1, :14]
        conditional = conditional / conditional.sum(dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(conditional, evidence[:, -1], atol=1e-6))
        self.assertTrue(torch.equal(
            conditional.argmax(dim=-1), evidence[:, -1].argmax(dim=-1)
        ))

    def test_no_bias_does_not_change_known_class_order(self):
        levels = build_balanced_tree_levels(np.eye(14, dtype=np.float32))
        model = HierarchicalVerificationSearch(level_masks(levels), dropout=0.0)
        model.eval()
        conditions = torch.randn(3, 5, 129)
        evidence = torch.softmax(torch.randn(3, 5, 14), dim=-1)
        low_no, _ = model(conditions, evidence, no_biases=[-4.0] * 5)
        high_no, _ = model(conditions, evidence, no_biases=[4.0] * 5)
        self.assertTrue(torch.equal(
            low_no[..., :14].argmax(dim=-1), high_no[..., :14].argmax(dim=-1)
        ))
        self.assertTrue(torch.all(high_no[..., 14] > low_no[..., 14]))


class DedicatedKnownnessTests(unittest.TestCase):
    def test_causal_features_do_not_use_future_observations(self):
        conditions = torch.randn(2, 5, 129)
        evidence = torch.softmax(torch.randn(2, 5, 14), dim=-1)
        changed_conditions = conditions.clone()
        changed_evidence = evidence.clone()
        changed_conditions[:, 3:] += 100.0
        changed_evidence[:, 3:] = torch.softmax(
            torch.randn_like(changed_evidence[:, 3:]), dim=-1
        )
        original = TemporalKnownnessDetector.causal_features(conditions, evidence)
        changed = TemporalKnownnessDetector.causal_features(
            changed_conditions, changed_evidence
        )
        self.assertTrue(torch.allclose(original[:, :3], changed[:, :3]))

    def test_dedicated_no_bias_preserves_action_distribution(self):
        conditional = np.random.default_rng(7).random((3, 5, 14))
        conditional /= conditional.sum(axis=-1, keepdims=True)
        logits = np.random.default_rng(8).normal(size=(3, 5))
        low_no = combine_probabilities(conditional, logits, [-3.0] * 5)
        high_no = combine_probabilities(conditional, logits, [3.0] * 5)
        low_conditional = low_no[..., :14] / low_no[..., :14].sum(
            axis=-1, keepdims=True
        )
        high_conditional = high_no[..., :14] / high_no[..., :14].sum(
            axis=-1, keepdims=True
        )
        self.assertTrue(np.allclose(low_conditional, conditional))
        self.assertTrue(np.allclose(high_conditional, conditional))
        self.assertTrue(np.all(high_no[..., 14] > low_no[..., 14]))


class TemporalPEOrderTests(unittest.TestCase):
    def test_normalized_temporal_pe_uses_valid_prefix_positions(self):
        encoding, mask = normalized_temporal_encoding(
            torch.tensor([2, 4]), 4, 8, torch.float32
        )
        self.assertEqual(tuple(encoding.shape), (2, 4, 8))
        self.assertEqual(mask.tolist(), [[True, True, False, False], [True] * 4])
        self.assertTrue(torch.all(encoding[0, 2:] == 0))
        self.assertFalse(torch.allclose(encoding[0, 1], encoding[1, 1]))

    def test_reverse_valid_frames_leaves_padding_in_place(self):
        frames = torch.arange(2 * 5, dtype=torch.float32).reshape(2, 5, 1)
        reversed_frames = reverse_valid_frames(frames, torch.tensor([3, 5]))
        self.assertEqual(reversed_frames[0, :, 0].tolist(), [2, 1, 0, 3, 4])
        self.assertEqual(reversed_frames[1, :, 0].tolist(), [9, 8, 7, 6, 5])

    def test_temporal_pe_order_model_shapes_and_normalization(self):
        model = TemporalPEOrderKnownness(dropout=0.0)
        conditions = torch.randn(3, 5, 129)
        evidence = torch.softmax(torch.randn(3, 5, 14), dim=-1)
        frames = torch.randn(3, 5, 16, 128)
        lengths = torch.tensor([[4, 8, 11, 13, 16]]).expand(3, -1)
        known_logits, order_logits = model(
            conditions, evidence, frames, lengths
        )
        reversed_logits = model.reversed_order_logits(frames, lengths)
        self.assertEqual(tuple(known_logits.shape), (3, 5))
        self.assertEqual(tuple(order_logits.shape), (3, 5))
        self.assertEqual(tuple(reversed_logits.shape), (3, 5))

    def test_reverse_label_mapping_is_maximum_support_and_involutive(self):
        counts = np.asarray([
            [1, 8, 0, 0],
            [9, 1, 0, 0],
            [0, 0, 7, 1],
            [0, 0, 1, 6],
        ])
        mapping = np.asarray(best_involutive_mapping(counts))
        self.assertEqual(mapping.tolist(), [1, 0, 2, 3])
        self.assertTrue(np.all(mapping[mapping] == np.arange(4)))


class ReverseAwareActionTests(unittest.TestCase):
    def test_adapter_has_fourteen_class_output_and_stage_gate(self):
        model = ReverseAwareActionAdapter(hidden=32, dropout=0.0)
        logits, gate_logits = model(
            torch.randn(3, 5, 129),
            torch.softmax(torch.randn(3, 5, 14), dim=-1),
            torch.randn(3, 5, 64),
            torch.randn(3, 5),
        )
        self.assertEqual(tuple(logits.shape), (3, 5, 14))
        self.assertEqual(tuple(gate_logits.shape), (3, 5))

    def test_hard_gate_preserves_original_posterior_exactly(self):
        original = np.asarray([[[0.8, 0.2], [0.7, 0.3]]])
        adapted = np.asarray([[[0.1, 0.9], [0.2, 0.8]]])
        gates = np.asarray([[0.49, 0.51]])
        selected = threshold_predictions(original, adapted, gates, 0.5)
        self.assertTrue(np.array_equal(selected[:, 0], original[:, 0]))
        self.assertTrue(np.array_equal(selected[:, 1], adapted[:, 1]))


class TemporalCandidateMembershipTests(unittest.TestCase):
    def test_predicts_membership_without_a_reverse_gate(self):
        levels = build_balanced_tree_levels(np.eye(14, dtype=np.float32))
        base = HierarchicalVerificationSearch(level_masks(levels), dropout=0.0)
        model = TemporalCandidateMembership(base, hidden=32, dropout=0.0)
        conditions = torch.randn(3, 5, 129)
        evidence = torch.softmax(torch.randn(3, 5, 14), dim=-1)
        temporal = torch.randn(3, 5, 64)
        probabilities, diagnostics = model(conditions, evidence, temporal)
        self.assertEqual(tuple(probabilities.shape), (3, 5, 15))
        self.assertEqual(tuple(diagnostics["support_logits"].shape), (3, 5))
        self.assertFalse(hasattr(model, "reverse_gate"))
        loss = support_loss(diagnostics, torch.tensor([0, 4, 13]))
        self.assertTrue(torch.isfinite(loss))

    def test_full_stage_preserves_original_action_posterior(self):
        levels = build_balanced_tree_levels(np.eye(14, dtype=np.float32))
        base = HierarchicalVerificationSearch(level_masks(levels), dropout=0.0)
        model = TemporalCandidateMembership(base, hidden=32, dropout=0.0)
        model.eval()
        evidence = torch.softmax(torch.randn(2, 5, 14), dim=-1)
        probabilities, _ = model(
            torch.randn(2, 5, 129), evidence, torch.randn(2, 5, 64)
        )
        conditional = probabilities[:, -1, :14]
        conditional /= conditional.sum(dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(conditional, evidence[:, -1], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
