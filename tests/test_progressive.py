import unittest

import numpy as np
import torch

from gesturegraph.progressive import (
    ClassDiffusionModel,
    DirectPrefixModel,
    GatedClassDiffusionModel,
    GRUEvidenceModel,
    ReliabilityGatedClassDiffusionModel,
    causal_prefix_view,
)
from gesturegraph.progressive_benchmark import error_recovery_metrics


class ProgressiveRecognitionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_causal_prefix_cannot_see_future_coordinates(self):
        sequence = np.zeros((20, 22, 3), dtype=np.float32)
        sequence[:, :, 0] = np.arange(20, dtype=np.float32)[:, None]
        changed_future = sequence.copy()
        changed_future[10:] += 1000.0
        first, length, ratio = causal_prefix_view(sequence, 0.5, frames=16)
        second, other_length, other_ratio = causal_prefix_view(changed_future, 0.5, frames=16)
        self.assertEqual((length, ratio), (8, 0.5))
        self.assertEqual((other_length, other_ratio), (8, 0.5))
        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(first[8:], np.repeat(first[7:8], 8, axis=0))

    def test_direct_and_gru_prefix_models_forward_backward(self):
        views = torch.randn(2, 3, 3, 16, 22)
        lengths = torch.tensor([[4, 8, 16], [4, 8, 16]])
        ratios = torch.tensor([[0.25, 0.5, 1.0], [0.25, 0.5, 1.0]])
        for model in (DirectPrefixModel(), GRUEvidenceModel()):
            logits = model(views, lengths, ratios)
            self.assertEqual(tuple(logits.shape), (2, 3, 14))
            logits.mean().backward()

    def test_direct_model_accepts_dataloader_float64_progress(self):
        model = DirectPrefixModel()
        logits = model(
            torch.randn(2, 3, 16, 22),
            torch.tensor([8, 16]),
            torch.tensor([0.5, 1.0], dtype=torch.float64),
        )
        self.assertEqual(tuple(logits.shape), (2, 14))

    def test_class_diffusion_is_normalized_and_trainable(self):
        model = ClassDiffusionModel()
        self.assertTrue(torch.allclose(
            model.transitions.sum(-1),
            torch.ones_like(model.transitions.sum(-1)),
        ))
        self.assertTrue(torch.allclose(
            model.posteriors.sum(-1),
            torch.ones(model.steps, 14, 14),
            atol=1e-5,
        ))
        views = torch.randn(1, 2, 3, 16, 22)
        lengths = torch.tensor([[8, 16]])
        ratios = torch.tensor([[0.5, 1.0]])
        log_probabilities, conditions = model(views, lengths, ratios, return_auxiliary=True)
        self.assertEqual(tuple(log_probabilities.shape), (1, 2, 14))
        self.assertTrue(torch.allclose(log_probabilities.exp().sum(-1), torch.ones(1, 2), atol=1e-5))
        targets = torch.tensor([3])
        loss = -log_probabilities[:, :, 3].mean() + model.denoising_loss(conditions, targets)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_four_step_clean_prediction_matches_complete_reverse_chain(self):
        model = ClassDiffusionModel()
        model.eval()
        condition = torch.randn(2, 129)
        initial = torch.softmax(torch.randn(2, 14), dim=-1)
        with torch.no_grad():
            fast_path = model._reverse_distribution(condition, initial)
            distribution = initial
            classes = torch.arange(model.num_classes)
            for step in range(model.steps, 0, -1):
                expanded = condition[:, None, :].expand(-1, model.num_classes, -1)
                noisy = classes[None, :].expand(len(condition), -1)
                logits = model._denoise_logits(
                    expanded.reshape(-1, condition.shape[-1]), noisy.reshape(-1), step
                ).reshape(len(condition), model.num_classes, model.num_classes)
                clean = logits.softmax(dim=-1)
                reverse = torch.einsum(
                    "bji,jik->bjk", clean, model.posteriors[step - 1]
                )
                distribution = torch.einsum("bj,bjk->bk", distribution, reverse)
                distribution = distribution / distribution.sum(dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(fast_path, distribution, atol=1e-5, rtol=1e-5))

    def test_online_steps_match_batched_sequence_in_eval_mode(self):
        views = torch.randn(1, 3, 3, 16, 22)
        lengths = torch.tensor([[4, 8, 16]])
        ratios = torch.tensor([[0.25, 0.5, 1.0]])
        for model in (
            DirectPrefixModel(), GRUEvidenceModel(), ClassDiffusionModel(),
            GatedClassDiffusionModel(), ReliabilityGatedClassDiffusionModel(),
        ):
            model.eval()
            with torch.no_grad():
                batched = model(views, lengths, ratios)
                state = None
                online = []
                for update in range(views.shape[1]):
                    output, state = model.online_step(
                        views[:, update], lengths[:, update], ratios[:, update], state
                    )
                    online.append(output)
                online = torch.stack(online, dim=1)
            self.assertTrue(torch.allclose(batched, online, atol=1e-5, rtol=1e-4))

    def test_gated_diffusion_starts_from_half_inheritance_and_backpropagates(self):
        model = GatedClassDiffusionModel()
        views = torch.randn(2, 3, 3, 16, 22)
        lengths = torch.tensor([[4, 8, 16], [4, 8, 16]])
        ratios = torch.tensor([[0.25, 0.5, 1.0], [0.25, 0.5, 1.0]])
        log_probabilities, diagnostics = model(
            views, lengths, ratios, return_diagnostics=True
        )
        self.assertEqual(tuple(log_probabilities.shape), (2, 3, 14))
        self.assertEqual(tuple(diagnostics["gates"].shape), (2, 2))
        self.assertTrue(torch.allclose(
            diagnostics["gates"], torch.full((2, 2), 0.5), atol=1e-6
        ))
        targets = torch.tensor([1, 2])
        _, conditions = model(views, lengths, ratios, return_auxiliary=True)
        loss = -log_probabilities[torch.arange(2), :, targets].mean()
        loss = loss + model.denoising_loss(conditions, targets)
        loss.backward()
        self.assertIsNotNone(model.inheritance_gate[-1].weight.grad)

    def test_reliability_gate_is_bounded_and_has_supervised_loss(self):
        model = ReliabilityGatedClassDiffusionModel()
        views = torch.randn(2, 3, 3, 16, 22)
        lengths = torch.tensor([[4, 8, 16], [4, 8, 16]])
        ratios = torch.tensor([[0.25, 0.5, 1.0], [0.25, 0.5, 1.0]])
        log_probabilities, conditions, diagnostics = model(
            views,
            lengths,
            ratios,
            return_auxiliary=True,
            return_diagnostics=True,
        )
        self.assertTrue(torch.all(diagnostics["gates"] >= 0.05))
        self.assertTrue(torch.all(diagnostics["gates"] <= 0.95))
        targets = torch.tensor([1, 2])
        loss = -log_probabilities[torch.arange(2), :, targets].mean()
        loss = loss + model.denoising_loss(conditions, targets)
        loss = loss + model.reliability_loss(diagnostics, targets)
        loss.backward()
        self.assertIsNotNone(model.inheritance_gate[-1].weight.grad)

    def test_error_recovery_metrics_track_recovery_and_relapse(self):
        targets = np.asarray([0, 1, 2, 3])
        predictions = np.asarray([
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 1],
            [2, 2, 2, 1, 1],
            [3, 3, 3, 3, 3],
        ])
        probabilities = np.eye(4, dtype=np.float32)[predictions]
        metrics = error_recovery_metrics(
            probabilities, targets, (0.25, 0.50, 0.65, 0.80, 1.00)
        )
        self.assertEqual(metrics["initial_wrong_samples"], 2)
        self.assertAlmostEqual(metrics["point_recovery"]["0.50"], 0.5)
        self.assertAlmostEqual(metrics["ever_recovered_by_ratio"]["0.65"], 1.0)
        self.assertAlmostEqual(metrics["stable_recovery_from_ratio"]["0.65"], 0.5)
        self.assertAlmostEqual(metrics["final_recovery_rate"], 1.0)
        self.assertAlmostEqual(metrics["initially_correct_final_retention"], 0.5)


if __name__ == "__main__":
    unittest.main()
