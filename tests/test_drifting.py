import unittest

import torch

from gesturegraph.drifting import (
    ConditionalDriftMemoryBank,
    OneStepClassDiffusionModel,
    conditional_categorical_drift_target,
    distillation_loss,
)


class DriftingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_one_step_student_is_normalized_trainable_and_online_equivalent(self):
        model = OneStepClassDiffusionModel()
        self.assertEqual(model.inference_steps, 1)
        views = torch.randn(2, 3, 3, 16, 22)
        lengths = torch.tensor([[4, 8, 16], [4, 8, 16]])
        ratios = torch.tensor([[0.25, 0.5, 1.0], [0.25, 0.5, 1.0]])
        log_probabilities, conditions = model(
            views, lengths, ratios, return_auxiliary=True
        )
        self.assertEqual(tuple(log_probabilities.shape), (2, 3, 14))
        self.assertTrue(torch.allclose(
            log_probabilities.exp().sum(-1), torch.ones(2, 3), atol=1e-5
        ))
        targets = torch.tensor([1, 2])
        loss = F_nll(log_probabilities, targets) + model.denoising_loss(conditions, targets)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

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
        self.assertTrue(torch.allclose(
            batched, torch.stack(online, dim=1), atol=1e-5, rtol=1e-4
        ))

    def test_conditional_drift_target_stays_on_simplex_and_is_frozen(self):
        student_logits = torch.randn(4, 3, 14, requires_grad=True)
        student_log = student_logits.log_softmax(dim=-1)
        teacher = torch.randn(4, 3, 14).softmax(dim=-1)
        contexts = torch.randn(4, 3, 129)
        labels = torch.tensor([0, 0, 1, 1])
        bank = ConditionalDriftMemoryBank(capacity_per_bucket=2)
        target = conditional_categorical_drift_target(
            student_log, teacher, contexts, labels, bank
        )
        self.assertEqual(tuple(target.shape), (4, 3, 14))
        self.assertFalse(target.requires_grad)
        self.assertTrue(torch.all(target >= 0))
        self.assertTrue(torch.allclose(target.sum(-1), torch.ones(4, 3), atol=1e-6))
        self.assertEqual(len(bank), 12)
        loss = -(target * student_log).sum(-1).mean()
        loss.backward()
        self.assertIsNotNone(student_logits.grad)

    def test_memory_bank_is_conditioned_and_capacity_bounded(self):
        bank = ConditionalDriftMemoryBank(capacity_per_bucket=1)
        contexts = torch.randn(2, 2, 5)
        positives = torch.randn(2, 2, 3).softmax(-1)
        negatives = torch.randn(2, 2, 3).softmax(-1)
        labels = torch.tensor([2, 3])
        bank.add(contexts, positives, negatives, labels)
        bank.add(contexts + 1, positives, negatives, labels)
        self.assertEqual(len(bank), 4)
        self.assertIsNotNone(bank.get(2, 0, torch.device("cpu"), torch.float32))
        self.assertIsNone(bank.get(2, 2, torch.device("cpu"), torch.float32))
        self.assertIsNone(bank.get(4, 0, torch.device("cpu"), torch.float32))

    def test_distillation_loss_is_zero_for_equal_distributions(self):
        logits = torch.randn(2, 3, 14)
        log_probabilities = logits.log_softmax(dim=-1)
        loss = distillation_loss(log_probabilities, log_probabilities)
        self.assertAlmostEqual(float(loss), 0.0, places=6)


def F_nll(log_probabilities, targets):
    repeated = targets[:, None].expand(-1, log_probabilities.shape[1]).reshape(-1)
    return torch.nn.functional.nll_loss(
        log_probabilities.reshape(-1, log_probabilities.shape[-1]), repeated
    )


if __name__ == "__main__":
    unittest.main()

