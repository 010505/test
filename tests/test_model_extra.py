import unittest

import torch

from gesturegraph.model import FlatMLP, HandSTGCN, build_model


class ModelInputValidationTests(unittest.TestCase):
    def test_wrong_number_of_channels_is_rejected(self):
        model = HandSTGCN(num_classes=6)
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 4, 64, 22))  # 4 channels instead of xyz

    def test_wrong_number_of_joints_is_rejected(self):
        model = HandSTGCN(num_classes=6)
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 3, 64, 21))  # one joint short

    def test_wrong_rank_is_rejected(self):
        model = HandSTGCN(num_classes=6)
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 3, 64))  # missing the joints axis entirely


class ModelDeterminismTests(unittest.TestCase):
    def test_eval_mode_is_deterministic_across_calls(self):
        # dropout/batchnorm can make repeated calls differ in train mode;
        # in eval mode the same input should always give the same output.
        model = HandSTGCN(num_classes=6).eval()
        x = torch.randn(2, 3, 64, 22)
        with torch.no_grad():
            first = model(x)
            second = model(x)
        self.assertTrue(torch.allclose(first, second))

    def test_batch_of_one_matches_slice_of_a_larger_batch(self):
        # a model with correct batchnorm/graph handling shouldn't let samples
        # in a batch leak into each other's predictions
        model = HandSTGCN(num_classes=6).eval()
        torch.manual_seed(0)
        x = torch.randn(4, 3, 64, 22)
        with torch.no_grad():
            batched = model(x)
            single = model(x[2:3])
        self.assertTrue(torch.allclose(batched[2:3], single, atol=1e-5))


class GradientFlowTests(unittest.TestCase):
    def test_every_parameter_receives_a_gradient(self):
        # catches accidentally-frozen layers or a broken residual path that
        # silently disconnects part of the network from the loss
        model = HandSTGCN(num_classes=6)
        x = torch.randn(2, 3, 64, 22)
        target = torch.tensor([0, 1])
        loss = torch.nn.functional.cross_entropy(model(x), target)
        loss.backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} got no gradient")
            self.assertGreater(param.grad.abs().sum().item(), 0, f"{name} gradient is all zero")


class BuildModelTests(unittest.TestCase):
    def test_unknown_model_name_raises(self):
        with self.assertRaises(ValueError):
            build_model("transformer", num_classes=6)

    def test_no_graph_ablation_uses_identity_adjacency(self):
        model = build_model("stgcn", num_classes=6, ablation="no_graph")
        self.assertTrue(torch.equal(model.adjacency, torch.eye(22)))

    def test_mlp_and_stgcn_have_different_parameter_counts(self):
        mlp = build_model("mlp", num_classes=14)
        stgcn = build_model("stgcn", num_classes=14)
        mlp_params = sum(p.numel() for p in mlp.parameters())
        stgcn_params = sum(p.numel() for p in stgcn.parameters())
        # sanity check more than an exact number, since either backbone
        # might get tuned later - just confirms build_model actually wires
        # up two structurally different networks
        self.assertNotEqual(mlp_params, stgcn_params)


if __name__ == "__main__":
    unittest.main()
