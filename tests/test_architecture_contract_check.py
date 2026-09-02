import unittest

from gesturegraph.architecture_contract_check import (
    check_all_known_architectures,
    check_architecture,
    check_batch_independence,
    check_deterministic_eval,
    check_gradient_flow,
    check_output_shape,
    KNOWN_ARCHITECTURES,
)
from gesturegraph.model import build_model


class SingleContractCheckTests(unittest.TestCase):
    def test_output_shape_passes_for_stgcn(self):
        model = build_model("stgcn", num_classes=6)
        self.assertEqual(check_output_shape(model, num_classes=6, frames=64), [])

    def test_deterministic_eval_passes_for_mlp(self):
        model = build_model("mlp", num_classes=6)
        self.assertEqual(check_deterministic_eval(model, frames=64), [])

    def test_gradient_flow_passes_for_stgcn(self):
        model = build_model("stgcn", num_classes=6)
        self.assertEqual(check_gradient_flow(model, num_classes=6, frames=64), [])

    def test_batch_independence_passes_for_stgcn(self):
        model = build_model("stgcn", num_classes=6)
        self.assertEqual(check_batch_independence(model, frames=64), [])


class CheckArchitectureTests(unittest.TestCase):
    def test_registered_architecture_is_checked_and_passes(self):
        result = check_architecture("stgcn", num_classes=6, frames=64)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["errors"], [])

    def test_unregistered_architecture_is_marked_pending_not_failed(self):
        result = check_architecture("agcrn_factorized_adjacency_v2", num_classes=6, frames=64)
        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["errors"], [])

    def test_completely_unknown_name_is_also_pending(self):
        result = check_architecture("not_a_real_model", num_classes=6, frames=64)
        self.assertEqual(result["status"], "pending")


class CheckAllKnownArchitecturesTests(unittest.TestCase):
    def test_covers_every_name_in_the_known_list(self):
        results = check_all_known_architectures(num_classes=6, frames=64)
        checked_names = {row["name"] for row in results}
        self.assertEqual(checked_names, set(KNOWN_ARCHITECTURES))

    def test_currently_registered_backbones_pass_today(self):
        results = check_all_known_architectures(num_classes=6, frames=64)
        by_name = {row["name"]: row for row in results}
        self.assertEqual(by_name["stgcn"]["status"], "pass")
        self.assertEqual(by_name["mlp"]["status"], "pass")
        self.assertEqual(by_name["velocity_gated_agcrn"]["status"], "pass")
        self.assertEqual(by_name["gated_agcrn"]["status"], "pass")
        self.assertEqual(by_name["velocity_agcrn"]["status"], "pass")
        self.assertEqual(by_name["spectral_pe_qkv_stable"]["status"], "pass")

    def test_not_yet_implemented_backbones_are_pending_not_failing(self):
        results = check_all_known_architectures(
            names=KNOWN_ARCHITECTURES + ("agcrn_factorized_adjacency_v2",),
            num_classes=6,
            frames=64,
        )
        by_name = {row["name"]: row for row in results}
        self.assertEqual(by_name["agcrn_factorized_adjacency_v2"]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
