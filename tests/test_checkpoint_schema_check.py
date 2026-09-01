import torch
import unittest

from gesturegraph.checkpoint_schema_check import (
    missing_keys,
    validate_best_accuracy,
    validate_checkpoint,
    validate_dropout,
    validate_frames,
    validate_labels,
    validate_model_config,
    validate_model_loads,
    validate_model_state,
)
from gesturegraph.model import build_model


def make_valid_checkpoint():
    return {
        "model_state": {"weight": torch.randn(4, 4), "bias": torch.zeros(4)},
        "labels": ["grab", "tap", "swipe"],
        "frames": 64,
        "best_accuracy": 0.81,
        "model_name": "mlp",
        "ablation": "none",
        "dropout": 0.15,
    }


def make_real_checkpoint(model_name, num_classes=6, frames=64, dropout=0.15, model_config=None):
    model = build_model(model_name, num_classes, frames, dropout, "none", model_config)
    checkpoint = {
        "model_state": model.state_dict(),
        "labels": [f"class_{i}" for i in range(num_classes)],
        "frames": frames,
        "best_accuracy": 0.75,
        "model_name": model_name,
        "ablation": "none",
        "dropout": dropout,
    }
    if model_config is not None:
        checkpoint["model_config"] = model_config
    return checkpoint


class MissingKeysTests(unittest.TestCase):
    def test_reports_every_missing_key(self):
        self.assertEqual(sorted(missing_keys({"labels": ["a"]})), sorted([
            "model_state", "frames", "best_accuracy", "model_name", "ablation", "dropout",
        ]))

    def test_complete_checkpoint_has_no_missing_keys(self):
        self.assertEqual(missing_keys(make_valid_checkpoint()), [])

    def test_model_config_is_not_required(self):
        checkpoint = make_valid_checkpoint()
        self.assertNotIn("model_config", checkpoint)
        self.assertEqual(missing_keys(checkpoint), [])


class ValidateLabelsTests(unittest.TestCase):
    def test_valid_labels_pass(self):
        self.assertEqual(validate_labels(["grab", "tap"]), [])

    def test_empty_list_is_rejected(self):
        self.assertIn("labels list is empty", validate_labels([]))

    def test_duplicate_labels_are_rejected(self):
        self.assertIn("labels contain duplicates", validate_labels(["grab", "grab"]))

    def test_non_list_is_rejected(self):
        errors = validate_labels("grab")
        self.assertTrue(any("must be a list" in e for e in errors))

    def test_non_string_entries_are_rejected(self):
        errors = validate_labels(["grab", 5])
        self.assertTrue(any("must all be strings" in e for e in errors))


class ValidateFramesTests(unittest.TestCase):
    def test_valid_frame_count_passes(self):
        self.assertEqual(validate_frames(64), [])

    def test_too_few_frames_is_rejected(self):
        self.assertTrue(validate_frames(1))

    def test_non_int_is_rejected(self):
        self.assertTrue(validate_frames(64.0))

    def test_bool_is_rejected_even_though_it_is_technically_an_int(self):
        self.assertTrue(validate_frames(True))


class ValidateBestAccuracyTests(unittest.TestCase):
    def test_in_range_value_passes(self):
        self.assertEqual(validate_best_accuracy(0.7), [])

    def test_out_of_range_value_is_rejected(self):
        self.assertTrue(validate_best_accuracy(1.5))

    def test_non_numeric_is_rejected(self):
        self.assertTrue(validate_best_accuracy("high"))


class ValidateDropoutTests(unittest.TestCase):
    def test_in_range_value_passes(self):
        self.assertEqual(validate_dropout(0.15), [])

    def test_value_of_one_is_rejected(self):
        self.assertTrue(validate_dropout(1.0))

    def test_negative_value_is_rejected(self):
        self.assertTrue(validate_dropout(-0.1))


class ValidateModelConfigTests(unittest.TestCase):
    def test_absent_config_is_fine(self):
        self.assertEqual(validate_model_config(None), [])

    def test_valid_config_passes(self):
        self.assertEqual(validate_model_config({"pe_dim": 8, "adaptive_dim": 10, "attention_heads": 4}), [])

    def test_non_dict_is_rejected(self):
        self.assertTrue(validate_model_config("not a dict"))

    def test_zero_pe_dim_is_rejected(self):
        errors = validate_model_config({"pe_dim": 0})
        self.assertTrue(any("pe_dim" in e for e in errors))

    def test_non_int_adaptive_dim_is_rejected(self):
        errors = validate_model_config({"adaptive_dim": 10.5})
        self.assertTrue(any("adaptive_dim" in e for e in errors))


class ValidateModelStateTests(unittest.TestCase):
    def test_valid_state_dict_passes(self):
        self.assertEqual(validate_model_state({"w": torch.ones(3)}), [])

    def test_empty_state_dict_is_rejected(self):
        self.assertTrue(validate_model_state({}))

    def test_non_tensor_value_is_rejected(self):
        errors = validate_model_state({"w": [1, 2, 3]})
        self.assertTrue(any("not a tensor" in e for e in errors))

    def test_nan_weights_are_rejected(self):
        broken = torch.ones(3)
        broken[0] = float("nan")
        errors = validate_model_state({"w": broken})
        self.assertTrue(any("NaN or infinite" in e for e in errors))


class ValidateModelLoadsTests(unittest.TestCase):
    def test_registered_name_with_matching_state_passes(self):
        checkpoint = make_real_checkpoint("mlp")
        errors = validate_model_loads("mlp", 6, 64, 0.15, "none", None, checkpoint["model_state"])
        self.assertEqual(errors, [])

    def test_unregistered_name_is_rejected(self):
        errors = validate_model_loads("not_a_real_architecture", 14, 64, 0.15, "none", None, None)
        self.assertTrue(any("not buildable" in e for e in errors))

    def test_empty_name_is_rejected(self):
        self.assertTrue(validate_model_loads("", 14, 64, 0.15, "none", None, None))

    def test_state_dict_from_a_different_architecture_is_rejected(self):
        wrong_state = make_real_checkpoint("stgcn", num_classes=6)["model_state"]
        errors = validate_model_loads("mlp", 6, 64, 0.15, "none", None, wrong_state)
        self.assertTrue(any("does not match" in e for e in errors))

    def test_matching_state_with_a_non_default_model_config_passes(self):
        config = {"pe_dim": 4, "adaptive_dim": 6, "attention_heads": 2}
        checkpoint = make_real_checkpoint("spectral_pe_qkv", num_classes=6, model_config=config)
        errors = validate_model_loads("spectral_pe_qkv", 6, 64, 0.15, "none", config, checkpoint["model_state"])
        self.assertEqual(errors, [])

    def test_non_default_model_config_mismatch_is_caught_not_silently_accepted(self):
        real_config = {"pe_dim": 4, "adaptive_dim": 6, "attention_heads": 2}
        checkpoint = make_real_checkpoint("spectral_pe_qkv", num_classes=6, model_config=real_config)
        errors = validate_model_loads("spectral_pe_qkv", 6, 64, 0.15, "none", None, checkpoint["model_state"])
        self.assertTrue(any("does not match" in e for e in errors))

    def test_missing_model_state_only_checks_buildability(self):
        errors = validate_model_loads("mlp", 6, 64, 0.15, "none", None, None)
        self.assertEqual(errors, [])


class ValidateCheckpointTests(unittest.TestCase):
    def test_a_fully_valid_checkpoint_passes(self):
        report = validate_checkpoint(make_real_checkpoint("mlp"))
        self.assertTrue(report["valid"])
        self.assertEqual(report["errors"], [])

    def test_reports_missing_keys_without_running_other_checks(self):
        report = validate_checkpoint({"labels": ["a"]})
        self.assertFalse(report["valid"])
        self.assertTrue(any("missing required key" in e for e in report["errors"]))

    def test_catches_an_unregistered_architecture_name(self):
        checkpoint = make_valid_checkpoint()
        checkpoint["model_name"] = "not_a_real_architecture"
        report = validate_checkpoint(checkpoint)
        self.assertFalse(report["valid"])
        self.assertTrue(any("not buildable" in e for e in report["errors"]))

    def test_collects_multiple_independent_problems_at_once(self):
        checkpoint = make_valid_checkpoint()
        checkpoint["dropout"] = 5.0
        checkpoint["labels"] = []
        report = validate_checkpoint(checkpoint)
        self.assertFalse(report["valid"])
        self.assertGreaterEqual(len(report["errors"]), 2)

    def test_a_real_experimental_checkpoint_with_custom_config_validates_end_to_end(self):
        config = {"pe_dim": 4, "adaptive_dim": 6, "attention_heads": 2}
        checkpoint = make_real_checkpoint("velocity_gated_agcrn", num_classes=6, model_config=config)
        report = validate_checkpoint(checkpoint)
        self.assertTrue(report["valid"])

    def test_catches_a_checkpoint_that_is_buildable_but_whose_weights_do_not_actually_load(self):
        real_config = {"pe_dim": 4, "adaptive_dim": 6, "attention_heads": 2}
        checkpoint = make_real_checkpoint("gwnet_adaptive_support", num_classes=6, model_config=real_config)
        del checkpoint["model_config"]
        report = validate_checkpoint(checkpoint)
        self.assertFalse(report["valid"])
        self.assertTrue(any("does not match" in e for e in report["errors"]))


if __name__ == "__main__":
    unittest.main()
