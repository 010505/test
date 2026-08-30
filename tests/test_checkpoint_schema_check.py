import torch
import unittest

from gesturegraph.checkpoint_schema_check import (
    missing_keys,
    validate_best_accuracy,
    validate_checkpoint,
    validate_dropout,
    validate_frames,
    validate_labels,
    validate_model_name_buildable,
    validate_model_state,
)


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


class MissingKeysTests(unittest.TestCase):
    def test_reports_every_missing_key(self):
        self.assertEqual(sorted(missing_keys({"labels": ["a"]})), sorted([
            "model_state", "frames", "best_accuracy", "model_name", "ablation", "dropout",
        ]))

    def test_complete_checkpoint_has_no_missing_keys(self):
        self.assertEqual(missing_keys(make_valid_checkpoint()), [])


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


class ValidateModelNameBuildableTests(unittest.TestCase):
    def test_registered_name_passes(self):
        self.assertEqual(validate_model_name_buildable("mlp", 14, 64, 0.15, "none"), [])

    def test_unregistered_name_is_rejected(self):
        errors = validate_model_name_buildable("velocity_gated_agcrn", 14, 64, 0.15, "none")
        self.assertTrue(any("not buildable" in e for e in errors))

    def test_empty_name_is_rejected(self):
        self.assertTrue(validate_model_name_buildable("", 14, 64, 0.15, "none"))


class ValidateCheckpointTests(unittest.TestCase):
    def test_a_fully_valid_checkpoint_passes(self):
        report = validate_checkpoint(make_valid_checkpoint())
        self.assertTrue(report["valid"])
        self.assertEqual(report["errors"], [])

    def test_reports_missing_keys_without_running_other_checks(self):
        report = validate_checkpoint({"labels": ["a"]})
        self.assertFalse(report["valid"])
        self.assertTrue(any("missing required key" in e for e in report["errors"]))

    def test_catches_an_unregistered_architecture_name(self):
        checkpoint = make_valid_checkpoint()
        checkpoint["model_name"] = "gated_agcrn"
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


if __name__ == "__main__":
    unittest.main()
