import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gesturegraph.data import (
    GestureDataset,
    GestureSample,
    discover_samples,
    load_sample,
    normalize_sequence,
    resample_sequence,
    stratified_split,
)


def make_sample_file(directory, name, label="grab", frames=20, bad_json=False, missing_label=False, with_nan=False):
    path = Path(directory) / name
    if bad_json:
        path.write_text("{not valid json", encoding="utf-8")
        return path
    sequence = np.random.default_rng(0).normal(size=(frames, 22, 3)).tolist()
    if with_nan:
        sequence[0][0][0] = float("nan")
    payload = {"sequence": sequence}
    if not missing_label:
        payload["label"] = label
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class NormalizeSequenceTests(unittest.TestCase):
    def test_translation_is_removed(self):
        rng = np.random.default_rng(3)
        sequence = rng.normal(size=(10, 22, 3)).astype(np.float32)
        sequence += np.array([100.0, -50.0, 7.0], dtype=np.float32)  # shove the whole hand across the room
        normalized = normalize_sequence(sequence)
        # first joint (the wrist) is the origin at every frame once centred
        self.assertTrue(np.allclose(normalized[:, 0, :], 0.0, atol=1e-5))

    def test_scale_is_bone_length_invariant(self):
        rng = np.random.default_rng(3)
        base = rng.normal(size=(10, 22, 3)).astype(np.float32)
        doubled = base * 2.0
        normal_base = normalize_sequence(base)
        normal_doubled = normalize_sequence(doubled)
        # scaling the whole hand up shouldn't change the normalized shape
        self.assertTrue(np.allclose(normal_base, normal_doubled, atol=1e-4))


class LoadSampleTests(unittest.TestCase):
    def test_rejects_missing_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = make_sample_file(directory, "sample.json", missing_label=True)
            with self.assertRaises(ValueError):
                load_sample(path)

    def test_rejects_nan_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = make_sample_file(directory, "sample.json", with_nan=True)
            with self.assertRaises(ValueError):
                load_sample(path)

    def test_valid_sample_resamples_to_requested_frame_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = make_sample_file(directory, "sample.json", frames=17)
            sample = load_sample(path, frames=32)
            self.assertEqual(sample.sequence.shape, (32, 22, 3))
            self.assertEqual(sample.label, "grab")


class DiscoverSamplesTests(unittest.TestCase):
    def test_collects_multiple_errors_before_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            make_sample_file(directory, "good.json")
            make_sample_file(directory, "broken.json", bad_json=True)
            make_sample_file(directory, "unlabeled.json", missing_label=True)
            with self.assertRaises(ValueError) as context:
                discover_samples(directory)
            # both bad files should be reported together, not just the first one hit
            message = str(context.exception)
            self.assertEqual(message.count("\n- "), 2)
            self.assertIn("unlabeled.json", message)
            # NOTE: the malformed-JSON branch currently reports json.JSONDecodeError's
            # own message, which doesn't include the file path (unlike the ValueError
            # branch, e.g. "missing label" above). Worth a small follow-up so both
            # error kinds are equally easy to locate in a large dataset.

    def test_raises_on_empty_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                discover_samples(directory)


class StratifiedSplitTests(unittest.TestCase):
    def test_a_single_item_label_stays_entirely_in_train(self):
        sequence = np.zeros((64, 22, 3), dtype=np.float32)
        samples = [GestureSample(Path("lonely.json"), "rare", sequence)]
        samples += [GestureSample(Path(f"common-{i}.json"), "common", sequence) for i in range(6)]
        train, val = stratified_split(samples, val_ratio=0.2)
        labels_in_train = {sample.label for sample in train}
        labels_in_val = {sample.label for sample in val}
        self.assertIn("rare", labels_in_train)
        self.assertNotIn("rare", labels_in_val)

    def test_split_is_reproducible_given_the_same_seed(self):
        sequence = np.zeros((64, 22, 3), dtype=np.float32)
        samples = [GestureSample(Path(f"a-{i}.json"), "a", sequence) for i in range(10)]
        first_train, first_val = stratified_split(samples, val_ratio=0.3, seed=7)
        second_train, second_val = stratified_split(samples, val_ratio=0.3, seed=7)
        self.assertEqual([s.path for s in first_train], [s.path for s in second_train])
        self.assertEqual([s.path for s in first_val], [s.path for s in second_val])


class GestureDatasetTests(unittest.TestCase):
    def test_length_and_tensor_shape(self):
        sequence = np.zeros((64, 22, 3), dtype=np.float32)
        samples = [GestureSample(Path("a.json"), "grab", sequence), GestureSample(Path("b.json"), "tap", sequence)]
        dataset = GestureDataset(samples, labels=["grab", "tap"])
        self.assertEqual(len(dataset), 2)
        tensor, label_index = dataset[0]
        self.assertEqual(tuple(tensor.shape), (3, 64, 22))  # channels, time, joints
        self.assertEqual(label_index, 0)

    def test_augmentation_does_not_change_the_frame_count(self):
        sequence = np.random.default_rng(4).normal(size=(64, 22, 3)).astype(np.float32)
        samples = [GestureSample(Path("a.json"), "grab", sequence)]
        dataset = GestureDataset(samples, labels=["grab"], augment=True)
        tensor, _ = dataset[0]
        self.assertEqual(tuple(tensor.shape), (3, 64, 22))


if __name__ == "__main__":
    unittest.main()
