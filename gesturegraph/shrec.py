from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import GestureSample, normalize_sequence, resample_sequence

LABELS_14 = [
    "grab", "expand", "pinch", "rotation_cw", "rotation_ccw", "tap",
    "swipe_right", "swipe_left", "swipe_up", "swipe_down", "swipe_x",
    "swipe_v", "swipe_plus", "shake",
]


def load_shrec17(root: str | Path, split: str, frames: int = 64, classes: int = 14) -> list[GestureSample]:
    """Load the official SHREC'17 skeleton files and split lists.

    Each split row is: gesture, finger, subject, trial, label14, label28, frames.
    """
    if split not in {"train", "test"}:
        raise ValueError("SHREC split must be train or test")
    if classes not in {14, 28}:
        raise ValueError("SHREC supports 14 or 28 classes")
    root = Path(root)
    list_path = root / f"{split}_gestures.txt"
    if not list_path.exists():
        raise FileNotFoundError(f"missing official split file: {list_path}")
    rows = np.loadtxt(list_path, dtype=np.int64).reshape(-1, 7)
    samples = []
    for gesture, finger, subject, trial, label14, label28, frame_count in rows:
        path = root / f"gesture_{gesture}" / f"finger_{finger}" / f"subject_{subject}" / f"essai_{trial}" / "skeletons_world.txt"
        values = np.loadtxt(path, dtype=np.float32)
        sequence = values.reshape(int(frame_count), 22, 3)
        label_index = int(label14 if classes == 14 else label28) - 1
        label = LABELS_14[label_index] if classes == 14 else f"class_{label_index:02d}"
        samples.append(GestureSample(path, label, resample_sequence(normalize_sequence(sequence), frames), str(subject), split))
    return samples


def load_shrec17_npz(root: str | Path, split: str, frames: int = 64, classes: int = 14) -> list[GestureSample]:
    """Load the safe numeric archive produced by scripts/convert_ddnet_shrec.py."""
    if split not in {"train", "test"}:
        raise ValueError("SHREC split must be train or test")
    if classes not in {14, 28}:
        raise ValueError("SHREC supports 14 or 28 classes")
    path = Path(root) / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing converted SHREC split: {path}")
    with np.load(path, allow_pickle=False) as archive:
        coordinates = np.asarray(archive["coordinates"], dtype=np.float32)
        offsets = np.asarray(archive["offsets"], dtype=np.int64)
        label_key = "labels14" if classes == 14 else "labels28"
        labels = np.asarray(archive[label_key], dtype=np.int64)
    if offsets.ndim != 1 or len(offsets) != len(labels) + 1:
        raise ValueError(f"{path}: invalid offsets/labels")
    if coordinates.ndim != 3 or coordinates.shape[1:] != (22, 3):
        raise ValueError(f"{path}: coordinates must have shape [all_frames, 22, 3]")
    if offsets[0] != 0 or offsets[-1] != len(coordinates) or np.any(np.diff(offsets) < 2):
        raise ValueError(f"{path}: invalid sequence offsets")

    samples = []
    for index, label_value in enumerate(labels):
        start, end = int(offsets[index]), int(offsets[index + 1])
        sequence = coordinates[start:end]
        label_index = int(label_value) - 1
        label = LABELS_14[label_index] if classes == 14 else f"class_{label_index:02d}"
        samples.append(
            GestureSample(
                Path(f"{path}#{index}"),
                label,
                resample_sequence(normalize_sequence(sequence), frames),
                None,
                split,
            )
        )
    return samples
