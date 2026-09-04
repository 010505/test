"""Generate a tiny SHREC-shaped dataset for integration tests only."""
from pathlib import Path

import numpy as np

root = Path("data/smoke_shrec/HandGestureDataset_SHREC2017")
rng = np.random.default_rng(7)


def make_sequence(label, subject, trial, frames=12):
    sequence = np.zeros((frames, 22, 3), dtype=np.float32)
    for frame in range(frames):
        progress = frame / (frames - 1)
        sequence[frame, :, 0] = np.linspace(-1, 1, 22)
        sequence[frame, :, 1] = np.arange(22) % 4
        if label == 1:
            sequence[frame, :, 2] = progress
        else:
            sequence[frame, :, 1] *= .2
            sequence[frame, :, 2] = 1 - progress
    return sequence + rng.normal(0, .01, sequence.shape), frames


def build(split, subjects, trials):
    rows = []
    for label in (1, 2):
        for subject in subjects:
            for trial in trials:
                sequence, frames = make_sequence(label, subject, trial)
                target = root / f"gesture_{label}/finger_1/subject_{subject}/essai_{trial}"
                target.mkdir(parents=True, exist_ok=True)
                np.savetxt(target / "skeletons_world.txt", sequence.reshape(frames, 66))
                rows.append([label, 1, subject, trial, label, label, frames])
    np.savetxt(root / f"{split}_gestures.txt", np.asarray(rows), fmt="%d")


build("train", range(1, 5), range(1, 3))
build("test", range(5, 7), range(1, 2))
print(root)
