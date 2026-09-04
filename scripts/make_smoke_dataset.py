"""Create a tiny deterministic dataset for testing the complete training pipeline."""
import json
from pathlib import Path

import numpy as np


def sample(label, index):
    rng = np.random.default_rng(index + (0 if label == "open_palm" else 100))
    sequence = np.zeros((64, 22, 3), dtype=np.float32)
    for frame in range(64):
        phase = frame / 63
        for node in range(22):
            finger = max(0, (node - 2) // 4)
            segment = max(0, (node - 2) % 4) / 3
            sequence[frame, node] = [(finger - 2) * .18, -segment, np.sin(phase * np.pi * 2) * .03]
        if label == "fist":
            sequence[frame, :, 1] *= .22
            sequence[frame, :, 2] += np.arange(22) % 4 * .08
    sequence += rng.normal(0, .008, sequence.shape)
    return {"format": "GestureGraph-22", "label": label, "fps": 30, "frames": 64, "sequence": sequence.round(6).tolist()}


root = Path("data/smoke")
root.mkdir(parents=True, exist_ok=True)
for label in ("open_palm", "fist"):
    for index in range(8):
        (root / f"{label}-{index:02d}.json").write_text(json.dumps(sample(label, index)), encoding="utf-8")
print(f"Wrote 16 smoke-test samples to {root}")
