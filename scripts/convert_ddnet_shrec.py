from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
try:
    from numpy._core import multiarray
except ImportError:  # NumPy 1.x
    from numpy.core import multiarray


class NumpyOnlyUnpickler(pickle.Unpickler):
    """Load legacy NumPy arrays without permitting arbitrary pickle globals."""

    ALLOWED = {
        ("numpy", "dtype"): np.dtype,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy.core.multiarray", "_reconstruct"): multiarray._reconstruct,
        ("numpy.core.multiarray", "scalar"): multiarray.scalar,
    }

    def find_class(self, module: str, name: str):
        try:
            return self.ALLOWED[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}") from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(source: Path, destination: Path) -> dict:
    with source.open("rb") as stream:
        payload = NumpyOnlyUnpickler(stream).load()
    expected = {"pose", "coarse_label", "fine_label"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{source}: expected keys {sorted(expected)}")
    poses = payload["pose"]
    labels14 = np.asarray(payload["coarse_label"], dtype=np.int16)
    labels28 = np.asarray(payload["fine_label"], dtype=np.int16)
    if not (len(poses) == len(labels14) == len(labels28)):
        raise ValueError(f"{source}: pose and label counts differ")

    sequences = []
    offsets = [0]
    for index, pose in enumerate(poses):
        sequence = np.asarray(pose, dtype=np.float32)
        if sequence.ndim == 2 and sequence.shape[1] == 66:
            sequence = sequence.reshape(-1, 22, 3)
        if sequence.ndim != 3 or sequence.shape[1:] != (22, 3):
            raise ValueError(f"{source}: sample {index} has shape {sequence.shape}")
        if len(sequence) < 2 or not np.isfinite(sequence).all():
            raise ValueError(f"{source}: sample {index} is empty or non-finite")
        sequences.append(sequence)
        offsets.append(offsets[-1] + len(sequence))

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        coordinates=np.concatenate(sequences, axis=0),
        offsets=np.asarray(offsets, dtype=np.int64),
        labels14=labels14,
        labels28=labels28,
    )
    return {
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "output": str(destination.resolve()),
        "samples": len(sequences),
        "frames": int(offsets[-1]),
        "label14_min": int(labels14.min()),
        "label14_max": int(labels14.max()),
        "label28_min": int(labels28.min()),
        "label28_max": int(labels28.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DD-Net SHREC pickles to safe numeric NPZ archives")
    parser.add_argument("--source", required=True, help="DD-Net data/SHREC directory")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()
    source = Path(args.source)
    output = Path(args.output)
    metadata = {
        "provenance": "fandulu/DD-Net data/SHREC",
        "preprocessing": "Each of the 66 coordinate channels was median-filtered by DD-Net before serialization.",
        "splits": {},
    }
    for split, expected_count in (("train", 1960), ("test", 840)):
        row = convert(source / f"{split}.pkl", output / f"{split}.npz")
        if row["samples"] != expected_count:
            raise ValueError(f"{split}: expected {expected_count} official samples, got {row['samples']}")
        metadata["splits"][split] = row
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
