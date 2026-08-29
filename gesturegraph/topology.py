import numpy as np

NUM_NODES = 22
EDGES = [
    (0, 1),
    (0, 2), (2, 3), (3, 4), (4, 5),
    (1, 6), (6, 7), (7, 8), (8, 9),
    (1, 10), (10, 11), (11, 12), (12, 13),
    (1, 14), (14, 15), (15, 16), (16, 17),
    (1, 18), (18, 19), (19, 20), (20, 21),
]


def normalized_adjacency() -> np.ndarray:
    """Return symmetric D^-1/2 (A+I) D^-1/2 for the 22-node hand graph."""
    adjacency = np.eye(NUM_NODES, dtype=np.float32)
    for source, target in EDGES:
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0
    degree = adjacency.sum(axis=1)
    inverse_sqrt = np.diag(np.power(degree, -0.5))
    return inverse_sqrt @ adjacency @ inverse_sqrt


def binary_adjacency(include_self: bool = True) -> np.ndarray:
    """Return the unnormalised physical hand graph as a binary matrix."""
    adjacency = np.eye(NUM_NODES, dtype=np.float32) if include_self else np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
    for source, target in EDGES:
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0
    return adjacency


def laplacian_eigenpairs(dimensions: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic non-trivial eigenpairs of the normalised Laplacian.

    The first (constant) eigenvector is excluded. Eigenvector signs are fixed by
    making the largest-magnitude entry in each vector positive, which keeps
    checkpoints and tests reproducible across runs.
    """
    if not 1 <= dimensions < NUM_NODES:
        raise ValueError(f"dimensions must be in [1, {NUM_NODES - 1}]")
    adjacency = binary_adjacency(include_self=False).astype(np.float64)
    degree = adjacency.sum(axis=1)
    inverse_sqrt = np.diag(np.power(np.maximum(degree, 1.0), -0.5))
    laplacian = np.eye(NUM_NODES, dtype=np.float64) - inverse_sqrt @ adjacency @ inverse_sqrt
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    values = eigenvalues[1:dimensions + 1].copy()
    encoding = eigenvectors[:, 1:dimensions + 1].copy()
    for column in range(encoding.shape[1]):
        pivot = int(np.argmax(np.abs(encoding[:, column])))
        if encoding[pivot, column] < 0:
            encoding[:, column] *= -1.0
    return values.astype(np.float32), encoding.astype(np.float32)


def laplacian_positional_encoding(dimensions: int = 8) -> np.ndarray:
    """Return only the eigenvector part of :func:`laplacian_eigenpairs`."""
    _, encoding = laplacian_eigenpairs(dimensions)
    return encoding
