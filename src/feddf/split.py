from __future__ import annotations

import numpy as np
from torch.utils.data import Subset


def iid_split(dataset, targets, *, n_clients: int, seed: int) -> list[Subset]:
    """Randomly shuffle and divide dataset equally across n_clients."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(targets))
    rng.shuffle(indices)
    splits = np.array_split(indices, n_clients)
    return [Subset(dataset, split.tolist()) for split in splits]


def dirichlet_split(dataset, targets, *, n_clients: int, alpha: float, seed: int) -> list[Subset]:
    """Non-IID split via Dirichlet distribution over class labels.

    Lower alpha means more heterogeneous (each client dominated by fewer classes).
    alpha=0.5 is a common research benchmark; alpha=0.1 is highly non-IID.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(targets)
    n_classes = int(np.max(labels)) + 1
    client_indices: list[list[int]] = [[] for _ in range(n_clients)]

    for class_id in range(n_classes):
        class_idx = np.flatnonzero(labels == class_id)
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(np.full(n_clients, alpha))
        split_points = (np.cumsum(proportions[:-1]) * len(class_idx)).astype(int)
        for client_id, split in enumerate(np.split(class_idx, split_points)):
            client_indices[client_id].extend(split.tolist())

    for i in range(n_clients):
        arr = np.asarray(client_indices[i])
        rng.shuffle(arr)
        client_indices[i] = arr.tolist()

    return [Subset(dataset, indices) for indices in client_indices]
