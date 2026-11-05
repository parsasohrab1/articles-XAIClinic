from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau


def topk_sets(importances: np.ndarray, k: int) -> list[frozenset]:
    """importances: (n_resamples, n_features) -> top-k feature set per resample."""
    return [frozenset(np.argsort(-row)[:k]) for row in importances]


def mean_jaccard(sets: list[frozenset]) -> float:
    n = len(sets)
    vals = [
        len(sets[i] & sets[j]) / len(sets[i] | sets[j]) for i in range(n) for j in range(i + 1, n)
    ]
    return float(np.mean(vals))


def mean_kendall(importances: np.ndarray) -> float:
    n = len(importances)
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            t, _ = kendalltau(importances[i], importances[j])
            if not np.isnan(t):
                vals.append(t)
    return float(np.mean(vals))


def selection_frequency(sets: list[frozenset], n_features: int) -> np.ndarray:
    f = np.zeros(n_features)
    for s in sets:
        for j in s:
            f[j] += 1
    return f / len(sets)
