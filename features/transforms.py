"""Feature transforms applied at training time.

- Subject-wise mean centring of the edge features: for each subject and edge,
  the mean over that subject's epochs is subtracted (no labels are used).
- Edge z-scoring with training-fold statistics only: per band, mean and SD over the
  full symmetric 60 x 60 matrices (zero diagonal included) of the fit epochs.
- Optional node z-scoring with training-fold statistics (a hyperparameter).
"""
from __future__ import annotations

import numpy as np


def center_edges(edge_up: np.ndarray, subj: np.ndarray) -> np.ndarray:
    """edge_up (E,1770,B) -> subject-wise mean-centred copy."""
    edge = np.abs(edge_up).astype(np.float32)
    for s in np.unique(subj):
        m = subj == s
        edge[m] = edge[m] - edge[m].mean(axis=0, keepdims=True)
    return edge


def edge_zscore_stats(edge_up: np.ndarray, fit_idx: np.ndarray, n_nodes: int = 60):
    """Per-band mean/SD over the full symmetric matrices of the fit epochs."""
    e = edge_up[fit_idx].astype(np.float64)
    n_full = len(fit_idx) * n_nodes * n_nodes
    mu = 2.0 * e.sum(axis=(0, 1), keepdims=True) / n_full
    ex2 = 2.0 * (e ** 2).sum(axis=(0, 1), keepdims=True) / n_full
    sd = np.sqrt(np.maximum(ex2 - mu ** 2, 0.0)) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def node_zscore(node: np.ndarray, fit_idx: np.ndarray) -> np.ndarray:
    n = node[fit_idx]
    return (node - n.mean(axis=(0, 1), keepdims=True)) / (n.std(axis=(0, 1), keepdims=True) + 1e-6)
