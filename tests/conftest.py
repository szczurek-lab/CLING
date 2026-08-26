"""Shared fixtures: small deterministic low-rank multi-view datasets."""

from __future__ import annotations

import numpy as np
import pytest

TRUE_K = 4


def _make_multiview(n_samples, dims, true_k=TRUE_K, noise=0.2, seed=0):
    """Return (views, Z_true): low-rank Gaussian views sharing latent Z."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_samples, true_k))
    views = []
    for d in dims:
        W = rng.standard_normal((d, true_k))
        views.append(Z @ W.T + noise * rng.standard_normal((n_samples, d)))
    return views, Z


@pytest.fixture
def views():
    """Three complete views, N=80, shared rank-4 structure."""
    v, _ = _make_multiview(80, [30, 24, 18])
    return v


@pytest.fixture
def views_with_missing():
    """Two views, N=80, with a scatter of missing (NaN) entries in view 0."""
    v, _ = _make_multiview(80, [30, 24], seed=1)
    rng = np.random.default_rng(2)
    mask = rng.random(v[0].shape) < 0.1
    v[0][mask] = np.nan
    return v
