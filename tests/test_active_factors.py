"""Active-factor selection: per-view R^2 keep rule."""

from __future__ import annotations

import numpy as np

import cling
from cling import find_inactive_factors


def test_keep_rule_is_per_view_any():
    R2 = np.array([[0.5, 0.5, 0.5], [0.02, 0.0, 0.0], [0.0, 0.0, 0.0]])
    keep = ~find_inactive_factors(R2, 0.01)
    assert keep.tolist() == [True, True, False]


def test_n_active_factors_matches_mask(views):
    fitted = cling.fit(views, K_init=12, seed=0, max_iter=80)
    mask = fitted.active_factor_mask(0.01)
    assert mask.dtype == bool
    assert mask.shape == (fitted.K,)
    assert fitted.n_active_factors(0.01) == max(int(mask.sum()), 1)
    assert 1 <= fitted.n_active_factors(0.01) <= fitted.K
