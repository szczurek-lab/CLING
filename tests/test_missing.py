"""Native missing-value (NaN mask) handling."""

from __future__ import annotations

import numpy as np

import cling


def test_fit_runs_with_missing_values(views_with_missing):
    assert np.isnan(views_with_missing[0]).any()
    fitted = cling.fit(views_with_missing, K_init=10, seed=0, max_iter=80)
    Z = fitted.get_factors()
    assert np.all(np.isfinite(Z))
    # Low-rank signal is still recovered despite ~10% missing entries.
    assert fitted.variance_explained_per_view()[0] > 0.5


def test_mask_marks_nan_as_unobserved(views_with_missing):
    ds = cling.MultiviewDataset.from_arrays(views_with_missing)
    obs_nan = np.isnan(views_with_missing[0])
    # Centered data keeps NaN where the input was missing.
    assert np.isnan(ds.centered_views[0])[obs_nan].all()
