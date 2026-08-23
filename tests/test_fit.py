"""Inference behaviour: ELBO, factor selection, determinism, diagnostics."""

from __future__ import annotations

import numpy as np

import cling


def test_elbo_is_finite_and_monotone(views):
    fitted = cling.fit(views, K_init=10, seed=0, max_iter=80)
    elbo = np.asarray(fitted.training.elbo_history)
    assert elbo.size > 0
    assert np.all(np.isfinite(elbo))
    # CAVI is coordinate ascent on the ELBO: non-decreasing up to fp noise.
    diffs = np.diff(elbo)
    assert np.all(diffs >= -1e-6 * np.abs(elbo[:-1]) - 1e-8)


def test_recovers_true_number_of_factors(views):
    # Data is exactly rank 4; the R2 >= 0.01 criterion should keep 4 factors.
    fitted = cling.fit(views, K_init=10, seed=0, max_iter=200)
    r2 = fitted.variance_explained_per_factor()
    n_active = int((r2 >= 0.01).sum())
    assert n_active == 4


def test_fit_is_deterministic_under_seed(views):
    a = cling.fit(views, K_init=10, seed=0, max_iter=60)
    b = cling.fit(views, K_init=10, seed=0, max_iter=60)
    assert np.array_equal(a.get_factors(), b.get_factors())
    assert np.array_equal(
        np.asarray(a.training.elbo_history), np.asarray(b.training.elbo_history)
    )


def test_variance_explained_in_unit_interval(views):
    fitted = cling.fit(views, K_init=10, seed=0, max_iter=60)
    for r2 in (
        fitted.variance_explained_per_view(),
        fitted.variance_explained_per_factor(),
        fitted.variance_explained_per_factor_view(),
    ):
        r2 = np.asarray(r2)
        assert np.all(r2 >= 0.0) and np.all(r2 <= 1.0)


def test_reconstruction_shapes(views):
    fitted = cling.fit(views, K_init=8, seed=0, max_iter=30)
    recon = fitted.reconstruct()
    assert [r.shape for r in recon] == [(80, 30), (80, 24), (80, 18)]
