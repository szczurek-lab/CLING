"""Import and basic-fit smoke tests."""

from __future__ import annotations

import numpy as np

import cling


def test_version_is_exposed():
    assert isinstance(cling.__version__, str)
    assert cling.__version__ == "0.1.0"


def test_public_api_exports():
    for name in ["build", "run", "fit", "MultiviewDataset", "FittedModel"]:
        assert hasattr(cling, name)


def test_fit_returns_expected_shapes(views):
    fitted = cling.fit(views, K_init=10, seed=0, max_iter=60)
    Z = fitted.get_factors()
    W = fitted.get_weights()
    assert Z.shape == (80, fitted.K)
    assert [w.shape for w in W] == [(30, fitted.K), (24, fitted.K), (18, fitted.K)]
    assert np.all(np.isfinite(Z))
