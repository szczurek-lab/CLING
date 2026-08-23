"""Saving and loading fitted models."""

from __future__ import annotations

import numpy as np

import cling


def test_save_load_roundtrip(tmp_path, views):
    fitted = cling.fit(views, K_init=8, seed=0, max_iter=40, view_names=["a", "b", "c"])
    path = fitted.save(tmp_path / "fit.npz")
    assert path.exists()

    loaded = cling.FittedModel.load(path)
    assert loaded.K == fitted.K
    assert loaded.view_names == fitted.view_names
    assert np.array_equal(loaded.get_factors(), fitted.get_factors())
    for w_loaded, w_orig in zip(loaded.get_weights(), fitted.get_weights()):
        assert np.array_equal(w_loaded, w_orig)
    assert np.array_equal(
        loaded.variance_explained_per_view(), fitted.variance_explained_per_view()
    )


def test_saved_variant_uses_display_name(tmp_path, views):
    fitted = cling.fit(views, K_init=6, seed=0, max_iter=20, variant="CLING-MGP")
    loaded = cling.FittedModel.load(fitted.save(tmp_path / "mgp.npz"))
    assert loaded.model.variant == "CLING-MGP"
