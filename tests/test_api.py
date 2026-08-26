"""Public variant selection and dataset API."""

from __future__ import annotations

import numpy as np
import pytest

import cling


@pytest.mark.parametrize("variant", ["CLING", "CLING-MGP", "CLING-ARD"])
def test_display_name_variants_run(views, variant):
    fitted = cling.fit(views, K_init=8, seed=0, max_iter=30, variant=variant)
    assert fitted.get_factors().shape[0] == 80


@pytest.mark.parametrize("alias", ["cling", "cling-mgp", "CLING_ARD"])
def test_variant_names_are_case_and_separator_insensitive(views, alias):
    model = cling.build(views, K_init=6, seed=0, variant=alias)
    assert model.K == 6


def test_unknown_variant_raises(views):
    with pytest.raises(ValueError):
        cling.build(views, K_init=6, seed=0, variant="not-a-variant")


def test_build_then_run_matches_fit_shapes(views):
    model = cling.build(views, K_init=8, seed=0)
    fitted = cling.run(model, cling.TrainingOptions(max_iter=20, verbose=False))
    assert fitted.get_factors().shape == (80, fitted.K)


def test_multiview_dataset_shapes(views):
    ds = cling.MultiviewDataset.from_arrays(views, view_names=["a", "b", "c"])
    assert ds.N == 80
    assert ds.M == 3
    assert ds.D == [30, 24, 18]
    assert ds.G == 1
    assert np.all(np.isfinite(ds.centered_views[0]))
