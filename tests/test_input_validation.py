"""Input validation on the dataset container."""

from __future__ import annotations

import numpy as np
import pytest

import cling


def test_empty_view_list_raises():
    with pytest.raises(ValueError):
        cling.MultiviewDataset.from_arrays([])


def test_mismatched_sample_counts_raise():
    with pytest.raises(ValueError):
        cling.MultiviewDataset.from_arrays(
            [np.zeros((10, 5)), np.zeros((9, 4))]
        )


def test_infinite_entries_raise():
    v = np.zeros((10, 5))
    v[0, 0] = np.inf
    with pytest.raises(ValueError):
        cling.MultiviewDataset.from_arrays([v])


def test_all_nan_feature_column_raises():
    v = np.zeros((10, 5))
    v[:, 2] = np.nan
    with pytest.raises(ValueError):
        cling.MultiviewDataset.from_arrays([v])
