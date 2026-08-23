"""Factor-pruning helpers.

A factor is dropped when its R^2 is below the threshold in every ``(group,
view)`` cell; among qualifying factors the one with the smallest
maximum-over-cells R^2 is removed, never below ``min_factors``. A ``threshold``
of ``None`` disables pruning entirely (mofapy2's ``dropR2=None`` default). At
``G = 1`` the rule reduces to the per-view rule.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..utils import R2_EPS


def find_inactive_factors(R2: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask flagging factors below ``threshold`` in every cell.

    Accepts ``(K, M)`` or ``(K, G, M)`` R^2 arrays, collapsing all axes after
    the factor axis via ``reshape(K, -1)``.
    """
    R2 = np.asarray(R2)
    K = R2.shape[0]
    return np.asarray((R2.reshape(K, -1) < float(threshold)).all(axis=1))


def prune_inactive_factors(
    model,
    threshold: Optional[float],
    min_factors: int = 1,
) -> int:
    """Drop the single worst inactive factor if any qualify; return its index,
    or ``-1`` if none qualify, the model is at ``min_factors``, or pruning is
    disabled (``threshold is None``).

    Ties within ``R2_EPS`` are broken toward the lower index so the dropped
    factor is a deterministic function of the model state rather than of BLAS
    reduction order (needed for cross-platform reproducibility).
    """
    if threshold is None:
        return -1
    if model.K <= int(min_factors):
        return -1
    R2_kgm = model.variance_explained_per_factor_view_group()
    inactive = find_inactive_factors(R2_kgm, threshold)
    if not inactive.any():
        return -1
    worst_per_factor = R2_kgm.reshape(model.K, -1).max(axis=1)
    candidates = np.where(inactive)[0]
    cand_scores = worst_per_factor[candidates]
    min_score = float(cand_scores.min())
    tied = candidates[cand_scores <= min_score + R2_EPS]
    drop_idx = int(tied.min())
    keep = np.ones(model.K, dtype=bool)
    keep[drop_idx] = False
    model.prune(keep)
    return drop_idx


__all__ = [
    "find_inactive_factors",
    "prune_inactive_factors",
]
