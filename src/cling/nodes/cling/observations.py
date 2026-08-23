"""Observed data ``Y`` for a single view."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ...utils import EPS, log_eps
from ..base import VariationalNode


class Observations(VariationalNode):
    """Observed-data node for view ``m``.

    Stores the centered data, the NaN mask, per-(group, feature) observation
    counts, and a cached mean residual ``R_m = mask * (Y - Z W^T)`` used by the
    rank-1 inner-loop of :class:`Factors` and :class:`Loadings`. The residual
    is rebuilt by :meth:`refresh_residual_mean` (after wiring and after each
    prune) and maintained incrementally by :meth:`apply_rank1_delta`.
    """

    def __init__(
        self,
        data: np.ndarray,
        m: int,
        N: int,
        K: int,
        M: int,
        D: List[int],
        data_mean: Optional[np.ndarray] = None,
        G: int = 1,
        group_ix: Optional[np.ndarray] = None,
    ):
        self.m = int(m)
        self.N = int(N)
        self.K = int(K)
        self.M = int(M)
        self.D = list(D)

        self.data = np.asarray(data, dtype=float)
        self.mask = np.isfinite(self.data).astype(float)
        self.Y = np.nan_to_num(self.data, nan=0.0)
        self.data_mean: Optional[np.ndarray] = data_mean

        self.G = int(G)
        if group_ix is None:
            group_ix = np.zeros(self.N, dtype=np.int64)
        self.group_ix = np.asarray(group_ix, dtype=np.int64)

        Dm = self.D[self.m]
        # Per-(group, feature) observed-entry count; at G=1 the row equals
        # ``mask.sum(axis=0)``.
        self.obs_count_per_group = np.zeros((self.G, Dm), dtype=float)
        np.add.at(self.obs_count_per_group, self.group_ix, self.mask)
        self.obs_count = self.obs_count_per_group.sum(axis=0)

        self.residual_mean = np.zeros((self.N, Dm), dtype=float)

    def set_neighbours(self, loadings, tau) -> None:
        self.loadings = loadings
        self.tau = tau

    def update(self) -> None:
        return None

    # --- residual cache management ----------------------------------
    def refresh_residual_mean(self, Z: np.ndarray, W: np.ndarray) -> None:
        """Recompute ``residual_mean = mask * (Y - Z W^T)`` from scratch."""
        self.residual_mean = self.mask * (self.Y - Z @ W.T)

    def apply_rank1_delta(self, dz: np.ndarray, dw: np.ndarray) -> None:
        """Add ``mask * outer(dz, dw)`` to the residual cache after a single
        column of ``Z`` or ``W`` changed. For Z: ``dz = z_old - z_new``,
        ``dw = W[:, k]``; for W: ``dz = Z[:, k]``, ``dw = w_old - w_new``.
        """
        self.residual_mean += self.mask * np.outer(dz, dw)

    def elbo(self) -> float:
        """Group-aware Gaussian log-likelihood term. The ``E[log tau]`` weight
        uses the per-(group, feature) observation count; at ``G = 1`` the sums
        reduce to the plain per-feature expressions.
        """
        log2pi = log_eps(2 * np.pi, EPS)
        const_term = -0.5 * float(np.sum(self.obs_count_per_group)) * log2pi
        tau_log_term = 0.5 * float(
            np.sum(self.obs_count_per_group * self.tau.E_log_tau)
        )
        quad_term = float(np.sum(self.tau.E_tau * self.tau.residual_squared_half))
        return const_term + tau_log_term - quad_term

    def get_raw_and_mean(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the raw view and the feature means used to un-centre R^2.

        ``mu`` is ``(1, D_m)`` for global centering / no centering, or expanded
        to ``(N, D_m)`` via ``group_ix`` for per-group means.
        """
        if self.data_mean is None:
            Y_raw = self.data
            mu = np.nanmean(Y_raw, axis=0, keepdims=True)
        elif self.data_mean.ndim == 2:
            mu = self.data_mean[self.group_ix, :]
            Y_raw = self.data + mu
        else:
            mu = self.data_mean.reshape(1, -1)
            Y_raw = self.data + mu
        return Y_raw, mu


__all__ = ["Observations"]
