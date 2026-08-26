"""Column-wise local precision ``alpha`` (one parameter per factor)."""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.special import digamma, gammaln

from ...utils import EPS, MIN_GAMMA_SHAPE, log_eps
from ..base import VariationalNode


class AlphaColumn(VariationalNode):
    """q(alpha^m_k) = Ga(a, b) with prior ``alpha^m_k ~ Ga(a_alpha, b_alpha)``.

    A single precision is shared across all features ``d`` within the view; the
    ``E_alpha_full`` / ``E_log_alpha_full`` accessors broadcast to ``(D_m, K)``
    so the loadings update sees the same interface as for per-feature alpha.
    """

    def __init__(
        self,
        a_alpha: float,
        b_alpha: float,
        m: int,
        N: int,
        K: int,
        M: int,
        D: List[int],
    ):
        self.a_alpha = float(a_alpha)
        self.b_alpha = float(b_alpha)
        self.m = int(m)
        self.N = int(N)
        self.K = int(K)
        self.M = int(M)
        self.D = list(D)
        self.Dm = self.D[self.m]

        self.vi_a = np.full(self.K, self.a_alpha + 0.5 * self.Dm, dtype=float)
        self.vi_b = np.full(self.K, max(self.b_alpha, EPS), dtype=float)
        self._refresh_moments()

    # --- wiring -------------------------------------------------------
    def set_neighbours(self, loadings, delta) -> None:
        self.loadings = loadings
        self.delta = delta

    # --- moments ------------------------------------------------------
    def _refresh_moments(self) -> None:
        self.vi_b = np.maximum(self.vi_b, EPS)
        vi_a_safe = np.maximum(self.vi_a, MIN_GAMMA_SHAPE)
        self.E_alpha = self.vi_a / self.vi_b
        self.E_log_alpha = digamma(self.vi_a) - log_eps(self.vi_b, EPS)
        self.E_inv_alpha = self.vi_b / (vi_a_safe - 1.0)

    def E_alpha_full(self) -> np.ndarray:
        return np.broadcast_to(self.E_alpha[np.newaxis, :], (self.Dm, self.K))

    def E_log_alpha_full(self) -> np.ndarray:
        return np.broadcast_to(self.E_log_alpha[np.newaxis, :], (self.Dm, self.K))

    # --- update -------------------------------------------------------
    def update(self) -> None:
        E_gamma = self.delta.E_gamma                       # (K,)
        sum_EW2_per_k = self.loadings.E_w_squared.sum(axis=0)  # (K,)
        self.vi_a = np.full(self.K, self.a_alpha + 0.5 * self.Dm, dtype=float)
        self.vi_b = self.b_alpha + 0.5 * (E_gamma * sum_EW2_per_k)
        self._refresh_moments()

    # --- ELBO ---------------------------------------------------------
    def elbo(self) -> float:
        a0 = self.a_alpha
        b0 = self.b_alpha
        logp = self.K * (a0 * np.log(b0 + EPS) - gammaln(a0)) + (a0 - 1.0) * float(
            np.sum(self.E_log_alpha)
        ) - b0 * float(np.sum(self.E_alpha))
        logq = float(
            np.sum(
                self.vi_a * log_eps(self.vi_b, EPS)
                - gammaln(self.vi_a)
                + (self.vi_a - 1.0) * self.E_log_alpha
                - self.vi_b * self.E_alpha
            )
        )
        return float(logp - logq)

    # --- pruning ------------------------------------------------------
    def prune(self, active_mask: np.ndarray) -> None:
        self.K = int(active_mask.sum())
        self.vi_a = self.vi_a[active_mask]
        self.vi_b = self.vi_b[active_mask]
        self._refresh_moments()


__all__ = ["AlphaColumn"]
