"""Factor-specific local scale ``phi^m_{d,k}`` for the main CLING model.

Each ``alpha^m_{d,k}`` has its own auxiliary scale ``phi^m_{d,k}`` through the
rate: ``alpha^m_{d,k} ~ Ga(a_alpha, b_alpha * phi^m_{d,k})``. Marginalising
``phi`` gives ``alpha^m_{d,k} ~ BetaPrime`` iid across the k-plate.
"""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.special import digamma, gammaln

from ...utils import EPS, MIN_GAMMA_SHAPE, log_eps
from ..base import VariationalNode


class PhiDK(VariationalNode):
    """q(phi^m_{d,k}) = Ga(a, b) with prior ``phi^m_{d,k} ~ Ga(a_phi, b_phi)``.

    Posterior shape is ``a_phi + a_alpha`` and rate ``b_phi + b_alpha
    <alpha^m_{d,k}>``. The variational parameters carry a ``K`` axis, so this
    node is pruned alongside ``alpha`` and ``delta``.
    """

    def __init__(
        self,
        a_phi: float,
        b_phi: float,
        b_alpha: float,
        m: int,
        N: int,
        K: int,
        M: int,
        D: List[int],
    ):
        self.a_phi = float(a_phi)
        self.b_phi = float(b_phi)
        self.b_alpha = float(b_alpha)
        self.m = int(m)
        self.N = int(N)
        self.K = int(K)
        self.M = int(M)
        self.D = list(D)
        Dm = self.D[self.m]

        self.vi_a_phi = np.full((Dm, K), self.a_phi, dtype=float)
        self.vi_b_phi = np.full((Dm, K), self.b_phi, dtype=float)
        self._refresh_moments()

    def set_neighbours(self, alpha) -> None:
        self.alpha = alpha

    # --- moments ------------------------------------------------------
    def _refresh_moments(self) -> None:
        self.vi_b_phi = np.maximum(self.vi_b_phi, EPS)
        vi_a_safe = np.maximum(self.vi_a_phi, MIN_GAMMA_SHAPE)
        self.E_phi = self.vi_a_phi / self.vi_b_phi
        self.E_log_phi = digamma(self.vi_a_phi) - log_eps(self.vi_b_phi, EPS)
        self.E_inv_phi = self.vi_b_phi / (vi_a_safe - 1.0)

    # --- update -------------------------------------------------------
    def update(self) -> None:
        a_alpha = self.alpha.a_alpha
        self.vi_a_phi = np.full(
            self.vi_a_phi.shape, self.a_phi + a_alpha, dtype=float
        )
        self.vi_b_phi = self.b_phi + self.b_alpha * self.alpha.E_alpha
        self._refresh_moments()

    # --- ELBO ---------------------------------------------------------
    def elbo(self) -> float:
        a0, b0 = self.a_phi, self.b_phi
        a, b = self.vi_a_phi, self.vi_b_phi
        E_log_phi = self.E_log_phi
        E_phi = self.E_phi
        logp = (
            a0 * np.log(b0 + EPS)
            + (a0 - 1.0) * E_log_phi
            - b0 * E_phi
            - gammaln(a0)
        )
        logq = (
            a * np.log(b + EPS)
            + (a - 1.0) * E_log_phi
            - b * E_phi
            - gammaln(a)
        )
        return float(np.sum(logp - logq))

    # --- pruning ------------------------------------------------------
    def prune(self, active_mask: np.ndarray) -> None:
        self.K = int(active_mask.sum())
        self.vi_a_phi = self.vi_a_phi[:, active_mask]
        self.vi_b_phi = self.vi_b_phi[:, active_mask]
        self._refresh_moments()


__all__ = ["PhiDK"]
