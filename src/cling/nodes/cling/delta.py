"""Multiplicative Gamma Process (MGP) increment ``delta`` for a single view.

Cumulative shrinkage is ``gamma^m_k = prod_{j<=k} delta^m_j``. The cumulative
product is maintained in log space (``exp(cumsum(log ...))``) rather than
``np.cumprod``: the two are algebraically equivalent, but log space avoids the
underflow-to-zero that a long chain of ``E[delta] < 1`` factors would cause,
which would make the ``E_gamma / E_delta`` ratio in the update undefined.
"""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.special import digamma, gammaln

from ...utils import EPS, MIN_GAMMA_SHAPE, log_eps
from ..base import VariationalNode


class Delta(VariationalNode):
    """q(delta^m_k) = Ga(a, b); the first column uses ``(a1, b1)``, the rest
    ``(a2, b2)``. Exposes the cumulative product ``gamma^m_k`` via
    :attr:`E_gamma`, :attr:`E_inv_gamma`, :attr:`E_log_gamma`.
    """

    def __init__(
        self,
        a1: float,
        b1: float,
        a2: float,
        b2: float,
        m: int,
        N: int,
        K: int,
        M: int,
        D: List[int],
    ):
        self.a1, self.b1 = float(a1), float(b1)
        self.a2, self.b2 = float(a2), float(b2)
        self.m = int(m)
        self.N = int(N)
        self.K = int(K)
        self.M = int(M)
        self.D = list(D)

        self.vi_a = np.full(self.K, self.a2, dtype=float)
        self.vi_a[0] = self.a1
        self.vi_b = np.full(self.K, self.b2, dtype=float)
        self.vi_b[0] = self.b1

        self._init_buffers()
        self._refresh_moments()

    def _init_buffers(self) -> None:
        self.E_delta = np.ones(self.K)
        self.E_log_delta = np.zeros(self.K)
        self.E_gamma = np.ones(self.K)
        self.E_log_gamma = np.zeros(self.K)
        self.E_inv_delta = np.ones(self.K)
        self.E_inv_gamma = np.ones(self.K)

    # --- wiring -------------------------------------------------------
    def set_neighbours(self, loadings, alpha) -> None:
        self.loadings = loadings
        self.alpha = alpha

    # --- moments ------------------------------------------------------
    def _refresh_moments(self) -> None:
        self.vi_b = np.maximum(self.vi_b, EPS)
        vi_a_safe = np.maximum(self.vi_a, MIN_GAMMA_SHAPE)

        self.E_delta = self.vi_a / self.vi_b
        self.E_log_delta = digamma(self.vi_a) - log_eps(self.vi_b, EPS)
        self.E_inv_delta = self.vi_b / (vi_a_safe - 1.0)

        log_E_delta = np.log(np.maximum(self.E_delta, EPS))
        log_E_inv_delta = np.log(np.maximum(self.E_inv_delta, EPS))
        self.E_gamma = np.exp(np.cumsum(log_E_delta))
        self.E_inv_gamma = np.exp(np.cumsum(log_E_inv_delta))
        self.E_log_gamma = np.cumsum(self.E_log_delta)

    # --- update -------------------------------------------------------
    def _ak_bk(self, k: int) -> tuple[float, float]:
        return (self.a1, self.b1) if k == 0 else (self.a2, self.b2)

    def update(self) -> None:
        """Sequential CAVI sweep over columns ``k``. The rate is
        ``b_k + 0.5 * sum_{j>=k} (E[gamma_j]/E[delta_k]) * S_j`` with
        ``S_j = sum_d E[alpha_{d,j}] E[W^2_{d,j}]``; the gamma/delta ratio is
        formed in log space. The log cache is seeded from the *cached*
        ``E_delta`` (not the freshly overwritten ``vi_a/vi_b``) to match the
        Jacobi-style trajectory on the first call.
        """
        Dm = self.D[self.m]
        E_W_sq_all = self.loadings.E_w_squared            # (D_m, K)
        E_alpha_all = self.alpha.E_alpha_full()           # (D_m, K)

        S = np.sum(E_alpha_all * E_W_sq_all, axis=0)      # (K,)

        a_prior = np.where(np.arange(self.K) == 0, self.a1, self.a2)
        b_prior = np.where(np.arange(self.K) == 0, self.b1, self.b2)
        self.vi_a = a_prior + (Dm * (self.K - np.arange(self.K))) / 2.0

        log_E_delta = np.log(np.maximum(self.E_delta, EPS))
        log_E_gamma = np.cumsum(log_E_delta)

        for k in range(self.K):
            log_ratio_tail = log_E_gamma[k:] - log_E_delta[k]
            gamma_over_delta_tail = np.exp(log_ratio_tail)

            self.vi_b[k] = b_prior[k] + 0.5 * float(
                np.sum(gamma_over_delta_tail * S[k:])
            )
            self.vi_b[k] = max(self.vi_b[k], EPS)

            new_log_E_delta_k = np.log(self.vi_a[k]) - np.log(self.vi_b[k])
            shift = new_log_E_delta_k - log_E_delta[k]
            log_E_delta[k] = new_log_E_delta_k
            log_E_gamma[k:] += shift

        self._refresh_moments()

    # --- ELBO ---------------------------------------------------------
    def elbo(self) -> float:
        prior = 0.0
        entropy = 0.0
        for k in range(self.K):
            a_k, b_k = self._ak_bk(k)
            prior += (
                (a_k - 1.0) * self.E_log_delta[k]
                - b_k * self.E_delta[k]
                + a_k * np.log(b_k + EPS)
                - gammaln(a_k)
            )
            entropy += (
                self.vi_a[k] * log_eps(self.vi_b[k], EPS)
                + (self.vi_a[k] - 1.0) * self.E_log_delta[k]
                - self.vi_b[k] * self.E_delta[k]
                - gammaln(self.vi_a[k])
            )
        return float(prior - entropy)

    # --- pruning ------------------------------------------------------
    def prune(self, active_mask: np.ndarray) -> None:
        self.K = int(active_mask.sum())
        self.vi_a = self.vi_a[active_mask]
        self.vi_b = self.vi_b[active_mask]
        self._init_buffers()
        self._refresh_moments()


__all__ = ["Delta"]
