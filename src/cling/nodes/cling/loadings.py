"""View-specific loading matrix ``W``."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ...utils import EPS, log_eps
from ..base import VariationalNode


class Loadings(VariationalNode):
    """q(W^m) = N(mu, diag(sigma2)) with prior
    ``W^m_{d,k} ~ N(0, 1 / (alpha^m_{d,k} * gamma^m_k))``.

    Per-feature and column-wise ``alpha`` are both supported: the latter
    exposes its moments broadcast to ``(D_m, K)`` so the update is identical.
    """

    def __init__(
        self,
        m: int,
        N: int,
        K: int,
        M: int,
        D: List[int],
        mu_init: np.ndarray,
        var_init: np.ndarray,
    ):
        self.m = int(m)
        self.N = int(N)
        self.K = int(K)
        self.M = int(M)
        self.D = list(D)

        self.vi_mu = np.asarray(mu_init, dtype=float).copy()
        self.vi_var = np.maximum(np.asarray(var_init, dtype=float).copy(), EPS)

        self._refresh_moments()

        Dm = self.D[self.m]
        self.E_y_hat = np.zeros((self.N, Dm))
        self.E_y_hat_squared = np.zeros((self.N, Dm))

        # Sweep-level cache ``mask * tau.E_tau_expanded()`` of shape (N, D_m),
        # filled by the model before each W sweep; ``update`` fills it on the
        # fly for direct callers (tests).
        self._mask_tau: Optional[np.ndarray] = None

    # --- moments ------------------------------------------------------
    def _refresh_moments(self) -> None:
        self.vi_var = np.maximum(self.vi_var, EPS)
        self.E_w = self.vi_mu
        self.E_w_squared = self.vi_var + self.vi_mu ** 2

    def refresh_y_hat_moments(self) -> None:
        """Recompute ``E[Y_hat]`` and ``E[Y_hat^2]`` where ``Y_hat = W Z^T``.

        Under mean field ``Var(Y_hat_{n,d}) = sum_k E[W^2] E[Z^2] -
        E[W]^2 E[Z]^2``; computed once per sweep (no clean rank-1 form).
        """
        Z = self.factors.E_z
        Z_sq = self.factors.E_z_squared

        term_tmp = self.E_w @ Z.T
        self.E_y_hat = term_tmp.T

        first_term = term_tmp ** 2
        second_term = self.E_w_squared @ Z_sq.T
        third_term = (self.E_w ** 2) @ (Z.T ** 2)
        self.E_y_hat_squared = (first_term + second_term - third_term).T

    # --- wiring -------------------------------------------------------
    def set_neighbours(self, alpha, factors, observations, tau, delta) -> None:
        self.alpha = alpha
        self.factors = factors
        self.observations = observations
        self.tau = tau
        self.delta = delta
        self._refresh_moments()
        self.refresh_y_hat_moments()

    # --- updates ------------------------------------------------------
    def update_k(self, k: int) -> None:
        """Coordinate-ascent update for column ``k`` of ``W^m``, reading the
        cached residual and the sweep-level ``_mask_tau`` (constant across the
        K columns of one sweep), then applying a rank-1 delta to the residual.
        """
        E_alpha_k = self.alpha.E_alpha_full()[:, k]
        E_gamma_k = self.delta.E_gamma[k]
        Z = self.factors.E_z
        Zk = Z[:, k]
        mask = self.observations.mask
        R_m = self.observations.residual_mean
        w_old_k = self.E_w[:, k].copy()

        if self._mask_tau is not None:
            mask_tau = self._mask_tau
        else:
            mask_tau = mask * self.tau.E_tau_expanded()

        partial_masked = R_m + mask * np.outer(Zk, w_old_k)

        E_z2_k = self.factors.E_z_squared[:, k]
        data_prec = (mask_tau * E_z2_k[:, None]).sum(axis=0)         # (D_m,)
        prior_prec = E_alpha_k * E_gamma_k                            # (D_m,)
        full_prec = np.maximum(data_prec + prior_prec, EPS)           # (D_m,)

        numerator = Zk @ (mask_tau * partial_masked)                  # (D_m,)

        w_new_k = numerator / full_prec
        self.vi_mu[:, k] = w_new_k
        self.vi_var[:, k] = np.maximum(1.0 / full_prec, EPS)
        self._refresh_moments()

        dw = w_old_k - w_new_k
        self.observations.apply_rank1_delta(Zk, dw)

    def update(self) -> None:
        """Sweep every column once (used by tests and ad-hoc scripts)."""
        self._mask_tau = self.observations.mask * self.tau.E_tau_expanded()
        for k in range(self.K):
            self.update_k(k)

    # --- ELBO ---------------------------------------------------------
    def elbo(self) -> float:
        Dm = self.D[self.m]

        E_log_alpha = self.alpha.E_log_alpha_full()
        E_alpha = self.alpha.E_alpha_full()
        E_log_gamma_k = self.delta.E_log_gamma
        E_gamma_k = self.delta.E_gamma

        E_log_alpha_gamma = E_log_alpha + E_log_gamma_k[np.newaxis, :]
        E_alpha_gamma = E_alpha * E_gamma_k[np.newaxis, :]

        logp = 0.5 * float(np.sum(E_log_alpha_gamma)) - 0.5 * float(
            np.sum(E_alpha_gamma * self.E_w_squared)
        )
        entropy = 0.5 * Dm * self.K + 0.5 * float(np.sum(log_eps(self.vi_var, EPS)))
        return logp + entropy

    # --- pruning ------------------------------------------------------
    def prune(self, active_mask: np.ndarray) -> None:
        self.K = int(active_mask.sum())
        self.vi_mu = self.vi_mu[:, active_mask]
        self.vi_var = self.vi_var[:, active_mask]
        self._refresh_moments()


__all__ = ["Loadings"]
