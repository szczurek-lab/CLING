"""Per-feature noise precision ``tau`` for a single view, group-aware."""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.special import digamma, gammaln

from ...utils import EPS, log_eps
from ..base import VariationalNode


class Tau(VariationalNode):
    """q(tau^{g,m}_d) = Ga(a, b) with prior Ga(a0, b0), stored as ``(G, D_m)``.

    :meth:`E_tau_expanded` broadcasts the posterior mean to ``(N, D_m)`` via
    ``group_ix`` so the Z / W updates treat tau as per-sample without knowing
    about groups. With ``G = 1`` this recovers per-feature tau exactly.
    """

    def __init__(
        self,
        a0: float,
        b0: float,
        m: int,
        N: int,
        K: int,
        M: int,
        D: List[int],
        G: int,
        group_ix: np.ndarray,
        N_per_group: np.ndarray,
    ):
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.m = int(m)
        self.N = int(N)
        self.K = int(K)
        self.M = int(M)
        self.D = list(D)

        self.G = int(G)
        self.group_ix = np.asarray(group_ix, dtype=np.int64)
        self.N_per_group = np.asarray(N_per_group, dtype=np.int64)

        Dm = self.D[self.m]
        # Placeholder moments; set_neighbours overwrites with the actual
        # per-(group, feature) observed counts.
        self.vi_a = self.a0 + np.broadcast_to(
            self.N_per_group[:, None].astype(float), (self.G, Dm)
        ).copy() / 2.0
        self.vi_b = self.vi_a.copy()
        self.residual_squared_half: np.ndarray = np.zeros((self.G, Dm))

    # --- wiring -------------------------------------------------------
    def set_neighbours(self, observations, loadings, factors) -> None:
        self.observations = observations
        self.loadings = loadings
        self.factors = factors

        self.vi_a = self.a0 + self.observations.obs_count_per_group / 2.0
        self._initialise_constants()
        self._refresh_moments()

    def _initialise_constants(self) -> None:
        Dm = self.D[self.m]
        n_cells = self.G * Dm
        self.log_gamma_a0 = gammaln(self.a0)
        self.log_gamma_vi_a = gammaln(self.vi_a)                    # (G, D_m)
        self.digamma_vi_a = digamma(self.vi_a)                      # (G, D_m)
        self.kl_const = (
            -n_cells * self.log_gamma_a0
            + n_cells * self.a0 * log_eps(self.b0, EPS)
        )
        self.entropy_const = (
            float(np.sum(self.vi_a))
            + float(np.sum(self.log_gamma_vi_a))
            + float(np.sum((1.0 - self.vi_a) * self.digamma_vi_a))
        )

    # --- moments ------------------------------------------------------
    def _refresh_moments(self) -> None:
        self.vi_b = np.maximum(self.vi_b, EPS)
        self.E_tau = self.vi_a / self.vi_b                          # (G, D_m)
        self.E_log_tau = -log_eps(self.vi_b, EPS) + self.digamma_vi_a

    def E_tau_expanded(self) -> np.ndarray:
        """Return ``E[tau]`` broadcast to ``(N, D_m)`` via ``group_ix``."""
        return self.E_tau[self.group_ix]

    def E_log_tau_expanded(self) -> np.ndarray:
        """Return ``E[log tau]`` broadcast to ``(N, D_m)`` via ``group_ix``."""
        return self.E_log_tau[self.group_ix]

    def refresh_residual(self) -> None:
        """Recompute the per-(group, feature) residual sum
        ``0.5 * sum_n mask * E[(Y - sum_k W Z)^2]``, scattered by group."""
        Y = self.observations.Y
        mask = self.observations.mask
        E_y_hat = self.loadings.E_y_hat
        E_y_hat_sq = self.loadings.E_y_hat_squared
        per_sample = 0.5 * mask * (
            Y * Y - 2.0 * Y * E_y_hat + E_y_hat_sq
        )                                                           # (N, D_m)

        Dm = self.D[self.m]
        residual = np.zeros((self.G, Dm))
        np.add.at(residual, self.group_ix, per_sample)
        self.residual_squared_half = residual

    # --- update -------------------------------------------------------
    def update(self) -> None:
        self.refresh_residual()
        self.vi_b = self.b0 + self.residual_squared_half
        self._refresh_moments()

    # --- ELBO ---------------------------------------------------------
    def elbo(self) -> float:
        kl = (
            self.kl_const
            + (self.a0 - 1.0) * float(np.sum(self.E_log_tau))
            - self.b0 * float(np.sum(self.E_tau))
        )
        entropy = self.entropy_const - float(np.sum(log_eps(self.vi_b, EPS)))
        return kl + entropy


__all__ = ["Tau"]
