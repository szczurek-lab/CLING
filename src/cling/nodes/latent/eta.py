"""Group-factor ARD precision ``eta_{g, k}`` on the latent factors Z.

q(eta_{g, k}) = Gamma(vi_a, vi_b) of shape (G, K) with prior Gamma(a_eta,
b_eta). CAVI update: ``vi_a = a_eta + N_g/2``, ``vi_b = b_eta + 0.5 *
sum_{n in g} <Z_{n,k}^2>``. :meth:`E_eta_expanded` broadcasts to (N, K) via
``group_ix`` so :class:`Factors` uses it as a per-sample prior precision.

``vi_a`` is materialised at full (G, K) shape (though constant across k) so
that axis-1 pruning and the (G, K)-cell ELBO sum are correct. Initialised with
``E[eta] = 1`` (the effective N(0, 1) prior); the first ``update`` - scheduled
before the first Z sweep - sets it data-consistently, which avoids the
``E[eta] ~ N_g * 1e14`` collapse the naive ``b_eta = 1e-14`` init would cause.
"""

from __future__ import annotations

import numpy as np
from scipy.special import digamma, gammaln

from ...utils import EPS, log_eps
from ..base import VariationalNode


class Eta(VariationalNode):
    """q(eta_{g, k}) = Ga(a, b), shape ``(G, K)``."""

    def __init__(
        self,
        a_eta: float,
        b_eta: float,
        N: int,
        K: int,
        G: int,
        group_ix: np.ndarray,
        N_per_group: np.ndarray,
    ):
        self.a_eta = float(a_eta)
        self.b_eta = float(b_eta)
        self.N = int(N)
        self.K = int(K)
        self.G = int(G)
        self.group_ix = np.asarray(group_ix, dtype=np.int64)
        self.N_per_group = np.asarray(N_per_group, dtype=np.int64)

        self.vi_a = np.ones((self.G, self.K))
        self.vi_b = np.ones((self.G, self.K))
        self._refresh_moments()

    # --- wiring -------------------------------------------------------
    def set_neighbours(self, factors) -> None:
        self.factors = factors

    # --- moments ------------------------------------------------------
    def _refresh_moments(self) -> None:
        self.vi_b = np.maximum(self.vi_b, EPS)
        self.E_eta = self.vi_a / self.vi_b                          # (G, K)
        self.E_log_eta = digamma(self.vi_a) - log_eps(self.vi_b, EPS)

    def E_eta_expanded(self) -> np.ndarray:
        """Return ``E[eta]`` broadcast to ``(N, K)`` via ``group_ix``."""
        return self.E_eta[self.group_ix]

    def E_log_eta_expanded(self) -> np.ndarray:
        """Return ``E[log eta]`` broadcast to ``(N, K)`` via ``group_ix``."""
        return self.E_log_eta[self.group_ix]

    # --- update -------------------------------------------------------
    def update(self) -> None:
        E_z2 = self.factors.E_z_squared                             # (N, K)
        S_gk = np.zeros((self.G, self.K))
        np.add.at(S_gk, self.group_ix, E_z2)                         # (G, K)

        vi_a_per_group = self.a_eta + self.N_per_group.astype(float) / 2.0
        self.vi_a = np.broadcast_to(
            vi_a_per_group[:, None], (self.G, self.K)
        ).copy()
        self.vi_b = self.b_eta + 0.5 * S_gk
        self._refresh_moments()

    # --- ELBO ---------------------------------------------------------
    def elbo(self) -> float:
        n_cells = self.G * self.K
        log_gamma_a0 = gammaln(self.a_eta)
        log_gamma_vi_a = gammaln(self.vi_a)
        digamma_vi_a = digamma(self.vi_a)

        kl = (
            -n_cells * log_gamma_a0
            + n_cells * self.a_eta * log_eps(self.b_eta, EPS)
            + (self.a_eta - 1.0) * float(np.sum(self.E_log_eta))
            - self.b_eta * float(np.sum(self.E_eta))
        )
        entropy = (
            float(np.sum(self.vi_a))
            + float(np.sum(log_gamma_vi_a))
            + float(np.sum((1.0 - self.vi_a) * digamma_vi_a))
            - float(np.sum(log_eps(self.vi_b, EPS)))
        )
        return kl + entropy

    # --- pruning ------------------------------------------------------
    def prune(self, active_mask: np.ndarray) -> None:
        self.K = int(active_mask.sum())
        self.vi_a = self.vi_a[:, active_mask]
        self.vi_b = self.vi_b[:, active_mask]
        self._refresh_moments()


__all__ = ["Eta"]
