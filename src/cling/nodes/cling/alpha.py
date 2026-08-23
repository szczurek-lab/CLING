"""Per-feature local precision ``alpha`` for the loadings ``W``."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np
from scipy.special import digamma, gammaln

from ...utils import EPS, MIN_GAMMA_SHAPE, log_eps
from ..base import VariationalNode

if TYPE_CHECKING:
    from .phi_dk import PhiDK


class Alpha(VariationalNode):
    """q(alpha^m_{d,k}) = Ga(a, b), per ``(d, k)``.

    Two prior modes, detected at runtime from the ``phi`` neighbour:

    * ``phi is None``: ``alpha^m_{d,k} ~ Ga(a_alpha, b_alpha)`` (CLING-MGP).
    * ``phi`` (:class:`PhiDK`, 2-D moments): ``alpha^m_{d,k} ~
      Ga(a_alpha, b_alpha * phi^m_{d,k})`` (main CLING Gamma-Gamma hierarchy).
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

        self.vi_a = np.full((self.Dm, self.K), self.a_alpha + 0.5, dtype=float)
        self.vi_b = np.full((self.Dm, self.K), max(self.b_alpha, EPS), dtype=float)

        self._phi: Optional[PhiDK] = None  # set by set_neighbours
        self._refresh_moments()

    # --- wiring -------------------------------------------------------
    def set_neighbours(self, loadings, delta, phi: Optional[PhiDK] = None) -> None:
        self.loadings = loadings
        self.delta = delta
        self._phi = phi

    # --- moments ------------------------------------------------------
    def _refresh_moments(self) -> None:
        self.vi_b = np.maximum(self.vi_b, EPS)
        vi_a_safe = np.maximum(self.vi_a, MIN_GAMMA_SHAPE)
        self.E_alpha = self.vi_a / self.vi_b
        self.E_log_alpha = digamma(self.vi_a) - log_eps(self.vi_b, EPS)
        self.E_inv_alpha = self.vi_b / (vi_a_safe - 1.0)

    def E_alpha_full(self) -> np.ndarray:
        """Return ``E[alpha]`` with shape ``(D_m, K)``."""
        return self.E_alpha

    def E_log_alpha_full(self) -> np.ndarray:
        """Return ``E[log alpha]`` with shape ``(D_m, K)``."""
        return self.E_log_alpha

    # --- update -------------------------------------------------------
    def _rate_offset(self) -> np.ndarray:
        """Constant additive term on the rate: ``(D_m, 1)`` when ``phi`` is
        absent (broadcasts over k), else the ``(D_m, K)`` ``b_alpha * phi``."""
        if self._phi is None:
            return np.full((self.Dm, 1), self.b_alpha)
        return self.b_alpha * self._phi.E_phi

    def update(self) -> None:
        E_gamma = self.delta.E_gamma[np.newaxis, :]
        E_W2 = self.loadings.E_w_squared
        self.vi_a = np.full((self.Dm, self.K), self.a_alpha + 0.5, dtype=float)
        self.vi_b = self._rate_offset() + 0.5 * (E_gamma * E_W2)
        self._refresh_moments()

    # --- ELBO ---------------------------------------------------------
    def elbo(self) -> float:
        a0 = self.a_alpha
        b0 = self.b_alpha
        E_log_alpha = self.E_log_alpha
        E_alpha = self.E_alpha

        if self._phi is None:
            logp = (
                a0 * np.log(b0 + EPS)
                + (a0 - 1.0) * E_log_alpha
                - b0 * E_alpha
                - gammaln(a0)
            )
        else:
            E_log_phi = self._phi.E_log_phi
            E_phi = self._phi.E_phi
            logp = (
                a0 * (np.log(b0 + EPS) + E_log_phi)
                + (a0 - 1.0) * E_log_alpha
                - (b0 * E_phi) * E_alpha
                - gammaln(a0)
            )

        logq = (
            self.vi_a * log_eps(self.vi_b, EPS)
            + (self.vi_a - 1.0) * E_log_alpha
            - self.vi_b * E_alpha
            - gammaln(self.vi_a)
        )
        return float(np.sum(logp - logq))

    # --- pruning ------------------------------------------------------
    def prune(self, active_mask: np.ndarray) -> None:
        self.K = int(active_mask.sum())
        self.vi_a = self.vi_a[:, active_mask]
        self.vi_b = self.vi_b[:, active_mask]
        self._refresh_moments()


__all__ = ["Alpha"]
