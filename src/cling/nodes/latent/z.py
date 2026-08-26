"""Latent factor matrix ``Z``."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np

from ...utils import EPS, log_eps
from ..base import VariationalNode

if TYPE_CHECKING:
    from .eta import Eta


class Factors(VariationalNode):
    """q(Z) = N(mu, diag(sigma2)), shape ``(N, K)``.

    The prior on ``z_{n, k}`` is ``N(0, 1)`` when :class:`Eta` is not wired
    (single-group default), or ``N(0, <eta_{g(n), k}>^{-1})`` when the
    group-factor ARD precision is wired in.
    """

    def __init__(
        self,
        N: int,
        K: int,
        M: int,
        D: List[int],
        mu_init: np.ndarray | None = None,
        var_init: np.ndarray | None = None,
    ):
        self.N = int(N)
        self.K = int(K)
        self.M = int(M)
        self.D = list(D)

        if mu_init is None:
            mu_init = np.zeros((self.N, self.K))
        if var_init is None:
            var_init = np.ones((self.N, self.K))

        self.vi_mu = np.asarray(mu_init, dtype=float).copy()
        self.vi_var = np.maximum(np.asarray(var_init, dtype=float).copy(), EPS)

        self.eta: Optional[Eta] = None

        # Sweep-level caches filled by the model before each Z sweep:
        # ``_mask_tau[m]`` of shape (N, D_m) per view, and ``_eta_expanded``
        # of shape (N, K) when eta is wired. ``update`` fills them on the fly
        # for direct callers (tests).
        self._mask_tau: List[np.ndarray] = []
        self._eta_expanded: Optional[np.ndarray] = None

        self._refresh_moments()

    # --- moments ------------------------------------------------------
    def _refresh_moments(self) -> None:
        self.vi_var = np.maximum(self.vi_var, EPS)
        self.E_z = self.vi_mu
        self.E_z_squared = self.vi_var + self.vi_mu ** 2

    # --- wiring -------------------------------------------------------
    def set_neighbours(self, observations, loadings, tau, eta=None) -> None:
        self.observations = observations
        self.loadings = loadings
        self.tau = tau
        self.eta = eta

    # --- updates ------------------------------------------------------
    def update_k(self, k: int) -> None:
        """Coordinate-ascent update for column ``k`` of ``Z`` using the cached
        per-view residual ``R_m = mask * (Y - Z W^T)`` and the sweep-level
        ``_mask_tau`` / ``_eta_expanded`` caches, then a rank-1 residual delta.
        The prior precision is ``_eta_expanded[:, k]`` when eta is wired, else
        the scalar ``1.0`` (N(0, 1) prior on Z).
        """
        z_old_k = self.E_z[:, k].copy()
        mean_num_n: np.ndarray = np.zeros(self.N)
        data_prec_n: np.ndarray = np.zeros(self.N)

        mask_tau_list = self._mask_tau if self._mask_tau else None

        for m in range(self.M):
            mask = self.observations[m].mask
            if mask_tau_list is not None:
                mask_tau = mask_tau_list[m]
            else:
                mask_tau = mask * self.tau[m].E_tau_expanded()
            Wm = self.loadings[m].E_w
            Ew2_k = self.loadings[m].E_w_squared[:, k]
            R_m = self.observations[m].residual_mean

            data_prec_n += (mask_tau * Ew2_k[None, :]).sum(axis=1)

            partial_masked = R_m + mask * np.outer(z_old_k, Wm[:, k])
            mean_num_n += (mask_tau * partial_masked) @ Wm[:, k]

        prior_prec_n: np.ndarray
        if self.eta is not None:
            if self._eta_expanded is not None:
                prior_prec_n = self._eta_expanded[:, k]
            else:
                prior_prec_n = self.eta.E_eta[self.eta.group_ix, k]
        else:
            prior_prec_n = np.ones(self.N)

        vi_var_k: np.ndarray = 1.0 / np.maximum(data_prec_n + prior_prec_n, EPS)
        vi_mu_k: np.ndarray = mean_num_n * vi_var_k

        self.vi_var[:, k] = vi_var_k
        self.vi_mu[:, k] = vi_mu_k
        self._refresh_moments()

        dz = z_old_k - vi_mu_k
        for m in range(self.M):
            Wm = self.loadings[m].E_w
            self.observations[m].apply_rank1_delta(dz, Wm[:, k])

    def update(self) -> None:
        """Sweep all columns once (used by tests and ad-hoc scripts)."""
        self._mask_tau = [
            obs.mask * tau.E_tau_expanded()
            for obs, tau in zip(self.observations, self.tau)
        ]
        self._eta_expanded = (
            self.eta.E_eta_expanded() if self.eta is not None else None
        )
        for k in range(self.K):
            self.update_k(k)

    # --- ELBO ---------------------------------------------------------
    def elbo(self) -> float:
        entropy = 0.5 * float(np.sum(log_eps(self.vi_var, EPS)))
        base = 0.5 * self.N * self.K + entropy

        if self.eta is None:
            return base - 0.5 * float(np.sum(self.E_z_squared))

        E_log_eta_exp = self.eta.E_log_eta_expanded()                # (N, K)
        E_eta_exp = self.eta.E_eta_expanded()                         # (N, K)
        return (
            base
            + 0.5 * float(np.sum(E_log_eta_exp))
            - 0.5 * float(np.sum(E_eta_exp * self.E_z_squared))
        )

    # --- pruning ------------------------------------------------------
    def prune(self, active_mask: np.ndarray) -> None:
        self.K = int(active_mask.sum())
        self.vi_mu = self.vi_mu[:, active_mask]
        self.vi_var = self.vi_var[:, active_mask]
        self._refresh_moments()


__all__ = ["Factors"]
