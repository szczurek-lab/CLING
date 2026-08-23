"""Shared base class for the CLING model variants.

The three variants (CLING, CLING-MGP, CLING-ARD) share the same Z / W / Y / Tau
update machinery, factor-pruning logic, and variance-explained diagnostics.
They differ only in how the local precision and shrinkage nodes (alpha, phi,
delta) are structured; each subclass overrides ``update_step``,
``compute_elbo``, and ``_prune_factor_nodes``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .config import CLINGHyperparameters
from .data import MultiviewDataset
from .nodes import Eta, Factors, Loadings, Observations, Tau
from .utils import R2_EPS


class _CLINGModelBase:
    """Common machinery for CLING model variants."""

    factors: Factors
    observations: List[Observations]
    loadings: List[Loadings]
    tau: List[Tau]
    eta: Optional[Eta]

    def __init__(
        self,
        dataset: MultiviewDataset,
        K: int,
        hp: CLINGHyperparameters,
        factors: Factors,
        observations: List[Observations],
        loadings: List[Loadings],
        tau: List[Tau],
        eta: Optional[Eta] = None,
    ):
        self.dataset = dataset
        self.K = int(K)
        self.hp = hp
        self.factors = factors
        self.observations = observations
        self.loadings = loadings
        self.tau = tau
        self.eta = eta

        # Sweep-level caches; populated by :meth:`_cache_sweep_arrays`.
        self._mask_tau: List[np.ndarray] = []
        self._eta_expanded: Optional[np.ndarray] = None

    # --- size properties ----------------------------------------------
    @property
    def N(self) -> int:
        return self.dataset.N

    @property
    def M(self) -> int:
        return self.dataset.M

    @property
    def D(self) -> List[int]:
        return self.dataset.D

    @property
    def G(self) -> int:
        return self.dataset.G

    # --- subclass hooks -----------------------------------------------
    def _wire(self) -> None:
        """Connect every node to its Markov blanket. Subclasses override."""
        raise NotImplementedError

    def update_step(self) -> None:
        """Run one CAVI sweep. Subclasses override."""
        raise NotImplementedError

    def compute_elbo(self) -> float:
        """Return the ELBO under the current variational parameters."""
        raise NotImplementedError

    def _prune_factor_nodes(self, active_mask: np.ndarray) -> None:
        """Prune variant-specific factor-indexed nodes (alpha, phi, delta)."""
        raise NotImplementedError

    # --- eta helper ---------------------------------------------------
    def _sweep_eta(self) -> None:
        """Update q(eta) when wired; must run before :meth:`_cache_sweep_arrays`
        so the cached ``_eta_expanded`` reflects the just-updated eta."""
        if self.eta is not None:
            self.eta.update()

    # --- sweep-level caching ------------------------------------------
    def _cache_sweep_arrays(self) -> None:
        """Populate the per-sweep caches read by Z and W's ``update_k``:
        ``_mask_tau[m] = mask * tau[m].E_tau_expanded()`` and, when eta is
        wired, ``_eta_expanded``. tau does not change within a Z/W sweep, so
        this is a pure performance cache with unchanged arithmetic.
        """
        self._mask_tau = [
            obs.mask * tau.E_tau_expanded()
            for obs, tau in zip(self.observations, self.tau)
        ]
        self._eta_expanded = (
            self.eta.E_eta_expanded() if self.eta is not None else None
        )
        self.factors._mask_tau = self._mask_tau
        self.factors._eta_expanded = self._eta_expanded
        for m in range(self.M):
            self.loadings[m]._mask_tau = self._mask_tau[m]

    # --- residual cache management ------------------------------------
    def _refresh_residual_cache(self) -> None:
        """Recompute every view's mean-residual cache ``mask * (Y - Z W^T)``.
        Called after wiring and after every pruning event."""
        Z = self.factors.E_z
        for m in range(self.M):
            Wm = self.loadings[m].E_w
            self.observations[m].refresh_residual_mean(Z, Wm)

    # --- shared Z / W sweeps ------------------------------------------
    def _sweep_factors(self) -> None:
        for k in range(self.K):
            self.factors.update_k(k)

    def _sweep_loadings(self) -> None:
        for k in range(self.K):
            for m in range(self.M):
                self.loadings[m].update_k(k)

    def _refresh_y_hat_moments_all(self) -> None:
        """Recompute ``E[Y_hat]`` and ``E[Y_hat^2]`` per view, once per sweep
        after the Z + W sweeps and before tau (no clean rank-1 form)."""
        for m in range(self.M):
            self.loadings[m].refresh_y_hat_moments()

    def _sweep_tau(self) -> None:
        for m in range(self.M):
            self.tau[m].update()

    # --- shared pruning -----------------------------------------------
    def prune(self, active_mask: np.ndarray) -> None:
        """Restrict every factor-indexed node to the selected columns and
        rebuild the K-dependent caches."""
        if active_mask.sum() == self.K:
            return
        self.factors.prune(active_mask)
        for m in range(self.M):
            self.loadings[m].prune(active_mask)
        self._prune_factor_nodes(active_mask)
        if self.eta is not None:
            self.eta.prune(active_mask)
        self.K = int(active_mask.sum())
        self._wire()
        self._refresh_residual_cache()
        for m in range(self.M):
            self.tau[m].refresh_residual()

    # --- variance explained -------------------------------------------
    @staticmethod
    def _nansumsq(A: np.ndarray) -> float:
        return float(np.nansum(A * A))

    def variance_explained_per_view(self) -> np.ndarray:
        """Per-view R^2 of the full reconstruction Y_hat = Z W^T."""
        R2m = np.zeros(self.M)
        Z = self.factors.E_z
        for m in range(self.M):
            Y_raw, mu = self.observations[m].get_raw_and_mean()
            Yc = Y_raw - mu
            mask = self.observations[m].mask
            Wm = self.loadings[m].E_w
            recon = Z @ Wm.T
            ss_res = self._nansumsq((Yc - recon) * mask)
            ss_tot = self._nansumsq(Yc * mask)
            R2m[m] = 1.0 - ss_res / (ss_tot + R2_EPS) if ss_tot > 0 else 0.0
        return np.clip(R2m, 0.0, 1.0)

    def _per_view_gain_and_denom(self, m: int) -> Tuple[np.ndarray, float]:
        """Vectorised per-factor variance gain and total denominator for view
        ``m``. ``gain[k]`` is the drop in masked squared residual from adding
        factor ``k`` on top of the others; ``denom`` is ``sum(mask * Y_c^2)``.
        Three BLAS matmuls per view replace the K per-column outer products.
        """
        Z = self.factors.E_z
        Y_raw, mu = self.observations[m].get_raw_and_mean()
        Yc = np.nan_to_num(Y_raw - mu, nan=0.0)
        mask = self.observations[m].mask
        Wm = self.loadings[m].E_w

        Yc_masked = Yc * mask
        denom = float(np.sum(Yc_masked * Yc_masked))
        if denom <= 0:
            return np.zeros(self.K), 0.0

        R = mask * (Yc - Z @ Wm.T)
        ZtR = Z.T @ R                                   # (K, D_m)
        gain_term1 = 2.0 * (ZtR * Wm.T).sum(axis=1)     # (K,)

        W_sq = Wm * Wm                                  # (D_m, K)
        mask_W_sq = mask @ W_sq                         # (N, K)
        gain_term2 = ((Z * Z) * mask_W_sq).sum(axis=0)  # (K,)

        return gain_term1 + gain_term2, denom

    def _per_view_gain_and_denom_per_group(
        self, m: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-(factor, group) variance gain ``(K, G)`` and per-group
        denominator ``(G,)`` for view ``m``. At ``G = 1`` this delegates to
        :meth:`_per_view_gain_and_denom` for bit-identical single-group output.
        """
        if self.G == 1:
            gain_k, denom = self._per_view_gain_and_denom(m)
            return gain_k[:, None], np.array([denom], dtype=float)

        Z = self.factors.E_z
        Y_raw, mu = self.observations[m].get_raw_and_mean()
        Yc = np.nan_to_num(Y_raw - mu, nan=0.0)
        mask = self.observations[m].mask
        Wm = self.loadings[m].E_w
        W_sq = Wm * Wm                                          # (D_m, K)

        gain_kg = np.zeros((self.K, self.G))
        denom_g = np.zeros(self.G)
        for g in range(self.G):
            rows = self.dataset.group_member_indices[g]
            if rows.size == 0:
                continue
            Z_g = Z[rows]                                        # (N_g, K)
            Yc_g = Yc[rows]                                      # (N_g, D_m)
            mask_g = mask[rows]                                  # (N_g, D_m)

            Yc_masked_g = Yc_g * mask_g
            d_g = float(np.sum(Yc_masked_g * Yc_masked_g))
            if d_g <= 0:
                continue
            denom_g[g] = d_g

            R_g = mask_g * (Yc_g - Z_g @ Wm.T)                   # (N_g, D_m)
            ZtR_g = Z_g.T @ R_g                                  # (K, D_m)
            gain_term1 = 2.0 * (ZtR_g * Wm.T).sum(axis=1)        # (K,)

            mask_W_sq_g = mask_g @ W_sq                          # (N_g, K)
            gain_term2 = ((Z_g * Z_g) * mask_W_sq_g).sum(axis=0)  # (K,)

            gain_kg[:, g] = gain_term1 + gain_term2

        return gain_kg, denom_g

    def variance_explained_per_factor(self) -> np.ndarray:
        """Per-factor R^2 marginalised across views."""
        gain_total = np.zeros(self.K)
        denom_total = 0.0
        for m in range(self.M):
            gain, denom = self._per_view_gain_and_denom(m)
            gain_total += gain
            denom_total += denom
        if denom_total <= 0:
            return np.zeros(self.K)
        return np.clip(gain_total / (denom_total + R2_EPS), 0.0, 1.0)

    def variance_explained_per_factor_view(self) -> np.ndarray:
        """Per-(factor, view) R^2 matrix of shape ``(K, M)``."""
        R2_km = np.zeros((self.K, self.M))
        for m in range(self.M):
            gain, denom = self._per_view_gain_and_denom(m)
            if denom <= 0:
                continue
            R2_km[:, m] = gain / (denom + R2_EPS)
        return np.clip(R2_km, 0.0, 1.0)

    def variance_explained_per_factor_view_group(self) -> np.ndarray:
        """Per-(factor, group, view) R^2 array of shape ``(K, G, M)``. At
        ``G = 1`` ``R2_kgm[:, 0, :]`` equals
        :meth:`variance_explained_per_factor_view` bit-identically."""
        R2_kgm = np.zeros((self.K, self.G, self.M))
        for m in range(self.M):
            gain_kg, denom_g = self._per_view_gain_and_denom_per_group(m)
            for g in range(self.G):
                if denom_g[g] <= 0:
                    continue
                R2_kgm[:, g, m] = gain_kg[:, g] / (denom_g[g] + R2_EPS)
        return np.clip(R2_kgm, 0.0, 1.0)

    # --- accessors ----------------------------------------------------
    def get_factors(self) -> np.ndarray:
        return self.factors.E_z

    def get_weights(self, view: Optional[int] = None):
        if view is None:
            return [wm.E_w for wm in self.loadings]
        return self.loadings[view].E_w

    def reconstruct(self, view: Optional[int] = None):
        Z = self.factors.E_z
        if view is None:
            return [Z @ wm.E_w.T for wm in self.loadings]
        return Z @ self.loadings[view].E_w.T


__all__ = ["_CLINGModelBase"]
