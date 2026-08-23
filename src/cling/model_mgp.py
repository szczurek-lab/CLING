"""CLING-MGP ablation: per-feature local precision without ``phi`` coupling."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .config import ABLATION_LOCAL_PRECISION, CLINGHyperparameters, ModelOptions
from .data import MultiviewDataset
from .model_base import _CLINGModelBase
from .nodes import Alpha, Delta, Eta, Factors, Loadings, Observations, Tau
from .utils import pca_initialization, random_initialization


class CLINGModelMGP(_CLINGModelBase):
    """CLING variant with a single-Gamma prior on the local precision:
    ``alpha^m_{d,k} ~ Ga(a_alpha, b_alpha)`` (no ``phi`` coupling)."""

    def __init__(
        self,
        dataset: MultiviewDataset,
        K: int,
        hp: CLINGHyperparameters,
        factors: Factors,
        observations: List[Observations],
        loadings: List[Loadings],
        alpha: List[Alpha],
        tau: List[Tau],
        delta: List[Delta],
        eta: Optional[Eta] = None,
    ):
        super().__init__(dataset, K, hp, factors, observations, loadings, tau, eta=eta)
        self.alpha = alpha
        self.delta = delta
        self._wire()
        self._refresh_residual_cache()

    def _wire(self) -> None:
        if self.eta is not None:
            self.eta.set_neighbours(self.factors)
        self.factors.set_neighbours(
            self.observations, self.loadings, self.tau, eta=self.eta,
        )
        for m in range(self.M):
            self.observations[m].set_neighbours(self.loadings[m], self.tau[m])
            self.loadings[m].set_neighbours(
                self.alpha[m], self.factors, self.observations[m], self.tau[m], self.delta[m]
            )
            self.alpha[m].set_neighbours(self.loadings[m], self.delta[m], phi=None)
            self.tau[m].set_neighbours(self.observations[m], self.loadings[m], self.factors)
            self.delta[m].set_neighbours(self.loadings[m], self.alpha[m])

    def update_step(self) -> None:
        self._sweep_eta()
        self._cache_sweep_arrays()
        self._sweep_factors()
        self._sweep_loadings()
        self._refresh_y_hat_moments_all()
        self._sweep_tau()
        for m in range(self.M):
            self.alpha[m].update()
        for m in range(self.M):
            self.delta[m].update()

    def compute_elbo(self) -> float:
        total = self.factors.elbo()
        if self.eta is not None:
            total += self.eta.elbo()
        for m in range(self.M):
            total += (
                self.loadings[m].elbo()
                + self.alpha[m].elbo()
                + self.observations[m].elbo()
                + self.tau[m].elbo()
                + self.delta[m].elbo()
            )
        return float(total)

    def _prune_factor_nodes(self, active_mask: np.ndarray) -> None:
        for m in range(self.M):
            self.alpha[m].prune(active_mask)
            self.delta[m].prune(active_mask)


def build_model_mgp(
    dataset: MultiviewDataset,
    options: ModelOptions,
    hp: Optional[CLINGHyperparameters] = None,
) -> CLINGModelMGP:
    """Construct a fresh CLING-MGP ablation model."""
    hp = hp or CLINGHyperparameters(precision=ABLATION_LOCAL_PRECISION)
    rng = np.random.default_rng(options.seed)

    N, M, D = dataset.N, dataset.M, dataset.D
    K = options.K_init

    if options.init_mode == "pca":
        Z_init, W_inits = pca_initialization(dataset.centered_views, K, seed=options.seed)
    elif options.init_mode == "random":
        Z_init, W_inits = random_initialization(rng, N, D, K)
    else:
        raise ValueError(
            f"init_mode must be 'pca' or 'random'; got {options.init_mode!r}."
        )

    ard_factors = options.ard_factors
    if ard_factors is None:
        ard_factors = dataset.G > 1
    else:
        ard_factors = bool(ard_factors)

    factors = Factors(N=N, K=K, M=M, D=D, mu_init=Z_init, var_init=np.ones((N, K)))

    observations: List[Observations] = []
    loadings: List[Loadings] = []
    alpha: List[Alpha] = []
    tau: List[Tau] = []
    delta: List[Delta] = []

    for m in range(M):
        observations.append(
            Observations(
                data=dataset.centered_views[m],
                m=m, N=N, K=K, M=M, D=D,
                data_mean=dataset.feature_means[m],
                G=dataset.G,
                group_ix=dataset.group_ix,
            )
        )
        loadings.append(
            Loadings(
                m=m, N=N, K=K, M=M, D=D,
                mu_init=W_inits[m], var_init=np.ones((D[m], K)),
            )
        )
        delta.append(
            Delta(
                a1=hp.shrinkage.a1, b1=hp.shrinkage.b1,
                a2=hp.shrinkage.a2, b2=hp.shrinkage.b2,
                m=m, N=N, K=K, M=M, D=D,
            )
        )
        alpha.append(
            Alpha(
                a_alpha=hp.precision.a_alpha, b_alpha=hp.precision.b_alpha,
                m=m, N=N, K=K, M=M, D=D,
            )
        )
        tau.append(
            Tau(
                a0=hp.noise.a_tau, b0=hp.noise.b_tau,
                m=m, N=N, K=K, M=M, D=D,
                G=dataset.G,
                group_ix=dataset.group_ix,
                N_per_group=dataset.N_per_group,
            )
        )

    eta: Optional[Eta] = None
    if ard_factors:
        eta = Eta(
            a_eta=hp.latent_ard.a_eta,
            b_eta=hp.latent_ard.b_eta,
            N=N, K=K,
            G=dataset.G,
            group_ix=dataset.group_ix,
            N_per_group=dataset.N_per_group,
        )

    return CLINGModelMGP(
        dataset=dataset, K=K, hp=hp,
        factors=factors, observations=observations,
        loadings=loadings, alpha=alpha, tau=tau, delta=delta,
        eta=eta,
    )


__all__ = ["CLINGModelMGP", "build_model_mgp"]
