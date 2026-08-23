"""Hyperparameters and training options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Preset relative-ELBO convergence thresholds (percentage of the reference
# |ELBO| below which the change must stay for two consecutive iterations).
# Aligned with mofapy2's ``assess_convergence``.
CONVERGENCE_THRESHOLDS = {
    "fast": 5e-4,
    "medium": 5e-5,
    "slow": 5e-6,
}


@dataclass(frozen=True)
class ShrinkageHyperparameters:
    """Multiplicative Gamma process prior on ``delta``.

    The first column uses ``(a1, b1)``; subsequent columns use ``(a2, b2)``.
    The default ``Gamma(3, 2.5)`` is the mild shrinkage recommended for
    ``N < 1000`` (see :func:`paper_defaults_for_n`); pass ``b1 = b2 = 1.0`` for
    the stronger ``Gamma(3, 1)`` used at ``N >= 1000``.
    """

    a1: float = 3.0
    b1: float = 2.5
    a2: float = 3.0
    b2: float = 2.5


@dataclass(frozen=True)
class LocalPrecisionHyperparameters:
    """Local precision ``alpha`` and feature scale ``phi`` (Gamma-Gamma).

    The default ``a_alpha = b_alpha = 0.1`` is the main-model operating point;
    the CLING-MGP / CLING-ARD ablations instead use ``Gamma(1.5, 1.5)`` (see
    :data:`ABLATION_LOCAL_PRECISION`).
    """

    a_alpha: float = 0.1
    b_alpha: float = 0.1
    a_phi: float = 0.5
    b_phi: float = 1.0


@dataclass(frozen=True)
class NoiseHyperparameters:
    """Gamma prior on the per-feature noise precision ``tau``."""

    a_tau: float = 1e-3
    b_tau: float = 1e-3


@dataclass(frozen=True)
class LatentARDHyperparameters:
    """Gamma prior on the group-factor ARD precision ``eta_{g, k}`` on Z.

    Defaults match MOFA+'s ``AlphaZ``: an essentially improper prior
    (``a = b = 1e-14``) that lets the data determine per-group, per-factor
    activity. Only used when the grouped extension is active (``G > 1``).
    """

    a_eta: float = 1e-14
    b_eta: float = 1e-14


@dataclass(frozen=True)
class CLINGHyperparameters:
    """Bundle of all hyperparameters for the CLING model family."""

    shrinkage: ShrinkageHyperparameters = ShrinkageHyperparameters()
    precision: LocalPrecisionHyperparameters = LocalPrecisionHyperparameters()
    noise: NoiseHyperparameters = NoiseHyperparameters()
    latent_ard: LatentARDHyperparameters = LatentARDHyperparameters()


#: Local/view precision prior for the CLING-MGP and CLING-ARD ablations: a
#: single ``Gamma(1.5, 1.5)`` prior (manuscript). The main CLING model instead
#: uses the Gamma-Gamma hierarchy default on
#: :class:`LocalPrecisionHyperparameters`.
ABLATION_LOCAL_PRECISION = LocalPrecisionHyperparameters(a_alpha=1.5, b_alpha=1.5)


@dataclass(frozen=True)
class ModelOptions:
    """Model-construction options.

    ``K_init``
        Initial (overcomplete) number of latent factors. The default is the
        paper's fixed truncation ceiling of 30; it bounds but does not
        preselect the number of active factors.
    ``init_mode``
        ``"pca"`` (default) or ``"random"``.
    ``ard_factors``
        Whether to wire the group-factor ARD precision ``eta`` into Z's prior.
        ``None`` resolves to ``True`` iff the dataset has ``G > 1``.
    ``center_groups``
        Per-group feature centering; a no-op at ``G = 1``.
    """

    K_init: int = 30
    init_mode: str = "pca"          # "pca" or "random"
    center: bool = True
    seed: int | None = None
    center_groups: bool = True
    ard_factors: Optional[bool] = None


@dataclass(frozen=True)
class TrainingOptions:
    """Mean-field VI training options.

    Convergence is declared when the relative ELBO change stays below the
    ``convergence_mode`` threshold for two consecutive iterations, gated behind
    ``prune_warmup``. ``convergence_ref`` selects the denominator of the test:

    * ``"first"`` (default): ``100 * |delta| / |first_elbo|`` (mofapy2 parity).
    * ``"last"``: ``|delta| / |last_elbo|`` (running reference).
    * ``"warmup"`` / ``"warmup_frac"``: fixed |ELBO| captured at
      ``convergence_ref_iter`` (or the warm-up iteration), as a percentage or
      raw fraction respectively.
    * ``"auto"``: resolved by sample size in :func:`cling.fit`.

    ``prune_threshold=None`` disables in-fit R^2 pruning (the library default,
    mirroring mofapy2's ``dropR2=None``); factor selection is then the single
    post-convergence per-view R^2 cut. ``max_iter`` is a safety cap.
    """

    max_iter: int = 4000
    convergence_mode: str = "slow"
    prune_warmup: int = 250
    prune_every: int = 1
    prune_threshold: Optional[float] = None
    prune_min_factors: int = 1
    verbose: bool = True
    convergence_ref: str = "first"
    convergence_ref_iter: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1; got {self.max_iter}.")
        if self.convergence_mode not in CONVERGENCE_THRESHOLDS:
            raise ValueError(
                f"convergence_mode must be one of "
                f"{sorted(CONVERGENCE_THRESHOLDS)}; got {self.convergence_mode!r}."
            )
        if self.convergence_ref not in ("first", "last", "warmup", "warmup_frac", "auto"):
            raise ValueError(
                f"convergence_ref must be 'first', 'last', 'warmup', "
                f"'warmup_frac', or 'auto'; got {self.convergence_ref!r}."
            )
        if self.convergence_ref_iter is not None and self.convergence_ref_iter < 1:
            raise ValueError(
                f"convergence_ref_iter must be None or >= 1; "
                f"got {self.convergence_ref_iter}."
            )
        if self.prune_warmup < 0:
            raise ValueError(f"prune_warmup must be >= 0; got {self.prune_warmup}.")
        if self.prune_every < 1:
            raise ValueError(f"prune_every must be >= 1; got {self.prune_every}.")
        if self.prune_threshold is not None and not 0.0 <= self.prune_threshold < 1.0:
            raise ValueError(
                f"prune_threshold must be None or lie in [0, 1); "
                f"got {self.prune_threshold}."
            )
        if self.prune_min_factors < 1:
            raise ValueError(
                f"prune_min_factors must be >= 1; got {self.prune_min_factors}."
            )


__all__ = [
    "CONVERGENCE_THRESHOLDS",
    "ShrinkageHyperparameters",
    "LocalPrecisionHyperparameters",
    "ABLATION_LOCAL_PRECISION",
    "NoiseHyperparameters",
    "LatentARDHyperparameters",
    "CLINGHyperparameters",
    "ModelOptions",
    "TrainingOptions",
    "SAMPLE_SIZE_SWITCH",
    "paper_defaults_for_n",
]


# Sample-size-adaptive defaults (paper operating point). Applied automatically
# by ``cling.fit`` when the caller supplies neither ``hp`` nor ``training``;
# fully overridable and documented here.
SAMPLE_SIZE_SWITCH: int = 1000


def paper_defaults_for_n(n_samples: int) -> dict:
    """Return the sample-size-adaptive default settings for ``n_samples``.

    * ``n_samples < SAMPLE_SIZE_SWITCH`` (1000): mild shrinkage ``Gamma(3, 2.5)``;
      convergence ``"first"`` with the ELBO check gated to iteration 250.
    * ``n_samples >= SAMPLE_SIZE_SWITCH``: stronger shrinkage ``Gamma(3, 1)``;
      convergence ``"warmup_frac"`` with a fixed reference captured at
      iteration 3 and the ELBO check gated to iteration 150.

    Both regimes disable in-fit R^2 pruning; dimensionality is set by the
    post-convergence R^2 cut. The tolerance (``convergence_mode``) is ``"slow"``
    (5e-6) in both.
    """
    if int(n_samples) >= SAMPLE_SIZE_SWITCH:
        return {
            "shrinkage": {"a1": 3.0, "b1": 1.0, "a2": 3.0, "b2": 1.0},
            "convergence_ref": "warmup_frac",
            "convergence_ref_iter": 3,
            "prune_warmup": 150,
            "convergence_mode": "slow",
            "prune_threshold": None,
        }
    return {
        "shrinkage": {"a1": 3.0, "b1": 2.5, "a2": 3.0, "b2": 2.5},
        "convergence_ref": "first",
        "convergence_ref_iter": None,
        "prune_warmup": 250,
        "convergence_mode": "slow",
        "prune_threshold": None,
    }
