"""Public user-facing API."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Optional

import numpy as np

from .config import (
    ABLATION_LOCAL_PRECISION,
    CLINGHyperparameters,
    LocalPrecisionHyperparameters,
    ModelOptions,
    ShrinkageHyperparameters,
    TrainingOptions,
    paper_defaults_for_n,
)
from .data import MultiviewDataset
from .inference import run_mfvi as _run_mfvi
from .model import CLINGModel, build_model
from .model_ard import CLINGModelARD, build_model_ard
from .model_mgp import CLINGModelMGP, build_model_mgp
from .results import FittedModel, TrainingSummary

logger = logging.getLogger(__name__)

# Public variant display names -> internal builders. The primary model of the
# paper is "CLING"; "CLING-MGP" and "CLING-ARD" are the two ablations.
_VARIANT_BUILDERS = {
    "CLING": build_model,
    "CLING-MGP": build_model_mgp,
    "CLING-ARD": build_model_ard,
}

_ABLATION_VARIANTS = frozenset({"CLING-MGP", "CLING-ARD"})


def _is_ablation(variant: str) -> bool:
    """True for the CLING-MGP / CLING-ARD ablations (case/separator-insensitive)."""
    return str(variant).strip().upper().replace("_", "-") in _ABLATION_VARIANTS


def _resolve_variant(variant: str):
    """Map a variant display name to its builder (case/separator-insensitive)."""
    key = str(variant).strip().upper().replace("_", "-")
    try:
        return _VARIANT_BUILDERS[key]
    except KeyError:
        raise ValueError(
            f"variant must be one of {sorted(_VARIANT_BUILDERS)}; got {variant!r}."
        ) from None


def _infer_n_samples(Y) -> Optional[int]:
    """Best-effort sample count (rows) from a views list or a dataset."""
    if isinstance(Y, MultiviewDataset):
        for attr in ("n_samples", "N", "num_samples"):
            v = getattr(Y, attr, None)
            if isinstance(v, (int, np.integer)):
                return int(v)
        views = getattr(Y, "views", None)
        if views is None:
            views = getattr(Y, "Y", None)
        if views:
            try:
                return int(np.asarray(views[0]).shape[0])
            except Exception:
                return None
        return None
    try:
        return int(np.asarray(Y[0]).shape[0])
    except Exception:
        return None


def build(
    Y,
    K_init: int = 30,
    *,
    variant: str = "CLING",
    view_names: Optional[Sequence[str]] = None,
    hp: Optional[CLINGHyperparameters] = None,
    center: bool = True,
    scale_views: bool = False,
    init_mode: str = "pca",
    seed: Optional[int] = None,
    groups: Optional[Sequence] = None,
    center_groups: bool = True,
    ard_factors: Optional[bool] = None,
):
    """Build a fresh CLING model from a list of view arrays (or a
    :class:`MultiviewDataset`).

    ``build`` does NOT apply the sample-size-adaptive default prior; when ``hp``
    is ``None`` it uses the library-default shrinkage. Call :func:`fit` for the
    automatic N-adaptive operating point, or pass an explicit ``hp`` here.
    """
    if isinstance(Y, MultiviewDataset):
        dataset = Y
    else:
        dataset = MultiviewDataset.from_arrays(
            Y,
            view_names=view_names,
            center=center,
            scale_views=scale_views,
            groups=groups,
            center_groups=center_groups,
        )

    options = ModelOptions(
        K_init=K_init,
        init_mode=init_mode,
        center=center,
        seed=seed,
        center_groups=center_groups,
        ard_factors=ard_factors,
    )

    builder = _resolve_variant(variant)
    return builder(dataset, options, hp=hp)


def run(
    model,
    options: Optional[TrainingOptions] = None,
    *,
    seed: Optional[int] = None,
    K_init: Optional[int] = None,
    init_mode: Optional[str] = None,
) -> FittedModel:
    """Run mean-field variational inference on ``model`` in place and return a
    :class:`FittedModel` exposing factors, loadings, and the training trace."""
    opts = options if options is not None else TrainingOptions()
    summary = _run_mfvi(model, opts)
    training_summary = TrainingSummary(
        n_iterations=int(summary["n_iterations"]),
        final_elbo=float(summary["final_elbo"]),
        converged=bool(summary["converged"]),
        elbo_history=tuple(summary["elbo_history"]),
        K_history=tuple(summary["K_history"]),
    )
    inferred_K_init = K_init
    if inferred_K_init is None and training_summary.K_history:
        inferred_K_init = int(training_summary.K_history[0])
    return FittedModel.from_model(
        model,
        training_summary,
        seed=seed,
        K_init=inferred_K_init,
        init_mode=init_mode,
        training_options=opts,
    )


def fit(
    Y,
    K_init: int = 30,
    *,
    variant: str = "CLING",
    view_names: Optional[Sequence[str]] = None,
    hp: Optional[CLINGHyperparameters] = None,
    center: bool = True,
    scale_views: bool = False,
    init_mode: str = "pca",
    seed: Optional[int] = None,
    groups: Optional[Sequence] = None,
    center_groups: bool = True,
    ard_factors: Optional[bool] = None,
    max_iter: int = 4000,
    convergence_mode: Optional[str] = None,
    prune_warmup: Optional[int] = None,
    prune_every: int = 1,
    prune_threshold: Optional[float] = None,
    training: Optional[TrainingOptions] = None,
) -> FittedModel:
    """Convenience wrapper: ``build`` immediately followed by ``run``.

    When the caller supplies neither an explicit ``hp`` nor an explicit
    ``training``, the operating point is resolved automatically from the sample
    size via :func:`cling.config.paper_defaults_for_n`: mild ``Gamma(3, 2.5)``
    shrinkage for ``N < 1000`` and stronger ``Gamma(3, 1)`` for ``N >= 1000``,
    each with its own convergence reference. Both regimes disable in-fit R^2
    pruning, so factor selection is the single post-convergence per-view
    ``R^2 >= 0.01`` cut. Pass an explicit ``hp`` / ``training`` (or the
    ``convergence_mode`` / ``prune_warmup`` shortcuts) to override. ``max_iter``
    is a safety cap.
    """
    n_samples = _infer_n_samples(Y)
    pd = paper_defaults_for_n(n_samples if n_samples is not None else 0)

    auto_ref: str = "first"
    auto_ref_iter: Optional[int] = None
    if hp is None and training is None:
        precision = (
            ABLATION_LOCAL_PRECISION
            if _is_ablation(variant)
            else LocalPrecisionHyperparameters()
        )
        hp = CLINGHyperparameters(
            shrinkage=ShrinkageHyperparameters(**pd["shrinkage"]),
            precision=precision,
        )
        auto_ref = pd["convergence_ref"]
        auto_ref_iter = pd["convergence_ref_iter"]
        if convergence_mode is None:
            convergence_mode = pd["convergence_mode"]
        if prune_warmup is None:
            prune_warmup = pd["prune_warmup"]
        if prune_threshold is None:
            prune_threshold = pd["prune_threshold"]
        logger.info(
            "cling.fit: sample-size-adaptive defaults for N=%s -> "
            "shrinkage=%s, convergence_ref=%s, convergence_ref_iter=%s, "
            "prune_warmup=%s, convergence_mode=%s",
            n_samples, pd["shrinkage"], auto_ref, auto_ref_iter,
            prune_warmup, convergence_mode,
        )

    if convergence_mode is None:
        convergence_mode = "slow"
    if prune_warmup is None:
        prune_warmup = 250

    model = build(
        Y,
        K_init=K_init,
        variant=variant,
        view_names=view_names,
        hp=hp,
        center=center,
        scale_views=scale_views,
        init_mode=init_mode,
        seed=seed,
        groups=groups,
        center_groups=center_groups,
        ard_factors=ard_factors,
    )
    if training is None:
        training = TrainingOptions(
            max_iter=max_iter,
            convergence_mode=convergence_mode,
            prune_warmup=prune_warmup,
            prune_every=prune_every,
            prune_threshold=prune_threshold,
            convergence_ref=auto_ref,
            convergence_ref_iter=auto_ref_iter,
        )
    return run(
        model,
        options=training,
        seed=seed,
        K_init=K_init,
        init_mode=init_mode,
    )


__all__ = [
    "build",
    "run",
    "fit",
    "CLINGModel",
    "CLINGModelMGP",
    "CLINGModelARD",
]
