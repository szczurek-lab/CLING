"""Mean-field variational inference loop."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import CONVERGENCE_THRESHOLDS, TrainingOptions
from .pruning import prune_inactive_factors

logger = logging.getLogger(__name__)

_CONVERGENCE_PATIENCE = 2


def run_mfvi(model, options: TrainingOptions) -> Dict[str, Any]:
    """Run mean-field variational inference on a CLING model in place.

    Returns a summary dict with ``elbo_history``, ``K_history``,
    ``n_iterations``, ``final_elbo``, and ``converged``. Convergence requires
    two consecutive relative-ELBO changes below the threshold, gated behind the
    ``prune_warmup`` warm-up and a post-prune cool-down. The reference for the
    relative change is selected by ``options.convergence_ref`` (see
    :class:`cling.config.TrainingOptions`).
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None

    threshold = CONVERGENCE_THRESHOLDS[options.convergence_mode]
    _use_last_ref: bool = (options.convergence_ref == "last")
    _use_warmup_ref: bool = (options.convergence_ref == "warmup")
    _use_warmup_frac: bool = (options.convergence_ref == "warmup_frac")
    _capture_warmup: bool = _use_warmup_ref or _use_warmup_frac

    elbo_history: List[float] = []
    K_history: List[int] = []

    first_elbo: float = float(model.compute_elbo())
    _first_elbo_is_valid: bool = bool(np.isfinite(first_elbo)) and abs(first_elbo) > 0.0

    min_iter_for_convergence: int = max(1, options.prune_warmup)

    _ref_capture_iter: int = (
        int(options.convergence_ref_iter)
        if options.convergence_ref_iter is not None
        else min_iter_for_convergence
    )
    _warmup_ref: Optional[float] = None

    last_successful_prune: int = -1
    convergence_token = 0
    converged = False

    pbar = (
        tqdm(total=options.max_iter, desc="CLING fit", disable=not options.verbose)
        if tqdm is not None
        else None
    )

    for it in range(options.max_iter):
        model.update_step()
        elbo = model.compute_elbo()
        elbo_history.append(elbo)
        K_history.append(model.K)

        if not _first_elbo_is_valid and np.isfinite(elbo) and abs(elbo) > 0.0:
            first_elbo = float(elbo)
            _first_elbo_is_valid = True

        if (
            _capture_warmup
            and _warmup_ref is None
            and it >= _ref_capture_iter
            and np.isfinite(elbo)
            and abs(elbo) > 0.0
        ):
            _warmup_ref = abs(float(elbo))

        if pbar is not None:
            try:
                pbar.set_postfix({"ELBO": f"{elbo:.1f}", "K": model.K})
            except Exception:
                pass
            pbar.update(1)

        # Scheduled pruning: at most one factor per call.
        if (
            it > 0
            and it >= options.prune_warmup
            and it % options.prune_every == 0
        ):
            dropped = prune_inactive_factors(
                model,
                threshold=options.prune_threshold,
                min_factors=options.prune_min_factors,
            )
            if dropped >= 0:
                logger.info(
                    "iter %d: pruned factor %d; K -> %d", it, dropped, model.K
                )
                convergence_token = 0
                last_successful_prune = it

        in_cooldown = (
            last_successful_prune >= 0
            and it - last_successful_prune < options.prune_every
        )
        if _use_last_ref:
            _ref = abs(elbo_history[-2]) if len(elbo_history) >= 2 else 0.0
            _ref_valid = bool(np.isfinite(_ref)) and _ref > 0.0
        elif _capture_warmup:
            _ref = _warmup_ref if _warmup_ref is not None else 0.0
            _ref_valid = (_warmup_ref is not None) and _warmup_ref > 0.0
        else:
            _ref = abs(first_elbo)
            _ref_valid = _first_elbo_is_valid
        if (
            it >= min_iter_for_convergence
            and not in_cooldown
            and len(elbo_history) >= 2
            and _ref_valid
        ):
            delta = elbo - elbo_history[-2]
            if _use_last_ref or _use_warmup_frac:
                rel = abs(delta) / (_ref + 1e-12)
            else:
                rel = 100.0 * abs(delta) / _ref
            if rel < threshold:
                convergence_token += 1
                if convergence_token >= _CONVERGENCE_PATIENCE:
                    final_dropped = prune_inactive_factors(
                        model,
                        threshold=options.prune_threshold,
                        min_factors=options.prune_min_factors,
                    )
                    if final_dropped >= 0:
                        logger.info(
                            "iter %d: ELBO stable but factor %d still "
                            "below threshold; pruned and continuing "
                            "(K -> %d)",
                            it, final_dropped, model.K,
                        )
                        convergence_token = 0
                        last_successful_prune = it
                    else:
                        logger.info(
                            "converged at iteration %d (ELBO=%.4f, K=%d)",
                            it + 1, elbo, model.K,
                        )
                        converged = True
                        break
            else:
                convergence_token = 0

    if pbar is not None:
        pbar.close()

    return {
        "elbo_history": tuple(elbo_history),
        "K_history": tuple(K_history),
        "n_iterations": len(elbo_history),
        "final_elbo": elbo_history[-1] if elbo_history else float("nan"),
        "converged": converged,
    }


__all__ = ["run_mfvi"]
