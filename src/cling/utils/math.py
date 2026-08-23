"""Numerical helpers shared across nodes."""

from __future__ import annotations

import numpy as np

#: Strictly positive floor used in :func:`log_eps` and to clip Gamma rates,
#: preventing ``log(0)`` and division by zero in moment computations.
EPS = 1e-10

#: Floor for the shape parameter ``a`` of a Gamma variational factor when
#: computing ``E[1/x] = b / (a - 1)`` (the inverse moment exists only for
#: ``a > 1``). Posterior shapes ``a = a_prior + 0.5 * D_m`` sit comfortably
#: above 1; the floor only guards the first pass before the data-driven update.
MIN_GAMMA_SHAPE = 1.0001

#: Stabilising offset for variance-explained denominators. A feature whose
#: total sum-of-squares is at or below this is treated as having no variance to
#: explain, so its per-factor R^2 is zero rather than NaN.
R2_EPS = 1e-12


def log_eps(x, eps: float = EPS):
    """Stable logarithm: ``log(max(x, eps))``."""
    return np.log(np.maximum(x, eps))


def clip_positive(x, eps: float = EPS):
    """Clip values to a strictly positive floor."""
    return np.maximum(x, eps)


__all__ = ["EPS", "MIN_GAMMA_SHAPE", "R2_EPS", "log_eps", "clip_positive"]
