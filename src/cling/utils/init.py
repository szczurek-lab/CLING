"""Initialization strategies for latent factors and loadings."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import scipy

# Default RNG seed for the truncated-SVD solvers, used only when the caller
# passes ``seed=None`` so the no-seed library path stays reproducible.
_SVDS_RANDOM_STATE = 0


def _scipy_version_at_least(major: int, minor: int) -> bool:
    """Return True iff ``scipy.__version__`` is at least ``major.minor``."""
    parts = scipy.__version__.split(".")
    try:
        sci_major = int(parts[0])
        sci_minor = int(parts[1])
    except (IndexError, ValueError):
        return True
    return (sci_major, sci_minor) >= (major, minor)


# SciPy 1.14 renamed ``random_state`` to ``rng`` on the ``scipy.sparse.linalg``
# random-using APIs; used only by the opt-in ``arpack`` solver below.
_SVDS_RNG_KW = "rng" if _scipy_version_at_least(1, 14) else "random_state"


def _randomized_svd(
    A: np.ndarray,
    n_components: int,
    seed: int,
    n_oversamples: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truncated randomized SVD (Halko, Martinsson & Tropp, 2011).

    Returns the top-``n_components`` singular triples ``(U, S, Vt)`` in
    descending order. Reproducible for a fixed ``seed`` and genuinely varying
    across seeds. Oversampling and power-iteration count follow scikit-learn's
    ``randomized_svd`` defaults so the approximation quality matches.
    """
    rng = np.random.default_rng(seed)
    n, d = A.shape
    n_random = min(n_components + n_oversamples, n, d)
    n_iter = 7 if n_components < 0.1 * min(A.shape) else 4

    Omega = rng.standard_normal(size=(d, n_random))
    Y = A @ Omega
    for _ in range(n_iter):
        Q, _ = np.linalg.qr(Y)
        Q2, _ = np.linalg.qr(A.T @ Q)
        Y = A @ Q2
    Q, _ = np.linalg.qr(Y)

    B = Q.T @ A
    U_b, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_b

    U = U[:, :n_components]
    S = S[:n_components]
    Vt = Vt[:n_components, :]

    # Deterministic sign convention: make the largest-magnitude entry of each
    # left singular vector positive so a fixed seed gives a stable sign.
    max_abs = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_abs, np.arange(U.shape[1])])
    signs[signs == 0] = 1.0
    U = U * signs
    Vt = Vt * signs[:, None]
    return U, S, Vt


def pca_initialization(
    views: List[np.ndarray],
    K: int,
    seed: Optional[int] = _SVDS_RANDOM_STATE,
    solver: str = "randomized",
) -> tuple[np.ndarray, List[np.ndarray]]:
    """PCA-based initial values for ``Z`` and view-specific ``W``.

    Concatenates centered views feature-wise and runs a truncated SVD to
    extract the top-``K`` components, returning ``Z = U * S`` (scores) and
    ``W = V`` (loadings) so ``Z @ W^T`` reconstructs the centered data. ``NaN``
    entries are replaced by zero for the decomposition only. ``solver`` is
    ``"randomized"`` (default, pure-numpy, seeded) or ``"arpack"`` (exact,
    via ``scipy.sparse.linalg.svds``, seed-invariant).
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}.")
    if solver not in ("randomized", "arpack"):
        raise ValueError(
            f"Unknown solver {solver!r}; expected 'randomized' or 'arpack'."
        )

    N = views[0].shape[0]
    D = [v.shape[1] for v in views]

    centered_blocks = []
    for v in views:
        v_centered = v - np.nanmean(v, axis=0, keepdims=True)
        centered_blocks.append(np.nan_to_num(v_centered))

    Y_concat = np.hstack(centered_blocks)

    min_dim = min(Y_concat.shape)
    K_eff = min(K, min_dim)
    if K_eff < min_dim - 1:
        svd_seed = _SVDS_RANDOM_STATE if seed is None else seed
        if solver == "randomized":
            U_k, S_k, Vt_k = _randomized_svd(Y_concat, K_eff, svd_seed)
        else:  # solver == "arpack"
            from scipy.sparse.linalg import svds

            U_k, S_k, Vt_k = svds(
                Y_concat, k=K_eff, **{_SVDS_RNG_KW: svd_seed}
            )
            order = np.argsort(-S_k)
            U_k = U_k[:, order]
            S_k = S_k[order]
            Vt_k = Vt_k[order, :]
    else:
        U_full, S_full, Vt_full = np.linalg.svd(Y_concat, full_matrices=False)
        U_k = U_full[:, :K_eff]
        S_k = S_full[:K_eff]
        Vt_k = Vt_full[:K_eff, :]

    Z_init = np.zeros((N, K))
    Z_init[:, :K_eff] = U_k * S_k

    W_inits: List[np.ndarray] = []
    start = 0
    for d in D:
        Wm = np.zeros((d, K))
        Wm[:, :K_eff] = Vt_k[:, start : start + d].T
        W_inits.append(Wm)
        start += d

    return Z_init, W_inits


def random_initialization(
    rng: np.random.Generator,
    N: int,
    D: List[int],
    K: int,
    scale: float = 1.0,
) -> tuple[np.ndarray, List[np.ndarray]]:
    """Random Gaussian initial values for ``Z`` and view-specific ``W``."""
    Z_init = rng.normal(0.0, 0.1, size=(N, K))
    W_inits = [rng.normal(0.0, scale, size=(d, K)) for d in D]
    return Z_init, W_inits


__all__ = ["pca_initialization", "random_initialization"]
