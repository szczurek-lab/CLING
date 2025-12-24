# Simulate multi-view Gaussian data under CLING-Sparsity.
import numpy as np

def sparsity_data(
    N,
    D,
    K,
    M=None,
    a_alpha=1, b_alpha=1,
    a_tau=1.0, b_tau=1.0,
    sparsity=0.75,
    noise_std=None,
    # --- MCAR missingness ---
    mcar_missing_frac=0.0,
    mcar_whole_view_frac=0.0,
    # --- Misc ---
    center=False,
    seed=1234,
    eps=1e-12,
):
    """
    Model (rate parameterization for Gamma; numpy uses scale=1/rate):

      Z_{n,k} ~ N(0,1)                               for n=1..N, k=1..K
      α_{d,k}^m ~ Ga(a_alpha, b_alpha)               for d=1..D_m, k=1..K
      W_{d,k}^m | α_{d,k}^m ~ N(0, 1/α_{d,k}^m)
      τ_d^m ~ Ga(a_tau, b_tau)
      Y_{n,d}^m ~ N( (Z W^{(m)T})_{n,d}, 1/τ_d^m )

    - If noise_std is provided, τ_d^m is fixed to 1/noise_std^2 for convenience.
    """

    # ----- validation -----
    for name, val in dict(a_alpha=a_alpha, b_alpha=b_alpha,
                          a_tau=a_tau, b_tau=b_tau).items():
        if val <= 0:
            raise ValueError(f"{name} must be > 0.")
    if N <= 0 or K <= 0:
        raise ValueError("N and K must be positive integers.")
    if noise_std is not None and noise_std <= 0:
        raise ValueError("noise_std must be > 0 when provided.")

    rng = np.random.default_rng(seed)

    # Determine M and D list
    if M is None:
        if isinstance(D, (list, tuple, np.ndarray)):
            M = len(D)
        else:
            raise ValueError("If M is None, D must be a list/tuple/array of per-view dimensions.")
    if isinstance(D, int):
        D = [D] * M
    D = list(map(int, D))
    if len(D) != M:
        raise ValueError("Length of D must equal M.")

    # Latent scores Z ~ N(0, I)
    Z = rng.normal(0.0, 1.0, size=(N, K))

    # containers
    W_list, taus, alphas = [], [], []
    F_list, Y_list = [], []

    # ---------- per view ----------
    for m in range(M):
        Dm = D[m]

        # ARD α^{m}_{d,k} (no φ term anymore)
        alpha = np.maximum(
            rng.gamma(shape=a_alpha, scale=1.0 / b_alpha, size=(Dm, K)),
            eps
        )  # (Dm, K)

        # Loadings W^{(m)}
        denom = np.sqrt(np.maximum(alpha, eps))
        Wm = rng.normal(0.0, 1.0, size=(Dm, K)) / denom
        mask = (np.random.rand(D[m], K) > sparsity).astype(float)
        Wm = Wm * mask

        # Mean matrix F = Z W^T
        Fm = Z @ Wm.T

        # Noise precisions τ^{m}_d
        if noise_std is None:
            tau_m = np.maximum(rng.gamma(shape=a_tau, scale=1.0 / b_tau, size=Dm), eps)
        else:
            tau_m = np.full(Dm, 1.0 / (noise_std ** 2), dtype=float)

        # Sample Y
        eps_mat = rng.normal(0.0, 1.0, size=(N, Dm)) / np.sqrt(tau_m[np.newaxis, :])
        Ym = Fm + eps_mat
        if center:
            Ym = Ym - Ym.mean(axis=0, keepdims=True)

        # Collect
        W_list.append(Wm); taus.append(tau_m); alphas.append(alpha)
        F_list.append(Fm); Y_list.append(Ym)

    # ---------- apply MCAR missingness ----------
    if not (0.0 <= mcar_missing_frac <= 1.0):
        raise ValueError("mcar_missing_frac must be in [0,1].")
    if not (0.0 <= mcar_whole_view_frac <= 1.0):
        raise ValueError("mcar_whole_view_frac must be in [0,1].")

    if mcar_missing_frac > 0.0:
        for m in range(M):
            Nm = N * D[m]
            mask_count = int(round(mcar_missing_frac * Nm))
            if mask_count > 0:
                idx = rng.choice(Nm, size=mask_count, replace=False)
                Ym = Y_list[m].copy()
                flat = Ym.reshape(-1)
                flat[idx] = np.nan
                Y_list[m] = flat.reshape(Ym.shape)

    if mcar_whole_view_frac > 0.0:
        for m in range(M):
            sample_count = int(round(mcar_whole_view_frac * N))
            if sample_count > 0:
                rows = rng.choice(N, size=sample_count, replace=False)
                Ym = Y_list[m].copy()
                Ym[rows, :] = np.nan
                Y_list[m] = Ym

    data = {f"M{m}": Y_list[m] for m in range(M)}

    return {
        "data": data,
        "Z": Z,
        "W": W_list,
        "alphas": alphas,
        "taus": taus,
        "means": F_list,
        "likelihood": "gaussian",
        "missingness": {
            "mcar_missing_frac": mcar_missing_frac,
            "mcar_whole_view_frac": mcar_whole_view_frac,
        },
        "rng_seed": seed,
        "hyperparams": {
            "a_alpha": a_alpha, "b_alpha": b_alpha,
            "a_tau": a_tau, "b_tau": b_tau,
            "K": K, "M": M, "D": D,
            "noise_std": noise_std,
        },
        "notes": "Gaussian likelihood + MGPS column shrinkage with ARD (φ removed)."
    }
