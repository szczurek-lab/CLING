"""Generate the paper's synthetic multi-view benchmark data.

The generator deliberately departs from CLING's own prior (no cumulative column
shrinkage; sparsity via an element-wise Bernoulli mask), so recovery is not
biased toward CLING. For each view m and each (n, d, k):

    Z[n, k]        ~ N(0, 1)
    alpha[d, k]    ~ Gamma(1, 1)
    S[d, k]        ~ Bernoulli(1 - sparsity)
    W_hat[d, k]    ~ N(0, 1 / alpha[d, k])
    W[d, k]        = S[d, k] * W_hat[d, k]
    Y[n, :]        = Z @ W.T + sigma * N(0, 1)

Baseline settings match the paper: M = 3, D = (1000, 1500, 2000), N = 300,
K = 10, sigma = 0.75, sparsity = 0.60.

Usage:
    python reproducibility/simulate.py --out sim.npz --seed 0
    python reproducibility/simulate.py --n 300 --dims 1000,1500,2000 \
        --k 10 --sigma 0.75 --sparsity 0.60 --seed 0 --out sim.npz
"""

from __future__ import annotations

import argparse

import numpy as np


def simulate(n, dims, k, sigma, sparsity, seed):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, k))
    views, weights, supports = [], [], []
    for d in dims:
        alpha = rng.gamma(shape=1.0, scale=1.0, size=(d, k))
        support = (rng.random((d, k)) < (1.0 - sparsity)).astype(float)
        w_hat = rng.standard_normal((d, k)) / np.sqrt(alpha)
        w = support * w_hat
        y = Z @ w.T + sigma * rng.standard_normal((n, d))
        views.append(y)
        weights.append(w)
        supports.append(support)
    return Z, views, weights, supports


def main():
    p = argparse.ArgumentParser(description="Simulate CLING benchmark data.")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--dims", type=str, default="1000,1500,2000")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--sigma", type=float, default=0.75)
    p.add_argument("--sparsity", type=float, default=0.60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    dims = [int(x) for x in args.dims.split(",")]
    Z, views, weights, _ = simulate(
        args.n, dims, args.k, args.sigma, args.sparsity, args.seed
    )

    arrays = {"Z_true": Z}
    for m, (v, w) in enumerate(zip(views, weights)):
        arrays[f"view_{m}"] = v
        arrays[f"W_true_{m}"] = w
    np.savez_compressed(args.out, **arrays)
    print(
        f"wrote {args.out}: N={args.n}, M={len(dims)}, D={dims}, "
        f"K={args.k}, sigma={args.sigma}, sparsity={args.sparsity}, seed={args.seed}"
    )


if __name__ == "__main__":
    main()
