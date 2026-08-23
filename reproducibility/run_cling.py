"""Fit CLING on a set of views stored in an ``.npz`` file, using the paper's
operating point, and report the effective factor count and variance explained.

The input archive must contain arrays named ``view_0``, ``view_1``, ... each of
shape ``(N, D_m)`` (``NaN`` = missing). This is the format written by
``simulate.py`` and the recommended format for real datasets after
preprocessing.

Usage:
    python reproducibility/run_cling.py --input sim.npz --output fit.npz
    python reproducibility/run_cling.py --input data.npz --variant CLING-MGP \
        --k-init 30 --seed 23
"""

from __future__ import annotations

import argparse

import numpy as np

import cling


def load_views(path):
    with np.load(path, allow_pickle=False) as data:
        keys = sorted(
            (k for k in data.files if k.startswith("view_")),
            key=lambda s: int(s.split("_")[1]),
        )
        if not keys:
            raise ValueError(
                f"{path} contains no 'view_<m>' arrays; found {sorted(data.files)}."
            )
        return [np.asarray(data[k], dtype=float) for k in keys]


def main():
    p = argparse.ArgumentParser(description="Fit CLING on multi-view .npz data.")
    p.add_argument("--input", required=True, help="input .npz with view_<m> arrays")
    p.add_argument("--output", default=None, help="optional .npz to save the fit")
    p.add_argument("--variant", default="CLING",
                   choices=["CLING", "CLING-MGP", "CLING-ARD"])
    p.add_argument("--k-init", type=int, default=30,
                   help="overcomplete truncation ceiling (paper: 30)")
    p.add_argument("--epsilon", type=float, default=0.01,
                   help="per-view R^2 threshold for the active-factor count")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-iter", type=int, default=4000)
    args = p.parse_args()

    views = load_views(args.input)
    print(f"loaded {len(views)} views, N={views[0].shape[0]}, "
          f"D={[v.shape[1] for v in views]}")

    fitted = cling.fit(
        views,
        K_init=args.k_init,
        variant=args.variant,
        seed=args.seed,
        max_iter=args.max_iter,
    )

    r2_factor = fitted.variance_explained_per_factor()
    k_eff = int((r2_factor >= args.epsilon).sum())
    print(f"variant={args.variant} seed={args.seed}")
    print(f"K_eff (R^2 >= {args.epsilon}) = {k_eff}")
    print(f"final ELBO = {fitted.training.final_elbo:.4f} "
          f"(converged={fitted.training.converged}, "
          f"iters={fitted.training.n_iterations})")
    print(f"per-view variance explained = "
          f"{np.round(fitted.variance_explained_per_view(), 4).tolist()}")

    if args.output:
        fitted.save(args.output)
        print(f"saved fit to {args.output}")


if __name__ == "__main__":
    main()
