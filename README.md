# CLING

[![CI](https://github.com/szczurek-lab/CLING/actions/workflows/ci.yml/badge.svg)](https://github.com/szczurek-lab/CLING/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/1343957839.svg)](https://doi.org/10.5281/zenodo.22072726)

**CLING** (Cross-view Latent Integration via Nonparametric Gamma Shrinkage) is an unsupervised Bayesian multi-view factor model for integrating large, noisy, and heterogeneous multi-omics data.

## Installation

Install from the repository:

```bash
pip install "git+https://github.com/szczurek-lab/CLING.git"
```

or clone the repository and run `pip install .`. Runtime dependencies: `numpy`,
`scipy`, `tqdm`. Optional extras: `.[tutorial]`, `.[testing]`,
`.[reproducibility]`.

## Input

Pass a list of `M` view matrices on the same `N` samples; view `m` is `N × D_m` (samples × features), with `D_m` varying across views. NaNs are treated as missing under a mask, so no imputation is needed, though each feature needs at least one observed value.

## Usage

```python
import numpy as np
import cling

rng = np.random.default_rng(0)
views = [rng.standard_normal((100, 80)),
         rng.standard_normal((100, 60)),
         rng.standard_normal((100, 40))]
views[0][5, 3] = np.nan                      # missing entries are allowed

fitted = cling.fit(views, K_init=30, seed=0, view_names=["rna", "atac", "met"])

Z  = fitted.get_factors()                    # (N, K) shared latent scores
W  = fitted.get_weights()                    # list of (D_m, K) loadings
R2 = fitted.variance_explained_per_view()    # (M,) per-view variance explained
print(fitted.n_active_factors())             # number of active factors

fitted.save("my_fit.npz")
reloaded = cling.FittedModel.load("my_fit.npz")
```

The returned `FittedModel` also exposes `variance_explained_per_factor()`, `variance_explained_per_factor_view()`, `reconstruct()`, and the `training.elbo_history` / `training.K_history` traces.

## Key parameters and defaults

- `K_init` (default `30`): the overcomplete truncation ceiling used in the paper. It bounds but does not preselect the number of factors.
- Factor selection: a factor is kept if its per-view explained variance is at least `epsilon = 0.01` in one or more views. By default this is applied once after the ELBO converges, and the active count is `fitted.n_active_factors()`. Passing `prune_threshold=` to `fit` instead prunes factors adaptively during inference (after a warm-up).
- Shrinkage (set automatically by `fit` from the sample size): `Gamma(3, 2.5)` for `N < 1000` and `Gamma(3, 1)` for `N >= 1000`.
- Convergence: `TrainingOptions(convergence_mode=...)` with `"fast"`, `"medium"`, or `"slow"` (default `"slow"`, dELBO 5e-6) and `max_iter = 4000`.
- Other defaults: noise `Gamma(1e-3, 1e-3)`; local precision `a = b = 0.1`; local scale `a = 0.5, b = 1.0`; PCA initialisation. Pass `seed=` for a deterministic fit.
