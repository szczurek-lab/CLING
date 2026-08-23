# Reproducibility

Scripts and configuration to reproduce the analyses in the CLING paper. No data
or generated results are committed; every dataset is downloaded from its public
source and every output is produced by running these scripts.

## Contents

- `simulate.py` — generate the synthetic multi-view benchmark (Figures 1–2).
- `run_cling.py` — fit CLING on a set of views (`.npz`) with the paper's
  operating point and report the effective factor count and variance explained.
- `configs/simulation.yaml` — synthetic benchmark settings and sweeps.
- `configs/real_data.yaml` — per-dataset settings for the real analyses.
- `environment.yml` — a conda environment for CLING and these scripts.

## Environment

```bash
conda env create -f reproducibility/environment.yml
conda activate cling-repro
```

or simply `pip install ".[reproducibility]"` from the repository root.

## Simulations (Figures 1–2)

```bash
python reproducibility/simulate.py --out sim.npz --seed 0
python reproducibility/run_cling.py --input sim.npz --k-init 30 --seed 0
```

`simulate.py` writes `view_<m>` arrays plus the ground-truth `Z_true` /
`W_true_<m>`; `run_cling.py` fits CLING and prints the effective `K` and per-view
R². Sweep the parameters in `configs/simulation.yaml` (factor count, noise,
sparsity, sample size) and aggregate over the 25 seeds to reproduce the figures.

## Real datasets

CLING operates on a list of `N × D_m` view matrices (`NaN` = missing). After
downloading and preprocessing each dataset into an `.npz` of `view_<m>` arrays,
fit it with, for example:

```bash
python reproducibility/run_cling.py --input gbm_views.npz --k-init 30 --seed 23
```

Data sources (download and preprocess per `configs/real_data.yaml`):

| Dataset | Reference | Source |
|---|---|---|
| Evo-Devo (bulk RNA-seq, 5 organs, N=83) | Cardoso-Moreira et al., *Nature*, 2019 | [MEFISTO analyses](https://github.com/bioFAM/MEFISTO_analyses) |
| scNMT-seq (mouse gastrulation, 3 modalities, N=1518) | Argelaguet et al., *Nature*, 2019 | [GEO GSE121708](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE121708) |
| 10x Multiome PBMC (RNA + ATAC, 12,012 cells) | 10x Genomics | [10x Genomics datasets](https://www.10xgenomics.com/datasets) |
| TCGA-GBM (expression + methylation, N=278) | Brennan et al., *Cell*, 2013 | [cBioPortal](https://www.cbioportal.org/study/summary?id=gbm_tcga) |

## Baselines

The paper compares CLING against MOFA (`mofapy2` 0.7.2), MuVI (0.1.5), PCA
(scikit-learn), a Python port of MOJITOO, Multigrate (1.0.1), and — on GBM
only — the semi-supervised SOFA. These pin incompatible dependency sets; install
each in its own environment from its upstream project (see `environment.yml`).
CLING itself needs only `numpy`, `scipy`, and `tqdm`.
