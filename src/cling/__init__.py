"""cling: Cross-view Latent Integration via Nonparametric Gamma Shrinkage."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cling")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .api import build, fit, run
from .config import (
    SAMPLE_SIZE_SWITCH,
    CLINGHyperparameters,
    LatentARDHyperparameters,
    LocalPrecisionHyperparameters,
    ModelOptions,
    NoiseHyperparameters,
    ShrinkageHyperparameters,
    TrainingOptions,
    paper_defaults_for_n,
)
from .data import MultiviewDataset
from .inference import (
    find_inactive_factors,
    prune_inactive_factors,
    run_mfvi,
)
from .model import CLINGModel, build_model
from .model_ard import CLINGModelARD, build_model_ard
from .model_mgp import CLINGModelMGP, build_model_mgp
from .results import FittedModel, FittedSnapshot, TrainingSummary

__all__ = [
    "__version__",
    # public API
    "build",
    "run",
    "fit",
    # models
    "CLINGModel",
    "CLINGModelMGP",
    "CLINGModelARD",
    "build_model",
    "build_model_mgp",
    "build_model_ard",
    # data
    "MultiviewDataset",
    # config
    "CLINGHyperparameters",
    "ShrinkageHyperparameters",
    "LocalPrecisionHyperparameters",
    "NoiseHyperparameters",
    "LatentARDHyperparameters",
    "ModelOptions",
    "TrainingOptions",
    "SAMPLE_SIZE_SWITCH",
    "paper_defaults_for_n",
    # results
    "FittedModel",
    "FittedSnapshot",
    "TrainingSummary",
    # inference helpers
    "run_mfvi",
    "find_inactive_factors",
    "prune_inactive_factors",
]
