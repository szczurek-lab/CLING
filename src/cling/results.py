"""Containers for fitted models and training summaries."""

from __future__ import annotations

import dataclasses
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import __version__
from .config import (
    CLINGHyperparameters,
    LatentARDHyperparameters,
    LocalPrecisionHyperparameters,
    NoiseHyperparameters,
    ShrinkageHyperparameters,
    TrainingOptions,
)

# Display-name variant identifiers recorded in snapshots.
_VARIANT_BY_CLASS = {
    "CLINGModel": "CLING",
    "CLINGModelMGP": "CLING-MGP",
    "CLINGModelARD": "CLING-ARD",
}


def _safe_version(pkg_name: str) -> Optional[str]:
    """Look up a package version without raising if it isn't installed."""
    try:
        return version(pkg_name)
    except PackageNotFoundError:
        return None


def _hp_to_dict(hp: Optional[CLINGHyperparameters]) -> Optional[Dict[str, Any]]:
    """Serialise a :class:`CLINGHyperparameters` bundle to a JSON-able dict."""
    if hp is None:
        return None
    return dataclasses.asdict(hp)


def _hp_from_dict(d: Optional[Dict[str, Any]]) -> Optional[CLINGHyperparameters]:
    """Reverse of :func:`_hp_to_dict`; ``latent_ard`` defaults when absent."""
    if not d:
        return None
    latent_ard_d = d.get("latent_ard")
    if latent_ard_d:
        latent_ard = LatentARDHyperparameters(**latent_ard_d)
    else:
        latent_ard = LatentARDHyperparameters()
    return CLINGHyperparameters(
        shrinkage=ShrinkageHyperparameters(**d["shrinkage"]),
        precision=LocalPrecisionHyperparameters(**d["precision"]),
        noise=NoiseHyperparameters(**d["noise"]),
        latent_ard=latent_ard,
    )


def _training_options_to_dict(
    opts: Optional[TrainingOptions],
) -> Optional[Dict[str, Any]]:
    if opts is None:
        return None
    return dataclasses.asdict(opts)


def _training_options_from_dict(
    d: Optional[Dict[str, Any]],
) -> Optional[TrainingOptions]:
    if not d:
        return None
    return TrainingOptions(**d)


@dataclass(frozen=True)
class TrainingSummary:
    """Lightweight summary of a training run."""

    n_iterations: int
    final_elbo: float
    converged: bool
    elbo_history: Tuple[float, ...] = field(default_factory=tuple)
    K_history: Tuple[int, ...] = field(default_factory=tuple)


@dataclass
class FittedSnapshot:
    """Post-training read-only view: factors, weights, cached variance-explained
    matrices, and (for grouped fits) group metadata and eta posterior. Mirrors
    the live model's accessor surface; does not support continued training.
    """

    Z: np.ndarray
    W: List[np.ndarray]
    view_names: Tuple[str, ...]
    variant: str
    R2_view: np.ndarray
    R2_factor: np.ndarray
    R2_factor_view: np.ndarray
    G: int = 1
    group_ix: Optional[np.ndarray] = None
    group_names: Optional[Tuple[str, ...]] = None
    eta_vi_a: Optional[np.ndarray] = None
    eta_vi_b: Optional[np.ndarray] = None
    R2_factor_view_group: Optional[np.ndarray] = None

    @property
    def K(self) -> int:
        return int(self.Z.shape[1])

    @property
    def N(self) -> int:
        return int(self.Z.shape[0])

    @property
    def M(self) -> int:
        return len(self.W)

    @property
    def D(self) -> List[int]:
        return [int(w.shape[0]) for w in self.W]

    # --- accessors (mirror the live model API) ------------------------
    def get_factors(self) -> np.ndarray:
        return self.Z

    def get_weights(self, view: Optional[int] = None):
        if view is None:
            return list(self.W)
        return self.W[view]

    def reconstruct(self, view: Optional[int] = None):
        if view is None:
            return [self.Z @ wm.T for wm in self.W]
        return self.Z @ self.W[view].T

    def variance_explained_per_view(self) -> np.ndarray:
        return self.R2_view

    def variance_explained_per_factor(self) -> np.ndarray:
        return self.R2_factor

    def variance_explained_per_factor_view(self) -> np.ndarray:
        return self.R2_factor_view

    def variance_explained_per_factor_view_group(self) -> Optional[np.ndarray]:
        """Persisted ``(K, G, M)`` R^2 array, or ``None`` for older archives."""
        return self.R2_factor_view_group


@dataclass
class FittedModel:
    """Post-training container exposing factors, loadings, and diagnostics.

    Beyond ``model`` / ``training`` / ``seed`` it stores enough provenance to
    reproduce the fit (``K_init``, ``init_mode``, ``center``, ``scale_views``,
    ``training_options``, ``hp``, and the group-aware ``groups`` /
    ``center_groups`` / resolved ``ard_factors``). All extra fields default to
    ``None`` so older archives still load.
    """

    model: Any
    training: TrainingSummary
    seed: Optional[int] = None
    K_init: Optional[int] = None
    init_mode: Optional[str] = None
    center: Optional[bool] = None
    scale_views: Optional[bool] = None
    training_options: Optional[TrainingOptions] = None
    hp: Optional[CLINGHyperparameters] = None
    groups: Optional[Tuple[Any, ...]] = None
    center_groups: Optional[bool] = None
    ard_factors: Optional[bool] = None

    @property
    def K(self) -> int:
        return int(self.model.K)

    @property
    def M(self) -> int:
        return int(self.model.M)

    @property
    def N(self) -> int:
        return int(self.model.N)

    @property
    def D(self) -> List[int]:
        return list(self.model.D)

    @property
    def G(self) -> int:
        """Number of groups in the fitted dataset (``1`` for ungrouped)."""
        return int(getattr(self.model, "G", 1))

    @property
    def view_names(self) -> Tuple[str, ...]:
        """Human-readable view names, from the dataset (live) or snapshot."""
        ds = getattr(self.model, "dataset", None)
        if ds is not None and getattr(ds, "view_names", None) is not None:
            return tuple(ds.view_names)
        names = getattr(self.model, "view_names", None)
        if names is not None:
            return tuple(names)
        return tuple(f"view_{m}" for m in range(self.M))

    @property
    def group_names(self) -> Optional[Tuple[str, ...]]:
        """Canonical group names, or ``None`` at ``G = 1``."""
        if self.G <= 1:
            return None
        ds = getattr(self.model, "dataset", None)
        if ds is not None and getattr(ds, "group_names", None) is not None:
            return tuple(ds.group_names)
        snap_names = getattr(self.model, "group_names", None)
        if snap_names is not None:
            return tuple(snap_names)
        return None

    # --- factor / weight accessors -----------------------------------
    def get_factors(self) -> np.ndarray:
        return self.model.get_factors()

    def get_weights(self, view: Optional[int] = None):
        return self.model.get_weights(view)

    def reconstruct(self, view: Optional[int] = None):
        return self.model.reconstruct(view)

    # --- variance explained ------------------------------------------
    def variance_explained_per_view(self) -> np.ndarray:
        return self.model.variance_explained_per_view()

    def variance_explained_per_factor(self) -> np.ndarray:
        return self.model.variance_explained_per_factor()

    def variance_explained_per_factor_view(self) -> np.ndarray:
        return self.model.variance_explained_per_factor_view()

    def variance_explained_per_factor_view_group(self) -> Optional[np.ndarray]:
        """Per-(factor, group, view) R^2 array ``(K, G, M)``; ``None`` only for
        snapshots loaded from a pre-grouping archive."""
        return self.model.variance_explained_per_factor_view_group()

    # --- factor selection --------------------------------------------
    def active_factor_mask(self, epsilon: float = 0.01) -> np.ndarray:
        """Boolean mask of factors kept by the per-view R^2 >= epsilon rule."""
        from .inference import find_inactive_factors

        r2 = self.variance_explained_per_factor_view_group()
        if r2 is None:
            r2 = self.variance_explained_per_factor_view()
        return ~find_inactive_factors(np.asarray(r2), epsilon)

    def n_active_factors(self, epsilon: float = 0.01, min_factors: int = 1) -> int:
        """Number of active factors (per-view R^2 >= epsilon in any view)."""
        return max(int(self.active_factor_mask(epsilon).sum()), int(min_factors))

    # --- eta posterior ------------------------------------------------
    def get_eta(self) -> Optional[np.ndarray]:
        """Posterior mean ``E[eta_{g, k}]`` of shape ``(G, K)``, or ``None`` if
        the fit did not wire in the group-factor ARD precision."""
        eta_node = getattr(self.model, "eta", None)
        if eta_node is not None:
            return np.asarray(eta_node.E_eta)
        a = getattr(self.model, "eta_vi_a", None)
        b = getattr(self.model, "eta_vi_b", None)
        if a is not None and b is not None:
            return np.asarray(a) / np.asarray(b)
        return None

    @classmethod
    def from_model(
        cls,
        model: Any,
        summary: TrainingSummary,
        *,
        seed: Optional[int] = None,
        K_init: Optional[int] = None,
        init_mode: Optional[str] = None,
        training_options: Optional[TrainingOptions] = None,
    ) -> FittedModel:
        """Construct from a live model, auto-pulling ``hp`` and the
        preprocessing / group metadata from ``model.dataset``. ``ard_factors``
        is recorded as the *resolved* value (presence of an eta node)."""
        ds = getattr(model, "dataset", None)
        center = getattr(ds, "center", None) if ds is not None else None
        scale_views = getattr(ds, "scale_views", None) if ds is not None else None

        groups_raw = getattr(ds, "groups", None) if ds is not None else None
        groups_tuple: Optional[Tuple[Any, ...]] = (
            tuple(groups_raw) if groups_raw is not None else None
        )
        center_groups = (
            getattr(ds, "center_groups", None) if ds is not None else None
        )
        ard_factors_resolved = getattr(model, "eta", None) is not None

        return cls(
            model=model,
            training=summary,
            seed=seed,
            K_init=K_init,
            init_mode=init_mode,
            center=center,
            scale_views=scale_views,
            training_options=training_options,
            hp=getattr(model, "hp", None),
            groups=groups_tuple,
            center_groups=center_groups,
            ard_factors=ard_factors_resolved,
        )

    # --- persistence --------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Persist the fitted model to a ``.npz`` archive: the factors, per-view
        loadings, variance-explained matrices, training trace, and (when
        present) group index and eta posterior, plus a JSON metadata blob with
        variant, sizes, seed, preprocessing flags, hyperparameters, training
        options, group metadata, and an environment fingerprint.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")

        Z = np.asarray(self.get_factors())
        W = self.get_weights()
        R2_view = np.asarray(self.variance_explained_per_view())
        R2_factor = np.asarray(self.variance_explained_per_factor())
        R2_factor_view = np.asarray(self.variance_explained_per_factor_view())
        r2_kgm_raw = self.variance_explained_per_factor_view_group()
        R2_factor_view_group: Optional[np.ndarray] = (
            np.asarray(r2_kgm_raw) if r2_kgm_raw is not None else None
        )

        variant = _VARIANT_BY_CLASS.get(
            type(self.model).__name__,
            getattr(self.model, "variant", "unknown"),
        )
        view_names: Optional[List[str]] = None
        view_scales: Optional[List[Optional[float]]] = None
        if hasattr(self.model, "dataset") and self.model.dataset is not None:
            view_names = list(self.model.dataset.view_names)
            raw_scales = getattr(self.model.dataset, "view_scales", None)
            if raw_scales is not None:
                view_scales = [
                    float(s) if s is not None else None for s in raw_scales
                ]
        elif hasattr(self.model, "view_names"):
            view_names = list(self.model.view_names)

        G = self.G
        group_names_list: Optional[List[str]] = None
        ds = getattr(self.model, "dataset", None)
        if ds is not None and getattr(ds, "group_names", None) is not None:
            group_names_list = [str(name) for name in ds.group_names]
        elif getattr(self.model, "group_names", None) is not None:
            group_names_list = [str(name) for name in self.model.group_names]
        groups_for_json: Optional[List[str]] = None
        if self.groups is not None:
            groups_for_json = [str(g) for g in self.groups]

        metadata = {
            "cling_version": __version__,
            "variant": variant,
            "view_names": view_names,
            "K": self.K,
            "K_init": self.K_init,
            "M": self.M,
            "N": self.N,
            "D": list(self.D),
            "seed": self.seed,
            "init_mode": self.init_mode,
            "center": self.center,
            "scale_views": self.scale_views,
            "view_scales": view_scales,
            "hp": _hp_to_dict(self.hp),
            "training_options": _training_options_to_dict(self.training_options),
            "G": G,
            "groups": groups_for_json,
            "group_names": group_names_list,
            "center_groups": self.center_groups,
            "ard_factors": self.ard_factors,
            "training": {
                "n_iterations": self.training.n_iterations,
                "final_elbo": self.training.final_elbo,
                "converged": self.training.converged,
            },
            "library_versions": {
                "cling": __version__,
                "numpy": _safe_version("numpy"),
                "scipy": _safe_version("scipy"),
                "python": sys.version.split()[0],
            },
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        arrays: dict[str, np.ndarray] = {
            "Z": Z,
            "R2_view": R2_view,
            "R2_factor": R2_factor,
            "R2_factor_view": R2_factor_view,
            "elbo_history": np.asarray(self.training.elbo_history, dtype=float),
            "K_history": np.asarray(self.training.K_history, dtype=int),
            "metadata_json": np.asarray(json.dumps(metadata)),
        }
        for m, Wm in enumerate(W):
            arrays[f"W_{m}"] = np.asarray(Wm)

        if R2_factor_view_group is not None:
            arrays["R2_factor_view_group"] = R2_factor_view_group

        if G > 1:
            group_ix_arr: Optional[np.ndarray] = None
            if ds is not None and getattr(ds, "group_ix", None) is not None:
                group_ix_arr = np.asarray(ds.group_ix, dtype=np.int64)
            elif getattr(self.model, "group_ix", None) is not None:
                group_ix_arr = np.asarray(self.model.group_ix, dtype=np.int64)
            if group_ix_arr is not None:
                arrays["group_ix"] = group_ix_arr

        eta_node = getattr(self.model, "eta", None)
        if eta_node is not None:
            arrays["eta_vi_a"] = np.asarray(eta_node.vi_a, dtype=float)
            arrays["eta_vi_b"] = np.asarray(eta_node.vi_b, dtype=float)

        np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
        return path

    @classmethod
    def load(cls, path: str | Path) -> FittedModel:
        """Reconstruct a :class:`FittedModel` from a ``.npz`` archive; the
        returned ``.model`` is a read-only :class:`FittedSnapshot`. Missing
        reproducibility fields default to ``None`` for older archives."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")

        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            Z = np.asarray(data["Z"])
            M = int(metadata["M"])
            W = [np.asarray(data[f"W_{m}"]) for m in range(M)]
            R2_view = np.asarray(data["R2_view"])
            R2_factor = np.asarray(data["R2_factor"])
            R2_factor_view = np.asarray(data["R2_factor_view"])
            elbo_history = tuple(np.asarray(data["elbo_history"]).tolist())
            K_history = tuple(int(k) for k in np.asarray(data["K_history"]).tolist())
            files = set(data.files)
            group_ix_loaded: Optional[np.ndarray] = (
                np.asarray(data["group_ix"], dtype=np.int64)
                if "group_ix" in files
                else None
            )
            eta_vi_a_loaded: Optional[np.ndarray] = (
                np.asarray(data["eta_vi_a"], dtype=float)
                if "eta_vi_a" in files
                else None
            )
            eta_vi_b_loaded: Optional[np.ndarray] = (
                np.asarray(data["eta_vi_b"], dtype=float)
                if "eta_vi_b" in files
                else None
            )
            R2_factor_view_group_loaded: Optional[np.ndarray] = (
                np.asarray(data["R2_factor_view_group"], dtype=float)
                if "R2_factor_view_group" in files
                else None
            )

        view_names = metadata.get("view_names")
        if view_names is None:
            view_names = tuple(f"view_{m}" for m in range(M))
        else:
            view_names = tuple(view_names)

        G_meta = int(metadata.get("G", 1))
        group_names_meta = metadata.get("group_names")
        group_names_tup: Optional[Tuple[str, ...]] = (
            tuple(group_names_meta) if group_names_meta is not None else None
        )
        groups_meta = metadata.get("groups")
        groups_tup: Optional[Tuple[Any, ...]] = (
            tuple(groups_meta) if groups_meta is not None else None
        )
        center_groups_val = metadata.get("center_groups")
        center_groups = (
            bool(center_groups_val) if center_groups_val is not None else None
        )
        ard_factors_val = metadata.get("ard_factors")
        ard_factors = bool(ard_factors_val) if ard_factors_val is not None else None

        snap = FittedSnapshot(
            Z=Z,
            W=W,
            view_names=view_names,
            variant=str(metadata.get("variant", "unknown")),
            R2_view=R2_view,
            R2_factor=R2_factor,
            R2_factor_view=R2_factor_view,
            G=G_meta,
            group_ix=group_ix_loaded,
            group_names=group_names_tup,
            eta_vi_a=eta_vi_a_loaded,
            eta_vi_b=eta_vi_b_loaded,
            R2_factor_view_group=R2_factor_view_group_loaded,
        )

        summary_dict = metadata.get("training", {})
        summary = TrainingSummary(
            n_iterations=int(summary_dict.get("n_iterations", len(elbo_history))),
            final_elbo=float(
                summary_dict.get(
                    "final_elbo", elbo_history[-1] if elbo_history else float("nan")
                )
            ),
            converged=bool(summary_dict.get("converged", False)),
            elbo_history=elbo_history,
            K_history=K_history,
        )

        K_init_val = metadata.get("K_init")
        K_init = int(K_init_val) if K_init_val is not None else None
        init_mode = metadata.get("init_mode")
        if init_mode is not None:
            init_mode = str(init_mode)
        center_val = metadata.get("center")
        center = bool(center_val) if center_val is not None else None
        scale_views_val = metadata.get("scale_views")
        scale_views = bool(scale_views_val) if scale_views_val is not None else None

        return cls(
            model=snap,
            training=summary,
            seed=metadata.get("seed"),
            K_init=K_init,
            init_mode=init_mode,
            center=center,
            scale_views=scale_views,
            training_options=_training_options_from_dict(
                metadata.get("training_options")
            ),
            hp=_hp_from_dict(metadata.get("hp")),
            groups=groups_tup,
            center_groups=center_groups,
            ard_factors=ard_factors,
        )


__all__ = ["FittedModel", "FittedSnapshot", "TrainingSummary"]
