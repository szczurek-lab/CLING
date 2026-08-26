"""Light-weight container for multi-view input data."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


class CLINGDataQualityWarning(UserWarning):
    """Soft warning about input-data quality (very high missingness, samples
    empty in a view, per-group all-NaN features). The fit can still proceed;
    silence with ``warnings.simplefilter("ignore", CLINGDataQualityWarning)``.
    """


# Fraction of missing entries in a single view above which a quality warning is
# emitted (sparse-but-fittable views stay silent).
_HIGH_MISSINGNESS_THRESHOLD = 0.8


def _validate_views(views: List[np.ndarray]) -> None:
    """Hard-error checks on the raw view list: non-empty; every view 2-D with a
    shared sample dimension and >= 1 feature; no ``+/-inf`` (NaN is the missing
    sentinel and is allowed); and no globally all-NaN feature column.
    """
    if not views:
        raise ValueError("At least one view is required.")

    N = views[0].shape[0]
    if N < 1:
        raise ValueError(
            f"Views must contain at least one sample, got N={N}."
        )

    for i, v in enumerate(views):
        if v.ndim != 2:
            raise ValueError(f"View {i} must be 2-D, got shape {v.shape}.")
        if v.shape[0] != N:
            raise ValueError(
                f"All views must share the sample dimension; view {i} "
                f"has {v.shape[0]} samples but view 0 has {N}."
            )
        if v.shape[1] < 1:
            raise ValueError(
                f"View {i} has zero features; each view must have D_m >= 1."
            )

        non_nan = ~np.isnan(v)
        finite_or_nan = np.isfinite(v) | ~non_nan
        if not np.all(finite_or_nan):
            n_bad = int(np.sum(~finite_or_nan))
            raise ValueError(
                f"View {i} contains {n_bad} non-finite, non-NaN values "
                f"(e.g. +inf or -inf). Replace these with NaN (for "
                f"missing entries) or with finite numbers before fitting."
            )

        col_finite_count = np.sum(np.isfinite(v), axis=0)
        empty_cols = np.where(col_finite_count == 0)[0]
        if empty_cols.size > 0:
            preview = empty_cols[:5].tolist()
            more = (
                f" (and {empty_cols.size - 5} more)"
                if empty_cols.size > 5
                else ""
            )
            raise ValueError(
                f"View {i} has {empty_cols.size} all-NaN feature(s) at "
                f"indices {preview}{more}. Drop these features or fill "
                f"in some observations before fitting."
            )


def _emit_quality_warnings(
    views: List[np.ndarray], view_names: Sequence[str]
) -> None:
    """Soft warnings about input quality that do not break the fit."""
    for v, name in zip(views, view_names):
        total = int(v.size)
        if total == 0:
            continue
        n_missing = int(np.sum(np.isnan(v)))
        frac_missing = n_missing / total
        if frac_missing >= _HIGH_MISSINGNESS_THRESHOLD:
            warnings.warn(
                f"View {name!r} is {100 * frac_missing:.1f}% missing; "
                f"the fit may be unstable. Consider dropping this view "
                f"or imputing the dominant features.",
                CLINGDataQualityWarning,
                stacklevel=3,
            )

        sample_finite_count = np.sum(np.isfinite(v), axis=1)
        empty_samples = np.where(sample_finite_count == 0)[0]
        if empty_samples.size > 0:
            preview = empty_samples[:5].tolist()
            more = (
                f" (and {empty_samples.size - 5} more)"
                if empty_samples.size > 5
                else ""
            )
            warnings.warn(
                f"View {name!r} has {empty_samples.size} sample(s) "
                f"entirely missing at row indices {preview}{more}; "
                f"these contribute nothing to the corresponding "
                f"view-specific likelihood.",
                CLINGDataQualityWarning,
                stacklevel=3,
            )


def _process_groups(
    groups: Optional[Sequence],
    N: int,
) -> Tuple[int, np.ndarray, Tuple[str, ...], np.ndarray, List[np.ndarray]]:
    """Resolve ``groups`` to ``(G, group_ix, group_names, N_per_group,
    group_member_indices)``. ``groups=None`` gives the single-group case."""
    if groups is None:
        return (
            1,
            np.zeros(N, dtype=np.int64),
            ("group_0",),
            np.array([N], dtype=np.int64),
            [np.arange(N, dtype=np.int64)],
        )

    groups_arr = np.asarray(groups)
    if groups_arr.ndim != 1 or groups_arr.shape[0] != N:
        raise ValueError(
            f"groups must be a 1-D sequence of length N={N}; "
            f"got shape {groups_arr.shape}."
        )

    unique_names, group_ix = np.unique(groups_arr, return_inverse=True)
    G = int(unique_names.shape[0])
    group_ix = group_ix.astype(np.int64)
    group_names = tuple(str(name) for name in unique_names)
    N_per_group = np.bincount(group_ix, minlength=G).astype(np.int64)
    group_member_indices = [
        np.where(group_ix == g)[0].astype(np.int64) for g in range(G)
    ]
    return G, group_ix, group_names, N_per_group, group_member_indices


def _center_view_per_group(
    v: np.ndarray,
    G: int,
    group_member_indices: List[np.ndarray],
    view_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """NaN-aware per-group per-feature centering for one view. All-NaN
    ``(group, feature)`` cells are filled with the across-group mean (with a
    quality warning); the all-NaN entries themselves stay NaN so the mask drops
    them from the likelihood. Returns ``(centered, feature_means (G, D_m))``.
    """
    Dm = v.shape[1]
    feature_means_gd = np.empty((G, Dm), dtype=float)
    for g in range(G):
        idx = group_member_indices[g]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            feature_means_gd[g] = np.nanmean(v[idx], axis=0)

    nan_mask = np.isnan(feature_means_gd)
    if np.any(nan_mask):
        n_problematic = int(np.any(nan_mask, axis=0).sum())
        warnings.warn(
            f"View {view_name!r}: {n_problematic} feature(s) are "
            f"all-NaN in at least one group; centering those "
            f"(group, feature) cells with the across-group mean as "
            f"fallback. The mask drops these entries from the "
            f"likelihood, so the fallback only affects predicted "
            f"reconstruction.",
            CLINGDataQualityWarning,
            stacklevel=3,
        )
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            global_mu = np.nanmean(feature_means_gd, axis=0)  # (D_m,)
        if np.any(np.isnan(global_mu)):
            raise ValueError(
                f"View {view_name!r}: at least one feature is all-NaN "
                f"globally; this should have been caught by validation."
            )
        feature_means_gd = np.where(
            nan_mask, global_mu[None, :], feature_means_gd
        )

    centered = v.copy()
    for g in range(G):
        idx = group_member_indices[g]
        centered[idx] = v[idx] - feature_means_gd[g][None, :]
    return centered, feature_means_gd


@dataclass
class MultiviewDataset:
    """Multi-view observations on a shared set of ``N`` samples.

    Parameters
    ----------
    views:
        List of ``M`` arrays of shape ``(N, D_m)``. ``NaN`` entries are treated
        as missing (operated on directly under a mask; no imputation required).
    view_names:
        Optional names; defaults to ``["view_0", ...]``.
    center:
        Subtract a per-feature mean from each view (default ``True``). This is
        CLING's required preprocessing.
    scale_views:
        After centering, divide each view by its overall RMS (default
        ``False``). Always global.
    groups:
        Optional length-``N`` sequence of group labels for the grouped
        extension. ``None`` (default) is the single-group case and reproduces
        the plain multi-view model exactly.
    center_groups:
        With ``G > 1``, subtract the per-group per-feature mean (default
        ``True``); a no-op at ``G = 1``.

    Attributes set in ``__post_init__``
    -----------------------------------
    centered_views:
        Per-view arrays after centering (and optional scaling); what the model
        fits against.
    feature_means:
        Per-view mean subtracted, ``(D_m,)`` (global) or ``(G, D_m)``
        (per-group), or ``None`` when ``center=False``.
    view_scales:
        Per-view divisor when ``scale_views=True``, else ``None``.
    G, group_ix, group_names, N_per_group, group_member_indices:
        Group bookkeeping (single group when ``groups=None``).
    """

    views: List[np.ndarray]
    view_names: Optional[Sequence[str]] = None
    center: bool = True
    scale_views: bool = False
    groups: Optional[Sequence] = None
    center_groups: bool = True

    # populated post-init
    centered_views: List[np.ndarray] = field(default_factory=list)
    feature_means: List[Optional[np.ndarray]] = field(default_factory=list)
    view_scales: List[Optional[float]] = field(default_factory=list)
    G: int = field(default=1)
    group_ix: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    group_names: Tuple[str, ...] = field(default_factory=tuple)
    N_per_group: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    group_member_indices: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.views = [np.asarray(v, dtype=float) for v in self.views]
        _validate_views(self.views)

        if self.view_names is None:
            self.view_names = tuple(f"view_{m}" for m in range(self.M))
        else:
            if len(self.view_names) != self.M:
                raise ValueError(
                    "view_names length must equal the number of views."
                )
            self.view_names = tuple(self.view_names)

        _emit_quality_warnings(self.views, self.view_names)

        (
            self.G,
            self.group_ix,
            self.group_names,
            self.N_per_group,
            self.group_member_indices,
        ) = _process_groups(self.groups, self.N)

        self.centered_views = []
        self.feature_means = []
        self.view_scales = []
        for m, v in enumerate(self.views):
            if self.center:
                if self.G > 1 and self.center_groups:
                    v_proc, mu = _center_view_per_group(
                        v,
                        self.G,
                        self.group_member_indices,
                        str(self.view_names[m]),
                    )
                    self.feature_means.append(mu)  # (G, D_m)
                else:
                    mu = np.nanmean(v, axis=0)
                    v_proc = v - mu
                    self.feature_means.append(mu)  # (D_m,)
            else:
                v_proc = v.copy()
                self.feature_means.append(None)

            if self.scale_views:
                sq_mean = float(np.nanmean(v_proc ** 2))
                if sq_mean > 0.0 and np.isfinite(sq_mean):
                    view_std = float(np.sqrt(sq_mean))
                    v_proc = v_proc / view_std
                    self.view_scales.append(view_std)
                else:
                    self.view_scales.append(None)
            else:
                self.view_scales.append(None)

            self.centered_views.append(v_proc)

    # --- accessors ----------------------------------------------------
    @property
    def N(self) -> int:
        return self.views[0].shape[0]

    @property
    def M(self) -> int:
        return len(self.views)

    @property
    def D(self) -> List[int]:
        return [v.shape[1] for v in self.views]

    def feature_means_per_group(self, m: int) -> Optional[np.ndarray]:
        """Return ``feature_means[m]`` normalised to shape ``(G, D_m)`` (the
        global mean broadcast across groups when centering was global), or
        ``None`` if ``center=False``."""
        mu = self.feature_means[m]
        if mu is None:
            return None
        if mu.ndim == 1:
            return np.broadcast_to(mu[None, :], (self.G, mu.shape[0])).copy()
        return mu

    # --- builders -----------------------------------------------------
    @classmethod
    def from_arrays(
        cls,
        views: Sequence[np.ndarray],
        view_names: Optional[Sequence[str]] = None,
        center: bool = True,
        scale_views: bool = False,
        groups: Optional[Sequence] = None,
        center_groups: bool = True,
    ) -> MultiviewDataset:
        return cls(
            list(views),
            view_names=view_names,
            center=center,
            scale_views=scale_views,
            groups=groups,
            center_groups=center_groups,
        )


__all__ = ["MultiviewDataset", "CLINGDataQualityWarning"]
