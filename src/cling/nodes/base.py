"""Base interface for variational nodes."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VariationalNode(ABC):
    """Minimal contract for a node in the variational model.

    Each node owns its variational parameters and exposes ``update`` (a CAVI
    step), ``elbo`` (its contribution to the ELBO), and ``prune`` (restrict to
    a subset of factor columns).
    """

    K: int

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Perform one coordinate-ascent update."""

    @abstractmethod
    def elbo(self) -> float:
        """Return this node's ELBO contribution."""

    def prune(self, active_mask: np.ndarray) -> None:
        """Restrict this node to factor columns selected by ``active_mask``.

        The default implementation is a no-op; nodes whose variational
        parameters depend on ``K`` override it.
        """


__all__ = ["VariationalNode"]
