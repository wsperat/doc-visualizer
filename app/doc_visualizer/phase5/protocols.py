"""Protocols for Phase 5 reduction and clustering engines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from doc_visualizer.phase2.models import Vector

Point2D = tuple[float, float]


class DimensionalityReducer(Protocol):
    """Reducer contract to project high-dimensional vectors to 2D points."""

    @property
    def name(self) -> str:
        """Stable reducer identifier."""

    def reduce(self, vectors: Sequence[Vector]) -> list[Point2D]:
        """Project input vectors to two-dimensional coordinates."""


class Clusterer(Protocol):
    """Clusterer contract for 2D points."""

    @property
    def name(self) -> str:
        """Stable clusterer identifier."""

    def cluster(self, points: Sequence[Point2D]) -> list[int]:
        """Assign one cluster id to each input point."""
