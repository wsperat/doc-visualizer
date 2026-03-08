"""Clusterer implementations for Phase 5."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Protocol, cast

from doc_visualizer.phase5.protocols import Clusterer, Point2D


class _HdbscanLike(Protocol):
    def fit_predict(self, x: Sequence[Sequence[float]]) -> Sequence[int]: ...


class HdbscanClusterer:
    """HDBSCAN clusterer over 2D map points."""

    def __init__(
        self,
        *,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
    ) -> None:
        if min_cluster_size <= 1:
            raise ValueError("min_cluster_size must be greater than 1")
        if min_samples is not None and min_samples <= 0:
            raise ValueError("min_samples must be positive when provided")

        self._min_cluster_size = min_cluster_size
        self._min_samples = min_samples

    @property
    def name(self) -> str:
        return "hdbscan"

    def cluster(self, points: Sequence[Point2D]) -> list[int]:
        if not points:
            raise ValueError("HDBSCAN requires at least one point")

        if len(points) < self._min_cluster_size:
            return [-1] * len(points)

        matrix = [[x, y] for x, y in points]
        hdbscan_ctor = _load_callable("hdbscan", "HDBSCAN")
        clusterer = cast(
            _HdbscanLike,
            hdbscan_ctor(
                min_cluster_size=self._min_cluster_size,
                min_samples=self._min_samples,
            ),
        )
        labels = [int(value) for value in clusterer.fit_predict(matrix)]

        if len(labels) != len(points):
            raise ValueError("HDBSCAN returned unexpected number of labels")

        return labels


def build_clusterer(
    clusterer_name: str,
    *,
    min_cluster_size: int,
    min_samples: int | None,
) -> Clusterer:
    """Construct clusterer from CLI-friendly identifier."""
    normalized_name = clusterer_name.strip().lower()

    if normalized_name == "hdbscan":
        return HdbscanClusterer(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )

    raise ValueError(f"Unsupported clusterer '{clusterer_name}'. Available: ['hdbscan']")


def _load_callable(module_name: str, attribute_name: str) -> Callable[..., object]:
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Required dependency '{module_name}' is not installed for this clusterer"
        ) from exc

    target = getattr(module, attribute_name, None)
    if not callable(target):
        raise RuntimeError(
            f"Dependency '{module_name}' is installed but '{attribute_name}' is unavailable"
        )

    return cast(Callable[..., object], target)
