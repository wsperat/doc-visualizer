"""Dimensionality reducers for Phase 5 knowledge mapping."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Protocol, cast

from doc_visualizer.phase2.models import Vector
from doc_visualizer.phase5.protocols import DimensionalityReducer, Point2D


class _UmapLike(Protocol):
    def fit_transform(self, x: Sequence[Sequence[float]]) -> Sequence[Sequence[float]]: ...


class UmapReducer:
    """UMAP reducer with deterministic defaults for repeatable mapping."""

    def __init__(
        self,
        *,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
    ) -> None:
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        if not 0.0 <= min_dist <= 1.0:
            raise ValueError("min_dist must be in the [0.0, 1.0] range")

        self._n_neighbors = n_neighbors
        self._min_dist = min_dist
        self._random_state = random_state

    @property
    def name(self) -> str:
        return "umap"

    def reduce(self, vectors: Sequence[Vector]) -> list[Point2D]:
        if not vectors:
            raise ValueError("UMAP reduction requires at least one vector")

        if len(vectors) < 3:
            return _project_small_sample(len(vectors))

        matrix = [list(vector) for vector in vectors]
        n_components = 2
        max_neighbors = max(2, min(self._n_neighbors, len(vectors) - 1))

        umap_ctor = _load_callable("umap", "UMAP")
        reducer = cast(
            _UmapLike,
            umap_ctor(
                n_components=n_components,
                n_neighbors=max_neighbors,
                min_dist=self._min_dist,
                random_state=self._random_state,
            ),
        )

        coordinates = reducer.fit_transform(matrix)
        return _to_points_2d(coordinates, expected=len(vectors))


def build_reducer(
    reducer_name: str,
    *,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> DimensionalityReducer:
    """Construct reducer from CLI-friendly identifier."""
    normalized_name = reducer_name.strip().lower()

    if normalized_name == "umap":
        return UmapReducer(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )

    raise ValueError(f"Unsupported reducer '{reducer_name}'. Available: ['umap']")


def _project_small_sample(sample_count: int) -> list[Point2D]:
    if sample_count <= 0:
        return []
    if sample_count == 1:
        return [(0.0, 0.0)]
    if sample_count == 2:
        return [(-1.0, 0.0), (1.0, 0.0)]
    raise ValueError("Small-sample projection only supports up to 2 vectors")


def _to_points_2d(
    coordinates: Sequence[Sequence[float]],
    *,
    expected: int,
) -> list[Point2D]:
    points: list[Point2D] = []
    for row in coordinates:
        if len(row) < 2:
            raise ValueError("UMAP returned fewer than two dimensions")
        points.append((float(row[0]), float(row[1])))

    if len(points) != expected:
        raise ValueError("UMAP returned unexpected number of projected points")

    return points


def _load_callable(module_name: str, attribute_name: str) -> Callable[..., object]:
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Required dependency '{module_name}' is not installed for this reducer"
        ) from exc

    target = getattr(module, attribute_name, None)
    if not callable(target):
        raise RuntimeError(
            f"Dependency '{module_name}' is installed but '{attribute_name}' is unavailable"
        )

    return cast(Callable[..., object], target)
