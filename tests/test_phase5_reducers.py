from __future__ import annotations

from doc_visualizer.phase5.reducers import UmapReducer


def test_umap_reducer_returns_origin_for_single_vector() -> None:
    reducer = UmapReducer()
    points = reducer.reduce([(1.0, 2.0, 3.0)])
    assert points == [(0.0, 0.0)]


def test_umap_reducer_returns_split_layout_for_two_vectors() -> None:
    reducer = UmapReducer()
    points = reducer.reduce([(1.0, 0.0), (0.0, 1.0)])
    assert points == [(-1.0, 0.0), (1.0, 0.0)]
