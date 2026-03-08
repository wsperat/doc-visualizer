from __future__ import annotations

from collections.abc import Sequence

import pytest

from doc_visualizer.phase2.models import EmbeddingInput, Vector
from doc_visualizer.phase4.models import StrategyDocument
from doc_visualizer.phase5.service import build_strategy_map


class MappingEmbeddingBackend:
    def __init__(self, vectors: dict[str, Vector]) -> None:
        self._vectors = dict(vectors)

    @property
    def name(self) -> str:
        return "mapping_embed"

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._vectors[text] for text in texts]


class LinearReducer:
    @property
    def name(self) -> str:
        return "linear"

    def reduce(self, vectors: Sequence[Vector]) -> list[tuple[float, float]]:
        return [(float(index), float(vector[0])) for index, vector in enumerate(vectors)]


class ThresholdClusterer:
    @property
    def name(self) -> str:
        return "threshold"

    def cluster(self, points: Sequence[tuple[float, float]]) -> list[int]:
        return [0 if y >= 1.5 else -1 for _, y in points]


def test_build_strategy_map_includes_qc_overlay() -> None:
    documents = (
        StrategyDocument(
            source="grobid",
            strategy="whole_doc_mean_pool",
            document_id="a",
            inputs=(EmbeddingInput(section="abstract", text="alpha"),),
            section_weights={},
        ),
        StrategyDocument(
            source="grobid",
            strategy="whole_doc_mean_pool",
            document_id="b",
            inputs=(EmbeddingInput(section="abstract", text="beta"),),
            section_weights={},
        ),
    )

    strategy_map = build_strategy_map(
        documents=documents,
        embedding_backend=MappingEmbeddingBackend({"alpha": (1.0, 0.0), "beta": (2.0, 0.0)}),
        reducer=LinearReducer(),
        clusterer=ThresholdClusterer(),
        parameters={"n_neighbors": 15},
        quality_control={"a": (0.91, False)},
    )

    assert strategy_map.source == "grobid"
    assert strategy_map.strategy == "whole_doc_mean_pool"
    assert len(strategy_map.points) == 2

    point_a, point_b = strategy_map.points
    assert point_a.document_id == "a"
    assert point_a.cluster_id == -1
    assert point_a.similarity_score == pytest.approx(0.91)
    assert point_a.is_below_threshold is False

    assert point_b.document_id == "b"
    assert point_b.cluster_id == 0
    assert point_b.similarity_score is None


def test_build_strategy_map_validates_group_consistency() -> None:
    documents = (
        StrategyDocument(
            source="grobid",
            strategy="whole_doc_mean_pool",
            document_id="a",
            inputs=(EmbeddingInput(section="abstract", text="alpha"),),
            section_weights={},
        ),
        StrategyDocument(
            source="raw",
            strategy="whole_doc_mean_pool",
            document_id="b",
            inputs=(EmbeddingInput(section="abstract", text="beta"),),
            section_weights={},
        ),
    )

    with pytest.raises(ValueError, match="identical source and strategy"):
        _ = build_strategy_map(
            documents=documents,
            embedding_backend=MappingEmbeddingBackend({"alpha": (1.0,), "beta": (2.0,)}),
            reducer=LinearReducer(),
            clusterer=ThresholdClusterer(),
            parameters={},
            quality_control={},
        )
