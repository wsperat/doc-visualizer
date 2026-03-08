"""Core service logic for Phase 5 document map generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from doc_visualizer.phase2.models import Vector
from doc_visualizer.phase4.models import StrategyDocument
from doc_visualizer.phase4.protocols import EmbeddingBackend
from doc_visualizer.phase4.service import build_document_vector
from doc_visualizer.phase5.models import MapPoint, StrategyMap
from doc_visualizer.phase5.protocols import Clusterer, DimensionalityReducer


def build_strategy_map(
    *,
    documents: Sequence[StrategyDocument],
    embedding_backend: EmbeddingBackend,
    reducer: DimensionalityReducer,
    clusterer: Clusterer,
    parameters: Mapping[str, object],
    quality_control: Mapping[str, tuple[float, bool]],
) -> StrategyMap:
    """Generate one projected and clustered map for a strategy corpus."""
    if not documents:
        raise ValueError("build_strategy_map requires at least one document")

    first_document = documents[0]
    _validate_single_group(documents, first_document.source, first_document.strategy)

    vectors = _build_vectors(documents=documents, embedding_backend=embedding_backend)
    points = reducer.reduce(vectors)
    labels = clusterer.cluster(points)

    if len(points) != len(documents):
        raise ValueError("Reducer returned unexpected number of points")
    if len(labels) != len(documents):
        raise ValueError("Clusterer returned unexpected number of labels")

    map_points: list[MapPoint] = []
    for document, (x, y), cluster_id in zip(documents, points, labels, strict=True):
        qc_record = quality_control.get(document.document_id)
        similarity_score = qc_record[0] if qc_record is not None else None
        is_below_threshold = qc_record[1] if qc_record is not None else None

        map_points.append(
            MapPoint(
                document_id=document.document_id,
                x=x,
                y=y,
                cluster_id=int(cluster_id),
                similarity_score=similarity_score,
                is_below_threshold=is_below_threshold,
            )
        )

    return StrategyMap(
        source=first_document.source,
        strategy=first_document.strategy,
        reducer=reducer.name,
        clusterer=clusterer.name,
        parameters=dict(parameters),
        points=tuple(map_points),
    )


def _build_vectors(
    *,
    documents: Sequence[StrategyDocument],
    embedding_backend: EmbeddingBackend,
) -> list[Vector]:
    vectors: list[Vector] = []
    for document in documents:
        vectors.append(
            build_document_vector(
                strategy_name=document.strategy,
                inputs=document.inputs,
                section_weights=document.section_weights,
                embedding_backend=embedding_backend,
            )
        )
    return vectors


def _validate_single_group(
    documents: Sequence[StrategyDocument],
    source: str,
    strategy: str,
) -> None:
    for document in documents:
        if document.source != source or document.strategy != strategy:
            raise ValueError("All documents must share identical source and strategy")
