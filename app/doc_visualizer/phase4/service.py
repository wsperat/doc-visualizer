"""Core service logic for Phase 4 summarization + semantic auditing."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from doc_visualizer.phase2.models import ContextStrategy, EmbeddingInput, Vector
from doc_visualizer.phase2.vector_math import mean_pool, weighted_pool
from doc_visualizer.phase4.models import AuditRecord, StrategyDocument
from doc_visualizer.phase4.protocols import EmbeddingBackend, SummarizationBackend
from doc_visualizer.phase4.vector_math import cosine_similarity


class PhaseFourAuditService:
    """Generate summary and semantic-audit record for transformed documents."""

    def __init__(
        self,
        *,
        summarizer: SummarizationBackend,
        embedding_backend: EmbeddingBackend,
        similarity_threshold: float = 0.8,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in the [0.0, 1.0] range")

        self._summarizer = summarizer
        self._embedding_backend = embedding_backend
        self._similarity_threshold = similarity_threshold

    def audit_document(self, document: StrategyDocument) -> AuditRecord:
        """Run summary + similarity audit for one strategy document."""
        document_text = build_document_text(document.inputs)
        summary = self._summarizer.summarize(document_text)

        document_vector = build_document_vector(
            strategy_name=document.strategy,
            inputs=document.inputs,
            section_weights=document.section_weights,
            embedding_backend=self._embedding_backend,
        )
        summary_vector = _embed_one(self._embedding_backend, summary)

        similarity_score = cosine_similarity(document_vector, summary_vector)
        is_below_threshold = similarity_score < self._similarity_threshold

        return AuditRecord(
            source=document.source,
            strategy=document.strategy,
            document_id=document.document_id,
            summary=summary,
            similarity_score=similarity_score,
            similarity_threshold=self._similarity_threshold,
            is_below_threshold=is_below_threshold,
            summary_backend=self._summarizer.name,
            embedding_backend=self._embedding_backend.name,
            input_count=len(document.inputs),
            document_vector_dimension=len(document_vector),
            summary_vector_dimension=len(summary_vector),
        )


def build_document_text(inputs: Sequence[EmbeddingInput]) -> str:
    """Build one deduplicated text body from strategy inputs."""
    ordered_unique_chunks: list[str] = []
    seen_chunks: set[str] = set()

    for item in inputs:
        normalized_chunk = item.text.strip()
        if not normalized_chunk:
            continue
        if normalized_chunk in seen_chunks:
            continue
        seen_chunks.add(normalized_chunk)
        ordered_unique_chunks.append(normalized_chunk)

    if not ordered_unique_chunks:
        raise ValueError("No non-empty text chunks found in strategy inputs")

    return "\n\n".join(ordered_unique_chunks)


def build_document_vector(
    *,
    strategy_name: str,
    inputs: Sequence[EmbeddingInput],
    section_weights: Mapping[str, float],
    embedding_backend: EmbeddingBackend,
) -> Vector:
    """Build strategy-specific document vector from Phase 2 inputs."""
    prepared_inputs = [item for item in inputs if item.text.strip()]
    if not prepared_inputs:
        raise ValueError("Cannot build document vector from empty strategy inputs")

    vectors = embedding_backend.embed([item.text for item in prepared_inputs])
    if len(vectors) != len(prepared_inputs):
        raise ValueError("Embedding backend returned unexpected number of vectors")

    if strategy_name in {
        ContextStrategy.WHOLE_DOC_MEAN_POOL.value,
        ContextStrategy.PARENT_CHILD_PREPEND.value,
    }:
        return mean_pool(vectors)

    if strategy_name == ContextStrategy.WEIGHTED_POOLING.value:
        section_vectors = _average_vectors_by_section(
            inputs=prepared_inputs,
            vectors=vectors,
        )
        return weighted_pool(
            section_vectors=section_vectors,
            section_weights=section_weights,
        )

    raise ValueError(f"Unsupported strategy for vector construction: {strategy_name}")


def _average_vectors_by_section(
    *,
    inputs: Sequence[EmbeddingInput],
    vectors: Sequence[Vector],
) -> dict[str, Vector]:
    grouped: dict[str, list[Vector]] = defaultdict(list)
    for embedding_input, vector in zip(inputs, vectors, strict=True):
        grouped[embedding_input.section].append(vector)

    return {
        section_name: mean_pool(section_vectors)
        for section_name, section_vectors in grouped.items()
    }


def _embed_one(embedding_backend: EmbeddingBackend, text: str) -> Vector:
    vectors = embedding_backend.embed([text])
    if len(vectors) != 1:
        raise ValueError("Embedding backend returned unexpected vector count for summary text")
    return vectors[0]
