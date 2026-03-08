"""Orchestration service for Phase 2 context strategies."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from doc_visualizer.phase2.models import ContextStrategy, EmbeddingInput, Vector
from doc_visualizer.phase2.protocols import EmbeddingBackend
from doc_visualizer.phase2.strategies import (
    build_parent_child_prepend_inputs,
    build_whole_doc_inputs,
)
from doc_visualizer.phase2.vector_math import mean_pool, weighted_pool


class PhaseTwoContextService:
    """Build strategy-specific embedding inputs and pooled document vectors."""

    def __init__(self, embedding_backend: EmbeddingBackend) -> None:
        self._embedding_backend = embedding_backend

    def build_inputs(
        self,
        strategy: ContextStrategy,
        *,
        title: str,
        sections: Mapping[str, str],
        max_tokens: int = 512,
    ) -> list[EmbeddingInput]:
        """Build embedding inputs for a selected strategy."""
        if strategy is ContextStrategy.WHOLE_DOC_MEAN_POOL:
            return build_whole_doc_inputs(sections)

        if strategy is ContextStrategy.PARENT_CHILD_PREPEND:
            return build_parent_child_prepend_inputs(
                title=title,
                abstract=sections.get("abstract", ""),
                sections=sections,
                max_tokens=max_tokens,
            )

        if strategy is ContextStrategy.WEIGHTED_POOLING:
            return build_whole_doc_inputs(sections)

        raise ValueError(f"Unsupported context strategy: {strategy}")

    def build_document_vector(
        self,
        strategy: ContextStrategy,
        *,
        title: str,
        sections: Mapping[str, str],
        section_weights: Mapping[str, float] | None = None,
        max_tokens: int = 512,
    ) -> Vector:
        """Create a pooled document vector for the selected strategy."""
        inputs = self.build_inputs(
            strategy,
            title=title,
            sections=sections,
            max_tokens=max_tokens,
        )
        if not inputs:
            raise ValueError("No embedding inputs could be built from the provided sections")

        vectors = self._embedding_backend.embed([item.text for item in inputs])
        if len(vectors) != len(inputs):
            raise ValueError("Embedding backend returned unexpected number of vectors")

        if strategy is ContextStrategy.WHOLE_DOC_MEAN_POOL:
            return mean_pool(vectors)

        if strategy is ContextStrategy.PARENT_CHILD_PREPEND:
            return mean_pool(vectors)

        if strategy is ContextStrategy.WEIGHTED_POOLING:
            section_vectors = _average_vectors_by_section(inputs=inputs, vectors=vectors)
            return weighted_pool(
                section_vectors=section_vectors,
                section_weights=section_weights or {},
            )

        raise ValueError(f"Unsupported context strategy: {strategy}")


def _average_vectors_by_section(
    inputs: list[EmbeddingInput],
    vectors: list[Vector],
) -> dict[str, Vector]:
    """Average vectors per section to support weighted pooling."""
    grouped: dict[str, list[Vector]] = defaultdict(list)
    for embedding_input, vector in zip(inputs, vectors, strict=True):
        grouped[embedding_input.section].append(vector)

    return {
        section_name: mean_pool(section_vectors)
        for section_name, section_vectors in grouped.items()
    }
