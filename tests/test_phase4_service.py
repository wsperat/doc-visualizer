from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from doc_visualizer.phase2.models import ContextStrategy, EmbeddingInput, Vector
from doc_visualizer.phase4.models import StrategyDocument
from doc_visualizer.phase4.service import (
    PhaseFourAuditService,
    build_document_text,
    build_document_vector,
)


class FakeSummarizer:
    def __init__(self, output: str) -> None:
        self._output = output
        self.received_text: str = ""

    @property
    def name(self) -> str:
        return "fake_summary"

    def summarize(self, text: str) -> str:
        self.received_text = text
        return self._output


class MappingEmbeddingBackend:
    def __init__(self, vectors: Mapping[str, Vector]) -> None:
        self._vectors = dict(vectors)

    @property
    def name(self) -> str:
        return "mapping_embed"

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._vectors[text] for text in texts]


def test_build_document_text_deduplicates_repeated_chunks() -> None:
    inputs = (
        EmbeddingInput(section="abstract", text="Alpha section."),
        EmbeddingInput(section="results", text="Beta section."),
        EmbeddingInput(section="results", text="Alpha section."),
    )

    merged_text = build_document_text(inputs)

    assert merged_text == "Alpha section.\n\nBeta section."


def test_build_document_vector_weighted_pooling_applies_section_weights() -> None:
    inputs = (
        EmbeddingInput(section="abstract", text="abstract text"),
        EmbeddingInput(section="results", text="results text"),
    )
    backend = MappingEmbeddingBackend(
        {
            "abstract text": (1.0, 0.0),
            "results text": (3.0, 0.0),
        }
    )

    vector = build_document_vector(
        strategy_name=ContextStrategy.WEIGHTED_POOLING.value,
        inputs=inputs,
        section_weights={"abstract": 1.0, "results": 3.0},
        embedding_backend=backend,
    )

    assert vector == (2.5, 0.0)


def test_audit_service_marks_low_similarity_when_score_is_below_threshold() -> None:
    summarizer = FakeSummarizer(output="summary text")
    backend = MappingEmbeddingBackend(
        {
            "abstract text": (1.0, 0.0),
            "summary text": (0.0, 1.0),
        }
    )
    service = PhaseFourAuditService(
        summarizer=summarizer,
        embedding_backend=backend,
        similarity_threshold=0.8,
    )

    document = StrategyDocument(
        source="grobid",
        strategy=ContextStrategy.WHOLE_DOC_MEAN_POOL.value,
        document_id="paper",
        inputs=(EmbeddingInput(section="abstract", text="abstract text"),),
        section_weights={},
    )

    record = service.audit_document(document)

    assert summarizer.received_text == "abstract text"
    assert record.is_below_threshold is True
    assert record.similarity_score == pytest.approx(0.0)
    assert record.summary_backend == "fake_summary"
    assert record.embedding_backend == "mapping_embed"
