"""Phase 4 domain models for summarization and semantic auditing."""

from __future__ import annotations

from dataclasses import dataclass

from doc_visualizer.phase2.models import EmbeddingInput


@dataclass(frozen=True, slots=True)
class StrategyDocument:
    """One Phase 2 transformed document ready for Phase 4 processing."""

    source: str
    strategy: str
    document_id: str
    inputs: tuple[EmbeddingInput, ...]
    section_weights: dict[str, float]


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """Summary + semantic audit payload for one document."""

    source: str
    strategy: str
    document_id: str
    summary: str
    similarity_score: float
    similarity_threshold: float
    is_below_threshold: bool
    summary_backend: str
    embedding_backend: str
    input_count: int
    document_vector_dimension: int
    summary_vector_dimension: int

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "strategy": self.strategy,
            "document_id": self.document_id,
            "summary": self.summary,
            "similarity_score": self.similarity_score,
            "similarity_threshold": self.similarity_threshold,
            "is_below_threshold": self.is_below_threshold,
            "summary_backend": self.summary_backend,
            "embedding_backend": self.embedding_backend,
            "input_count": self.input_count,
            "document_vector_dimension": self.document_vector_dimension,
            "summary_vector_dimension": self.summary_vector_dimension,
        }


@dataclass(frozen=True, slots=True)
class PhaseFourRunResult:
    """Execution status for one source+strategy+document audit."""

    source: str
    strategy: str
    document_id: str
    status: str
    message: str
    similarity_score: float | None = None
    output_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source": self.source,
            "strategy": self.strategy,
            "document_id": self.document_id,
            "status": self.status,
            "message": self.message,
        }
        if self.similarity_score is not None:
            payload["similarity_score"] = self.similarity_score
        if self.output_path is not None:
            payload["output_path"] = self.output_path
        return payload
