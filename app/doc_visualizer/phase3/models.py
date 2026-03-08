"""Phase 3 domain models for multi-track topic modeling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TopicDocument:
    """One transformed document consumed by topic engines."""

    document_id: str
    text: str


@dataclass(frozen=True, slots=True)
class TopicTerm:
    """Weighted term representation for one topic."""

    term: str
    weight: float

    def to_dict(self) -> dict[str, object]:
        return {
            "term": self.term,
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class TopicCluster:
    """Topic cluster with top terms and assigned document ids."""

    topic_id: int
    terms: tuple[TopicTerm, ...]
    document_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "topic_id": self.topic_id,
            "terms": [term.to_dict() for term in self.terms],
            "document_ids": list(self.document_ids),
        }


@dataclass(frozen=True, slots=True)
class EngineOutput:
    """Normalized topic-model output from one engine over one corpus."""

    engine: str
    topics: tuple[TopicCluster, ...]
    assignments: dict[str, int]
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "engine": self.engine,
            "topics": [topic.to_dict() for topic in self.topics],
            "assignments": [
                {"document_id": document_id, "topic_id": topic_id}
                for document_id, topic_id in sorted(self.assignments.items())
            ],
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class StrategyCorpus:
    """Corpus for one phase2 source+strategy pair."""

    source: str
    strategy: str
    documents: tuple[TopicDocument, ...]


@dataclass(frozen=True, slots=True)
class PhaseThreeRunResult:
    """Execution status for one source+strategy+engine run."""

    source: str
    strategy: str
    engine: str
    status: str
    message: str
    output_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source": self.source,
            "strategy": self.strategy,
            "engine": self.engine,
            "status": self.status,
            "message": self.message,
        }
        if self.output_path is not None:
            payload["output_path"] = self.output_path
        return payload
