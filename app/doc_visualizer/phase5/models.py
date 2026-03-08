"""Phase 5 domain models for dimensionality reduction and mapping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MapPoint:
    """One projected point in the 2D knowledge map."""

    document_id: str
    x: float
    y: float
    cluster_id: int
    similarity_score: float | None = None
    is_below_threshold: bool | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "document_id": self.document_id,
            "x": self.x,
            "y": self.y,
            "cluster_id": self.cluster_id,
        }
        if self.similarity_score is not None:
            payload["similarity_score"] = self.similarity_score
        if self.is_below_threshold is not None:
            payload["is_below_threshold"] = self.is_below_threshold
        return payload


@dataclass(frozen=True, slots=True)
class StrategyMap:
    """Projected and clustered map for one source+strategy corpus."""

    source: str
    strategy: str
    reducer: str
    clusterer: str
    parameters: dict[str, object]
    points: tuple[MapPoint, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "strategy": self.strategy,
            "reducer": self.reducer,
            "clusterer": self.clusterer,
            "parameters": self.parameters,
            "document_count": len(self.points),
            "points": [point.to_dict() for point in self.points],
        }


@dataclass(frozen=True, slots=True)
class PhaseFiveRunResult:
    """Execution status for one source+strategy Phase 5 run."""

    source: str
    strategy: str
    status: str
    message: str
    document_count: int | None = None
    output_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source": self.source,
            "strategy": self.strategy,
            "status": self.status,
            "message": self.message,
        }
        if self.document_count is not None:
            payload["document_count"] = self.document_count
        if self.output_path is not None:
            payload["output_path"] = self.output_path
        return payload
