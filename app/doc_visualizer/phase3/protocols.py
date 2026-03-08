"""Protocols for Phase 3 topic-model engines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from doc_visualizer.phase3.models import EngineOutput, TopicDocument


class TopicModelEngine(Protocol):
    """Engine contract for topic modeling tracks."""

    @property
    def name(self) -> str:
        """Stable engine identifier."""

    def fit(
        self,
        documents: Sequence[TopicDocument],
        *,
        top_n_terms: int,
        n_topics: int | None,
    ) -> EngineOutput:
        """Fit the model on documents and return normalized engine output."""
