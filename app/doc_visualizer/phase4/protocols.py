"""Protocols for Phase 4 summarization and embedding backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from doc_visualizer.phase2.models import Vector


class SummarizationBackend(Protocol):
    """Abstract summarization engine."""

    @property
    def name(self) -> str:
        """Stable backend identifier."""

    def summarize(self, text: str) -> str:
        """Produce a summary for one input document text."""


class EmbeddingBackend(Protocol):
    """Abstract embedding engine."""

    @property
    def name(self) -> str:
        """Stable backend identifier."""

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Embed all provided texts."""
