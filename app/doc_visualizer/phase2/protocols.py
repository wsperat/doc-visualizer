"""Protocols for embedding backends used in Phase 2."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from doc_visualizer.phase2.models import Vector


class EmbeddingBackend(Protocol):
    """Abstraction over embedding model implementations."""

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Return an embedding vector for each input text."""
