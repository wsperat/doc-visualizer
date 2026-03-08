"""Phase 2 data models and strategy enums."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ContextStrategy(StrEnum):
    """User-selectable context strategies for document embeddings."""

    WHOLE_DOC_MEAN_POOL = "whole_doc_mean_pool"
    PARENT_CHILD_PREPEND = "parent_child_prepend"
    WEIGHTED_POOLING = "weighted_pooling"


@dataclass(frozen=True, slots=True)
class EmbeddingInput:
    """Text unit to be sent to an embedding model."""

    section: str
    text: str
    chunk_index: int | None = None


Vector = tuple[float, ...]
