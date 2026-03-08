"""Phase 2: User-selectable context strategies."""

from doc_visualizer.phase2.models import ContextStrategy, EmbeddingInput, Vector
from doc_visualizer.phase2.service import PhaseTwoContextService
from doc_visualizer.phase2.strategies import (
    build_parent_child_prepend_inputs,
    build_whole_doc_inputs,
)
from doc_visualizer.phase2.vector_math import mean_pool, weighted_pool

__all__ = [
    "ContextStrategy",
    "EmbeddingInput",
    "PhaseTwoContextService",
    "Vector",
    "build_parent_child_prepend_inputs",
    "build_whole_doc_inputs",
    "mean_pool",
    "weighted_pool",
]
