"""Vector math helpers for Phase 4 semantic auditing."""

from __future__ import annotations

import math

from doc_visualizer.phase2.models import Vector


def cosine_similarity(lhs: Vector, rhs: Vector) -> float:
    """Compute cosine similarity between two vectors."""
    if not lhs or not rhs:
        raise ValueError("cosine_similarity requires non-empty vectors")
    if len(lhs) != len(rhs):
        raise ValueError("cosine_similarity requires equal vector dimensionality")

    numerator = sum(left * right for left, right in zip(lhs, rhs, strict=True))
    lhs_norm = math.sqrt(sum(value * value for value in lhs))
    rhs_norm = math.sqrt(sum(value * value for value in rhs))

    if lhs_norm == 0 or rhs_norm == 0:
        return 0.0

    return numerator / (lhs_norm * rhs_norm)
