"""Vector aggregation helpers for Phase 2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from doc_visualizer.phase2.models import Vector


def mean_pool(vectors: Sequence[Vector]) -> Vector:
    """Compute arithmetic mean across vectors."""
    if not vectors:
        raise ValueError("mean_pool requires at least one vector")

    _validate_dimensions(vectors)
    dimension = len(vectors[0])

    sums = [0.0] * dimension
    for vector in vectors:
        for index, value in enumerate(vector):
            sums[index] += value

    divisor = float(len(vectors))
    return tuple(total / divisor for total in sums)


def weighted_pool(
    section_vectors: Mapping[str, Vector],
    section_weights: Mapping[str, float],
) -> Vector:
    """Compute weighted pooled vector across sections.

    Missing section weights default to `1.0`.
    Sections with non-positive weights are ignored.
    """
    if not section_vectors:
        raise ValueError("weighted_pool requires at least one section vector")

    vectors = list(section_vectors.values())
    _validate_dimensions(vectors)
    dimension = len(vectors[0])

    weighted_sums = [0.0] * dimension
    denominator = 0.0

    for section_name, vector in section_vectors.items():
        weight = section_weights.get(section_name, 1.0)
        if weight <= 0:
            continue

        denominator += weight
        for index, value in enumerate(vector):
            weighted_sums[index] += value * weight

    if denominator == 0:
        raise ValueError("weighted_pool requires at least one positive section weight")

    return tuple(total / denominator for total in weighted_sums)


def _validate_dimensions(vectors: Sequence[Vector]) -> None:
    """Ensure all vectors share identical dimensionality."""
    expected_dimension = len(vectors[0])
    for vector in vectors:
        if len(vector) != expected_dimension:
            raise ValueError("all vectors must have equal dimensionality")
