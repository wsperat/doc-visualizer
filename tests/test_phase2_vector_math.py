from __future__ import annotations

import pytest

from doc_visualizer.phase2.vector_math import mean_pool, weighted_pool


def test_mean_pool_computes_average() -> None:
    pooled = mean_pool(((1.0, 3.0), (3.0, 5.0)))

    assert pooled == (2.0, 4.0)


def test_mean_pool_raises_for_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="equal dimensionality"):
        mean_pool(((1.0, 2.0), (3.0,)))


def test_weighted_pool_uses_weights_and_defaults() -> None:
    section_vectors = {
        "abstract": (1.0, 1.0),
        "results": (5.0, 3.0),
    }

    pooled = weighted_pool(section_vectors=section_vectors, section_weights={"results": 3.0})

    # abstract weight defaults to 1.0, results weight is 3.0
    assert pooled == (4.0, 2.5)


def test_weighted_pool_raises_when_all_weights_non_positive() -> None:
    with pytest.raises(ValueError, match="positive section weight"):
        weighted_pool(
            section_vectors={"abstract": (1.0, 2.0)},
            section_weights={"abstract": 0.0},
        )
