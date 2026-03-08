from __future__ import annotations

import pytest

from doc_visualizer.phase4.vector_math import cosine_similarity


def test_cosine_similarity_returns_expected_value() -> None:
    score = cosine_similarity((1.0, 1.0), (1.0, 1.0))
    assert score == pytest.approx(1.0)


def test_cosine_similarity_validates_dimensions() -> None:
    with pytest.raises(ValueError, match="equal vector dimensionality"):
        _ = cosine_similarity((1.0, 2.0), (1.0,))
