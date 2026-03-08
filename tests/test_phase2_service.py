from __future__ import annotations

from collections.abc import Sequence

from doc_visualizer.phase2.models import ContextStrategy, Vector
from doc_visualizer.phase2.service import PhaseTwoContextService


class FakeEmbeddingBackend:
    def embed(self, texts: Sequence[str]) -> list[Vector]:
        # 2D vector: [token_count, text_index]
        vectors: list[Vector] = []
        for index, text in enumerate(texts):
            vectors.append((float(len(text.split())), float(index + 1)))
        return vectors


def test_service_builds_whole_doc_mean_pool_vector() -> None:
    service = PhaseTwoContextService(embedding_backend=FakeEmbeddingBackend())
    sections = {
        "abstract": "a b",
        "results": "c d e",
    }

    vector = service.build_document_vector(
        ContextStrategy.WHOLE_DOC_MEAN_POOL,
        title="ignored",
        sections=sections,
    )

    # vectors: (2,1), (3,2) -> mean (2.5, 1.5)
    assert vector == (2.5, 1.5)


def test_service_builds_weighted_pooling_vector() -> None:
    service = PhaseTwoContextService(embedding_backend=FakeEmbeddingBackend())
    sections = {
        "abstract": "a b",
        "results": "c d e",
    }

    vector = service.build_document_vector(
        ContextStrategy.WEIGHTED_POOLING,
        title="ignored",
        sections=sections,
        section_weights={"abstract": 1.0, "results": 3.0},
    )

    # section vectors are (2,1) and (3,2) -> weighted: ((2*1 + 3*3)/4, (1*1 + 2*3)/4)
    assert vector == (2.75, 1.75)


def test_service_builds_parent_child_prepend_vector() -> None:
    service = PhaseTwoContextService(embedding_backend=FakeEmbeddingBackend())
    sections = {
        "abstract": "why context",
        "chapter": "alpha beta gamma delta",
    }

    vector = service.build_document_vector(
        ContextStrategy.PARENT_CHILD_PREPEND,
        title="Doc",
        sections=sections,
        max_tokens=2,
    )

    # Expect at least 3 inputs (1 abstract chunk + 2 chapter chunks)
    assert len(vector) == 2
    assert vector[0] > 0
    assert vector[1] > 0
