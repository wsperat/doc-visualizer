"""Embedding backends for Phase 4 semantic auditing."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Protocol, cast

from doc_visualizer.phase2.models import Vector
from doc_visualizer.phase4.protocols import EmbeddingBackend


class _SparseMatrixLike(Protocol):
    def toarray(self) -> Sequence[Sequence[float]]: ...


class _HashingVectorizerLike(Protocol):
    def transform(self, raw_documents: Sequence[str]) -> _SparseMatrixLike: ...


class _SentenceTransformerLike(Protocol):
    def encode(
        self,
        texts: Sequence[str],
        *,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
    ) -> Sequence[Sequence[float]]: ...


class HashingEmbeddingBackend:
    """Deterministic embedding backend using sklearn hashing vectorizer."""

    def __init__(self, n_features: int = 512) -> None:
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        self._n_features = n_features
        self._vectorizer: _HashingVectorizerLike | None = None

    @property
    def name(self) -> str:
        return "hashing"

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []

        if self._vectorizer is None:
            vectorizer_ctor = _load_callable("sklearn.feature_extraction.text", "HashingVectorizer")
            self._vectorizer = cast(
                _HashingVectorizerLike,
                vectorizer_ctor(
                    n_features=self._n_features,
                    alternate_sign=False,
                    norm=None,
                ),
            )

        sparse_matrix = self._vectorizer.transform(texts)
        dense_rows = sparse_matrix.toarray()
        return [tuple(float(value) for value in row) for row in dense_rows]


class SentenceTransformerEmbeddingBackend:
    """Optional sentence-transformers backend."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
    ) -> None:
        self._model_name = model_name
        self._model: _SentenceTransformerLike | None = None

    @property
    def name(self) -> str:
        return "sentence_transformers"

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []

        if self._model is None:
            model_ctor = _load_callable("sentence_transformers", "SentenceTransformer")
            self._model = cast(_SentenceTransformerLike, model_ctor(self._model_name))

        raw_vectors = self._model.encode(
            texts,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return [tuple(float(value) for value in vector) for vector in raw_vectors]


def build_embedding_backend(
    backend_name: str,
    *,
    model_name: str | None = None,
    hashing_features: int = 512,
) -> EmbeddingBackend:
    """Construct embedding backend from CLI-friendly identifier."""
    normalized_name = backend_name.strip().lower()

    if normalized_name == "hashing":
        return HashingEmbeddingBackend(n_features=hashing_features)

    if normalized_name in {"sentence_transformers", "sentence-transformers"}:
        resolved_model = model_name or "sentence-transformers/all-mpnet-base-v2"
        return SentenceTransformerEmbeddingBackend(model_name=resolved_model)

    raise ValueError(
        "Unsupported embedding backend "
        f"'{backend_name}'. Available: ['hashing', 'sentence_transformers']"
    )


def _load_callable(module_name: str, attribute_name: str) -> Callable[..., object]:
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Required dependency '{module_name}' is not installed for this backend"
        ) from exc

    target = getattr(module, attribute_name, None)
    if not callable(target):
        raise RuntimeError(
            f"Dependency '{module_name}' is installed but '{attribute_name}' is unavailable"
        )

    return cast(Callable[..., object], target)
