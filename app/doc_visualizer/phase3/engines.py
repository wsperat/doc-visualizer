"""Topic-model engine implementations for Phase 3."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Protocol, cast

from doc_visualizer.phase3.models import EngineOutput, TopicCluster, TopicDocument, TopicTerm
from doc_visualizer.phase3.preprocessing import clean_for_lda
from doc_visualizer.phase3.protocols import TopicModelEngine


class _CountVectorizerLike(Protocol):
    def fit_transform(self, raw_documents: Sequence[str]) -> object: ...

    def get_feature_names_out(self) -> Sequence[str]: ...


class _LdaLike(Protocol):
    components_: Sequence[Sequence[float]]

    def fit_transform(self, x: object) -> Sequence[Sequence[float]]: ...


class _BertopicLike(Protocol):
    def fit_transform(self, documents: Sequence[str]) -> tuple[Sequence[int], object]: ...

    def get_topic(self, topic_id: int) -> Sequence[tuple[str, float]] | None: ...


class _Top2VecLike(Protocol):
    def get_documents_topics(self, doc_ids: Sequence[int]) -> tuple[Sequence[int], object]: ...

    def get_topics(
        self,
        num_topics: int | None = None,
    ) -> tuple[Sequence[Sequence[str]], Sequence[Sequence[float]], Sequence[int]]: ...


class LdaSklearnEngine:
    """LDA engine backed by scikit-learn."""

    def __init__(
        self,
        default_topics: int = 8,
        max_features: int = 5000,
        random_state: int = 42,
    ) -> None:
        self._default_topics = default_topics
        self._max_features = max_features
        self._random_state = random_state

    @property
    def name(self) -> str:
        return "lda"

    def fit(
        self,
        documents: Sequence[TopicDocument],
        *,
        top_n_terms: int,
        n_topics: int | None,
    ) -> EngineOutput:
        if not documents:
            raise ValueError("LDA requires at least one document")

        cleaned_texts = [clean_for_lda(document.text) for document in documents]
        if not any(text.strip() for text in cleaned_texts):
            raise ValueError("LDA preprocessing produced empty text for all documents")

        vectorizer_ctor = _load_callable("sklearn.feature_extraction.text", "CountVectorizer")
        vectorizer = cast(
            _CountVectorizerLike,
            vectorizer_ctor(max_features=self._max_features),
        )
        document_term_matrix = vectorizer.fit_transform(cleaned_texts)

        feature_names = list(vectorizer.get_feature_names_out())
        if not feature_names:
            raise ValueError("LDA vocabulary is empty after preprocessing")

        requested_topics = n_topics if n_topics is not None else self._default_topics
        effective_topics = max(1, min(requested_topics, len(documents)))

        lda_ctor = _load_callable("sklearn.decomposition", "LatentDirichletAllocation")
        lda_model = cast(
            _LdaLike,
            lda_ctor(n_components=effective_topics, random_state=self._random_state),
        )

        document_topic_distribution = [
            tuple(float(value) for value in row)
            for row in lda_model.fit_transform(document_term_matrix)
        ]

        assignments: dict[str, int] = {}
        for document, scores in zip(documents, document_topic_distribution, strict=True):
            assignments[document.document_id] = _argmax_index(scores)

        topics: list[TopicCluster] = []
        for topic_id, term_weights in enumerate(lda_model.components_):
            top_indices = _top_indices(term_weights, limit=top_n_terms)
            topic_terms = tuple(
                TopicTerm(term=feature_names[index], weight=float(term_weights[index]))
                for index in top_indices
            )
            topic_documents = tuple(
                document_id
                for document_id, assigned_topic in assignments.items()
                if assigned_topic == topic_id
            )
            topics.append(
                TopicCluster(
                    topic_id=topic_id,
                    terms=topic_terms,
                    document_ids=topic_documents,
                )
            )

        return EngineOutput(
            engine=self.name,
            topics=tuple(topics),
            assignments=assignments,
            metadata={
                "effective_topics": effective_topics,
                "vocabulary_size": len(feature_names),
                "document_count": len(documents),
            },
        )


class BertopicEngine:
    """BERTopic engine wrapper with lazy imports."""

    def __init__(self, default_topics: int = 8) -> None:
        self._default_topics = default_topics

    @property
    def name(self) -> str:
        return "bertopic"

    def fit(
        self,
        documents: Sequence[TopicDocument],
        *,
        top_n_terms: int,
        n_topics: int | None,
    ) -> EngineOutput:
        if not documents:
            raise ValueError("BERTopic requires at least one document")

        bertopic_ctor = _load_callable("bertopic", "BERTopic")
        target_topics = n_topics if n_topics is not None else self._default_topics
        model = cast(
            _BertopicLike,
            bertopic_ctor(
                nr_topics=target_topics,
                calculate_probabilities=True,
                verbose=False,
            ),
        )

        texts = [document.text for document in documents]
        topic_ids, _ = model.fit_transform(texts)

        assignments: dict[str, int] = {}
        for document, topic_id in zip(documents, topic_ids, strict=True):
            assignments[document.document_id] = int(topic_id)

        unique_topic_ids = sorted({int(topic_id) for topic_id in topic_ids})
        topics: list[TopicCluster] = []

        for topic_id in unique_topic_ids:
            model_terms = model.get_topic(topic_id) or ()
            topic_terms = tuple(
                TopicTerm(term=term, weight=float(weight))
                for term, weight in model_terms[:top_n_terms]
                if isinstance(term, str)
            )
            topic_documents = tuple(
                document_id
                for document_id, assigned_topic in assignments.items()
                if assigned_topic == topic_id
            )
            topics.append(
                TopicCluster(
                    topic_id=topic_id,
                    terms=topic_terms,
                    document_ids=topic_documents,
                )
            )

        return EngineOutput(
            engine=self.name,
            topics=tuple(topics),
            assignments=assignments,
            metadata={
                "requested_topics": target_topics,
                "document_count": len(documents),
            },
        )


class Top2VecEngine:
    """Optional Top2Vec engine wrapper."""

    def __init__(self, default_topics: int = 8) -> None:
        self._default_topics = default_topics

    @property
    def name(self) -> str:
        return "top2vec"

    def fit(
        self,
        documents: Sequence[TopicDocument],
        *,
        top_n_terms: int,
        n_topics: int | None,
    ) -> EngineOutput:
        if not documents:
            raise ValueError("Top2Vec requires at least one document")

        top2vec_ctor = _load_callable("top2vec", "Top2Vec")
        texts = [document.text for document in documents]

        model = cast(
            _Top2VecLike,
            top2vec_ctor(
                documents=texts,
                speed="learn",
                min_count=1,
                workers=1,
            ),
        )

        doc_indices = list(range(len(documents)))
        document_topics, _ = model.get_documents_topics(doc_indices)

        assignments: dict[str, int] = {}
        for document, topic_id in zip(documents, document_topics, strict=True):
            assignments[document.document_id] = int(topic_id)

        target_topics = n_topics if n_topics is not None else self._default_topics
        topic_words, topic_scores, topic_ids = model.get_topics(num_topics=target_topics)

        topics: list[TopicCluster] = []
        for topic_word_list, topic_score_list, topic_id in zip(
            topic_words,
            topic_scores,
            topic_ids,
            strict=True,
        ):
            top_terms = tuple(
                TopicTerm(term=term, weight=float(score))
                for term, score in zip(topic_word_list, topic_score_list, strict=True)
            )[:top_n_terms]
            topic_documents = tuple(
                document_id
                for document_id, assigned_topic in assignments.items()
                if assigned_topic == int(topic_id)
            )
            topics.append(
                TopicCluster(
                    topic_id=int(topic_id),
                    terms=top_terms,
                    document_ids=topic_documents,
                )
            )

        return EngineOutput(
            engine=self.name,
            topics=tuple(topics),
            assignments=assignments,
            metadata={
                "requested_topics": target_topics,
                "document_count": len(documents),
            },
        )


def build_engines(engine_names: Sequence[str], *, default_topics: int) -> list[TopicModelEngine]:
    """Build engine instances from user-provided engine names."""
    engine_map: dict[str, Callable[[], TopicModelEngine]] = {
        "lda": lambda: LdaSklearnEngine(default_topics=default_topics),
        "bertopic": lambda: BertopicEngine(default_topics=default_topics),
        "top2vec": lambda: Top2VecEngine(default_topics=default_topics),
    }

    engines: list[TopicModelEngine] = []
    for engine_name in engine_names:
        normalized_name = engine_name.strip().lower()
        if not normalized_name:
            continue
        if normalized_name not in engine_map:
            raise ValueError(
                f"Unsupported engine '{engine_name}'. Available: {sorted(engine_map.keys())}"
            )
        engines.append(engine_map[normalized_name]())

    if not engines:
        raise ValueError("No valid topic engines were requested")

    return engines


def _load_callable(module_name: str, attribute_name: str) -> Callable[..., object]:
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Required dependency '{module_name}' is not installed for this engine"
        ) from exc

    target = getattr(module, attribute_name, None)
    if not callable(target):
        raise RuntimeError(
            f"Dependency '{module_name}' is installed but '{attribute_name}' is unavailable"
        )

    return cast(Callable[..., object], target)


def _argmax_index(values: Sequence[float]) -> int:
    if not values:
        raise ValueError("Cannot compute argmax of an empty sequence")
    return max(range(len(values)), key=lambda index: values[index])


def _top_indices(weights: Sequence[float], limit: int) -> list[int]:
    scored_indices = sorted(
        range(len(weights)),
        key=lambda index: weights[index],
        reverse=True,
    )
    return scored_indices[: max(1, limit)]
