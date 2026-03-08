"""Summarization backends for Phase 4."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from typing import Protocol, cast

from doc_visualizer.phase4.protocols import SummarizationBackend

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")


class _PipelineLike(Protocol):
    def __call__(
        self,
        inputs: str,
        *,
        max_length: int,
        min_length: int,
        do_sample: bool,
        truncation: bool,
    ) -> Sequence[Mapping[str, object]]: ...


class ExtractiveLeadSummarizer:
    """Simple deterministic sentence-leading summarizer."""

    def __init__(self, max_sentences: int = 5, max_chars: int = 5000) -> None:
        if max_sentences <= 0:
            raise ValueError("max_sentences must be positive")
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self._max_sentences = max_sentences
        self._max_chars = max_chars

    @property
    def name(self) -> str:
        return "extractive"

    def summarize(self, text: str) -> str:
        normalized = _normalize_whitespace(text)
        if not normalized:
            raise ValueError("Cannot summarize an empty document")

        sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(normalized)]
        non_empty_sentences = [sentence for sentence in sentences if sentence]
        if not non_empty_sentences:
            raise ValueError("No sentence-like content available for summarization")

        summary = " ".join(non_empty_sentences[: self._max_sentences]).strip()
        if len(summary) <= self._max_chars:
            return summary

        truncated = summary[: self._max_chars].rstrip()
        if truncated.endswith((".", "!", "?")):
            return truncated
        return f"{truncated}..."


class TransformersSeq2SeqSummarizer:
    """Lazy-loaded transformer summarizer wrapper."""

    def __init__(
        self,
        *,
        backend_name: str,
        model_name: str,
        max_input_chars: int = 12000,
        max_output_tokens: int = 256,
        min_output_tokens: int = 64,
    ) -> None:
        if max_input_chars <= 0:
            raise ValueError("max_input_chars must be positive")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if min_output_tokens <= 0:
            raise ValueError("min_output_tokens must be positive")
        if min_output_tokens > max_output_tokens:
            raise ValueError("min_output_tokens cannot exceed max_output_tokens")

        self._backend_name = backend_name
        self._model_name = model_name
        self._max_input_chars = max_input_chars
        self._max_output_tokens = max_output_tokens
        self._min_output_tokens = min_output_tokens
        self._pipeline: _PipelineLike | None = None

    @property
    def name(self) -> str:
        return self._backend_name

    def summarize(self, text: str) -> str:
        normalized = _normalize_whitespace(text)
        if not normalized:
            raise ValueError("Cannot summarize an empty document")

        model_input = normalized[: self._max_input_chars]
        pipeline = self._get_pipeline()
        raw_output = pipeline(
            model_input,
            max_length=self._max_output_tokens,
            min_length=min(self._min_output_tokens, self._max_output_tokens),
            do_sample=False,
            truncation=True,
        )

        if not raw_output:
            raise RuntimeError("Summarization backend returned no output")

        summary_value = raw_output[0].get("summary_text")
        if not isinstance(summary_value, str) or not summary_value.strip():
            raise RuntimeError("Summarization backend returned malformed output")

        return summary_value.strip()

    def _get_pipeline(self) -> _PipelineLike:
        if self._pipeline is None:
            pipeline_ctor = _load_callable("transformers", "pipeline")
            self._pipeline = cast(
                _PipelineLike,
                pipeline_ctor(
                    "summarization",
                    model=self._model_name,
                    tokenizer=self._model_name,
                ),
            )
        return self._pipeline


def build_summarizer(
    backend_name: str,
    *,
    model_name: str | None = None,
    max_sentences: int = 5,
) -> SummarizationBackend:
    """Construct summarizer backend from CLI-friendly identifier."""
    normalized_name = backend_name.strip().lower()

    if normalized_name == "extractive":
        return ExtractiveLeadSummarizer(max_sentences=max_sentences)

    if normalized_name == "led":
        return TransformersSeq2SeqSummarizer(
            backend_name="led",
            model_name=model_name or "allenai/led-base-16384",
        )

    if normalized_name == "longt5":
        return TransformersSeq2SeqSummarizer(
            backend_name="longt5",
            model_name=model_name or "google/long-t5-tglobal-base",
        )

    raise ValueError(
        "Unsupported summarization backend "
        f"'{backend_name}'. Available: ['extractive', 'led', 'longt5']"
    )


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _load_callable(module_name: str, attribute_name: str) -> Callable[..., object]:
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Required dependency '{module_name}' is not installed for this summarizer"
        ) from exc

    target = getattr(module, attribute_name, None)
    if not callable(target):
        raise RuntimeError(
            f"Dependency '{module_name}' is installed but '{attribute_name}' is unavailable"
        )

    return cast(Callable[..., object], target)
