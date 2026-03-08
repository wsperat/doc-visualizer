"""Context strategy builders for Phase 2."""

from __future__ import annotations

from collections.abc import Mapping

from doc_visualizer.phase2.chunking import join_non_empty, split_into_token_chunks
from doc_visualizer.phase2.models import EmbeddingInput


def build_whole_doc_inputs(sections: Mapping[str, str]) -> list[EmbeddingInput]:
    """Return one embedding input per non-empty section."""
    inputs: list[EmbeddingInput] = []
    for section_name, section_text in sections.items():
        normalized_text = section_text.strip()
        if not normalized_text:
            continue
        inputs.append(EmbeddingInput(section=section_name, text=normalized_text))
    return inputs


def build_parent_child_prepend_inputs(
    title: str,
    abstract: str,
    sections: Mapping[str, str],
    max_tokens: int = 512,
) -> list[EmbeddingInput]:
    """Build parent-child prepend chunk inputs.

    For every section chunk, prepend title and abstract context.
    """
    parent_context = join_non_empty((f"Title: {title}", f"Abstract: {abstract}"))

    inputs: list[EmbeddingInput] = []
    for section_name, section_text in sections.items():
        normalized_text = section_text.strip()
        if not normalized_text:
            continue

        chunks = split_into_token_chunks(normalized_text, max_tokens=max_tokens)
        for chunk_index, chunk in enumerate(chunks):
            payload = join_non_empty(
                (
                    parent_context,
                    f"Section: {section_name}",
                    chunk,
                )
            )
            inputs.append(
                EmbeddingInput(
                    section=section_name,
                    text=payload,
                    chunk_index=chunk_index,
                )
            )

    return inputs
