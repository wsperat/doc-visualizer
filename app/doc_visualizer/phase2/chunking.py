"""Token chunking utilities for Phase 2 context strategies."""

from __future__ import annotations

from collections.abc import Sequence


def split_into_token_chunks(text: str, max_tokens: int = 512) -> list[str]:
    """Split text into whitespace-token chunks of at most `max_tokens` tokens."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    tokens = text.split()
    if not tokens:
        return []

    chunks: list[str] = []
    for start_index in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[start_index : start_index + max_tokens]
        chunks.append(" ".join(chunk_tokens))

    return chunks


def count_tokens(text: str) -> int:
    """Count whitespace-delimited tokens."""
    return len(text.split())


def join_non_empty(parts: Sequence[str]) -> str:
    """Join non-empty text parts with blank lines."""
    normalized = [part.strip() for part in parts if part.strip()]
    return "\n\n".join(normalized)
