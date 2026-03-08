"""Pure text utilities for parser normalization."""

from __future__ import annotations

import re
from collections.abc import Iterable

_WHITESPACE_RE = re.compile(r"\s+")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim the result."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def merge_text(existing_text: str, incoming_text: str) -> str:
    """Join non-empty text blocks using paragraph separators."""
    left = normalize_whitespace(existing_text)
    right = normalize_whitespace(incoming_text)
    if not left:
        return right
    if not right:
        return left
    return f"{left}\n\n{right}"


def join_non_empty(chunks: Iterable[str]) -> str:
    """Join normalized non-empty chunks using spaces."""
    normalized = [normalize_whitespace(chunk) for chunk in chunks]
    return " ".join(chunk for chunk in normalized if chunk)


def extract_first_year(text: str) -> int | None:
    """Extract the first year (1900-2099) from a text blob."""
    match = _YEAR_RE.search(text)
    if match is None:
        return None
    return int(match.group(0))
