"""Text preprocessing helpers for Phase 3 engines."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z]+")

# Compact stopword set for LDA-oriented cleaning.
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def clean_for_lda(text: str) -> str:
    """Apply deterministic bag-of-words cleaning for LDA modeling."""
    tokens = tokenize_and_normalize(text)
    return " ".join(tokens)


def tokenize_and_normalize(text: str) -> list[str]:
    """Tokenize, lowercase, remove stopwords, and normalize tokens."""
    lowered_text = text.lower()
    raw_tokens = _TOKEN_RE.findall(lowered_text)

    normalized_tokens: list[str] = []
    for token in raw_tokens:
        if token in _STOPWORDS:
            continue
        if len(token) < 3:
            continue
        normalized = _normalize_token(token)
        if normalized and normalized not in _STOPWORDS:
            normalized_tokens.append(normalized)

    return normalized_tokens


def _normalize_token(token: str) -> str:
    """Cheap lemmatization-like normalization for frequent suffixes."""
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    return token
