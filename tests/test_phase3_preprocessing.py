from __future__ import annotations

from doc_visualizer.phase3.preprocessing import clean_for_lda, tokenize_and_normalize


def test_tokenize_and_normalize_removes_stopwords_and_normalizes_suffixes() -> None:
    tokens = tokenize_and_normalize("The studies are testing methods and findings")

    assert "study" in tokens
    assert "test" in tokens
    assert "method" in tokens
    assert "finding" in tokens
    assert "the" not in tokens


def test_clean_for_lda_returns_space_joined_tokens() -> None:
    cleaned = clean_for_lda("This is a simple methods section")

    assert cleaned == "simple method section"
