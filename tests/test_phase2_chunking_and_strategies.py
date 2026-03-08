from __future__ import annotations

from doc_visualizer.phase2.chunking import count_tokens, split_into_token_chunks
from doc_visualizer.phase2.strategies import (
    build_parent_child_prepend_inputs,
    build_whole_doc_inputs,
)


def test_split_into_token_chunks_respects_limit() -> None:
    text = "one two three four five"

    chunks = split_into_token_chunks(text, max_tokens=2)

    assert chunks == ["one two", "three four", "five"]


def test_count_tokens_uses_whitespace_tokenization() -> None:
    assert count_tokens("a  b\n c") == 3


def test_build_whole_doc_inputs_skips_empty_sections() -> None:
    sections = {
        "abstract": "A",
        "methods": "",
        "results": "R",
    }

    inputs = build_whole_doc_inputs(sections)

    assert [(item.section, item.text) for item in inputs] == [
        ("abstract", "A"),
        ("results", "R"),
    ]


def test_build_parent_child_prepend_inputs_chunks_and_prepends_context() -> None:
    sections = {
        "abstract": "why it matters",
        "chapter_1": "alpha beta gamma delta epsilon",
    }

    inputs = build_parent_child_prepend_inputs(
        title="Doc Title",
        abstract=sections["abstract"],
        sections=sections,
        max_tokens=2,
    )

    chapter_inputs = [item for item in inputs if item.section == "chapter_1"]
    assert len(chapter_inputs) == 3
    assert chapter_inputs[0].chunk_index == 0
    assert chapter_inputs[1].chunk_index == 1
    assert chapter_inputs[2].chunk_index == 2

    first_payload = chapter_inputs[0].text
    assert "Title: Doc Title" in first_payload
    assert "Abstract: why it matters" in first_payload
    assert "Section: chapter_1" in first_payload
    assert "alpha beta" in first_payload
