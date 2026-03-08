from __future__ import annotations

import pytest

from doc_visualizer.phase1.section_classifier import classify_section_heading, normalize_heading
from doc_visualizer.phase1.types import SectionName


@pytest.mark.parametrize(
    ("raw_heading", "expected"),
    [
        ("Introduction", "introduction"),
        ("Background", "introduction"),
        ("Materials and Methods", "methods"),
        ("Methodology", "methods"),
        ("Results and Discussion", "results"),
        ("Discussion", "conclusion"),
        ("Concluding Remarks", "conclusion"),
        ("Abstract", "abstract"),
    ],
)
def test_classify_section_heading(raw_heading: str, expected: SectionName) -> None:
    assert classify_section_heading(raw_heading) == expected


def test_classify_section_heading_returns_none_for_untracked_sections() -> None:
    assert classify_section_heading("Related Work") is None


def test_normalize_heading_cleans_punctuation() -> None:
    assert normalize_heading("  Materials & Methods (v2) ") == "materials methods v2"
