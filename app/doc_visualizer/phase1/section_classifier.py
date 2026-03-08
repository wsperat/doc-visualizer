"""Heading normalization and section classification."""

from __future__ import annotations

import re
from typing import Final

from doc_visualizer.phase1.text_utils import normalize_whitespace
from doc_visualizer.phase1.types import SectionName

_PUNCTUATION_RE = re.compile(r"[^a-z0-9]+")

_SECTION_ALIASES: Final[dict[SectionName, tuple[str, ...]]] = {
    "abstract": ("abstract",),
    "introduction": ("introduction", "background"),
    "methods": (
        "methods",
        "materials methods",
        "material methods",
        "methodology",
        "experimental methods",
    ),
    "results": ("results", "findings", "results discussion"),
    "conclusion": (
        "conclusion",
        "conclusions",
        "discussion",
        "concluding remarks",
    ),
}


def normalize_heading(raw_heading: str) -> str:
    """Normalize a heading so aliases can be matched robustly."""
    collapsed = normalize_whitespace(raw_heading).lower()
    return _PUNCTUATION_RE.sub(" ", collapsed).strip()


def classify_section_heading(raw_heading: str) -> SectionName | None:
    """Map a heading to one of the five target sections."""
    normalized = normalize_heading(raw_heading)
    if not normalized:
        return None

    for section_name, aliases in _SECTION_ALIASES.items():
        if normalized in aliases:
            return section_name

    if normalized.startswith("introduction"):
        return "introduction"
    if normalized.startswith("method") or normalized.startswith("materials"):
        return "methods"
    if normalized.startswith("result"):
        return "results"
    if normalized.startswith("conclusion") or normalized.startswith("discussion"):
        return "conclusion"
    if normalized.startswith("abstract"):
        return "abstract"

    return None
