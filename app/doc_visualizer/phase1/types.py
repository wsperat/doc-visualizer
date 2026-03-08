"""Shared typing primitives for Phase 1."""

from __future__ import annotations

from typing import Final, Literal, TypeAlias

SectionName: TypeAlias = Literal[
    "abstract",
    "introduction",
    "methods",
    "results",
    "conclusion",
]

SECTION_NAMES: Final[tuple[SectionName, ...]] = (
    "abstract",
    "introduction",
    "methods",
    "results",
    "conclusion",
)


def empty_section_map() -> dict[SectionName, str]:
    """Return an empty section map containing all supported section keys."""
    return dict.fromkeys(SECTION_NAMES, "")
