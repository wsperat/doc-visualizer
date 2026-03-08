"""Domain models for parsed documents and metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from doc_visualizer.phase1.types import SectionName, empty_section_map


@dataclass(frozen=True, slots=True)
class PaperMetadata:
    """Bibliographic metadata detached from the main section content."""

    title: str
    authors: tuple[str, ...]
    year: int | None
    references: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "references": list(self.references),
        }


@dataclass(frozen=True, slots=True)
class DocumentSections:
    """Core paper sections used by downstream embeddings."""

    abstract: str = ""
    introduction: str = ""
    methods: str = ""
    results: str = ""
    conclusion: str = ""

    @classmethod
    def from_mapping(cls, sections: Mapping[SectionName, str]) -> DocumentSections:
        """Create section DTO from a section mapping while preserving defaults."""
        defaults = empty_section_map()
        defaults.update(sections)
        return cls(
            abstract=defaults["abstract"],
            introduction=defaults["introduction"],
            methods=defaults["methods"],
            results=defaults["results"],
            conclusion=defaults["conclusion"],
        )

    def to_mapping(self) -> dict[SectionName, str]:
        """Convert to a mutable mapping for functional transforms."""
        return {
            "abstract": self.abstract,
            "introduction": self.introduction,
            "methods": self.methods,
            "results": self.results,
            "conclusion": self.conclusion,
        }

    def non_empty(self) -> dict[SectionName, str]:
        """Return only sections with non-empty text."""
        return {section: text for section, text in self.to_mapping().items() if text}


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Result object for Phase 1 parsing."""

    sections: DocumentSections
    metadata: PaperMetadata

    def content_payload(self) -> dict[SectionName, str]:
        """Embeddings payload without metadata or references."""
        return self.sections.non_empty()

    def metadata_payload(self) -> dict[str, object]:
        """Dashboard payload including references and bibliographic fields."""
        return self.metadata.to_dict()
