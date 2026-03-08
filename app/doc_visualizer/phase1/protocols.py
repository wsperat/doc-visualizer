"""Abstractions used to keep parsing pipeline composable and testable."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from doc_visualizer.phase1.models import ParsedDocument


class GrobidXmlSource(Protocol):
    """Dependency for converting a PDF into GROBID TEI XML."""

    def process_pdf(self, pdf_path: Path) -> str:
        """Return TEI XML produced by GROBID for the provided PDF path."""


class TeiDocumentParser(Protocol):
    """Dependency for converting TEI XML into strongly typed domain objects."""

    def parse(self, tei_xml: str) -> ParsedDocument:
        """Parse TEI XML into core sections and isolated metadata."""
