"""Phase 1 orchestration service."""

from __future__ import annotations

from pathlib import Path

from doc_visualizer.phase1.models import ParsedDocument
from doc_visualizer.phase1.protocols import GrobidXmlSource, TeiDocumentParser


class PhaseOnePipeline:
    """Coordinate PDF -> TEI extraction and TEI -> domain parsing."""

    def __init__(self, xml_source: GrobidXmlSource, tei_parser: TeiDocumentParser) -> None:
        self._xml_source = xml_source
        self._tei_parser = tei_parser

    def process_pdf(self, pdf_path: Path) -> ParsedDocument:
        """Execute Phase 1 for a single PDF."""
        tei_xml = self._xml_source.process_pdf(pdf_path)
        return self._tei_parser.parse(tei_xml)
