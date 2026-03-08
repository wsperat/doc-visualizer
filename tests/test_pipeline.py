from __future__ import annotations

from pathlib import Path

from doc_visualizer.phase1.models import DocumentSections, PaperMetadata, ParsedDocument
from doc_visualizer.phase1.pipeline import PhaseOnePipeline
from doc_visualizer.phase1.protocols import GrobidXmlSource, TeiDocumentParser


class StubXmlSource(GrobidXmlSource):
    def __init__(self, tei_xml: str) -> None:
        self.tei_xml = tei_xml
        self.called_with: Path | None = None

    def process_pdf(self, pdf_path: Path) -> str:
        self.called_with = pdf_path
        return self.tei_xml


class StubTeiParser(TeiDocumentParser):
    def __init__(self, parsed_document: ParsedDocument) -> None:
        self.parsed_document = parsed_document
        self.called_with: str | None = None

    def parse(self, tei_xml: str) -> ParsedDocument:
        self.called_with = tei_xml
        return self.parsed_document


def test_phase_one_pipeline_orchestrates_dependencies() -> None:
    expected_document = ParsedDocument(
        sections=DocumentSections(introduction="Structured introduction"),
        metadata=PaperMetadata(
            title="Example",
            authors=("Alice Smith",),
            year=2024,
            references=("Alice Smith (2020) Ref",),
        ),
    )
    xml_source = StubXmlSource("<TEI/>")
    parser = StubTeiParser(expected_document)
    pipeline = PhaseOnePipeline(xml_source=xml_source, tei_parser=parser)

    result = pipeline.process_pdf(Path("paper.pdf"))

    assert result == expected_document
    assert xml_source.called_with == Path("paper.pdf")
    assert parser.called_with == "<TEI/>"
