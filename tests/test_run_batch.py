from __future__ import annotations

import json
from pathlib import Path

from doc_visualizer.phase1.metadata import MetadataJsonWriter
from doc_visualizer.phase1.models import DocumentSections, PaperMetadata, ParsedDocument, RawSection
from doc_visualizer.phase1.run_batch import _process_one_pdf


class StubPipeline:
    def process_pdf(self, pdf_path: Path) -> ParsedDocument:
        _ = pdf_path
        return ParsedDocument(
            sections=DocumentSections(introduction="Intro text"),
            metadata=PaperMetadata(
                title="Sample",
                authors=("Alice",),
                year=2024,
                references=("Ref A",),
            ),
            raw_sections=(
                RawSection(title="Chapter 1", text="Body", level=1, position=1),
                RawSection(title="Defining terms", text="Glossary", level=2, position=2),
            ),
        )


def test_process_one_pdf_writes_grobid_and_raw_content(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"pdf")

    result = _process_one_pdf(
        pdf_path=pdf_path,
        pipeline=StubPipeline(),
        metadata_writer=MetadataJsonWriter(),
        metadata_dir=tmp_path / "metadata",
        grobid_content_dir=tmp_path / "grobid_content",
        raw_content_dir=tmp_path / "raw_content",
    )

    assert result.status == "ok"

    grobid_payload = json.loads(
        (tmp_path / "grobid_content" / "paper.json").read_text(encoding="utf-8")
    )
    raw_payload = json.loads((tmp_path / "raw_content" / "paper.json").read_text(encoding="utf-8"))
    metadata_payload = json.loads(
        (tmp_path / "metadata" / "paper.json").read_text(encoding="utf-8")
    )

    assert grobid_payload == {"introduction": "Intro text"}
    assert raw_payload == [
        {"level": 1, "position": 1, "text": "Body", "title": "Chapter 1"},
        {"level": 2, "position": 2, "text": "Glossary", "title": "Defining terms"},
    ]
    assert metadata_payload["title"] == "Sample"
