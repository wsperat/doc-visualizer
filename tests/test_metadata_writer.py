from __future__ import annotations

import json
from pathlib import Path

from doc_visualizer.phase1.metadata import MetadataJsonWriter
from doc_visualizer.phase1.models import PaperMetadata


def test_metadata_writer_persists_json_payload(tmp_path: Path) -> None:
    writer = MetadataJsonWriter()
    metadata = PaperMetadata(
        title="Paper",
        authors=("Alice Smith", "Bob Jones"),
        year=2024,
        references=("A Ref", "B Ref"),
    )
    output_path = tmp_path / "metadata" / "paper.json"

    writer.write(metadata, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "title": "Paper",
        "authors": ["Alice Smith", "Bob Jones"],
        "year": 2024,
        "references": ["A Ref", "B Ref"],
    }
