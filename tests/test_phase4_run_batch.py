from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from doc_visualizer.phase2.models import Vector
from doc_visualizer.phase4.run_batch import load_phase2_documents, run_phase4_batch


class ConstantSummarizer:
    @property
    def name(self) -> str:
        return "constant_summary"

    def summarize(self, text: str) -> str:
        _ = text
        return "shared summary"


class MappingEmbeddingBackend:
    def __init__(self, vectors: Mapping[str, Vector]) -> None:
        self._vectors = dict(vectors)

    @property
    def name(self) -> str:
        return "mapping_embed"

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._vectors[text] for text in texts]


def test_load_phase2_documents_supports_namespaced_sources(tmp_path: Path) -> None:
    phase2_output = tmp_path / "phase2_output"
    (phase2_output / "whole_doc_mean_pool").mkdir(parents=True)
    (phase2_output / "grobid" / "whole_doc_mean_pool").mkdir(parents=True)
    (phase2_output / "raw" / "weighted_pooling").mkdir(parents=True)

    payload = {"inputs": [{"section": "abstract", "text": "alpha text", "chunk_index": None}]}
    (phase2_output / "whole_doc_mean_pool" / "paper.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    (phase2_output / "grobid" / "whole_doc_mean_pool" / "paper.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    (phase2_output / "raw" / "weighted_pooling" / "paper.json").write_text(
        json.dumps(
            {
                "inputs": [
                    {"section": "chapter_1", "text": "raw chapter text", "chunk_index": None}
                ],
                "section_weights": {"chapter_1": 2.0},
            }
        ),
        encoding="utf-8",
    )

    documents = load_phase2_documents(phase2_output)

    assert {(document.source, document.strategy) for document in documents} == {
        ("grobid", "whole_doc_mean_pool"),
        ("raw", "weighted_pooling"),
    }

    weighted_document = next(
        document for document in documents if document.strategy == "weighted_pooling"
    )
    assert weighted_document.section_weights == {"chapter_1": 2.0}


def test_run_phase4_batch_writes_outputs_and_flags_low_similarity(tmp_path: Path) -> None:
    phase2_output = tmp_path / "phase2_output"
    (phase2_output / "whole_doc_mean_pool").mkdir(parents=True)
    (phase2_output / "weighted_pooling").mkdir(parents=True)

    (phase2_output / "whole_doc_mean_pool" / "paper.json").write_text(
        json.dumps(
            {
                "inputs": [
                    {"section": "abstract", "text": "alpha text", "chunk_index": None},
                    {"section": "results", "text": "beta text", "chunk_index": None},
                ]
            }
        ),
        encoding="utf-8",
    )
    (phase2_output / "weighted_pooling" / "paper.json").write_text(
        json.dumps(
            {
                "inputs": [
                    {"section": "abstract", "text": "alpha text", "chunk_index": None},
                    {"section": "results", "text": "beta text", "chunk_index": None},
                ],
                "section_weights": {"abstract": 1.0, "results": 3.0},
            }
        ),
        encoding="utf-8",
    )

    phase4_output = tmp_path / "phase4_output"
    results = run_phase4_batch(
        phase2_output_dir=phase2_output,
        phase4_output_dir=phase4_output,
        summarizer=ConstantSummarizer(),
        embedding_backend=MappingEmbeddingBackend(
            {
                "alpha text": (1.0, 0.0),
                "beta text": (0.0, 1.0),
                "shared summary": (1.0, 1.0),
            }
        ),
        similarity_threshold=0.95,
    )

    assert len(results) == 2
    assert {result.status for result in results} == {"ok", "low_similarity"}

    whole_doc_output = phase4_output / "default" / "whole_doc_mean_pool" / "paper.json"
    weighted_output = phase4_output / "default" / "weighted_pooling" / "paper.json"
    report_path = phase4_output / "report.json"

    assert whole_doc_output.exists()
    assert weighted_output.exists()
    assert report_path.exists()

    whole_doc_payload = json.loads(whole_doc_output.read_text(encoding="utf-8"))
    weighted_payload = json.loads(weighted_output.read_text(encoding="utf-8"))
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert whole_doc_payload["is_below_threshold"] is False
    assert weighted_payload["is_below_threshold"] is True
    assert len(report_payload) == 2
