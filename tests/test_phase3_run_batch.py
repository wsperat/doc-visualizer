from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from doc_visualizer.phase3.models import EngineOutput, TopicCluster, TopicDocument, TopicTerm
from doc_visualizer.phase3.run_batch import load_phase2_corpora, run_phase3_batch


class FakeTopicEngine:
    @property
    def name(self) -> str:
        return "fake"

    def fit(
        self,
        documents: Sequence[TopicDocument],
        *,
        top_n_terms: int,
        n_topics: int | None,
    ) -> EngineOutput:
        _ = top_n_terms
        _ = n_topics
        assignments = {document.document_id: 0 for document in documents}
        topic = TopicCluster(
            topic_id=0,
            terms=(TopicTerm(term="topic", weight=1.0),),
            document_ids=tuple(assignments.keys()),
        )
        return EngineOutput(
            engine=self.name,
            topics=(topic,),
            assignments=assignments,
            metadata={"engine": "fake"},
        )


def test_load_phase2_corpora_supports_namespaced_sources(tmp_path: Path) -> None:
    phase2_output = tmp_path / "phase2_output"
    (phase2_output / "whole_doc_mean_pool").mkdir(parents=True)
    (phase2_output / "grobid" / "whole_doc_mean_pool").mkdir(parents=True)
    (phase2_output / "raw" / "weighted_pooling").mkdir(parents=True)

    (phase2_output / "grobid" / "whole_doc_mean_pool" / "paper.json").write_text(
        json.dumps(
            {
                "inputs": [
                    {"section": "abstract", "text": "grobid abstract text", "chunk_index": None}
                ]
            }
        ),
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
    (phase2_output / "whole_doc_mean_pool" / "paper.json").write_text(
        json.dumps(
            {"inputs": [{"section": "abstract", "text": "legacy root text", "chunk_index": None}]}
        ),
        encoding="utf-8",
    )

    corpora = load_phase2_corpora(phase2_output)

    assert {(corpus.source, corpus.strategy) for corpus in corpora} == {
        ("grobid", "whole_doc_mean_pool"),
        ("raw", "weighted_pooling"),
    }

    weighted_corpus = next(corpus for corpus in corpora if corpus.strategy == "weighted_pooling")
    assert weighted_corpus.documents[0].text == "raw chapter text\n\nraw chapter text"


def test_run_phase3_batch_writes_engine_outputs_and_report(tmp_path: Path) -> None:
    phase2_output = tmp_path / "phase2_output"
    (phase2_output / "whole_doc_mean_pool").mkdir(parents=True)
    (phase2_output / "parent_child_prepend").mkdir(parents=True)

    payload = {
        "inputs": [
            {"section": "abstract", "text": "alpha beta", "chunk_index": None},
            {"section": "results", "text": "gamma delta", "chunk_index": None},
        ]
    }
    (phase2_output / "whole_doc_mean_pool" / "paper.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    (phase2_output / "parent_child_prepend" / "paper.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    phase3_output = tmp_path / "phase3_output"
    results = run_phase3_batch(
        phase2_output_dir=phase2_output,
        phase3_output_dir=phase3_output,
        engines=[FakeTopicEngine()],
        top_n_terms=5,
        n_topics=3,
    )

    assert len(results) == 2
    assert all(result.status == "ok" for result in results)

    whole_doc_engine_output = phase3_output / "default" / "whole_doc_mean_pool" / "fake.json"
    parent_child_engine_output = phase3_output / "default" / "parent_child_prepend" / "fake.json"
    report_path = phase3_output / "report.json"

    assert whole_doc_engine_output.exists()
    assert parent_child_engine_output.exists()
    assert report_path.exists()

    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(report_payload) == 2
    assert all(item["status"] == "ok" for item in report_payload)
