from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc_visualizer.phase2.run_batch import ContentSource, _load_section_weights, run_phase2_batch


def test_run_phase2_batch_writes_all_strategy_outputs(tmp_path: Path) -> None:
    phase1_output_dir = tmp_path / "phase1_output"
    (phase1_output_dir / "grobid_content").mkdir(parents=True)
    (phase1_output_dir / "raw_content").mkdir(parents=True)
    (phase1_output_dir / "metadata").mkdir(parents=True)

    (phase1_output_dir / "grobid_content" / "paper.json").write_text(
        json.dumps(
            {
                "abstract": "why this matters",
                "introduction": "intro section text",
                "results": "result text section",
            }
        ),
        encoding="utf-8",
    )
    (phase1_output_dir / "metadata" / "paper.json").write_text(
        json.dumps({"title": "Paper Title"}),
        encoding="utf-8",
    )

    phase2_output_dir = tmp_path / "phase2_output"
    results = run_phase2_batch(
        phase1_output_dir=phase1_output_dir,
        phase2_output_dir=phase2_output_dir,
        max_tokens=2,
        section_weights={"results": 3.0},
    )

    assert len(results) == 1
    assert results[0].status == "ok"

    whole_doc_path = phase2_output_dir / "whole_doc_mean_pool" / "paper.json"
    parent_child_path = phase2_output_dir / "parent_child_prepend" / "paper.json"
    weighted_path = phase2_output_dir / "weighted_pooling" / "paper.json"
    report_path = phase2_output_dir / "report.json"

    assert whole_doc_path.exists()
    assert parent_child_path.exists()
    assert weighted_path.exists()
    assert report_path.exists()

    parent_payload = json.loads(parent_child_path.read_text(encoding="utf-8"))
    assert parent_payload["strategy"] == "parent_child_prepend"
    assert parent_payload["max_tokens"] == 2
    assert len(parent_payload["inputs"]) > 1
    assert "Title: Paper Title" in parent_payload["inputs"][0]["text"]

    weighted_payload = json.loads(weighted_path.read_text(encoding="utf-8"))
    assert weighted_payload["section_weights"]["results"] == 3.0
    assert weighted_payload["section_weights"]["abstract"] == 1.0


def test_run_phase2_batch_can_use_raw_content_hierarchy(tmp_path: Path) -> None:
    phase1_output_dir = tmp_path / "phase1_output"
    (phase1_output_dir / "grobid_content").mkdir(parents=True)
    (phase1_output_dir / "raw_content").mkdir(parents=True)
    (phase1_output_dir / "metadata").mkdir(parents=True)

    (phase1_output_dir / "raw_content" / "paper.json").write_text(
        json.dumps(
            [
                {
                    "title": "Chapter 1",
                    "text": "Top-level chapter text.",
                    "level": 1,
                    "position": 1,
                },
                {
                    "title": "Defining terms",
                    "text": "Section detail text.",
                    "level": 2,
                    "position": 2,
                },
                {
                    "title": "Types and recommendations",
                    "text": "Subsection detail text.",
                    "level": 3,
                    "position": 3,
                },
            ]
        ),
        encoding="utf-8",
    )
    (phase1_output_dir / "metadata" / "paper.json").write_text(
        json.dumps({"title": "Paper Title"}),
        encoding="utf-8",
    )

    phase2_output_dir = tmp_path / "phase2_output"
    results = run_phase2_batch(
        phase1_output_dir=phase1_output_dir,
        phase2_output_dir=phase2_output_dir,
        content_source=ContentSource.RAW,
    )

    assert len(results) == 1
    assert results[0].status == "ok"

    whole_doc_payload = json.loads(
        (phase2_output_dir / "whole_doc_mean_pool" / "paper.json").read_text(encoding="utf-8")
    )
    weighted_payload = json.loads(
        (phase2_output_dir / "weighted_pooling" / "paper.json").read_text(encoding="utf-8")
    )

    sections = [item["section"] for item in whole_doc_payload["inputs"]]
    assert sections == [
        "chapter_1",
        "chapter_1_defining_terms",
        "chapter_1_defining_terms_types_and_recommendations",
    ]

    hierarchy = whole_doc_payload["raw_section_hierarchy"]
    assert hierarchy["chapter_1"]["level"] == 1
    assert hierarchy["chapter_1_defining_terms"]["level"] == 2
    assert hierarchy["chapter_1_defining_terms_types_and_recommendations"]["level"] == 3

    assert weighted_payload["section_weights"]["chapter_1"] == 1.0
    assert weighted_payload["section_weights"]["chapter_1_defining_terms"] == 0.85
    assert (
        weighted_payload["section_weights"]["chapter_1_defining_terms_types_and_recommendations"]
        == 0.7
    )


def test_run_phase2_batch_hybrid_merges_grobid_and_raw_content(tmp_path: Path) -> None:
    phase1_output_dir = tmp_path / "phase1_output"
    (phase1_output_dir / "grobid_content").mkdir(parents=True)
    (phase1_output_dir / "raw_content").mkdir(parents=True)
    (phase1_output_dir / "metadata").mkdir(parents=True)

    (phase1_output_dir / "grobid_content" / "paper.json").write_text(
        json.dumps({"abstract": "why this matters"}),
        encoding="utf-8",
    )
    (phase1_output_dir / "raw_content" / "paper.json").write_text(
        json.dumps(
            [
                {
                    "title": "Chapter 1",
                    "text": "Top-level chapter text.",
                    "level": 1,
                    "position": 1,
                }
            ]
        ),
        encoding="utf-8",
    )
    (phase1_output_dir / "metadata" / "paper.json").write_text(
        json.dumps({"title": "Paper Title"}),
        encoding="utf-8",
    )

    phase2_output_dir = tmp_path / "phase2_output"
    results = run_phase2_batch(
        phase1_output_dir=phase1_output_dir,
        phase2_output_dir=phase2_output_dir,
        content_source=ContentSource.HYBRID,
    )

    assert len(results) == 1
    assert results[0].status == "ok"

    whole_doc_payload = json.loads(
        (phase2_output_dir / "whole_doc_mean_pool" / "paper.json").read_text(encoding="utf-8")
    )
    sections = [item["section"] for item in whole_doc_payload["inputs"]]

    assert "abstract" in sections
    assert "chapter_1" in sections


def test_run_phase2_batch_both_writes_source_specific_folders(tmp_path: Path) -> None:
    phase1_output_dir = tmp_path / "phase1_output"
    (phase1_output_dir / "grobid_content").mkdir(parents=True)
    (phase1_output_dir / "raw_content").mkdir(parents=True)
    (phase1_output_dir / "metadata").mkdir(parents=True)

    (phase1_output_dir / "grobid_content" / "paper.json").write_text(
        json.dumps({"abstract": "why this matters"}),
        encoding="utf-8",
    )
    (phase1_output_dir / "raw_content" / "paper.json").write_text(
        json.dumps(
            [
                {
                    "title": "Chapter 1",
                    "text": "Top-level chapter text.",
                    "level": 1,
                    "position": 1,
                }
            ]
        ),
        encoding="utf-8",
    )
    (phase1_output_dir / "metadata" / "paper.json").write_text(
        json.dumps({"title": "Paper Title"}),
        encoding="utf-8",
    )

    phase2_output_dir = tmp_path / "phase2_output"
    results = run_phase2_batch(
        phase1_output_dir=phase1_output_dir,
        phase2_output_dir=phase2_output_dir,
        content_source=ContentSource.BOTH,
    )

    assert len(results) == 2
    assert all(result.status == "ok" for result in results)
    assert {result.content_source for result in results} == {"grobid", "raw"}

    grobid_whole_doc = phase2_output_dir / "grobid" / "whole_doc_mean_pool" / "paper.json"
    raw_whole_doc = phase2_output_dir / "raw" / "whole_doc_mean_pool" / "paper.json"
    combined_report = phase2_output_dir / "report.json"
    grobid_report = phase2_output_dir / "grobid" / "report.json"
    raw_report = phase2_output_dir / "raw" / "report.json"

    assert grobid_whole_doc.exists()
    assert raw_whole_doc.exists()
    assert combined_report.exists()
    assert grobid_report.exists()
    assert raw_report.exists()


def test_load_section_weights_validates_json_shape(tmp_path: Path) -> None:
    bad_weights_path = tmp_path / "weights.json"
    bad_weights_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(ValueError, match="must contain an object"):
        _load_section_weights(bad_weights_path)
