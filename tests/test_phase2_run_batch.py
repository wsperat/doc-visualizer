from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc_visualizer.phase2.run_batch import _load_section_weights, run_phase2_batch


def test_run_phase2_batch_writes_all_strategy_outputs(tmp_path: Path) -> None:
    phase1_output_dir = tmp_path / "phase1_output"
    (phase1_output_dir / "grobid_content").mkdir(parents=True)
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


def test_load_section_weights_validates_json_shape(tmp_path: Path) -> None:
    bad_weights_path = tmp_path / "weights.json"
    bad_weights_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(ValueError, match="must contain an object"):
        _load_section_weights(bad_weights_path)
