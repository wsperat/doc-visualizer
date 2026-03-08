from __future__ import annotations

import json
import time
from pathlib import Path
from typing import cast

from fastapi.testclient import TestClient

from doc_visualizer.api.app import app


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_phase2_job_runs_asynchronously_and_writes_outputs(tmp_path: Path) -> None:
    client = TestClient(app)

    phase1_output = tmp_path / "phase1_output"
    (phase1_output / "grobid_content").mkdir(parents=True)
    (phase1_output / "raw_content").mkdir(parents=True)
    (phase1_output / "metadata").mkdir(parents=True)

    (phase1_output / "grobid_content" / "paper.json").write_text(
        json.dumps({"abstract": "short abstract text"}),
        encoding="utf-8",
    )
    (phase1_output / "metadata" / "paper.json").write_text(
        json.dumps({"title": "Paper Title"}),
        encoding="utf-8",
    )

    phase2_output = tmp_path / "phase2_output"
    response = client.post(
        "/phases/2/run",
        json={
            "phase1_output_dir": str(phase1_output),
            "phase2_output_dir": str(phase2_output),
            "content_source": "grobid",
            "max_tokens": 128,
        },
    )

    assert response.status_code == 202
    response_payload = cast(dict[str, object], response.json())
    job_id = response_payload["job_id"]
    assert isinstance(job_id, str)
    completed_job = _wait_for_job_completion(client, job_id)
    completed_result = completed_job["result"]
    assert isinstance(completed_result, dict)

    assert completed_job["status"] == "completed"
    assert completed_result["succeeded"] == 1
    assert completed_result["failed"] == 0
    assert (phase2_output / "whole_doc_mean_pool" / "paper.json").exists()
    assert (phase2_output / "report.json").exists()


def test_phase5_plot_job_generates_html(tmp_path: Path) -> None:
    client = TestClient(app)

    phase5_output = tmp_path / "phase5_output"
    (phase5_output / "grobid" / "whole_doc_mean_pool").mkdir(parents=True)
    (phase5_output / "grobid" / "whole_doc_mean_pool" / "map.json").write_text(
        json.dumps(
            {
                "points": [
                    {
                        "document_id": "paper",
                        "cluster_id": 0,
                        "x": 1.0,
                        "y": 2.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "paper.json").write_text(
        json.dumps({"title": "Paper Title"}),
        encoding="utf-8",
    )

    output_html = tmp_path / "document_map.html"
    response = client.post(
        "/phases/5/plot",
        json={
            "phase5_output_dir": str(phase5_output),
            "metadata_dir": str(metadata_dir),
            "output_html": str(output_html),
        },
    )

    assert response.status_code == 202
    response_payload = cast(dict[str, object], response.json())
    job_id = response_payload["job_id"]
    assert isinstance(job_id, str)
    completed_job = _wait_for_job_completion(client, job_id)

    assert completed_job["status"] == "completed"
    assert output_html.exists()
    html_content = output_html.read_text(encoding="utf-8")
    assert "Paper Title" in html_content


def _wait_for_job_completion(client: TestClient, job_id: str) -> dict[str, object]:
    for _ in range(100):
        response = client.get(f"/jobs/{job_id}")
        response.raise_for_status()
        payload = cast(dict[str, object], response.json())
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"Job {job_id} did not complete in time")
