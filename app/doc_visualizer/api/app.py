"""Async FastAPI app exposing Doc Visualizer phase pipelines."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse

from doc_visualizer.api.executor import (
    execute_phase1,
    execute_phase2,
    execute_phase3,
    execute_phase4,
    execute_phase5,
    execute_phase5_plot,
)
from doc_visualizer.api.jobs import InMemoryJobManager
from doc_visualizer.api.models import (
    JobResponse,
    Phase1RunRequest,
    Phase2RunRequest,
    Phase3RunRequest,
    Phase4RunRequest,
    Phase5PlotRequest,
    Phase5RunRequest,
    UploadResponse,
)
from doc_visualizer.phase5.plot_records import load_plot_records

app = FastAPI(
    title="Doc Visualizer API",
    version="0.1.0",
    description="Asynchronous API for running document-processing phases.",
)
job_manager = InMemoryJobManager()


@app.get("/health")
async def health() -> dict[str, str]:
    """Service health-check endpoint."""
    return {"status": "ok"}


@app.post("/files/upload", response_model=UploadResponse)
async def upload_files(
    files: Annotated[list[UploadFile], File(...)],
    target_dir: Annotated[str, Form()] = "data/uploads",
) -> UploadResponse:
    """Upload PDF files into a backend-accessible directory."""
    resolved_dir = Path(target_dir).resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[str] = []
    for upload in files:
        if not upload.filename:
            continue

        safe_name = Path(upload.filename).name
        if not safe_name.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Only PDF uploads are supported: '{safe_name}'",
            )

        target_path = resolved_dir / safe_name
        file_bytes = await upload.read()
        target_path.write_bytes(file_bytes)
        saved_files.append(str(target_path))

    return UploadResponse(saved_files=saved_files, target_dir=str(resolved_dir))


@app.get("/phase5/plot-html", response_class=HTMLResponse)
async def get_phase5_plot_html(
    output_html: str = "data/phase5_output/document_map.html",
) -> HTMLResponse:
    """Return generated Phase 5 plot HTML for frontend embedding."""
    html_path = Path(output_html).resolve()
    if not html_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plot HTML not found at '{html_path}'",
        )
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/phase5/map-records")
async def get_phase5_map_records(
    phase5_output_dir: str = "data/phase5_output",
    metadata_dir: str = "data/phase1_output/metadata",
) -> dict[str, object]:
    """Return title-enriched map points for custom frontend visualizations."""
    records = load_plot_records(
        phase5_output_dir=Path(phase5_output_dir).resolve(),
        metadata_dir=Path(metadata_dir).resolve(),
    )
    return {
        "records": [record.to_dict() for record in records],
        "count": len(records),
    }


@app.get("/jobs", response_model=list[JobResponse])
async def list_jobs() -> list[JobResponse]:
    records = await job_manager.list()
    return [record.to_response() for record in records]


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobResponse:
    record = await job_manager.get(job_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' was not found",
        )
    return record.to_response()


@app.post("/phases/1/run", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def run_phase1(
    request: Phase1RunRequest | None = None,
) -> JobResponse:
    resolved_request = request or Phase1RunRequest()
    return await _submit("phase1", lambda: execute_phase1(resolved_request))


@app.post("/phases/2/run", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def run_phase2(
    request: Phase2RunRequest | None = None,
) -> JobResponse:
    resolved_request = request or Phase2RunRequest()
    return await _submit("phase2", lambda: execute_phase2(resolved_request))


@app.post("/phases/3/run", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def run_phase3(
    request: Phase3RunRequest | None = None,
) -> JobResponse:
    resolved_request = request or Phase3RunRequest()
    return await _submit("phase3", lambda: execute_phase3(resolved_request))


@app.post("/phases/4/run", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def run_phase4(
    request: Phase4RunRequest | None = None,
) -> JobResponse:
    resolved_request = request or Phase4RunRequest()
    return await _submit("phase4", lambda: execute_phase4(resolved_request))


@app.post("/phases/5/run", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def run_phase5(
    request: Phase5RunRequest | None = None,
) -> JobResponse:
    resolved_request = request or Phase5RunRequest()
    return await _submit("phase5", lambda: execute_phase5(resolved_request))


@app.post(
    "/phases/5/plot",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def run_phase5_plot(
    request: Phase5PlotRequest | None = None,
) -> JobResponse:
    resolved_request = request or Phase5PlotRequest()
    return await _submit("phase5_plot", lambda: execute_phase5_plot(resolved_request))


async def _submit(
    phase: str,
    runner: Callable[[], Coroutine[object, object, dict[str, object]]],
) -> JobResponse:
    record = await job_manager.submit(phase=phase, runner=runner)
    return record.to_response()
