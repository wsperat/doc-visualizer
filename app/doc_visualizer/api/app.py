"""Async FastAPI app exposing Doc Visualizer phase pipelines."""

from __future__ import annotations

from collections.abc import Callable, Coroutine

from fastapi import FastAPI, HTTPException, status

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
)

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
