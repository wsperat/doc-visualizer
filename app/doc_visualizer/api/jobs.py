"""In-memory job manager for long-running pipeline tasks."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from threading import Lock, Thread
from typing import cast
from uuid import uuid4

from doc_visualizer.api.models import JobResponse, JobState

_UNSET = object()


@dataclass(frozen=True, slots=True)
class JobRecord:
    """Internal immutable job state."""

    job_id: str
    phase: str
    status: JobState
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, object] | None = None
    error: str | None = None

    def to_response(self) -> JobResponse:
        return JobResponse(
            job_id=self.job_id,
            phase=self.phase,
            status=self.status,
            created_at=self.created_at.isoformat(),
            started_at=self.started_at.isoformat() if self.started_at else None,
            finished_at=self.finished_at.isoformat() if self.finished_at else None,
            result=self.result,
            error=self.error,
        )


class InMemoryJobManager:
    """Thread-safe job registry backed by worker threads."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._threads: dict[str, Thread] = {}
        self._lock = Lock()

    async def submit(
        self,
        *,
        phase: str,
        runner: Callable[[], Coroutine[object, object, dict[str, object]]],
    ) -> JobRecord:
        """Queue a job and run it asynchronously."""
        job_id = str(uuid4())
        record = JobRecord(
            job_id=job_id,
            phase=phase,
            status=JobState.QUEUED,
            created_at=_utc_now(),
        )
        with self._lock:
            self._jobs[job_id] = record

        thread = Thread(
            target=self._execute_thread,
            kwargs={"job_id": job_id, "runner": runner},
            daemon=True,
        )
        with self._lock:
            self._threads[job_id] = thread
        thread.start()

        return record

    async def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    async def list(self) -> list[JobRecord]:
        with self._lock:
            return [self._jobs[job_id] for job_id in sorted(self._jobs.keys())]

    def _execute_thread(
        self,
        *,
        job_id: str,
        runner: Callable[[], Coroutine[object, object, dict[str, object]]],
    ) -> None:
        self._update(
            job_id,
            status=JobState.RUNNING,
            started_at=_utc_now(),
            finished_at=_UNSET,
            result=_UNSET,
            error=_UNSET,
        )
        try:
            result: dict[str, object] = asyncio.run(runner())
            self._update(
                job_id,
                status=JobState.COMPLETED,
                finished_at=_utc_now(),
                result=result,
                error=_UNSET,
            )
        except Exception as exc:  # pragma: no cover - integration failure path
            self._update(
                job_id,
                status=JobState.FAILED,
                finished_at=_utc_now(),
                error=str(exc),
            )
        finally:
            with self._lock:
                self._threads.pop(job_id, None)

    def _update(
        self,
        job_id: str,
        *,
        status: JobState | object = _UNSET,
        started_at: datetime | object = _UNSET,
        finished_at: datetime | object = _UNSET,
        result: dict[str, object] | object = _UNSET,
        error: str | object = _UNSET,
    ) -> None:
        with self._lock:
            existing = self._jobs[job_id]
            updated = replace(
                existing,
                status=existing.status if status is _UNSET else cast(JobState, status),
                started_at=(
                    existing.started_at
                    if started_at is _UNSET
                    else cast(datetime | None, started_at)
                ),
                finished_at=(
                    existing.finished_at
                    if finished_at is _UNSET
                    else cast(datetime | None, finished_at)
                ),
                result=(
                    existing.result if result is _UNSET else cast(dict[str, object] | None, result)
                ),
                error=existing.error if error is _UNSET else cast(str | None, error),
            )
            self._jobs[job_id] = updated


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)
