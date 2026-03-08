"""Streamlit frontend service for orchestrating Doc Visualizer pipelines."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypedDict, cast

import httpx
import streamlit as st
from streamlit.components.v1 import html as render_html

POLL_INTERVAL_SECONDS = 1.0
DEFAULT_TIMEOUT_SECONDS = 3600.0


class JobPayload(TypedDict):
    job_id: str
    phase: str
    status: str
    result: dict[str, object] | None
    error: str | None


class UploadedFileLike(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


class StatusPlaceholderLike(Protocol):
    def info(self, body: str) -> object: ...

    def success(self, body: str) -> object: ...


class DetailsPlaceholderLike(Protocol):
    def write(self, body: str) -> object: ...


@dataclass(frozen=True, slots=True)
class PipelineStep:
    """One API job step executed by the frontend orchestrator."""

    name: str
    endpoint: str
    payload: dict[str, object]


def main() -> None:
    st.set_page_config(page_title="Doc Visualizer Frontend", layout="wide")
    st.title("Doc Visualizer Frontend Service")
    st.caption("Upload PDFs, run async backend phases, and inspect the final interactive map.")

    with st.sidebar:
        st.subheader("Backend")
        default_api_url = os.getenv("DOC_VIS_API_BASE_URL", "http://localhost:8000")
        api_base_url = st.text_input("FastAPI base URL", value=default_api_url)
        upload_dir = st.text_input("Backend upload directory", value="data/uploads")
        st.divider()
        st.subheader("Output Dirs")
        phase1_output = st.text_input("Phase 1 output", value="data/phase1_output")
        phase2_output = st.text_input("Phase 2 output", value="data/phase2_output")
        phase3_output = st.text_input("Phase 3 output", value="data/phase3_output")
        phase4_output = st.text_input("Phase 4 output", value="data/phase4_output")
        phase5_output = st.text_input("Phase 5 output", value="data/phase5_output")

    _render_upload_section(api_base_url=api_base_url, upload_dir=upload_dir)
    st.divider()
    _render_run_section(
        api_base_url=api_base_url,
        upload_dir=upload_dir,
        phase1_output=phase1_output,
        phase2_output=phase2_output,
        phase3_output=phase3_output,
        phase4_output=phase4_output,
        phase5_output=phase5_output,
    )
    st.divider()
    _render_plot_section(api_base_url=api_base_url, phase5_output=phase5_output)


def _render_upload_section(*, api_base_url: str, upload_dir: str) -> None:
    st.header("1. Upload Files")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Files are uploaded to the backend so Phase 1 can process them.",
    )
    if st.button("Upload to Backend", type="primary"):
        if not uploaded_files:
            st.warning("Select at least one PDF file.")
            return

        with st.spinner("Uploading files..."):
            try:
                saved_files = _upload_files(
                    api_base_url=api_base_url,
                    upload_dir=upload_dir,
                    uploaded_files=uploaded_files,
                )
            except Exception as exc:
                st.error(f"Upload failed: {exc}")
                return

        st.success(f"Uploaded {len(saved_files)} file(s) to {upload_dir}.")
        for saved_file in saved_files:
            st.write(f"- {saved_file}")


def _render_run_section(
    *,
    api_base_url: str,
    upload_dir: str,
    phase1_output: str,
    phase2_output: str,
    phase3_output: str,
    phase4_output: str,
    phase5_output: str,
) -> None:
    st.header("2. Run Processing Pipeline")
    content_source = st.selectbox(
        "Phase 2 content source",
        options=["both", "grobid", "raw", "hybrid"],
    )
    engines_csv = st.text_input("Phase 3 engines (comma separated)", value="lda")

    if st.button("Run Full Pipeline", type="primary"):
        steps = [
            PipelineStep(
                name="Phase 1: Structural Parsing",
                endpoint="/phases/1/run",
                payload={
                    "input_dir": upload_dir,
                    "output_dir": phase1_output,
                },
            ),
            PipelineStep(
                name="Phase 2: Context Strategies",
                endpoint="/phases/2/run",
                payload={
                    "phase1_output_dir": phase1_output,
                    "phase2_output_dir": phase2_output,
                    "content_source": content_source,
                },
            ),
            PipelineStep(
                name="Phase 3: Topic Modeling",
                endpoint="/phases/3/run",
                payload={
                    "phase2_output_dir": phase2_output,
                    "phase3_output_dir": phase3_output,
                    "engines": [part.strip() for part in engines_csv.split(",") if part.strip()],
                },
            ),
            PipelineStep(
                name="Phase 4: Summarization + QC",
                endpoint="/phases/4/run",
                payload={
                    "phase2_output_dir": phase2_output,
                    "phase4_output_dir": phase4_output,
                },
            ),
            PipelineStep(
                name="Phase 5: Mapping",
                endpoint="/phases/5/run",
                payload={
                    "phase2_output_dir": phase2_output,
                    "phase4_output_dir": phase4_output,
                    "phase5_output_dir": phase5_output,
                },
            ),
            PipelineStep(
                name="Phase 5 Plot: HTML Generation",
                endpoint="/phases/5/plot",
                payload={
                    "phase5_output_dir": phase5_output,
                    "metadata_dir": f"{phase1_output}/metadata",
                    "output_html": f"{phase5_output}/document_map.html",
                },
            ),
        ]

        status_placeholder = st.empty()
        details_placeholder = st.empty()
        try:
            _run_steps(
                api_base_url=api_base_url,
                steps=steps,
                status_placeholder=status_placeholder,
                details_placeholder=details_placeholder,
            )
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            return

        st.success("Pipeline completed successfully.")


def _render_plot_section(*, api_base_url: str, phase5_output: str) -> None:
    st.header("3. Interactive Document Map")
    if st.button("Load Latest Plot"):
        with st.spinner("Loading plot..."):
            try:
                plot_html = _fetch_plot_html(
                    api_base_url=api_base_url,
                    output_html=f"{phase5_output}/document_map.html",
                )
            except Exception as exc:
                st.error(f"Could not load plot: {exc}")
                return
        render_html(plot_html, height=780, scrolling=True)


def _upload_files(
    *,
    api_base_url: str,
    upload_dir: str,
    uploaded_files: Sequence[UploadedFileLike],
) -> list[str]:
    multipart_files: list[tuple[str, tuple[str, bytes, str]]] = []
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_content = uploaded_file.getvalue()
        multipart_files.append(("files", (file_name, file_content, "application/pdf")))

    response = _request(
        api_base_url=api_base_url,
        method="POST",
        endpoint="/files/upload",
        data={"target_dir": upload_dir},
        files=multipart_files,
    )
    payload = cast(dict[str, object], response.json())
    saved_files = payload.get("saved_files")
    if not isinstance(saved_files, list):
        raise RuntimeError("Invalid upload response payload")
    return [str(item) for item in saved_files]


def _run_steps(
    *,
    api_base_url: str,
    steps: list[PipelineStep],
    status_placeholder: StatusPlaceholderLike,
    details_placeholder: DetailsPlaceholderLike,
) -> None:
    for index, step in enumerate(steps, start=1):
        status_placeholder.info(f"[{index}/{len(steps)}] Submitting {step.name}...")
        job_id = _submit_job(
            api_base_url=api_base_url,
            endpoint=step.endpoint,
            payload=step.payload,
        )

        while True:
            job = _get_job(api_base_url=api_base_url, job_id=job_id)
            details_placeholder.write(f"{step.name}: status={job['status']} (job_id={job_id})")

            if job["status"] == "completed":
                break
            if job["status"] == "failed":
                raise RuntimeError(f"{step.name} failed: {job.get('error')}")
            time.sleep(POLL_INTERVAL_SECONDS)

    status_placeholder.success("All steps completed.")


def _submit_job(*, api_base_url: str, endpoint: str, payload: dict[str, object]) -> str:
    response = _request(
        api_base_url=api_base_url,
        method="POST",
        endpoint=endpoint,
        json=payload,
    )
    response_payload = cast(dict[str, object], response.json())
    job_id = response_payload.get("job_id")
    if not isinstance(job_id, str):
        raise RuntimeError(f"Invalid job response from {endpoint}")
    return job_id


def _get_job(*, api_base_url: str, job_id: str) -> JobPayload:
    response = _request(
        api_base_url=api_base_url,
        method="GET",
        endpoint=f"/jobs/{job_id}",
    )
    payload = cast(dict[str, object], response.json())
    return {
        "job_id": str(payload.get("job_id", "")),
        "phase": str(payload.get("phase", "")),
        "status": str(payload.get("status", "")),
        "result": cast(dict[str, object] | None, payload.get("result")),
        "error": cast(str | None, payload.get("error")),
    }


def _fetch_plot_html(*, api_base_url: str, output_html: str) -> str:
    response = _request(
        api_base_url=api_base_url,
        method="GET",
        endpoint="/phase5/plot-html",
        params={"output_html": output_html},
    )
    return response.text


def _request(
    *,
    api_base_url: str,
    method: str,
    endpoint: str,
    json: dict[str, object] | None = None,
    data: dict[str, str] | None = None,
    files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
    params: dict[str, str] | None = None,
) -> httpx.Response:
    with httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
        response = client.request(
            method=method,
            url=f"{api_base_url.rstrip('/')}{endpoint}",
            json=json,
            data=data,
            files=files,
            params=params,
        )
    response.raise_for_status()
    return response


if __name__ == "__main__":
    main()
