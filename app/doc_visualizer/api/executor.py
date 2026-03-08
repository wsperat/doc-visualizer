"""Async execution helpers that bridge API requests to phase pipelines."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from doc_visualizer.api.models import (
    Phase1RunRequest,
    Phase2RunRequest,
    Phase3RunRequest,
    Phase4RunRequest,
    Phase5PlotRequest,
    Phase5RunRequest,
)
from doc_visualizer.phase1.run_batch import run_phase1_batch
from doc_visualizer.phase2.run_batch import ContentSource, run_phase2_batch
from doc_visualizer.phase3.engines import build_engines
from doc_visualizer.phase3.run_batch import run_phase3_batch
from doc_visualizer.phase4.embeddings import build_embedding_backend
from doc_visualizer.phase4.run_batch import run_phase4_batch
from doc_visualizer.phase4.summarizers import build_summarizer
from doc_visualizer.phase5.clusterers import build_clusterer
from doc_visualizer.phase5.generate_plot import build_plot_html
from doc_visualizer.phase5.reducers import build_reducer
from doc_visualizer.phase5.run_batch import run_phase5_batch


async def execute_phase1(request: Phase1RunRequest) -> dict[str, object]:
    return await asyncio.to_thread(_run_phase1_sync, request)


async def execute_phase2(request: Phase2RunRequest) -> dict[str, object]:
    return await asyncio.to_thread(_run_phase2_sync, request)


async def execute_phase3(request: Phase3RunRequest) -> dict[str, object]:
    return await asyncio.to_thread(_run_phase3_sync, request)


async def execute_phase4(request: Phase4RunRequest) -> dict[str, object]:
    return await asyncio.to_thread(_run_phase4_sync, request)


async def execute_phase5(request: Phase5RunRequest) -> dict[str, object]:
    return await asyncio.to_thread(_run_phase5_sync, request)


async def execute_phase5_plot(request: Phase5PlotRequest) -> dict[str, object]:
    return await asyncio.to_thread(_run_phase5_plot_sync, request)


def _run_phase1_sync(request: Phase1RunRequest) -> dict[str, object]:
    resolved_config_path = request.config_path or os.getenv("GROBID_CONFIG_PATH")
    results = run_phase1_batch(
        input_dir=Path(request.input_dir).resolve(),
        output_dir=Path(request.output_dir).resolve(),
        config_path=resolved_config_path,
    )
    succeeded = sum(1 for result in results if result.status == "ok")
    failed = len(results) - succeeded
    return {
        "total": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "output_dir": str(Path(request.output_dir).resolve()),
    }


def _run_phase2_sync(request: Phase2RunRequest) -> dict[str, object]:
    results = run_phase2_batch(
        phase1_output_dir=Path(request.phase1_output_dir).resolve(),
        phase2_output_dir=Path(request.phase2_output_dir).resolve(),
        max_tokens=request.max_tokens,
        section_weights=request.section_weights,
        content_source=ContentSource(request.content_source),
    )
    succeeded = sum(1 for result in results if result.status == "ok")
    failed = len(results) - succeeded
    return {
        "total": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "output_dir": str(Path(request.phase2_output_dir).resolve()),
    }


def _run_phase3_sync(request: Phase3RunRequest) -> dict[str, object]:
    engines = build_engines(request.engines, default_topics=request.n_topics)
    results = run_phase3_batch(
        phase2_output_dir=Path(request.phase2_output_dir).resolve(),
        phase3_output_dir=Path(request.phase3_output_dir).resolve(),
        engines=engines,
        top_n_terms=request.top_n_terms,
        n_topics=request.n_topics,
    )
    succeeded = sum(1 for result in results if result.status == "ok")
    failed = len(results) - succeeded
    return {
        "total": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "output_dir": str(Path(request.phase3_output_dir).resolve()),
        "engines": [engine.name for engine in engines],
    }


def _run_phase4_sync(request: Phase4RunRequest) -> dict[str, object]:
    summarizer = build_summarizer(
        request.summary_backend,
        model_name=request.summary_model,
        max_sentences=request.summary_max_sentences,
    )
    embedding_backend = build_embedding_backend(
        request.embedding_backend,
        model_name=request.embedding_model,
        hashing_features=request.hashing_features,
    )
    results = run_phase4_batch(
        phase2_output_dir=Path(request.phase2_output_dir).resolve(),
        phase4_output_dir=Path(request.phase4_output_dir).resolve(),
        summarizer=summarizer,
        embedding_backend=embedding_backend,
        similarity_threshold=request.similarity_threshold,
    )
    succeeded = sum(1 for result in results if result.status == "ok")
    low_similarity = sum(1 for result in results if result.status == "low_similarity")
    failed = sum(1 for result in results if result.status == "error")
    return {
        "total": len(results),
        "succeeded": succeeded,
        "low_similarity": low_similarity,
        "failed": failed,
        "output_dir": str(Path(request.phase4_output_dir).resolve()),
    }


def _run_phase5_sync(request: Phase5RunRequest) -> dict[str, object]:
    embedding_backend = build_embedding_backend(
        request.embedding_backend,
        model_name=request.embedding_model,
        hashing_features=request.hashing_features,
    )
    reducer = build_reducer(
        request.reducer,
        n_neighbors=request.n_neighbors,
        min_dist=request.min_dist,
        random_state=request.random_state,
    )
    clusterer = build_clusterer(
        request.clusterer,
        min_cluster_size=request.min_cluster_size,
        min_samples=request.min_samples,
    )
    phase4_output_dir = (
        Path(request.phase4_output_dir).resolve() if request.phase4_output_dir is not None else None
    )
    results = run_phase5_batch(
        phase2_output_dir=Path(request.phase2_output_dir).resolve(),
        phase5_output_dir=Path(request.phase5_output_dir).resolve(),
        embedding_backend=embedding_backend,
        reducer=reducer,
        clusterer=clusterer,
        phase4_output_dir=phase4_output_dir,
    )
    succeeded = sum(1 for result in results if result.status == "ok")
    failed = len(results) - succeeded
    total_documents = sum(result.document_count or 0 for result in results if result.status == "ok")
    return {
        "total_groups": len(results),
        "total_documents": total_documents,
        "succeeded": succeeded,
        "failed": failed,
        "output_dir": str(Path(request.phase5_output_dir).resolve()),
    }


def _run_phase5_plot_sync(request: Phase5PlotRequest) -> dict[str, object]:
    output_html = Path(request.output_html).resolve()
    build_plot_html(
        phase5_output_dir=Path(request.phase5_output_dir).resolve(),
        metadata_dir=Path(request.metadata_dir).resolve(),
        output_html=output_html,
    )
    return {
        "output_html": str(output_html),
    }
