"""Pydantic models for async pipeline API."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class JobState(StrEnum):
    """Execution state for submitted jobs."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResponse(BaseModel):
    """API payload for one job record."""

    job_id: str
    phase: str
    status: JobState
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, object] | None = None
    error: str | None = None


class Phase1RunRequest(BaseModel):
    """Request payload for Phase 1."""

    input_dir: str = "data"
    output_dir: str = "data/phase1_output"
    config_path: str | None = None


class Phase2RunRequest(BaseModel):
    """Request payload for Phase 2."""

    phase1_output_dir: str = "data/phase1_output"
    phase2_output_dir: str = "data/phase2_output"
    max_tokens: int = 512
    section_weights: dict[str, float] = Field(default_factory=dict)
    content_source: str = "grobid"


class Phase3RunRequest(BaseModel):
    """Request payload for Phase 3."""

    phase2_output_dir: str = "data/phase2_output"
    phase3_output_dir: str = "data/phase3_output"
    engines: list[str] = Field(default_factory=lambda: ["lda"])
    top_n_terms: int = 10
    n_topics: int = 8


class Phase4RunRequest(BaseModel):
    """Request payload for Phase 4."""

    phase2_output_dir: str = "data/phase2_output"
    phase4_output_dir: str = "data/phase4_output"
    summary_backend: str = "extractive"
    summary_model: str | None = None
    summary_max_sentences: int = 5
    embedding_backend: str = "hashing"
    embedding_model: str | None = None
    hashing_features: int = 512
    similarity_threshold: float = 0.8


class Phase5RunRequest(BaseModel):
    """Request payload for Phase 5."""

    phase2_output_dir: str = "data/phase2_output"
    phase4_output_dir: str | None = "data/phase4_output"
    phase5_output_dir: str = "data/phase5_output"
    embedding_backend: str = "hashing"
    embedding_model: str | None = None
    hashing_features: int = 512
    reducer: str = "umap"
    n_neighbors: int = 15
    min_dist: float = 0.1
    random_state: int = 42
    clusterer: str = "hdbscan"
    min_cluster_size: int = 5
    min_samples: int | None = None


class Phase5PlotRequest(BaseModel):
    """Request payload for Phase 5 HTML plot generation."""

    phase5_output_dir: str = "data/phase5_output"
    metadata_dir: str = "data/phase1_output/metadata"
    output_html: str = "data/phase5_output/document_map.html"


class UploadResponse(BaseModel):
    """Response payload for file uploads."""

    saved_files: list[str]
    target_dir: str
