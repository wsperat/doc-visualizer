"""Utilities for loading Phase 5 map records enriched with document titles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PlotRecord:
    """One plot point enriched with display metadata."""

    view: str
    source: str
    strategy: str
    document_id: str
    title: str
    x: float
    y: float
    cluster_id: int
    similarity_score: float | None
    is_below_threshold: bool | None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "view": self.view,
            "source": self.source,
            "strategy": self.strategy,
            "document_id": self.document_id,
            "title": self.title,
            "x": self.x,
            "y": self.y,
            "cluster_id": self.cluster_id,
        }
        if self.similarity_score is not None:
            payload["similarity_score"] = self.similarity_score
        if self.is_below_threshold is not None:
            payload["is_below_threshold"] = self.is_below_threshold
        return payload


def load_plot_records(
    *,
    phase5_output_dir: Path,
    metadata_dir: Path,
) -> list[PlotRecord]:
    """Load all map points from Phase 5 outputs enriched with metadata titles."""
    title_by_document_id = _load_titles(metadata_dir)

    records: list[PlotRecord] = []
    for map_file in sorted(phase5_output_dir.glob("*/*/map.json")):
        source = map_file.parent.parent.name
        strategy = map_file.parent.name
        view = f"{source} / {strategy}"

        payload = json.loads(map_file.read_text(encoding="utf-8"))
        points = payload.get("points")
        if not isinstance(points, list):
            continue

        for point in points:
            if not isinstance(point, dict):
                continue
            document_id = point.get("document_id")
            x_value = point.get("x")
            y_value = point.get("y")
            cluster_id = point.get("cluster_id")
            if not isinstance(document_id, str):
                continue
            if not isinstance(x_value, int | float):
                continue
            if not isinstance(y_value, int | float):
                continue
            if not isinstance(cluster_id, int):
                continue

            similarity_raw = point.get("similarity_score")
            is_below_raw = point.get("is_below_threshold")
            similarity_score = (
                float(similarity_raw) if isinstance(similarity_raw, int | float) else None
            )
            is_below_threshold = is_below_raw if isinstance(is_below_raw, bool) else None
            title = title_by_document_id.get(document_id, document_id)

            records.append(
                PlotRecord(
                    view=view,
                    source=source,
                    strategy=strategy,
                    document_id=document_id,
                    title=title,
                    x=float(x_value),
                    y=float(y_value),
                    cluster_id=cluster_id,
                    similarity_score=similarity_score,
                    is_below_threshold=is_below_threshold,
                )
            )

    return records


def _load_titles(metadata_dir: Path) -> dict[str, str]:
    titles: dict[str, str] = {}
    for metadata_file in metadata_dir.glob("*.json"):
        payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        title = payload.get("title")
        if isinstance(title, str) and title.strip():
            titles[metadata_file.stem] = title.strip()
    return titles
