from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from doc_visualizer.phase2.models import Vector
from doc_visualizer.phase5.run_batch import run_phase5_batch


class MappingEmbeddingBackend:
    def __init__(self, vectors: dict[str, Vector]) -> None:
        self._vectors = dict(vectors)

    @property
    def name(self) -> str:
        return "mapping_embed"

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._vectors[text] for text in texts]


class IndexReducer:
    @property
    def name(self) -> str:
        return "index_reducer"

    def reduce(self, vectors: Sequence[Vector]) -> list[tuple[float, float]]:
        return [(float(index), float(index + 1)) for index, _ in enumerate(vectors)]


class SingleClusterer:
    @property
    def name(self) -> str:
        return "single_clusterer"

    def cluster(self, points: Sequence[tuple[float, float]]) -> list[int]:
        if len(points) == 1:
            return [-1]
        return [0] * len(points)


def test_run_phase5_batch_writes_map_outputs_and_report(tmp_path: Path) -> None:
    phase2_output = tmp_path / "phase2_output"
    (phase2_output / "grobid" / "whole_doc_mean_pool").mkdir(parents=True)
    (phase2_output / "raw" / "weighted_pooling").mkdir(parents=True)

    (phase2_output / "grobid" / "whole_doc_mean_pool" / "paper_a.json").write_text(
        json.dumps(
            {
                "inputs": [{"section": "abstract", "text": "alpha text", "chunk_index": None}],
            }
        ),
        encoding="utf-8",
    )
    (phase2_output / "grobid" / "whole_doc_mean_pool" / "paper_b.json").write_text(
        json.dumps(
            {
                "inputs": [{"section": "abstract", "text": "beta text", "chunk_index": None}],
            }
        ),
        encoding="utf-8",
    )
    (phase2_output / "raw" / "weighted_pooling" / "paper_c.json").write_text(
        json.dumps(
            {
                "inputs": [{"section": "chapter_1", "text": "gamma text", "chunk_index": None}],
                "section_weights": {"chapter_1": 2.0},
            }
        ),
        encoding="utf-8",
    )

    phase4_output = tmp_path / "phase4_output"
    (phase4_output / "grobid" / "whole_doc_mean_pool").mkdir(parents=True)
    (phase4_output / "grobid" / "whole_doc_mean_pool" / "paper_a.json").write_text(
        json.dumps(
            {
                "similarity_score": 0.92,
                "is_below_threshold": False,
            }
        ),
        encoding="utf-8",
    )

    phase5_output = tmp_path / "phase5_output"
    results = run_phase5_batch(
        phase2_output_dir=phase2_output,
        phase5_output_dir=phase5_output,
        phase4_output_dir=phase4_output,
        embedding_backend=MappingEmbeddingBackend(
            {
                "alpha text": (1.0, 0.0),
                "beta text": (0.0, 1.0),
                "gamma text": (2.0, 2.0),
            }
        ),
        reducer=IndexReducer(),
        clusterer=SingleClusterer(),
    )

    assert len(results) == 2
    assert all(result.status == "ok" for result in results)

    grobid_map = phase5_output / "grobid" / "whole_doc_mean_pool" / "map.json"
    raw_map = phase5_output / "raw" / "weighted_pooling" / "map.json"
    report = phase5_output / "report.json"

    assert grobid_map.exists()
    assert raw_map.exists()
    assert report.exists()

    grobid_payload = json.loads(grobid_map.read_text(encoding="utf-8"))
    raw_payload = json.loads(raw_map.read_text(encoding="utf-8"))
    report_payload = json.loads(report.read_text(encoding="utf-8"))

    assert grobid_payload["document_count"] == 2
    assert raw_payload["document_count"] == 1
    assert len(report_payload) == 2

    point_a = next(point for point in grobid_payload["points"] if point["document_id"] == "paper_a")
    point_b = next(point for point in grobid_payload["points"] if point["document_id"] == "paper_b")
    assert point_a["similarity_score"] == 0.92
    assert point_a["is_below_threshold"] is False
    assert "similarity_score" not in point_b
