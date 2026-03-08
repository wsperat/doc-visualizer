"""Batch pipeline for Phase 5 dimensionality reduction and mapping."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from doc_visualizer.phase4.embeddings import build_embedding_backend
from doc_visualizer.phase4.models import StrategyDocument
from doc_visualizer.phase4.protocols import EmbeddingBackend
from doc_visualizer.phase4.run_batch import load_phase2_documents
from doc_visualizer.phase5.clusterers import build_clusterer
from doc_visualizer.phase5.models import PhaseFiveRunResult, StrategyMap
from doc_visualizer.phase5.protocols import Clusterer, DimensionalityReducer
from doc_visualizer.phase5.reducers import build_reducer
from doc_visualizer.phase5.service import build_strategy_map


def main() -> int:
    """Run Phase 5 UMAP+HDBSCAN map generation pipeline."""
    args = _build_arg_parser().parse_args()

    embedding_backend = build_embedding_backend(
        args.embedding_backend,
        model_name=args.embedding_model,
        hashing_features=args.hashing_features,
    )
    reducer = build_reducer(
        args.reducer,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
    )
    clusterer = build_clusterer(
        args.clusterer,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )

    phase4_output_dir = (
        args.phase4_output_dir.resolve() if args.phase4_output_dir is not None else None
    )
    results = run_phase5_batch(
        phase2_output_dir=args.phase2_output_dir.resolve(),
        phase5_output_dir=args.phase5_output_dir.resolve(),
        embedding_backend=embedding_backend,
        reducer=reducer,
        clusterer=clusterer,
        phase4_output_dir=phase4_output_dir,
    )

    succeeded = sum(1 for result in results if result.status == "ok")
    failed = len(results) - succeeded
    print(f"Mapped {len(results)} source+strategy corpora: {succeeded} succeeded, {failed} failed.")
    return 0 if failed == 0 else 2


def run_phase5_batch(
    *,
    phase2_output_dir: Path,
    phase5_output_dir: Path,
    embedding_backend: EmbeddingBackend,
    reducer: DimensionalityReducer,
    clusterer: Clusterer,
    phase4_output_dir: Path | None = None,
) -> list[PhaseFiveRunResult]:
    """Build 2D maps from Phase 2 vectors and optional Phase 4 QC metadata."""
    documents = load_phase2_documents(phase2_output_dir)
    grouped = _group_documents_by_source_strategy(documents)

    parameter_payload: dict[str, object] = {
        "embedding_backend": embedding_backend.name,
        "reducer": reducer.name,
        "clusterer": clusterer.name,
    }

    results: list[PhaseFiveRunResult] = []
    for (source, strategy), strategy_documents in sorted(grouped.items()):
        quality_control = (
            _load_quality_control_for_group(
                phase4_output_dir=phase4_output_dir,
                source=source,
                strategy=strategy,
            )
            if phase4_output_dir is not None
            else {}
        )

        results.append(
            _run_one_group(
                source=source,
                strategy=strategy,
                documents=strategy_documents,
                embedding_backend=embedding_backend,
                reducer=reducer,
                clusterer=clusterer,
                phase5_output_dir=phase5_output_dir,
                parameter_payload=parameter_payload,
                quality_control=quality_control,
            )
        )

    _write_report(results, phase5_output_dir / "report.json")
    return results


def _run_one_group(
    *,
    source: str,
    strategy: str,
    documents: Sequence[StrategyDocument],
    embedding_backend: EmbeddingBackend,
    reducer: DimensionalityReducer,
    clusterer: Clusterer,
    phase5_output_dir: Path,
    parameter_payload: dict[str, object],
    quality_control: dict[str, tuple[float, bool]],
) -> PhaseFiveRunResult:
    try:
        map_output = build_strategy_map(
            documents=documents,
            embedding_backend=embedding_backend,
            reducer=reducer,
            clusterer=clusterer,
            parameters=parameter_payload,
            quality_control=quality_control,
        )
        output_path = _write_map_output(map_output, phase5_output_dir)
        return PhaseFiveRunResult(
            source=source,
            strategy=strategy,
            status="ok",
            message=f"mapped {len(documents)} documents",
            output_path=str(output_path),
        )
    except Exception as exc:  # pragma: no cover - integration failure path
        return PhaseFiveRunResult(
            source=source,
            strategy=strategy,
            status="error",
            message=str(exc),
        )


def _group_documents_by_source_strategy(
    documents: Sequence[StrategyDocument],
) -> dict[tuple[str, str], list[StrategyDocument]]:
    grouped: dict[tuple[str, str], list[StrategyDocument]] = defaultdict(list)
    for document in documents:
        grouped[(document.source, document.strategy)].append(document)

    for key in grouped:
        grouped[key].sort(key=lambda item: item.document_id)

    return dict(grouped)


def _load_quality_control_for_group(
    *,
    phase4_output_dir: Path,
    source: str,
    strategy: str,
) -> dict[str, tuple[float, bool]]:
    strategy_dir = phase4_output_dir / source / strategy
    if not strategy_dir.is_dir():
        return {}

    records: dict[str, tuple[float, bool]] = {}
    for payload_file in sorted(strategy_dir.glob("*.json")):
        payload = _read_json_object(payload_file)
        similarity_score = payload.get("similarity_score")
        is_below_threshold = payload.get("is_below_threshold")

        if not isinstance(similarity_score, int | float):
            continue
        if not isinstance(is_below_threshold, bool):
            continue
        records[payload_file.stem] = (float(similarity_score), is_below_threshold)

    return records


def _write_map_output(map_output: StrategyMap, phase5_output_dir: Path) -> Path:
    target_dir = phase5_output_dir / map_output.source / map_output.strategy
    target_dir.mkdir(parents=True, exist_ok=True)

    output_path = target_dir / "map.json"
    output_path.write_text(
        json.dumps(map_output.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def _write_report(results: Sequence[PhaseFiveRunResult], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.to_dict() for result in results]
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 5 dimensionality reduction and mapping pipeline."
    )
    parser.add_argument(
        "--phase2-output-dir",
        type=Path,
        default=Path("data/phase2_output"),
        help="Directory containing Phase 2 transformed outputs.",
    )
    parser.add_argument(
        "--phase4-output-dir",
        type=Path,
        default=Path("data/phase4_output"),
        help="Optional Phase 4 output directory for QC overlay data.",
    )
    parser.add_argument(
        "--phase5-output-dir",
        type=Path,
        default=Path("data/phase5_output"),
        help="Directory where Phase 5 map outputs will be written.",
    )
    parser.add_argument(
        "--embedding-backend",
        type=str,
        default="hashing",
        help="Embedding backend: hashing, sentence_transformers.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Optional model id for sentence_transformers backend.",
    )
    parser.add_argument(
        "--hashing-features",
        type=int,
        default=512,
        help="Feature dimensionality for hashing embedding backend.",
    )
    parser.add_argument(
        "--reducer",
        type=str,
        default="umap",
        help="Reducer engine (currently: umap).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors hyperparameter.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist hyperparameter.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reducer reproducibility.",
    )
    parser.add_argument(
        "--clusterer",
        type=str,
        default="hdbscan",
        help="Clusterer engine (currently: hdbscan).",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="HDBSCAN minimum cluster size.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="Optional HDBSCAN min_samples.",
    )
    return parser


__all__ = [
    "run_phase5_batch",
]


if __name__ == "__main__":
    raise SystemExit(main())
