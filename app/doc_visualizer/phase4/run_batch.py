"""Batch pipeline for Phase 4 summarization and semantic auditing."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from doc_visualizer.phase2.models import ContextStrategy, EmbeddingInput
from doc_visualizer.phase4.embeddings import build_embedding_backend
from doc_visualizer.phase4.models import AuditRecord, PhaseFourRunResult, StrategyDocument
from doc_visualizer.phase4.protocols import EmbeddingBackend, SummarizationBackend
from doc_visualizer.phase4.service import PhaseFourAuditService
from doc_visualizer.phase4.summarizers import build_summarizer


def main() -> int:
    """Run Phase 4 summary + semantic-audit pipeline."""
    args = _build_arg_parser().parse_args()

    summarizer = build_summarizer(
        args.summary_backend,
        model_name=args.summary_model,
        max_sentences=args.summary_max_sentences,
    )
    embedding_backend = build_embedding_backend(
        args.embedding_backend,
        model_name=args.embedding_model,
        hashing_features=args.hashing_features,
    )

    results = run_phase4_batch(
        phase2_output_dir=args.phase2_output_dir.resolve(),
        phase4_output_dir=args.phase4_output_dir.resolve(),
        summarizer=summarizer,
        embedding_backend=embedding_backend,
        similarity_threshold=args.similarity_threshold,
    )

    succeeded = sum(1 for result in results if result.status == "ok")
    below_threshold = sum(1 for result in results if result.status == "low_similarity")
    failed = sum(1 for result in results if result.status == "error")
    print(
        "Audited "
        f"{len(results)} documents: {succeeded} ok, "
        f"{below_threshold} below threshold, {failed} failed."
    )
    return 0 if failed == 0 else 2


def run_phase4_batch(
    *,
    phase2_output_dir: Path,
    phase4_output_dir: Path,
    summarizer: SummarizationBackend,
    embedding_backend: EmbeddingBackend,
    similarity_threshold: float = 0.8,
) -> list[PhaseFourRunResult]:
    """Execute summary + semantic-audit pass for all Phase 2 documents."""
    documents = load_phase2_documents(phase2_output_dir)
    service = PhaseFourAuditService(
        summarizer=summarizer,
        embedding_backend=embedding_backend,
        similarity_threshold=similarity_threshold,
    )

    results: list[PhaseFourRunResult] = []
    for document in documents:
        results.append(
            _run_one_document(
                document=document,
                service=service,
                phase4_output_dir=phase4_output_dir,
            )
        )

    _write_report(results, phase4_output_dir / "report.json")
    return results


def load_phase2_documents(phase2_output_dir: Path) -> list[StrategyDocument]:
    """Load all strategy payloads from Phase 2 output structure."""
    source_dirs = _discover_source_dirs(phase2_output_dir)
    strategy_names = {strategy.value for strategy in ContextStrategy}

    documents: list[StrategyDocument] = []
    for source_name, source_dir in source_dirs.items():
        for strategy_name in sorted(strategy_names):
            strategy_dir = source_dir / strategy_name
            if not strategy_dir.is_dir():
                continue

            for payload_file in sorted(strategy_dir.glob("*.json")):
                payload = _read_json_object(payload_file)
                inputs = _parse_inputs(payload.get("inputs"))
                if not inputs:
                    continue

                section_weights = _parse_section_weights(payload.get("section_weights"))
                documents.append(
                    StrategyDocument(
                        source=source_name,
                        strategy=strategy_name,
                        document_id=payload_file.stem,
                        inputs=inputs,
                        section_weights=section_weights,
                    )
                )

    if not documents:
        raise FileNotFoundError(f"No Phase 2 strategy documents found under: {phase2_output_dir}")

    return documents


def _run_one_document(
    *,
    document: StrategyDocument,
    service: PhaseFourAuditService,
    phase4_output_dir: Path,
) -> PhaseFourRunResult:
    try:
        audit_record = service.audit_document(document)
        output_path = _write_audit_output(audit_record, phase4_output_dir)
        status = "low_similarity" if audit_record.is_below_threshold else "ok"
        message = (
            "summary drift risk"
            if audit_record.is_below_threshold
            else "summary passed similarity threshold"
        )
        return PhaseFourRunResult(
            source=document.source,
            strategy=document.strategy,
            document_id=document.document_id,
            status=status,
            message=message,
            similarity_score=audit_record.similarity_score,
            output_path=str(output_path),
        )
    except Exception as exc:  # pragma: no cover - integration failure path
        return PhaseFourRunResult(
            source=document.source,
            strategy=document.strategy,
            document_id=document.document_id,
            status="error",
            message=str(exc),
        )


def _write_audit_output(audit_record: AuditRecord, phase4_output_dir: Path) -> Path:
    target_dir = phase4_output_dir / audit_record.source / audit_record.strategy
    target_dir.mkdir(parents=True, exist_ok=True)

    output_path = target_dir / f"{audit_record.document_id}.json"
    output_path.write_text(
        json.dumps(audit_record.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def _write_report(results: Sequence[PhaseFourRunResult], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.to_dict() for result in results]
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _discover_source_dirs(phase2_output_dir: Path) -> dict[str, Path]:
    """Support both direct and namespaced Phase 2 output layouts."""
    strategy_names = {strategy.value for strategy in ContextStrategy}

    source_dirs: dict[str, Path] = {}
    for child in sorted(phase2_output_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name in strategy_names:
            continue
        if any((child / strategy_name).is_dir() for strategy_name in strategy_names):
            source_dirs[child.name] = child

    if source_dirs:
        return source_dirs

    root_has_strategies = any(
        (phase2_output_dir / strategy_name).is_dir() for strategy_name in strategy_names
    )
    if root_has_strategies:
        return {"default": phase2_output_dir}

    raise FileNotFoundError(
        f"No strategy directories found under phase2 output directory: {phase2_output_dir}"
    )


def _parse_inputs(raw_inputs: object) -> tuple[EmbeddingInput, ...]:
    if not isinstance(raw_inputs, list):
        return ()

    parsed_inputs: list[EmbeddingInput] = []
    for item in raw_inputs:
        if not isinstance(item, dict):
            continue

        section_value = item.get("section")
        text_value = item.get("text")
        chunk_index_value = item.get("chunk_index")

        if not isinstance(section_value, str):
            continue
        if not isinstance(text_value, str):
            continue

        if isinstance(chunk_index_value, int):
            chunk_index: int | None = chunk_index_value
        else:
            chunk_index = None

        parsed_inputs.append(
            EmbeddingInput(
                section=section_value,
                text=text_value,
                chunk_index=chunk_index,
            )
        )

    return tuple(parsed_inputs)


def _parse_section_weights(raw_weights: object) -> dict[str, float]:
    if not isinstance(raw_weights, dict):
        return {}

    section_weights: dict[str, float] = {}
    for key, value in raw_weights.items():
        if not isinstance(key, str):
            continue
        if not isinstance(value, int | float):
            continue
        section_weights[key] = float(value)
    return section_weights


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 4 summarization and semantic-audit pipeline."
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
        help="Directory where Phase 4 outputs will be written.",
    )
    parser.add_argument(
        "--summary-backend",
        type=str,
        default="extractive",
        help="Summary backend: extractive, led, longt5.",
    )
    parser.add_argument(
        "--summary-model",
        type=str,
        default=None,
        help="Optional model id for led/longt5 backends.",
    )
    parser.add_argument(
        "--summary-max-sentences",
        type=int,
        default=5,
        help="Maximum sentence count for extractive summarizer.",
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
        help="Feature dimensionality used by hashing embedding backend.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.8,
        help="Cosine similarity threshold for summary drift alert.",
    )
    return parser


__all__ = [
    "load_phase2_documents",
    "run_phase4_batch",
]


if __name__ == "__main__":
    raise SystemExit(main())
