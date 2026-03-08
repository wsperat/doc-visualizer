"""Batch pipeline for Phase 2 context-strategy transformations."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from doc_visualizer.phase2.models import ContextStrategy, EmbeddingInput
from doc_visualizer.phase2.strategies import (
    build_parent_child_prepend_inputs,
    build_whole_doc_inputs,
)


@dataclass(frozen=True, slots=True)
class TransformationResult:
    """Result status for one Phase 1 document transformed into Phase 2 outputs."""

    document_id: str
    status: str
    message: str


def main() -> int:
    """Run Phase 2 transformations over all Phase 1 JSON outputs."""
    args = _build_arg_parser().parse_args()

    section_weights = _load_section_weights(args.weights_json)
    results = run_phase2_batch(
        phase1_output_dir=args.phase1_output_dir.resolve(),
        phase2_output_dir=args.phase2_output_dir.resolve(),
        max_tokens=args.max_tokens,
        section_weights=section_weights,
    )

    succeeded = sum(1 for result in results if result.status == "ok")
    failed = len(results) - succeeded
    print(f"Transformed {len(results)} documents: {succeeded} succeeded, {failed} failed.")
    return 0 if failed == 0 else 2


def run_phase2_batch(
    *,
    phase1_output_dir: Path,
    phase2_output_dir: Path,
    max_tokens: int = 512,
    section_weights: dict[str, float] | None = None,
) -> list[TransformationResult]:
    """Apply all Phase 2 transformations and persist strategy-specific outputs."""
    grobid_content_dir = phase1_output_dir / "grobid_content"
    metadata_dir = phase1_output_dir / "metadata"

    document_files = sorted(grobid_content_dir.glob("*.json"))
    if not document_files:
        raise FileNotFoundError(f"No Phase 1 grobid_content files found at {grobid_content_dir}")

    results: list[TransformationResult] = []
    for content_file in document_files:
        results.append(
            _transform_one_document(
                content_file=content_file,
                metadata_file=metadata_dir / content_file.name,
                phase2_output_dir=phase2_output_dir,
                max_tokens=max_tokens,
                section_weights=section_weights or {},
            )
        )

    _write_report(results, phase2_output_dir / "report.json")
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 2 transformation pipeline.")
    parser.add_argument(
        "--phase1-output-dir",
        type=Path,
        default=Path("data/phase1_output"),
        help="Directory containing Phase 1 outputs (grobid_content + metadata).",
    )
    parser.add_argument(
        "--phase2-output-dir",
        type=Path,
        default=Path("data/phase2_output"),
        help="Directory where Phase 2 strategy outputs will be written.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Token chunk size for parent-child-prepend strategy.",
    )
    parser.add_argument(
        "--weights-json",
        type=Path,
        default=None,
        help="Optional JSON file mapping section names to weights.",
    )
    return parser


def _load_section_weights(weights_json_path: Path | None) -> dict[str, float]:
    """Load optional section weights from a JSON mapping file."""
    if weights_json_path is None:
        return {}

    raw_payload = json.loads(weights_json_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise ValueError("weights JSON must contain an object mapping section names to weights")

    weights: dict[str, float] = {}
    for key, value in raw_payload.items():
        if not isinstance(key, str):
            raise ValueError("weights JSON keys must be strings")
        if not isinstance(value, int | float):
            raise ValueError("weights JSON values must be numeric")
        weights[key] = float(value)

    return weights


def _transform_one_document(
    *,
    content_file: Path,
    metadata_file: Path,
    phase2_output_dir: Path,
    max_tokens: int,
    section_weights: dict[str, float],
) -> TransformationResult:
    try:
        sections = _load_sections(content_file)
        title = _load_title(metadata_file)
        document_id = content_file.stem

        whole_doc_inputs = build_whole_doc_inputs(sections)
        parent_child_inputs = build_parent_child_prepend_inputs(
            title=title,
            abstract=sections.get("abstract", ""),
            sections=sections,
            max_tokens=max_tokens,
        )
        weighted_inputs = build_whole_doc_inputs(sections)
        weighted_payload = _build_weighted_payload(weighted_inputs, section_weights)

        _write_strategy_output(
            strategy=ContextStrategy.WHOLE_DOC_MEAN_POOL,
            document_id=document_id,
            output_dir=phase2_output_dir,
            payload={
                "document_id": document_id,
                "strategy": ContextStrategy.WHOLE_DOC_MEAN_POOL.value,
                "inputs": _serialize_inputs(whole_doc_inputs),
            },
        )
        _write_strategy_output(
            strategy=ContextStrategy.PARENT_CHILD_PREPEND,
            document_id=document_id,
            output_dir=phase2_output_dir,
            payload={
                "document_id": document_id,
                "strategy": ContextStrategy.PARENT_CHILD_PREPEND.value,
                "inputs": _serialize_inputs(parent_child_inputs),
                "max_tokens": max_tokens,
            },
        )
        _write_strategy_output(
            strategy=ContextStrategy.WEIGHTED_POOLING,
            document_id=document_id,
            output_dir=phase2_output_dir,
            payload={
                "document_id": document_id,
                "strategy": ContextStrategy.WEIGHTED_POOLING.value,
                "inputs": _serialize_inputs(weighted_inputs),
                "section_weights": weighted_payload,
            },
        )

        return TransformationResult(document_id=document_id, status="ok", message="transformed")
    except Exception as exc:  # pragma: no cover - integration failure path
        return TransformationResult(document_id=content_file.stem, status="error", message=str(exc))


def _load_sections(content_file: Path) -> dict[str, str]:
    payload = json.loads(content_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid content payload at {content_file}: expected object")

    sections: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError(f"Invalid section key type in {content_file}")
        if not isinstance(value, str):
            raise ValueError(f"Invalid section text type for '{key}' in {content_file}")
        sections[key] = value
    return sections


def _load_title(metadata_file: Path) -> str:
    if not metadata_file.exists():
        return ""

    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return ""

    title = payload.get("title")
    if not isinstance(title, str):
        return ""

    return title


def _build_weighted_payload(
    inputs: list[EmbeddingInput],
    provided_weights: dict[str, float],
) -> dict[str, float]:
    """Resolve section weights per current document (default = 1.0)."""
    section_names = {item.section for item in inputs}
    resolved: dict[str, float] = {}
    for section_name in section_names:
        resolved[section_name] = provided_weights.get(section_name, 1.0)
    return resolved


def _write_strategy_output(
    *,
    strategy: ContextStrategy,
    document_id: str,
    output_dir: Path,
    payload: dict[str, Any],
) -> None:
    strategy_dir = output_dir / strategy.value
    strategy_dir.mkdir(parents=True, exist_ok=True)
    output_path = strategy_dir / f"{document_id}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _serialize_inputs(inputs: list[EmbeddingInput]) -> list[dict[str, Any]]:
    return [
        {
            "section": embedding_input.section,
            "text": embedding_input.text,
            "chunk_index": embedding_input.chunk_index,
        }
        for embedding_input in inputs
    ]


def _write_report(results: list[TransformationResult], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "document_id": result.document_id,
            "status": result.status,
            "message": result.message,
        }
        for result in results
    ]
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
