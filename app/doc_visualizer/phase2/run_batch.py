"""Batch pipeline for Phase 2 context-strategy transformations."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from doc_visualizer.phase2.models import ContextStrategy, EmbeddingInput
from doc_visualizer.phase2.strategies import (
    build_parent_child_prepend_inputs,
    build_whole_doc_inputs,
)

_SECTION_KEY_RE = re.compile(r"[^a-z0-9]+")


class ContentSource(StrEnum):
    """Available content sources for Phase 2 transformations."""

    GROBID = "grobid"
    RAW = "raw"
    HYBRID = "hybrid"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class TransformationResult:
    """Result status for one Phase 1 document transformed into Phase 2 outputs."""

    document_id: str
    status: str
    message: str
    content_source: str = ""


@dataclass(frozen=True, slots=True)
class RawSectionRecord:
    """Typed representation of one raw section node."""

    title: str
    text: str
    level: int
    position: int
    breadcrumb: str


def main() -> int:
    """Run Phase 2 transformations over all Phase 1 JSON outputs."""
    args = _build_arg_parser().parse_args()

    section_weights = _load_section_weights(args.weights_json)
    results = run_phase2_batch(
        phase1_output_dir=args.phase1_output_dir.resolve(),
        phase2_output_dir=args.phase2_output_dir.resolve(),
        max_tokens=args.max_tokens,
        section_weights=section_weights,
        content_source=ContentSource(args.content_source),
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
    content_source: ContentSource = ContentSource.GROBID,
) -> list[TransformationResult]:
    """Apply all Phase 2 transformations and persist strategy-specific outputs."""
    if content_source is ContentSource.BOTH:
        grobid_results = _run_single_source_batch(
            phase1_output_dir=phase1_output_dir,
            phase2_output_dir=phase2_output_dir / ContentSource.GROBID.value,
            max_tokens=max_tokens,
            section_weights=section_weights,
            content_source=ContentSource.GROBID,
        )
        raw_results = _run_single_source_batch(
            phase1_output_dir=phase1_output_dir,
            phase2_output_dir=phase2_output_dir / ContentSource.RAW.value,
            max_tokens=max_tokens,
            section_weights=section_weights,
            content_source=ContentSource.RAW,
        )
        results = [*grobid_results, *raw_results]
        _write_report(results, phase2_output_dir / "report.json")
        return results

    return _run_single_source_batch(
        phase1_output_dir=phase1_output_dir,
        phase2_output_dir=phase2_output_dir,
        max_tokens=max_tokens,
        section_weights=section_weights,
        content_source=content_source,
    )


def _run_single_source_batch(
    *,
    phase1_output_dir: Path,
    phase2_output_dir: Path,
    max_tokens: int,
    section_weights: dict[str, float] | None,
    content_source: ContentSource,
) -> list[TransformationResult]:
    """Run transformation pipeline for one content source."""
    grobid_content_dir = phase1_output_dir / "grobid_content"
    raw_content_dir = phase1_output_dir / "raw_content"
    metadata_dir = phase1_output_dir / "metadata"

    document_ids = _discover_document_ids(
        grobid_content_dir=grobid_content_dir,
        raw_content_dir=raw_content_dir,
        content_source=content_source,
    )

    results: list[TransformationResult] = []
    for document_id in document_ids:
        results.append(
            _transform_one_document(
                document_id=document_id,
                grobid_content_file=grobid_content_dir / f"{document_id}.json",
                raw_content_file=raw_content_dir / f"{document_id}.json",
                metadata_file=metadata_dir / f"{document_id}.json",
                phase2_output_dir=phase2_output_dir,
                max_tokens=max_tokens,
                section_weights=section_weights or {},
                content_source=content_source,
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
        help="Directory containing Phase 1 outputs (grobid_content/raw_content/metadata).",
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
    parser.add_argument(
        "--content-source",
        type=str,
        choices=[source.value for source in ContentSource],
        default=ContentSource.GROBID.value,
        help="Choose content source: grobid, raw, hybrid, or both (grobid+raw).",
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


def _discover_document_ids(
    *,
    grobid_content_dir: Path,
    raw_content_dir: Path,
    content_source: ContentSource,
) -> list[str]:
    """Resolve document ids based on selected content source."""
    grobid_ids = {path.stem for path in grobid_content_dir.glob("*.json")}
    raw_ids = {path.stem for path in raw_content_dir.glob("*.json")}

    if content_source is ContentSource.GROBID:
        if not grobid_ids:
            raise FileNotFoundError(
                f"No Phase 1 grobid_content files found at {grobid_content_dir}"
            )
        return sorted(grobid_ids)

    if content_source is ContentSource.RAW:
        if not raw_ids:
            raise FileNotFoundError(f"No Phase 1 raw_content files found at {raw_content_dir}")
        return sorted(raw_ids)

    hybrid_ids = grobid_ids | raw_ids
    if not hybrid_ids:
        raise FileNotFoundError(
            "No Phase 1 content files found in either grobid_content or raw_content"
        )
    return sorted(hybrid_ids)


def _transform_one_document(
    *,
    document_id: str,
    grobid_content_file: Path,
    raw_content_file: Path,
    metadata_file: Path,
    phase2_output_dir: Path,
    max_tokens: int,
    section_weights: dict[str, float],
    content_source: ContentSource,
) -> TransformationResult:
    try:
        title = _load_title(metadata_file)
        grobid_sections = (
            _load_sections(grobid_content_file) if grobid_content_file.exists() else {}
        )
        raw_sections = _load_raw_sections(raw_content_file) if raw_content_file.exists() else []

        sections, hierarchy_payload, raw_levels = _select_sections(
            grobid_sections=grobid_sections,
            raw_sections=raw_sections,
            content_source=content_source,
        )

        if not sections:
            raise ValueError("No section content available for selected content source")

        whole_doc_inputs = build_whole_doc_inputs(sections)
        parent_child_inputs = build_parent_child_prepend_inputs(
            title=title,
            abstract=sections.get("abstract", ""),
            sections=sections,
            max_tokens=max_tokens,
        )
        weighted_inputs = build_whole_doc_inputs(sections)
        weighted_payload = _build_weighted_payload(
            inputs=weighted_inputs,
            provided_weights=section_weights,
            section_levels=raw_levels,
        )

        _write_strategy_output(
            strategy=ContextStrategy.WHOLE_DOC_MEAN_POOL,
            document_id=document_id,
            output_dir=phase2_output_dir,
            payload={
                "document_id": document_id,
                "strategy": ContextStrategy.WHOLE_DOC_MEAN_POOL.value,
                "content_source": content_source.value,
                "inputs": _serialize_inputs(whole_doc_inputs),
                "raw_section_hierarchy": hierarchy_payload,
            },
        )
        _write_strategy_output(
            strategy=ContextStrategy.PARENT_CHILD_PREPEND,
            document_id=document_id,
            output_dir=phase2_output_dir,
            payload={
                "document_id": document_id,
                "strategy": ContextStrategy.PARENT_CHILD_PREPEND.value,
                "content_source": content_source.value,
                "inputs": _serialize_inputs(parent_child_inputs),
                "max_tokens": max_tokens,
                "raw_section_hierarchy": hierarchy_payload,
            },
        )
        _write_strategy_output(
            strategy=ContextStrategy.WEIGHTED_POOLING,
            document_id=document_id,
            output_dir=phase2_output_dir,
            payload={
                "document_id": document_id,
                "strategy": ContextStrategy.WEIGHTED_POOLING.value,
                "content_source": content_source.value,
                "inputs": _serialize_inputs(weighted_inputs),
                "section_weights": weighted_payload,
                "raw_section_hierarchy": hierarchy_payload,
            },
        )

        return TransformationResult(
            document_id=document_id,
            status="ok",
            message="transformed",
            content_source=content_source.value,
        )
    except Exception as exc:  # pragma: no cover - integration failure path
        return TransformationResult(
            document_id=document_id,
            status="error",
            message=str(exc),
            content_source=content_source.value,
        )


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


def _load_raw_sections(raw_content_file: Path) -> list[RawSectionRecord]:
    payload = json.loads(raw_content_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Invalid raw content payload at {raw_content_file}: expected list")

    parsed: list[tuple[int, int, str, str]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid raw section entry at index {index} in {raw_content_file}")

        title = item.get("title", "")
        text = item.get("text", "")
        level = item.get("level", 1)
        position = item.get("position", index + 1)

        if not isinstance(title, str):
            title = ""
        if not isinstance(text, str):
            text = ""
        if not isinstance(level, int):
            raise ValueError(f"Invalid raw section level at index {index} in {raw_content_file}")
        if not isinstance(position, int):
            raise ValueError(f"Invalid raw section position at index {index} in {raw_content_file}")

        parsed.append((position, max(level, 1), title.strip(), text.strip()))

    parsed.sort(key=lambda item: item[0])

    stack: list[str] = []
    sections: list[RawSectionRecord] = []
    for position, level, title, text in parsed:
        effective_title = title if title else f"untitled_section_{position}"

        while len(stack) >= level:
            stack.pop()
        stack.append(effective_title)

        breadcrumb = " > ".join(stack)
        sections.append(
            RawSectionRecord(
                title=effective_title,
                text=text,
                level=level,
                position=position,
                breadcrumb=breadcrumb,
            )
        )

    return sections


def _select_sections(
    *,
    grobid_sections: dict[str, str],
    raw_sections: list[RawSectionRecord],
    content_source: ContentSource,
) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, int]]:
    """Build transformed section map based on selected content source."""
    if content_source is ContentSource.GROBID:
        return dict(grobid_sections), {}, {}

    raw_map, hierarchy_payload, raw_levels = _build_raw_section_map(raw_sections)

    if content_source is ContentSource.RAW:
        return raw_map, hierarchy_payload, raw_levels

    merged = dict(grobid_sections)
    merged_hierarchy: dict[str, dict[str, Any]] = {}
    merged_levels: dict[str, int] = {}

    for key, value in raw_map.items():
        target_key = key if key not in merged else f"raw__{key}"
        merged[target_key] = value

        if key in hierarchy_payload:
            metadata = dict(hierarchy_payload[key])
            metadata["source_key"] = key
            merged_hierarchy[target_key] = metadata

        if key in raw_levels:
            merged_levels[target_key] = raw_levels[key]

    return merged, merged_hierarchy, merged_levels


def _build_raw_section_map(
    raw_sections: list[RawSectionRecord],
) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, int]]:
    """Convert raw section list into enriched section mapping with hierarchy metadata."""
    section_map: dict[str, str] = {}
    hierarchy_payload: dict[str, dict[str, Any]] = {}
    section_levels: dict[str, int] = {}

    for section in raw_sections:
        if not section.text:
            continue

        base_key = _normalize_section_key(section.breadcrumb)
        section_key = _make_unique_key(base_key, section_map)

        section_map[section_key] = _join_non_empty(
            (
                f"Heading: {section.title}",
                f"Hierarchy: {section.breadcrumb}",
                section.text,
            )
        )
        hierarchy_payload[section_key] = {
            "title": section.title,
            "breadcrumb": section.breadcrumb,
            "level": section.level,
            "position": section.position,
        }
        section_levels[section_key] = section.level

    return section_map, hierarchy_payload, section_levels


def _normalize_section_key(title: str) -> str:
    normalized = _SECTION_KEY_RE.sub("_", title.lower()).strip("_")
    return normalized or "untitled"


def _make_unique_key(base_key: str, existing: dict[str, str]) -> str:
    if base_key not in existing:
        return base_key

    suffix = 2
    while f"{base_key}_{suffix}" in existing:
        suffix += 1
    return f"{base_key}_{suffix}"


def _join_non_empty(parts: tuple[str, ...]) -> str:
    normalized = [part.strip() for part in parts if part.strip()]
    return "\n\n".join(normalized)


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
    *,
    inputs: list[EmbeddingInput],
    provided_weights: dict[str, float],
    section_levels: dict[str, int],
) -> dict[str, float]:
    """Resolve section weights per current document.

    Defaults:
    - explicit user weight if provided
    - otherwise depth-aware default for raw hierarchy sections
    - otherwise 1.0
    """
    section_names = {item.section for item in inputs}
    resolved: dict[str, float] = {}
    for section_name in section_names:
        if section_name in provided_weights:
            resolved[section_name] = provided_weights[section_name]
            continue

        if section_name in section_levels:
            resolved[section_name] = _default_weight_for_level(section_levels[section_name])
            continue

        resolved[section_name] = 1.0

    return resolved


def _default_weight_for_level(level: int) -> float:
    if level <= 1:
        return 1.0
    if level == 2:
        return 0.85
    return 0.7


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
            "content_source": result.content_source,
        }
        for result in results
    ]
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
