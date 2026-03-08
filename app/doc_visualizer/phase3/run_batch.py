"""Batch pipeline for Phase 3 multi-track topic modeling."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from doc_visualizer.phase2.models import ContextStrategy
from doc_visualizer.phase3.engines import build_engines
from doc_visualizer.phase3.models import (
    EngineOutput,
    PhaseThreeRunResult,
    StrategyCorpus,
    TopicDocument,
)
from doc_visualizer.phase3.protocols import TopicModelEngine


def main() -> int:
    """Run Phase 3 topic-model tracks over Phase 2 transformed corpora."""
    args = _build_arg_parser().parse_args()

    engine_names = [part.strip() for part in args.engines.split(",")]
    engines = build_engines(engine_names, default_topics=args.n_topics)

    results = run_phase3_batch(
        phase2_output_dir=args.phase2_output_dir.resolve(),
        phase3_output_dir=args.phase3_output_dir.resolve(),
        engines=engines,
        top_n_terms=args.top_n_terms,
        n_topics=args.n_topics,
    )

    succeeded = sum(1 for result in results if result.status == "ok")
    failed = len(results) - succeeded
    print(f"Ran {len(results)} engine tracks: {succeeded} succeeded, {failed} failed.")
    return 0 if failed == 0 else 2


def run_phase3_batch(
    *,
    phase2_output_dir: Path,
    phase3_output_dir: Path,
    engines: Sequence[TopicModelEngine],
    top_n_terms: int,
    n_topics: int,
) -> list[PhaseThreeRunResult]:
    """Execute configured topic engines for each source+strategy corpus."""
    corpora = load_phase2_corpora(phase2_output_dir)

    results: list[PhaseThreeRunResult] = []
    for corpus in corpora:
        for engine in engines:
            results.append(
                _run_one_track(
                    corpus=corpus,
                    engine=engine,
                    phase3_output_dir=phase3_output_dir,
                    top_n_terms=top_n_terms,
                    n_topics=n_topics,
                )
            )

    _write_report(results, phase3_output_dir / "report.json")
    return results


def load_phase2_corpora(phase2_output_dir: Path) -> list[StrategyCorpus]:
    """Load source+strategy corpora from Phase 2 output folder structure."""
    source_dirs = _discover_source_dirs(phase2_output_dir)
    strategy_names = {strategy.value for strategy in ContextStrategy}

    corpora: list[StrategyCorpus] = []
    for source_name, source_dir in source_dirs.items():
        for strategy_name in sorted(strategy_names):
            strategy_dir = source_dir / strategy_name
            if not strategy_dir.is_dir():
                continue

            documents: list[TopicDocument] = []
            for payload_file in sorted(strategy_dir.glob("*.json")):
                payload = _read_json_object(payload_file)
                document_text = _strategy_payload_to_text(
                    strategy_name=strategy_name,
                    payload=payload,
                )
                if not document_text:
                    continue
                documents.append(TopicDocument(document_id=payload_file.stem, text=document_text))

            if documents:
                corpora.append(
                    StrategyCorpus(
                        source=source_name,
                        strategy=strategy_name,
                        documents=tuple(documents),
                    )
                )

    if not corpora:
        raise FileNotFoundError(
            f"No strategy corpora found under phase2 output directory: {phase2_output_dir}"
        )

    return corpora


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


def _strategy_payload_to_text(strategy_name: str, payload: Mapping[str, object]) -> str:
    """Convert a Phase 2 strategy payload into one consolidated model document text."""
    raw_inputs = payload.get("inputs")
    if not isinstance(raw_inputs, list):
        return ""

    chunks: list[str] = []
    section_weights = _extract_section_weights(payload)

    for raw_item in raw_inputs:
        if not isinstance(raw_item, dict):
            continue

        text_value = raw_item.get("text")
        section_value = raw_item.get("section")
        if not isinstance(text_value, str):
            continue

        normalized_text = text_value.strip()
        if not normalized_text:
            continue

        if strategy_name == ContextStrategy.WEIGHTED_POOLING.value:
            weight = section_weights.get(section_value, 1.0)
            repetitions = _weight_to_repetitions(weight)
            chunks.extend(normalized_text for _ in range(repetitions))
            continue

        chunks.append(normalized_text)

    return "\n\n".join(chunks).strip()


def _extract_section_weights(payload: Mapping[str, object]) -> dict[object, float]:
    raw_weights = payload.get("section_weights")
    if not isinstance(raw_weights, dict):
        return {}

    weights: dict[object, float] = {}
    for key, value in raw_weights.items():
        if not isinstance(value, int | float):
            continue
        weights[key] = float(value)
    return weights


def _weight_to_repetitions(weight: float) -> int:
    if weight <= 0:
        return 0
    return max(1, round(weight))


def _run_one_track(
    *,
    corpus: StrategyCorpus,
    engine: TopicModelEngine,
    phase3_output_dir: Path,
    top_n_terms: int,
    n_topics: int,
) -> PhaseThreeRunResult:
    try:
        output = engine.fit(
            corpus.documents,
            top_n_terms=top_n_terms,
            n_topics=n_topics,
        )
        output_path = _write_engine_output(
            source=corpus.source,
            strategy=corpus.strategy,
            output=output,
            phase3_output_dir=phase3_output_dir,
        )
        return PhaseThreeRunResult(
            source=corpus.source,
            strategy=corpus.strategy,
            engine=engine.name,
            status="ok",
            message="completed",
            output_path=str(output_path),
        )
    except Exception as exc:  # pragma: no cover - integration failure path
        return PhaseThreeRunResult(
            source=corpus.source,
            strategy=corpus.strategy,
            engine=engine.name,
            status="error",
            message=str(exc),
        )


def _write_engine_output(
    *,
    source: str,
    strategy: str,
    output: EngineOutput,
    phase3_output_dir: Path,
) -> Path:
    target_dir = phase3_output_dir / source / strategy
    target_dir.mkdir(parents=True, exist_ok=True)

    output_payload = {
        "source": source,
        "strategy": strategy,
        "document_count": len(output.assignments),
        **output.to_dict(),
    }

    output_path = target_dir / f"{output.engine}.json"
    output_path.write_text(
        json.dumps(output_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def _write_report(results: Sequence[PhaseThreeRunResult], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.to_dict() for result in results]
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 3 multi-track topic modeling pipeline.")
    parser.add_argument(
        "--phase2-output-dir",
        type=Path,
        default=Path("data/phase2_output"),
        help="Directory containing Phase 2 transformed outputs.",
    )
    parser.add_argument(
        "--phase3-output-dir",
        type=Path,
        default=Path("data/phase3_output"),
        help="Directory where Phase 3 topic-model outputs will be written.",
    )
    parser.add_argument(
        "--engines",
        type=str,
        default="lda",
        help="Comma-separated engine list: lda, bertopic, top2vec.",
    )
    parser.add_argument(
        "--top-n-terms",
        type=int,
        default=10,
        help="Number of top terms per topic in serialized outputs.",
    )
    parser.add_argument(
        "--n-topics",
        type=int,
        default=8,
        help="Target topic count per engine run.",
    )
    return parser


__all__ = [
    "load_phase2_corpora",
    "run_phase3_batch",
]


if __name__ == "__main__":
    raise SystemExit(main())
