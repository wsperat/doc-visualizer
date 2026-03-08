"""Batch runner for Phase 1 processing."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from doc_visualizer.phase1.grobid_gateway import GrobidClientGateway
from doc_visualizer.phase1.metadata import MetadataJsonWriter
from doc_visualizer.phase1.pipeline import PhaseOnePipeline
from doc_visualizer.phase1.tei_parser import BeautifulSoupTeiParser


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    """Result status for one input PDF."""

    pdf_path: Path
    status: str
    message: str


def main() -> int:
    """Run Phase 1 processing for all PDFs in the input directory."""
    args = _build_arg_parser().parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    metadata_dir = output_dir / "metadata"
    content_dir = output_dir / "content"

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return 1

    pipeline = PhaseOnePipeline(
        xml_source=GrobidClientGateway(config_path=args.config_path),
        tei_parser=BeautifulSoupTeiParser(),
    )
    metadata_writer = MetadataJsonWriter()

    results: list[ProcessingResult] = []
    for pdf_path in pdf_files:
        results.append(
            _process_one_pdf(
                pdf_path=pdf_path,
                pipeline=pipeline,
                metadata_writer=metadata_writer,
                metadata_dir=metadata_dir,
                content_dir=content_dir,
            )
        )

    _write_report(results, output_dir / "report.json")

    succeeded = sum(1 for result in results if result.status == "ok")
    failed = len(results) - succeeded
    print(f"Processed {len(results)} files: {succeeded} succeeded, {failed} failed.")
    return 0 if failed == 0 else 2


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Doc Visualizer Phase 1 on a directory of PDFs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing input PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/phase1_output"),
        help="Directory where content and metadata JSON outputs are written.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional grobid-client-python config path.",
    )
    return parser


def _process_one_pdf(
    pdf_path: Path,
    pipeline: PhaseOnePipeline,
    metadata_writer: MetadataJsonWriter,
    metadata_dir: Path,
    content_dir: Path,
) -> ProcessingResult:
    try:
        parsed = pipeline.process_pdf(pdf_path)

        metadata_path = metadata_dir / f"{pdf_path.stem}.json"
        metadata_writer.write(parsed.metadata, metadata_path)

        content_payload = parsed.content_payload()
        content_path = content_dir / f"{pdf_path.stem}.json"
        content_path.parent.mkdir(parents=True, exist_ok=True)
        content_path.write_text(
            json.dumps(content_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return ProcessingResult(pdf_path=pdf_path, status="ok", message="processed")
    except Exception as exc:  # pragma: no cover - integration failure path
        return ProcessingResult(pdf_path=pdf_path, status="error", message=str(exc))


def _write_report(results: list[ProcessingResult], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"pdf_path": str(result.pdf_path), "status": result.status, "message": result.message}
        for result in results
    ]
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
