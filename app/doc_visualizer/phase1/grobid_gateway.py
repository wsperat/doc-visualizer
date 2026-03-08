"""GROBID gateway implementation using grobid-client-python."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory
from typing import cast

from doc_visualizer.phase1.protocols import GrobidXmlSource


class GrobidClientGateway(GrobidXmlSource):
    """Convert PDFs into TEI XML via GROBID client."""

    def __init__(self, config_path: str | None = None) -> None:
        self._config_path = config_path

    def process_pdf(self, pdf_path: Path) -> str:
        """Process one PDF and return its TEI XML payload as a string."""
        client_class = _load_grobid_client_class()
        client = _build_client(client_class, self._config_path)

        with (
            TemporaryDirectory(prefix="grobid_input_") as input_dir,
            TemporaryDirectory(prefix="grobid_output_") as output_dir,
        ):
            copied_pdf = Path(input_dir) / pdf_path.name
            copy2(pdf_path, copied_pdf)
            _invoke_process(client, input_dir=input_dir, output_dir=output_dir)

            tei_path = _resolve_tei_path(output_dir=Path(output_dir), pdf_stem=pdf_path.stem)

            return tei_path.read_text(encoding="utf-8")


def _load_grobid_client_class() -> Callable[..., object]:
    """Load GrobidClient class lazily to keep imports optional in tests."""
    try:
        module = import_module("grobid_client.grobid_client")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "grobid-client-python is not installed. Install project dependencies first."
        ) from exc

    client_class = getattr(module, "GrobidClient", None)
    if callable(client_class):
        return cast(Callable[..., object], client_class)

    raise RuntimeError("grobid-client-python is installed but GrobidClient was not found.")


def _build_client(client_class: Callable[..., object], config_path: str | None) -> object:
    """Instantiate GrobidClient using either explicit or default configuration."""
    if config_path is None:
        return client_class()
    return client_class(config_path=config_path)


def _invoke_process(client: object, input_dir: str, output_dir: str) -> None:
    """Invoke the client process method across compatible API variants."""
    process_fn = getattr(client, "process", None)
    if not callable(process_fn):
        raise RuntimeError("GROBID client instance does not expose a callable 'process' method.")

    process = process_fn
    _call_process(process=process, input_dir=input_dir, output_dir=output_dir)


def _call_process(process: Callable[..., object], input_dir: str, output_dir: str) -> None:
    """Call process using the most common signatures used by grobid-client-python."""
    try:
        process(
            "processFulltextDocument",
            input_dir,
            output_dir,
            force=True,
            consolidate_header=True,
            consolidate_citations=True,
        )
        return
    except TypeError:
        pass

    process(
        service="processFulltextDocument",
        input_path=input_dir,
        output=output_dir,
        force=True,
        consolidate_header=True,
        consolidate_citations=True,
    )


def _resolve_tei_path(output_dir: Path, pdf_stem: str) -> Path:
    """Locate the generated TEI file across known grobid-client naming variants."""
    direct_candidates = (
        output_dir / f"{pdf_stem}.grobid.tei.xml",
        output_dir / f"{pdf_stem}.tei.xml",
    )
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    glob_candidates = sorted(output_dir.glob(f"{pdf_stem}*.tei.xml"))
    if len(glob_candidates) == 1:
        return glob_candidates[0]

    all_tei_candidates = sorted(output_dir.glob("*.tei.xml"))
    if len(all_tei_candidates) == 1:
        return all_tei_candidates[0]

    expected_hint = output_dir / f"{pdf_stem}.grobid.tei.xml"
    raise FileNotFoundError(
        "GROBID processing finished without generating expected TEI file. "
        f"Looked for matches under: {expected_hint}"
    )
