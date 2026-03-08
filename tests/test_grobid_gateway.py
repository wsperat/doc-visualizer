from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from doc_visualizer.phase1.grobid_gateway import _call_process, _resolve_tei_path


class PositionalProcessRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> None:
        self.calls.append((args, kwargs))


class KeywordOnlyProcessRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(
        self,
        *,
        service: str,
        input_path: str,
        output: str,
        force: bool,
        consolidate_header: bool,
        consolidate_citations: bool,
    ) -> None:
        self.calls.append(
            (
                (),
                {
                    "service": service,
                    "input_path": input_path,
                    "output": output,
                    "force": force,
                    "consolidate_header": consolidate_header,
                    "consolidate_citations": consolidate_citations,
                },
            )
        )


def test_call_process_supports_positional_signature() -> None:
    process = PositionalProcessRecorder()

    _call_process(process=process, input_dir="in_dir", output_dir="out_dir")

    assert len(process.calls) == 1
    args, kwargs = process.calls[0]
    assert args[:3] == ("processFulltextDocument", "in_dir", "out_dir")
    assert kwargs["force"] is True


def test_call_process_supports_keyword_signature() -> None:
    process = KeywordOnlyProcessRecorder()

    _call_process(process=process, input_dir="in_dir", output_dir="out_dir")

    assert len(process.calls) == 1
    _, kwargs = process.calls[0]
    assert kwargs["service"] == "processFulltextDocument"
    assert kwargs["input_path"] == "in_dir"
    assert kwargs["output"] == "out_dir"


def test_call_process_falls_back_when_positional_call_raises_type_error() -> None:
    class HybridProcess:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def __call__(self, *args: object, **kwargs: object) -> None:
            if args:
                self.calls.append("positional")
                raise TypeError("keyword-only signature")
            self.calls.append("keyword")

    process = HybridProcess()
    process_callable: Callable[..., object] = process

    _call_process(process=process_callable, input_dir="in", output_dir="out")

    assert process.calls == ["positional", "keyword"]


def test_resolve_tei_path_accepts_grobid_suffix(tmp_path: Path) -> None:
    tei_path = tmp_path / "paper.grobid.tei.xml"
    tei_path.write_text("<TEI/>", encoding="utf-8")

    resolved = _resolve_tei_path(output_dir=tmp_path, pdf_stem="paper")

    assert resolved == tei_path
