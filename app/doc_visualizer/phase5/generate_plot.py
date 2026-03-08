"""Generate interactive Plotly HTML map from Phase 5 outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Protocol, cast

from doc_visualizer.phase5.plot_records import load_plot_records


class _FigureLike(Protocol):
    def add_trace(self, trace: object) -> None: ...

    def update_layout(self, **kwargs: object) -> None: ...


class _GraphObjectsLike(Protocol):
    def Figure(self) -> _FigureLike: ...  # noqa: N802

    def Scatter(self, **kwargs: object) -> object: ...  # noqa: N802


class _WriteHtmlCallable(Protocol):
    def __call__(
        self,
        figure: _FigureLike,
        file: str,
        *,
        include_plotlyjs: str,
        full_html: bool,
    ) -> None: ...


def main() -> int:
    args = _build_arg_parser().parse_args()
    build_plot_html(
        phase5_output_dir=args.phase5_output_dir.resolve(),
        metadata_dir=args.metadata_dir.resolve(),
        output_html=args.output_html.resolve(),
    )
    print(f"Wrote interactive map: {args.output_html.resolve()}")
    return 0


def build_plot_html(
    *,
    phase5_output_dir: Path,
    metadata_dir: Path,
    output_html: Path,
) -> None:
    graph_objects = _load_plotly_graph_objects()
    html_writer = _load_plotly_write_html()

    records = load_plot_records(
        phase5_output_dir=phase5_output_dir,
        metadata_dir=metadata_dir,
    )
    if not records:
        raise FileNotFoundError(f"No map points found under {phase5_output_dir}")

    figure = graph_objects.Figure()
    views = sorted({record.view for record in records})

    trace_names: list[str] = []
    for view in views:
        view_records = [record for record in records if record.view == view]
        for cluster_id in sorted({record.cluster_id for record in view_records}):
            cluster_records = [record for record in view_records if record.cluster_id == cluster_id]
            figure.add_trace(
                graph_objects.Scatter(
                    x=[record.x for record in cluster_records],
                    y=[record.y for record in cluster_records],
                    mode="markers",
                    marker={"size": 11, "line": {"width": 0.5, "color": "#1a1a1a"}},
                    name=f"{view} | cluster {cluster_id}",
                    visible=view == views[0],
                    customdata=[
                        [
                            record.title,
                            record.document_id,
                            record.source,
                            record.strategy,
                            record.cluster_id,
                            (
                                f"{record.similarity_score:.3f}"
                                if record.similarity_score is not None
                                else "n/a"
                            ),
                            (
                                str(record.is_below_threshold)
                                if record.is_below_threshold is not None
                                else "n/a"
                            ),
                        ]
                        for record in cluster_records
                    ],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Document ID: %{customdata[1]}<br>"
                        "Source: %{customdata[2]}<br>"
                        "Strategy: %{customdata[3]}<br>"
                        "Cluster: %{customdata[4]}<br>"
                        "Similarity: %{customdata[5]}<br>"
                        "Below threshold: %{customdata[6]}<extra></extra>"
                    ),
                )
            )
            trace_names.append(view)

    buttons = []
    for view in views:
        visible = [trace_view == view for trace_view in trace_names]
        buttons.append(
            {
                "label": view,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"Document Map ({view})"},
                ],
            }
        )

    figure.update_layout(
        title=f"Document Map ({views[0]})",
        xaxis_title="UMAP X",
        yaxis_title="UMAP Y",
        template="plotly_white",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "y": 1.18,
            }
        ],
        margin={"l": 60, "r": 30, "t": 90, "b": 60},
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    html_writer(figure, str(output_html), include_plotlyjs="cdn", full_html=True)


def _load_plotly_graph_objects() -> _GraphObjectsLike:
    try:
        import plotly.graph_objects as graph_objects  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("plotly is not installed") from exc
    return cast(_GraphObjectsLike, graph_objects)


def _load_plotly_write_html() -> _WriteHtmlCallable:
    try:
        from plotly.io import write_html  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("plotly is not installed") from exc
    return cast(_WriteHtmlCallable, write_html)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML from Phase 5 map outputs."
    )
    parser.add_argument(
        "--phase5-output-dir",
        type=Path,
        default=Path("data/phase5_output"),
        help="Directory containing Phase 5 map outputs.",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("data/phase1_output/metadata"),
        help="Phase 1 metadata directory used for document titles.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("data/phase5_output/document_map.html"),
        help="Target HTML path.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
