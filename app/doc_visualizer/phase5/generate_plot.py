"""Generate interactive Plotly HTML map from Phase 5 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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

    title_by_document_id = _load_titles(metadata_dir)
    records = _load_map_records(phase5_output_dir, title_by_document_id)
    if not records:
        raise FileNotFoundError(f"No map points found under {phase5_output_dir}")

    figure = graph_objects.Figure()
    views = sorted({record["view"] for record in records})

    trace_names: list[str] = []
    for view in views:
        view_records = [record for record in records if record["view"] == view]
        for cluster_id in sorted({record["cluster_id"] for record in view_records}):
            cluster_records = [
                record for record in view_records if record["cluster_id"] == cluster_id
            ]
            figure.add_trace(
                graph_objects.Scatter(
                    x=[record["x"] for record in cluster_records],
                    y=[record["y"] for record in cluster_records],
                    mode="markers",
                    marker={"size": 11, "line": {"width": 0.5, "color": "#1a1a1a"}},
                    name=f"{view} | cluster {cluster_id}",
                    visible=view == views[0],
                    customdata=[
                        [
                            record["title"],
                            record["document_id"],
                            record["source"],
                            record["strategy"],
                            record["cluster_id"],
                            record.get("similarity_score"),
                            record.get("is_below_threshold"),
                        ]
                        for record in cluster_records
                    ],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Document ID: %{customdata[1]}<br>"
                        "Source: %{customdata[2]}<br>"
                        "Strategy: %{customdata[3]}<br>"
                        "Cluster: %{customdata[4]}<br>"
                        "Similarity: %{customdata[5]:.3f}<br>"
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


def _load_titles(metadata_dir: Path) -> dict[str, str]:
    titles: dict[str, str] = {}
    for metadata_file in metadata_dir.glob("*.json"):
        payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        title = payload.get("title")
        if isinstance(title, str) and title.strip():
            titles[metadata_file.stem] = title.strip()
    return titles


def _load_map_records(
    phase5_output_dir: Path,
    titles: dict[str, str],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for map_file in sorted(phase5_output_dir.glob("*/*/map.json")):
        source = map_file.parent.parent.name
        strategy = map_file.parent.name
        view = f"{source} / {strategy}"

        payload = json.loads(map_file.read_text(encoding="utf-8"))
        points = payload.get("points")
        if not isinstance(points, list):
            continue

        for point in points:
            if not isinstance(point, dict):
                continue
            document_id = point.get("document_id")
            x_value = point.get("x")
            y_value = point.get("y")
            cluster_id = point.get("cluster_id")
            if not isinstance(document_id, str):
                continue
            if not isinstance(x_value, int | float):
                continue
            if not isinstance(y_value, int | float):
                continue
            if not isinstance(cluster_id, int):
                continue

            title = titles.get(document_id, document_id)
            records.append(
                {
                    "view": view,
                    "source": source,
                    "strategy": strategy,
                    "document_id": document_id,
                    "title": title,
                    "x": float(x_value),
                    "y": float(y_value),
                    "cluster_id": cluster_id,
                    "similarity_score": point.get("similarity_score"),
                    "is_below_threshold": point.get("is_below_threshold"),
                }
            )
    return records


def _load_plotly_graph_objects() -> object:
    try:
        import plotly.graph_objects as graph_objects
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("plotly is not installed") from exc
    return graph_objects


def _load_plotly_write_html():
    try:
        from plotly.io import write_html
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("plotly is not installed") from exc
    return write_html


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate interactive HTML from Phase 5 map outputs.")
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
