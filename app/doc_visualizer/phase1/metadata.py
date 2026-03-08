"""Metadata persistence helpers for dashboard integration."""

from __future__ import annotations

import json
from pathlib import Path

from doc_visualizer.phase1.models import PaperMetadata


class MetadataJsonWriter:
    """Write isolated metadata (including references) to JSON."""

    def write(self, metadata: PaperMetadata, output_path: Path) -> None:
        """Persist metadata to a JSON file with deterministic ordering."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(metadata.to_dict(), indent=2, sort_keys=True)
        output_path.write_text(f"{serialized}\n", encoding="utf-8")
