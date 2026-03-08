# doc-visualizer
Apply different strategies for document visualization in an interactive way.

## Current Process (Phase 1)
The current implementation processes scientific PDFs into structured section content and isolated metadata.

### What it does
1. Reads all `*.pdf` files from `data/`.
2. Sends each PDF to a local GROBID server (`http://localhost:8070`).
3. Parses returned TEI XML.
4. Extracts core sections:
   - `abstract`
   - `introduction`
   - `methods`
   - `results`
   - `conclusion`
5. Extracts metadata:
   - `title`
   - `authors`
   - `year`
   - `references`
6. Removes in-text bibliography reference markers from content text.
7. Extracts all detected raw section headings/subsections (including non-standard titles such as
   `Chapter 1`, `Defining terms`, `Types and recommendations`).
8. Writes per-document JSON outputs + batch report.

### Prerequisites
- Docker running locally.
- GROBID container available on port `8070`.
- Python dependencies installed (`uv sync --all-extras`).

### Run processing
```bash
UV_CACHE_DIR=.uv-cache uv run python -m doc_visualizer.phase1.run_batch \
  --input-dir data \
  --output-dir data/phase1_output
```

### Output structure
- `data/phase1_output/grobid_content/*.json`
  - Section-only payload for embeddings.
  - Includes only non-empty section keys.
- `data/phase1_output/raw_content/*.json`
  - Full section payload for all discovered body sections/subsections.
  - Each entry includes: `title`, `text`, `level`, `position`.
- `data/phase1_output/metadata/*.json`
  - Metadata payload for dashboard/paper info.
  - Keys: `title`, `authors`, `year`, `references`.
- `data/phase1_output/report.json`
  - One entry per input PDF with:
    - `pdf_path`
    - `status` (`ok` or `error`)
    - `message`

### Git tracking rules
Generated processing outputs and PDFs are ignored by git:
- `data/phase1_output/`
- `*.pdf`
