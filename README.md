# doc-visualizer
Apply different strategies for document visualization in an interactive way.

## Async API Service
All phases are available through an async FastAPI service. Each request creates a background job
and immediately returns a `job_id`; status/results are polled via `/jobs/{job_id}`.

Run API server:
```bash
UV_CACHE_DIR=.uv-cache uv run uvicorn doc_visualizer.api.main:app --host 0.0.0.0 --port 8000
```

Core endpoints:
- `GET /health`
- `GET /jobs`
- `GET /jobs/{job_id}`
- `POST /phases/1/run`
- `POST /phases/2/run`
- `POST /phases/3/run`
- `POST /phases/4/run`
- `POST /phases/5/run`
- `POST /phases/5/plot`

## Implemented Phases

### Phase 1: Structural Parsing and Metadata Isolation
Phase 1 processes scientific PDFs into structured text and isolated metadata using GROBID.

#### What it does
1. Reads all `*.pdf` files from `data/`.
2. Sends each PDF to a local GROBID server (`http://localhost:8070`).
3. Parses returned TEI XML.
4. Extracts mapped scientific sections:
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
7. Extracts all raw section/subsection nodes (including non-standard headings such as `Chapter 1`,
   `Defining terms`, `Types and recommendations`).
8. Writes JSON outputs and a run report.

#### Prerequisites
- Docker running locally.
- GROBID container available on port `8070`.
- Python dependencies installed (`uv sync --all-extras`).

#### Run Phase 1 batch processing
```bash
UV_CACHE_DIR=.uv-cache uv run python -m doc_visualizer.phase1.run_batch \
  --input-dir data \
  --output-dir data/phase1_output
```

#### Phase 1 output structure
- `data/phase1_output/grobid_content/*.json`
  - Non-empty mapped section payload for downstream embeddings.
- `data/phase1_output/raw_content/*.json`
  - Full raw section list.
  - Each entry includes: `title`, `text`, `level`, `position`.
- `data/phase1_output/metadata/*.json`
  - Metadata payload with keys: `title`, `authors`, `year`, `references`.
- `data/phase1_output/report.json`
  - One entry per PDF: `pdf_path`, `status`, `message`.

### Phase 2: User-Selectable Context Strategies
Phase 2 provides strategy-specific context construction and pooling logic for embeddings.

#### Strategies implemented
- `whole_doc_mean_pool`
  - Build one embedding input per non-empty section and mean-pool all vectors.
- `parent_child_prepend`
  - Split section text into `max_tokens` chunks (default `512`).
  - Prepend `Title` + `Abstract` + section label to every chunk.
  - Mean-pool chunk vectors.
- `weighted_pooling`
  - Build section-level vectors and apply user weights:
  - `V_doc = sum(w_i * V_i) / sum(w_i)`

#### Phase 2 code entry points
- `doc_visualizer.phase2.ContextStrategy`
- `doc_visualizer.phase2.PhaseTwoContextService`
- `doc_visualizer.phase2.build_whole_doc_inputs`
- `doc_visualizer.phase2.build_parent_child_prepend_inputs`
- `doc_visualizer.phase2.mean_pool`
- `doc_visualizer.phase2.weighted_pool`

#### Run Phase 2 transformation pipeline
```bash
UV_CACHE_DIR=.uv-cache uv run python -m doc_visualizer.phase2.run_batch \
  --phase1-output-dir data/phase1_output \
  --phase2-output-dir data/phase2_output \
  --max-tokens 512 \
  --content-source grobid
```

Optional section weights:
```bash
UV_CACHE_DIR=.uv-cache uv run python -m doc_visualizer.phase2.run_batch \
  --phase1-output-dir data/phase1_output \
  --phase2-output-dir data/phase2_output \
  --weights-json data/section_weights.json
```

Use richer hierarchical content from Phase 1 raw extraction:
```bash
UV_CACHE_DIR=.uv-cache uv run python -m doc_visualizer.phase2.run_batch \
  --phase1-output-dir data/phase1_output \
  --phase2-output-dir data/phase2_output \
  --content-source raw
```

Or combine standard + raw sources:
```bash
UV_CACHE_DIR=.uv-cache uv run python -m doc_visualizer.phase2.run_batch \
  --phase1-output-dir data/phase1_output \
  --phase2-output-dir data/phase2_output \
  --content-source hybrid
```

Run standard and rich raw pipelines in one execution:
```bash
UV_CACHE_DIR=.uv-cache uv run python -m doc_visualizer.phase2.run_batch \
  --phase1-output-dir data/phase1_output \
  --phase2-output-dir data/phase2_output \
  --content-source both
```

#### Phase 2 output structure
- `data/phase2_output/whole_doc_mean_pool/*.json`
  - One transformed payload per document.
- `data/phase2_output/parent_child_prepend/*.json`
  - Chunked and prepended inputs (title + abstract + section).
- `data/phase2_output/weighted_pooling/*.json`
  - Section-level inputs + resolved weights for each document.
- Raw/hybrid outputs include hierarchy metadata per section (`title`, `breadcrumb`, `level`, `position`)
  so chapter/section/subsection relationships are preserved.
- `data/phase2_output/report.json`
  - Per-document transformation status report.
- In `both` mode, outputs are namespaced by source:
  - `data/phase2_output/grobid/...`
  - `data/phase2_output/raw/...`
  - with a combined top-level `data/phase2_output/report.json`.

#### Minimal usage example
```python
from collections.abc import Sequence

from doc_visualizer.phase2 import ContextStrategy, PhaseTwoContextService
from doc_visualizer.phase2.models import Vector


class MyEmbeddingBackend:
    def embed(self, texts: Sequence[str]) -> list[Vector]:
        # Replace with sentence-transformers or your preferred backend.
        return [(float(len(text.split())), 1.0) for text in texts]


service = PhaseTwoContextService(embedding_backend=MyEmbeddingBackend())

sections = {
    "abstract": "Short abstract text.",
    "introduction": "Intro text.",
    "results": "Result text.",
}

vector = service.build_document_vector(
    ContextStrategy.WEIGHTED_POOLING,
    title="Example Paper",
    sections=sections,
    section_weights={"abstract": 2.0, "results": 3.0, "introduction": 1.0},
)
```

### Phase 3: Multi-Track Topic Modeling
Phase 3 consumes Phase 2 outputs and runs multiple topic engines per source/strategy corpus.

#### Engines
- `lda` (implemented via scikit-learn LDA with deterministic text cleaning)
- `bertopic` (lazy-loaded wrapper)
- `top2vec` (optional lazy-loaded wrapper)

#### Run Phase 3
```bash
UV_CACHE_DIR=.uv-cache uv run python -m doc_visualizer.phase3.run_batch \
  --phase2-output-dir data/phase2_output \
  --phase3-output-dir data/phase3_output \
  --engines lda \
  --n-topics 8 \
  --top-n-terms 10
```

Run multiple tracks (when optional engine deps are installed):
```bash
UV_CACHE_DIR=.uv-cache uv run python -m doc_visualizer.phase3.run_batch \
  --phase2-output-dir data/phase2_output \
  --phase3-output-dir data/phase3_output \
  --engines lda,bertopic
```

#### Phase 3 output structure
- `data/phase3_output/<source>/<strategy>/<engine>.json`
  - Topic clusters, top terms, document-topic assignments, and engine metadata.
- `data/phase3_output/report.json`
  - Status report for each source+strategy+engine track.

### Phase 4: Summarization and Semantic Auditing
Phase 4 summarizes each transformed strategy document and compares summary semantics against
strategy document vectors using cosine similarity.

#### What it does
1. Loads all Phase 2 strategy payloads (`inputs`) across all sources (`grobid`, `raw`, `default`, etc.).
2. Builds one document text per payload (deduplicated chunk join).
3. Generates a summary (`extractive`, `led`, or `longt5` backend).
4. Builds the strategy document vector:
   - `whole_doc_mean_pool`: mean of chunk embeddings.
   - `parent_child_prepend`: mean of chunk embeddings.
   - `weighted_pooling`: section mean vectors + weighted pooling from `section_weights`.
5. Embeds the summary text and computes cosine similarity against the document vector.
6. Flags drift risk when similarity is below threshold (default `0.80`).

#### Run Phase 4
```bash
UV_CACHE_DIR=.uv-cache uv run python -m doc_visualizer.phase4.run_batch \
  --phase2-output-dir data/phase2_output \
  --phase4-output-dir data/phase4_output \
  --summary-backend extractive \
  --embedding-backend hashing \
  --similarity-threshold 0.80
```

Optional transformer summarizers (requires additional deps/models):
```bash
UV_CACHE_DIR=.uv-cache uv run python -m doc_visualizer.phase4.run_batch \
  --phase2-output-dir data/phase2_output \
  --phase4-output-dir data/phase4_output \
  --summary-backend led
```

#### Phase 4 output structure
- `data/phase4_output/<source>/<strategy>/<document_id>.json`
  - Summary text, cosine similarity score, threshold flag, backend metadata.
- `data/phase4_output/report.json`
  - Status report for each source+strategy+document audit.

### Phase 5: Dimensionality Reduction and Mapping
Phase 5 converts strategy-level document vectors into 2D map coordinates and clusters.

#### What it does
1. Loads Phase 2 strategy payloads per source (`grobid`, `raw`, etc.).
2. Reconstructs one document vector (`V_doc`) for each document using the same strategy logic as Phase 4.
3. Projects vectors into 2D using UMAP (`n_neighbors`, `min_dist` configurable).
4. Clusters projected points with HDBSCAN (no pre-defined `k` required).
5. Optionally overlays Phase 4 QC flags (`similarity_score`, `is_below_threshold`) onto each point.

#### Run Phase 5
```bash
UV_CACHE_DIR=.uv-cache uv run python -m doc_visualizer.phase5.run_batch \
  --phase2-output-dir data/phase2_output \
  --phase4-output-dir data/phase4_output \
  --phase5-output-dir data/phase5_output \
  --embedding-backend hashing \
  --n-neighbors 15 \
  --min-dist 0.1 \
  --min-cluster-size 5
```

#### Phase 5 output structure
- `data/phase5_output/<source>/<strategy>/map.json`
  - 2D points (`x`, `y`), cluster labels, and optional QC overlay fields.
- `data/phase5_output/report.json`
  - Status report for each source+strategy map generation run.

## Git tracking rules
Generated processing outputs and PDFs are ignored by git:
- `data/phase1_output/`
- `data/phase2_output/`
- `data/phase3_output/`
- `data/phase4_output/`
- `data/phase5_output/`
- `*.pdf`
