# doc-visualizer
Apply different strategies for document visualization in an interactive way.

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
  --max-tokens 512
```

Optional section weights:
```bash
UV_CACHE_DIR=.uv-cache uv run python -m doc_visualizer.phase2.run_batch \
  --phase1-output-dir data/phase1_output \
  --phase2-output-dir data/phase2_output \
  --weights-json data/section_weights.json
```

#### Phase 2 output structure
- `data/phase2_output/whole_doc_mean_pool/*.json`
  - One transformed payload per document.
- `data/phase2_output/parent_child_prepend/*.json`
  - Chunked and prepended inputs (title + abstract + section).
- `data/phase2_output/weighted_pooling/*.json`
  - Section-level inputs + resolved weights for each document.
- `data/phase2_output/report.json`
  - Per-document transformation status report.

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

## Git tracking rules
Generated processing outputs and PDFs are ignored by git:
- `data/phase1_output/`
- `*.pdf`
