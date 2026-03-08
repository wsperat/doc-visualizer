# Doc-visualizer
## Phase 1: Data Ingestion & Preprocessing

Before the "smart" stuff happens, you need clean text. PDF structures can be messy (columns, headers, footers).

* **Tools:** `PyMuPDF` (fastest) or `marker` (best for preserving layout/math).
* **Workflow:** 1.  Extract text from PDF.
2.  Clean noise (OCR errors, extra whitespace, stop-word removal for LDA).
3.  **Chunking:** If documents are long, decide if you're modeling at the *page* level or *document* level.

---

## Phase 2: Dual-Track Topic Modeling

To meet the "different strategies" requirement, I recommend implementing both a probabilistic model and a transformer-based model.

| Strategy | Technology | Best For... |
| --- | --- | --- |
| **Probabilistic (LDA)** | `Gensim` | Interpretable keywords, fast, works on smaller datasets. |
| **Neural (BERTopic)** | `BERTopic` | Capturing semantic nuances and context using embeddings. |
| **Matrix Factorization (NMF)** | `Scikit-learn` | Short documents or very distinct categories. |

---

## Phase 3: Summarization & Quality Control

This is where it gets interesting. You aren't just summarizing; you're *auditing* the summary.

* **Summarization:** Use `HuggingFace` transformers (e.g., `BART` or `T5`) or an LLM API (OpenAI/Claude).
* **QC Logic:** Generate embeddings for both the original text ($V_{orig}$) and the summary ($V_{sum}$).
* **Metric:** Use **Cosine Similarity** to check if the summary stayed "on topic."

$$\text{Similarity} = \frac{V_{orig} \cdot V_{sum}}{\|V_{orig}\| \|V_{sum}\|}$$

> **Pro-tip:** If the similarity score drops below a certain threshold (e.g., 0.80), your dashboard should flag that summary as "low fidelity."

---

## Phase 4: Dimensionality Reduction & Viz

To plot documents in 2D, you need to squash high-dimensional embeddings.

1. **UMAP (Uniform Manifold Approximation and Projection):** Generally superior to t-SNE for preserving both local and global structure. It’s faster and handles large datasets better.
2. **PCA (Principal Component Analysis):** Good as a baseline, but often fails to capture the "clusters" in natural language.

---

## Phase 5: The Dashboard (Plotly + Streamlit)

While you mentioned Plotly, I highly recommend using **Streamlit** as the framework to host your Plotly charts. It allows you to build a Python UI in minutes.

* **Interactive Map:** A Plotly scatter plot where each point is a document. Hovering reveals the summary and the top topics.
* **Intertopic Distance Map:** A visualization showing how "close" different topics are to one another.
* **QC Heatmap:** A chart showing the similarity scores across the corpus to identify where the summarizer struggled.

---

### Suggested Tech Stack

* **Language:** Python 3.10+
* **NLP:** `BERTopic`, `Spacy`, `Transformers`
* **Math/Stats:** `Scikit-learn`, `UMAP-learn`
* **Frontend:** `Streamlit`
* **Plotting:** `Plotly`
