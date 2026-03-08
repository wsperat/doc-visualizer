# Doc Visualizer
## Phase 1: Structural Parsing & Metadata Isolation

Scientific PDFs are not "blocks of text"; they are hierarchies. We must treat them as such.

* **Primary Tool:** **GROBID** (via `grobid-client-python`). It is the industry standard for parsing scientific PDFs into structured XML/TEI.
* **The Extraction Logic:** * **Content:** Extract `Abstract`, `Introduction`, `Methods`, `Results`, and `Conclusion` into a dictionary.
* **Metadata:** Extract `Title`, `Authors`, `Year`, and **References**.
* **Reference Handling:** References are stripped from the text body to prevent "citation noise" in the embeddings but stored in a metadata JSON for the dashboard's "Paper Info" view.



---

## Phase 2: User-Selectable Context Strategies

Your system will allow users to choose how the model "perceives" the document structure.

| Strategy | Implementation Logic | Best For... |
| --- | --- | --- |
| **Whole Doc (Mean Pool)** | Embed all sections; take the mathematical average of all vectors. | High-level categorization. |
| **Parent-Child Prepend** | Split sections into 512-token chunks. **Prepend** the Title and Abstract to *every* chunk before embedding. | Preserving "Why" the research matters in technical sections. |
| **Weighted Pooling** | User assigns weights ($w$) to sections. $V_{doc} = \frac{\sum (w_i \cdot V_i)}{\sum w_i}$. | Prioritizing the "Results" and "Abstract" over "Introduction." |

---

## Phase 3: Multi-Track Topic Modeling

To compare strategies, the pipeline will run two (or three) parallel engines:

1. **LDA (Latent Dirichlet Allocation):**
* **Mechanism:** Probabilistic/Bag-of-Words.
* **Preprocessing:** Requires heavy cleaning (stop-words, lemmatization).
* **Output:** Word-cloud style topics (e.g., "protein, cell, signaling").


2. **BERTopic (Neural):**
* **Mechanism:** SBERT Embeddings + UMAP + HDBSCAN + c-TF-IDF.
* **Benefit:** Captures context and polysemy (words with multiple meanings).


3. **Top2Vec (Optional):**
* **Mechanism:** Jointly learns word and document vectors. Good for very dense, high-signal corpora.



---

## Phase 4: Summarization & Semantic Auditing (QC)

This is the "closed-loop" part of your project where the AI checks its own homework.

* **Summarization Engine:** Use **LED (Longformer Encoder-Decoder)** or **LongT5**. These models handle up to 16,384 tokens, allowing for full-paper context.
* **Quality Control Logic:**
1. Generate Summary ($S$).
2. Embed $S$ into vector $V_s$.
3. Compare $V_s$ against the **Document Vector** ($V_{doc}$) generated in Phase 2.
4. **Metric:** Cosine Similarity score.


$$\text{Similarity Score} = \frac{V_{doc} \cdot V_s}{\|V_{doc}\| \|V_s\|}$$



> **Dashboard Action:** Papers with a similarity score $< 0.80$ should be highlighted in **red** on the dashboard to warn the user of potential "summary drift" or hallucinations.

---

## Phase 5: Dimensionality Reduction & Mapping

Transforming high-dimensional embeddings into a 2D "Knowledge Map."

* **UMAP (Uniform Manifold Approximation and Projection):** * Used to project the $V_{doc}$ vectors.
* **Hyperparameter Control:** Expose `n_neighbors` (Balance local vs. global structure) and `min_dist` (How tightly dots cluster) to the user.


* **Clustering:** Use **HDBSCAN** to automatically identify topic clusters on the 2D plane without needing to pre-define the number of topics ($k$).

---

## Phase 6: The Plotly-Streamlit Dashboard

The "Command Center" for the user.

* **View A: The Galaxy Map (Plotly Scatter):**
* **X/Y:** UMAP Coordinates.
* **Color:** Topic Cluster.
* **Hover:** Title, Abstract, and QC Score.
* **Click:** Opens a sidebar with the **Full Summary** and the list of **References** (Metadata).


* **View B: Strategy Comparison:**
* Side-by-side bar charts showing how the topic distribution changes between LDA and BERTopic.


* **View C: Fidelity Audit:**
* A ranked list of documents by their Cosine Similarity score.

---

### Tech Stack

* **Parsing:** `grobid-client-python`, `BeautifulSoup` (for XML).
* **Embeddings:** `sentence-transformers` (Model: `all-mpnet-base-v2`).
* **Topic Modeling:** `bertopic`, `gensim`, `scikit-learn`.
* **Reduction:** `umap-learn`, `hdbscan`.
* **UI/Viz:** `streamlit`, `plotly`.
