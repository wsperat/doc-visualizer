"""Phase 4: Summarization and semantic auditing."""

from doc_visualizer.phase4.embeddings import (
    HashingEmbeddingBackend,
    SentenceTransformerEmbeddingBackend,
    build_embedding_backend,
)
from doc_visualizer.phase4.models import AuditRecord, PhaseFourRunResult, StrategyDocument
from doc_visualizer.phase4.service import PhaseFourAuditService
from doc_visualizer.phase4.summarizers import (
    ExtractiveLeadSummarizer,
    TransformersSeq2SeqSummarizer,
    build_summarizer,
)
from doc_visualizer.phase4.vector_math import cosine_similarity

__all__ = [
    "AuditRecord",
    "ExtractiveLeadSummarizer",
    "HashingEmbeddingBackend",
    "PhaseFourAuditService",
    "PhaseFourRunResult",
    "SentenceTransformerEmbeddingBackend",
    "StrategyDocument",
    "TransformersSeq2SeqSummarizer",
    "build_embedding_backend",
    "build_summarizer",
    "cosine_similarity",
]
