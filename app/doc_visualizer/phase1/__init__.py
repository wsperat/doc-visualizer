"""Phase 1: Structural parsing and metadata isolation."""

from doc_visualizer.phase1.grobid_gateway import GrobidClientGateway
from doc_visualizer.phase1.metadata import MetadataJsonWriter
from doc_visualizer.phase1.models import (
    DocumentSections,
    PaperMetadata,
    ParsedDocument,
    RawSection,
)
from doc_visualizer.phase1.pipeline import PhaseOnePipeline
from doc_visualizer.phase1.tei_parser import BeautifulSoupTeiParser
from doc_visualizer.phase1.types import SECTION_NAMES, SectionName

__all__ = [
    "SECTION_NAMES",
    "BeautifulSoupTeiParser",
    "DocumentSections",
    "GrobidClientGateway",
    "MetadataJsonWriter",
    "PaperMetadata",
    "ParsedDocument",
    "PhaseOnePipeline",
    "RawSection",
    "SectionName",
]
