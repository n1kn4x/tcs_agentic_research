"""Modular literature-ingestion, extraction, and retrieval services."""

from .fetchers import LiteratureFetcher
from .index import LiteratureIndex
from .openalex import OpenAlexClient
from .pdf_text import PDFTextExtractor
from .retrieval import LiteratureRetriever, detect_duplicate_results

__all__ = [
    "LiteratureFetcher",
    "LiteratureIndex",
    "LiteratureRetriever",
    "OpenAlexClient",
    "PDFTextExtractor",
    "detect_duplicate_results",
]
