"""Modular literature-ingestion, extraction, retrieval, and notation services."""

from .fetchers import LiteratureFetcher
from .nomenclature import NomenclatureMapper
from .openalex import OpenAlexClient
from .pdf_text import PDFTextExtractor
from .retrieval import LiteratureRetriever, detect_duplicate_results

__all__ = [
    "LiteratureFetcher",
    "LiteratureRetriever",
    "NomenclatureMapper",
    "OpenAlexClient",
    "PDFTextExtractor",
    "detect_duplicate_results",
]
