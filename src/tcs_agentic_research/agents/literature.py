"""Modular literature agent with fetching, extraction, indexing, and retrieval.

The agent is deliberately usable outside the LangGraph loop: network/PDF/retrieval pieces
live in ``tcs_agentic_research.literature`` services, while this class coordinates durable
artifacts and optional LLM extraction. Other agents should call ``answer_query`` rather than
reading raw literature records so they receive stable statement/support IDs and quote provenance.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from ..artifact_store import ArtifactStore
from ..literature.fetchers import (
    LiteratureFetcher,
    _citation_key_from_doi,
    _crossref_authors,
    _crossref_title,
    _crossref_venue,
    _crossref_year,
    normalize_arxiv_id,
    normalize_doi,
    parse_arxiv_id,
    parse_doi,
)
from ..literature.index import LiteratureIndex
from ..literature.openalex import OpenAlexClient
from ..literature.pdf_text import PDFTextExtractor
from ..literature.retrieval import LiteratureRetriever
from ..llm import LLMRouter
from ..prompt_loader import render_prompt
from ..schemas import (
    ArtifactRef,
    LiteratureCandidate,
    LiteratureExtract,
    LiteratureQueryAnswer,
    LiteratureQuote,
    LiteratureSource,
    LiteratureStatement,
    PaperMetadata,
    new_id,
)


class PaperMetadataMismatchError(ValueError):
    """Raised when fetched paper metadata does not match an expected target paper."""


class LiteratureResearcher:
    """Independent literature pipeline.

    Public methods are intentionally tool-like so they can later be exposed through LangGraph tool
    calling: ``search_papers``, ``discover_related``, ``import_candidate``, ``import_url``,
    ``import_arxiv``, ``extract_pdf_text``, ``extract_paper``, and ``answer_query``.
    """

    def __init__(self, store: ArtifactStore, router: LLMRouter, *, prompt_dir: str | None = None):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir
        self.fetcher = LiteratureFetcher(store)
        self.pdf_extractor = PDFTextExtractor(store)
        needs_index_rebuild = not store.exists(LiteratureIndex.INDEX_PATH)
        self.index = LiteratureIndex(store)
        has_literature_artifacts = bool(
            store.read_jsonl("LiteratureDB/papers.jsonl")
            or store.read_jsonl("LiteratureDB/statements.jsonl")
        )
        if (needs_index_rebuild or self.index.is_empty()) and has_literature_artifacts:
            self.index.rebuild()
        self.retriever = LiteratureRetriever(store, index=self.index)
        self.openalex = OpenAlexClient()

    # ------------------------------------------------------------------
    # Import/fetch capabilities
    # ------------------------------------------------------------------
    def import_source(self, source: LiteratureSource) -> PaperMetadata:
        token = source.source.strip()
        citation_key = source.citation_key or None
        arxiv_id = parse_arxiv_id(token) or (token if source.source_type == "arxiv" else None)
        if arxiv_id:
            return self.import_arxiv(
                str(arxiv_id),
                citation_key=citation_key,
                extract_text=source.extract_text,
                expected_title=source.title or None,
            )
        doi = parse_doi(token) or (token if source.source_type == "doi" else None)
        if doi:
            return self.import_doi(
                str(doi),
                citation_key=citation_key,
                extract_text=source.extract_text,
                expected_title=source.title or None,
            )
        return self.import_url(
            token,
            citation_key=citation_key,
            title=source.title or None,
            extract_text=source.extract_text,
            expected_title=source.title or None,
        )

    def import_paper(self, paper: PaperMetadata) -> PaperMetadata:
        """Register already-known paper metadata in ``LiteratureDB``.

        The append-only JSONL ledger remains an audit trail; the SQLite literature index records
        canonical paper aliases so duplicate citation keys/DOIs/arXiv IDs resolve to one paper.
        """
        existing = self._find_existing_paper_for_metadata(paper)
        if existing is not None and existing.paper_id != paper.paper_id:
            self.index.add_alias(existing.paper_id, "citation_key", paper.citation_key)
            merged, changed = _merge_duplicate_paper_metadata(existing, paper)
            if changed:
                self._update_paper_record(merged)
                return merged
            return existing
        if paper.metadata_path:
            metadata_ref = self.store.write_json(paper.metadata_path, paper)
        else:
            rel_dir = f"LiteratureDB/papers/{_safe_slug(paper.citation_key)}"
            metadata_ref = self.store.write_json(f"{rel_dir}/metadata.json", paper)
            paper.metadata_path = metadata_ref.path
        if metadata_ref.path not in {ref.path for ref in paper.artifact_refs}:
            paper.artifact_refs.append(metadata_ref)
        self.store.write_json(paper.metadata_path, paper)
        self.store.append_jsonl("LiteratureDB/papers.jsonl", paper)
        self.index.upsert_paper(paper)
        if paper.text_path and self.store.exists(paper.text_path):
            self.index.index_paper_text(paper)
        return paper

    def import_url(
        self,
        url: str,
        *,
        citation_key: str | None = None,
        title: str | None = None,
        doi: str | None = None,
        extract_text: bool = False,
        expected_title: str | None = None,
        expected_authors: list[str] | None = None,
        expected_year: int | None = None,
        expected_venue: str | None = None,
    ) -> PaperMetadata:
        """Fetch a URL/DOI/arXiv/PDF source into ``LiteratureDB/papers/``.

        When expected metadata is provided, the fetched record is rejected before it is committed
        to the paper ledger if title/authors/year/venue disagree with the target paper.
        """
        arxiv_id = parse_arxiv_id(url)
        doi_value = doi or parse_doi(url)
        if arxiv_id:
            return self.import_arxiv(
                arxiv_id,
                citation_key=citation_key,
                extract_text=extract_text,
                expected_title=expected_title,
                expected_authors=expected_authors,
                expected_year=expected_year,
                expected_venue=expected_venue,
            )
        if doi_value and (parse_doi(url) or not url.lower().startswith(("http://", "https://"))):
            return self.import_doi(
                doi_value,
                citation_key=citation_key,
                extract_text=extract_text,
                expected_title=expected_title,
                expected_authors=expected_authors,
                expected_year=expected_year,
                expected_venue=expected_venue,
            )
        existing = None
        if doi_value:
            existing = self.index.find_paper(doi=doi_value)
        elif citation_key:
            existing = self.index.find_paper(citation_key=citation_key)
        if existing is not None:
            self._validate_expected_metadata(
                existing,
                expected_title=expected_title,
                expected_authors=expected_authors,
                expected_year=expected_year,
                expected_venue=expected_venue,
            )
            if citation_key:
                self.index.add_alias(existing.paper_id, "citation_key", citation_key)
            return self._extract_text_if_requested(existing, extract_text=extract_text)
        paper = self.fetcher.import_url(
            url, citation_key=citation_key, title=title, doi=doi, commit=False
        )
        self._validate_expected_metadata(
            paper,
            expected_title=expected_title,
            expected_authors=expected_authors,
            expected_year=expected_year,
            expected_venue=expected_venue,
        )
        canonical = self.import_paper(paper)
        return self._extract_text_if_requested(canonical, extract_text=extract_text)

    def import_arxiv(
        self,
        arxiv_id: str,
        *,
        citation_key: str | None = None,
        extract_text: bool = False,
        expected_title: str | None = None,
        expected_authors: list[str] | None = None,
        expected_year: int | None = None,
        expected_venue: str | None = None,
    ) -> PaperMetadata:
        clean_id = normalize_arxiv_id(arxiv_id)
        existing = self.index.find_paper(arxiv_id=clean_id)
        if existing is not None:
            self._validate_expected_metadata(
                existing,
                expected_title=expected_title,
                expected_authors=expected_authors,
                expected_year=expected_year,
                expected_venue=expected_venue,
            )
            if citation_key:
                self.index.add_alias(existing.paper_id, "citation_key", citation_key)
            return self._extract_text_if_requested(existing, extract_text=extract_text)
        # Transactional validation: inspect arXiv metadata before downloading/writing PDFs.
        metadata = self.fetcher._fetch_arxiv_metadata(clean_id)
        provisional = PaperMetadata(
            citation_key=citation_key or f"arxiv_{_safe_slug(clean_id)}",
            title=str(metadata.get("title") or f"arXiv:{clean_id}"),
            authors=list(metadata.get("authors") or []),
            year=metadata.get("year"),
            venue="arXiv",
            url=str(metadata.get("url") or f"https://arxiv.org/abs/{clean_id}"),
            arxiv_id=clean_id,
            abstract=str(metadata.get("abstract") or ""),
            source_type="arxiv",
        )
        self._validate_expected_metadata(
            provisional,
            expected_title=expected_title,
            expected_authors=expected_authors,
            expected_year=expected_year,
            expected_venue=expected_venue,
        )
        paper = self.fetcher.import_arxiv(clean_id, citation_key=citation_key, commit=False)
        canonical = self.import_paper(paper)
        return self._extract_text_if_requested(canonical, extract_text=extract_text)

    def import_doi(
        self,
        doi: str,
        *,
        citation_key: str | None = None,
        extract_text: bool = False,
        expected_title: str | None = None,
        expected_authors: list[str] | None = None,
        expected_year: int | None = None,
        expected_venue: str | None = None,
    ) -> PaperMetadata:
        clean_doi = normalize_doi(doi)
        existing = self.index.find_paper(doi=clean_doi)
        if existing is not None:
            self._validate_expected_metadata(
                existing,
                expected_title=expected_title,
                expected_authors=expected_authors,
                expected_year=expected_year,
                expected_venue=expected_venue,
            )
            if citation_key:
                self.index.add_alias(existing.paper_id, "citation_key", citation_key)
            return self._extract_text_if_requested(existing, extract_text=extract_text)
        # Transactional validation: inspect Crossref metadata before downloading/writing assets.
        metadata = self.fetcher._fetch_crossref_metadata(clean_doi)
        provisional = PaperMetadata(
            citation_key=citation_key or _citation_key_from_doi(clean_doi, metadata),
            title=_crossref_title(metadata) or f"DOI:{clean_doi}",
            authors=_crossref_authors(metadata),
            year=_crossref_year(metadata),
            venue=_crossref_venue(metadata),
            url=f"https://doi.org/{clean_doi}",
            doi=clean_doi,
            abstract=str(metadata.get("abstract") or ""),
            source_type="doi",
        )
        self._validate_expected_metadata(
            provisional,
            expected_title=expected_title,
            expected_authors=expected_authors,
            expected_year=expected_year,
            expected_venue=expected_venue,
        )
        paper = self.fetcher.import_doi(clean_doi, citation_key=citation_key, commit=False)
        canonical = self.import_paper(paper)
        return self._extract_text_if_requested(canonical, extract_text=extract_text)

    def _extract_text_if_requested(
        self, paper: PaperMetadata, *, extract_text: bool
    ) -> PaperMetadata:
        if not extract_text or not paper.pdf_path:
            return paper
        self.extract_pdf_text(citation_key=paper.citation_key)
        return self.require_paper(citation_key=paper.citation_key)

    def extract_pdf_text(
        self,
        pdf_path: str | Path | None = None,
        *,
        citation_key: str | None = None,
        paper_id: str | None = None,
    ) -> str:
        """Extract text from an imported or explicit PDF and record/update metadata."""
        paper: PaperMetadata | None = None
        if citation_key or paper_id:
            paper = self.require_paper(citation_key=citation_key, paper_id=paper_id)
            if not paper.pdf_path:
                raise ValueError(f"Paper {paper.citation_key} has no pdf_path to extract")
            pdf_path = paper.pdf_path
        if pdf_path is None:
            raise ValueError("Provide pdf_path, citation_key, or paper_id")
        text_path = self.pdf_extractor.extract_pdf_text(pdf_path)
        if paper is not None:
            paper.text_path = text_path
            text_ref = self.store.artifact_ref(text_path)
            if text_ref.path not in {ref.path for ref in paper.artifact_refs}:
                paper.artifact_refs.append(text_ref)
            self._update_paper_record(paper)
            self.index.index_paper_text(paper)
        return text_path

    # ------------------------------------------------------------------
    # Extraction and statement indexing
    # ------------------------------------------------------------------
    def extract_paper(
        self,
        *,
        citation_key: str | None = None,
        paper_id: str | None = None,
        use_llm: bool = True,
    ) -> LiteratureExtract:
        """Run theorem/algorithm extraction for an imported paper.

        If a PDF exists but text has not yet been extracted, ``extract_pdf_text`` is invoked first.
        """
        paper = self.require_paper(citation_key=citation_key, paper_id=paper_id)
        if not paper.text_path:
            if not paper.pdf_path:
                raise ValueError(f"Paper {paper.citation_key} has neither text_path nor pdf_path")
            self.extract_pdf_text(citation_key=paper.citation_key)
            paper = self.require_paper(citation_key=paper.citation_key)
        text = self.store.read_text(paper.text_path)
        return self.extract_from_text(
            citation_key=paper.citation_key,
            paper_text=text,
            paper_id=paper.paper_id,
            text_artifact_path=paper.text_path,
            use_llm=use_llm,
        )

    def extract_imported_papers(
        self,
        *,
        max_papers: int = 8,
        only_missing: bool = True,
        use_llm: bool = False,
        citation_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        """Deterministically extract statements from imported papers with available text/PDF.

        This is the non-agentic extraction loop used by literature obligations: imported full-text
        sources should not remain unindexed just because the model forgot to call ``extract_paper``.
        The method is conservative, idempotent by citation key when ``only_missing`` is true, and
        returns a compact audit payload suitable for a tool observation or prompt context.
        """
        batch_id = new_id("lit_extract_batch")
        latest_by_key: dict[str, PaperMetadata] = {}
        for paper in self.list_papers():
            if paper.citation_key:
                latest_by_key[paper.citation_key] = paper
        already_extracted = self._citation_keys_with_statement_extractions()
        processed: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        errors: list[dict[str, str]] = []
        support_ids: list[str] = []
        processed_citation_keys: list[str] = []

        selected_papers = (
            [latest_by_key[key] for key in citation_keys or [] if key in latest_by_key]
            if citation_keys
            else list(latest_by_key.values())
        )
        for paper in selected_papers:
            if len(processed) >= max_papers:
                skipped.append(
                    {"citation_key": paper.citation_key, "reason": "max_papers limit reached"}
                )
                continue
            if only_missing and paper.citation_key in already_extracted:
                skipped.append({"citation_key": paper.citation_key, "reason": "already extracted"})
                continue
            try:
                current = paper
                if not current.text_path:
                    if not current.pdf_path:
                        skipped.append(
                            {"citation_key": current.citation_key, "reason": "no text_path or pdf_path"}
                        )
                        continue
                    self.extract_pdf_text(citation_key=current.citation_key)
                    current = self.require_paper(citation_key=current.citation_key)
                if not current.text_path or not self.store.exists(current.text_path):
                    skipped.append(
                        {"citation_key": current.citation_key, "reason": "text artifact missing"}
                    )
                    continue
                extract = self.extract_paper(
                    citation_key=current.citation_key, use_llm=use_llm
                )
                statements = [
                    *extract.theorem_statements,
                    *extract.algorithm_statements,
                    *extract.lower_bound_statements,
                ]
                statement_support_ids = [
                    statement.support_id for statement in statements if statement.support_id
                ]
                support_ids.extend(statement_support_ids)
                processed_citation_keys.append(extract.citation_key)
                processed.append(
                    {
                        "citation_key": extract.citation_key,
                        "paper_id": extract.paper_id,
                        "extract_id": extract.extract_id,
                        "support_ids": statement_support_ids,
                        "theorem_count": len(extract.theorem_statements),
                        "algorithm_count": len(extract.algorithm_statements),
                        "lower_bound_count": len(extract.lower_bound_statements),
                    }
                )
                already_extracted.add(extract.citation_key)
            except Exception as exc:  # noqa: BLE001 - batch extraction should continue
                errors.append(
                    {
                        "citation_key": paper.citation_key,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

        return {
            "tool_result_id": batch_id,
            "batch_id": batch_id,
            "processed_count": len(processed),
            "skipped_count": len(skipped),
            "error_count": len(errors),
            "citation_keys": list(dict.fromkeys(processed_citation_keys)),
            "support_ids": list(dict.fromkeys(support_ids)),
            "processed": processed,
            "skipped": skipped[:20],
            "errors": errors[:20],
        }

    def _citation_keys_with_statement_extractions(self) -> set[str]:
        keys: set[str] = set()
        for record in self.store.read_jsonl("LiteratureDB/statements.jsonl"):
            key = str(record.get("citation_key") or "")
            if key:
                keys.add(key)
        return keys

    def extract_from_text(
        self,
        *,
        citation_key: str,
        paper_text: str,
        paper_id: str = "",
        text_artifact_path: str | None = None,
        use_llm: bool = True,
    ) -> LiteratureExtract:
        """Extract statements/claims and commit statement-level quote provenance.

        A local heuristic extractor is used as deterministic dry-run output and as a safety net for
        theorem/algorithm statement presence. Literature claims are downgraded unless at least one
        theorem/algorithm statement was extracted for the same paper.
        """
        text_ref = self.store.artifact_ref(text_artifact_path) if text_artifact_path else None
        heuristic = self._heuristic_extract(
            citation_key=citation_key,
            paper_text=paper_text,
            paper_id=paper_id,
            text_ref=text_ref,
        )
        if use_llm:
            messages = [
                {
                    "role": "system",
                    "content": render_prompt("literature_researcher", override_dir=self.prompt_dir),
                },
                {
                    "role": "user",
                    "content": (
                        f"Citation key: {citation_key}\n"
                        f"Paper ID: {paper_id}\n"
                        "Paper text/excerpt (extract exact quote provenance with offsets/locators "
                        "when possible):\n"
                        f"{paper_text[:22000]}"
                    ),
                },
            ]
            try:
                extract = self.router.complete_structured(
                    task_type="literature_extraction",
                    messages=messages,
                    schema=LiteratureExtract,
                    mock_output=heuristic if self.router.dry_run else None,
                )
            except Exception:
                # Literature obligations depend on extraction being robust.  If the LLM backend is
                # unavailable or over context, keep the deterministic statement scan rather than
                # failing the whole research run.
                extract = heuristic
        else:
            extract = heuristic
        extract, support_ids_by_statement = self._postprocess_extract(
            extract,
            heuristic=heuristic,
            citation_key=citation_key,
            paper_id=paper_id,
            text_ref=text_ref,
        )
        self._commit_extract(extract, support_ids_by_statement=support_ids_by_statement)
        return extract

    def answer_query(self, query: str, *, limit: int = 10) -> LiteratureQueryAnswer:
        """Answer a local query. The calling run owns persistence of the bounded answer."""
        return self.retriever.answer_query(query, limit=limit)

    # ------------------------------------------------------------------
    # OpenAlex discovery / candidate queue
    # ------------------------------------------------------------------
    def search_papers(self, query: str, *, limit: int = 10) -> list[LiteratureCandidate]:
        """Search OpenAlex and queue candidate papers for optional import."""
        candidates = self.openalex.search(query, limit=limit)
        return self._queue_candidates(candidates)

    def discover_related(
        self,
        *,
        citation_key: str,
        direction: str = "both",
        limit: int = 20,
    ) -> list[LiteratureCandidate]:
        """Queue papers referenced by or citing one imported paper.

        This is deliberately one-hop and non-recursive. It expands the candidate horizon but does
        not import papers or accept any literature claims.
        """
        if direction not in {"cited", "cited_by", "both"}:
            raise ValueError("direction must be one of: cited, cited_by, both")
        paper = self.require_paper(citation_key=citation_key)
        lookup = _openalex_lookup_value(paper)
        candidates: list[LiteratureCandidate] = []
        if direction in {"cited", "both"}:
            cited_limit = limit if direction == "cited" else max(1, limit // 2)
            candidates.extend(self.openalex.references(lookup, limit=cited_limit))
        if direction in {"cited_by", "both"}:
            cited_by_limit = limit if direction == "cited_by" else max(1, limit - len(candidates))
            candidates.extend(self.openalex.citations(lookup, limit=cited_by_limit))
        return self._queue_candidates(candidates[:limit])

    def import_candidate(self, candidate_id: str, *, extract_text: bool = False) -> PaperMetadata:
        """Import one queued candidate through the existing paper-ingestion pipeline."""
        candidate = self.require_candidate(candidate_id)
        if candidate.status == "duplicate":
            raise ValueError(
                f"Candidate {candidate_id} is marked duplicate; import explicitly by URL/DOI"
            )
        key = _candidate_key(candidate)
        imported_keys = {k for paper in self.list_papers() for k in _paper_candidate_keys(paper)}
        if key and key in imported_keys:
            candidate.status = "duplicate"
            self.store.append_jsonl("LiteratureDB/candidates.jsonl", candidate)
            raise ValueError(f"Candidate {candidate_id} already appears to be imported")
        expected_title = candidate.title or None
        expected_authors = candidate.authors or None
        # Discovery and fetched records commonly disagree by one year between preprint,
        # online-first, and proceedings publication. Title/author identity is the safer automatic
        # gate; exact year remains available for explicit manual imports.
        expected_year = None
        # Venue metadata often differs between preprint and proceedings records; use it only
        # when explicitly supplied through import_* tools, not for candidate auto-import.
        expected_venue = None
        if candidate.arxiv_id:
            # Discovery already supplied arXiv metadata. Re-querying the Atom API here both wastes a
            # request and can fail under an API throttle after the candidate was successfully found.
            provisional = self.fetcher.import_discovered_arxiv(
                candidate.arxiv_id,
                title=candidate.title,
                authors=candidate.authors,
                year=candidate.year,
                abstract=candidate.abstract,
                pdf_url=candidate.pdf_url,
                landing_url=candidate.landing_url,
                commit=False,
            )
            self._validate_expected_metadata(
                provisional,
                expected_title=expected_title,
                expected_authors=expected_authors,
                expected_year=expected_year,
                expected_venue=expected_venue,
            )
            paper = self.import_paper(provisional)
            paper = self._extract_text_if_requested(paper, extract_text=extract_text)
        elif candidate.pdf_url:
            try:
                paper = self.import_url(
                    candidate.pdf_url,
                    title=candidate.title,
                    doi=candidate.doi or None,
                    extract_text=extract_text,
                    expected_title=expected_title,
                    expected_authors=expected_authors,
                    expected_year=expected_year,
                    expected_venue=expected_venue,
                )
            except PaperMetadataMismatchError:
                raise
            except Exception:
                if not candidate.doi:
                    raise
                paper = self.import_url(
                    candidate.pdf_url,
                    title=candidate.title,
                    extract_text=extract_text,
                    expected_title=expected_title,
                    expected_authors=expected_authors,
                    expected_year=expected_year,
                    expected_venue=expected_venue,
                )
        elif candidate.doi:
            paper = self.import_doi(
                candidate.doi,
                extract_text=extract_text,
                expected_title=expected_title,
                expected_authors=expected_authors,
                expected_year=expected_year,
                expected_venue=expected_venue,
            )
        elif candidate.landing_url:
            paper = self.import_url(
                candidate.landing_url,
                title=candidate.title,
                extract_text=extract_text,
                expected_title=expected_title,
                expected_authors=expected_authors,
                expected_year=expected_year,
                expected_venue=expected_venue,
            )
        else:
            raise ValueError(f"Candidate {candidate_id} has no importable arXiv/DOI/URL")

        paper.title = paper.title or candidate.title
        paper.authors = paper.authors or candidate.authors
        paper.year = paper.year or candidate.year
        paper.venue = paper.venue or candidate.venue
        paper.abstract = paper.abstract or candidate.abstract
        candidate_urls = [
            candidate.openalex_id,
            candidate.landing_url,
            candidate.pdf_url,
            *candidate.source_urls,
        ]
        for url in candidate_urls:
            if url and url not in paper.source_urls:
                paper.source_urls.append(url)
        self._update_paper_record(paper)

        candidate.status = "imported"
        candidate.imported_paper_id = paper.paper_id
        self.store.append_jsonl("LiteratureDB/candidates.jsonl", candidate)
        return paper

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def _validate_expected_metadata(
        self,
        paper: PaperMetadata,
        *,
        expected_title: str | None = None,
        expected_authors: list[str] | None = None,
        expected_year: int | None = None,
        expected_venue: str | None = None,
    ) -> None:
        """Reject imports whose fetched metadata clearly points at a different paper."""
        issues = _metadata_mismatch_issues(
            paper,
            expected_title=expected_title,
            expected_authors=expected_authors,
            expected_year=expected_year,
            expected_venue=expected_venue,
        )
        if issues:
            raise PaperMetadataMismatchError(
                "Imported metadata does not match expected paper: "
                + "; ".join(issues)
                + f". fetched_title={paper.title!r}, citation_key={paper.citation_key!r}"
            )

    def list_papers(self) -> list[PaperMetadata]:
        papers = []
        for record in self.store.read_jsonl("LiteratureDB/papers.jsonl"):
            try:
                papers.append(PaperMetadata.model_validate(record))
            except Exception:
                continue
        return papers

    def require_paper(
        self, *, citation_key: str | None = None, paper_id: str | None = None
    ) -> PaperMetadata:
        for paper in reversed(self.list_papers()):
            if citation_key and paper.citation_key == citation_key:
                return paper
            if paper_id and paper.paper_id == paper_id:
                return paper
        ident = citation_key or paper_id or "<missing>"
        raise KeyError(f"No imported paper found for {ident}")

    def list_candidates(self) -> list[LiteratureCandidate]:
        candidates = []
        for record in self.store.read_jsonl("LiteratureDB/candidates.jsonl"):
            try:
                candidates.append(LiteratureCandidate.model_validate(record))
            except Exception:
                continue
        return candidates

    def require_candidate(self, candidate_id: str) -> LiteratureCandidate:
        for candidate in reversed(self.list_candidates()):
            if candidate.candidate_id == candidate_id:
                return candidate
        raise KeyError(f"No literature candidate found for {candidate_id}")

    def _queue_candidates(
        self, candidates: list[LiteratureCandidate]
    ) -> list[LiteratureCandidate]:
        queued: list[LiteratureCandidate] = []
        existing_candidates = self.list_candidates()
        imported_papers = self.list_papers()
        seen_keys = {
            _candidate_key(candidate)
            for candidate in existing_candidates
            if _candidate_key(candidate)
        }
        seen_keys.update(
            key for paper in imported_papers for key in _paper_candidate_keys(paper) if key
        )
        for candidate in candidates:
            key = _candidate_key(candidate)
            if key and key in seen_keys:
                candidate.status = "duplicate"
            elif key:
                seen_keys.add(key)
            self.store.append_jsonl("LiteratureDB/candidates.jsonl", candidate)
            queued.append(candidate)
        return queued

    def _update_paper_record(self, paper: PaperMetadata) -> None:
        if paper.metadata_path:
            metadata_ref = self.store.write_json(paper.metadata_path, paper)
        else:
            metadata_ref = self.store.write_json(
                f"LiteratureDB/papers/{_safe_slug(paper.citation_key)}/metadata.json", paper
            )
            paper.metadata_path = metadata_ref.path
            self.store.write_json(paper.metadata_path, paper)
        if metadata_ref.path not in {ref.path for ref in paper.artifact_refs}:
            paper.artifact_refs.append(metadata_ref)
        self.store.append_jsonl("LiteratureDB/papers.jsonl", paper)
        self.index.upsert_paper(paper)
        if paper.text_path and self.store.exists(paper.text_path):
            self.index.index_paper_text(paper)

    # ------------------------------------------------------------------
    # Extraction internals
    # ------------------------------------------------------------------
    def _postprocess_extract(
        self,
        extract: LiteratureExtract,
        *,
        heuristic: LiteratureExtract,
        citation_key: str,
        paper_id: str,
        text_ref: ArtifactRef | None,
    ) -> tuple[LiteratureExtract, dict[str, str]]:
        extract.citation_key = citation_key
        extract.paper_id = extract.paper_id or paper_id
        extract.text_artifact_ref = extract.text_artifact_ref or text_ref

        # If optional model extraction misses explicit statements, retain the deterministic
        # exact-quote scanner output. The worker normally uses the deterministic path directly.
        if not extract.theorem_statements and heuristic.theorem_statements:
            extract.theorem_statements = heuristic.theorem_statements
        if not extract.algorithm_statements and heuristic.algorithm_statements:
            extract.algorithm_statements = heuristic.algorithm_statements
        if not extract.lower_bound_statements and heuristic.lower_bound_statements:
            extract.lower_bound_statements = heuristic.lower_bound_statements

        for statements in [
            extract.theorem_statements,
            extract.algorithm_statements,
            extract.lower_bound_statements,
        ]:
            for statement in statements:
                self._normalize_statement(statement, citation_key, extract.paper_id, text_ref)

        support_ids_by_statement = self.index.index_extract(extract)
        return extract, support_ids_by_statement

    def _commit_extract(
        self, extract: LiteratureExtract, *, support_ids_by_statement: dict[str, str]
    ) -> None:
        # Canonical literature memory has one statement ledger. The richer extraction object is
        # returned to the bounded caller but is not duplicated into a second claims ledger.
        self._append_flat_statement_rows(extract, support_ids_by_statement=support_ids_by_statement)

    def _append_flat_statement_rows(
        self, extract: LiteratureExtract, *, support_ids_by_statement: dict[str, str]
    ) -> None:
        """Atomically replace one paper's canonical statement snapshot."""
        path = "LiteratureDB/statements.jsonl"
        retained = [
            row
            for row in self.store.read_jsonl(path)
            if str(row.get("paper_id") or "") != extract.paper_id
            and str(row.get("citation_key") or "") != extract.citation_key
        ]
        rows: list[dict[str, Any]] = []
        for statement in [
            *extract.theorem_statements,
            *extract.algorithm_statements,
            *extract.lower_bound_statements,
        ]:
            quote = statement.provenance[0] if statement.provenance else None
            support_id = statement.support_id or support_ids_by_statement.get(
                statement.statement_id, ""
            )
            rows.append(
                {
                    "statement_id": statement.statement_id,
                    "support_id": support_id,
                    "citation_key": statement.citation_key or extract.citation_key,
                    "paper_id": statement.paper_id or extract.paper_id,
                    "kind": statement.kind,
                    "label": statement.label,
                    "original_statement": statement.original_statement,
                    "statement_text": statement.statement_text or statement.original_statement,
                    "quote_id": quote.quote_id if quote else "",
                    "quote": quote.quote if quote else statement.original_statement,
                    "locator": quote.locator if quote else statement.label,
                    "char_start": quote.char_start if quote else None,
                    "char_end": quote.char_end if quote else None,
                    "source_sha256": quote.source_sha256 if quote else "",
                    "validated_exact_substring": bool(quote.validated) if quote else False,
                    "confidence": statement.confidence,
                }
            )
        content = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in [*retained, *rows]
        )
        self.store.write_text(path, content)

    def _normalize_statement(
        self,
        statement: LiteratureStatement,
        citation_key: str,
        paper_id: str,
        text_ref: ArtifactRef | None,
    ) -> None:
        statement.citation_key = statement.citation_key or citation_key
        statement.paper_id = statement.paper_id or paper_id
        statement.statement_text = statement.statement_text or statement.original_statement
        if not statement.provenance:
            statement.provenance = [
                LiteratureQuote(
                    citation_key=citation_key,
                    paper_id=paper_id,
                    locator=statement.label,
                    quote=statement.original_statement,
                    source_sha256=text_ref.sha256 if text_ref and text_ref.sha256 else "",
                    artifact_refs=[text_ref] if text_ref else [],
                )
            ]
        else:
            for quote in statement.provenance:
                quote.citation_key = quote.citation_key or citation_key
                quote.paper_id = quote.paper_id or paper_id
                if text_ref and text_ref.sha256 and not quote.source_sha256:
                    quote.source_sha256 = text_ref.sha256
                if text_ref and text_ref.path not in {ref.path for ref in quote.artifact_refs}:
                    quote.artifact_refs.append(text_ref)
        # Stable content-derived handles make deterministic re-extraction idempotent and keep
        # evidence references valid across index rebuilds.
        primary_quote = statement.provenance[0]
        primary_quote.quote_id = _stable_literature_id(
            "quote",
            citation_key,
            str(primary_quote.char_start),
            str(primary_quote.char_end),
            primary_quote.quote,
        )
        statement.statement_id = _stable_literature_id(
            "lit_stmt",
            citation_key,
            statement.kind,
            statement.label,
            statement.statement_text,
        )

    def _find_existing_paper_for_metadata(self, paper: PaperMetadata) -> PaperMetadata | None:
        existing = None
        if paper.doi:
            existing = self.index.find_paper(doi=paper.doi)
        if existing is None and paper.arxiv_id:
            existing = self.index.find_paper(arxiv_id=paper.arxiv_id)
        if existing is None and paper.citation_key:
            existing = self.index.find_paper(citation_key=paper.citation_key)
        if existing is None:
            existing = self.index.find_paper(title=paper.title)
        return existing

    def _heuristic_extract(
        self,
        *,
        citation_key: str,
        paper_text: str,
        paper_id: str,
        text_ref: ArtifactRef | None,
    ) -> LiteratureExtract:
        theorem_statements: list[LiteratureStatement] = []
        algorithm_statements: list[LiteratureStatement] = []
        lower_bound_statements: list[LiteratureStatement] = []
        scan_text = _strip_reference_sections(paper_text)
        for match in _iter_statement_matches(scan_text):
            kind = _statement_kind(match.label)
            if kind != "algorithm" and _LOWER_BOUND_RE.search(match.text):
                kind = "lower_bound"
            quote = LiteratureQuote(
                citation_key=citation_key,
                paper_id=paper_id,
                locator=match.locator,
                quote=match.text,
                char_start=match.start,
                char_end=match.end,
                artifact_refs=[text_ref] if text_ref else [],
            )
            statement = LiteratureStatement(
                citation_key=citation_key,
                paper_id=paper_id,
                kind=kind,
                label=match.label,
                original_statement=match.text,
                statement_text=match.text,
                provenance=[quote],
                confidence=0.55,
            )
            if kind == "algorithm":
                algorithm_statements.append(statement)
            elif kind == "lower_bound":
                lower_bound_statements.append(statement)
            else:
                theorem_statements.append(statement)
        extract = LiteratureExtract(
            citation_key=citation_key,
            paper_id=paper_id,
            text_artifact_ref=text_ref,
            theorem_statements=theorem_statements,
            algorithm_statements=algorithm_statements,
            lower_bound_statements=lower_bound_statements,
            provenance_notes=(
                "Deterministic regex extraction; review before relying on fine details."
            ),
        )
        # Support IDs are assigned during postprocessing when the quote spans are indexed.
        return extract


class _StatementMatch:
    def __init__(self, label: str, text: str, start: int, end: int, locator: str):
        self.label = label
        self.text = text
        self.start = start
        self.end = end
        self.locator = locator


_STATEMENT_START_RE = re.compile(
    r"(?im)^\s*(?:(?:I|▶|▸|□)\s+)?"
    r"(?P<label>(?:(?:Theorem|Lemma|Corollary|Proposition|Algorithm|Definition)"
    r"\s*(?:[0-9A-Za-z][0-9A-Za-z().-]*)?)|"
    r"(?:(?:Hypothesis|Assumption|Conjecture)\s*[A-Za-z0-9().-]*))"
    r"\s*(?:\([^\n]{0,80}\))?\s*(?:[:.]|—|-)"
)
_LOWER_BOUND_RE = re.compile(
    r"(?i)(?:\b(?:lower bound|impossibility theorem|no-go theorem)\b|"
    r"\bno\b.{0,100}\b(?:subquadratic|subexponential|polynomial[- ]time)?\s*algorithm\b)"
)
_REFERENCE_HEADING_RE = re.compile(
    r"(?im)^\s*(references|bibliography|works cited|literature cited)\s*$"
)


def _strip_reference_sections(text: str) -> str:
    """Drop bibliography/reference tail before deterministic statement extraction."""
    match = _REFERENCE_HEADING_RE.search(text)
    if match:
        return text[: match.start()]
    return text


def _iter_statement_matches(text: str) -> list[_StatementMatch]:
    matches = list(_STATEMENT_START_RE.finditer(text))
    results: list[_StatementMatch] = []
    for idx, match in enumerate(matches):
        start = match.start("label")
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        proof_pos = _find_after(text, start, next_start, ["\nProof", "\n proof", "\nPROOF"])
        # PDF layout often puts a blank line immediately after a theorem heading and before its
        # body. Starting the paragraph search at match.end() then extracts only the heading.
        content_start = match.end()
        while content_start < next_start and text[content_start].isspace():
            content_start += 1
        paragraph_pos = text.find("\n\n", content_start, next_start)
        end_candidates = [pos for pos in [proof_pos, paragraph_pos, next_start] if pos != -1]
        end = min(end_candidates) if end_candidates else next_start
        # Avoid swallowing whole papers when PDFs have poor paragraphing.
        end = min(end, start + 3000)
        label = re.sub(r"\s+", " ", match.group("label")).strip()
        if _statement_kind(label) == "claim":
            end = _named_claim_sentence_end(text, start, end)
        statement_text = re.sub(r"\s+", " ", text[start:end]).strip()
        if len(statement_text) < 20:
            continue
        results.append(
            _StatementMatch(
                label=label,
                text=statement_text,
                start=start,
                end=end,
                locator=_locator_for(text, start),
            )
        )
    return results


def _named_claim_sentence_end(text: str, start: int, end: int) -> int:
    """Keep a named conjecture/hypothesis statement, not its following discussion."""
    segment = text[start:end]
    boundaries = list(re.finditer(r"\.\s+(?=[A-Z])", segment))
    if not boundaries:
        return end
    first = boundaries[0]
    prefix = segment[: first.end()]
    # A parenthesized display name commonly ends the label sentence; the mathematical
    # statement is the next sentence.
    chosen = boundaries[1] if "(" in prefix and ")" in prefix and len(boundaries) > 1 else first
    return start + chosen.start() + 1


def _find_after(text: str, start: int, end: int, needles: list[str]) -> int:
    positions = [text.find(needle, start, end) for needle in needles]
    positions = [pos for pos in positions if pos != -1]
    return min(positions) if positions else -1


def _locator_for(text: str, char_pos: int) -> str:
    page = ""
    for match in re.finditer(r"--- page (\d+) ---", text, flags=re.IGNORECASE):
        if match.start() <= char_pos:
            page = match.group(1)
        else:
            break
    return f"page {page}" if page else f"char {char_pos}"


def _statement_kind(label: str) -> str:
    lowered = label.lower()
    if lowered.startswith("algorithm"):
        return "algorithm"
    if _LOWER_BOUND_RE.search(label):
        return "lower_bound"
    if lowered.startswith("definition"):
        return "definition"
    if "hypothesis" in lowered or "conjecture" in lowered or lowered.startswith("assumption"):
        return "claim"
    if lowered.startswith("lemma"):
        return "lemma"
    if lowered.startswith("corollary"):
        return "corollary"
    if lowered.startswith("proposition"):
        return "proposition"
    return "theorem"


def _merge_duplicate_paper_metadata(
    existing: PaperMetadata, incoming: PaperMetadata
) -> tuple[PaperMetadata, bool]:
    """Merge useful artifacts/metadata from a duplicate import into the canonical paper."""
    merged = existing.model_copy(deep=True)
    changed = False
    for field in [
        "title",
        "url",
        "arxiv_id",
        "doi",
        "abstract",
        "venue",
        "pdf_path",
        "text_path",
        "metadata_path",
    ]:
        if not getattr(merged, field) and getattr(incoming, field):
            setattr(merged, field, getattr(incoming, field))
            changed = True
    if merged.year is None and incoming.year is not None:
        merged.year = incoming.year
        changed = True
    for author in incoming.authors:
        if author and author not in merged.authors:
            merged.authors.append(author)
            changed = True
    for url in incoming.source_urls:
        if url and url not in merged.source_urls:
            merged.source_urls.append(url)
            changed = True
    for ref in incoming.artifact_refs:
        if ref.path not in {existing_ref.path for existing_ref in merged.artifact_refs}:
            merged.artifact_refs.append(ref)
            changed = True
    return merged, changed


def _metadata_mismatch_issues(
    paper: PaperMetadata,
    *,
    expected_title: str | None = None,
    expected_authors: list[str] | None = None,
    expected_year: int | None = None,
    expected_venue: str | None = None,
) -> list[str]:
    issues: list[str] = []
    if expected_title and paper.title and not _titles_match(expected_title, paper.title):
        issues.append(f"title mismatch: expected {expected_title!r}")
    if expected_year is not None and paper.year is not None and int(expected_year) != int(paper.year):
        issues.append(f"year mismatch: expected {expected_year}, got {paper.year}")
    if expected_venue and paper.venue and not _venue_matches(expected_venue, paper.venue):
        issues.append(f"venue mismatch: expected {expected_venue!r}, got {paper.venue!r}")
    expected_authors = [author for author in (expected_authors or []) if author]
    if expected_authors and paper.authors and not _authors_overlap(expected_authors, paper.authors):
        issues.append(
            "author mismatch: expected one of "
            + repr(expected_authors[:4])
            + ", got "
            + repr(paper.authors[:4])
        )
    return issues


def _titles_match(expected: str, actual: str) -> bool:
    expected_norm = _normalize_title(expected)
    actual_norm = _normalize_title(actual)
    if not expected_norm or not actual_norm:
        return True
    if expected_norm in actual_norm or actual_norm in expected_norm:
        return True
    expected_terms = set(expected_norm.split())
    actual_terms = set(actual_norm.split())
    if not expected_terms or not actual_terms:
        return True
    return len(expected_terms & actual_terms) / len(expected_terms | actual_terms) >= 0.65


def _venue_matches(expected: str, actual: str) -> bool:
    expected_norm = _normalize_title(expected)
    actual_norm = _normalize_title(actual)
    return bool(
        expected_norm
        and actual_norm
        and (expected_norm in actual_norm or actual_norm in expected_norm)
    )


def _authors_overlap(expected: list[str], actual: list[str]) -> bool:
    """Match author names independent of `family, given` ordering and initials."""
    expected_names = [_author_tokens(author) for author in expected]
    actual_names = [_author_tokens(author) for author in actual]
    expected_names = [tokens for tokens in expected_names if tokens]
    actual_names = [tokens for tokens in actual_names if tokens]
    if not expected_names or not actual_names:
        return True
    matches = 0
    for wanted in expected_names:
        if any(
            len(wanted & found) >= 1
            and len(wanted & found) / max(1, min(len(wanted), len(found))) >= 0.75
            for found in actual_names
        ):
            matches += 1
    return matches >= max(1, min(len(expected_names), 2))


def _author_tokens(author: str) -> set[str]:
    folded = unicodedata.normalize("NFKD", author).encode("ascii", "ignore").decode("ascii")
    return {
        token
        for token in re.findall(r"[a-z]+", folded.lower())
        if len(token) > 1
    }


def _openalex_lookup_value(paper: PaperMetadata) -> str:
    for url in paper.source_urls:
        if "openalex.org/W" in url:
            return url
    if paper.doi:
        return paper.doi
    if paper.arxiv_id:
        return f"arXiv:{paper.arxiv_id}"
    if paper.title:
        return paper.title
    raise ValueError(f"Paper {paper.citation_key} has no DOI/arXiv/title for OpenAlex lookup")


def _candidate_key(candidate: LiteratureCandidate) -> str:
    if candidate.openalex_id:
        return "openalex:" + candidate.openalex_id.rstrip("/").rsplit("/", 1)[-1].lower()
    if candidate.doi:
        return "doi:" + candidate.doi.lower()
    if candidate.arxiv_id:
        return "arxiv:" + candidate.arxiv_id.lower()
    title_key = _normalize_title(candidate.title)
    return "title:" + title_key if title_key else ""


def _paper_candidate_keys(paper: PaperMetadata) -> list[str]:
    keys = []
    if paper.doi:
        keys.append("doi:" + paper.doi.lower())
    if paper.arxiv_id:
        keys.append("arxiv:" + paper.arxiv_id.lower())
    for url in paper.source_urls:
        if "openalex.org/W" in url:
            keys.append("openalex:" + url.rstrip("/").rsplit("/", 1)[-1].lower())
    title_key = _normalize_title(paper.title)
    if title_key:
        keys.append("title:" + title_key)
    return keys


def _normalize_title(title: str) -> str:
    folded = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", folded.lower()).strip()


def _safe_slug(value: str, *, default: str = "paper") -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return (slug or default)[:120]


def _stable_literature_id(prefix: str, *parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8", errors="replace")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:12]}"
