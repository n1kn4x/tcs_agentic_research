"""Modular literature agent with fetching, extraction, notation mapping, and retrieval.

The agent is deliberately usable outside the LangGraph loop: network/PDF/retrieval pieces
live in ``tcs_agentic_research.literature`` services, while this class coordinates durable
artifacts and optional LLM extraction. Other agents should call ``answer_query`` rather than
reading raw literature records so they always receive mapped nomenclature and quote provenance.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..artifact_store import ArtifactStore
from ..literature.fetchers import (
    LiteratureFetcher,
    normalize_arxiv_id,
    normalize_doi,
    parse_arxiv_id,
    parse_doi,
)
from ..literature.index import LiteratureIndex
from ..literature.nomenclature import NomenclatureMapper
from ..literature.openalex import OpenAlexClient
from ..literature.pdf_text import PDFTextExtractor
from ..literature.retrieval import LiteratureRetriever
from ..llm import LLMRouter
from ..prompt_loader import render_prompt
from ..schemas import (
    ArtifactRef,
    ClaimRecord,
    ClaimStatus,
    ClaimType,
    EvidenceRecord,
    EvidenceType,
    LiteratureCandidate,
    LiteratureExtract,
    LiteratureQueryAnswer,
    LiteratureQuote,
    LiteratureSource,
    LiteratureStatement,
    PaperMetadata,
)


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
        self.mapper = NomenclatureMapper(store)
        needs_index_rebuild = not store.exists(LiteratureIndex.INDEX_PATH)
        self.index = LiteratureIndex(store)
        has_literature_artifacts = bool(
            store.read_jsonl("LiteratureDB/papers.jsonl")
            or store.read_jsonl("LiteratureDB/extracted_claims.jsonl")
        )
        if (needs_index_rebuild or self.index.is_empty()) and has_literature_artifacts:
            self.index.rebuild()
        self.retriever = LiteratureRetriever(store, self.mapper, index=self.index)
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
                str(arxiv_id), citation_key=citation_key, extract_text=source.extract_text
            )
        doi = parse_doi(token) or (token if source.source_type == "doi" else None)
        if doi:
            return self.import_doi(str(doi), citation_key=citation_key, extract_text=source.extract_text)
        return self.import_url(
            token,
            citation_key=citation_key,
            title=source.title or None,
            extract_text=source.extract_text,
        )

    def import_paper(self, paper: PaperMetadata) -> PaperMetadata:
        """Register already-known paper metadata in ``LiteratureDB``.

        The append-only JSONL ledger remains an audit trail; the SQLite literature index records
        canonical paper aliases so duplicate citation keys/DOIs/arXiv IDs resolve to one paper.
        """
        existing = self._find_existing_paper_for_metadata(paper)
        if existing is not None and existing.paper_id != paper.paper_id:
            self.index.add_alias(existing.paper_id, "citation_key", paper.citation_key)
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
    ) -> PaperMetadata:
        """Fetch a URL/DOI/arXiv/PDF source into ``LiteratureDB/papers/``."""
        arxiv_id = parse_arxiv_id(url)
        doi_value = doi or parse_doi(url)
        existing = None
        if arxiv_id:
            existing = self.index.find_paper(arxiv_id=arxiv_id)
        elif doi_value:
            existing = self.index.find_paper(doi=doi_value)
        if existing is not None:
            if citation_key:
                self.index.add_alias(existing.paper_id, "citation_key", citation_key)
            return self._extract_text_if_requested(existing, extract_text=extract_text)
        paper = self.fetcher.import_url(url, citation_key=citation_key, title=title, doi=doi)
        self.index.upsert_paper(paper)
        return self._extract_text_if_requested(paper, extract_text=extract_text)

    def import_arxiv(
        self, arxiv_id: str, *, citation_key: str | None = None, extract_text: bool = False
    ) -> PaperMetadata:
        clean_id = normalize_arxiv_id(arxiv_id)
        existing = self.index.find_paper(arxiv_id=clean_id)
        if existing is not None:
            if citation_key:
                self.index.add_alias(existing.paper_id, "citation_key", citation_key)
            return self._extract_text_if_requested(existing, extract_text=extract_text)
        paper = self.fetcher.import_arxiv(clean_id, citation_key=citation_key)
        self.index.upsert_paper(paper)
        return self._extract_text_if_requested(paper, extract_text=extract_text)

    def import_doi(
        self, doi: str, *, citation_key: str | None = None, extract_text: bool = False
    ) -> PaperMetadata:
        clean_doi = normalize_doi(doi)
        existing = self.index.find_paper(doi=clean_doi)
        if existing is not None:
            if citation_key:
                self.index.add_alias(existing.paper_id, "citation_key", citation_key)
            return self._extract_text_if_requested(existing, extract_text=extract_text)
        paper = self.fetcher.import_doi(clean_doi, citation_key=citation_key)
        self.index.upsert_paper(paper)
        return self._extract_text_if_requested(paper, extract_text=extract_text)

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
    # Extraction and nomenclature mapping
    # ------------------------------------------------------------------
    def extract_paper(
        self, *, citation_key: str | None = None, paper_id: str | None = None
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
        )

    def extract_from_text(
        self,
        *,
        citation_key: str,
        paper_text: str,
        nomenclature_yaml: str | None = None,
        paper_id: str = "",
        text_artifact_path: str | None = None,
    ) -> LiteratureExtract:
        """Extract statements/claims and commit only nomenclature-mapped records.

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
                    f"Nomenclature (canonical notation; map all outputs to it):\n"
                    f"{nomenclature_yaml or self.store.read_text(ArtifactStore.NOMENCLATURE)}\n\n"
                    "Paper text/excerpt (extract exact quote provenance with offsets/locators "
                    "when possible):\n"
                    f"{paper_text[:70000]}"
                ),
            },
        ]
        extract = self.router.complete_structured(
            task_type="literature_extraction",
            messages=messages,
            schema=LiteratureExtract,
            mock_output=heuristic if self.router.dry_run else None,
        )
        extract = self._postprocess_extract(
            extract,
            heuristic=heuristic,
            citation_key=citation_key,
            paper_id=paper_id,
            text_ref=text_ref,
        )
        self._commit_extract(extract)
        return extract

    def answer_query(self, query: str, *, limit: int = 10) -> LiteratureQueryAnswer:
        """Answer a literature query with mapped notation, quote provenance, and duplicates."""
        answer = self.retriever.answer_query(query, limit=limit)
        self.store.append_jsonl("LiteratureDB/query_answers.jsonl", answer)
        return answer

    def query_local(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Compatibility wrapper returning mapped query results as plain dicts."""
        answer = self.answer_query(query, limit=limit)
        return [result.model_dump(mode="json") for result in answer.results]

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
        if candidate.arxiv_id:
            paper = self.import_arxiv(candidate.arxiv_id, extract_text=extract_text)
        elif candidate.pdf_url:
            try:
                paper = self.import_url(
                    candidate.pdf_url,
                    title=candidate.title,
                    doi=candidate.doi or None,
                    extract_text=extract_text,
                )
            except Exception:
                if not candidate.doi:
                    raise
                paper = self.import_url(
                    candidate.pdf_url,
                    title=candidate.title,
                    extract_text=extract_text,
                )
        elif candidate.doi:
            paper = self.import_doi(candidate.doi, extract_text=extract_text)
        elif candidate.landing_url:
            paper = self.import_url(
                candidate.landing_url, title=candidate.title, extract_text=extract_text
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
    ) -> LiteratureExtract:
        extract.citation_key = citation_key
        extract.paper_id = extract.paper_id or paper_id
        extract.text_artifact_ref = extract.text_artifact_ref or text_ref

        # If the model missed explicit statements but the deterministic scanner found them, keep
        # the scanner output so claims cannot be accepted without statement-level provenance.
        if not extract.theorem_statements and heuristic.theorem_statements:
            extract.theorem_statements = heuristic.theorem_statements
        if not extract.algorithm_statements and heuristic.algorithm_statements:
            extract.algorithm_statements = heuristic.algorithm_statements
        if not extract.lower_bound_statements and heuristic.lower_bound_statements:
            extract.lower_bound_statements = heuristic.lower_bound_statements
        if not extract.extracted_claims and heuristic.extracted_claims:
            extract.extracted_claims = heuristic.extracted_claims

        all_mappings = dict(extract.notation_mappings)
        for statements in [
            extract.theorem_statements,
            extract.algorithm_statements,
            extract.lower_bound_statements,
        ]:
            for statement in statements:
                all_mappings.update(statement.notation_mappings)
        extract.notation_mappings = all_mappings

        source_refs = [text_ref] if text_ref else []
        changed_entries = self.mapper.update_from_mappings(
            all_mappings,
            source_refs=source_refs,
            new_entries=extract.new_nomenclature_entries,
        )
        for entry in changed_entries:
            self.store.append_jsonl(
                "LiteratureDB/notation_mappings.jsonl",
                {
                    "citation_key": citation_key,
                    "symbol": entry.symbol,
                    "canonical_name": entry.canonical_name,
                    "aliases": entry.aliases,
                    "source_refs": [ref.model_dump(mode="json") for ref in source_refs],
                },
            )

        all_statement_fields = [
            extract.theorem_statements,
            extract.algorithm_statements,
            extract.lower_bound_statements,
        ]
        for statements in all_statement_fields:
            for statement in statements:
                self._normalize_statement(
                    statement,
                    citation_key,
                    extract.paper_id,
                    extract.notation_mappings,
                    text_ref,
                )

        support_ids_by_statement = self.index.index_extract(extract)
        has_acceptance_statement = bool(support_ids_by_statement)
        if has_acceptance_statement and not extract.extracted_claims:
            extract.extracted_claims = self._claims_from_statements(
                extract, support_ids_by_statement=support_ids_by_statement
            )
        for claim in extract.extracted_claims:
            self._normalize_literature_claim(
                claim,
                citation_key=citation_key,
                has_acceptance_statement=has_acceptance_statement,
                text_ref=text_ref,
                notation_mappings=extract.notation_mappings,
                support_ids=self._matching_support_ids_for_claim(claim, extract, support_ids_by_statement),
            )
        return extract

    def _commit_extract(self, extract: LiteratureExtract) -> None:
        self.store.append_jsonl("LiteratureDB/extracted_claims.jsonl", extract)
        for symbol, canonical in extract.notation_mappings.items():
            self.store.append_jsonl(
                "LiteratureDB/notation_mappings.jsonl",
                {"citation_key": extract.citation_key, "symbol": symbol, "canonical": canonical},
            )
        # Do not auto-promote literature-derived statements into the global ClaimLedger.
        # LiteratureDB records are evidence objects; research/obligation code must explicitly
        # create claims that cite stable support IDs.

    def _normalize_statement(
        self,
        statement: LiteratureStatement,
        citation_key: str,
        paper_id: str,
        notation_mappings: dict[str, str],
        text_ref: ArtifactRef | None,
    ) -> None:
        statement.citation_key = statement.citation_key or citation_key
        statement.paper_id = statement.paper_id or paper_id
        merged_mappings = {**notation_mappings, **statement.notation_mappings}
        statement.notation_mappings = merged_mappings
        statement.mapped_statement = self.mapper.map_text(
            statement.mapped_statement or statement.original_statement,
            merged_mappings,
        )
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

    def _normalize_literature_claim(
        self,
        claim: ClaimRecord,
        *,
        citation_key: str,
        has_acceptance_statement: bool,
        text_ref: ArtifactRef | None,
        notation_mappings: dict[str, str],
        support_ids: list[str],
    ) -> None:
        claim.claim_type = ClaimType.literature
        claim.normalized_statement = self.mapper.map_text(
            claim.normalized_statement or claim.statement,
            notation_mappings,
        )
        if "literature" not in claim.tags:
            claim.tags.append("literature")
        if not any(ev.evidence_type == EvidenceType.citation for ev in claim.evidence):
            claim.evidence.append(
                EvidenceRecord(
                    evidence_type=EvidenceType.citation,
                    summary=(
                        f"Extracted from {citation_key} with quote-level literature provenance."
                    ),
                    artifact_refs=[text_ref] if text_ref else [],
                    citation_keys=[citation_key],
                    literature_support_ids=support_ids,
                    verifier="LiteratureResearcher",
                    confidence=0.7 if has_acceptance_statement and support_ids else 0.3,
                )
            )
        else:
            for ev in claim.evidence:
                if ev.evidence_type == EvidenceType.citation:
                    if citation_key not in ev.citation_keys:
                        ev.citation_keys.append(citation_key)
                    if text_ref and text_ref.path not in {ref.path for ref in ev.artifact_refs}:
                        ev.artifact_refs.append(text_ref)
                    for support_id in support_ids:
                        if support_id not in ev.literature_support_ids:
                            ev.literature_support_ids.append(support_id)
        if has_acceptance_statement and support_ids:
            claim.status = ClaimStatus.cited
        else:
            claim.status = ClaimStatus.needs_review
            if "unaccepted_no_extracted_theorem_or_algorithm" not in claim.tags:
                claim.tags.append("unaccepted_no_extracted_theorem_or_algorithm")

    def _claims_from_statements(
        self, extract: LiteratureExtract, *, support_ids_by_statement: dict[str, str]
    ) -> list[ClaimRecord]:
        claims: list[ClaimRecord] = []
        for statement in [
            *extract.theorem_statements,
            *extract.algorithm_statements,
            *extract.lower_bound_statements,
        ]:
            refs = statement.provenance[0].artifact_refs if statement.provenance else []
            support_id = support_ids_by_statement.get(statement.statement_id, "")
            claims.append(
                ClaimRecord(
                    claim_type=ClaimType.literature,
                    statement=f"{extract.citation_key} states: {statement.mapped_statement}",
                    normalized_statement=statement.mapped_statement,
                    status=ClaimStatus.cited if support_id else ClaimStatus.needs_review,
                    evidence=[
                        EvidenceRecord(
                            evidence_type=EvidenceType.citation,
                            summary=(
                                "Statement-level extraction from "
                                f"{extract.citation_key}: {statement.label}"
                            ),
                            artifact_refs=refs,
                            citation_keys=[extract.citation_key],
                            literature_support_ids=[support_id] if support_id else [],
                            verifier="LiteratureResearcher",
                            confidence=statement.confidence or 0.7,
                        )
                    ],
                    tags=["literature", statement.kind],
                )
            )
        return claims

    def _matching_support_ids_for_claim(
        self,
        claim: ClaimRecord,
        extract: LiteratureExtract,
        support_ids_by_statement: dict[str, str],
    ) -> list[str]:
        """Return support IDs whose extracted statement text matches a literature claim."""
        claim_text = _normalize_title(claim.normalized_statement or claim.statement)
        if not claim_text:
            return []
        matched: list[str] = []
        for statement in [
            *extract.theorem_statements,
            *extract.algorithm_statements,
            *extract.lower_bound_statements,
        ]:
            stmt_text = _normalize_title(statement.mapped_statement or statement.original_statement)
            if not stmt_text:
                continue
            support_id = support_ids_by_statement.get(statement.statement_id)
            if not support_id:
                continue
            if stmt_text in claim_text or claim_text in stmt_text:
                matched.append(support_id)
        return list(dict.fromkeys(matched))

    def _find_existing_paper_for_metadata(self, paper: PaperMetadata) -> PaperMetadata | None:
        existing = None
        if paper.doi:
            existing = self.index.find_paper(doi=paper.doi)
        if existing is None and paper.arxiv_id:
            existing = self.index.find_paper(arxiv_id=paper.arxiv_id)
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
        for match in _iter_statement_matches(paper_text):
            kind = _statement_kind(match.label)
            if kind != "algorithm" and _LOWER_BOUND_RE.search(match.text):
                kind = "lower_bound"
            mapped = self.mapper.map_text(match.text)
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
                mapped_statement=mapped,
                notation_mappings=self.mapper.used_mappings_for_text(match.text),
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
    r"(?im)^\s*(?P<label>(?:Theorem|Lemma|Corollary|Proposition|Algorithm)"
    r"\s*(?:[0-9A-Za-z][0-9A-Za-z().-]*)?)\s*(?:\([^\n]{0,80}\))?\s*(?:[:.]|—|-)"
)
_LOWER_BOUND_RE = re.compile(r"(?i)\b(lower bound|impossibility theorem|no-go theorem)\b")


def _iter_statement_matches(text: str) -> list[_StatementMatch]:
    matches = list(_STATEMENT_START_RE.finditer(text))
    results: list[_StatementMatch] = []
    for idx, match in enumerate(matches):
        start = match.start()
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        proof_pos = _find_after(text, start, next_start, ["\nProof", "\n proof", "\nPROOF"])
        paragraph_pos = text.find("\n\n", match.end(), next_start)
        end_candidates = [pos for pos in [proof_pos, paragraph_pos, next_start] if pos != -1]
        end = min(end_candidates) if end_candidates else next_start
        # Avoid swallowing whole papers when PDFs have poor paragraphing.
        end = min(end, start + 3000)
        statement_text = re.sub(r"\s+", " ", text[start:end]).strip()
        if len(statement_text) < 20:
            continue
        label = re.sub(r"\s+", " ", match.group("label")).strip()
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
    if lowered.startswith("lemma"):
        return "lemma"
    if lowered.startswith("corollary"):
        return "corollary"
    if lowered.startswith("proposition"):
        return "proposition"
    return "theorem"


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
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _safe_slug(value: str, *, fallback: str = "paper") -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return (slug or fallback)[:120]
