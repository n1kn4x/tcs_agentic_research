"""Small OpenAlex client for paper search and one-hop citation discovery."""

from __future__ import annotations

import re
import time
from typing import Any

import httpx

from ..schemas import LiteratureCandidate
from .fetchers import normalize_doi, parse_arxiv_id, parse_doi

OPENALEX_API = "https://api.openalex.org"
USER_AGENT = "agentic-tcs-research-system/0.1 (OpenAlex discovery)"


class OpenAlexClient:
    """Minimal OpenAlex wrapper.

    This intentionally exposes only simple Scholar-like search plus one-hop references/citations.
    It can later grow into cursor pagination, scoring policies, and provider abstraction.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._cache: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, Any]] = {}

    def search(self, query: str, *, limit: int = 10) -> list[LiteratureCandidate]:
        query = query.strip()
        if not query:
            raise ValueError("query must be non-empty")
        payload = self._get_json(
            "/works",
            params={"search": query, "per-page": str(min(max(limit, 1), 200))},
        )
        return [
            _candidate_from_work(
                work,
                discovery_reason=f"OpenAlex search result for query: {query}",
                score=_openalex_score(work),
            )
            for work in payload.get("results", [])[:limit]
        ]

    def references(self, value: str, *, limit: int = 20) -> list[LiteratureCandidate]:
        work = self.resolve_work(value)
        refs = list(work.get("referenced_works") or [])[:limit]
        candidates: list[LiteratureCandidate] = []
        for ref in refs:
            try:
                ref_work = self._get_work(ref)
            except Exception:
                continue
            candidates.append(
                _candidate_from_work(
                    ref_work,
                    discovery_reason=(
                        f"Referenced by {work.get('display_name') or work.get('title')}"
                    ),
                )
            )
        return candidates

    def citations(self, value: str, *, limit: int = 20) -> list[LiteratureCandidate]:
        work = self.resolve_work(value)
        work_id = _openalex_short_id(str(work.get("id") or value))
        payload = self._get_json(
            "/works",
            params={"filter": f"cites:{work_id}", "per-page": str(min(max(limit, 1), 200))},
        )
        return [
            _candidate_from_work(
                citing,
                discovery_reason=f"Cites {work.get('display_name') or work.get('title')}",
                score=_openalex_score(citing),
            )
            for citing in payload.get("results", [])[:limit]
        ]

    def resolve_work(self, value: str) -> dict[str, Any]:
        value = value.strip()
        if not value:
            raise ValueError("OpenAlex work lookup value must be non-empty")
        if _looks_like_openalex_id(value):
            return self._get_work(value)

        doi = parse_doi(value)
        if doi:
            work = self._first_by_filter(f"doi:{normalize_doi(doi)}")
            if work is not None:
                return work
            work = self._first_by_filter(f"doi:https://doi.org/{normalize_doi(doi)}")
            if work is not None:
                return work

        arxiv_id = parse_arxiv_id(value)
        search_value = f"arXiv:{arxiv_id}" if arxiv_id else value
        payload = self._get_json("/works", params={"search": search_value, "per-page": "1"})
        results = payload.get("results") or []
        if results:
            return dict(results[0])
        raise LookupError(f"OpenAlex could not resolve work: {value}")

    def _first_by_filter(self, filter_value: str) -> dict[str, Any] | None:
        payload = self._get_json("/works", params={"filter": filter_value, "per-page": "1"})
        results = payload.get("results") or []
        return dict(results[0]) if results else None

    def _get_work(self, value: str) -> dict[str, Any]:
        short_id = _openalex_short_id(value)
        return self._get_json(f"/works/{short_id}")

    def _get_json(self, path: str, *, params: dict[str, str] | None = None) -> dict[str, Any]:
        cache_key = (path, tuple(sorted((params or {}).items())))
        if cache_key in self._cache:
            return dict(self._cache[cache_key])
        headers = {"User-Agent": USER_AGENT}
        last_exc: Exception | None = None
        with httpx.Client(timeout=self.timeout_seconds, headers=headers) as client:
            for attempt in range(self.max_retries + 1):
                response = client.get(OPENALEX_API.rstrip("/") + path, params=params)
                if response.status_code < 400:
                    payload = dict(response.json())
                    self._cache[cache_key] = payload
                    return dict(payload)
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    last_exc = exc
                    if response.status_code not in {429, 500, 502, 503, 504} or attempt >= self.max_retries:
                        raise
                    retry_after = _retry_after_seconds(response)
                    delay = retry_after if retry_after is not None else self.backoff_seconds * (2**attempt)
                    time.sleep(min(delay, 30.0))
            if last_exc is not None:
                raise last_exc
        raise RuntimeError(f"OpenAlex request failed for {path}")


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _candidate_from_work(
    work: dict[str, Any], *, discovery_reason: str, score: float = 0.0
) -> LiteratureCandidate:
    title = str(work.get("display_name") or work.get("title") or "Untitled")
    doi = _doi_from_work(work)
    landing_url, pdf_url, source_urls = _urls_from_work(work)
    arxiv_id = _arxiv_from_work(work, [landing_url, pdf_url, *source_urls])
    return LiteratureCandidate(
        title=title,
        authors=_authors_from_work(work),
        year=work.get("publication_year"),
        venue=_venue_from_work(work),
        doi=doi,
        arxiv_id=arxiv_id,
        openalex_id=str(work.get("id") or ""),
        abstract=_abstract_from_inverted_index(work.get("abstract_inverted_index")),
        landing_url=landing_url,
        pdf_url=pdf_url,
        source_urls=source_urls,
        cited_by_count=int(work.get("cited_by_count") or 0),
        discovery_reason=discovery_reason,
        score=score,
    )


def _authors_from_work(work: dict[str, Any]) -> list[str]:
    authors = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        name = str(author.get("display_name") or "").strip()
        if name:
            authors.append(name)
    return authors


def _venue_from_work(work: dict[str, Any]) -> str:
    for location_key in ["primary_location", "best_oa_location"]:
        source = ((work.get(location_key) or {}).get("source") or {})
        name = str(source.get("display_name") or "").strip()
        if name:
            return name
    return ""


def _doi_from_work(work: dict[str, Any]) -> str:
    doi = str(work.get("doi") or "")
    return normalize_doi(doi) if doi else ""


def _urls_from_work(work: dict[str, Any]) -> tuple[str, str, list[str]]:
    urls: list[str] = []
    landing_url = ""
    pdf_url = ""
    for key in ["best_oa_location", "primary_location"]:
        location = work.get(key) or {}
        landing_url = landing_url or str(location.get("landing_page_url") or "")
        pdf_url = pdf_url or str(location.get("pdf_url") or "")
        urls.extend([landing_url, pdf_url])
    open_access = work.get("open_access") or {}
    urls.append(str(open_access.get("oa_url") or ""))
    doi = str(work.get("doi") or "")
    if doi:
        urls.append(doi)
    openalex_id = str(work.get("id") or "")
    if openalex_id:
        urls.append(openalex_id)
    deduped = [url for url in dict.fromkeys(urls) if url]
    return landing_url, pdf_url, deduped


def _arxiv_from_work(work: dict[str, Any], urls: list[str]) -> str:
    ids = work.get("ids") or {}
    for value in [ids.get("arxiv"), *urls]:
        if not value:
            continue
        arxiv_id = parse_arxiv_id(str(value))
        if arxiv_id:
            return arxiv_id
    return ""


def _abstract_from_inverted_index(index: Any) -> str:
    if not isinstance(index, dict):
        return ""
    positions: dict[int, str] = {}
    for word, offsets in index.items():
        if not isinstance(offsets, list):
            continue
        for offset in offsets:
            if isinstance(offset, int):
                positions[offset] = str(word)
    return " ".join(positions[i] for i in sorted(positions))


def _openalex_score(work: dict[str, Any]) -> float:
    score = work.get("relevance_score")
    if isinstance(score, (int, float)):
        return float(score)
    return float(work.get("cited_by_count") or 0)


def _looks_like_openalex_id(value: str) -> bool:
    return bool(re.search(r"(?:openalex\.org/)?W\d+", value, flags=re.IGNORECASE))


def _openalex_short_id(value: str) -> str:
    match = re.search(r"W\d+", value, flags=re.IGNORECASE)
    if match:
        return match.group(0).upper()
    return value.rstrip("/").rsplit("/", 1)[-1]
