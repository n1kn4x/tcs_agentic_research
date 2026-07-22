"""Network importers for arXiv, DOI, URL, and direct-PDF literature sources."""

from __future__ import annotations

import re
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ..artifact_store import ArtifactStore
from ..schemas import PaperMetadata

ARXIV_API = "https://export.arxiv.org/api/query"
CROSSREF_API = "https://api.crossref.org/works"
USER_AGENT = "agentic-tcs-research-system/0.1 (literature importer)"


@dataclass(frozen=True)
class DownloadedResource:
    final_url: str
    content_type: str
    content: bytes


class LiteratureFetcher:
    """Fetch papers and metadata into ``LiteratureDB/papers/``.

    The fetcher does only deterministic network/file work. It does not call an LLM and can be
    tested independently from the literature extraction and query-answering agent.
    """

    def __init__(self, store: ArtifactStore, *, timeout_seconds: float = 60.0):
        self.store = store
        self.timeout_seconds = timeout_seconds

    def import_url(
        self,
        url: str,
        *,
        citation_key: str | None = None,
        title: str | None = None,
        doi: str | None = None,
        commit: bool = True,
    ) -> PaperMetadata:
        """Import a paper from a URL, DOI URL/string, arXiv URL, or direct PDF URL."""
        normalized = url.strip()
        if not normalized:
            raise ValueError("url must be non-empty")

        arxiv_id = parse_arxiv_id(normalized)
        if arxiv_id:
            return self.import_arxiv(arxiv_id, citation_key=citation_key, commit=commit)

        doi_value = doi or parse_doi(normalized)
        if doi_value:
            return self.import_doi(
                doi_value, citation_key=citation_key, source_url=normalized, commit=commit
            )

        resource = self._download(normalized)
        key = citation_key or _citation_key_from_url(resource.final_url)
        rel_dir = self._paper_dir(key)
        artifact_refs = []
        content_type = resource.content_type.lower()
        final_path = resource.final_url.lower().split("?")[0]
        is_pdf = "application/pdf" in content_type or final_path.endswith(".pdf")
        text_path = ""
        if is_pdf:
            pdf_ref = self.store.write_bytes(f"{rel_dir}/paper.pdf", resource.content)
            artifact_refs.append(pdf_ref)
            pdf_path = pdf_ref.path
            source_type = "pdf"
        else:
            suffix = ".html" if "html" in content_type else ".txt"
            source_ref = self.store.write_bytes(f"{rel_dir}/source{suffix}", resource.content)
            artifact_refs.append(source_ref)
            pdf_path = ""
            text_path = source_ref.path if suffix == ".txt" else ""
            source_type = "url"

        paper = PaperMetadata(
            citation_key=key,
            title=title or _title_from_url(resource.final_url),
            url=resource.final_url,
            doi=doi or "",
            source_type=source_type,
            source_urls=list(dict.fromkeys([normalized, resource.final_url])),
            pdf_path=pdf_path,
            text_path=text_path,
            artifact_refs=artifact_refs,
        )
        metadata_ref = self._write_paper_metadata(paper)
        paper.metadata_path = metadata_ref.path
        paper.artifact_refs.append(metadata_ref)
        self._rewrite_paper_metadata(paper)
        if commit:
            self.store.append_jsonl("LiteratureDB/papers.jsonl", paper)
        return paper

    def import_arxiv(
        self, arxiv_id: str, *, citation_key: str | None = None, commit: bool = True
    ) -> PaperMetadata:
        """Fetch arXiv Atom metadata and the canonical arXiv PDF."""
        clean_id = normalize_arxiv_id(arxiv_id)
        metadata = self._fetch_arxiv_metadata(clean_id)
        key = citation_key or f"arxiv_{_safe_slug(clean_id)}"
        rel_dir = self._paper_dir(key)
        pdf_url = metadata.get("pdf_url") or f"https://arxiv.org/pdf/{clean_id}.pdf"
        pdf = self._download(pdf_url)
        pdf_ref = self.store.write_bytes(f"{rel_dir}/paper.pdf", pdf.content)
        paper = PaperMetadata(
            citation_key=key,
            title=str(metadata.get("title") or f"arXiv:{clean_id}"),
            authors=list(metadata.get("authors") or []),
            year=metadata.get("year"),
            venue="arXiv",
            url=str(metadata.get("url") or f"https://arxiv.org/abs/{clean_id}"),
            arxiv_id=clean_id,
            abstract=str(metadata.get("abstract") or ""),
            source_type="arxiv",
            source_urls=[
                str(metadata.get("url") or f"https://arxiv.org/abs/{clean_id}"),
                pdf.final_url,
            ],
            pdf_path=pdf_ref.path,
            artifact_refs=[pdf_ref],
        )
        metadata_ref = self._write_paper_metadata(paper)
        paper.metadata_path = metadata_ref.path
        paper.artifact_refs.append(metadata_ref)
        self._rewrite_paper_metadata(paper)
        if commit:
            self.store.append_jsonl("LiteratureDB/papers.jsonl", paper)
        return paper

    def import_discovered_arxiv(
        self,
        arxiv_id: str,
        *,
        title: str,
        authors: list[str],
        year: int | None,
        abstract: str,
        pdf_url: str = "",
        landing_url: str = "",
        citation_key: str | None = None,
        commit: bool = True,
    ) -> PaperMetadata:
        """Import an arXiv search result without fetching its Atom metadata a second time."""
        clean_id = normalize_arxiv_id(arxiv_id)
        key = citation_key or f"arxiv_{_safe_slug(clean_id)}"
        rel_dir = self._paper_dir(key)
        source_pdf = pdf_url or f"https://arxiv.org/pdf/{clean_id}.pdf"
        pdf = self._download(source_pdf)
        pdf_ref = self.store.write_bytes(f"{rel_dir}/paper.pdf", pdf.content)
        source_landing = landing_url or f"https://arxiv.org/abs/{clean_id}"
        paper = PaperMetadata(
            citation_key=key,
            title=title or f"arXiv:{clean_id}",
            authors=authors,
            year=year,
            venue="arXiv",
            url=source_landing,
            arxiv_id=clean_id,
            abstract=abstract,
            source_type="arxiv",
            source_urls=list(dict.fromkeys([source_landing, pdf.final_url])),
            pdf_path=pdf_ref.path,
            artifact_refs=[pdf_ref],
        )
        metadata_ref = self._write_paper_metadata(paper)
        paper.metadata_path = metadata_ref.path
        paper.artifact_refs.append(metadata_ref)
        self._rewrite_paper_metadata(paper)
        if commit:
            self.store.append_jsonl("LiteratureDB/papers.jsonl", paper)
        return paper

    def import_doi(
        self,
        doi: str,
        *,
        citation_key: str | None = None,
        source_url: str | None = None,
        commit: bool = True,
    ) -> PaperMetadata:
        """Fetch DOI metadata through Crossref and download a PDF when a direct link exists."""
        clean_doi = normalize_doi(doi)
        crossref = self._fetch_crossref_metadata(clean_doi)
        key = citation_key or _citation_key_from_doi(clean_doi, crossref)
        rel_dir = self._paper_dir(key)
        artifact_refs = []
        pdf_path = ""
        source_urls: list[str] = []

        pdf_url = _crossref_pdf_url(crossref)
        if pdf_url:
            try:
                pdf = self._download(pdf_url)
                pdf_ref = self.store.write_bytes(f"{rel_dir}/paper.pdf", pdf.content)
                artifact_refs.append(pdf_ref)
                pdf_path = pdf_ref.path
                source_urls.append(pdf.final_url)
            except Exception:
                # Restricted DOI PDFs are common. Keep metadata/landing-page provenance instead.
                pass

        landing_url = source_url or f"https://doi.org/{clean_doi}"
        try:
            landing = self._download(landing_url)
            source_urls.append(landing.final_url)
            landing_is_pdf = (
                "application/pdf" in landing.content_type.lower()
                or landing.final_url.lower().endswith(".pdf")
            )
            if not pdf_path and landing_is_pdf:
                pdf_ref = self.store.write_bytes(f"{rel_dir}/paper.pdf", landing.content)
                artifact_refs.append(pdf_ref)
                pdf_path = pdf_ref.path
            else:
                suffix = ".html" if "html" in landing.content_type.lower() else ".txt"
                landing_ref = self.store.write_bytes(f"{rel_dir}/landing{suffix}", landing.content)
                artifact_refs.append(landing_ref)
        except Exception:
            source_urls.append(landing_url)

        paper = PaperMetadata(
            citation_key=key,
            title=_crossref_title(crossref) or f"DOI:{clean_doi}",
            authors=_crossref_authors(crossref),
            year=_crossref_year(crossref),
            venue=_crossref_venue(crossref),
            url=landing_url,
            doi=clean_doi,
            abstract=str(crossref.get("abstract") or ""),
            source_type="doi",
            source_urls=list(dict.fromkeys(source_urls or [landing_url])),
            pdf_path=pdf_path,
            artifact_refs=artifact_refs,
        )
        metadata_ref = self._write_paper_metadata(paper)
        paper.metadata_path = metadata_ref.path
        paper.artifact_refs.append(metadata_ref)
        self._rewrite_paper_metadata(paper)
        if commit:
            self.store.append_jsonl("LiteratureDB/papers.jsonl", paper)
        return paper

    def _paper_dir(self, citation_key: str) -> str:
        return f"LiteratureDB/papers/{_safe_slug(citation_key)}"

    def _write_paper_metadata(self, paper: PaperMetadata):
        return self.store.write_json(f"{self._paper_dir(paper.citation_key)}/metadata.json", paper)

    def _rewrite_paper_metadata(self, paper: PaperMetadata) -> None:
        self.store.write_json(f"{self._paper_dir(paper.citation_key)}/metadata.json", paper)

    def _download(self, url: str) -> DownloadedResource:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/pdf,text/html,text/plain,*/*"}
        with httpx.Client(
            timeout=self.timeout_seconds, follow_redirects=True, headers=headers
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "application/octet-stream")
            return DownloadedResource(str(response.url), content_type, response.content)

    def _fetch_arxiv_metadata(self, arxiv_id: str) -> dict[str, Any]:
        params = {"id_list": arxiv_id, "max_results": "1"}
        with httpx.Client(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            response = client.get(ARXIV_API, params=params)
            response.raise_for_status()
        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            raise RuntimeError(f"arXiv returned no entry for {arxiv_id}")
        title = _compact_ws(entry.findtext("atom:title", default="", namespaces=ns))
        abstract = _compact_ws(entry.findtext("atom:summary", default="", namespaces=ns))
        published = entry.findtext("atom:published", default="", namespaces=ns)
        year = int(published[:4]) if published[:4].isdigit() else None
        authors = [
            _compact_ws(author.findtext("atom:name", default="", namespaces=ns))
            for author in entry.findall("atom:author", ns)
        ]
        authors = [a for a in authors if a]
        abs_url = ""
        pdf_url = ""
        for link in entry.findall("atom:link", ns):
            href = link.attrib.get("href", "")
            rel = link.attrib.get("rel", "")
            title_attr = link.attrib.get("title", "")
            if rel == "alternate":
                abs_url = href
            if title_attr == "pdf" or href.endswith(".pdf"):
                pdf_url = href
        return {
            "title": title,
            "abstract": abstract,
            "year": year,
            "authors": authors,
            "url": abs_url,
            "pdf_url": pdf_url,
        }

    def _fetch_crossref_metadata(self, doi: str) -> dict[str, Any]:
        url = f"{CROSSREF_API}/{urllib.parse.quote(doi, safe='')}"
        with httpx.Client(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
        payload = response.json()
        return dict(payload.get("message") or {})


def parse_arxiv_id(value: str) -> str | None:
    text = value.strip()
    # arXiv identifiers, optionally embedded in abs/pdf URLs.
    patterns = [
        r"arxiv\.org/(?:abs|pdf)/([^?#/]+)",
        r"^arXiv:([^\s]+)$",
        r"^(\d{4}\.\d{4,5}(?:v\d+)?)$",
        r"^([a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_arxiv_id(match.group(1).removesuffix(".pdf"))
    return None


def normalize_arxiv_id(arxiv_id: str) -> str:
    return arxiv_id.strip().removeprefix("arXiv:").removesuffix(".pdf")


def parse_doi(value: str) -> str | None:
    text = urllib.parse.unquote(value.strip())
    doi_url = re.search(r"doi\.org/(10\.\d{4,9}/[^\s?#]+)", text, flags=re.IGNORECASE)
    if doi_url:
        return normalize_doi(doi_url.group(1))
    doi = re.search(r"\b(10\.\d{4,9}/[^\s]+)", text, flags=re.IGNORECASE)
    if doi:
        return normalize_doi(doi.group(1).rstrip(".))]"))
    return None


def normalize_doi(doi: str) -> str:
    text = doi.strip()
    text = re.sub(r"^doi:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.IGNORECASE)
    return urllib.parse.unquote(text).strip()


def _safe_slug(value: str, *, default: str = "paper") -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return (slug or default)[:120]


def _citation_key_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    stem = Path(parsed.path).stem or parsed.netloc or "paper"
    host = parsed.netloc.replace("www.", "").split(":")[0]
    return _safe_slug(f"{host}_{stem}")


def _title_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    stem = Path(parsed.path).stem.replace("_", " ").replace("-", " ").strip()
    return stem or parsed.netloc or url


def _citation_key_from_doi(doi: str, metadata: dict[str, Any]) -> str:
    year = _crossref_year(metadata)
    first_author = ""
    authors = metadata.get("author") or []
    if authors:
        first_author = str(authors[0].get("family") or authors[0].get("name") or "")
    title = _crossref_title(metadata)
    title_word = re.sub(r"[^A-Za-z0-9]+", " ", title).split()
    stem = title_word[0] if title_word else doi.split("/")[-1]
    pieces = [p for p in [first_author, str(year or ""), stem] if p]
    return _safe_slug("_".join(pieces), default="doi_" + _safe_slug(doi))


def _crossref_title(metadata: dict[str, Any]) -> str:
    title = metadata.get("title") or []
    if isinstance(title, list) and title:
        return _compact_ws(str(title[0]))
    return _compact_ws(str(title or ""))


def _crossref_authors(metadata: dict[str, Any]) -> list[str]:
    authors = []
    for author in metadata.get("author") or []:
        given = author.get("given") or ""
        family = author.get("family") or author.get("name") or ""
        name = _compact_ws(f"{given} {family}".strip())
        if name:
            authors.append(name)
    return authors


def _crossref_year(metadata: dict[str, Any]) -> int | None:
    for field in ["published-print", "published-online", "issued"]:
        date_parts = ((metadata.get(field) or {}).get("date-parts") or [])
        if date_parts and date_parts[0] and isinstance(date_parts[0][0], int):
            return date_parts[0][0]
    return None


def _crossref_venue(metadata: dict[str, Any]) -> str:
    venue = metadata.get("container-title") or []
    if isinstance(venue, list) and venue:
        return _compact_ws(str(venue[0]))
    return _compact_ws(str(venue or ""))


def _crossref_pdf_url(metadata: dict[str, Any]) -> str:
    for link in metadata.get("link") or []:
        content_type = str(link.get("content-type") or "").lower()
        url = str(link.get("URL") or "")
        if url and ("pdf" in content_type or url.lower().split("?")[0].endswith(".pdf")):
            return url
    return ""


def _compact_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
