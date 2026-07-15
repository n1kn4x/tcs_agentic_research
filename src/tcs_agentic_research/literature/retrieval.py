"""Transparent local information retrieval over LiteratureDB artifacts."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Any, Iterable

from ..artifact_store import ArtifactStore
from ..schemas import (
    LiteratureDuplicateGroup,
    LiteratureQueryAnswer,
    LiteratureQueryResult,
    LiteratureQuote,
    LiteratureStatement,
    PaperMetadata,
)
from .index import LiteratureIndex
from .nomenclature import NomenclatureMapper


class LiteratureRetriever:
    """Keyword/BM25-lite retrieval with quote-level provenance.

    This is intentionally simple and auditable. The canonical JSONL records remain the source of
    truth; a vector index can later be added behind the same result schema.
    """

    def __init__(
        self,
        store: ArtifactStore,
        mapper: NomenclatureMapper | None = None,
        *,
        index: LiteratureIndex | None = None,
    ):
        self.store = store
        self.mapper = mapper or NomenclatureMapper(store)
        self.index = index or LiteratureIndex(store)

    def answer_query(self, query: str, *, limit: int = 10) -> LiteratureQueryAnswer:
        query = query.strip()
        if not query:
            raise ValueError("query must be non-empty")
        results = self.retrieve(query, limit=limit)
        duplicates = detect_duplicate_results(results)
        used = self.mapper.used_mappings_for_text(query)
        for result in results:
            used.update(result.notation_mappings)
            used.update(self.mapper.used_mappings_for_text(result.mapped_statement))
        answer_text = _render_answer(query, results, duplicates)
        limitations = []
        if not results:
            limitations.append("No matching local LiteratureDB records were found.")
        limitations.append(
            "Retrieval is local to imported papers and extracted statements; "
            "import more papers for broader coverage."
        )
        return LiteratureQueryAnswer(
            query=query,
            answer=answer_text,
            results=results,
            duplicate_results=duplicates,
            limitations=limitations,
            used_nomenclature=used,
        )

    def retrieve(self, query: str, *, limit: int = 10) -> list[LiteratureQueryResult]:
        indexed = self._indexed_results(query, limit=limit)
        if indexed:
            return indexed[:limit]

        # Backward-compatible fallback for legacy workspaces without a populated index.
        terms = _terms(query)
        papers = _paper_index(self.store)
        candidates = list(self._statement_candidates(papers)) + list(
            self._text_chunk_candidates(papers)
        )
        scored: list[LiteratureQueryResult] = []
        for result, searchable in candidates:
            score = _score(terms, searchable)
            if score <= 0:
                continue
            result.score = score
            result.mapped_statement = self.mapper.map_text(
                result.mapped_statement, result.notation_mappings
            )
            scored.append(result)
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def _indexed_results(self, query: str, *, limit: int) -> list[LiteratureQueryResult]:
        rows = self.index.search(query, limit=max(limit, 1))
        results: list[LiteratureQueryResult] = []
        for row in rows:
            citation_key = str(row.get("citation_key") or "")
            paper_id = str(row.get("paper_id") or "")
            quote = str(row.get("quote") or row.get("mapped_statement") or "")
            quote_obj = LiteratureQuote(
                citation_key=citation_key,
                paper_id=paper_id,
                locator=str(row.get("locator") or row.get("label") or ""),
                quote=quote,
                char_start=row.get("char_start"),
                char_end=row.get("char_end"),
                source_sha256=str(row.get("text_sha256") or ""),
                validated=bool(row.get("validated") or False),
            )
            if row.get("quote_id"):
                quote_obj.quote_id = str(row.get("quote_id"))
            mapped = str(row.get("mapped_statement") or "")
            notation_mappings: dict[str, str] = {}
            if row.get("notation_json"):
                try:
                    parsed = json.loads(str(row.get("notation_json")))
                    if isinstance(parsed, dict):
                        notation_mappings = {str(k): str(v) for k, v in parsed.items()}
                except Exception:
                    notation_mappings = {}
            result = LiteratureQueryResult(
                citation_key=citation_key,
                paper_id=paper_id,
                title=str(row.get("paper_title") or ""),
                year=row.get("year"),
                kind=str(row.get("kind") or row.get("result_kind") or "text_chunk"),
                label=str(row.get("label") or row.get("locator") or ""),
                mapped_statement=self.mapper.map_text(mapped, notation_mappings),
                summary=_summary(mapped),
                score=float(row.get("score") or 0.0),
                statement_id=str(row.get("statement_id") or ""),
                quote_id=quote_obj.quote_id,
                support_id=str(row.get("support_id") or ""),
                support_level=str(row.get("support_level") or ""),
                relation=str(row.get("relation") or ""),
                provenance=[quote_obj],
                notation_mappings=notation_mappings,
            )
            results.append(result)
        return results

    def _statement_candidates(
        self, papers: dict[str, PaperMetadata]
    ) -> Iterable[tuple[LiteratureQueryResult, str]]:
        for record in self.store.read_jsonl("LiteratureDB/extracted_claims.jsonl"):
            citation_key = str(record.get("citation_key") or "")
            paper = papers.get(citation_key)
            paper_id = str(record.get("paper_id") or (paper.paper_id if paper else ""))
            all_statements: list[Any] = []
            for field in ["theorem_statements", "algorithm_statements", "lower_bound_statements"]:
                all_statements.extend(record.get(field) or [])
            for raw_statement in all_statements:
                statement = _coerce_statement(
                    raw_statement, citation_key=citation_key, paper_id=paper_id
                )
                if statement is None:
                    continue
                mapped = statement.mapped_statement or self.mapper.map_text(
                    statement.original_statement, statement.notation_mappings
                )
                provenance = statement.provenance or [
                    LiteratureQuote(
                        citation_key=citation_key,
                        paper_id=paper_id,
                        locator=statement.label,
                        quote=statement.original_statement,
                    )
                ]
                title = paper.title if paper else ""
                year = paper.year if paper else None
                result = LiteratureQueryResult(
                    citation_key=citation_key,
                    paper_id=paper_id,
                    title=title,
                    year=year,
                    kind=statement.kind,
                    label=statement.label,
                    mapped_statement=mapped,
                    summary=_summary(mapped),
                    provenance=provenance,
                    notation_mappings=(
                        statement.notation_mappings or dict(record.get("notation_mappings") or {})
                    ),
                )
                searchable = "\n".join(
                    [
                        citation_key,
                        title,
                        statement.kind,
                        statement.label,
                        statement.original_statement,
                        mapped,
                        " ".join(q.quote for q in provenance),
                    ]
                )
                yield result, searchable

    def _text_chunk_candidates(
        self, papers: dict[str, PaperMetadata]
    ) -> Iterable[tuple[LiteratureQueryResult, str]]:
        for paper in papers.values():
            if not paper.text_path or not self.store.exists(paper.text_path):
                continue
            text = self.store.read_text(paper.text_path)
            for idx, chunk in enumerate(_chunks(text)):
                mapped = self.mapper.map_text(chunk.text)
                quote = LiteratureQuote(
                    citation_key=paper.citation_key,
                    paper_id=paper.paper_id,
                    locator=chunk.locator or f"chunk {idx + 1}",
                    quote=chunk.text,
                    char_start=chunk.start,
                    char_end=chunk.end,
                    artifact_refs=[self.store.artifact_ref(paper.text_path)],
                )
                result = LiteratureQueryResult(
                    citation_key=paper.citation_key,
                    paper_id=paper.paper_id,
                    title=paper.title,
                    year=paper.year,
                    kind="text_chunk",
                    label=quote.locator,
                    mapped_statement=mapped,
                    summary=_summary(mapped),
                    provenance=[quote],
                    notation_mappings=self.mapper.used_mappings_for_text(chunk.text),
                )
                yield result, "\n".join([paper.citation_key, paper.title, chunk.text, mapped])


def detect_duplicate_results(
    results: list[LiteratureQueryResult],
) -> list[LiteratureDuplicateGroup]:
    groups: list[LiteratureDuplicateGroup] = []
    by_key: dict[str, list[LiteratureQueryResult]] = defaultdict(list)
    for result in results:
        key = _canonical_statement_key(result.mapped_statement)
        if key:
            by_key[key].append(result)
    for key, grouped in by_key.items():
        if len(grouped) <= 1:
            continue
        anchor = grouped[0].result_id
        for duplicate in grouped[1:]:
            duplicate.duplicate_of = anchor
        groups.append(
            LiteratureDuplicateGroup(
                result_ids=[item.result_id for item in grouped],
                canonical_key=key[:160],
                reason="Exact normalized mapped statement match.",
            )
        )

    # Near-duplicate pass for paraphrases in mapped notation.
    already_grouped = {rid for group in groups for rid in group.result_ids}
    for i, left in enumerate(results):
        if left.result_id in already_grouped:
            continue
        near = [left]
        for right in results[i + 1 :]:
            if right.result_id in already_grouped:
                continue
            if _jaccard(_terms(left.mapped_statement), _terms(right.mapped_statement)) >= 0.86:
                near.append(right)
        if len(near) > 1:
            anchor = near[0].result_id
            for duplicate in near[1:]:
                duplicate.duplicate_of = anchor
            already_grouped.update(item.result_id for item in near)
            groups.append(
                LiteratureDuplicateGroup(
                    result_ids=[item.result_id for item in near],
                    canonical_key=_canonical_statement_key(near[0].mapped_statement)[:160],
                    reason="High token-overlap near duplicate after nomenclature mapping.",
                )
            )
    return groups


def _paper_index(store: ArtifactStore) -> dict[str, PaperMetadata]:
    papers: dict[str, PaperMetadata] = {}
    for record in store.read_jsonl("LiteratureDB/papers.jsonl"):
        try:
            paper = PaperMetadata.model_validate(record)
        except Exception:
            continue
        papers[paper.citation_key] = paper
    return papers


def _coerce_statement(raw: Any, *, citation_key: str, paper_id: str) -> LiteratureStatement | None:
    if isinstance(raw, str):
        return LiteratureStatement(
            citation_key=citation_key,
            paper_id=paper_id,
            kind="other",
            original_statement=raw,
            mapped_statement=raw,
        )
    if isinstance(raw, dict):
        payload = dict(raw)
        payload.setdefault("citation_key", citation_key)
        payload.setdefault("paper_id", paper_id)
        try:
            return LiteratureStatement.model_validate(payload)
        except Exception:
            return None
    return None


def _score(terms: list[str], text: str) -> float:
    if not terms:
        return 0.0
    lowered = text.lower()
    length_norm = 1.0 + math.log(max(len(lowered), 20) / 20)
    score = 0.0
    for term in terms:
        count = lowered.count(term)
        if count:
            score += 1.0 + math.log(count)
    # Reward phrase hits.
    phrase = " ".join(terms)
    if len(phrase) > 4 and phrase in lowered:
        score += 2.0
    return round(score / length_norm, 4)


def _terms(text: str) -> list[str]:
    return [term for term in re.findall(r"[A-Za-z0-9_\\-]{3,}", text.lower()) if term not in _STOP]


_STOP = {
    "and",
    "are",
    "for",
    "from",
    "into",
    "that",
    "the",
    "then",
    "this",
    "with",
    "where",
}


def _summary(text: str, *, limit: int = 300) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def _canonical_statement_key(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(theorem|lemma|corollary|proposition|algorithm)\s+[0-9a-z.:-]+", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _jaccard(left: list[str], right: list[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class _Chunk:
    def __init__(self, text: str, start: int, end: int, locator: str = ""):
        self.text = text
        self.start = start
        self.end = end
        self.locator = locator


def _chunks(text: str, *, size: int = 1600, overlap: int = 200) -> Iterable[_Chunk]:
    page_positions = list(re.finditer(r"--- page (\d+) ---", text, flags=re.IGNORECASE))
    if len(text) <= size:
        yield _Chunk(
            text=text.strip(), start=0, end=len(text), locator=_page_locator(page_positions, 0)
        )
        return
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
            if boundary > start + size // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            yield _Chunk(
                text=chunk, start=start, end=end, locator=_page_locator(page_positions, start)
            )
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)


def _page_locator(page_positions: list[re.Match[str]], char_pos: int) -> str:
    page = ""
    for match in page_positions:
        if match.start() <= char_pos:
            page = match.group(1)
        else:
            break
    return f"page {page}" if page else ""


def _render_answer(
    query: str, results: list[LiteratureQueryResult], duplicates: list[LiteratureDuplicateGroup]
) -> str:
    if not results:
        return f"No local LiteratureDB result answers `{query}`."
    lines = ["Mapped-nomenclature literature results:"]
    for idx, result in enumerate(results, start=1):
        duplicate = f" duplicate_of={result.duplicate_of}" if result.duplicate_of else ""
        quote = result.provenance[0].quote if result.provenance else ""
        quote = _summary(quote, limit=220)
        locator = result.provenance[0].locator if result.provenance else result.label
        support = f", support_id={result.support_id}" if result.support_id else ""
        validated = "validated" if result.provenance and result.provenance[0].validated else "unvalidated"
        lines.append(
            f"{idx}. [{result.citation_key}] {result.mapped_statement}"
            f" (kind={result.kind}, locator={locator}, score={result.score}{support}, {validated}{duplicate})"
        )
        if quote:
            lines.append(f"   quote (original notation, provenance only): \"{quote}\"")
    if duplicates:
        lines.append("Duplicate-result groups detected:")
        for group in duplicates:
            lines.append(f"- {', '.join(group.result_ids)}: {group.reason}")
    return "\n".join(lines)


def dumps_for_debug(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)
