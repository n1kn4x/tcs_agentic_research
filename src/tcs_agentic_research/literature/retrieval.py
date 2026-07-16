"""Transparent local information retrieval over LiteratureDB artifacts."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from ..artifact_store import ArtifactStore
from ..schemas import (
    LiteratureDuplicateGroup,
    LiteratureQueryAnswer,
    LiteratureQueryResult,
    LiteratureQuote,
)
from .index import LiteratureIndex


class LiteratureRetriever:
    """Keyword/BM25-lite retrieval with quote-level provenance."""

    def __init__(self, store: ArtifactStore, *, index: LiteratureIndex | None = None):
        self.store = store
        self.index = index or LiteratureIndex(store)

    def answer_query(self, query: str, *, limit: int = 10) -> LiteratureQueryAnswer:
        query = query.strip()
        if not query:
            raise ValueError("query must be non-empty")
        results = self.retrieve(query, limit=limit)
        duplicates = detect_duplicate_results(results)
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
        )

    def retrieve(self, query: str, *, limit: int = 10) -> list[LiteratureQueryResult]:
        return self._indexed_results(query, limit=max(limit, 1))[:limit]

    def _indexed_results(self, query: str, *, limit: int) -> list[LiteratureQueryResult]:
        rows = self.index.search(query, limit=limit)
        results: list[LiteratureQueryResult] = []
        for row in rows:
            citation_key = str(row.get("citation_key") or "")
            paper_id = str(row.get("paper_id") or "")
            statement_text = str(row.get("statement_text") or "")
            quote = str(row.get("quote") or statement_text)
            quote_obj = LiteratureQuote(
                citation_key=citation_key,
                paper_id=paper_id,
                locator=str(row.get("locator") or row.get("label") or ""),
                quote=quote,
                char_start=row.get("char_start"),
                char_end=row.get("char_end"),
                source_sha256=str(row.get("text_sha256") or ""),
                validated=bool(row.get("quote_validated") or False),
            )
            if row.get("quote_id"):
                quote_obj.quote_id = str(row.get("quote_id"))
            results.append(
                LiteratureQueryResult(
                    citation_key=citation_key,
                    paper_id=paper_id,
                    title=str(row.get("paper_title") or ""),
                    year=row.get("year"),
                    kind=str(row.get("kind") or row.get("result_kind") or "text_chunk"),
                    label=str(row.get("label") or row.get("locator") or ""),
                    statement_text=statement_text,
                    summary=_summary(statement_text),
                    score=float(row.get("score") or 0.0),
                    statement_id=str(row.get("statement_id") or ""),
                    quote_id=quote_obj.quote_id,
                    support_id=str(row.get("support_id") or ""),
                    support_level=str(row.get("support_level") or ""),
                    relation=str(row.get("relation") or ""),
                    provenance=[quote_obj],
                )
            )
        return results


def detect_duplicate_results(
    results: list[LiteratureQueryResult],
) -> list[LiteratureDuplicateGroup]:
    groups: list[LiteratureDuplicateGroup] = []
    by_key: dict[str, list[LiteratureQueryResult]] = defaultdict(list)
    for result in results:
        key = _canonical_statement_key(result.statement_text)
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
                reason="Exact normalized statement match.",
            )
        )

    already_grouped = {rid for group in groups for rid in group.result_ids}
    for i, left in enumerate(results):
        if left.result_id in already_grouped:
            continue
        near = [left]
        for right in results[i + 1 :]:
            if right.result_id in already_grouped:
                continue
            if _jaccard(_terms(left.statement_text), _terms(right.statement_text)) >= 0.86:
                near.append(right)
        if len(near) > 1:
            anchor = near[0].result_id
            for duplicate in near[1:]:
                duplicate.duplicate_of = anchor
            already_grouped.update(item.result_id for item in near)
            groups.append(
                LiteratureDuplicateGroup(
                    result_ids=[item.result_id for item in near],
                    canonical_key=_canonical_statement_key(near[0].statement_text)[:160],
                    reason="High token-overlap near duplicate statement text.",
                )
            )
    return groups


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


def _render_answer(
    query: str, results: list[LiteratureQueryResult], duplicates: list[LiteratureDuplicateGroup]
) -> str:
    if not results:
        return f"No local LiteratureDB result answers `{query}`."
    lines = ["Local LiteratureDB results with quote provenance:"]
    for idx, result in enumerate(results, start=1):
        duplicate = f" duplicate_of={result.duplicate_of}" if result.duplicate_of else ""
        quote = result.provenance[0].quote if result.provenance else ""
        quote = _summary(quote, limit=220)
        locator = result.provenance[0].locator if result.provenance else result.label
        support = f", support_id={result.support_id}" if result.support_id else ""
        validated = "validated" if result.provenance and result.provenance[0].validated else "unvalidated"
        lines.append(
            f"{idx}. [{result.citation_key}] {result.statement_text}"
            f" (kind={result.kind}, locator={locator}, score={result.score}{support}, {validated}{duplicate})"
        )
        if quote:
            lines.append(f"   quote: \"{quote}\"")
    if duplicates:
        lines.append("Duplicate-result groups detected:")
        for group in duplicates:
            lines.append(f"- {', '.join(group.result_ids)}: {group.reason}")
    return "\n".join(lines)


def dumps_for_debug(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)
