"""Deterministic, schema-light literature audit pipeline.

The audit imports a small source plan, extracts exact statement/quote rows, and renders a report
that separates supported claims, gaps, and follow-up work.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from ..agents.literature import LiteratureResearcher
from ..artifact_store import ArtifactStore
from ..schemas import ArtifactRef, PaperMetadata, utc_now
from .fetchers import normalize_doi


SOURCE_PLAN_PATH = "LiteratureDB/SourcePlan.yml"
AUDIT_TABLE_PATH = "LiteratureDB/audit_table.jsonl"
GAPS_PATH = "LiteratureDB/gaps.jsonl"
REPORT_PATH = "Reports/literature_audit.md"


class LiteratureAuditRunner:
    """Run a deterministic literature-audit workflow for one workspace."""

    def __init__(self, store: ArtifactStore, literature: LiteratureResearcher):
        self.store = store
        self.literature = literature

    def run(
        self,
        *,
        bibliography_path: str | Path | None = None,
        import_sources: bool = True,
        extract_statements: bool = True,
    ) -> dict[str, Any]:
        self.store.initialize_layout()
        task = (
            self.store.read_text(ArtifactStore.RESEARCH_TASK)
            if self.store.exists(ArtifactStore.RESEARCH_TASK)
            else ""
        )
        source_plan = self._source_plan(task, bibliography_path=bibliography_path)
        source_plan_ref = self.store.write_yaml(SOURCE_PLAN_PATH, source_plan)

        import_results = self._import_sources(source_plan, enabled=import_sources)
        extraction_summary: dict[str, Any] = {"status": "skipped", "reason": "not requested"}
        if extract_statements:
            extraction_summary = self.literature.extract_imported_papers(
                max_papers=50,
                only_missing=True,
            )
            extraction_summary["status"] = "ok"

        papers = self.literature.list_papers()
        statements = _read_flat_statements(self.store)
        audit_rows, gaps = self._build_audit(task, source_plan, import_results, papers, statements)

        audit_ref = _write_jsonl_overwrite(self.store, AUDIT_TABLE_PATH, audit_rows)
        gaps_ref = _write_jsonl_overwrite(self.store, GAPS_PATH, gaps)
        report_ref = self.store.write_text(
            REPORT_PATH,
            _render_literature_audit_report(
                task=task,
                source_plan=source_plan,
                import_results=import_results,
                extraction_summary=extraction_summary,
                papers=papers,
                audit_rows=audit_rows,
                gaps=gaps,
            ),
        )
        return {
            "status": "ok",
            "source_plan_path": source_plan_ref.path,
            "audit_table_path": audit_ref.path,
            "gaps_path": gaps_ref.path,
            "report_path": report_ref.path,
            "imported_or_existing_sources": sum(
                1 for result in import_results if result.get("status") in {"imported", "existing"}
            ),
            "failed_or_missing_sources": sum(
                1 for result in import_results if result.get("status") not in {"imported", "existing"}
            ),
            "statement_count": len(statements),
            "supported_claim_count": len([row for row in audit_rows if row["status"] == "supported"]),
            "gap_count": len(gaps),
            "artifact_refs": [
                source_plan_ref.model_dump(mode="json"),
                audit_ref.model_dump(mode="json"),
                gaps_ref.model_dump(mode="json"),
                report_ref.model_dump(mode="json"),
            ],
        }

    def _source_plan(
        self, task: str, *, bibliography_path: str | Path | None
    ) -> dict[str, Any]:
        if bibliography_path is not None:
            raw = yaml.safe_load(Path(bibliography_path).read_text(encoding="utf-8")) or {}
            return _normalize_source_plan(raw, source=f"bibliography:{bibliography_path}")
        return _builtin_source_plan_for_task(task)

    def _import_sources(self, source_plan: dict[str, Any], *, enabled: bool) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for source in [
            *source_plan.get("required_sources", []),
            *source_plan.get("optional_sources", []),
        ]:
            result = dict(source)
            result["required"] = bool(source.get("required", True))
            if not enabled:
                result.update({"status": "skipped", "reason": "not requested"})
                results.append(result)
                continue
            existing = self._find_existing_source(source)
            if existing is not None:
                result.update(
                    {
                        "status": "existing",
                        "paper_id": existing.paper_id,
                        "citation_key": existing.citation_key,
                    }
                )
                results.append(result)
                continue
            try:
                paper = self._import_one_source(source)
            except Exception as exc:  # noqa: BLE001 - failure is a structured audit gap
                result.update(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            else:
                result.update(
                    {
                        "status": "imported",
                        "paper_id": paper.paper_id,
                        "citation_key": paper.citation_key,
                    }
                )
            results.append(result)
        return results

    def _find_existing_source(self, source: dict[str, Any]) -> PaperMetadata | None:
        if source.get("citation_key"):
            existing = self.literature.index.find_paper(citation_key=str(source["citation_key"]))
            if existing is not None:
                return existing
        if source.get("doi"):
            existing = self.literature.index.find_paper(doi=normalize_doi(str(source["doi"])))
            if existing is not None:
                return existing
        if source.get("arxiv_id"):
            existing = self.literature.index.find_paper(arxiv_id=str(source["arxiv_id"]))
            if existing is not None:
                return existing
        if source.get("title"):
            return self.literature.index.find_paper(title=str(source["title"]))
        return None

    def _import_one_source(self, source: dict[str, Any]) -> PaperMetadata:
        expected_title = str(source.get("title") or "") or None
        expected_authors = [str(author) for author in source.get("authors") or []] or None
        expected_year = source.get("year")
        expected_year = int(expected_year) if expected_year is not None else None
        citation_key = str(source.get("citation_key") or "") or None
        if source.get("arxiv_id"):
            return self.literature.import_arxiv(
                str(source["arxiv_id"]),
                citation_key=citation_key,
                extract_text=True,
                expected_title=expected_title,
                expected_authors=expected_authors,
                expected_year=expected_year,
            )
        if source.get("doi"):
            return self.literature.import_doi(
                str(source["doi"]),
                citation_key=citation_key,
                extract_text=True,
                expected_title=expected_title,
                expected_authors=expected_authors,
                expected_year=expected_year,
            )
        if source.get("url"):
            return self.literature.import_url(
                str(source["url"]),
                citation_key=citation_key,
                title=expected_title,
                extract_text=True,
                expected_title=expected_title,
                expected_authors=expected_authors,
                expected_year=expected_year,
            )
        raise ValueError("source has no arxiv_id, doi, or url")

    def _build_audit(
        self,
        task: str,
        source_plan: dict[str, Any],
        import_results: list[dict[str, Any]],
        papers: list[PaperMetadata],
        statements: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        paper_by_key = {paper.citation_key: paper for paper in papers}
        task_terms = _task_terms(task)
        audit_rows: list[dict[str, Any]] = []
        for statement in statements:
            text = str(statement.get("statement_text") or statement.get("original_statement") or "")
            if not _is_relevant_statement(text, task_terms):
                continue
            citation_key = str(statement.get("citation_key") or "")
            paper = paper_by_key.get(citation_key)
            audit_rows.append(
                {
                    "audit_id": _stable_row_id("audit", statement.get("statement_id"), text),
                    "status": "supported",
                    "normalized_claim": _compact_statement(text),
                    "citation_key": citation_key,
                    "title": paper.title if paper else "",
                    "year": paper.year if paper else None,
                    "statement_id": statement.get("statement_id") or "",
                    "support_id": statement.get("support_id") or statement.get("statement_id") or "",
                    "quote_id": statement.get("quote_id") or "",
                    "locator": statement.get("locator") or "",
                    "dimension_regime": _dimension_regime(text),
                    "assumption": _assumption(text),
                    "kind": statement.get("kind") or "",
                    "validated_exact_substring": bool(statement.get("validated_exact_substring")),
                    "created_at": utc_now(),
                }
            )
        audit_rows = _dedupe_rows(audit_rows, key="statement_id")

        gaps: list[dict[str, Any]] = []
        statements_by_key: dict[str, list[dict[str, Any]]] = {}
        for statement in statements:
            statements_by_key.setdefault(str(statement.get("citation_key") or ""), []).append(statement)

        for result in import_results:
            status = str(result.get("status") or "")
            if status not in {"imported", "existing"}:
                gaps.append(
                    {
                        "gap_id": _stable_row_id("gap", result.get("title"), result.get("doi"), result.get("arxiv_id")),
                        "kind": "missing_source",
                        "status": "open",
                        "citation_key": result.get("citation_key") or "",
                        "title": result.get("title") or "",
                        "reason": result.get("reason") or result.get("error") or "source import failed",
                        "follow_up": "Provide a correct DOI/arXiv/PDF URL or manually import the source.",
                        "created_at": utc_now(),
                    }
                )
                continue
            key = str(result.get("citation_key") or "")
            if key and key not in statements_by_key:
                gaps.append(
                    {
                        "gap_id": _stable_row_id("gap", key, "no_statements"),
                        "kind": "missing_statement_extraction",
                        "status": "open",
                        "citation_key": key,
                        "title": result.get("title") or "",
                        "reason": "paper is imported but no theorem/definition/lower-bound statement was extracted",
                        "follow_up": "Inspect the PDF/text and add exact statement quotes manually if needed.",
                        "created_at": utc_now(),
                    }
                )

        all_text = "\n".join(row["normalized_claim"] for row in audit_rows).lower()
        for concept, pattern, follow_up in [
            (
                "OV definition",
                r"orthogonal vectors|\bov\b",
                "Extract an exact definition of Boolean Orthogonal Vectors from a primary source.",
            ),
            (
                "SETH statement",
                r"strong exponential time hypothesis|\bseth\b",
                "Extract the exact SETH formulation used by the imported papers.",
            ),
            (
                "dimension regimes",
                r"logarithmic|polylog|moderate|n\^|dimension",
                "Extract explicit statements separating logarithmic, polylogarithmic, and polynomial/moderate dimensions.",
            ),
            (
                "OV lower-bound implication",
                r"lower bound|no .*algorithm|cannot be solved|conjecture",
                "Import/extract a primary quote supporting the SETH-to-OV lower-bound implication.",
            ),
        ]:
            if not re.search(pattern, all_text):
                gaps.append(
                    {
                        "gap_id": _stable_row_id("gap", concept),
                        "kind": "missing_concept",
                        "status": "open",
                        "citation_key": "",
                        "title": concept,
                        "reason": f"no supported audit row currently covers: {concept}",
                        "follow_up": follow_up,
                        "created_at": utc_now(),
                    }
                )

        # Keep only source-plan relevant imported keys in a metadata row for traceability.
        if source_plan.get("source"):
            for row in audit_rows:
                row["source_plan"] = source_plan["source"]
        return audit_rows, _dedupe_rows(gaps, key="gap_id")


def _normalize_source_plan(raw: dict[str, Any], *, source: str) -> dict[str, Any]:
    required = raw.get("required_sources") or raw.get("sources") or []
    optional = raw.get("optional_sources") or []
    normalized_required = [_normalize_source(item, required=True) for item in required]
    normalized_optional = [_normalize_source(item, required=False) for item in optional]
    return {
        "version": 1,
        "source": raw.get("source") or source,
        "created_at": utc_now(),
        "required_sources": normalized_required,
        "optional_sources": normalized_optional,
        "search_queries": [str(item) for item in raw.get("search_queries") or []],
    }


def _normalize_source(item: dict[str, Any], *, required: bool) -> dict[str, Any]:
    title = str(item.get("title") or "").strip()
    return {
        "citation_key": str(item.get("citation_key") or _citation_key_from_title(title)).strip(),
        "title": title,
        "authors": [str(author) for author in item.get("authors") or []],
        "year": item.get("year"),
        "venue": str(item.get("venue") or ""),
        "doi": str(item.get("doi") or ""),
        "arxiv_id": str(item.get("arxiv_id") or ""),
        "url": str(item.get("url") or ""),
        "required": required,
        "reason_needed": str(item.get("reason_needed") or item.get("reason") or ""),
    }


def _builtin_source_plan_for_task(task: str) -> dict[str, Any]:
    lowered = task.lower()
    if "orthogonal" in lowered and "vector" in lowered and "seth" in lowered:
        return _normalize_source_plan(
            {
                "source": "builtin:ov-seth-literature-audit",
                "required_sources": [
                    {
                        "citation_key": "Williams_2005_2CSP",
                        "title": "A new algorithm for optimal 2-constraint satisfaction and its implications",
                        "authors": ["Ryan Williams"],
                        "year": 2005,
                        "doi": "10.1016/j.tcs.2005.09.023",
                        "reason": "Primary source commonly cited for the SAT/OV connection.",
                    },
                    {
                        "citation_key": "Abboud_Bringmann_Dell_Nederlof_2018",
                        "title": "More Consequences of Falsifying SETH and the Orthogonal Vectors Conjecture",
                        "authors": ["Amir Abboud", "Karl Bringmann", "Holger Dell", "Jesper Nederlof"],
                        "year": 2018,
                        "arxiv_id": "1805.08554",
                        "reason": "Defines and uses the moderate-dimension OV conjecture.",
                    },
                    {
                        "citation_key": "Chen_Williams_2018_OV_Equivalence",
                        "title": "An Equivalence Class for Orthogonal Vectors",
                        "authors": ["Lijie Chen", "Ryan Williams"],
                        "year": 2018,
                        "arxiv_id": "1811.12017",
                        "reason": "Separates sparse/logarithmic and moderate-dimensional OV regimes.",
                    },
                    {
                        "citation_key": "Impagliazzo_Paturi_Zane_2001",
                        "title": "Which Problems Have Strongly Exponential Complexity?",
                        "authors": ["Russell Impagliazzo", "Ramamohan Paturi", "Francis Zane"],
                        "year": 2001,
                        "doi": "10.1006/jcss.2001.1774",
                        "reason": "Primary SETH/IPZ background source.",
                    },
                ],
                "optional_sources": [
                    {
                        "citation_key": "Vassilevska_Williams_ICM_2018",
                        "title": "On some fine-grained questions in algorithms and complexity",
                        "authors": ["Virginia Vassilevska Williams"],
                        "year": 2018,
                        "reason": "Survey context; optional because primary statements should be preferred.",
                    }
                ],
                "search_queries": [
                    "Orthogonal Vectors SETH lower bound Williams 2005",
                    "Orthogonal Vectors Conjecture moderate dimension",
                    "SETH Orthogonal Vectors conjecture fine grained complexity",
                ],
            },
            source="builtin:ov-seth-literature-audit",
        )
    return _normalize_source_plan(
        {
            "source": "builtin:empty-literature-audit",
            "required_sources": [],
            "optional_sources": [],
            "search_queries": [],
        },
        source="builtin:empty-literature-audit",
    )


def _read_flat_statements(store: ArtifactStore) -> list[dict[str, Any]]:
    rows = store.read_jsonl("LiteratureDB/statements.jsonl")
    return list(_latest_by(rows, key="statement_id").values())


def _write_jsonl_overwrite(store: ArtifactStore, path: str, rows: list[dict[str, Any]]) -> ArtifactRef:
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    return store.write_text(path, content)


def _latest_by(rows: list[dict[str, Any]], *, key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if value:
            result[value] = row
    return result


def _dedupe_rows(rows: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    return list(_latest_by(rows, key=key).values())


def _task_terms(task: str) -> set[str]:
    terms = {term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", task)}
    terms.update({"orthogonal", "vectors", "ov", "seth", "dimension", "lower", "bound", "conjecture"})
    return {term for term in terms if term not in _STOPWORDS}


def _is_relevant_statement(text: str, task_terms: set[str]) -> bool:
    lowered = text.lower()
    if any(token in lowered for token in ["orthogonal", " ov", "seth", "dimension", "lower bound"]):
        return True
    statement_terms = set(re.findall(r"[a-z][a-z0-9-]{2,}", lowered))
    return len(statement_terms & task_terms) >= 2


def _compact_statement(text: str, *, limit: int = 900) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _dimension_regime(text: str) -> str:
    lowered = text.lower().replace(" ", "")
    if "logarithmic" in text.lower() or "o(logn)" in lowered or "o(log n)" in lowered or "clogn" in lowered:
        return "logarithmic"
    if "polylog" in lowered or "log^" in lowered or "logc" in lowered:
        return "polylogarithmic"
    if "moderate" in text.lower() or "n^" in lowered or "nδ" in lowered or "n^delta" in lowered:
        return "moderate_or_polynomial"
    if "dimension" in text.lower() or "dimensional" in text.lower():
        return "dimension_mentioned_unspecified"
    return "unspecified"


def _assumption(text: str) -> str:
    lowered = text.lower()
    if "seth" in lowered or "strong exponential time hypothesis" in lowered:
        return "SETH"
    if "ov conjecture" in lowered or "orthogonal vectors conjecture" in lowered:
        return "OV conjecture"
    if "conjecture" in lowered:
        return "conjectural"
    return "none_or_unspecified"


def _citation_key_from_title(title: str) -> str:
    key = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
    return key[:80] or "source"


def _stable_row_id(prefix: str, *parts: Any) -> str:
    import hashlib

    payload = "\x1f".join(str(part or "") for part in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def _render_literature_audit_report(
    *,
    task: str,
    source_plan: dict[str, Any],
    import_results: list[dict[str, Any]],
    extraction_summary: dict[str, Any],
    papers: list[PaperMetadata],
    audit_rows: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# Deterministic Literature Audit",
        "",
        "This report was produced by the deterministic literature-audit pipeline.",
        "Supported rows cite statement/support IDs for later promotion.",
        "",
        "## Source plan",
        f"- Source: `{source_plan.get('source', '')}`",
        f"- Required sources: {len(source_plan.get('required_sources', []))}",
        f"- Optional sources: {len(source_plan.get('optional_sources', []))}",
        "",
        "## Import results",
    ]
    if import_results:
        for result in import_results:
            title = result.get("title") or result.get("citation_key") or "<untitled>"
            status = result.get("status") or "unknown"
            detail = result.get("error") or result.get("reason") or result.get("paper_id") or ""
            lines.append(f"- **{status}** — {title} {('— ' + detail) if detail else ''}")
    else:
        lines.append("- No source-plan entries were available.")
    lines.extend(
        [
            "",
            "## Extraction summary",
            "```json",
            json.dumps(extraction_summary, indent=2, sort_keys=True),
            "```",
            "",
            "## Supported claims / extracted statements",
        ]
    )
    if audit_rows:
        for row in audit_rows:
            lines.extend(
                [
                    f"### {row.get('citation_key', '')}: {row.get('label', row.get('kind', 'statement'))}",
                    f"- Status: `{row['status']}`",
                    f"- Statement/support: `{row.get('statement_id', '')}` / `{row.get('support_id', '')}`",
                    f"- Quote: `{row.get('quote_id', '')}` at {row.get('locator', '')}",
                    f"- Assumption: {row.get('assumption', '')}",
                    f"- Dimension regime: {row.get('dimension_regime', '')}",
                    f"- Exact quote span validated: {row.get('validated_exact_substring')}",
                    "",
                    row.get("normalized_claim", ""),
                    "",
                ]
            )
    else:
        lines.append("No supported rows were extracted.")
    lines.extend(["", "## Unsupported / gap claims"])
    if gaps:
        for gap in gaps:
            lines.extend(
                [
                    f"- **{gap.get('kind', 'gap')}** `{gap.get('gap_id', '')}`: {gap.get('title') or gap.get('citation_key') or 'gap'}",
                    f"  - Reason: {gap.get('reason', '')}",
                    f"  - Follow-up: {gap.get('follow_up', '')}",
                ]
            )
    else:
        lines.append("No gaps were recorded.")
    lines.extend(
        [
            "",
            "## Imported papers currently in LiteratureDB",
        ]
    )
    for paper in papers:
        lines.append(f"- `{paper.citation_key}` — {paper.title} ({paper.year or 'n.d.'})")
    lines.extend(
        [
            "",
            "## Files",
            f"- `{SOURCE_PLAN_PATH}`",
            "- `LiteratureDB/statements.jsonl`",
            f"- `{AUDIT_TABLE_PATH}`",
            f"- `{GAPS_PATH}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "bound",
    "bounds",
    "claim",
    "claims",
    "for",
    "from",
    "into",
    "lower",
    "paper",
    "papers",
    "problem",
    "produce",
    "research",
    "should",
    "source",
    "sources",
    "that",
    "the",
    "this",
    "with",
}
