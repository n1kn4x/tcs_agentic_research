"""Reusable per-agent toolsets for native OpenAI/vLLM tool calling."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from ..artifact_store import ArtifactStore, to_plain
from ..prompt_serialization import compact_json_dumps
from ..schemas import (
    ArtifactRef,
    LeanStatement,
    LiteratureCandidate,
    LiteratureExtract,
    LiteratureQueryAnswer,
    PaperMetadata,
    ResearchReportSubmission,
    StrictModel,
)
from ..tooling import AgentTool, Toolset, final_submission_tool
from .experiment import ExperimentAgent
from .literature import LiteratureResearcher
from .theorem_prover import TheoremProverAgent


# ---------------------------------------------------------------------------
# Shared literature tools
# ---------------------------------------------------------------------------


class QueryLiteratureArgs(StrictModel):
    query: str


class SearchPapersArgs(StrictModel):
    query: str


class ImportUrlArgs(StrictModel):
    url: str
    extract_text: bool = True


class ImportArxivArgs(StrictModel):
    arxiv_id: str
    extract_text: bool = True


class ImportDoiArgs(StrictModel):
    doi: str
    extract_text: bool = True


class ImportCandidateArgs(StrictModel):
    candidate_id: str
    extract_text: bool = True


class ExtractPaperArgs(StrictModel):
    citation_key: str = ""
    paper_id: str = ""


class ExtractImportedPapersArgs(StrictModel):
    max_papers: int = Field(default=8, ge=1, le=50)
    only_missing: bool = True


class AttemptLeanProofArgs(StrictModel):
    statement: str
    name: str = ""
    context: str = ""
    supports_claim_statement: str = ""


class RunExperimentArgs(StrictModel):
    description: str
    name: str = "experiment"
    supports_claim_ids: list[str] = Field(default_factory=list)
    timeout_seconds: int | None = None


class ReadArtifactArgs(StrictModel):
    path: str
    offset: int = Field(default=0, ge=0)
    max_chars: int = Field(default=8000, ge=1, le=30000)


class ReadJsonlRecordsArgs(StrictModel):
    path: str
    limit: int = Field(default=10, ge=1, le=50)
    id_field: str | None = None
    id_value: str | None = None
    offset: int = Field(default=0, ge=0)
    max_chars: int = Field(default=8000, ge=1, le=30000)


# ---------------------------------------------------------------------------
# Shared artifact retrieval tools
# ---------------------------------------------------------------------------


def artifact_retrieval_toolset(*, store: ArtifactStore) -> Toolset:
    """Build simple workspace-memory tools backed by canonical artifacts."""

    def read_artifact(arguments: dict[str, Any]) -> dict[str, Any]:
        args = ReadArtifactArgs.model_validate(arguments)
        try:
            target = store.resolve(args.path)
        except Exception as exc:  # noqa: BLE001 - return as tool observation
            return {
                "status": "error",
                "tool": "read_artifact",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        if not target.exists():
            return {
                "status": "error",
                "tool": "read_artifact",
                "path": args.path,
                "error_type": "FileNotFoundError",
                "error": "Artifact does not exist.",
            }
        if not target.is_file():
            return {
                "status": "error",
                "tool": "read_artifact",
                "path": store.relpath(target),
                "error_type": "IsADirectoryError",
                "error": "Artifact path is not a file.",
            }
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return {
                "status": "error",
                "tool": "read_artifact",
                "path": store.relpath(target),
                "error_type": "UnicodeDecodeError",
                "error": "Artifact is not UTF-8 text; inspect metadata or a text extraction artifact instead.",
                "artifact_ref": store.artifact_ref(target).model_dump(mode="json"),
            }
        start = min(args.offset, len(text))
        end = min(start + args.max_chars, len(text))
        return {
            "status": "ok",
            "tool": "read_artifact",
            "path": store.relpath(target),
            "offset": start,
            "max_chars": args.max_chars,
            "size_chars": len(text),
            "content": text[start:end],
            "truncated": end < len(text),
            "next_offset": end if end < len(text) else None,
            "artifact_ref": store.artifact_ref(target).model_dump(mode="json"),
            "instruction": "Use next_offset to continue reading this artifact if truncated.",
        }

    def read_jsonl_records(arguments: dict[str, Any]) -> dict[str, Any]:
        args = ReadJsonlRecordsArgs.model_validate(arguments)
        try:
            target = store.resolve(args.path)
        except Exception as exc:  # noqa: BLE001 - return as tool observation
            return {
                "status": "error",
                "tool": "read_jsonl_records",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        if not target.exists() or not target.is_file():
            return {
                "status": "error",
                "tool": "read_jsonl_records",
                "path": args.path,
                "error_type": "FileNotFoundError",
                "error": "JSONL artifact does not exist or is not a file.",
            }
        matches: list[dict[str, Any]] = []
        malformed_lines: list[int] = []
        for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines.append(line_number)
                continue
            if args.id_field and args.id_value is not None:
                if str(record.get(args.id_field)) != args.id_value:
                    continue
            matches.append({"line_number": line_number, "record": record})
        selected = matches[: args.limit] if args.id_field and args.id_value is not None else matches[-args.limit :]
        serialized = json.dumps(selected, indent=2, sort_keys=True, ensure_ascii=False)
        start = min(args.offset, len(serialized))
        end = min(start + args.max_chars, len(serialized))
        return {
            "status": "ok",
            "tool": "read_jsonl_records",
            "path": store.relpath(target),
            "id_filter": {"field": args.id_field, "value": args.id_value}
            if args.id_field and args.id_value is not None
            else None,
            "matching_record_count": len(matches),
            "returned_record_count": len(selected),
            "malformed_line_numbers": malformed_lines[:20],
            "offset": start,
            "max_chars": args.max_chars,
            "size_chars": len(serialized),
            "content": serialized[start:end],
            "truncated": end < len(serialized),
            "next_offset": end if end < len(serialized) else None,
            "artifact_ref": store.artifact_ref(target).model_dump(mode="json"),
            "instruction": "Use next_offset to continue reading the selected JSONL records if truncated.",
        }

    return Toolset(
        [
            AgentTool(
                "read_artifact",
                (
                    "Read a UTF-8 text artifact from the research workspace by path. "
                    "Use this to inspect durable workspace memory listed in artifact_manifest."
                ),
                ReadArtifactArgs,
                read_artifact,
                strip_system_owned_fields=False,
            ),
            AgentTool(
                "read_jsonl_records",
                (
                    "Read selected records from a JSONL artifact by path. Without an id filter, "
                    "returns the last records. With id_field/id_value, returns matching records."
                ),
                ReadJsonlRecordsArgs,
                read_jsonl_records,
                strip_system_owned_fields=False,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Shared literature tools
# ---------------------------------------------------------------------------


def literature_toolset(
    *,
    store: ArtifactStore,
    literature: LiteratureResearcher,
    include_discovery_tools: bool = True,
    include_extraction_tools: bool = False,
    auto_extract_after_import: bool = False,
) -> Toolset:
    """Build literature tools for a single agent call.

    The caller chooses whether discovery/import/extraction tools are visible. This keeps the
    interface unified while preserving per-agent access control.  For literature obligations,
    ``auto_extract_after_import`` should be enabled so newly imported full-text papers are
    immediately statement-extracted instead of sitting idle in LiteratureDB.
    """

    def query_literature(arguments: dict[str, Any]) -> dict[str, Any]:
        args = QueryLiteratureArgs.model_validate(arguments)
        answer = literature.answer_query(args.query, limit=5)
        return _compact_literature_answer(
            answer,
            ledger_ref=store.artifact_ref("LiteratureDB/query_answers.jsonl"),
        )

    tools: list[AgentTool] = [
        AgentTool(
            "query_literature",
            (
                "Query the local LiteratureDB in canonical notation. Use this before relying "
                "on prior work, barriers, novelty, or known results. Returned result handles "
                "may be referenced in final-submission tool_result_ids."
            ),
            QueryLiteratureArgs,
            query_literature,
            strip_system_owned_fields=False,
        )
    ]

    def _maybe_extract_imported_paper(tool_name: str, paper: PaperMetadata, *, requested: bool) -> dict[str, Any]:
        extraction: LiteratureExtract | None = None
        extraction_error: str = ""
        if auto_extract_after_import and requested and paper.text_path:
            try:
                extraction = literature.extract_paper(citation_key=paper.citation_key, use_llm=False)
            except Exception as exc:  # noqa: BLE001 - preserve the successful import observation
                extraction_error = f"{type(exc).__name__}: {exc}"
        return _compact_imported_paper(
            tool_name,
            paper,
            extraction=extraction,
            extraction_error=extraction_error,
            extracted_claims_ref=store.artifact_ref("LiteratureDB/extracted_claims.jsonl")
            if extraction is not None
            else None,
        )

    if include_discovery_tools:

        def search_papers(arguments: dict[str, Any]) -> dict[str, Any]:
            args = SearchPapersArgs.model_validate(arguments)
            candidates = literature.search_papers(args.query, limit=8)
            return {
                "status": "ok",
                "tool": "search_papers",
                "query": args.query,
                "candidate_count": len(candidates),
                "candidates": [_compact_candidate(candidate) for candidate in candidates],
                "ledger_ref": store.artifact_ref("LiteratureDB/candidates.jsonl").model_dump(
                    mode="json"
                ),
            }

        def import_url(arguments: dict[str, Any]) -> dict[str, Any]:
            args = ImportUrlArgs.model_validate(arguments)
            paper = literature.import_url(args.url, extract_text=args.extract_text)
            return _maybe_extract_imported_paper("import_url", paper, requested=args.extract_text)

        def import_arxiv(arguments: dict[str, Any]) -> dict[str, Any]:
            args = ImportArxivArgs.model_validate(arguments)
            paper = literature.import_arxiv(args.arxiv_id, extract_text=args.extract_text)
            return _maybe_extract_imported_paper("import_arxiv", paper, requested=args.extract_text)

        def import_doi(arguments: dict[str, Any]) -> dict[str, Any]:
            args = ImportDoiArgs.model_validate(arguments)
            paper = literature.import_doi(args.doi, extract_text=args.extract_text)
            return _maybe_extract_imported_paper("import_doi", paper, requested=args.extract_text)

        def import_candidate(arguments: dict[str, Any]) -> dict[str, Any]:
            args = ImportCandidateArgs.model_validate(arguments)
            paper = literature.import_candidate(args.candidate_id, extract_text=args.extract_text)
            return _maybe_extract_imported_paper(
                "import_candidate", paper, requested=args.extract_text
            )

        tools.extend(
            [
                AgentTool(
                    "search_papers",
                    (
                        "Search external paper metadata and queue candidate papers for possible "
                        "import. This does not certify claims."
                    ),
                    SearchPapersArgs,
                    search_papers,
                    strip_system_owned_fields=False,
                ),
                AgentTool(
                    "import_url",
                    "Import a useful paper from a URL or PDF URL into LiteratureDB.",
                    ImportUrlArgs,
                    import_url,
                    strip_system_owned_fields=False,
                ),
                AgentTool(
                    "import_arxiv",
                    "Import an arXiv paper into LiteratureDB.",
                    ImportArxivArgs,
                    import_arxiv,
                    strip_system_owned_fields=False,
                ),
                AgentTool(
                    "import_doi",
                    "Import a DOI into LiteratureDB.",
                    ImportDoiArgs,
                    import_doi,
                    strip_system_owned_fields=False,
                ),
                AgentTool(
                    "import_candidate",
                    "Import a previously queued literature candidate into LiteratureDB.",
                    ImportCandidateArgs,
                    import_candidate,
                    strip_system_owned_fields=False,
                ),
            ]
        )

    if include_extraction_tools:

        def extract_paper(arguments: dict[str, Any]) -> dict[str, Any]:
            args = ExtractPaperArgs.model_validate(arguments)
            if not args.citation_key and not args.paper_id:
                raise ValueError("Provide citation_key or paper_id.")
            extract = literature.extract_paper(
                citation_key=args.citation_key or None,
                paper_id=args.paper_id or None,
                use_llm=False,
            )
            return _compact_literature_extract(
                "extract_paper",
                extract,
                extracted_claims_ref=store.artifact_ref("LiteratureDB/extracted_claims.jsonl"),
            )

        def extract_imported_papers(arguments: dict[str, Any]) -> dict[str, Any]:
            args = ExtractImportedPapersArgs.model_validate(arguments)
            summary = literature.extract_imported_papers(
                max_papers=args.max_papers,
                only_missing=args.only_missing,
            )
            summary["tool"] = "extract_imported_papers"
            summary["status"] = "ok"
            summary["ledger_ref"] = store.artifact_ref(
                "LiteratureDB/extracted_claims.jsonl"
            ).model_dump(mode="json")
            summary["instruction"] = (
                "Use returned support_ids and citation_keys in a subsequent query_literature "
                "call or in the final citation evidence handles."
            )
            return summary

        tools.extend(
            [
                AgentTool(
                    "extract_paper",
                    (
                        "Extract theorem/algorithm/lower-bound statements with quote provenance "
                        "from one imported paper, indexing support IDs in LiteratureDB."
                    ),
                    ExtractPaperArgs,
                    extract_paper,
                    strip_system_owned_fields=False,
                ),
                AgentTool(
                    "extract_imported_papers",
                    (
                        "Deterministically extract statements from imported papers that have PDF/text "
                        "artifacts but no extracted statements yet. Use after imports and before "
                        "query_literature."
                    ),
                    ExtractImportedPapersArgs,
                    extract_imported_papers,
                    strip_system_owned_fields=False,
                ),
            ]
        )

    return Toolset(tools)


# ---------------------------------------------------------------------------
# Research execution tools
# ---------------------------------------------------------------------------


def research_execution_toolset(
    *,
    store: ArtifactStore,
    literature: LiteratureResearcher,
    theorem_prover: TheoremProverAgent,
    experiment: ExperimentAgent,
    final_tool_name: str = "submit_research_report",
    final_schema: type[BaseModel] = ResearchReportSubmission,
    final_tool_description: str | None = None,
    include_literature_discovery_tools: bool = False,
    include_literature_extraction_tools: bool = False,
) -> Toolset:
    """Toolset visible to the research agent's native thinking loop."""

    def attempt_lean_proof(arguments: dict[str, Any]) -> dict[str, Any]:
        args = AttemptLeanProofArgs.model_validate(arguments)
        statement = _strip_lean_prefix(args.statement)
        name = _lean_safe_name(args.name or args.supports_claim_statement or statement)
        goal = LeanStatement(name=name, statement=statement)
        context = {
            "tool": "attempt_lean_proof",
            "supports_claim_statement": args.supports_claim_statement,
            "additional_context": args.context,
        }
        result = theorem_prover.prove(goal, context=compact_json_dumps(context))
        refs = _unique_refs([*result.proved_artifacts, *result.artifact_refs])
        return {
            "status": "ok",
            "tool": "attempt_lean_proof",
            "tool_result_id": result.result_id,
            "proof_status": result.status,
            "root_goal": result.root_goal.model_dump(mode="json"),
            "supports_claim_statement": args.supports_claim_statement,
            "proved_artifacts": [ref.model_dump(mode="json") for ref in result.proved_artifacts],
            "artifact_refs": [ref.model_dump(mode="json") for ref in refs],
            "open_goals": [goal.model_dump(mode="json") for goal in result.open_goals[:5]],
            "proof_dag_summary": result.proof_dag_summary,
            "recommended_next_steps": result.recommended_next_steps[:10],
            "instruction": (
                "If you use this result in the final submission, put the tool_result_id in "
                "tool_result_ids. A proved result can support Lean proof evidence; "
                "partial/failed results should be reported as unresolved work."
            ),
        }

    def run_experiment(arguments: dict[str, Any]) -> dict[str, Any]:
        args = RunExperimentArgs.model_validate(arguments)
        result = experiment.run_experiment(
            description=args.description,
            name=args.name,
            supports_claim_ids=args.supports_claim_ids,
            timeout_seconds=args.timeout_seconds,
        )
        return {
            "status": "ok",
            "tool": "run_experiment",
            "tool_result_id": result.run_id,
            "summary": result.summary,
            "description": args.description,
            "experiment_result": result.model_dump(mode="json"),
            "artifact_refs": [ref.model_dump(mode="json") for ref in result.artifact_refs],
            "instruction": (
                "If you use this result in the final submission, put the tool_result_id in "
                "tool_result_ids. The system will attach reproducible run artifacts from the "
                "trace. Experiments support empirical claims only; they do not prove mathematical claims."
            ),
        }

    artifact_tools = list(artifact_retrieval_toolset(store=store))
    literature_tools = list(
        literature_toolset(
            store=store,
            literature=literature,
            include_discovery_tools=include_literature_discovery_tools,
            include_extraction_tools=include_literature_extraction_tools,
            auto_extract_after_import=include_literature_extraction_tools,
        )
    )
    return Toolset(
        [
            *artifact_tools,
            *literature_tools,
            AgentTool(
                "attempt_lean_proof",
                (
                    "Attempt to prove a Lean statement through the LEAP harness. Use for central "
                    "mathematical obligations that are already precise enough to formalize."
                ),
                AttemptLeanProofArgs,
                attempt_lean_proof,
                strip_system_owned_fields=False,
            ),
            AgentTool(
                "run_experiment",
                (
                    "Run a simulation, numerical experiment, or small-instance search in the "
                    "project experimenter: a persistent Docker container running the pi coding "
                    "agent with shell and internet access. The research workspace is mounted "
                    "read-only at /research; /workspace is a writable bind mount backed by "
                    ".experimenter/workspace inside the research workspace, so it is portable "
                    "when the workspace is copied. The system imports completed run artifacts "
                    "into ExperimentRuns/. Missing Docker or experimenter "
                    "configuration is a fatal error, not a blocked placeholder."
                ),
                RunExperimentArgs,
                run_experiment,
                strip_system_owned_fields=False,
            ),
            final_submission_tool(
                final_tool_name,
                final_tool_description
                or (
                    "Commit the final flat research-report submission. Use simple strings/lists "
                    "rather than nested ClaimRecord/EvidenceRecord objects; reference any used "
                    "tool_result_id values in tool_result_ids."
                ),
                final_schema,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Compact observations
# ---------------------------------------------------------------------------


def _compact_literature_answer(
    answer: LiteratureQueryAnswer, *, ledger_ref: ArtifactRef
) -> dict[str, Any]:
    return {
        "status": "ok",
        "tool": "query_literature",
        "tool_result_id": answer.answer_id,
        "query": answer.query,
        "answer_id": answer.answer_id,
        "answer": _compact_text(answer.answer, 2500),
        "result_count": len(answer.results),
        "results": [
            {
                "citation_key": result.citation_key,
                "paper_id": result.paper_id,
                "title": result.title,
                "year": result.year,
                "kind": result.kind,
                "label": result.label,
                "mapped_statement": _compact_text(result.mapped_statement, 1200),
                "summary": _compact_text(result.summary, 800),
                "score": result.score,
                "statement_id": result.statement_id,
                "quote_id": result.quote_id,
                "support_id": result.support_id,
                "support_level": result.support_level,
                "relation": result.relation,
                "duplicate_of": result.duplicate_of,
                "provenance": [
                    {
                        "quote_id": quote.quote_id,
                        "locator": quote.locator,
                        "quote_excerpt": _compact_text(quote.quote, 500),
                        "validated": quote.validated,
                    }
                    for quote in result.provenance[:2]
                ],
            }
            for result in answer.results[:5]
        ],
        "duplicate_results": [group.model_dump(mode="json") for group in answer.duplicate_results[:5]],
        "limitations": answer.limitations[:5],
        "ledger_ref": ledger_ref.model_dump(mode="json"),
        "instruction": (
            "If a final submission uses this literature query, put answer_id/tool_result_id in "
            "tool_result_ids and cite returned support_id values (preferred) plus citation_key values."
        ),
    }


def _compact_candidate(candidate: LiteratureCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "title": candidate.title,
        "authors": candidate.authors[:6],
        "year": candidate.year,
        "venue": candidate.venue,
        "doi": candidate.doi,
        "arxiv_id": candidate.arxiv_id,
        "landing_url": candidate.landing_url,
        "pdf_url": candidate.pdf_url,
        "abstract": _compact_text(candidate.abstract, 1200),
        "cited_by_count": candidate.cited_by_count,
        "discovery_reason": candidate.discovery_reason,
        "status": candidate.status,
        "score": candidate.score,
    }


def _compact_imported_paper(
    tool: str,
    paper: PaperMetadata,
    *,
    extraction: LiteratureExtract | None = None,
    extraction_error: str = "",
    extracted_claims_ref: ArtifactRef | None = None,
) -> dict[str, Any]:
    refs = [ref.model_dump(mode="json") for ref in paper.artifact_refs]
    payload: dict[str, Any] = {
        "status": "ok",
        "tool": tool,
        "tool_result_id": paper.paper_id,
        "citation_keys": [paper.citation_key] if paper.citation_key else [],
        "paper": {
            "paper_id": paper.paper_id,
            "citation_key": paper.citation_key,
            "title": paper.title,
            "authors": paper.authors[:10],
            "year": paper.year,
            "venue": paper.venue,
            "url": paper.url,
            "arxiv_id": paper.arxiv_id,
            "doi": paper.doi,
            "abstract": _compact_text(paper.abstract, 1200),
            "pdf_path": paper.pdf_path,
            "text_path": paper.text_path,
            "metadata_path": paper.metadata_path,
            "artifact_refs": refs,
        },
    }
    if extraction is not None:
        payload["extraction"] = _compact_literature_extract_payload(extraction)
        if extracted_claims_ref is not None:
            payload["extraction"]["ledger_ref"] = extracted_claims_ref.model_dump(mode="json")
    if extraction_error:
        payload["extraction_error"] = extraction_error
    return payload


def _compact_literature_extract(
    tool: str,
    extract: LiteratureExtract,
    *,
    extracted_claims_ref: ArtifactRef | None = None,
) -> dict[str, Any]:
    payload = _compact_literature_extract_payload(extract)
    payload.update(
        {
            "status": "ok",
            "tool": tool,
            "tool_result_id": extract.extract_id,
        }
    )
    if extracted_claims_ref is not None:
        payload["ledger_ref"] = extracted_claims_ref.model_dump(mode="json")
    return payload


def _compact_literature_extract_payload(extract: LiteratureExtract) -> dict[str, Any]:
    statements = [
        *extract.theorem_statements,
        *extract.algorithm_statements,
        *extract.lower_bound_statements,
    ]
    support_ids = [statement.statement_id for statement in statements if statement.statement_id]
    return {
        "extract_id": extract.extract_id,
        "citation_key": extract.citation_key,
        "citation_keys": [extract.citation_key] if extract.citation_key else [],
        "paper_id": extract.paper_id,
        "support_ids": support_ids,
        "theorem_count": len(extract.theorem_statements),
        "algorithm_count": len(extract.algorithm_statements),
        "lower_bound_count": len(extract.lower_bound_statements),
        "claim_count": len(extract.extracted_claims),
        "text_artifact_ref": extract.text_artifact_ref.model_dump(mode="json")
        if extract.text_artifact_ref
        else None,
        "statements": [
            {
                "statement_id": statement.statement_id,
                "support_id": statement.statement_id,
                "kind": statement.kind,
                "label": statement.label,
                "mapped_statement": _compact_text(statement.mapped_statement, 900),
                "quote_id": statement.provenance[0].quote_id if statement.provenance else "",
                "locator": statement.provenance[0].locator if statement.provenance else "",
            }
            for statement in statements[:8]
        ],
    }


def _strip_lean_prefix(statement: str) -> str:
    text = statement.strip()
    for prefix in ["lean:", "Lean:", "LEAN:"]:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def _lean_safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_']+", "_", value).strip("_")
    if not name or not re.match(r"[A-Za-z_]", name):
        name = "goal_" + name
    return name[:80]


def _unique_refs(refs: list[ArtifactRef]) -> list[ArtifactRef]:
    unique: list[ArtifactRef] = []
    seen: set[str] = set()
    for ref in refs:
        if ref.path in seen:
            continue
        seen.add(ref.path)
        unique.append(ref)
    return unique


def _compact_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n...[truncated {omitted} characters]"


def artifact_refs_from_observation(observation: Any) -> list[ArtifactRef]:
    """Best-effort extraction of real artifact refs from a tool observation."""

    payload = to_plain(observation)
    refs: list[ArtifactRef] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("path"), str):
                try:
                    refs.append(ArtifactRef.model_validate(node))
                except Exception:
                    pass
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return _unique_refs(refs)
