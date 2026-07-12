"""Reusable per-agent toolsets for native OpenAI/vLLM tool calling."""

from __future__ import annotations

import re
from typing import Any

from pydantic import Field

from ..artifact_store import ArtifactStore, to_plain
from ..prompt_serialization import compact_json_dumps
from ..schemas import (
    ArtifactRef,
    LeanStatement,
    LiteratureCandidate,
    LiteratureQueryAnswer,
    PaperMetadata,
    ResearchReport,
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


def literature_toolset(
    *,
    store: ArtifactStore,
    literature: LiteratureResearcher,
    include_discovery_tools: bool = True,
) -> Toolset:
    """Build literature tools for a single agent call.

    The caller chooses whether discovery/import tools are visible. This keeps the interface
    unified while preserving per-agent access control.
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
                "may be referenced in EvidenceRecord.tool_result_ids."
            ),
            QueryLiteratureArgs,
            query_literature,
            strip_system_owned_fields=False,
        )
    ]

    if not include_discovery_tools:
        return Toolset(tools)

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
        return _compact_imported_paper("import_url", paper)

    def import_arxiv(arguments: dict[str, Any]) -> dict[str, Any]:
        args = ImportArxivArgs.model_validate(arguments)
        paper = literature.import_arxiv(args.arxiv_id, extract_text=args.extract_text)
        return _compact_imported_paper("import_arxiv", paper)

    def import_doi(arguments: dict[str, Any]) -> dict[str, Any]:
        args = ImportDoiArgs.model_validate(arguments)
        paper = literature.import_doi(args.doi, extract_text=args.extract_text)
        return _compact_imported_paper("import_doi", paper)

    def import_candidate(arguments: dict[str, Any]) -> dict[str, Any]:
        args = ImportCandidateArgs.model_validate(arguments)
        paper = literature.import_candidate(args.candidate_id, extract_text=args.extract_text)
        return _compact_imported_paper("import_candidate", paper)

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
                "If you use this result in the final report, put the tool_result_id in "
                "EvidenceRecord.tool_result_ids. A proved result can support matching Lean "
                "proof evidence; partial/failed results should be reported as unresolved work."
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
                "If you use this result in the final report, include this ExperimentResult in "
                "ResearchReport.experimental_results and put the tool_result_id in supporting "
                "EvidenceRecord.tool_result_ids. Experiments support empirical claims only; "
                "they do not prove mathematical claims."
            ),
        }

    literature_tools = list(
        literature_toolset(
            store=store,
            literature=literature,
            include_discovery_tools=False,
        )
    )
    return Toolset(
        [
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
                "submit_research_report",
                (
                    "Commit the final structured ResearchReport. The arguments must be the "
                    "ResearchReport object itself, not wrapped under another key. Reference any "
                    "used tool_result_id values in EvidenceRecord.tool_result_ids."
                ),
                ResearchReport,
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
                "duplicate_of": result.duplicate_of,
                "provenance": [
                    {
                        "locator": quote.locator,
                        "quote_excerpt": _compact_text(quote.quote, 500),
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
            "If a final report uses this literature query, put answer_id/tool_result_id in "
            "EvidenceRecord.tool_result_ids and cite returned citation_key values."
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


def _compact_imported_paper(tool: str, paper: PaperMetadata) -> dict[str, Any]:
    refs = [ref.model_dump(mode="json") for ref in paper.artifact_refs]
    return {
        "status": "ok",
        "tool": tool,
        "tool_result_id": paper.paper_id,
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
