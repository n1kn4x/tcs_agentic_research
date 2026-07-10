"""Proposal generation and proposal critic agents."""

from __future__ import annotations

import json
from typing import Any, Literal

from ..artifact_store import ArtifactStore
from ..llm import LLMRouter, openai_tool_from_schema
from ..prompt_loader import render_prompt
from ..prompt_serialization import compact_json_dumps
from ..render import render_proposal_markdown
from ..schemas import (
    ArtifactRef,
    CriticDecision,
    LiteratureCandidate,
    LiteratureQueryAnswer,
    PaperMetadata,
    ProposalCritique,
    ProposalLedgerEntry,
    ProposalRisk,
    ResearchProposal,
    ResearchState,
    StrictModel,
)
from .literature import LiteratureResearcher


FINAL_PROPOSAL_TOOL_NAME = "submit_research_proposal"


class ProposalQueryLiteratureArgs(StrictModel):
    query: str


class ProposalSearchPapersArgs(StrictModel):
    query: str


class ProposalImportUrlArgs(StrictModel):
    url: str
    extract_text: bool = True


class ProposalImportArxivArgs(StrictModel):
    arxiv_id: str
    extract_text: bool = True


class ProposalImportDoiArgs(StrictModel):
    doi: str
    extract_text: bool = True


class ProposalImportCandidateArgs(StrictModel):
    candidate_id: str
    extract_text: bool = True


class ProposalAgent:
    def __init__(self, store: ArtifactStore, router: LLMRouter, *, prompt_dir: str | None = None):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir
        self.literature = LiteratureResearcher(store, router, prompt_dir=prompt_dir)

    def generate_and_review(
        self,
        state: ResearchState,
        *,
        max_revisions: int = 2,
    ) -> tuple[ResearchProposal, ProposalCritique, str]:
        task = self.store.read_text(ArtifactStore.RESEARCH_TASK)
        context = self._context_payload(state, task)
        iteration = state.iteration + 1
        iteration_dir = self.store.create_iteration_dir(iteration)
        critique: ProposalCritique | None = None
        proposal = self._mock_proposal(state)

        for attempt in range(max_revisions + 1):
            proposal = self._generate_proposal(
                context=context,
                iteration_dir=iteration_dir,
                attempt=attempt,
                previous_critique=critique,
                dry_run_seed=proposal,
            )
            proposal_ref = self._write_proposal_artifacts(iteration_dir, proposal)
            event_refs = [proposal_ref]
            self._record_proposal_event(
                proposal_id=proposal.proposal_id,
                event_type="generated" if attempt == 0 else "revised",
                artifact_refs=event_refs,
                proposal=proposal,
            )

            critique = self._review_proposal(context, proposal)
            self._record_proposal_event(
                proposal_id=proposal.proposal_id,
                event_type="critic_review",
                artifact_refs=event_refs,
                critique=critique,
                reason=critique.summary,
            )

            if critique.decision == CriticDecision.accept:
                return self._accept_proposal(
                    state,
                    iteration,
                    proposal,
                    critique,
                    proposal_ref,
                    reason="Accepted by proposal critic.",
                )
            if critique.decision == CriticDecision.reject:
                self._record_proposal_event(
                    proposal_id=proposal.proposal_id,
                    event_type="rejected",
                    artifact_refs=event_refs,
                    proposal=proposal,
                    critique=critique,
                    reason="Rejected by proposal critic.",
                )
                break

        if not self.router.dry_run:
            reason = critique.summary if critique is not None else "No acceptable proposal was produced."
            raise RuntimeError(
                "Proposal generation/revision failed in a real run. "
                f"Last critique: {reason}"
            )

        proposal = self._mock_proposal(state)
        critique = self._mock_critique(proposal)
        proposal_ref = self._write_proposal_artifacts(iteration_dir, proposal)
        return self._accept_proposal(
            state,
            iteration,
            proposal,
            critique,
            proposal_ref,
            reason="Accepted dry-run mock proposal after failed proposal revisions.",
        )

    def _generate_proposal(
        self,
        *,
        context: Any,
        iteration_dir: str,
        attempt: int,
        previous_critique: ProposalCritique | None,
        dry_run_seed: ResearchProposal,
    ) -> ResearchProposal:
        prompt_payload = {
            "proposal_generation_attempt": attempt + 1,
            "instruction": (
                "Think privately. Use native tool calls when they materially improve "
                "the proposal. Finish by calling the final proposal-submission tool."
            ),
            "context": context,
            "previous_critique": previous_critique.model_dump(mode="json")
            if previous_critique
            else None,
        }
        messages = [
            {
                "role": "system",
                "content": render_prompt("proposal_generator", override_dir=self.prompt_dir),
            },
            {"role": "user", "content": compact_json_dumps(prompt_payload)},
        ]
        proposal, trace = self.router.complete_structured_with_tools(
            task_type="proposal_generation",
            messages=messages,
            tools=self._proposal_tools(),
            tool_executors=self._proposal_tool_executors(),
            schema=ResearchProposal,
            final_tool_name=FINAL_PROPOSAL_TOOL_NAME,
            mock_output=dry_run_seed if self.router.dry_run else None,
        )
        self._write_proposal_tool_trace(
            iteration_dir,
            attempt=attempt,
            proposal=proposal,
            trace=trace,
        )
        return proposal

    def _proposal_tools(self) -> list[dict[str, Any]]:
        return [
            openai_tool_from_schema(
                "query_literature",
                (
                    "Query the local LiteratureDB in canonical notation. Use this before "
                    "relying on prior work, barriers, or known results."
                ),
                ProposalQueryLiteratureArgs,
                strip_system_owned_fields=False,
            ),
            openai_tool_from_schema(
                "search_papers",
                (
                    "Search external paper metadata and queue candidate papers for possible "
                    "import. This does not certify claims."
                ),
                ProposalSearchPapersArgs,
                strip_system_owned_fields=False,
            ),
            openai_tool_from_schema(
                "import_url",
                "Import a useful paper from a URL or PDF URL into LiteratureDB.",
                ProposalImportUrlArgs,
                strip_system_owned_fields=False,
            ),
            openai_tool_from_schema(
                "import_arxiv",
                "Import an arXiv paper into LiteratureDB.",
                ProposalImportArxivArgs,
                strip_system_owned_fields=False,
            ),
            openai_tool_from_schema(
                "import_doi",
                "Import a DOI into LiteratureDB.",
                ProposalImportDoiArgs,
                strip_system_owned_fields=False,
            ),
            openai_tool_from_schema(
                "import_candidate",
                "Import a previously queued literature candidate into LiteratureDB.",
                ProposalImportCandidateArgs,
                strip_system_owned_fields=False,
            ),
            openai_tool_from_schema(
                FINAL_PROPOSAL_TOOL_NAME,
                (
                    "Commit the final concrete research proposal. The arguments must be the "
                    "ResearchProposal object itself, not wrapped under another key."
                ),
                ResearchProposal,
            ),
        ]

    def _proposal_tool_executors(self) -> dict[str, Any]:
        return {
            "query_literature": self._tool_query_literature,
            "search_papers": self._tool_search_papers,
            "import_url": self._tool_import_url,
            "import_arxiv": self._tool_import_arxiv,
            "import_doi": self._tool_import_doi,
            "import_candidate": self._tool_import_candidate,
        }

    def _tool_query_literature(self, arguments: dict[str, Any]) -> dict[str, Any]:
        args = ProposalQueryLiteratureArgs.model_validate(arguments)
        answer = self.literature.answer_query(args.query, limit=5)
        return _compact_literature_answer(
            answer,
            ledger_ref=self.store.artifact_ref("LiteratureDB/query_answers.jsonl"),
        )

    def _tool_search_papers(self, arguments: dict[str, Any]) -> dict[str, Any]:
        args = ProposalSearchPapersArgs.model_validate(arguments)
        candidates = self.literature.search_papers(args.query, limit=8)
        return {
            "status": "ok",
            "tool": "search_papers",
            "query": args.query,
            "candidate_count": len(candidates),
            "candidates": [_compact_candidate(candidate) for candidate in candidates],
            "ledger_ref": self.store.artifact_ref("LiteratureDB/candidates.jsonl").model_dump(
                mode="json"
            ),
        }

    def _tool_import_url(self, arguments: dict[str, Any]) -> dict[str, Any]:
        args = ProposalImportUrlArgs.model_validate(arguments)
        paper = self.literature.import_url(args.url, extract_text=args.extract_text)
        return _compact_imported_paper("import_url", paper)

    def _tool_import_arxiv(self, arguments: dict[str, Any]) -> dict[str, Any]:
        args = ProposalImportArxivArgs.model_validate(arguments)
        paper = self.literature.import_arxiv(args.arxiv_id, extract_text=args.extract_text)
        return _compact_imported_paper("import_arxiv", paper)

    def _tool_import_doi(self, arguments: dict[str, Any]) -> dict[str, Any]:
        args = ProposalImportDoiArgs.model_validate(arguments)
        paper = self.literature.import_doi(args.doi, extract_text=args.extract_text)
        return _compact_imported_paper("import_doi", paper)

    def _tool_import_candidate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        args = ProposalImportCandidateArgs.model_validate(arguments)
        paper = self.literature.import_candidate(
            args.candidate_id,
            extract_text=args.extract_text,
        )
        return _compact_imported_paper("import_candidate", paper)

    def _review_proposal(self, context: Any, proposal: ResearchProposal) -> ProposalCritique:
        prompt_payload = {
            "task_and_state": context,
            "proposal": proposal.model_dump(mode="json"),
        }
        messages = [
            {
                "role": "system",
                "content": render_prompt("proposal_critic", override_dir=self.prompt_dir),
            },
            {
                "role": "user",
                "content": "Review this proposal against the task/state payload.\n"
                + compact_json_dumps(prompt_payload),
            },
        ]
        return self.router.complete_structured(
            task_type="proposal_critique",
            messages=messages,
            schema=ProposalCritique,
            mock_output=self._mock_critique(proposal) if self.router.dry_run else None,
        )

    def _write_proposal_artifacts(
        self, iteration_dir: str, proposal: ResearchProposal
    ) -> ArtifactRef:
        proposal_ref = self.store.write_json(
            f"{iteration_dir}/proposal_{proposal.proposal_id}.json", proposal
        )
        self.store.write_text(
            f"{iteration_dir}/proposal_{proposal.proposal_id}.md",
            render_proposal_markdown(proposal),
        )
        return proposal_ref

    def _write_proposal_tool_trace(
        self,
        iteration_dir: str,
        *,
        attempt: int,
        proposal: ResearchProposal,
        trace: dict[str, Any],
    ) -> list[ArtifactRef]:
        """Write an operational tool trace, without adding it to future prompt context."""
        payload = {
            "proposal_id": proposal.proposal_id,
            "attempt": attempt + 1,
            "private_reasoning": "redacted_not_logged_or_replayed",
            "trace": trace,
        }
        json_ref = self.store.write_json(
            f"{iteration_dir}/proposal_tool_trace_{proposal.proposal_id}.json",
            payload,
        )
        lines = [f"# Proposal Tool Trace `{proposal.proposal_id}`", ""]
        lines.extend(
            [
                "Private chain-of-thought/reasoning is intentionally not logged.",
                "Only external tool calls, arguments, observations, and finalization metadata appear here.",
                "",
            ]
        )
        for item in trace.get("tool_calls", []):
            lines.append(f"## Turn {item.get('turn', '?')}: `{item.get('name', '')}`")
            lines.extend(
                [
                    "",
                    "Arguments:",
                    "",
                    "```json",
                    json.dumps(item.get("arguments", {}), indent=2, sort_keys=True),
                    "```",
                    "",
                    "Observation:",
                    "",
                    "```json",
                    json.dumps(item.get("observation", {}), indent=2, sort_keys=True),
                    "```",
                    "",
                ]
            )
        if trace.get("finalization"):
            lines.extend(
                [
                    "## Finalization",
                    "",
                    "```json",
                    json.dumps(trace["finalization"], indent=2, sort_keys=True),
                    "```",
                    "",
                ]
            )
        md_ref = self.store.write_text(
            f"{iteration_dir}/proposal_tool_trace_{proposal.proposal_id}.md",
            "\n".join(lines).rstrip() + "\n",
        )
        return [json_ref, md_ref]

    def _record_proposal_event(
        self,
        *,
        proposal_id: str,
        event_type: Literal["generated", "revised", "accepted", "rejected", "critic_review"],
        artifact_refs: list[ArtifactRef],
        proposal: ResearchProposal | None = None,
        critique: ProposalCritique | None = None,
        reason: str = "",
    ) -> None:
        self.store.append_proposal_event(
            ProposalLedgerEntry(
                proposal_id=proposal_id,
                event_type=event_type,
                proposal=proposal,
                critique=critique,
                reason=reason,
                artifact_refs=artifact_refs,
            )
        )

    def _accept_proposal(
        self,
        state: ResearchState,
        iteration: int,
        proposal: ResearchProposal,
        critique: ProposalCritique,
        proposal_ref: ArtifactRef,
        *,
        reason: str,
    ) -> tuple[ResearchProposal, ProposalCritique, str]:
        self._record_proposal_event(
            proposal_id=proposal.proposal_id,
            event_type="accepted",
            artifact_refs=[proposal_ref],
            proposal=proposal,
            critique=critique,
            reason=reason,
        )
        state.iteration = iteration
        state.current_proposal_id = proposal.proposal_id
        state.artifact_refs.append(proposal_ref)
        self.store.save_state(state)
        return proposal, critique, proposal_ref.path

    def _context_payload(self, state: ResearchState, task: str) -> dict[str, object]:
        claim_tail = self.store.read_jsonl(ArtifactStore.CLAIM_LEDGER, limit=20)
        proposal_tail = self.store.read_jsonl(ArtifactStore.PROPOSAL_LEDGER, limit=20)
        return {
            "research_task_md": task,
            "research_state": state.model_dump(mode="json"),
            "recent_claim_ledger_entries": claim_tail,
            "recent_proposal_ledger_entries": proposal_tail,
            "literature_papers": self.store.read_jsonl("LiteratureDB/papers.jsonl", limit=20),
            "literature_candidates": self.store.read_jsonl(
                "LiteratureDB/candidates.jsonl", limit=20
            ),
            "recent_literature_query_answers": self.store.read_jsonl(
                "LiteratureDB/query_answers.jsonl", limit=20
            ),
        }

    def _mock_proposal(self, state: ResearchState) -> ResearchProposal:
        return ResearchProposal(
            title="Audit-first scoping pass for literature and formalizable subclaims",
            precise_goal=(
                "Identify one precise, low-risk path for progress by auditing relevant literature, "
                "known barriers, complexity models, and formalizable definitions before attempting any "
                "breakthrough claim."
            ),
            relevant_assumptions_and_model=[
                "Use the computational model and assumptions in ResearchTask.md.",
                "Do not introduce stronger assumptions without explicit ledger entries and critic review.",
            ],
            expected_intermediate_lemmas=[
                "A normalized statement of the central problem in the project nomenclature.",
                "A list of lower-bound or no-go results from the literature that constrain the attempted improvement.",
            ],
            algorithmic_subgoals=[
                "Define baseline algorithms and resource measures for comparison.",
                "If applicable, design small-instance experiments only as conjecture generators.",
            ],
            plausibility_argument=(
                "Hard TCS tasks often fail because of hidden model changes, duplicate literature, or "
                "implicit complexity costs. An audit-first pass increases correctness and can produce useful "
                "formalization targets."
            ),
            success_criteria=[
                "A structured report with claims classified as cited, conjectural, informal, or needing proof.",
                "At least one concrete next research proposal or vetted barrier with provenance.",
            ],
            partial_success_criteria=[
                "Updated nomenclature mappings.",
                "Open proof obligations for LEAP.",
                "Complexity-derivation checklist for subsequent algorithm claims.",
            ],
            required_tools=[
                "literature_search",
                "theorem_prover_if_formalizable",
                "experiment_runner_if_needed",
            ],
            known_risks_and_barriers=[
                ProposalRisk(
                    risk="The pass may not produce a new theorem.",
                    mitigation="Treat literature-vetted barriers and formalization targets as valid partial progress.",
                    severity="low",
                )
            ],
            literature_queries=[
                "known best algorithms and lower bounds for the task",
                "quantum speedups and query lower bounds relevant to the model",
                "complexity-theoretic barriers and reductions",
            ],
            resource_model="Use explicit asymptotic time, space, query, circuit, proof-size, and quantum resources when relevant.",
        )

    def _mock_critique(self, proposal: ResearchProposal) -> ProposalCritique:
        return ProposalCritique(
            decision=CriticDecision.accept,
            summary="Dry-run mock proposal is conservative, auditable, and suitable for bootstrapping.",
            consistency_with_task="It preserves the task assumptions and asks for provenance before strong claims.",
            plausibility="High as a scoping and verification pass, not as a claimed breakthrough.",
            barrier_risks=[],
            missing_complexity_model=[]
            if proposal.resource_model
            else ["Complexity/resource model should be explicit."],
            unclear_success_criteria=[] if proposal.success_criteria else ["Success criteria missing."],
            required_revisions=[],
            confidence=0.7,
        )


def _compact_literature_answer(
    answer: LiteratureQueryAnswer, *, ledger_ref: ArtifactRef
) -> dict[str, Any]:
    return {
        "status": "ok",
        "tool": "query_literature",
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
    return {
        "status": "ok",
        "tool": tool,
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
            "artifact_refs": [ref.model_dump(mode="json") for ref in paper.artifact_refs],
        },
    }


def _compact_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n...[truncated {omitted} characters]"
