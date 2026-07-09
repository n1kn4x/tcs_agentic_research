"""Proposal generation and proposal critic agents."""

from __future__ import annotations

import json
from typing import Any, Literal

from ..artifact_store import ArtifactStore
from ..llm import LLMRouter
from ..prompt_loader import render_prompt
from ..prompt_serialization import compact_json_dumps
from ..render import render_proposal_markdown
from ..schemas import (
    ArtifactRef,
    CriticDecision,
    ProposalCritique,
    ProposalLedgerEntry,
    ProposalLoopAction,
    ProposalRisk,
    ResearchProposal,
    ResearchState,
)
from .literature import LiteratureResearcher


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
        max_thinking_loop_rounds: int = 15,
    ) -> tuple[ResearchProposal, ProposalCritique, str]:
        task = self.store.read_text(ArtifactStore.RESEARCH_TASK)
        context = self._context_payload(state, task)
        iteration = state.iteration + 1
        iteration_dir = self.store.create_iteration_dir(iteration)
        critique: ProposalCritique | None = None
        proposal = self._mock_proposal(state)

        for attempt in range(max_revisions + 1):
            proposal, trace_refs = self._generate_proposal(
                context=context,
                iteration_dir=iteration_dir,
                attempt=attempt,
                previous_critique=critique,
                dry_run_seed=proposal,
                max_rounds=max_thinking_loop_rounds,
            )
            proposal_ref = self._write_proposal_artifacts(iteration_dir, proposal)
            event_refs = [proposal_ref, *trace_refs]
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
                    trace_refs=trace_refs,
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
            trace_refs=[],
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
        max_rounds: int = 15,
    ) -> tuple[ResearchProposal, list[ArtifactRef]]:
        trace: list[dict[str, object]] = []
        if self.router.dry_run:
            trace.append(
                {
                    "round": 0,
                    "action": "commit_proposal",
                    "observation": "dry-run returned deterministic proposal seed",
                }
            )
            return dry_run_seed, self._write_proposal_trace(
                iteration_dir, attempt=attempt, proposal=dry_run_seed, trace=trace
            )

        max_rounds = max(1, max_rounds)
        for round_index in range(1, max_rounds + 1):
            action = self._choose_loop_action(
                context=context,
                attempt=attempt,
                round_index=round_index,
                max_rounds=max_rounds,
                previous_critique=previous_critique,
                trace=trace,
            )
            trace.append({"round": round_index, "action": action.model_dump(mode="json")})
            if action.action_type == "commit_proposal":
                if action.proposal is None:
                    trace.append(
                        {
                            "round": round_index,
                            "observation": "commit_proposal action omitted the required proposal field",
                        }
                    )
                    continue
                proposal = action.proposal
                trace.append({"round": round_index, "observation": "proposal committed"})
                return proposal, self._write_proposal_trace(
                    iteration_dir, attempt=attempt, proposal=proposal, trace=trace
                )
            observation = self._execute_loop_action(action)
            trace.append({"round": round_index, "observation": observation})

        trace_refs = self._write_uncommitted_proposal_trace(
            iteration_dir, attempt=attempt, trace=trace
        )
        raise RuntimeError(
            "Proposal generator did not commit a proposal within "
            f"{max_rounds} thinking-loop round(s). Trace: "
            f"{', '.join(ref.path for ref in trace_refs)}"
        )

    def _choose_loop_action(
        self,
        *,
        context: Any,
        attempt: int,
        round_index: int,
        max_rounds: int,
        previous_critique: ProposalCritique | None,
        trace: list[dict[str, object]],
    ) -> ProposalLoopAction:
        prompt_payload = {
            "proposal_generation_attempt": attempt + 1,
            "loop_round": round_index,
            "max_rounds": max_rounds,
            "instruction": (
                "If this is the final round, return `commit_proposal` with the `proposal` "
                "field populated."
            ),
            "context": context,
            "previous_critique": previous_critique.model_dump(mode="json")
            if previous_critique
            else None,
            "loop_trace_so_far": trace[-12:],
        }
        messages = [
            {
                "role": "system",
                "content": render_prompt("proposal_generator", override_dir=self.prompt_dir),
            },
            {"role": "user", "content": compact_json_dumps(prompt_payload)},
        ]
        return self.router.complete_structured(
            task_type="proposal_generation",
            messages=messages,
            schema=ProposalLoopAction,
        )

    def _execute_loop_action(self, action: ProposalLoopAction) -> dict[str, object]:
        try:
            if action.action_type == "query_literature":
                answer = self.literature.answer_query(action.query, limit=5)
                return {
                    "status": "ok",
                    "tool": "query_literature",
                    "query": action.query,
                    "result_count": len(answer.results),
                    "answer": answer.model_dump(mode="json"),
                }
            if action.action_type == "search_papers":
                candidates = self.literature.search_papers(action.query, limit=8)
                return {
                    "status": "ok",
                    "tool": "search_papers",
                    "query": action.query,
                    "candidate_count": len(candidates),
                    "candidates": [c.model_dump(mode="json") for c in candidates],
                }
            if action.action_type == "import_url":
                paper = self.literature.import_url(action.url, extract_text=action.extract_text)
                return {"status": "ok", "tool": "import_url", "paper": paper.model_dump(mode="json")}
            if action.action_type == "import_arxiv":
                paper = self.literature.import_arxiv(
                    action.arxiv_id, extract_text=action.extract_text
                )
                return {"status": "ok", "tool": "import_arxiv", "paper": paper.model_dump(mode="json")}
            if action.action_type == "import_doi":
                paper = self.literature.import_doi(action.doi, extract_text=action.extract_text)
                return {"status": "ok", "tool": "import_doi", "paper": paper.model_dump(mode="json")}
            if action.action_type == "import_candidate":
                paper = self.literature.import_candidate(
                    action.candidate_id, extract_text=action.extract_text
                )
                return {
                    "status": "ok",
                    "tool": "import_candidate",
                    "paper": paper.model_dump(mode="json"),
                }
        except Exception as exc:  # noqa: BLE001 - proposal exploration can recover from tool failures
            return {
                "status": "error",
                "tool": action.action_type,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        return {"status": "ignored", "tool": action.action_type}

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

    def _write_proposal_trace(
        self,
        iteration_dir: str,
        *,
        attempt: int,
        proposal: ResearchProposal,
        trace: list[dict[str, object]],
    ) -> list[ArtifactRef]:
        payload = {
            "proposal_id": proposal.proposal_id,
            "attempt": attempt + 1,
            "trace": trace,
        }
        json_ref = self.store.write_json(
            f"{iteration_dir}/proposal_thinking_trace_{proposal.proposal_id}.json", payload
        )
        lines = [f"# Proposal Thinking Trace `{proposal.proposal_id}`", ""]
        for item in trace:
            lines.append(f"## Round {item.get('round', '?')}")
            if "action" in item:
                lines.extend(
                    [
                        "",
                        "Action:",
                        "",
                        "```json",
                        json.dumps(item["action"], indent=2, sort_keys=True),
                        "```",
                    ]
                )
            if "observation" in item:
                lines.extend(
                    [
                        "",
                        "Observation:",
                        "",
                        "```json",
                        json.dumps(item["observation"], indent=2, sort_keys=True),
                        "```",
                    ]
                )
            lines.append("")
        md_ref = self.store.write_text(
            f"{iteration_dir}/proposal_thinking_trace_{proposal.proposal_id}.md",
            "\n".join(lines).rstrip() + "\n",
        )
        return [json_ref, md_ref]

    def _write_uncommitted_proposal_trace(
        self,
        iteration_dir: str,
        *,
        attempt: int,
        trace: list[dict[str, object]],
    ) -> list[ArtifactRef]:
        stem = f"proposal_thinking_trace_uncommitted_attempt_{attempt + 1:02d}"
        payload = {"attempt": attempt + 1, "committed": False, "trace": trace}
        json_ref = self.store.write_json(f"{iteration_dir}/{stem}.json", payload)
        lines = ["# Uncommitted Proposal Thinking Trace", ""]
        for item in trace:
            lines.append(f"## Round {item.get('round', '?')}")
            if "action" in item:
                lines.extend(
                    [
                        "",
                        "Action:",
                        "",
                        "```json",
                        json.dumps(item["action"], indent=2, sort_keys=True),
                        "```",
                    ]
                )
            if "observation" in item:
                lines.extend(
                    [
                        "",
                        "Observation:",
                        "",
                        "```json",
                        json.dumps(item["observation"], indent=2, sort_keys=True),
                        "```",
                    ]
                )
            lines.append("")
        md_ref = self.store.write_text(
            f"{iteration_dir}/{stem}.md", "\n".join(lines).rstrip() + "\n"
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
        trace_refs: list[ArtifactRef],
        reason: str,
    ) -> tuple[ResearchProposal, ProposalCritique, str]:
        self._record_proposal_event(
            proposal_id=proposal.proposal_id,
            event_type="accepted",
            artifact_refs=[proposal_ref, *trace_refs],
            proposal=proposal,
            critique=critique,
            reason=reason,
        )
        state.iteration = iteration
        state.current_proposal_id = proposal.proposal_id
        state.artifact_refs.extend([proposal_ref, *trace_refs])
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
            missing_complexity_model=[] if proposal.resource_model else ["Complexity/resource model should be explicit."],
            unclear_success_criteria=[] if proposal.success_criteria else ["Success criteria missing."],
            required_revisions=[],
            confidence=0.7,
        )
