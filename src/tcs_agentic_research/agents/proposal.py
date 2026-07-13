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
    ProposalRisk,
    ResearchProposal,
    ResearchState,
)
from ..tooling import Toolset, final_submission_tool
from .literature import LiteratureResearcher
from .toolsets import artifact_retrieval_toolset, literature_toolset


FINAL_PROPOSAL_TOOL_NAME = "submit_research_proposal"


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
        toolset = self._proposal_toolset()
        proposal, trace = self.router.complete_structured_with_tools(
            task_type="proposal_generation",
            messages=messages,
            tools=toolset.openai_tools(),
            tool_executors=toolset.executors(),
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

    def _proposal_toolset(self) -> Toolset:
        return artifact_retrieval_toolset(store=self.store) + literature_toolset(
            store=self.store,
            literature=self.literature,
            include_discovery_tools=True,
        ) + Toolset(
            [
                final_submission_tool(
                    FINAL_PROPOSAL_TOOL_NAME,
                    (
                        "Commit the final concrete research proposal. The arguments must be the "
                        "ResearchProposal object itself, not wrapped under another key."
                    ),
                    ResearchProposal,
                )
            ]
        )

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
        return {
            "research_task_md": task,
            "research_state": state.model_dump(mode="json"),
            "artifact_manifest": self.store.artifact_manifest(max_items=200),
            "workspace_memory_instructions": (
                "The artifact_manifest is a compact index of durable workspace memory. "
                "Do not assume artifact contents that are not included in this prompt. "
                "Use read_artifact or read_jsonl_records when details from prior proposals, "
                "claims, literature answers, reports, or traces materially affect the proposal."
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
