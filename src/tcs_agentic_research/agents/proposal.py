"""Proposal generation and proposal critic agents."""

from __future__ import annotations

import json

from ..artifact_store import ArtifactStore
from ..llm import LLMRouter
from ..prompt_loader import render_prompt
from ..render import render_proposal_markdown
from ..schemas import (
    CriticDecision,
    ProposalCritique,
    ProposalLedgerEntry,
    ProposalRisk,
    ResearchProposal,
    ResearchState,
)


class ProposalAgent:
    def __init__(self, store: ArtifactStore, router: LLMRouter, *, prompt_dir: str | None = None):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir

    def generate_and_review(
        self, state: ResearchState, *, max_revisions: int = 2
    ) -> tuple[ResearchProposal, ProposalCritique, str]:
        task = self.store.read_text(ArtifactStore.RESEARCH_TASK)
        context = self._context_blob(state, task)
        iteration = state.iteration + 1
        iteration_dir = self.store.create_iteration_dir(iteration)
        critique: ProposalCritique | None = None
        proposal = self._mock_proposal(state)

        for attempt in range(max_revisions + 1):
            generation_messages = [
                {"role": "system", "content": render_prompt("proposal_generator", override_dir=self.prompt_dir)},
                {
                    "role": "user",
                    "content": (
                        f"Context for proposal generation (attempt {attempt + 1}):\n{context}\n\n"
                        f"Previous critique, if any:\n{critique.model_dump_json(indent=2) if critique else 'None'}"
                    ),
                },
            ]
            proposal = self.router.complete_structured(
                task_type="proposal_generation",
                messages=generation_messages,
                schema=ResearchProposal,
                mock_output=proposal if self.router.dry_run else None,
            )
            proposal_ref = self.store.write_json(
                f"{iteration_dir}/proposal_{proposal.proposal_id}.json", proposal
            )
            self.store.write_text(
                f"{iteration_dir}/proposal_{proposal.proposal_id}.md", render_proposal_markdown(proposal)
            )
            self.store.append_proposal_event(
                ProposalLedgerEntry(
                    proposal_id=proposal.proposal_id,
                    event_type="generated" if attempt == 0 else "revised",
                    proposal=proposal,
                    artifact_refs=[proposal_ref],
                )
            )

            critic_messages = [
                {"role": "system", "content": render_prompt("proposal_critic", override_dir=self.prompt_dir)},
                {"role": "user", "content": f"Task and state:\n{context}\n\nProposal:\n{proposal.model_dump_json(indent=2)}"},
            ]
            critique = self.router.complete_structured(
                task_type="proposal_critique",
                messages=critic_messages,
                schema=ProposalCritique,
                mock_output=self._mock_critique(proposal) if self.router.dry_run else None,
            )
            self.store.append_proposal_event(
                ProposalLedgerEntry(
                    proposal_id=proposal.proposal_id,
                    event_type="critic_review",
                    critique=critique,
                    reason=critique.summary,
                    artifact_refs=[proposal_ref],
                )
            )
            if critique.decision == CriticDecision.accept:
                self.store.append_proposal_event(
                    ProposalLedgerEntry(
                        proposal_id=proposal.proposal_id,
                        event_type="accepted",
                        proposal=proposal,
                        critique=critique,
                        reason="Accepted by proposal critic.",
                        artifact_refs=[proposal_ref],
                    )
                )
                state.iteration = iteration
                state.current_proposal_id = proposal.proposal_id
                state.artifact_refs.append(proposal_ref)
                self.store.save_state(state)
                return proposal, critique, proposal_ref.path
            if critique.decision == CriticDecision.reject:
                self.store.append_proposal_event(
                    ProposalLedgerEntry(
                        proposal_id=proposal.proposal_id,
                        event_type="rejected",
                        proposal=proposal,
                        critique=critique,
                        reason="Rejected by proposal critic.",
                        artifact_refs=[proposal_ref],
                    )
                )
                break

        # If we get here: No proposal was accepted; revisions were exhausted or the proposal was rejected;
        # Therefore proposal generation failed to produce an acceptable real proposal
        # If we are in a real run, refuse to commit a mock proposal
        if not self.router.dry_run:
            reason = critique.summary if critique is not None else "No acceptable proposal was produced."
            raise RuntimeError(
                "Proposal generation/revision failed in a real run. "
                f"Last critique: {reason}"
            )

        # Dry-run mock proposal: useful for exercising graph/artifact plumbing only.
        proposal = self._mock_proposal(state)
        critique = self._mock_critique(proposal)
        proposal_ref = self.store.write_json(f"{iteration_dir}/proposal_{proposal.proposal_id}.json", proposal)
        self.store.write_text(
            f"{iteration_dir}/proposal_{proposal.proposal_id}.md", render_proposal_markdown(proposal)
        )
        self.store.append_proposal_event(
            ProposalLedgerEntry(
                proposal_id=proposal.proposal_id,
                event_type="accepted",
                proposal=proposal,
                critique=critique,
                reason="Accepted dry-run mock proposal after failed proposal revisions.",
                artifact_refs=[proposal_ref],
            )
        )
        state.iteration = iteration
        state.current_proposal_id = proposal.proposal_id
        state.artifact_refs.append(proposal_ref)
        self.store.save_state(state)
        return proposal, critique, proposal_ref.path

    def _context_blob(self, state: ResearchState, task: str) -> str:
        claim_tail = self.store.read_jsonl(ArtifactStore.CLAIM_LEDGER, limit=20)
        proposal_tail = self.store.read_jsonl(ArtifactStore.PROPOSAL_LEDGER, limit=20)
        return json.dumps(
            {
                "research_task_md": task,
                "research_state": state.model_dump(mode="json"),
                "recent_claim_ledger_entries": claim_tail,
                "recent_proposal_ledger_entries": proposal_tail,
            },
            indent=2,
        )

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
