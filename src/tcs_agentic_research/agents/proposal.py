"""Proposal generation and proposal critic agents."""

from __future__ import annotations

import json
from typing import Any, Literal

from ..artifact_store import ArtifactStore
from ..llm import LLMRouter, StructuredLLMError
from ..prompt_loader import render_prompt
from ..prompt_serialization import compact_json_dumps
from ..obligations import ObligationBoardManager
from ..render import render_proposal_markdown
from ..schemas import (
    ArtifactRef,
    CriticDecision,
    ProposalCritique,
    ProposalKind,
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
        proposal: ResearchProposal | None = self._mock_proposal(state)
        generation_failures: list[str] = []

        for attempt in range(max_revisions + 1):
            try:
                proposal = self._generate_proposal(
                    context=context,
                    iteration_dir=iteration_dir,
                    attempt=attempt,
                    previous_critique=critique,
                    dry_run_seed=proposal or self._mock_proposal(state),
                )
            except StructuredLLMError as exc:
                generation_failures.append(
                    f"attempt {attempt + 1}: {type(exc).__name__}: {_truncate_text(str(exc), 1200)}"
                )
                break

            proposal_ref = self._write_proposal_artifacts(iteration_dir, proposal)
            event_refs = [proposal_ref]
            self._record_proposal_event(
                proposal_id=proposal.proposal_id,
                event_type="generated" if attempt == 0 else "revised",
                artifact_refs=event_refs,
                proposal=proposal,
            )

            try:
                critique = self._review_proposal(context, proposal)
            except StructuredLLMError as exc:
                generation_failures.append(
                    f"critic attempt {attempt + 1}: {type(exc).__name__}: {_truncate_text(str(exc), 1200)}"
                )
                break
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
                    reason="Rejected by proposal critic; converting objections into a barrier-analysis fallback.",
                )
                break

        if self.router.dry_run:
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

        fallback = self._critic_driven_obstruction_proposal(
            state=state,
            task=task,
            prior_proposal=proposal,
            critique=critique,
            generation_failures=generation_failures,
        )
        fallback_ref = self._write_proposal_artifacts(iteration_dir, fallback)
        fallback_critique = self._fallback_accept_critique(
            fallback,
            prior_critique=critique,
            generation_failures=generation_failures,
        )
        self._record_proposal_event(
            proposal_id=fallback.proposal_id,
            event_type="generated",
            artifact_refs=[fallback_ref],
            proposal=fallback,
            reason="Deterministic fallback generated from proposal critic objections.",
        )
        self._record_proposal_event(
            proposal_id=fallback.proposal_id,
            event_type="critic_review",
            artifact_refs=[fallback_ref],
            critique=fallback_critique,
            reason=fallback_critique.summary,
        )
        return self._accept_proposal(
            state,
            iteration,
            fallback,
            fallback_critique,
            fallback_ref,
            reason="Accepted critic-driven obstruction-analysis fallback after failed proposal revisions.",
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
        obligation_context = ObligationBoardManager(self.store).context_for_proposal()
        return {
            "research_task_md": task,
            "research_state": state.model_dump(mode="json"),
            "obligation_board_context": obligation_context,
            "artifact_manifest": self.store.artifact_manifest(max_items=200),
            "workspace_memory_instructions": (
                "The artifact_manifest is a compact index of durable workspace memory. "
                "Do not assume artifact contents that are not included in this prompt. "
                "Use read_artifact or read_jsonl_records when details from prior proposals, "
                "claims, literature answers, reports, obligation runs, or traces materially "
                "affect the proposal. Accepted claims are established; blocked candidate "
                "claims and failed obligations are diagnostic input only."
            ),
        }

    def _critic_driven_obstruction_proposal(
        self,
        *,
        state: ResearchState,
        task: str,
        prior_proposal: ResearchProposal | None,
        critique: ProposalCritique | None,
        generation_failures: list[str],
    ) -> ResearchProposal:
        """Create an executable proposal from critic objections instead of oscillating.

        The fallback deliberately avoids claiming that the previous positive route is correct.
        It asks the research agent to resolve the disputed assumptions, probability/resource
        calculations, or literature gaps and to report a negative/bottleneck result if warranted.
        """
        critique_constraints = _critique_constraints(critique)
        failure_constraints = [f"Proposal-generation failure to account for: {item}" for item in generation_failures]
        prior_title = prior_proposal.title if prior_proposal is not None else "the attempted proposal"
        task_header = next((line.strip() for line in task.splitlines() if line.strip()), "ResearchTask.md")
        open_obligation_notes = [
            f"Carry forward open obligation: {obligation}"
            for obligation in state.open_proof_obligations[:5]
        ]
        disputed_hypotheses = []
        if prior_proposal is not None:
            disputed_hypotheses.extend(prior_proposal.expected_intermediate_lemmas[:5])
            disputed_hypotheses.extend(prior_proposal.hypotheses_to_test[:5])
        if not disputed_hypotheses:
            disputed_hypotheses.append(
                "The previously attempted route may contain a repairable technical obstruction."
            )

        return ResearchProposal(
            title=f"Critic-driven obstruction analysis for {prior_title}",
            proposal_kind=ProposalKind.barrier_analysis,
            precise_goal=(
                "Turn the previous proposal critique into an executable research step: identify the "
                "disputed mathematical/resource assumptions, verify or refute them with explicit "
                "derivations and provenance, and produce either a repaired next-step route or a clear "
                "bottleneck/negative-result statement."
            ),
            relevant_assumptions_and_model=[
                f"Primary task artifact consulted: {task_header}",
                "Use exactly the model, assumptions, success criteria, and forbidden shortcuts stated in ResearchTask.md.",
                "Treat the previous proposal as a source of hypotheses, not as established truth.",
                "A negative result, bottleneck theorem, or precise dead-end diagnosis is a valid successful outcome for this iteration.",
                *open_obligation_notes,
            ],
            expected_intermediate_lemmas=[
                "A normalized list of the previous proposal's disputed assumptions, each classified as cited, derived, refuted, or open.",
                "For each central probability/resource calculation in the disputed route, an explicit derivation or a counter-derivation.",
                "A statement of the strongest conclusion justified after the obstruction analysis: repaired route, conditional route, bottleneck, or dead end.",
            ],
            algorithmic_subgoals=[
                "Read the latest proposal, critic review, ResearchTask.md, and relevant literature/query artifacts before deriving conclusions.",
                "Convert each critic objection into a concrete verification task with evidence requirements.",
                "Where algebraic or small-instance checks help, use the experimenter/bash environment to sanity-check formulas; treat experiments as heuristic support only.",
                "Update claims and proof obligations so later proposal rounds can build on the obstruction analysis without repeating the same loop.",
            ],
            hypotheses_to_test=list(dict.fromkeys(disputed_hypotheses)),
            questions_to_answer=[
                "Which criticised claims are actually true under the task's model, and which are false or unsupported?",
                "Does the obstruction force an uncosted resource, hidden oracle, index-erasure-like step, or other disallowed assumption?",
                "If the positive route fails, what is the cleanest formal bottleneck or conditional theorem suggested by the failure?",
                "What exact next proposal kind should follow: repaired algorithm attempt, lemma derivation, literature audit, formalization, or stop/dead-end?",
            ],
            assertions_used_as_assumptions=[
                "Only the assumptions explicitly recorded in ResearchTask.md and facts supported by local LiteratureDB provenance may be used as established premises.",
                "The fallback does not assume that the previous proposal's algorithmic claims or success probabilities are correct.",
            ],
            must_not_assume=[
                "Do not assume any shortcut explicitly disallowed by ResearchTask.md.",
                "Do not assume an ideal quantum sample, index erasure, quantum oracle access, or uncosted classical-to-quantum state preparation unless the task explicitly permits it.",
                "Do not claim an asymptotic improvement unless all preprocessing, state preparation, repetition, verification, and sample costs are included.",
            ],
            critic_constraints=list(dict.fromkeys([*critique_constraints, *failure_constraints])),
            plausibility_argument=(
                "The proposal loop failed because the critic identified unresolved technical assumptions. "
                "Those assumptions are now the object of study. Executing this proposal should produce "
                "useful progress even if the original positive route is refuted, because the report will "
                "pin down the obstruction and prevent subsequent rounds from reasserting the same flaw."
            ),
            success_criteria=[
                "A research report classifying each critic objection as resolved, refuted, or still open, with evidence records.",
                "Explicit derivations or cited provenance for the central probability/resource calculations under dispute.",
                "A concrete next-step recommendation, or a justified negative/bottleneck result if the route fails.",
            ],
            partial_success_criteria=[
                "At least one disputed assumption is refuted or repaired with a clear derivation.",
                "The report records open proof obligations precise enough for a subsequent lemma-derivation or formalization proposal.",
                "The workspace ledgers distinguish established facts from hypotheses to test in later rounds.",
            ],
            required_tools=[
                "artifact_retrieval",
                "literature_query",
                "experimenter/bash for algebraic or small-instance sanity checks when useful",
                "Lean/LEAP only if a clean formal proposition emerges",
            ],
            known_risks_and_barriers=[
                ProposalRisk(
                    risk="The obstruction may show that the attempted positive route cannot work under the task model.",
                    mitigation="Treat a clear bottleneck or negative-result statement as successful progress.",
                    severity="medium",
                ),
                ProposalRisk(
                    risk="The critic objections may be too broad to settle in one iteration.",
                    mitigation="Prioritize the objections that invalidate correctness or asymptotic improvement, then record narrower open obligations.",
                    severity="medium",
                ),
            ],
            literature_queries=[
                "local provenance for the central theorem or algorithm used in the criticised proposal",
                "known lower bounds, oracle barriers, or state-preparation barriers relevant to the criticised route",
                "baseline algorithm resource bounds needed for comparison",
            ],
            resource_model=(
                "The research report must include all resource terms needed by the disputed route, "
                "including sample count, classical preprocessing, quantum state preparation, quantum "
                "gates/queries, repetitions/amplification, and candidate verification."
            ),
        )

    def _fallback_accept_critique(
        self,
        proposal: ResearchProposal,
        *,
        prior_critique: ProposalCritique | None,
        generation_failures: list[str],
    ) -> ProposalCritique:
        constraints = _critique_constraints(prior_critique)
        if generation_failures:
            constraints.extend(f"Account for generation failure: {item}" for item in generation_failures)
        return ProposalCritique(
            decision=CriticDecision.accept,
            summary=(
                "Accepted deterministic fallback: the proposal converts unresolved critic objections "
                "into a bounded barrier-analysis research step instead of requiring another proposal revision."
            ),
            consistency_with_task=(
                "The fallback preserves the original task assumptions and explicitly forbids disallowed shortcuts."
            ),
            plausibility=(
                "High as a diagnostic research step: it does not assume the disputed route is correct and "
                "allows a repaired route, bottleneck theorem, or negative result as useful outcomes."
            ),
            barrier_risks=constraints,
            missing_complexity_model=[] if proposal.resource_model else ["Resource model missing."],
            unclear_success_criteria=[] if proposal.success_criteria else ["Success criteria missing."],
            required_revisions=[],
            confidence=0.8,
        )

    def _mock_proposal(self, state: ResearchState) -> ResearchProposal:
        return ResearchProposal(
            title="Audit-first scoping pass for literature and formalizable subclaims",
            proposal_kind=ProposalKind.literature_audit,
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
            hypotheses_to_test=[
                "There is at least one low-risk next step that can be supported by literature provenance or explicit derivation."
            ],
            questions_to_answer=[
                "Which facts are already cited or proved, and which are merely conjectural?",
                "Which resource terms or hidden assumptions are most likely to invalidate a proposed route?",
            ],
            assertions_used_as_assumptions=[
                "Only the computational model and assumptions in ResearchTask.md are established at the start of this pass."
            ],
            must_not_assume=[
                "Do not introduce stronger assumptions without explicit ledger entries and critic review."
            ],
            critic_constraints=[
                "Separate supported facts from hypotheses before proposing a breakthrough claim."
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


def _critique_constraints(critique: ProposalCritique | None) -> list[str]:
    if critique is None:
        return []
    items: list[str] = []
    for prefix, values in [
        ("Required revision", critique.required_revisions),
        ("Barrier risk", critique.barrier_risks),
        ("Missing complexity/model issue", critique.missing_complexity_model),
        ("Unclear success criterion", critique.unclear_success_criteria),
    ]:
        for value in values:
            items.append(f"{prefix}: {value}")
    if critique.summary:
        items.append(f"Critic summary: {_truncate_text(critique.summary, 1200)}")
    return items


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"...[truncated {len(text) - limit} chars]"
