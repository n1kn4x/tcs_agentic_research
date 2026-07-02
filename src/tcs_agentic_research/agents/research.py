"""Research execution agent and durable report writer."""

from __future__ import annotations

import json
import re

from ..artifact_store import ArtifactStore
from ..llm import LLMRouter
from ..prompt_loader import render_prompt
from ..render import render_report_markdown
from ..schemas import (
    ArtifactRef,
    ClaimRecord,
    ClaimStatus,
    ClaimType,
    EvidenceRecord,
    EvidenceType,
    LeanStatement,
    ProofObligation,
    ReportOutcome,
    ResearchProposal,
    ResearchReport,
    ResearchState,
    TheoremProverResult,
    utc_now,
)
from .critics import ResearchCriticAgent
from .literature import LiteratureResearcher
from .theorem_prover import TheoremProverAgent


class ResearchAgent:
    def __init__(self, store: ArtifactStore, router: LLMRouter, *, prompt_dir: str | None = None):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir
        self.literature = LiteratureResearcher(store, router, prompt_dir=prompt_dir)
        self.critic = ResearchCriticAgent(store, router, prompt_dir=prompt_dir)
        self.theorem_prover = TheoremProverAgent(store, router, prompt_dir=prompt_dir)

    def run(self, proposal: ResearchProposal, state: ResearchState) -> tuple[ResearchReport, str]:
        task = self.store.read_text(ArtifactStore.RESEARCH_TASK)
        literature_answers = []
        for query in proposal.literature_queries[:5]:
            literature_answers.append(
                self.literature.answer_query(query, limit=3).model_dump(mode="json")
            )
        context = json.dumps(
            {
                "research_task_md": task,
                "research_state": state.model_dump(mode="json"),
                "proposal": proposal.model_dump(mode="json"),
                # Literature context is only supplied through mapped-nomenclature answers,
                # with quote-level provenance and duplicate-result flags.
                "local_literature_answers": literature_answers,
                "recent_claims": self.store.read_jsonl(ArtifactStore.CLAIM_LEDGER, limit=30),
            },
            indent=2,
        )
        mock_output = self._mock_report(proposal)
        messages = [
            {"role": "system", "content": render_prompt("research_agent", override_dir=self.prompt_dir)},
            {
                "role": "user",
                "content": (
                    "Execute the selected proposal using only evidence that can be referenced "
                    "by durable artifacts.\n"
                    f"Context:\n{context}"
                ),
            },
        ]
        report = self.router.complete_structured(
            task_type="research_execution",
            messages=messages,
            schema=ResearchReport,
            mock_output=mock_output if self.router.dry_run else None,
        )
        report = self._normalize_report(report, proposal)
        report = self._add_complexity_verification_requirements(report)

        # First critic pass: deterministic evidence downgrades plus LLM-requested obligations.
        report, critique = self.critic.review(report, context=context)
        for obligation in critique.forced_verifications:
            if obligation.obligation_id not in {o.obligation_id for o in report.proof_obligations}:
                report.proof_obligations.append(obligation)

        # Execute already-existing verification subsystems. At present the only fully structured
        # automatic path is LEAP for Lean obligations; experiment commands are intentionally not
        # inferred from prose to avoid running unreviewed LLM-generated code.
        self._run_lean_verifications(report, context=context)
        report = self.critic.enforce_evidence_statuses(report)

        iteration_dir = self.store.create_iteration_dir(state.iteration)
        critique_ref = self.store.write_json(
            f"{iteration_dir}/research_critique_{report.report_id}.json", critique
        )
        if critique_ref.path not in {ref.path for ref in report.artifact_refs}:
            report.artifact_refs.append(critique_ref)
        report_ref = self.store.write_json(
            f"{iteration_dir}/research_report_{report.report_id}.json", report
        )
        self.store.write_text(
            f"{iteration_dir}/research_report_{report.report_id}.md", render_report_markdown(report)
        )

        self._attach_report_refs_to_claims(report, report_ref=report_ref, critique_ref=critique_ref)
        self.store.append_claims(report.claims_generated)
        return report, report_ref.path

    def _normalize_report(self, report: ResearchReport, proposal: ResearchProposal) -> ResearchReport:
        if report.proposal_id != proposal.proposal_id:
            report.proposal_id = proposal.proposal_id
        for claim in report.claims_generated:
            if proposal.proposal_id not in claim.related_proposal_ids:
                claim.related_proposal_ids.append(proposal.proposal_id)
        return report

    def _add_complexity_verification_requirements(self, report: ResearchReport) -> ResearchReport:
        for estimate in report.complexity_estimates:
            if estimate.needs_derivation_review:
                message = (
                    f"Complexity estimate for {estimate.resource}={estimate.bound} "
                    "needs derivation review."
                )
                if message not in report.required_verifications:
                    report.required_verifications.append(message)
        return report

    def _run_lean_verifications(self, report: ResearchReport, *, context: str) -> None:
        attempted = 0
        for obligation in report.proof_obligations:
            if obligation.suggested_tool != "lean":
                continue
            if obligation.status not in {"open", "in_progress"}:
                continue
            if attempted >= 3:
                deferred_message = "Additional Lean obligations deferred after the per-iteration cap."
                if deferred_message not in report.required_verifications:
                    report.required_verifications.append(deferred_message)
                continue
            attempted += 1
            goal = self._lean_goal_from_obligation(obligation)
            result = self.theorem_prover.prove(
                goal,
                context=(
                    context
                    + "\n\nCurrent report before LEAP verification:\n"
                    + report.model_dump_json(indent=2)
                ),
            )
            self._record_theorem_prover_result(report, obligation, result)

    def _record_theorem_prover_result(
        self,
        report: ResearchReport,
        obligation: ProofObligation,
        result: TheoremProverResult,
    ) -> None:
        result_refs = _unique_refs([*result.proved_artifacts, *result.artifact_refs])
        for ref in result_refs:
            if ref.path not in {existing.path for existing in obligation.artifact_refs}:
                obligation.artifact_refs.append(ref)
        if result.status == "proved":
            obligation.status = "proved"
            for claim in report.claims_generated:
                if claim.claim_id not in obligation.claim_ids:
                    continue
                claim.evidence.append(
                    EvidenceRecord(
                        evidence_type=EvidenceType.lean_proof,
                        summary=(
                            f"LEAP verified proof obligation `{obligation.obligation_id}` "
                            f"for Lean goal `{result.root_goal.name}`."
                        ),
                        artifact_refs=result_refs,
                        verifier="LEAPHarness",
                        confidence=1.0,
                    )
                )
                claim.status = ClaimStatus.proved_by_lean
        elif result.status == "partially_proved":
            obligation.status = "in_progress"
        else:
            obligation.status = "blocked"
        if result.recommended_next_steps:
            for step in result.recommended_next_steps:
                if step not in report.required_verifications and obligation.status != "proved":
                    report.required_verifications.append(step)

    def _lean_goal_from_obligation(self, obligation: ProofObligation) -> LeanStatement:
        statement = obligation.statement.strip()
        for prefix in ["lean:", "Lean:", "LEAN:"]:
            if statement.startswith(prefix):
                statement = statement[len(prefix) :].strip()
                break
        name = _lean_safe_name(obligation.obligation_id)
        return LeanStatement(name=name, statement=statement)

    def _attach_report_refs_to_claims(
        self,
        report: ResearchReport,
        *,
        report_ref: ArtifactRef,
        critique_ref: ArtifactRef,
    ) -> None:
        refs = [report_ref, critique_ref]
        for claim in report.claims_generated:
            if report.report_id not in claim.related_report_ids:
                claim.related_report_ids.append(report.report_id)
            if report.proposal_id not in claim.related_proposal_ids:
                claim.related_proposal_ids.append(report.proposal_id)
            claim.evidence.append(
                EvidenceRecord(
                    evidence_type=EvidenceType.critic_review,
                    summary=(
                        "Research report and critic review committed as durable artifacts. "
                        "This audit record is not certifying proof evidence."
                    ),
                    artifact_refs=refs,
                    verifier="ResearchAgent",
                    confidence=0.0,
                )
            )
            claim.updated_at = utc_now()

    def _mock_report(self, proposal: ResearchProposal) -> ResearchReport:
        claim = ClaimRecord(
            claim_type=ClaimType.other,
            statement=(
                f"Proposal {proposal.proposal_id} has not yet produced a verified main-task solution; "
                "current progress is an auditable scoping pass."
            ),
            status=ClaimStatus.informal_argument,
            related_proposal_ids=[proposal.proposal_id],
            evidence=[
                EvidenceRecord(
                    evidence_type=EvidenceType.informal_argument,
                    summary="Dry-run mock research execution records process progress only.",
                    confidence=0.2,
                )
            ],
        )
        return ResearchReport(
            proposal_id=proposal.proposal_id,
            outcome=ReportOutcome.partially_succeeded,
            executive_summary=(
                "Dry-run mock execution completed a conservative scoping iteration. It does not claim a "
                "breakthrough."
            ),
            claims_generated=[claim],
            proof_obligations=[
                ProofObligation(
                    statement="Formalize any central mathematical claim before treating it as established.",
                    claim_ids=[claim.claim_id],
                    suggested_tool="lean",
                )
            ],
            unresolved_issues=[
                "Literature claims need provenance-bearing extraction.",
                "Any algorithmic improvement requires an explicit complexity derivation.",
            ],
            proposed_next_steps=[
                "Import and normalize the most relevant papers into LiteratureDB.",
                "Select a specific lemma, reduction, or algorithmic subgoal for proof or literature review.",
            ],
            required_verifications=[
                "No conjecture or informal argument may be upgraded without Lean, citation, "
                "derivation, or experimental evidence as appropriate."
            ],
        )


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
