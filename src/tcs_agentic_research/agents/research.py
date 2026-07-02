"""Research execution agent and durable report writer."""

from __future__ import annotations

import json

from ..artifact_store import ArtifactStore
from ..llm import LLMRouter
from ..prompt_loader import render_prompt
from ..render import render_report_markdown
from ..schemas import (
    ClaimRecord,
    ClaimStatus,
    ClaimType,
    EvidenceRecord,
    EvidenceType,
    ProofObligation,
    ReportOutcome,
    ResearchProposal,
    ResearchReport,
    ResearchState,
)
from .critics import ResearchCriticAgent
from .literature import LiteratureResearcher


class ResearchAgent:
    def __init__(self, store: ArtifactStore, router: LLMRouter, *, prompt_dir: str | None = None):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir
        self.literature = LiteratureResearcher(store, router, prompt_dir=prompt_dir)
        self.critic = ResearchCriticAgent(store, router, prompt_dir=prompt_dir)

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
                    "Execute the selected proposal using only evidence that can be referenced by durable artifacts.\n"
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
        if report.proposal_id != proposal.proposal_id:
            report.proposal_id = proposal.proposal_id
        for estimate in report.complexity_estimates:
            if estimate.needs_derivation_review:
                report.required_verifications.append(
                    f"Complexity estimate for {estimate.resource}={estimate.bound} needs derivation review."
                )
        report, critique = self.critic.review(report, context=context)
        for obligation in critique.forced_verifications:
            if obligation.obligation_id not in {o.obligation_id for o in report.proof_obligations}:
                report.proof_obligations.append(obligation)
        self.store.append_claims(report.claims_generated)
        iteration_dir = self.store.create_iteration_dir(state.iteration)
        report_ref = self.store.write_json(f"{iteration_dir}/research_report_{report.report_id}.json", report)
        self.store.write_text(
            f"{iteration_dir}/research_report_{report.report_id}.md", render_report_markdown(report)
        )
        critique_ref = self.store.write_json(
            f"{iteration_dir}/research_critique_{report.report_id}.json", critique
        )
        report.artifact_refs.extend([report_ref, critique_ref])
        # Rewrite report so its own artifact references include the JSON/critique files.
        report_ref = self.store.write_json(f"{iteration_dir}/research_report_{report.report_id}.json", report)
        return report, report_ref.path

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
                "No conjecture or informal argument may be upgraded without Lean, citation, derivation, or experimental evidence as appropriate."
            ],
        )
