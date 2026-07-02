"""Critic agents for scientific fidelity and conservative solved checks."""

from __future__ import annotations

from ..artifact_store import ArtifactStore
from ..llm import LLMRouter
from ..prompt_loader import render_prompt
from ..schemas import (
    ClaimRecord,
    ClaimStatus,
    ClaimType,
    EvidenceType,
    ProofObligation,
    ReportOutcome,
    ResearchCritique,
    ResearchReport,
    ResearchState,
    SolvedOutcome,
    SolvedVerdict,
)


class ResearchCriticAgent:
    def __init__(self, store: ArtifactStore, router: LLMRouter, *, prompt_dir: str | None = None):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir

    def review(self, report: ResearchReport, context: str = "") -> tuple[ResearchReport, ResearchCritique]:
        report = self._downgrade_or_accept_claims(report)
        mock_output = self._mock_critique(report)
        messages = [
            {"role": "system", "content": render_prompt("research_critic", override_dir=self.prompt_dir)},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nReport to critique:\n{report.model_dump_json(indent=2)}",
            },
        ]
        critique = self.router.complete_structured(
            task_type="research_critique",
            messages=messages,
            schema=ResearchCritique,
            mock_output=mock_output if self.router.dry_run else None,
        )
        return report, critique

    def _downgrade_or_accept_claims(self, report: ResearchReport) -> ResearchReport:
        for claim in report.claims_generated:
            # Only claim-local evidence can certify a claim. Report-level evidence is context and
            # must be copied into the claim by an agent before it upgrades status.
            evidence_types = {ev.evidence_type for ev in claim.evidence}
            if EvidenceType.counterexample in evidence_types:
                claim.status = ClaimStatus.refuted
            elif EvidenceType.lean_proof in evidence_types:
                claim.status = ClaimStatus.proved_by_lean
            elif (
                EvidenceType.citation in evidence_types
                and claim.claim_type == ClaimType.literature
            ):
                if self._literature_claim_has_statement_support(claim):
                    claim.status = ClaimStatus.cited
                else:
                    claim.status = ClaimStatus.needs_review
                    if "unaccepted_no_extracted_theorem_or_algorithm" not in claim.tags:
                        claim.tags.append("unaccepted_no_extracted_theorem_or_algorithm")
            elif EvidenceType.resource_accounting in evidence_types and claim.claim_type in {
                ClaimType.complexity,
                ClaimType.resource,
            }:
                claim.status = ClaimStatus.resource_checked
            elif EvidenceType.experiment in evidence_types:
                claim.status = ClaimStatus.experimentally_supported
            elif EvidenceType.informal_argument in evidence_types:
                claim.status = ClaimStatus.informal_argument
            else:
                claim.status = ClaimStatus.conjecture
        return report

    def _literature_claim_has_statement_support(self, claim: ClaimRecord) -> bool:
        """Require statement-level extraction before accepting literature claims as cited."""
        citation_keys = {
            key
            for evidence in claim.evidence
            if evidence.evidence_type == EvidenceType.citation
            for key in evidence.citation_keys
        }
        if not citation_keys:
            return False
        for record in self.store.read_jsonl("LiteratureDB/extracted_claims.jsonl"):
            if record.get("citation_key") not in citation_keys:
                continue
            if (
                record.get("theorem_statements")
                or record.get("algorithm_statements")
                or record.get("lower_bound_statements")
            ):
                return True
        return False

    def _mock_critique(self, report: ResearchReport) -> ResearchCritique:
        accepted: list[str] = []
        downgraded: list[str] = []
        forced: list[ProofObligation] = []
        for claim in report.claims_generated:
            if claim.status in {
                ClaimStatus.proved_by_lean,
                ClaimStatus.cited,
                ClaimStatus.resource_checked,
                ClaimStatus.experimentally_supported,
            }:
                accepted.append(claim.claim_id)
            else:
                downgraded.append(claim.claim_id)
                if claim.claim_type in {ClaimType.mathematical, ClaimType.algorithmic, ClaimType.complexity}:
                    forced.append(
                        ProofObligation(
                            statement=f"Verify or refute claim `{claim.claim_id}`: {claim.statement}",
                            claim_ids=[claim.claim_id],
                            suggested_tool="lean" if claim.claim_type == ClaimType.mathematical else "resource_accounting",
                        )
                    )
        return ResearchCritique(
            accepted_claim_ids=accepted,
            downgraded_claim_ids=downgraded,
            forced_verifications=forced,
            summary=(
                "Dry-run mock critic classified claims by evidence type. Unproved mathematical, "
                "algorithmic, and complexity claims are not accepted as established facts."
            ),
            rejects_report=False,
            reasons=[],
        )


class SolvedCheckAgent:
    def __init__(self, store: ArtifactStore, router: LLMRouter, *, prompt_dir: str | None = None):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir

    def check(self, state: ResearchState, report: ResearchReport | None, context: str = "") -> SolvedVerdict:
        mock_output = self._mock_verdict(state, report)
        messages = [
            {"role": "system", "content": render_prompt("solved_checker", override_dir=self.prompt_dir)},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\nState:\n{state.model_dump_json(indent=2)}\n\n"
                    f"Latest report:\n{report.model_dump_json(indent=2) if report else 'None'}"
                ),
            },
        ]
        verdict = self.router.complete_structured(
            task_type="solved_check",
            messages=messages,
            schema=SolvedVerdict,
            mock_output=mock_output if self.router.dry_run else None,
        )
        return self._enforce_conservatism(verdict, state, report)

    def _mock_verdict(self, state: ResearchState, report: ResearchReport | None) -> SolvedVerdict:
        if report is None:
            return SolvedVerdict(
                outcomes=[SolvedOutcome.partial_progress],
                rationale="No report exists yet.",
                blocking_issues=["Run at least one research iteration."],
            )
        outcomes: list[SolvedOutcome] = []
        blockers: list[str] = []
        if report.outcome == ReportOutcome.counterexample_found:
            outcomes.append(SolvedOutcome.counterexample_found)
        elif report.outcome == ReportOutcome.useful_obstruction:
            outcomes.append(SolvedOutcome.negative_result)
        elif report.outcome == ReportOutcome.succeeded:
            outcomes.append(SolvedOutcome.solves_main_task)
        elif report.outcome == ReportOutcome.partially_succeeded:
            outcomes.append(SolvedOutcome.partial_progress)
        else:
            outcomes.append(SolvedOutcome.dead_end)
        if report.proof_obligations:
            outcomes.append(SolvedOutcome.needs_formalization)
            blockers.append("Open proof obligations remain.")
        if any(e.needs_accounting_review for e in report.complexity_estimates):
            outcomes.append(SolvedOutcome.needs_resource_accounting)
            blockers.append("Complexity/resource estimates require accounting review.")
        if report.experimental_results and not any(
            ev.evidence_type == EvidenceType.lean_proof for ev in report.evidence
        ):
            outcomes.append(SolvedOutcome.needs_experiment)
        possible = report.outcome == ReportOutcome.succeeded and not blockers
        if "replication_failed_or_incomplete" in state.outcome_flags and not state.confirmed_by_replication:
            possible = False
            blockers.append("Independent replication failed or was incomplete; do not repeat breakthrough check without new evidence.")
        confirmed = bool(report.outcome == ReportOutcome.succeeded and state.confirmed_by_replication and not blockers)
        return SolvedVerdict(
            outcomes=list(dict.fromkeys(outcomes)),
            possible_breakthrough=possible and not confirmed,
            confirmed_solved=confirmed,
            requires_independent_replication=possible and not confirmed,
            rationale="Conservative dry-run/mock solved check based on report outcome and verification blockers.",
            blocking_issues=blockers,
            next_action="stop_confirmed" if confirmed else ("independent_replication" if possible else "continue"),
        )

    def _enforce_conservatism(
        self, verdict: SolvedVerdict, state: ResearchState, report: ResearchReport | None
    ) -> SolvedVerdict:
        # The graph may not terminate as solved unless independent replication has confirmed it.
        if verdict.confirmed_solved and not state.confirmed_by_replication:
            verdict.confirmed_solved = False
            verdict.possible_breakthrough = True
            verdict.requires_independent_replication = True
            verdict.next_action = "independent_replication"
            verdict.blocking_issues.append("Independent replication has not confirmed the result.")
        if "replication_failed_or_incomplete" in state.outcome_flags and not state.confirmed_by_replication:
            verdict.confirmed_solved = False
            verdict.possible_breakthrough = False
            verdict.requires_independent_replication = False
            verdict.next_action = "continue"
            verdict.blocking_issues.append("Prior independent replication failed or was incomplete.")
        if report and report.proof_obligations and SolvedOutcome.needs_formalization not in verdict.outcomes:
            verdict.outcomes.append(SolvedOutcome.needs_formalization)
            verdict.confirmed_solved = False
            verdict.possible_breakthrough = False
            verdict.next_action = "continue"
        if report and any(e.needs_accounting_review for e in report.complexity_estimates):
            if SolvedOutcome.needs_resource_accounting not in verdict.outcomes:
                verdict.outcomes.append(SolvedOutcome.needs_resource_accounting)
            verdict.confirmed_solved = False
            verdict.possible_breakthrough = False
            verdict.next_action = "continue"
        return verdict
