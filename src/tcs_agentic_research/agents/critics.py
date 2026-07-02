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


REJECTED_CLAIM_STATUSES = {
    ClaimStatus.refuted,
    ClaimStatus.withdrawn,
    ClaimStatus.duplicate,
}


CENTRAL_SOLVED_CLAIM_TYPES = {
    ClaimType.mathematical,
    ClaimType.algorithmic,
    ClaimType.complexity,
    ClaimType.resource,
    ClaimType.novelty,
    ClaimType.experimental,
    ClaimType.theorem_statement,
    ClaimType.literature,
}


def derive_claim_status_from_evidence(claim: ClaimRecord, store: ArtifactStore) -> ClaimStatus:
    """Return the conservative status justified by claim-local evidence.

    This function is intentionally one-way: evidence can certify or refute a claim, but an LLM
    critic cannot upgrade a claim beyond the evidence attached to that claim. Report-level context
    is not enough; the relevant artifact/citation must be copied into the claim evidence first.
    """
    evidence_types = {ev.evidence_type for ev in claim.evidence}
    if EvidenceType.counterexample in evidence_types:
        return ClaimStatus.refuted
    if EvidenceType.lean_proof in evidence_types:
        if any(ev.artifact_refs for ev in claim.evidence if ev.evidence_type == EvidenceType.lean_proof):
            return ClaimStatus.proved_by_lean
        _add_tag(claim, "lean_proof_missing_artifact")
        return ClaimStatus.needs_review
    if EvidenceType.citation in evidence_types and claim.claim_type == ClaimType.literature:
        if _literature_claim_has_statement_support(store, claim):
            return ClaimStatus.cited
        _add_tag(claim, "unaccepted_no_extracted_theorem_or_algorithm")
        return ClaimStatus.needs_review
    if EvidenceType.experiment in evidence_types:
        if any(ev.artifact_refs for ev in claim.evidence if ev.evidence_type == EvidenceType.experiment):
            return ClaimStatus.experimentally_supported
        _add_tag(claim, "experiment_missing_reproducible_artifact")
        return ClaimStatus.needs_review
    if EvidenceType.informal_argument in evidence_types:
        return ClaimStatus.informal_argument
    if claim.claim_type == ClaimType.definition:
        return ClaimStatus.proposed
    return ClaimStatus.conjecture


def is_claim_acceptably_supported(claim: ClaimRecord, store: ArtifactStore) -> bool:
    """Whether a claim is strong enough to enter ``ResearchState.accepted_claim_ids``.

    Informal arguments and critic reviews are deliberately not certifying evidence. This keeps the
    global state useful as a list of tool/literature-backed facts rather than persuasive text.
    """
    if is_claim_rejected(claim):
        return False
    status = derive_claim_status_from_evidence(claim.model_copy(deep=True), store)
    if status == ClaimStatus.proved_by_lean:
        return True
    if status == ClaimStatus.cited and claim.claim_type == ClaimType.literature:
        return True
    if status == ClaimStatus.experimentally_supported:
        return True
    return False


def is_claim_rejected(claim: ClaimRecord) -> bool:
    return (
        claim.status in REJECTED_CLAIM_STATUSES
        or any(ev.evidence_type == EvidenceType.counterexample for ev in claim.evidence)
    )


class ResearchCriticAgent:
    def __init__(self, store: ArtifactStore, router: LLMRouter, *, prompt_dir: str | None = None):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir

    def review(self, report: ResearchReport, context: str = "") -> tuple[ResearchReport, ResearchCritique]:
        report = self.enforce_evidence_statuses(report)
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
        report = self.reconcile_report_with_critique(report, critique)
        return report, critique

    def enforce_evidence_statuses(self, report: ResearchReport) -> ResearchReport:
        for claim in report.claims_generated:
            claim.status = derive_claim_status_from_evidence(claim, self.store)
        return report

    def reconcile_report_with_critique(
        self, report: ResearchReport, critique: ResearchCritique
    ) -> ResearchReport:
        """Apply critic downgrades/refutations without allowing critic-only upgrades."""
        claims = {claim.claim_id: claim for claim in report.claims_generated}
        for claim_id in critique.refuted_claim_ids:
            if claim_id in claims:
                claims[claim_id].status = ClaimStatus.refuted
                _add_tag(claims[claim_id], "critic_refuted")
        for claim_id in critique.downgraded_claim_ids:
            claim = claims.get(claim_id)
            if claim is None or claim.status in {ClaimStatus.refuted, ClaimStatus.proved_by_lean}:
                continue
            # Do not upgrade or trust model acceptance. Keep the evidence-derived status, but ensure
            # unsupported central claims remain visibly review-blocked.
            if not is_claim_acceptably_supported(claim, self.store):
                if claim.status in {ClaimStatus.cited, ClaimStatus.experimentally_supported}:
                    claim.status = ClaimStatus.needs_review
                _add_tag(claim, "critic_downgraded")
        for claim_id in critique.accepted_claim_ids:
            claim = claims.get(claim_id)
            if claim is not None and not is_claim_acceptably_supported(claim, self.store):
                _add_tag(claim, "critic_acceptance_not_certifying")
        return report

    def _mock_critique(self, report: ResearchReport) -> ResearchCritique:
        accepted: list[str] = []
        downgraded: list[str] = []
        forced: list[ProofObligation] = []
        for claim in report.claims_generated:
            if is_claim_acceptably_supported(claim, self.store):
                accepted.append(claim.claim_id)
            else:
                downgraded.append(claim.claim_id)
                if claim.claim_type in {
                    ClaimType.mathematical,
                    ClaimType.algorithmic,
                    ClaimType.complexity,
                    ClaimType.theorem_statement,
                }:
                    forced.append(
                        ProofObligation(
                            statement=f"Verify or refute claim `{claim.claim_id}`: {claim.statement}",
                            claim_ids=[claim.claim_id],
                            suggested_tool="lean"
                            if claim.claim_type in {ClaimType.mathematical, ClaimType.theorem_statement}
                            else "informal",
                        )
                    )
        return ResearchCritique(
            accepted_claim_ids=accepted,
            downgraded_claim_ids=downgraded,
            forced_verifications=forced,
            summary=(
                "Dry-run mock critic classified claims by claim-local evidence. Unproved "
                "mathematical, algorithmic, and complexity claims are not accepted as established facts."
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
        outcomes = self._outcomes_from_report(report)
        blockers_without_replication = self._hard_blockers(
            state, report, include_replication=False
        )
        prior_replication_failed = (
            "replication_failed_or_incomplete" in state.outcome_flags
            and not state.confirmed_by_replication
        )
        possible = (
            report.outcome == ReportOutcome.succeeded
            and not blockers_without_replication
            and not state.confirmed_by_replication
            and not prior_replication_failed
        )
        confirmed = (
            report.outcome == ReportOutcome.succeeded
            and not blockers_without_replication
            and state.confirmed_by_replication
        )
        blockers = list(blockers_without_replication)
        if possible and not state.confirmed_by_replication:
            blockers.append("Independent replication has not confirmed the result.")
        if prior_replication_failed:
            blockers.append("Independent replication failed or was incomplete.")
        return SolvedVerdict(
            outcomes=outcomes,
            possible_breakthrough=possible,
            confirmed_solved=confirmed,
            requires_independent_replication=possible,
            rationale="Conservative solved check based on hard evidence gates.",
            blocking_issues=list(dict.fromkeys(blockers)),
            next_action="stop_confirmed"
            if confirmed
            else ("independent_replication" if possible else "continue"),
        )

    def _enforce_conservatism(
        self, verdict: SolvedVerdict, state: ResearchState, report: ResearchReport | None
    ) -> SolvedVerdict:
        if report is None:
            verdict.confirmed_solved = False
            verdict.possible_breakthrough = False
            verdict.requires_independent_replication = False
            verdict.next_action = "continue"
            _append_unique(verdict.blocking_issues, "No report exists yet.")
            return verdict

        blockers_without_replication = self._hard_blockers(
            state, report, include_replication=False
        )
        prior_replication_failed = (
            "replication_failed_or_incomplete" in state.outcome_flags
            and not state.confirmed_by_replication
        )
        for blocker in blockers_without_replication:
            _append_unique(verdict.blocking_issues, blocker)

        if blockers_without_replication or report.outcome != ReportOutcome.succeeded:
            verdict.confirmed_solved = False
            verdict.possible_breakthrough = False
            verdict.requires_independent_replication = False
            verdict.next_action = "continue"
        elif state.confirmed_by_replication:
            verdict.confirmed_solved = True
            verdict.possible_breakthrough = False
            verdict.requires_independent_replication = False
            verdict.next_action = "stop_confirmed"
        elif prior_replication_failed:
            verdict.confirmed_solved = False
            verdict.possible_breakthrough = False
            verdict.requires_independent_replication = False
            verdict.next_action = "continue"
            _append_unique(
                verdict.blocking_issues,
                "Prior independent replication failed or was incomplete.",
            )
        else:
            verdict.confirmed_solved = False
            verdict.possible_breakthrough = True
            verdict.requires_independent_replication = True
            verdict.next_action = "independent_replication"
            _append_unique(
                verdict.blocking_issues,
                "Independent replication has not confirmed the result.",
            )

        for outcome in self._outcomes_from_report(report):
            if outcome not in verdict.outcomes:
                verdict.outcomes.append(outcome)
        if any("proof obligation" in blocker.lower() for blocker in verdict.blocking_issues):
            if SolvedOutcome.needs_formalization not in verdict.outcomes:
                verdict.outcomes.append(SolvedOutcome.needs_formalization)
        needs_resource_review = any(
            "complexity" in blocker.lower() or "resource" in blocker.lower()
            for blocker in verdict.blocking_issues
        )
        if needs_resource_review and SolvedOutcome.needs_complexity_review not in verdict.outcomes:
            verdict.outcomes.append(SolvedOutcome.needs_complexity_review)
        return verdict

    def _hard_blockers(
        self, state: ResearchState, report: ResearchReport, *, include_replication: bool
    ) -> list[str]:
        blockers: list[str] = []
        if report.outcome != ReportOutcome.succeeded:
            blockers.append(f"Latest report outcome is `{report.outcome.value}`, not `succeeded`.")
        if report.outcome == ReportOutcome.succeeded and not report.claims_generated:
            blockers.append("Succeeded reports must contain generated claims with evidence.")
        open_obligations = [
            obligation
            for obligation in report.proof_obligations
            if obligation.status in {"open", "in_progress", "blocked"}
        ]
        if open_obligations:
            blockers.append("Open or blocked proof obligations remain.")
        if any(estimate.needs_derivation_review for estimate in report.complexity_estimates):
            blockers.append("Complexity/resource estimates require derivation review.")
        if report.required_verifications:
            blockers.append("Required verification items remain unresolved.")
        if report.unresolved_issues:
            blockers.append("Report still lists unresolved issues.")
        unsupported = []
        for claim in report.claims_generated:
            if claim.claim_type not in CENTRAL_SOLVED_CLAIM_TYPES:
                continue
            if claim.status in REJECTED_CLAIM_STATUSES:
                blockers.append(f"Central claim `{claim.claim_id}` is {claim.status.value}.")
                continue
            if not is_claim_acceptably_supported(claim, self.store):
                unsupported.append(claim.claim_id)
        if unsupported:
            blockers.append(
                "Central claims lack certifying claim-local evidence: " + ", ".join(unsupported)
            )
        if include_replication and not state.confirmed_by_replication:
            blockers.append("Independent replication has not confirmed the result.")
        return list(dict.fromkeys(blockers))

    def _outcomes_from_report(self, report: ResearchReport) -> list[SolvedOutcome]:
        outcomes: list[SolvedOutcome] = []
        if report.outcome == ReportOutcome.counterexample_found:
            outcomes.append(SolvedOutcome.counterexample_found)
        elif report.outcome == ReportOutcome.negative_result:
            outcomes.append(SolvedOutcome.negative_result)
        elif report.outcome == ReportOutcome.succeeded:
            outcomes.append(SolvedOutcome.solves_main_task)
        elif report.outcome == ReportOutcome.partially_succeeded:
            outcomes.append(SolvedOutcome.partial_progress)
        else:
            outcomes.append(SolvedOutcome.dead_end)
        if any(o.status in {"open", "in_progress", "blocked"} for o in report.proof_obligations):
            outcomes.append(SolvedOutcome.needs_formalization)
        if any(e.needs_derivation_review for e in report.complexity_estimates):
            outcomes.append(SolvedOutcome.needs_complexity_review)
        if report.experimental_results and not any(
            ev.evidence_type == EvidenceType.lean_proof for ev in report.evidence
        ):
            outcomes.append(SolvedOutcome.needs_experiment)
        return list(dict.fromkeys(outcomes))


def _literature_claim_has_statement_support(store: ArtifactStore, claim: ClaimRecord) -> bool:
    """Require statement-level extraction before accepting literature claims as cited."""
    citation_keys = {
        key
        for evidence in claim.evidence
        if evidence.evidence_type == EvidenceType.citation
        for key in evidence.citation_keys
    }
    if not citation_keys:
        return False
    for record in store.read_jsonl("LiteratureDB/extracted_claims.jsonl"):
        if record.get("citation_key") not in citation_keys:
            continue
        if (
            record.get("theorem_statements")
            or record.get("algorithm_statements")
            or record.get("lower_bound_statements")
        ):
            return True
    return False


def _add_tag(claim: ClaimRecord, tag: str) -> None:
    if tag not in claim.tags:
        claim.tags.append(tag)


def _append_unique(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)
