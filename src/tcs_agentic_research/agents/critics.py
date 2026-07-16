"""Claim-evidence checks and deterministic solved verdicts."""

from __future__ import annotations

from typing import TypeVar

from ..artifact_store import ArtifactStore
from ..schemas import (
    ClaimRecord,
    ClaimStatus,
    ClaimType,
    EvidenceType,
    ReportOutcome,
    ResearchReport,
    ResearchState,
    SolvedOutcome,
    SolvedVerdict,
)


T = TypeVar("T")


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

    This function is intentionally one-way: evidence can certify or refute a claim, but report
    context cannot upgrade a claim beyond the evidence attached to that claim. The relevant
    artifact/citation must be copied into claim-local evidence first.
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
    if EvidenceType.external_tool in evidence_types:
        if any(
            ev.artifact_refs
            and ev.verifier in {"ObligationRunValidator", "CommitManager"}
            for ev in claim.evidence
            if ev.evidence_type == EvidenceType.external_tool
        ):
            return ClaimStatus.proved_informally
        _add_tag(claim, "external_tool_missing_validator_artifact")
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
    if status == ClaimStatus.proved_informally:
        return True
    if status == ClaimStatus.cited and claim.claim_type == ClaimType.literature:
        return True
    if status == ClaimStatus.experimentally_supported:
        return claim.claim_type == ClaimType.experimental
    return False


def is_claim_rejected(claim: ClaimRecord) -> bool:
    return (
        claim.status in REJECTED_CLAIM_STATUSES
        or any(ev.evidence_type == EvidenceType.counterexample for ev in claim.evidence)
    )


def check_solved_deterministically(
    store: ArtifactStore, state: ResearchState, report: ResearchReport | None
) -> SolvedVerdict:
    """Compute the solved verdict from hard evidence gates without an LLM call."""
    if report is None:
        return SolvedVerdict(
            outcomes=[SolvedOutcome.partial_progress],
            rationale="No report exists yet.",
            blocking_issues=["Run at least one research iteration."],
            next_action="continue",
        )

    blockers = _solved_hard_blockers(store, report)
    prior_replication_failed = (
        "replication_failed_or_incomplete" in state.outcome_flags
        and not state.confirmed_by_replication
    )
    outcomes = _solved_outcomes_from_report(report)

    if blockers or report.outcome != ReportOutcome.succeeded:
        possible = False
        confirmed = False
        requires_replication = False
        next_action = "continue"
        if prior_replication_failed:
            _append_unique(blockers, "Independent replication failed or was incomplete.")
    elif state.confirmed_by_replication:
        possible = False
        confirmed = True
        requires_replication = False
        next_action = "stop_confirmed"
    elif prior_replication_failed:
        possible = False
        confirmed = False
        requires_replication = False
        next_action = "continue"
        _append_unique(blockers, "Prior independent replication failed or was incomplete.")
    else:
        possible = True
        confirmed = False
        requires_replication = True
        next_action = "independent_replication"
        _append_unique(blockers, "Independent replication has not confirmed the result.")

    if any("proof obligation" in blocker.lower() for blocker in blockers):
        _append_unique(outcomes, SolvedOutcome.needs_formalization)
    needs_resource_review = any(
        "complexity" in blocker.lower() or "resource" in blocker.lower() for blocker in blockers
    )
    if needs_resource_review:
        _append_unique(outcomes, SolvedOutcome.needs_complexity_review)

    return SolvedVerdict(
        outcomes=outcomes,
        possible_breakthrough=possible,
        confirmed_solved=confirmed,
        requires_independent_replication=requires_replication,
        rationale="Deterministic solved verdict based on hard evidence gates.",
        blocking_issues=blockers,
        next_action=next_action,
    )


def _solved_hard_blockers(store: ArtifactStore, report: ResearchReport) -> list[str]:
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
        if not is_claim_acceptably_supported(claim, store):
            unsupported.append(claim.claim_id)
    if unsupported:
        blockers.append(
            "Central claims lack certifying claim-local evidence: " + ", ".join(unsupported)
        )
    return list(dict.fromkeys(blockers))


def _solved_outcomes_from_report(report: ResearchReport) -> list[SolvedOutcome]:
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
    """Require stable statement/quote/support handles before accepting literature claims."""
    support_ids = {
        support_id
        for evidence in claim.evidence
        if evidence.evidence_type == EvidenceType.citation
        for support_id in evidence.literature_support_ids
    }
    if not support_ids:
        return False
    try:
        from ..literature.index import LiteratureIndex

        index = LiteratureIndex(store)
        return any(index.support_exists(support_id) for support_id in support_ids)
    except Exception:
        return False


def _add_tag(claim: ClaimRecord, tag: str) -> None:
    if tag not in claim.tags:
        claim.tags.append(tag)


def _append_unique(items: list[T], item: T) -> None:
    if item not in items:
        items.append(item)
