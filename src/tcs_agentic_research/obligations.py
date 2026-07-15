"""Deterministic obligation-first harness for claim-centered research.

The proposal stage now creates executable obligations, not proposal-level claims.  The research
agent runs exactly one obligation and may submit flat factual findings.  This module owns the
only transition from an obligation run into accepted ClaimLedger records.

The design is intentionally small:
- proposals append obligations to ``ObligationBoard.json``;
- each run attempts one obligation and supplies simple evidence handles;
- deterministic gates validate scope, evidence, and consistency;
- only validated runs can append generated claims to ``ClaimLedger.jsonl``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from .agents.critics import is_claim_acceptably_supported, is_claim_rejected
from .artifact_store import ArtifactStore
from .schemas import (
    ArtifactRef,
    ClaimRecord,
    ClaimStatus,
    ClaimType,
    EvidenceRecord,
    EvidenceType,
    ObligationBoard,
    ObligationRun,
    ResearchObligation,
    ResearchProposal,
    ValidationGateStatus,
    ValidationResult,
    utc_now,
)


class ObligationBoardManager:
    """Small deterministic facade around ``ObligationBoard.json``."""

    def __init__(self, store: ArtifactStore):
        self.store = store

    def load(self) -> ObligationBoard:
        return self.store.load_obligation_board()

    def save(self, board: ObligationBoard) -> ArtifactRef:
        return self.store.save_obligation_board(board)

    def next_open_obligation(self, board: ObligationBoard | None = None) -> ResearchObligation | None:
        board = board or self.load()
        for obligation in board.obligations:
            if obligation.status == "open":
                return obligation
        return None

    def get_obligation(self, board: ObligationBoard, obligation_id: str) -> ResearchObligation | None:
        for obligation in board.obligations:
            if obligation.obligation_id == obligation_id:
                return obligation
        return None

    def ensure_obligations_from_proposal(
        self, proposal: ResearchProposal
    ) -> tuple[list[ResearchObligation], bool]:
        """Create obligation-first work items for a proposal if absent.

        The proposal is treated as a plan for what to verify next.  It is not converted into a
        claim and never yields a meta-claim of the form "proposal succeeds".
        """
        board = self.load()
        existing = [
            obligation
            for obligation in board.obligations
            if obligation.proposal_id == proposal.proposal_id
        ]
        if existing:
            return existing, False

        obligations = _obligations_from_proposal(proposal)
        board.obligations.extend(obligations)
        self.save(board)
        return obligations, True

    def context_for_proposal(self, *, max_items: int = 12) -> dict[str, Any]:
        """Compact deterministic context for the proposal generator."""
        board = self.load()
        accepted = []
        for claim in self.store.latest_claims_by_id().values():
            if is_claim_acceptably_supported(claim, self.store):
                accepted.append(
                    {
                        "claim_id": claim.claim_id,
                        "statement": claim.statement,
                        "status": claim.status.value,
                        "claim_type": claim.claim_type.value,
                    }
                )
        blocked_obligations = [
            {
                "obligation_id": obligation.obligation_id,
                "proposal_id": obligation.proposal_id,
                "statement": obligation.statement,
                "kind": obligation.kind,
                "status": obligation.status,
                "failure_reason": obligation.failure_reason,
                "last_run_id": obligation.last_run_id,
            }
            for obligation in board.obligations
            if obligation.status in {"blocked", "failed"}
        ]
        open_obligations = [
            {
                "obligation_id": obligation.obligation_id,
                "proposal_id": obligation.proposal_id,
                "statement": obligation.statement,
                "kind": obligation.kind,
                "required_evidence": [ev.value for ev in obligation.required_evidence],
            }
            for obligation in board.obligations
            if obligation.status == "open"
        ]
        return {
            "accepted_claims": accepted[-max_items:],
            "blocked_or_failed_obligations": blocked_obligations[-max_items:],
            "open_obligations": open_obligations[:max_items],
            "instruction": (
                "Use accepted claims as established context. Treat blocked/failed obligations as "
                "diagnostic lessons. New proposals should add concrete obligations, not meta-claims "
                "about the proposal succeeding."
            ),
        }


class ObligationRunValidator:
    """Deterministic scope/provenance, evidence, and consistency gates."""

    def __init__(self, store: ArtifactStore):
        self.store = store

    def validate(
        self,
        *,
        run: ObligationRun,
        obligation: ResearchObligation,
        trace: dict[str, Any] | None = None,
    ) -> ValidationResult:
        gate_results = [
            self._scope_provenance_gate(run, obligation, trace or {}),
            self._evidence_gate(run, obligation, trace or {}),
            self._consistency_gate(run.claims_generated),
        ]
        blockers = [issue for gate in gate_results if not gate.passed for issue in gate.issues]
        return ValidationResult(
            ok=not blockers,
            gate_results=gate_results,
            blocking_issues=blockers,
        )

    def _scope_provenance_gate(
        self,
        run: ObligationRun,
        obligation: ResearchObligation,
        trace: dict[str, Any],
    ) -> ValidationGateStatus:
        issues: list[str] = []
        if run.obligation_id != obligation.obligation_id:
            issues.append("Obligation run does not reference the assigned obligation_id.")
        if run.proposal_id and obligation.proposal_id and run.proposal_id != obligation.proposal_id:
            issues.append("Obligation run proposal_id does not match the assigned obligation.")
        if _is_placeholder(run.summary):
            issues.append("Obligation run summary is empty or placeholder text.")
        for blocker in run.unresolved_blockers:
            if _is_placeholder(blocker):
                issues.append("Obligation run contains placeholder unresolved-blocker text.")
        for claim in run.claims_generated:
            if _is_placeholder(claim.statement):
                issues.append("Generated claim statement is empty or placeholder text.")
            if _looks_like_proposal_meta_claim(claim.statement):
                issues.append(
                    "Generated claim is a proposal-success meta-claim; state the factual theorem, "
                    "algorithmic fact, literature fact, counterexample, or resource bound instead."
                )
            for evidence in claim.evidence:
                self._check_evidence_provenance(evidence, trace, issues)
        for evidence in run.evidence:
            self._check_evidence_provenance(evidence, trace, issues)
        for ref in run.artifact_refs:
            if not self._artifact_exists(ref):
                issues.append(f"Run artifact does not exist: `{ref.path}`.")
        for child in run.child_obligations:
            if _is_placeholder(child.statement):
                issues.append("Child obligation statement is empty or placeholder text.")
            if _looks_like_proposal_meta_claim(child.statement):
                issues.append("Child obligation is a proposal-success meta-obligation.")
        return ValidationGateStatus(
            gate="scope_provenance", passed=not issues, issues=list(dict.fromkeys(issues))
        )

    def _check_evidence_provenance(
        self, evidence: EvidenceRecord, trace: dict[str, Any], issues: list[str]
    ) -> None:
        trace_ids = _tool_result_ids_from_trace(trace)
        for result_id in evidence.tool_result_ids:
            if result_id not in trace_ids:
                issues.append(f"Evidence references unknown tool_result_id `{result_id}`.")
        for ref in evidence.artifact_refs:
            if not self._artifact_exists(ref):
                issues.append(f"Evidence artifact does not exist: `{ref.path}`.")
        for key in evidence.citation_keys:
            if key not in _imported_citation_keys(self.store):
                issues.append(f"Citation key `{key}` is not imported in LiteratureDB/papers.jsonl.")
        if evidence.evidence_type == EvidenceType.experiment:
            experiment_ids = _tool_result_ids_from_trace(trace, tool_name="run_experiment")
            for result_id in evidence.tool_result_ids:
                if result_id not in experiment_ids:
                    issues.append(
                        f"Experiment evidence `{result_id}` was not produced by run_experiment in this run."
                    )

    def _evidence_gate(
        self, run: ObligationRun, obligation: ResearchObligation, trace: dict[str, Any]
    ) -> ValidationGateStatus:
        issues: list[str] = []
        if run.outcome != "fulfilled":
            issues.append(f"Obligation run outcome is `{run.outcome}`, not `fulfilled`.")
        if run.outcome == "fulfilled" and not run.claims_generated:
            issues.append("Fulfilled obligations must generate at least one factual claim.")
        required = obligation.required_evidence or [EvidenceType.informal_argument]
        evidence_pool = [*run.evidence]
        for claim in run.claims_generated:
            evidence_pool.extend(claim.evidence)
        for evidence_type in required:
            if evidence_type == EvidenceType.citation:
                if not _has_citation_evidence(evidence_pool):
                    issues.append("Literature obligation lacks citation evidence with citation keys.")
                for key in _citation_keys(evidence_pool):
                    if not _has_extracted_statement(self.store, key):
                        issues.append(
                            f"Citation key `{key}` has no extracted theorem/algorithm/lower-bound statement."
                        )
            elif evidence_type == EvidenceType.lean_proof:
                if not any(
                    ev.evidence_type == EvidenceType.lean_proof and ev.artifact_refs
                    for ev in evidence_pool
                ):
                    issues.append("Proof obligation lacks Lean proof evidence with artifact refs.")
            elif evidence_type == EvidenceType.experiment:
                if not any(
                    ev.evidence_type == EvidenceType.experiment
                    and ev.tool_result_ids
                    and ev.artifact_refs
                    for ev in evidence_pool
                ):
                    issues.append("Experiment obligation lacks reproducible experiment evidence.")
            elif evidence_type == EvidenceType.informal_argument:
                if len(run.summary.strip()) < 80 and not any(
                    ev.evidence_type == EvidenceType.informal_argument for ev in evidence_pool
                ):
                    issues.append("Derivation/informal obligation lacks a substantive argument summary.")
            elif evidence_type == EvidenceType.external_tool:
                if not any(ev.tool_result_ids or ev.artifact_refs for ev in evidence_pool):
                    issues.append("External-tool obligation lacks tool result IDs or artifacts.")
        return ValidationGateStatus(
            gate="evidence", passed=not issues, issues=list(dict.fromkeys(issues))
        )

    def _consistency_gate(self, generated_claims: Iterable[ClaimRecord]) -> ValidationGateStatus:
        issues: list[str] = []
        accepted_claims = [
            claim
            for claim in self.store.latest_claims_by_id().values()
            if not is_claim_rejected(claim) and is_claim_acceptably_supported(claim, self.store)
        ]
        for candidate in generated_claims:
            candidate_norm = _normalize_statement(candidate.statement)
            candidate_polarity = _polarity(candidate.statement)
            if not candidate_norm:
                continue
            for accepted in accepted_claims:
                accepted_norm = _normalize_statement(accepted.statement)
                if not accepted_norm:
                    continue
                if candidate_norm == accepted_norm:
                    # Duplicates are harmless; CommitManager skips appending another copy.
                    continue
                overlap = _jaccard(_terms(candidate_norm), _terms(accepted_norm))
                if overlap >= 0.82 and candidate_polarity != _polarity(accepted.statement):
                    issues.append(
                        f"Generated claim may contradict accepted claim `{accepted.claim_id}`; "
                        "create a conflict-resolution obligation instead of committing it."
                    )
        return ValidationGateStatus(
            gate="consistency", passed=not issues, issues=list(dict.fromkeys(issues))
        )

    def _artifact_exists(self, ref: ArtifactRef) -> bool:
        try:
            return self.store.exists(ref.path)
        except Exception:
            return False


class CommitManager:
    """The only deterministic path from obligation attempts to canonical claims."""

    def __init__(self, store: ArtifactStore):
        self.store = store
        self.board_manager = ObligationBoardManager(store)

    def apply_obligation_run(
        self,
        *,
        run: ObligationRun,
        validation: ValidationResult,
        run_ref: ArtifactRef,
    ) -> dict[str, Any]:
        board = self.board_manager.load()
        obligation = self.board_manager.get_obligation(board, run.obligation_id)
        if obligation is None:
            raise KeyError(f"Unknown obligation_id `{run.obligation_id}`")

        run.validation = validation
        if run_ref.path not in {ref.path for ref in run.artifact_refs}:
            run.artifact_refs.append(run_ref)
        board.runs.append(run)
        obligation.last_run_id = run.run_id
        obligation.updated_at = utc_now()

        accepted_claim_ids: list[str] = []
        if validation.ok:
            obligation.status = "fulfilled"
            obligation.failure_reason = ""
            for ref in _evidence_refs(run):
                if ref.path not in {existing.path for existing in obligation.evidence_refs}:
                    obligation.evidence_refs.append(ref)
            self._append_child_obligations(board, parent=obligation, run=run)
            accepted_claim_ids = self._append_claims_from_run(run, obligation, run_ref)
            obligation.generated_claim_ids = list(
                dict.fromkeys([*obligation.generated_claim_ids, *accepted_claim_ids])
            )
            outcome = "claims_accepted" if accepted_claim_ids else "obligation_fulfilled"
        else:
            obligation.status = "failed" if run.outcome == "failed" else "blocked"
            obligation.failure_reason = "; ".join(validation.blocking_issues)
            outcome = "obligation_blocked"

        self.board_manager.save(board)
        self._refresh_research_state_from_board(board)
        return {
            "outcome": outcome,
            "obligation_id": obligation.obligation_id,
            "accepted_claim_ids": accepted_claim_ids,
            "blocking_issues": validation.blocking_issues,
        }

    def _append_claims_from_run(
        self, run: ObligationRun, obligation: ResearchObligation, run_ref: ArtifactRef
    ) -> list[str]:
        accepted_ids: list[str] = []
        existing = self.store.latest_claims_by_id()
        existing_norms = {
            _normalize_statement(claim.statement): claim
            for claim in existing.values()
            if is_claim_acceptably_supported(claim, self.store)
        }
        certification = EvidenceRecord(
            evidence_type=EvidenceType.external_tool,
            summary=(
                "This claim was generated by a fulfilled obligation run that passed the "
                "deterministic scope/provenance, evidence, and consistency gates."
            ),
            artifact_refs=[run_ref],
            verifier="CommitManager",
            confidence=1.0,
        )
        to_append: list[ClaimRecord] = []
        for claim in run.claims_generated:
            norm = _normalize_statement(claim.statement)
            if not norm:
                continue
            if norm in existing_norms:
                continue
            prepared = claim.model_copy(deep=True)
            if obligation.proposal_id and obligation.proposal_id not in prepared.related_proposal_ids:
                prepared.related_proposal_ids.append(obligation.proposal_id)
            prepared.evidence.extend(ev.model_copy(deep=True) for ev in run.evidence)
            prepared.evidence.append(certification.model_copy(deep=True))
            prepared.status = _status_from_evidence(prepared.evidence, prepared.claim_type)
            prepared.updated_at = utc_now()
            to_append.append(prepared)
            accepted_ids.append(prepared.claim_id)
            existing_norms[norm] = prepared
        if to_append:
            self.store.append_claims(to_append)
        return accepted_ids

    def _append_child_obligations(
        self, board: ObligationBoard, *, parent: ResearchObligation, run: ObligationRun
    ) -> None:
        existing = {_normalize_statement(obligation.statement) for obligation in board.obligations}
        for child in run.child_obligations:
            norm = _normalize_statement(child.statement)
            if not norm or norm in existing:
                continue
            child.proposal_id = child.proposal_id or parent.proposal_id
            board.obligations.append(child)
            existing.add(norm)

    def _refresh_research_state_from_board(self, board: ObligationBoard) -> None:
        state = self.store.load_state()
        if state is None:
            return
        latest = self.store.latest_claims_by_id()
        state.accepted_claim_ids = [
            claim_id
            for claim_id, claim in latest.items()
            if is_claim_acceptably_supported(claim, self.store)
        ]
        state.rejected_claim_ids = [
            claim_id for claim_id, claim in latest.items() if is_claim_rejected(claim)
        ]
        state.active_claim_ids = list(state.accepted_claim_ids)
        state.open_proof_obligations = [
            obligation.statement for obligation in board.obligations if obligation.status == "open"
        ]
        if self.store.exists(ArtifactStore.OBLIGATION_BOARD):
            board_ref = self.store.artifact_ref(ArtifactStore.OBLIGATION_BOARD)
            if board_ref.path not in {ref.path for ref in state.artifact_refs}:
                state.artifact_refs.append(board_ref)
        self.store.save_state(state)


def _obligations_from_proposal(proposal: ResearchProposal) -> list[ResearchObligation]:
    raw_items: list[str] = []
    raw_items.extend(proposal.obligation_statements)
    if not raw_items:
        raw_items.extend(proposal.expected_intermediate_lemmas[:6])
        raw_items.extend(proposal.questions_to_answer[:4])
    if not raw_items:
        raw_items.extend(proposal.success_criteria[:4])
    if not raw_items:
        raw_items.append(proposal.precise_goal)

    obligations: list[ResearchObligation] = []
    seen: set[str] = set()
    assumptions = list(
        dict.fromkeys(
            [
                *proposal.assertions_used_as_assumptions,
                *proposal.relevant_assumptions_and_model[:4],
            ]
        )
    )
    for item in raw_items:
        statement = item.strip()
        if not statement or _looks_like_proposal_meta_claim(statement):
            continue
        key = _normalize_statement(statement)
        if key in seen:
            continue
        seen.add(key)
        kind = _classify_obligation_kind(statement)
        obligations.append(
            ResearchObligation(
                proposal_id=proposal.proposal_id,
                statement=statement,
                kind=kind,
                required_evidence=_required_evidence_for_kind(kind),
                success_criteria=proposal.success_criteria[:3],
                assumptions=assumptions,
            )
        )
    if not obligations:
        kind = _classify_obligation_kind(proposal.precise_goal)
        obligations.append(
            ResearchObligation(
                proposal_id=proposal.proposal_id,
                statement=proposal.precise_goal,
                kind=kind,
                required_evidence=_required_evidence_for_kind(kind),
                success_criteria=proposal.success_criteria[:3],
                assumptions=assumptions,
            )
        )
    return obligations


def _classify_obligation_kind(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ["citation", "literature", "paper", "theorem from"]):
        return "literature"
    if any(word in lowered for word in ["lean", "formal", "proof", "prove theorem"]):
        return "proof"
    if any(word in lowered for word in ["experiment", "simulation", "numerical", "small-instance"]):
        return "experiment"
    if any(
        word in lowered
        for word in ["complexity", "runtime", "asymptotic", "derive", "bound", "lemma"]
    ):
        return "derivation"
    if any(word in lowered for word in ["consistent", "contradict", "conflict"]):
        return "consistency"
    return "derivation"


def _required_evidence_for_kind(kind: str) -> list[EvidenceType]:
    if kind == "literature":
        return [EvidenceType.citation]
    if kind == "proof":
        return [EvidenceType.lean_proof]
    if kind == "experiment":
        return [EvidenceType.experiment]
    if kind == "consistency":
        return [EvidenceType.external_tool]
    return [EvidenceType.informal_argument]


def _status_from_evidence(evidence: Iterable[EvidenceRecord], claim_type: ClaimType) -> ClaimStatus:
    evidence_types = {ev.evidence_type for ev in evidence}
    if EvidenceType.counterexample in evidence_types:
        return ClaimStatus.refuted
    if EvidenceType.lean_proof in evidence_types:
        return ClaimStatus.proved_by_lean
    if EvidenceType.citation in evidence_types and claim_type == ClaimType.literature:
        return ClaimStatus.cited
    if EvidenceType.experiment in evidence_types:
        return ClaimStatus.experimentally_supported
    if EvidenceType.external_tool in evidence_types:
        return ClaimStatus.proved_informally
    if EvidenceType.informal_argument in evidence_types:
        return ClaimStatus.informal_argument
    return ClaimStatus.needs_review


def _tool_result_ids_from_trace(trace: dict[str, Any], *, tool_name: str | None = None) -> set[str]:
    ids: set[str] = set()
    for item in trace.get("tool_calls") or []:
        if not isinstance(item, dict):
            continue
        if tool_name is not None and item.get("name") != tool_name:
            continue
        observation = item.get("observation")
        if not isinstance(observation, dict):
            continue
        for key in ["tool_result_id", "answer_id", "run_id"]:
            value = observation.get(key)
            if value:
                ids.add(str(value))
    return ids


def _imported_citation_keys(store: ArtifactStore) -> set[str]:
    return {str(record.get("citation_key")) for record in store.read_jsonl("LiteratureDB/papers.jsonl")}


def _has_extracted_statement(store: ArtifactStore, citation_key: str) -> bool:
    for record in store.read_jsonl("LiteratureDB/extracted_claims.jsonl"):
        if record.get("citation_key") != citation_key:
            continue
        if (
            record.get("theorem_statements")
            or record.get("algorithm_statements")
            or record.get("lower_bound_statements")
        ):
            return True
    return False


def _has_citation_evidence(evidence: Iterable[EvidenceRecord]) -> bool:
    return any(ev.evidence_type == EvidenceType.citation and ev.citation_keys for ev in evidence)


def _citation_keys(evidence: Iterable[EvidenceRecord]) -> set[str]:
    return {key for ev in evidence for key in ev.citation_keys if ev.evidence_type == EvidenceType.citation}


def _evidence_refs(run: ObligationRun) -> list[ArtifactRef]:
    refs = list(run.artifact_refs)
    for evidence in run.evidence:
        refs.extend(evidence.artifact_refs)
    for claim in run.claims_generated:
        for evidence in claim.evidence:
            refs.extend(evidence.artifact_refs)
    unique: list[ArtifactRef] = []
    seen: set[str] = set()
    for ref in refs:
        if ref.path in seen:
            continue
        seen.add(ref.path)
        unique.append(ref)
    return unique


def _looks_like_proposal_meta_claim(statement: str) -> bool:
    lowered = _normalize_statement(statement)
    return lowered.startswith("proposal ") and any(
        phrase in lowered for phrase in [" succeeds", " success", " is successful", " has succeeded"]
    )


def _is_placeholder(text: str) -> bool:
    normalized = " ".join(text.strip().lower().split())
    if not normalized:
        return True
    return normalized in {"test", "todo", "tbd", "n/a", "na", "placeholder"}


def _normalize_statement(statement: str) -> str:
    text = statement.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _terms(statement: str) -> list[str]:
    return [term for term in statement.split() if len(term) >= 3 and term not in _STOP]


_STOP = {"and", "are", "for", "from", "that", "the", "this", "with", "where"}


def _jaccard(left: list[str], right: list[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _polarity(statement: str) -> int:
    lowered = f" {_normalize_statement(statement)} "
    negators = [" no ", " not ", " cannot ", " impossible ", " refutes ", " false ", " fails "]
    return -1 if any(token in lowered for token in negators) else 1
