"""Deterministic obligation-board harness for claim-centered research.

The LLM agents may propose candidate claims and attempt individual obligations, but this module
owns the state transitions from an obligation attempt into canonical workspace state.  It is
intentionally conservative: uncertain provenance, missing evidence, or possible contradictions
block a candidate claim instead of appending it to the accepted claim ledger.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from .agents.critics import is_claim_acceptably_supported, is_claim_rejected
from .artifact_store import ArtifactStore
from .schemas import (
    ArtifactRef,
    CandidateClaim,
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

    def get_claim(self, board: ObligationBoard, claim_id: str) -> CandidateClaim | None:
        for claim in board.candidate_claims:
            if claim.claim_id == claim_id:
                return claim
        return None

    def get_obligation(self, board: ObligationBoard, obligation_id: str) -> ResearchObligation | None:
        for obligation in board.obligations:
            if obligation.obligation_id == obligation_id:
                return obligation
        return None

    def ensure_candidate_from_proposal(
        self, proposal: ResearchProposal
    ) -> tuple[CandidateClaim, list[ResearchObligation], bool]:
        """Create one candidate claim and linked obligations for a proposal if absent.

        This keeps the existing proposal agent intact.  The proposal is interpreted as a source of
        a candidate research claim plus concrete obligations; only fulfilled obligations can later
        promote that claim into the canonical claim ledger.
        """
        board = self.load()
        for claim in board.candidate_claims:
            if claim.source_proposal_id == proposal.proposal_id:
                obligations = [
                    obligation
                    for obligation in board.obligations
                    if obligation.obligation_id in claim.obligation_ids
                ]
                return claim, obligations, False

        claim = CandidateClaim(
            statement=_candidate_statement_from_proposal(proposal),
            claim_type=_claim_type_from_proposal(proposal),
            source_proposal_id=proposal.proposal_id,
            status="in_progress",
        )
        obligations = _obligations_from_proposal(proposal, claim.claim_id)
        claim.obligation_ids = [obligation.obligation_id for obligation in obligations]
        board.candidate_claims.append(claim)
        board.obligations.extend(obligations)
        self.save(board)
        return claim, obligations, True

    def context_for_proposal(self, *, max_items: int = 12) -> dict[str, Any]:
        """Compact deterministic context for the existing proposal generator."""
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
        blocked_claims = [
            {
                "claim_id": claim.claim_id,
                "statement": claim.statement,
                "status": claim.status,
                "blocked_reason": claim.blocked_reason,
                "source_proposal_id": claim.source_proposal_id,
            }
            for claim in board.candidate_claims
            if claim.status in {"blocked", "refuted"}
        ]
        failed_obligations = [
            {
                "obligation_id": obligation.obligation_id,
                "claim_id": obligation.claim_id,
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
                "claim_id": obligation.claim_id,
                "statement": obligation.statement,
                "kind": obligation.kind,
                "required_evidence": [ev.value for ev in obligation.required_evidence],
            }
            for obligation in board.obligations
            if obligation.status == "open"
        ]
        return {
            "accepted_claims": accepted[-max_items:],
            "blocked_candidate_claims": blocked_claims[-max_items:],
            "failed_obligations": failed_obligations[-max_items:],
            "open_obligations": open_obligations[:max_items],
            "instruction": (
                "Use accepted claims as established context. Treat blocked candidate claims and "
                "failed obligations as lessons for selecting a better next proposal; do not treat "
                "them as facts."
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
        candidate_claim: CandidateClaim,
        trace: dict[str, Any] | None = None,
    ) -> ValidationResult:
        gate_results = [
            self._scope_provenance_gate(run, obligation, candidate_claim, trace or {}),
            self._evidence_gate(run, obligation, trace or {}),
            self._consistency_gate(candidate_claim),
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
        candidate_claim: CandidateClaim,
        trace: dict[str, Any],
    ) -> ValidationGateStatus:
        issues: list[str] = []
        if run.obligation_id != obligation.obligation_id:
            issues.append("Obligation run does not reference the assigned obligation_id.")
        if run.claim_id != candidate_claim.claim_id or obligation.claim_id != candidate_claim.claim_id:
            issues.append("Obligation run, obligation, and candidate claim IDs do not agree.")
        if _is_placeholder(run.summary):
            issues.append("Obligation run summary is empty or placeholder text.")
        for blocker in run.unresolved_blockers:
            if _is_placeholder(blocker):
                issues.append("Obligation run contains placeholder unresolved-blocker text.")

        trace_ids = _tool_result_ids_from_trace(trace)
        for evidence in run.evidence:
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
        for ref in run.artifact_refs:
            if not self._artifact_exists(ref):
                issues.append(f"Run artifact does not exist: `{ref.path}`.")
        return ValidationGateStatus(
            gate="scope_provenance", passed=not issues, issues=list(dict.fromkeys(issues))
        )

    def _evidence_gate(
        self, run: ObligationRun, obligation: ResearchObligation, trace: dict[str, Any]
    ) -> ValidationGateStatus:
        issues: list[str] = []
        if run.outcome != "fulfilled":
            issues.append(f"Obligation run outcome is `{run.outcome}`, not `fulfilled`.")
        required = obligation.required_evidence or [EvidenceType.informal_argument]
        for evidence_type in required:
            if evidence_type == EvidenceType.citation:
                if not _has_citation_evidence(run.evidence):
                    issues.append("Literature obligation lacks citation evidence with citation keys.")
                for key in _citation_keys(run.evidence):
                    if not _has_extracted_statement(self.store, key):
                        issues.append(
                            f"Citation key `{key}` has no extracted theorem/algorithm/lower-bound statement."
                        )
            elif evidence_type == EvidenceType.lean_proof:
                if not any(
                    ev.evidence_type == EvidenceType.lean_proof and ev.artifact_refs
                    for ev in run.evidence
                ):
                    issues.append("Proof obligation lacks Lean proof evidence with artifact refs.")
            elif evidence_type == EvidenceType.experiment:
                if not any(
                    ev.evidence_type == EvidenceType.experiment
                    and ev.tool_result_ids
                    and ev.artifact_refs
                    for ev in run.evidence
                ):
                    issues.append("Experiment obligation lacks reproducible experiment evidence.")
            elif evidence_type == EvidenceType.informal_argument:
                if len(run.summary.strip()) < 80 and not any(
                    ev.evidence_type == EvidenceType.informal_argument for ev in run.evidence
                ):
                    issues.append("Derivation/informal obligation lacks a substantive argument summary.")
            elif evidence_type == EvidenceType.external_tool:
                if not any(ev.tool_result_ids or ev.artifact_refs for ev in run.evidence):
                    issues.append("External-tool obligation lacks tool result IDs or artifacts.")
        return ValidationGateStatus(
            gate="evidence", passed=not issues, issues=list(dict.fromkeys(issues))
        )

    def _consistency_gate(self, candidate_claim: CandidateClaim) -> ValidationGateStatus:
        issues: list[str] = []
        candidate_norm = _normalize_statement(candidate_claim.statement)
        candidate_polarity = _polarity(candidate_claim.statement)
        for accepted in self.store.latest_claims_by_id().values():
            if is_claim_rejected(accepted) or not is_claim_acceptably_supported(accepted, self.store):
                continue
            accepted_norm = _normalize_statement(accepted.statement)
            if not candidate_norm or not accepted_norm:
                continue
            if candidate_norm == accepted_norm:
                issues.append(
                    f"Candidate duplicates already accepted claim `{accepted.claim_id}`; do not append a second copy."
                )
                continue
            overlap = _jaccard(_terms(candidate_norm), _terms(accepted_norm))
            if overlap >= 0.82 and candidate_polarity != _polarity(accepted.statement):
                issues.append(
                    f"Candidate may contradict accepted claim `{accepted.claim_id}`; create a conflict-resolution obligation."
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
        claim = self.board_manager.get_claim(board, obligation.claim_id)
        if claim is None:
            raise KeyError(f"Unknown candidate claim_id `{obligation.claim_id}`")

        run.validation = validation
        if run_ref.path not in {ref.path for ref in run.artifact_refs}:
            run.artifact_refs.append(run_ref)
        board.runs.append(run)
        obligation.last_run_id = run.run_id
        obligation.updated_at = utc_now()

        if validation.ok:
            obligation.status = "fulfilled"
            obligation.failure_reason = ""
            for ref in _evidence_refs(run):
                if ref.path not in {existing.path for existing in obligation.evidence_refs}:
                    obligation.evidence_refs.append(ref)
            outcome = "obligation_fulfilled"
            accepted_claim_id = None
            if self._all_claim_obligations_fulfilled(board, claim):
                accepted_claim_id = self._accept_candidate_claim(board, claim, run_ref)
                outcome = "claim_accepted" if accepted_claim_id else "claim_already_present"
        else:
            obligation.status = "blocked"
            obligation.failure_reason = "; ".join(validation.blocking_issues)
            # If one linked obligation fails, the candidate claim is blocked.  Do not keep
            # running sibling obligations for a claim that cannot currently be promoted.
            for sibling in board.obligations:
                if sibling.claim_id != claim.claim_id or sibling.obligation_id == obligation.obligation_id:
                    continue
                if sibling.status in {"open", "in_progress"}:
                    sibling.status = "blocked"
                    sibling.failure_reason = (
                        "Sibling obligation blocked because candidate claim failed obligation "
                        f"`{obligation.obligation_id}`."
                    )
                    sibling.updated_at = utc_now()
            claim.status = "blocked"
            claim.blocked_reason = obligation.failure_reason
            claim.updated_at = utc_now()
            outcome = "claim_blocked"
            accepted_claim_id = None

        self.board_manager.save(board)
        self._refresh_research_state_from_board(board)
        return {
            "outcome": outcome,
            "claim_id": claim.claim_id,
            "obligation_id": obligation.obligation_id,
            "accepted_claim_id": accepted_claim_id,
            "blocking_issues": validation.blocking_issues,
        }

    def _all_claim_obligations_fulfilled(
        self, board: ObligationBoard, claim: CandidateClaim
    ) -> bool:
        by_id = {obligation.obligation_id: obligation for obligation in board.obligations}
        return bool(claim.obligation_ids) and all(
            by_id.get(obligation_id) is not None
            and by_id[obligation_id].status == "fulfilled"
            for obligation_id in claim.obligation_ids
        )

    def _accept_candidate_claim(
        self, board: ObligationBoard, claim: CandidateClaim, run_ref: ArtifactRef
    ) -> str | None:
        candidate_norm = _normalize_statement(claim.statement)
        for accepted in self.store.latest_claims_by_id().values():
            if _normalize_statement(accepted.statement) == candidate_norm and is_claim_acceptably_supported(
                accepted, self.store
            ):
                claim.status = "proven"
                claim.blocked_reason = f"Duplicate of accepted claim `{accepted.claim_id}`."
                claim.updated_at = utc_now()
                return None

        evidence = list(claim.evidence)
        evidence.append(
            EvidenceRecord(
                evidence_type=EvidenceType.external_tool,
                summary=(
                    "All linked obligations for this candidate claim were fulfilled and passed "
                    "the deterministic scope/provenance, evidence, and consistency gates."
                ),
                artifact_refs=[run_ref, self.store.artifact_ref(ArtifactStore.OBLIGATION_BOARD)],
                verifier="CommitManager",
                confidence=1.0,
            )
        )
        record = ClaimRecord(
            claim_id=claim.claim_id,
            claim_type=claim.claim_type,
            statement=claim.statement,
            status=ClaimStatus.proved_informally,
            evidence=evidence,
            related_proposal_ids=[claim.source_proposal_id] if claim.source_proposal_id else [],
        )
        self.store.append_claims([record])
        claim.status = "proven"
        claim.blocked_reason = ""
        claim.updated_at = utc_now()
        return claim.claim_id

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


def _candidate_statement_from_proposal(proposal: ResearchProposal) -> str:
    if proposal.success_criteria:
        return f"Proposal `{proposal.title}` succeeds: {proposal.success_criteria[0]}"
    return f"Proposal `{proposal.title}` succeeds: {proposal.precise_goal}"


def _claim_type_from_proposal(proposal: ResearchProposal) -> ClaimType:
    if proposal.proposal_kind.value in {"positive_algorithm_attempt", "counterexample_search"}:
        return ClaimType.algorithmic
    if proposal.proposal_kind.value in {"barrier_analysis", "lemma_derivation", "formalization"}:
        return ClaimType.mathematical
    return ClaimType.literature


def _obligations_from_proposal(proposal: ResearchProposal, claim_id: str) -> list[ResearchObligation]:
    raw_items = []
    raw_items.extend(proposal.expected_intermediate_lemmas[:6])
    raw_items.extend(proposal.questions_to_answer[:4])
    if not raw_items:
        raw_items.extend(proposal.success_criteria[:4])
    if not raw_items:
        raw_items.append(proposal.precise_goal)
    obligations: list[ResearchObligation] = []
    seen: set[str] = set()
    for item in raw_items:
        statement = item.strip()
        key = _normalize_statement(statement)
        if not statement or key in seen:
            continue
        seen.add(key)
        kind = _classify_obligation_kind(statement)
        obligations.append(
            ResearchObligation(
                claim_id=claim_id,
                statement=statement,
                kind=kind,
                required_evidence=_required_evidence_for_kind(kind),
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
    unique: list[ArtifactRef] = []
    seen: set[str] = set()
    for ref in refs:
        if ref.path in seen:
            continue
        seen.add(ref.path)
        unique.append(ref)
    return unique


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
