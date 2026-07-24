"""Small, subsystem-neutral contracts for the research kernel.

The kernel knows how to schedule and persist actions.  It deliberately has no model of questions,
deliverables, obligations, or scientific completion.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from ..schemas import ArtifactRef, StrictModel, new_id, utc_now


class KernelPhase(str, Enum):
    running = "running"
    idle = "idle"
    error = "error"


class ActionStatus(str, Enum):
    proposed = "proposed"
    running = "running"
    succeeded = "succeeded"
    no_progress = "no_progress"
    failed = "failed"
    duplicate = "duplicate"
    interrupted = "interrupted"


class RecordKind(str, Enum):
    question = "question"
    conjecture = "conjecture"
    analysis = "analysis"
    synthesis = "synthesis"
    source = "source"
    source_quote = "source_quote"
    formal_theorem = "formal_theorem"
    experiment = "experiment"
    counterexample = "counterexample"
    obstruction = "obstruction"


class RecordStatus(str, Enum):
    """Epistemic status assigned by deterministic admission policy, never by a model."""

    tentative = "tentative"
    observed = "observed"
    verified = "verified"


class RecordRelation(str, Enum):
    none = "none"
    extends = "extends"
    supports = "supports"
    challenges = "challenges"
    answers = "answers"
    documents = "documents"
    replicates = "replicates"


class EvidenceType(str, Enum):
    model = "model"
    source_metadata = "source_metadata"
    source_quote = "source_quote"
    lean = "lean"
    execution = "execution"
    system = "system"


class EvidenceReceipt(StrictModel):
    """Machine-inspectable reason why a draft may receive a non-tentative status."""

    evidence_type: EvidenceType = EvidenceType.model
    details: dict[str, Any] = Field(default_factory=dict)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)


class RecordDraft(StrictModel):
    kind: RecordKind
    title: str = Field(min_length=3, max_length=300)
    summary: str = Field(min_length=3, max_length=3000)
    body: str = Field(default="", max_length=30_000)
    relation: RecordRelation = RecordRelation.none
    parent_ids: list[str] = Field(default_factory=list, max_length=30)
    evidence: EvidenceReceipt = Field(default_factory=EvidenceReceipt)

    @field_validator("parent_ids")
    @classmethod
    def unique_parents(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))


class ResearchRecord(StrictModel):
    record_id: str = Field(default_factory=lambda: new_id("record"))
    task_revision: int = Field(ge=1)
    producer: str
    action_id: str
    kind: RecordKind
    status: RecordStatus
    title: str
    summary: str
    body: str = ""
    relation: RecordRelation = RecordRelation.none
    parent_ids: list[str] = Field(default_factory=list)
    evidence_type: EvidenceType
    evidence_details: dict[str, Any] = Field(default_factory=dict)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    fingerprint: str
    created_at: str = Field(default_factory=utc_now)


class ActionProposal(StrictModel):
    subsystem: str
    action_type: str = Field(min_length=2, max_length=100)
    title: str = Field(min_length=3, max_length=300)
    rationale: str = Field(default="", max_length=3000)
    payload: dict[str, Any] = Field(default_factory=dict)
    parent_ids: list[str] = Field(default_factory=list, max_length=30)
    fingerprint: str = ""

    @field_validator("subsystem")
    @classmethod
    def safe_subsystem(cls, value: str) -> str:
        value = value.strip().lower()
        if not re.fullmatch(r"[a-z][a-z0-9_-]{1,63}", value):
            raise ValueError("subsystem must be a safe lowercase identifier")
        return value

    @model_validator(mode="after")
    def derive_fingerprint(self) -> "ActionProposal":
        if not self.fingerprint:
            self.fingerprint = content_fingerprint(
                {
                    "subsystem": self.subsystem,
                    "action_type": self.action_type,
                    "payload": self.payload,
                    "parents": sorted(set(self.parent_ids)),
                }
            )
        return self


class ActionOutcome(StrictModel):
    summary: str = Field(min_length=1, max_length=5000)
    records: list[RecordDraft] = Field(default_factory=list, max_length=30)
    state_patch: dict[str, Any] = Field(default_factory=dict)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    retryable: bool = False
    error: str = Field(default="", max_length=5000)


class ActionRecord(StrictModel):
    action_id: str = Field(default_factory=lambda: new_id("action"))
    cycle: int = Field(ge=1)
    task_revision: int = Field(ge=1)
    subsystem: str
    proposal: ActionProposal
    status: ActionStatus = ActionStatus.proposed
    summary: str = ""
    error: str = ""
    admitted_record_ids: list[str] = Field(default_factory=list)
    rejected_drafts: list[str] = Field(default_factory=list)
    run_dir: str = ""
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class KernelState(StrictModel):
    task_sha256: str
    task_revision: int = Field(default=1, ge=1)
    cycle: int = Field(default=0, ge=0)
    phase: KernelPhase = KernelPhase.running
    next_subsystem_index: int = Field(default=0, ge=0)
    active_action_id: str | None = None
    consecutive_idle: int = Field(default=0, ge=0)
    enabled_subsystems: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class RecordCard(StrictModel):
    record_id: str
    producer: str
    kind: RecordKind
    status: RecordStatus
    title: str
    summary: str
    relation: RecordRelation
    parent_ids: list[str]
    created_at: str


class ResearchView(StrictModel):
    task: str
    task_sha256: str
    task_revision: int
    cycle: int
    subsystem: str
    records: list[RecordCard] = Field(default_factory=list)
    subsystem_state: dict[str, Any] = Field(default_factory=dict)
    recent_actions: list[dict[str, Any]] = Field(default_factory=list)


class AdmissionResult(StrictModel):
    records: list[ResearchRecord] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)


def content_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def record_fingerprint(draft: RecordDraft) -> str:
    evidence_identity = _evidence_identity(draft.evidence)
    if draft.evidence.evidence_type not in {EvidenceType.model, EvidenceType.system}:
        # The same theorem, quote, paper, or exact execution is one record even if a model changes
        # its title, prose summary, theorem name, or intended parent link.
        return content_fingerprint(
            {
                "kind": draft.kind.value,
                "evidence_type": draft.evidence.evidence_type.value,
                "evidence_identity": evidence_identity,
            }
        )
    normalized = {
        "kind": draft.kind.value,
        "title": _normalize(draft.title),
        "summary": _normalize(draft.summary),
        "body": _normalize(draft.body),
        "relation": draft.relation.value,
        "parents": sorted(set(draft.parent_ids)),
        "evidence_type": draft.evidence.evidence_type.value,
    }
    return content_fingerprint(normalized)


def _evidence_identity(receipt: EvidenceReceipt) -> dict[str, Any]:
    details = receipt.details
    if receipt.evidence_type == EvidenceType.source_metadata:
        identifier = (
            details.get("doi")
            or details.get("arxiv_id")
            or details.get("citation_key")
            or details.get("paper_id")
        )
        return {"source": _normalize(str(identifier or ""))}
    if receipt.evidence_type == EvidenceType.source_quote:
        return {
            "citation_key": _normalize(str(details.get("citation_key") or "")),
            "quote": _normalize(str(details.get("quote") or "")),
            "source_sha256": details.get("source_sha256"),
        }
    if receipt.evidence_type == EvidenceType.lean:
        return {
            "statement": _normalize(str(details.get("statement") or "")),
            "environment_fingerprint": details.get("environment_fingerprint"),
        }
    if receipt.evidence_type == EvidenceType.execution:
        return {
            "program_sha256": details.get("program_sha256"),
            "output_sha256": details.get("output_sha256"),
        }
    return {}


def _normalize(value: str) -> str:
    return " ".join(value.lower().split())
