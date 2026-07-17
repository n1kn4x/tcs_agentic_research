"""Typed contracts used internally by LEAP.

The graph and Lean compiler own proof state.  Models may propose plans and proof terms, but they
never assign node IDs, statuses, fingerprints, or artifact paths.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from ..schemas import LeanStatement, StrictModel


class OrStatus(str, Enum):
    open = "open"
    solving = "solving"
    proved = "proved"
    abandoned = "abandoned"


class AndStatus(str, Enum):
    open = "open"
    solving = "solving"
    proved = "proved"
    paused = "paused"
    rejected = "rejected"


class InformalProof(StrictModel):
    strategy: str = Field(min_length=1, max_length=8000)
    steps: list[str] = Field(default_factory=list, max_length=16)
    useful_concepts: list[str] = Field(default_factory=list, max_length=16)
    search_queries: list[str] = Field(default_factory=list, max_length=8)


class FormalProofCandidate(StrictModel):
    """A proof body only; declaration text is always rendered by the application."""

    proof: str = Field(min_length=2, max_length=30_000)
    # Retained as useful provenance when a formal prover wants to annotate its translation.  The
    # controller normally obtains the informal plan in a separate call.
    informal_proof: str = Field(default="", max_length=8000)
    notes: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("proof", mode="before")
    @classmethod
    def normalize_proof_term(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        text = value.strip()
        fenced = re.fullmatch(r"```(?:lean)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
        proof = fenced.group(1).strip() if fenced else text
        if not proof.startswith("by"):
            raise ValueError("proof must be one Lean proof term beginning with `by`")
        if re.search(
            r"(?m)^\s*(?:import|namespace|section|end|theorem|lemma|axiom|opaque|def)\b",
            proof,
        ):
            raise ValueError("proof must not contain Lean commands or declarations")
        return proof


class BlueprintChild(StrictModel):
    label: str = Field(min_length=1, max_length=80)
    statement: str = Field(
        min_length=1,
        max_length=4000,
        description="A closed Lean proposition with all variables explicitly bound.",
    )
    required: bool = True
    rationale: str = Field(min_length=1, max_length=1600)

    @field_validator("label")
    @classmethod
    def normalize_label(cls, value: str) -> str:
        label = re.sub(r"[^A-Za-z0-9_']+", "_", value.strip()).strip("_")
        if not label:
            raise ValueError("child label must contain an identifier character")
        if label[0].isdigit():
            label = "lemma_" + label
        return label[:80]

    @field_validator("statement")
    @classmethod
    def statement_is_type_only(cls, value: str) -> str:
        statement = value.strip()
        if re.match(
            r"^(?:```|theorem\b|lemma\b|example\b|def\b|axiom\b|import\b)",
            statement,
            flags=re.IGNORECASE,
        ):
            raise ValueError("child statement must be a proposition, not a declaration")
        if ":=" in statement or re.search(r"\b(?:sorry|admit)\b", statement):
            raise ValueError("child statement must not contain a proof or placeholder")
        return statement


class BlueprintCandidate(StrictModel):
    overview: str = Field(min_length=1, max_length=8000)
    parent_strategy: str = Field(min_length=1, max_length=6000)
    children: list[BlueprintChild] = Field(min_length=1, max_length=8)
    library_notes: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def has_a_required_child(self) -> "BlueprintCandidate":
        if not any(child.required for child in self.children):
            raise ValueError("a decomposition needs at least one required child")
        return self


class SketchCandidate(StrictModel):
    """The placeholder-free parent proof from a formal decomposition sketch."""

    parent_proof: str = Field(min_length=2, max_length=30_000)
    notes: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("parent_proof", mode="before")
    @classmethod
    def normalize_parent_proof(cls, value: object) -> object:
        # Reuse the stricter proof-body normalizer.
        if not isinstance(value, str):
            return value
        return FormalProofCandidate.normalize_proof_term(value)


class DecompositionReview(StrictModel):
    accept: bool
    score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list, max_length=10)
    suggested_direction: str = Field(default="", max_length=2400)


class LeanDiagnostic(StrictModel):
    severity: Literal["error", "warning", "information", "unknown"] = "unknown"
    message: str
    line: int | None = None
    column: int | None = None
    end_line: int | None = None
    end_column: int | None = None


class VerificationResult(StrictModel):
    accepted: bool
    reason: str = ""
    source_path: str = ""
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    diagnostics: list[LeanDiagnostic] = Field(default_factory=list)
    log_path: str = ""


class RetrievalHit(StrictModel):
    name: str
    statement: str
    source: str
    score: float
    module: str = ""
    proved_or_id: str | None = None


class ProofContext(StrictModel):
    node_id: str
    goal: LeanStatement
    environment_fingerprint: str
    user_context: str = ""
    ancestors: list[LeanStatement] = Field(default_factory=list)
    proved_lemmas: list[RetrievalHit] = Field(default_factory=list)
    library_results: list[RetrievalHit] = Field(default_factory=list)
    previous_failures: list[str] = Field(default_factory=list)
    existing_decompositions: list[str] = Field(default_factory=list)
    remaining_nodes: int
    remaining_seconds: int


class OrNode(StrictModel):
    node_id: str
    fingerprint: str
    goal: LeanStatement
    environment_fingerprint: str
    status: OrStatus
    proof_kind: Literal["direct", "decomposition"] | None = None
    proof_content: str = ""
    proof_artifact_path: str = ""
    selected_and_id: str | None = None


class AndChild(StrictModel):
    child_or_id: str
    required: bool
    position: int
    local_name: str


class AndNode(StrictModel):
    node_id: str
    parent_or_id: str
    fingerprint: str
    status: AndStatus
    blueprint: BlueprintCandidate
    parent_proof: str
    sketch_artifact_path: str
    review: DecompositionReview
    children: list[AndChild] = Field(default_factory=list)


class AttemptRecord(StrictModel):
    attempt_id: str
    or_id: str
    mode: str
    ordinal: int
    outcome: str
    candidate_sha256: str = ""
    candidate_artifact_path: str = ""
    diagnostics: list[LeanDiagnostic] = Field(default_factory=list)
    retrieval: list[RetrievalHit] = Field(default_factory=list)
    parent_attempt_id: str | None = None
    duration_seconds: float = 0.0
    note: str = ""


class RunRecord(StrictModel):
    run_id: str
    root_or_id: str
    target: LeanStatement
    status: str
    user_context: str = ""
    final_artifact_path: str = ""
