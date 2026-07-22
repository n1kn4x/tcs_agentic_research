"""Small, strict data contracts for the research engine and its subsystems.

The language model is never trusted with IDs, timestamps, evidence status, or artifact paths.
Those fields are assigned by deterministic application code after validation.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from enum import Enum
import re
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ---------------------------------------------------------------------------
# Artifacts and core research state
# ---------------------------------------------------------------------------


class ArtifactKind(str, Enum):
    markdown = "markdown"
    json = "json"
    jsonl = "jsonl"
    yaml = "yaml"
    lean = "lean"
    python = "python"
    sqlite = "sqlite"
    log = "log"
    directory = "directory"
    other = "other"


class ArtifactRef(StrictModel):
    path: str
    kind: ArtifactKind = ArtifactKind.other
    sha256: str | None = None
    summary: str = ""
    created_at: str = Field(default_factory=utc_now)


class WorkKind(str, Enum):
    literature = "literature"
    proof = "proof"
    experiment = "experiment"
    derivation = "derivation"
    synthesis = "synthesis"


class WorkStatus(str, Enum):
    open = "open"
    running = "running"
    done = "done"
    partial = "partial"
    blocked = "blocked"
    failed = "failed"
    superseded = "superseded"


class RequirementStatus(str, Enum):
    open = "open"
    in_progress = "in_progress"
    satisfied = "satisfied"
    blocked = "blocked"


class EvidenceRequirement(StrictModel):
    """One independently auditable gap. Progress is measured against these records."""

    requirement_id: str
    description: str
    acceptance_criteria: list[str] = Field(min_length=1, max_length=5)
    acceptable_methods: list[WorkKind] = Field(min_length=1, max_length=4)
    mandatory: bool = True
    status: RequirementStatus = RequirementStatus.open
    finding_ids: list[str] = Field(default_factory=list)
    attempted_strategy_fingerprints: list[str] = Field(default_factory=list)
    attempt_count: int = 0
    blocker: str = ""
    updated_at: str = Field(default_factory=utc_now)


class WorkItemDraft(StrictModel):
    """A falsifiable strategy for one evidence requirement."""

    question_id: str = Field(min_length=2, max_length=40)
    requirement_id: str = Field(min_length=4, max_length=60)
    kind: WorkKind
    title: str = Field(min_length=3, max_length=160)
    instruction: str = Field(min_length=10, max_length=4000)
    strategy: str = Field(min_length=5, max_length=1200)
    hypothesis: str = Field(min_length=5, max_length=1200)
    falsification_criterion: str = Field(min_length=5, max_length=1200)
    expected_information_gain: str = Field(min_length=5, max_length=1200)
    success_criteria: list[str] = Field(min_length=1, max_length=6)


class PlanSubmission(StrictModel):
    """The next small batch of non-duplicate strategies for explicit evidence gaps."""

    decision: Literal["continue", "review"] = "continue"
    objective: str = Field(min_length=3, max_length=1000)
    work_items: list[WorkItemDraft] = Field(default_factory=list, max_length=6)
    reason: str = Field(default="", max_length=1200)


class ResearchQuestionDraft(StrictModel):
    question: str = Field(min_length=10, max_length=1200)
    hypotheses: list[str] = Field(min_length=1, max_length=4)
    evidence_needed: list[str] = Field(min_length=1, max_length=8)
    preferred_methods: list[WorkKind] = Field(min_length=1, max_length=4)


class ResearchAgendaDraft(StrictModel):
    """A conservative decomposition of an uncertain user request."""

    objective: str = Field(min_length=10, max_length=2000)
    constraints: list[str] = Field(default_factory=list, max_length=20)
    questions: list[ResearchQuestionDraft] = Field(min_length=1, max_length=24)
    deliverables: list[str] = Field(min_length=1, max_length=30)


class ResearchQuestion(StrictModel):
    question_id: str
    question: str
    hypotheses: list[str]
    preferred_methods: list[WorkKind]
    requirements: list[EvidenceRequirement]
    finding_ids: list[str] = Field(default_factory=list)


class ResearchAgenda(StrictModel):
    task_sha256: str
    objective: str
    constraints: list[str] = Field(default_factory=list)
    questions: list[ResearchQuestion]
    deliverables: list[str]
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class WorkItem(StrictModel):
    work_id: str = Field(default_factory=lambda: new_id("work"))
    question_id: str
    requirement_id: str
    kind: WorkKind
    title: str
    instruction: str
    strategy: str
    hypothesis: str
    falsification_criterion: str
    expected_information_gain: str
    success_criteria: list[str]
    strategy_fingerprint: str
    parent_work_id: str | None = None
    revision: int = 0
    prior_result_ids: list[str] = Field(default_factory=list)
    status: WorkStatus = WorkStatus.open
    attempts: int = 0
    operational_failures: int = 0
    last_result_id: str | None = None
    blocked_reason: str = ""
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class WorkQueue(StrictModel):
    items: list[WorkItem] = Field(default_factory=list)
    updated_at: str = Field(default_factory=utc_now)


class ResearchPhase(str, Enum):
    planning = "planning"
    working = "working"
    needs_input = "needs_input"
    system_error = "system_error"
    complete = "complete"


class WorkspaceState(StrictModel):
    task_id: str = Field(default_factory=lambda: new_id("task"))
    task_sha256: str
    task_summary: str
    phase: ResearchPhase = ResearchPhase.planning
    cycle: int = 0
    plan_round: int = 0
    active_work_id: str | None = None
    last_result_id: str | None = None
    no_progress_steps: int = 0
    last_progress_cycle: int = 0
    contribution_count: int = 0
    diversification_count: int = 0
    human_replan_count: int = 0
    notes: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class FindingStatus(str, Enum):
    hypothesis = "hypothesis"
    observed = "observed"
    supported = "supported"
    derived = "derived"
    verified = "verified"
    refuted = "refuted"


class FindingPolarity(str, Enum):
    supports = "supports"
    contradicts = "contradicts"
    null = "null"
    characterizes = "characterizes"
    inconclusive = "inconclusive"


class EvidenceStrength(str, Enum):
    preliminary = "preliminary"
    substantive = "substantive"
    strong = "strong"
    conclusive = "conclusive"


class Finding(StrictModel):
    finding_id: str = Field(default_factory=lambda: new_id("finding"))
    work_id: str
    question_id: str
    requirement_id: str
    kind: WorkKind
    statement: str
    status: FindingStatus
    polarity: FindingPolarity = FindingPolarity.characterizes
    strength: EvidenceStrength = EvidenceStrength.preliminary
    scope: str = ""
    evidence_refs: list[ArtifactRef] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class CriterionResult(StrictModel):
    criterion: str = Field(min_length=3, max_length=1200)
    satisfied: bool
    detail: str = Field(min_length=3, max_length=1600)


class WorkResult(StrictModel):
    result_id: str = Field(default_factory=lambda: new_id("result"))
    work_id: str
    outcome: Literal["done", "partial", "blocked", "failed"]
    progress: Literal["meaningful", "none"] = "none"
    failure_class: Literal[
        "none", "operational", "engineering", "method", "evidence_gap", "invalid"
    ] = "none"
    attempt_class: Literal["engineering", "scientific"] = "scientific"
    continue_work: bool = False
    evidence_level: Literal["none", "preliminary", "substantive", "conclusive"] = "none"
    requirement_satisfied: bool = False
    criteria: list[CriterionResult] = Field(default_factory=list)
    summary: str
    findings: list[Finding] = Field(default_factory=list)
    contribution_ids: list[str] = Field(default_factory=list)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_evidence_claim(self) -> "WorkResult":
        if self.requirement_satisfied:
            if not self.findings:
                raise ValueError("a satisfied requirement needs at least one finding")
            if self.criteria and any(not item.satisfied for item in self.criteria):
                raise ValueError("a satisfied requirement cannot have a failed criterion")
            if self.evidence_level not in {"substantive", "conclusive"}:
                raise ValueError("a satisfied requirement needs substantive evidence")
        return self


class Contribution(StrictModel):
    contribution_id: str = Field(default_factory=lambda: new_id("contribution"))
    fingerprint: str
    work_id: str
    result_id: str
    question_id: str
    requirement_id: str
    kind: Literal[
        "positive_result", "negative_result", "null_result", "characterization",
        "verified_subgoal", "source_evidence", "derived_result",
    ]
    summary: str
    finding_ids: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class AnalysisClaim(StrictModel):
    statement: str = Field(min_length=3, max_length=1200)
    basis_finding_ids: list[str] = Field(default_factory=list, max_length=8)
    caveat: str = Field(default="", max_length=800)


class AnalysisSubmission(StrictModel):
    summary: str = Field(min_length=10, max_length=5000)
    candidate_claims: list[AnalysisClaim] = Field(default_factory=list, max_length=6)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=8)
    suggested_next_steps: list[str] = Field(default_factory=list, max_length=6)


class DerivationStep(StrictModel):
    label: str = Field(min_length=1, max_length=80)
    statement: str = Field(min_length=5, max_length=1600)
    justification: str = Field(min_length=5, max_length=2000)
    depends_on: list[str] = Field(default_factory=list, max_length=8)


class DerivationSubmission(StrictModel):
    title: str = Field(min_length=5, max_length=200)
    result_kind: Literal[
        "theorem", "bound", "counterexample", "equivalence", "obstruction", "characterization"
    ]
    target_claim: str = Field(min_length=10, max_length=1600)
    assumptions: list[str] = Field(min_length=1, max_length=12)
    definitions: list[str] = Field(default_factory=list, max_length=12)
    steps: list[DerivationStep] = Field(min_length=2, max_length=20)
    conclusion: str = Field(min_length=10, max_length=2000)
    falsification_attempt: str = Field(min_length=10, max_length=2000)
    limitations: list[str] = Field(min_length=1, max_length=10)


class DerivationReview(StrictModel):
    accepted: bool
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=10, max_length=2000)
    criteria: list[CriterionResult] = Field(min_length=1, max_length=12)
    fatal_issues: list[str] = Field(default_factory=list, max_length=10)
    attempted_counterexample: str = Field(min_length=5, max_length=2000)
    required_revisions: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def consistent_verdict(self) -> "DerivationReview":
        if self.accepted and (self.fatal_issues or any(not item.satisfied for item in self.criteria)):
            raise ValueError("accepted derivation review has a fatal issue or failed criterion")
        if not self.accepted and not (self.fatal_issues or self.required_revisions):
            raise ValueError("rejected derivation review must explain what failed")
        return self


class LiteraturePlan(StrictModel):
    search_queries: list[str] = Field(min_length=1, max_length=4)
    known_source_titles: list[str] = Field(default_factory=list, max_length=4)
    focus_questions: list[str] = Field(min_length=1, max_length=4)


class LiteratureSelection(StrictModel):
    support_id: str
    relevant: bool
    relation: Literal["supports", "contradicts", "characterizes", "unrelated"]
    rationale: str = Field(min_length=5, max_length=800)


class LiteratureEvidenceReview(StrictModel):
    selections: list[LiteratureSelection] = Field(default_factory=list, max_length=20)
    search_assessment: str = Field(min_length=10, max_length=1600)
    next_queries: list[str] = Field(default_factory=list, max_length=5)


class ProofGoalReview(StrictModel):
    accepted: bool
    relevance: str = Field(min_length=5, max_length=1200)
    route_to_requirement: str = Field(min_length=5, max_length=1200)
    issues: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def rejected_goal_has_issues(self) -> "ProofGoalReview":
        if not self.accepted and not self.issues:
            raise ValueError("rejected proof goal must name a relevance or formulation issue")
        return self


class LeanGoalDraft(StrictModel):
    name: str = Field(
        min_length=1,
        max_length=80,
        description="One unqualified Lean identifier, without `theorem` or binders.",
    )
    statement: str = Field(
        min_length=1,
        max_length=2000,
        description=(
            "Only the theorem type with every variable explicitly bound; never a theorem/lemma "
            "declaration and never a proof. Example: `∀ (a b : Bool), (a && b) = (b && a)`."
        ),
    )
    namespace: str | None = "TCSResearch"

    @model_validator(mode="before")
    @classmethod
    def unwrap_common_declaration_form(cls, value: Any) -> Any:
        """Conservatively recover a type when a model wraps it in one Lean declaration."""
        if not isinstance(value, dict) or not isinstance(value.get("statement"), str):
            return value
        data = dict(value)
        parsed = _parse_lean_declaration(data["statement"])
        if parsed is not None:
            name, statement = parsed
            data["name"] = name
            data["statement"] = statement
        return data

    @field_validator("name")
    @classmethod
    def validate_declaration_name(cls, value: str) -> str:
        name = value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_']*", name):
            raise ValueError("name must be one unqualified Lean identifier")
        return name

    @field_validator("statement")
    @classmethod
    def validate_type_only_statement(cls, value: str) -> str:
        statement = value.strip()
        if re.match(
            r"^(?:```|theorem\b|lemma\b|example\b|def\b|axiom\b|import\b|"
            r"variables?\b|section\b|namespace\b|open\b)",
            statement,
            flags=re.IGNORECASE,
        ):
            raise ValueError(
                "statement must contain only the theorem type, not a Lean declaration or code fence"
            )
        if ":=" in statement or re.search(r"\b(?:sorry|admit)\b", statement):
            raise ValueError("statement must not contain a proof body or placeholder")
        return _parenthesize_bool_equality(statement)


def _parenthesize_bool_equality(statement: str) -> str:
    """Disambiguate a common Lean precedence trap without otherwise rewriting the goal."""
    prefix = ""
    body = statement
    if statement.startswith("∀") or statement.startswith("forall "):
        split = _first_top_level_character(statement, ",")
        if split < 0:
            return statement
        prefix, body = statement[: split + 1], statement[split + 1 :].strip()
    equality = _first_top_level_character(body, "=")
    if equality < 0 or _first_top_level_character(body[equality + 1 :], "=") >= 0:
        return statement
    left = body[:equality].strip()
    right = body[equality + 1 :].strip()
    if not left or not right or not any(token in left + right for token in ("&&", "||", "!")):
        return statement
    if any(token in right for token in ("→", "↔", "<->")):
        return statement
    separator = " " if prefix else ""
    return f"{prefix}{separator}({left}) = ({right})"


def _first_top_level_character(text: str, wanted: str) -> int:
    depth = 0
    for index, character in enumerate(text):
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth = max(0, depth - 1)
        elif character == wanted and depth == 0:
            if wanted != "=" or (index == 0 or text[index - 1] not in ("!", ":", "<", ">")):
                return index
    return -1


def _parse_lean_declaration(value: str) -> tuple[str, str] | None:
    """Parse only a simple, single `theorem`/`lemma` wrapper; leave all other text untouched."""
    text = value.strip()
    match = re.match(r"^(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_']*)\s*(.*)$", text)
    if match is None:
        return None
    name, remainder = match.groups()
    depth = 0
    separator = -1
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = set(pairs.values())
    for index, character in enumerate(remainder):
        if character in pairs:
            depth += 1
        elif character in closing:
            depth = max(0, depth - 1)
        elif character == ":" and depth == 0:
            separator = index
            break
    if separator < 0:
        return None
    binders = remainder[:separator].strip()
    statement = remainder[separator + 1 :].strip()
    if ":=" in statement:
        statement = statement.split(":=", 1)[0].strip()
    if not statement:
        return None
    if binders:
        statement = f"∀ {binders}, {statement}"
    return name, statement


class NamedDescription(StrictModel):
    """A stable identifier plus human-readable scientific meaning."""

    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
    description: str = Field(min_length=3, max_length=1200)


class ExperimentProtocol(StrictModel):
    """The scientific design is frozen and reviewed before code is generated."""

    title: str = Field(min_length=5, max_length=200)
    hypothesis: str = Field(min_length=10, max_length=1200)
    null_outcome: str = Field(min_length=10, max_length=1200)
    experimental_unit: str = Field(min_length=3, max_length=600)
    conditions: list[NamedDescription] = Field(min_length=2, max_length=12)
    baselines: list[NamedDescription] = Field(min_length=1, max_length=8)
    metrics: list[NamedDescription] = Field(min_length=1, max_length=12)
    correctness_checks: list[NamedDescription] = Field(min_length=1, max_length=10)
    sample_sizes: list[int] = Field(min_length=1, max_length=10)
    seeds: list[int] = Field(min_length=1, max_length=20)
    analysis_plan: str = Field(min_length=10, max_length=2000)
    decision_rule: str = Field(min_length=10, max_length=1200)
    wall_seconds: int = Field(ge=1, le=604800)
    memory_mb: int = Field(ge=64, le=262144)
    cpus: float = Field(gt=0, le=256)
    known_limitations: list[str] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def unique_component_ids(self) -> "ExperimentProtocol":
        for label, values in [
            ("conditions", self.conditions),
            ("baselines", self.baselines),
            ("metrics", self.metrics),
            ("correctness_checks", self.correctness_checks),
        ]:
            ids = [value.id for value in values]
            if len(ids) != len(set(ids)):
                raise ValueError(f"{label} must use unique ids")
        condition_ids = {value.id for value in self.conditions}
        baseline_ids = {value.id for value in self.baselines}
        missing = sorted(baseline_ids - condition_ids)
        if missing:
            raise ValueError("every baseline id must also name a condition: " + ", ".join(missing))
        if baseline_ids == condition_ids:
            raise ValueError(
                "baseline conditions must be a proper subset of conditions; include at least one "
                "distinct treatment condition"
            )
        return self


class ExperimentCriterionAssessment(StrictModel):
    criterion_id: str = Field(pattern=r"^[A-Z][A-Z0-9_]{1,63}$")
    satisfied: bool
    detail: str = Field(min_length=3, max_length=1600)


class ExperimentProtocolReview(StrictModel):
    """Python computes acceptance from exact criterion-id coverage and these assessments."""

    criteria: list[ExperimentCriterionAssessment] = Field(min_length=1, max_length=20)
    issues: list[str] = Field(default_factory=list, max_length=10)
    required_revisions: list[str] = Field(default_factory=list, max_length=10)


class ExperimentProgramReview(StrictModel):
    accepted: bool
    objective_alignment: str = Field(min_length=3, max_length=1000)
    issues: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def rejected_program_has_issues(self) -> "ExperimentProgramReview":
        if not self.accepted and not self.issues:
            self.issues = [self.objective_alignment]
        return self


class ExperimentEvidenceReview(StrictModel):
    usable: Literal["full", "preliminary", "unusable"]
    outcome: Literal["supports", "contradicts", "null", "inconclusive", "characterizes"]
    scientific_summary: str = Field(min_length=10, max_length=2400)
    criteria: list[ExperimentCriterionAssessment] = Field(min_length=1, max_length=20)
    issues: list[str] = Field(default_factory=list, max_length=10)
    caveats: list[str] = Field(default_factory=list, max_length=10)
    follow_up: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def consistent_usability(self) -> "ExperimentEvidenceReview":
        failed = any(not item.satisfied for item in self.criteria)
        if self.usable == "full" and (failed or self.issues):
            raise ValueError("fully usable evidence cannot have failed criteria or fatal issues")
        if self.usable == "unusable" and not self.issues:
            self.issues = [
                (self.caveats or self.follow_up or [self.scientific_summary])[0]
            ]
        if self.usable == "preliminary" and not self.follow_up:
            self.follow_up = [
                "Extend the sound pilot to every acceptance criterion and requested regime."
            ]
        return self


class ExperimentCheck(StrictModel):
    name: str = Field(min_length=2, max_length=200)
    passed: bool
    detail: str = Field(default="", max_length=1000)


JSONScalar = str | int | float | bool | None
JSONParameter = JSONScalar | list[JSONScalar]


class ExperimentObservation(StrictModel):
    condition: str = Field(min_length=1, max_length=200)
    sample_size: int = Field(ge=1)
    metrics: dict[str, JSONScalar] = Field(min_length=1, max_length=40)


class ExperimentConclusion(StrictModel):
    hypothesis: str = Field(min_length=5, max_length=1200)
    outcome: Literal["supports", "contradicts", "null", "inconclusive", "characterizes"]
    basis_metrics: list[str] = Field(min_length=1, max_length=12)
    statement: str = Field(min_length=10, max_length=1600)


class ExperimentOutput(StrictModel):
    """Required output: raw condition-level observations are never discarded."""

    schema_version: Literal[2] = 2
    experiment: str = Field(min_length=3, max_length=500)
    status: Literal["completed", "capped"] = "completed"
    parameters: dict[str, JSONParameter] = Field(min_length=1, max_length=50)
    aggregate_metrics: dict[str, JSONScalar] = Field(min_length=1, max_length=200)
    observations: list[ExperimentObservation] = Field(default_factory=list, max_length=1000)
    checks: list[ExperimentCheck] = Field(min_length=1, max_length=200)
    conclusion: ExperimentConclusion
    limitations: list[str] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def finite_metrics_and_distinct_conditions(self) -> "ExperimentOutput":
        import math

        values: list[JSONScalar] = [*self.aggregate_metrics.values()]
        for value in self.parameters.values():
            values.extend(value if isinstance(value, list) else [value])
        for observation in self.observations:
            values.extend(observation.metrics.values())
        if any(isinstance(value, float) and not math.isfinite(value) for value in values):
            raise ValueError("experiment metrics must be finite")
        return self


class ExperimentImplementationPlan(StrictModel):
    """A bounded reasoning pass before emitting a complete experiment source file."""

    approach: str = Field(min_length=10, max_length=2000)
    components: list[str] = Field(min_length=1, max_length=12)
    correctness_strategy: str = Field(min_length=10, max_length=2000)
    output_strategy: str = Field(min_length=10, max_length=1600)
    defect_repairs: list[str] = Field(default_factory=list, max_length=12)


class ExperimentProgram(StrictModel):
    """Model-generated scientific implementation; trusted code owns execution and output writing."""

    description: str = Field(min_length=10, max_length=1500)
    source: str = Field(
        min_length=20,
        max_length=20_000,
        description=(
            "Python source defining run_experiment(mode: str) -> dict. The function returns the "
            "ExperimentOutput v2 payload; a trusted wrapper writes results.json."
        ),
    )
    seeds: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=20)

    @field_validator("source", mode="before")
    @classmethod
    def normalize_source(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        normalized = re.sub(r"^```(?:python|py)?\s*\n", "", normalized)
        normalized = re.sub(r"\n```$", "", normalized).strip()
        # Large generated programs leave too little request budget for focused repairs. Python's
        # own parser/unparser removes comments and formatting while preserving executable semantics.
        if len(normalized) > 16_000:
            try:
                normalized = ast.unparse(ast.parse(normalized))
            except SyntaxError:
                pass
        return normalized.strip()

    @property
    def python_code(self) -> str:
        return self.source


# ---------------------------------------------------------------------------
# Literature records
# ---------------------------------------------------------------------------


class LiteratureSource(StrictModel):
    source: str
    source_type: Literal["url", "arxiv", "doi", "pdf", "unknown"] = "unknown"
    citation_key: str = ""
    title: str = ""
    role: str = ""
    extract_text: bool = True


class LiteratureCandidate(StrictModel):
    candidate_id: str = Field(default_factory=lambda: new_id("cand"))
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    openalex_id: str = ""
    abstract: str = ""
    landing_url: str = ""
    pdf_url: str = ""
    source_urls: list[str] = Field(default_factory=list)
    cited_by_count: int = 0
    discovery_reason: str = ""
    score: float = 0.0
    status: Literal["queued", "imported", "rejected", "duplicate"] = "queued"
    imported_paper_id: str | None = None
    created_at: str = Field(default_factory=utc_now)


class PaperMetadata(StrictModel):
    paper_id: str = Field(default_factory=lambda: new_id("paper"))
    citation_key: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str = ""
    url: str = ""
    arxiv_id: str = ""
    doi: str = ""
    abstract: str = ""
    source_type: Literal["manual", "url", "arxiv", "doi", "pdf"] = "manual"
    source_urls: list[str] = Field(default_factory=list)
    pdf_path: str = ""
    text_path: str = ""
    metadata_path: str = ""
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    imported_at: str = Field(default_factory=utc_now)


class LiteratureQuote(StrictModel):
    quote_id: str = Field(default_factory=lambda: new_id("quote"))
    citation_key: str = ""
    paper_id: str = ""
    locator: str = ""
    quote: str
    char_start: int | None = None
    char_end: int | None = None
    source_sha256: str = ""
    validated: bool = False
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)


class LiteratureStatement(StrictModel):
    statement_id: str = Field(default_factory=lambda: new_id("lit_stmt"))
    support_id: str = ""
    citation_key: str = ""
    paper_id: str = ""
    kind: Literal[
        "theorem", "lemma", "corollary", "proposition", "algorithm",
        "lower_bound", "definition", "claim", "other",
    ] = "other"
    label: str = ""
    title: str = ""
    original_statement: str
    statement_text: str = ""
    provenance: list[LiteratureQuote] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class LiteratureExtract(StrictModel):
    extract_id: str = Field(default_factory=lambda: new_id("lit_extract"))
    citation_key: str
    paper_id: str = ""
    text_artifact_ref: ArtifactRef | None = None
    theorem_statements: list[LiteratureStatement] = Field(default_factory=list)
    algorithm_statements: list[LiteratureStatement] = Field(default_factory=list)
    lower_bound_statements: list[LiteratureStatement] = Field(default_factory=list)
    provenance_notes: str = ""
    created_at: str = Field(default_factory=utc_now)


class LiteratureDuplicateGroup(StrictModel):
    duplicate_id: str = Field(default_factory=lambda: new_id("lit_dup"))
    result_ids: list[str]
    canonical_key: str
    reason: str = ""


class LiteratureQueryResult(StrictModel):
    result_id: str = Field(default_factory=lambda: new_id("lit_result"))
    citation_key: str
    paper_id: str = ""
    title: str = ""
    year: int | None = None
    kind: str = ""
    label: str = ""
    statement_text: str
    summary: str = ""
    score: float = 0.0
    statement_id: str = ""
    quote_id: str = ""
    support_id: str = ""
    support_level: str = ""
    relation: str = ""
    provenance: list[LiteratureQuote] = Field(default_factory=list)
    duplicate_of: str | None = None


class LiteratureQueryAnswer(StrictModel):
    answer_id: str = Field(default_factory=lambda: new_id("lit_answer"))
    query: str
    answer: str
    results: list[LiteratureQueryResult] = Field(default_factory=list)
    duplicate_results: list[LiteratureDuplicateGroup] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


# ---------------------------------------------------------------------------
# Lean and experiments
# ---------------------------------------------------------------------------


class LeanStatement(StrictModel):
    name: str
    statement: str
    imports: list[str] = Field(default_factory=lambda: ["TCSResearch.Basic"])
    namespace: str | None = "TCSResearch"

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        name = value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_']*", name):
            raise ValueError("Lean theorem name must be one unqualified identifier")
        return name

    @field_validator("statement")
    @classmethod
    def closed_type_source(cls, value: str) -> str:
        statement = value.strip()
        if not statement:
            raise ValueError("Lean statement cannot be empty")
        if ":=" in statement or re.search(r"\b(?:sorry|admit)\b", statement):
            raise ValueError("Lean statement must not contain a proof body or placeholder")
        if re.search(
            r"(?m)^\s*(?:import|namespace|section|end|theorem|lemma|axiom|def|opaque)\b",
            statement,
        ):
            raise ValueError("Lean statement must contain only one proposition type")
        return statement

    @field_validator("imports")
    @classmethod
    def valid_imports(cls, values: list[str]) -> list[str]:
        for value in values:
            if not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*", value
            ):
                raise ValueError(f"invalid Lean module import: {value!r}")
        return values

    @field_validator("namespace")
    @classmethod
    def valid_namespace(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*", value
        ):
            raise ValueError("invalid Lean namespace")
        return value


class ProofGoal(StrictModel):
    goal_id: str = Field(default_factory=lambda: new_id("goal"))
    lean_statement: LeanStatement
    status: Literal["open", "proved", "blocked", "failed"] = "open"


class LeanCompilerLog(StrictModel):
    command: list[str]
    cwd: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    success: bool = False
    artifact_ref: ArtifactRef | None = None
    created_at: str = Field(default_factory=utc_now)


class TheoremProverResult(StrictModel):
    result_id: str = Field(default_factory=lambda: new_id("lean_result"))
    status: Literal["proved", "partial", "exhausted", "unavailable"]
    root_goal: LeanStatement
    proved_artifacts: list[ArtifactRef] = Field(default_factory=list)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    open_goals: list[ProofGoal] = Field(default_factory=list)
    proof_dag_summary: str = ""
    compiler_logs: list[LeanCompilerLog] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class ExperimentResult(StrictModel):
    run_id: str = Field(default_factory=lambda: new_id("experiment"))
    success: bool = False
    failure_class: Literal["none", "infrastructure", "program", "contract"] = "none"
    summary: str
    validated_output: ExperimentOutput | None = None
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class ExperimentState(StrictModel):
    """Durable state machine for one cumulative experimental research strategy."""

    work_id: str
    stage: Literal[
        "protocol_design",
        "protocol_review",
        "protocol_revision",
        "program_design",
        "program_review",
        "program_revision",
        "smoke_execution",
        "full_execution",
        "evidence_review",
        "complete",
    ] = "protocol_design"
    protocol: ExperimentProtocol | None = None
    protocol_sha256: str = ""
    protocol_review: ExperimentProtocolReview | None = None
    program: ExperimentProgram | None = None
    program_review: ExperimentProgramReview | None = None
    smoke_result: ExperimentResult | None = None
    execution_result: ExperimentResult | None = None
    final_result: WorkResult | None = None
    last_error: str = ""
    engineering_failures: int = 0
    engineering_blocked: bool = False
    scientific_attempts: int = 0
    protocol_revision: int = 0
    program_revision: int = 0
    last_protocol_candidate_sha256: str = ""
    repeated_protocol_candidates: int = 0
    last_program_candidate_sha256: str = ""
    repeated_program_candidates: int = 0
    repeated_defect_failures: int = 0
    last_defect_signature: str = ""
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


# ---------------------------------------------------------------------------
# Configuration and model-call telemetry
# ---------------------------------------------------------------------------


class ModelProfile(StrictModel):
    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.6
    max_tokens: int = 8192
    task_types: list[str] = Field(default_factory=list)
    extra_body: dict[str, Any] = Field(default_factory=dict)


class RouterSettings(StrictModel):
    default_profile: str = "reasoning"
    repair_profile: str = "format"
    timeout_seconds: float = 600.0
    max_input_chars: int = 30000
    max_output_tokens: int = 8192
    repair_attempts: int = Field(default=1, ge=0, le=1)
    profiles: dict[str, ModelProfile]


class CoreSettings(StrictModel):
    max_model_calls_per_step: int = Field(default=12, ge=1, le=32)
    max_plan_items: int = Field(default=4, ge=1, le=6)
    # This threshold triggers diversification; it never halts while untried requirements remain.
    max_no_progress_steps: int = Field(default=4, ge=2, le=100)
    max_operational_retries: int = Field(default=2, ge=0, le=5)
    max_experiment_engineering_retries: int = Field(default=4, ge=1, le=20)
    max_strategy_revisions: int = Field(default=2, ge=0, le=5)
    max_method_attempts_per_requirement: int = Field(default=4, ge=1, le=12)
    literature_max_imports: int = Field(default=3, ge=0, le=10)
    literature_import_attempts: int = Field(default=8, ge=1, le=20)
    literature_results_per_query: int = Field(default=5, ge=1, le=10)


class LeapSettings(StrictModel):
    """Budgets for one resumable LEAP search invocation.

    These are safety limits, not claims that a theorem should finish within one invocation.  The
    SQLite graph preserves all verified progress for later invocations.
    """

    max_model_calls_per_run: int = Field(default=64, ge=1, le=100_000)
    direct_attempts_per_node: int = Field(default=2, ge=0, le=20)
    direct_revisions: int = Field(default=5, ge=0, le=30)
    blueprint_attempts_per_node: int = Field(default=3, ge=0, le=20)
    sketch_revisions: int = Field(default=4, ge=0, le=30)
    max_depth: int = Field(default=20, ge=1, le=100)
    max_nodes: int = Field(default=500, ge=1, le=100_000)
    max_children: int = Field(default=6, ge=1, le=20)
    max_wall_seconds: int = Field(default=21_600, ge=1, le=1_209_600)
    compiler_timeout_seconds: int = Field(default=300, ge=5, le=7200)
    compiler_memory_mb: int = Field(default=16384, ge=1024, le=262_144)
    reviewer_min_score: float = Field(default=0.55, ge=0.0, le=1.0)


class ExperimenterSettings(StrictModel):
    enabled: bool = True
    image: str = "tcs-agentic-research-experimenter:v2"
    dockerfile: str = ""
    network: str = "none"
    memory: str = "4g"
    cpus: float = 2.0
    timeout_seconds: int = 600
    max_output_bytes: int = 500_000
    container_name_prefix: str = "tcs-exp"
    add_host_gateway: bool = False
    environment: dict[str, str] = Field(default_factory=dict)


class AppConfig(StrictModel):
    router: RouterSettings
    core: CoreSettings = Field(default_factory=CoreSettings)
    leap: LeapSettings = Field(default_factory=LeapSettings)
    experimenter: ExperimenterSettings | None = None


class ModelCallRecord(StrictModel):
    call_id: str = Field(default_factory=lambda: new_id("model_call"))
    step_id: str = ""
    task_type: str
    profile_name: str
    model: str
    input_chars: int
    latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    structured_schema: str | None = None
    valid: bool = False
    execution_mode: Literal["real", "dry_run"] = "real"
    failure: str = ""
    created_at: str = Field(default_factory=utc_now)
