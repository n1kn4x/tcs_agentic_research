"""Small, strict data contracts for the research engine and its subsystems.

The language model is never trusted with IDs, timestamps, evidence status, or artifact paths.
Those fields are assigned by deterministic application code after validation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from pathlib import PurePosixPath
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
    analysis = "analysis"


class WorkStatus(str, Enum):
    open = "open"
    running = "running"
    done = "done"
    partial = "partial"
    blocked = "blocked"
    failed = "failed"


class WorkItemDraft(StrictModel):
    """Model-authored bounded unit of work. IDs and status are application-owned."""

    kind: WorkKind
    title: str = Field(min_length=3, max_length=160)
    instruction: str = Field(min_length=10, max_length=3000)
    success_criteria: list[str] = Field(default_factory=list, max_length=4)


class PlanSubmission(StrictModel):
    """A deliberately small planning response: at most four independent work items."""

    decision: Literal["continue", "review"] = "continue"
    objective: str = Field(min_length=3, max_length=1000)
    work_items: list[WorkItemDraft] = Field(default_factory=list, max_length=4)
    reason: str = Field(default="", max_length=1200)

    @model_validator(mode="after")
    def work_kinds_are_unique(self) -> "PlanSubmission":
        kinds = [item.kind for item in self.work_items]
        if len(kinds) != len(set(kinds)):
            raise ValueError(
                "work_items must contain at most one item of each kind; combine dependent "
                "work into one self-contained bounded item"
            )
        return self


class WorkItem(StrictModel):
    work_id: str = Field(default_factory=lambda: new_id("work"))
    kind: WorkKind
    title: str
    instruction: str
    success_criteria: list[str] = Field(default_factory=list)
    status: WorkStatus = WorkStatus.open
    attempts: int = 0
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
    review = "review"
    needs_input = "needs_input"


class WorkspaceState(StrictModel):
    task_id: str = Field(default_factory=lambda: new_id("task"))
    task_sha256: str
    task_summary: str
    phase: ResearchPhase = ResearchPhase.planning
    cycle: int = 0
    plan_round: int = 0
    active_work_id: str | None = None
    last_result_id: str | None = None
    notes: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class FindingStatus(str, Enum):
    hypothesis = "hypothesis"
    observed = "observed"
    supported = "supported"
    verified = "verified"
    refuted = "refuted"


class Finding(StrictModel):
    finding_id: str = Field(default_factory=lambda: new_id("finding"))
    work_id: str
    kind: WorkKind
    statement: str
    status: FindingStatus
    evidence_refs: list[ArtifactRef] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class WorkResult(StrictModel):
    result_id: str = Field(default_factory=lambda: new_id("result"))
    work_id: str
    outcome: Literal["done", "partial", "blocked", "failed"]
    summary: str
    findings: list[Finding] = Field(default_factory=list)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
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


class LiteraturePlan(StrictModel):
    search_queries: list[str] = Field(default_factory=list, min_length=1, max_length=3)
    known_source_titles: list[str] = Field(default_factory=list, max_length=3)
    focus_questions: list[str] = Field(default_factory=list, min_length=1, max_length=4)


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
    imports: list[str] = Field(default_factory=lambda: ["TCSResearch.Basic"], max_length=6)
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
            r"^(?:```|theorem\b|lemma\b|example\b|def\b|axiom\b|import\b)",
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


class ExperimentProgram(StrictModel):
    description: str = Field(min_length=10, max_length=1500)
    python_lines: list[str] = Field(
        min_length=1,
        max_length=500,
        description=(
            "Complete Python source as an ordered array containing exactly one source line per "
            "element. Preserve leading indentation; do not use Markdown fences."
        ),
    )
    seeds: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=20)
    expected_outputs: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("python_lines", mode="before")
    @classmethod
    def normalize_python_lines(cls, value: Any) -> Any:
        lines = value.splitlines() if isinstance(value, str) else value
        if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
            return lines
        normalized = list(lines)
        if normalized and re.fullmatch(r"```(?:python|py)?\s*", normalized[0].strip()):
            normalized.pop(0)
        if normalized and normalized[-1].strip() == "```":
            normalized.pop()
        return normalized

    @property
    def python_code(self) -> str:
        return "\n".join(self.python_lines).strip()

    @field_validator("expected_outputs")
    @classmethod
    def validate_expected_output_paths(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            path = PurePosixPath(value.strip())
            if not value.strip() or path.is_absolute() or ".." in path.parts:
                raise ValueError("expected_outputs must be relative paths inside the run directory")
            normalized.append(str(path))
        return normalized


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
    summary: str
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


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
    max_model_calls_per_step: int = Field(default=5, ge=1, le=8)
    max_plan_rounds: int = Field(default=3, ge=1, le=10)
    max_plan_items: int = Field(default=4, ge=1, le=4)
    literature_max_imports: int = Field(default=3, ge=0, le=10)
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
