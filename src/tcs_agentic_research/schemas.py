"""Strict contracts shared by independent low-level services.

Research-kernel contracts live in :mod:`tcs_agentic_research.core.models`.  This module contains
only artifact, literature, Lean, experiment-execution, configuration, and telemetry types.
"""

from __future__ import annotations

import ast
import json
import math
import re
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


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


# ---------------------------------------------------------------------------
# Generic experiment execution.  There is intentionally no global study blueprint.
# ---------------------------------------------------------------------------


JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list[JSONScalar] | dict[str, JSONScalar | list[JSONScalar]]


class ExperimentObservation(StrictModel):
    unit_id: str | int
    condition: str = Field(default="default", min_length=1, max_length=200)
    values: dict[str, JSONValue] = Field(min_length=1, max_length=100)


class ExperimentOutput(StrictModel):
    """Raw measurements emitted by a self-contained experiment.

    The contract deliberately has no pass bit, expected direction, or model-owned validity field.
    Execution proves only that this exact program produced these exact measurements.
    """

    schema_version: Literal[1] = 1
    experiment: str = Field(min_length=3, max_length=500)
    status: Literal["completed", "capped"] = "completed"
    protocol: str = Field(min_length=10, max_length=5000)
    parameters: dict[str, JSONValue] = Field(default_factory=dict, max_length=100)
    observations: list[ExperimentObservation] = Field(min_length=1, max_length=20_000)
    summaries: dict[str, JSONValue] = Field(default_factory=dict, max_length=200)
    interpretation: str = Field(min_length=10, max_length=5000)
    limitations: list[str] = Field(min_length=1, max_length=30)

    @model_validator(mode="after")
    def finite_json_measurements(self) -> "ExperimentOutput":
        try:
            encoded = json.dumps(self.model_dump(mode="json"), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("experiment output must be finite JSON") from exc
        if len(encoded) > 10_000_000:
            raise ValueError("experiment output exceeds the 10 MB structured-data limit")
        for observation in self.observations:
            for value in _walk_values(observation.values):
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("experiment measurements must be finite")
        return self


class ExperimentProgram(StrictModel):
    description: str = Field(min_length=10, max_length=2000)
    source: str = Field(
        min_length=20,
        max_length=30_000,
        description=(
            "Python source defining run_experiment(mode: str) -> dict. The dict must match "
            "ExperimentOutput schema version 1; the trusted wrapper writes results.json."
        ),
    )
    seeds: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=100)

    @field_validator("source", mode="before")
    @classmethod
    def normalize_source(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        normalized = re.sub(r"^```(?:python|py)?\s*\n", "", normalized)
        normalized = re.sub(r"\n```$", "", normalized).strip()
        if len(normalized) > 14_000:
            try:
                normalized = ast.unparse(ast.parse(normalized))
            except SyntaxError:
                pass
        return normalized

    @property
    def python_code(self) -> str:
        return self.source


class ExperimentResult(StrictModel):
    run_id: str = Field(default_factory=lambda: new_id("experiment"))
    success: bool = False
    failure_class: Literal["none", "infrastructure", "program", "contract"] = "none"
    summary: str
    validated_output: ExperimentOutput | None = None
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


def _walk_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [leaf for child in value.values() for leaf in _walk_values(child)]
    if isinstance(value, list):
        return [leaf for child in value for leaf in _walk_values(child)]
    return [value]


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
# Lean proof records
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


# ---------------------------------------------------------------------------
# Configuration and telemetry
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
    max_input_chars: int = 50000
    max_output_tokens: int = 8192
    repair_attempts: int = Field(default=1, ge=0, le=1)
    profiles: dict[str, ModelProfile]


class CoreSettings(StrictModel):
    max_model_calls_per_action: int = Field(default=128, ge=1, le=100_000)
    record_context_limit: int = Field(default=80, ge=10, le=1000)
    literature_max_imports: int = Field(default=2, ge=0, le=10)


class LeapSettings(StrictModel):
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
