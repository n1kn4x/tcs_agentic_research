"""Typed schemas for the artifact-driven TCS research workflow.

All state-changing agent outputs should be represented by these Pydantic models and then
serialized to durable files by :mod:`tcs_agentic_research.artifact_store`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal, NotRequired, TypedDict
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


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
    report = "report"
    directory = "directory"
    other = "other"


class ArtifactRef(StrictModel):
    path: str
    kind: ArtifactKind = ArtifactKind.other
    sha256: str | None = None
    summary: str = ""
    created_at: str = Field(default_factory=utc_now)


class EvidenceType(str, Enum):
    lean_proof = "lean_proof"
    citation = "citation"
    experiment = "experiment"
    informal_argument = "informal_argument"
    counterexample = "counterexample"
    critic_review = "critic_review"
    external_tool = "external_tool"
    none = "none"


class EvidenceRecord(StrictModel):
    evidence_id: str = Field(default_factory=lambda: new_id("ev"))
    evidence_type: EvidenceType
    summary: str
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    citation_keys: list[str] = Field(default_factory=list)
    tool_result_ids: list[str] = Field(default_factory=list)
    verifier: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: str = Field(default_factory=utc_now)


class ClaimType(str, Enum):
    mathematical = "mathematical"
    algorithmic = "algorithmic"
    complexity = "complexity"
    resource = "resource"
    literature = "literature"
    novelty = "novelty"
    experimental = "experimental"
    definition = "definition"
    theorem_statement = "theorem_statement"
    other = "other"


class ClaimStatus(str, Enum):
    proposed = "proposed"
    needs_review = "needs_review"
    informal_argument = "informal_argument"
    conjecture = "conjecture"
    cited = "cited"
    experimentally_supported = "experimentally_supported"
    proved_by_lean = "proved_by_lean"
    proved_informally = "proved_informally"
    refuted = "refuted"
    duplicate = "duplicate"
    blocked = "blocked"
    withdrawn = "withdrawn"


class ClaimRecord(StrictModel):
    claim_id: str = Field(default_factory=lambda: new_id("claim"))
    claim_type: ClaimType
    statement: str
    normalized_statement: str = ""
    status: ClaimStatus = ClaimStatus.needs_review
    assumptions: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    depends_on_claim_ids: list[str] = Field(default_factory=list)
    supersedes_claim_ids: list[str] = Field(default_factory=list)
    related_proposal_ids: list[str] = Field(default_factory=list)
    related_report_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class NomenclatureEntry(StrictModel):
    symbol: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    definition: str = ""
    convention: str = ""
    source_refs: list[ArtifactRef] = Field(default_factory=list)


class LiteratureSource(StrictModel):
    source: str
    source_type: Literal["url", "arxiv", "doi", "pdf", "unknown"] = "unknown"
    citation_key: str = ""
    title: str = ""
    role: str = ""
    user_supplied: bool = True
    import_required: bool = True
    extract_text: bool = True


class InitializationBundle(StrictModel):
    research_task_markdown: str
    nomenclature_entries: list[NomenclatureEntry] = Field(default_factory=list)
    literature_sources: list[LiteratureSource] = Field(default_factory=list)
    initial_state_notes: list[str] = Field(default_factory=list)
    initial_claims: list[ClaimRecord] = Field(default_factory=list)
    fallback_publishable_outcomes: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)


class InitializationInterviewTurn(StrictModel):
    ready_to_initialize: bool = False
    assistant_message: str
    missing_information: list[str] = Field(default_factory=list)
    relevant_information: list[str] = Field(default_factory=list)
    rationale: str = ""


class ResearchState(StrictModel):
    task_id: str = Field(default_factory=lambda: new_id("task"))
    task_summary: str = ""
    solved: bool = False
    confirmed_by_replication: bool = False
    iteration: int = 0
    current_proposal_id: str | None = None
    active_claim_ids: list[str] = Field(default_factory=list)
    accepted_claim_ids: list[str] = Field(default_factory=list)
    rejected_claim_ids: list[str] = Field(default_factory=list)
    open_proof_obligations: list[str] = Field(default_factory=list)
    outcome_flags: list[str] = Field(default_factory=list)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    last_report_ref: ArtifactRef | None = None
    last_verdict_ref: ArtifactRef | None = None
    notes: list[str] = Field(default_factory=list)
    updated_at: str = Field(default_factory=utc_now)


class ProposalRisk(StrictModel):
    risk: str
    mitigation: str = ""
    severity: Literal["low", "medium", "high"] = "medium"


class ResearchProposal(StrictModel):
    proposal_id: str = Field(default_factory=lambda: new_id("proposal"))
    title: str
    precise_goal: str
    relevant_assumptions_and_model: list[str] = Field(default_factory=list)
    expected_intermediate_lemmas: list[str] = Field(default_factory=list)
    algorithmic_subgoals: list[str] = Field(default_factory=list)
    plausibility_argument: str = ""
    success_criteria: list[str] = Field(default_factory=list)
    partial_success_criteria: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    known_risks_and_barriers: list[ProposalRisk] = Field(default_factory=list)
    literature_queries: list[str] = Field(default_factory=list)
    resource_model: str = ""
    created_at: str = Field(default_factory=utc_now)


class CriticDecision(str, Enum):
    accept = "accept"
    revise = "revise"
    reject = "reject"


class ProposalCritique(StrictModel):
    decision: CriticDecision
    summary: str
    consistency_with_task: str
    plausibility: str
    barrier_risks: list[str] = Field(default_factory=list)
    missing_complexity_model: list[str] = Field(default_factory=list)
    unclear_success_criteria: list[str] = Field(default_factory=list)
    required_revisions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: str = Field(default_factory=utc_now)


class ProposalLedgerEntry(StrictModel):
    event_id: str = Field(default_factory=lambda: new_id("proposal_event"))
    proposal_id: str = ""
    event_type: Literal["generated", "revised", "accepted", "rejected", "critic_review"]
    proposal: ResearchProposal | None = None
    critique: ProposalCritique | None = None
    reason: str = ""
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class ReportOutcome(str, Enum):
    succeeded = "succeeded"
    partially_succeeded = "partially_succeeded"
    failed = "failed"
    negative_result = "negative_result"
    counterexample_found = "counterexample_found"
    needs_more_work = "needs_more_work"


class ComplexityEstimate(StrictModel):
    claim_id: str | None = None
    resource: str
    bound: str
    model: str
    assumptions: list[str] = Field(default_factory=list)
    derivation_summary: str = ""
    needs_derivation_review: bool = True


class LiteratureDependency(StrictModel):
    citation_key: str
    title: str = ""
    used_for: str
    provenance: str = ""
    supports_claim_ids: list[str] = Field(default_factory=list)
    notation_mappings: dict[str, str] = Field(default_factory=dict)


class ExperimentResult(StrictModel):
    run_id: str = Field(default_factory=lambda: new_id("run"))
    summary: str
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=list)
    supports_claim_ids: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class ProofObligation(StrictModel):
    obligation_id: str = Field(default_factory=lambda: new_id("obl"))
    statement: str
    claim_ids: list[str] = Field(default_factory=list)
    suggested_tool: Literal["lean", "informal", "literature", "experiment"] = "lean"
    status: Literal[
        "open",
        "in_progress",
        "proved",
        "experimentally_supported",
        "blocked",
        "refuted",
    ] = "open"
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)


class ResearchReport(StrictModel):
    report_id: str = Field(default_factory=lambda: new_id("report"))
    proposal_id: str = ""
    outcome: ReportOutcome
    executive_summary: str
    claims_generated: list[ClaimRecord] = Field(default_factory=list)
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    proof_obligations: list[ProofObligation] = Field(default_factory=list)
    complexity_estimates: list[ComplexityEstimate] = Field(default_factory=list)
    literature_dependencies: list[LiteratureDependency] = Field(default_factory=list)
    experimental_results: list[ExperimentResult] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)
    proposed_next_steps: list[str] = Field(default_factory=list)
    required_verifications: list[str] = Field(default_factory=list)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class ResearchCritique(StrictModel):
    accepted_claim_ids: list[str] = Field(default_factory=list)
    downgraded_claim_ids: list[str] = Field(default_factory=list)
    refuted_claim_ids: list[str] = Field(default_factory=list)
    forced_verifications: list[ProofObligation] = Field(default_factory=list)
    summary: str
    rejects_report: bool = False
    reasons: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class SolvedOutcome(str, Enum):
    solves_main_task = "solves_main_task"
    partial_progress = "partial_progress"
    publishable_side_result = "publishable_side_result"
    negative_result = "negative_result"
    counterexample_found = "counterexample_found"
    literature_duplicate = "literature_duplicate"
    needs_formalization = "needs_formalization"
    needs_complexity_review = "needs_complexity_review"
    needs_experiment = "needs_experiment"
    dead_end = "dead_end"


class SolvedVerdict(StrictModel):
    verdict_id: str = Field(default_factory=lambda: new_id("verdict"))
    outcomes: list[SolvedOutcome]
    possible_breakthrough: bool = False
    confirmed_solved: bool = False
    requires_independent_replication: bool = False
    rationale: str
    blocking_issues: list[str] = Field(default_factory=list)
    next_action: Literal[
        "continue", "independent_replication", "stop_confirmed", "await_user", "revise_task"
    ] = "continue"
    created_at: str = Field(default_factory=utc_now)


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
    """Exact quote-level provenance for literature-derived statements."""

    quote_id: str = Field(default_factory=lambda: new_id("quote"))
    citation_key: str = ""
    paper_id: str = ""
    locator: str = ""
    quote: str
    char_start: int | None = None
    char_end: int | None = None
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)


class LiteratureStatement(StrictModel):
    """A theorem/algorithm/lower-bound statement normalized to canonical notation."""

    statement_id: str = Field(default_factory=lambda: new_id("lit_stmt"))
    citation_key: str = ""
    paper_id: str = ""
    kind: Literal[
        "theorem",
        "lemma",
        "corollary",
        "proposition",
        "algorithm",
        "lower_bound",
        "definition",
        "claim",
        "other",
    ] = "other"
    label: str = ""
    title: str = ""
    original_statement: str
    mapped_statement: str = ""
    notation_mappings: dict[str, str] = Field(default_factory=dict)
    provenance: list[LiteratureQuote] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


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
    mapped_statement: str
    summary: str = ""
    score: float = 0.0
    provenance: list[LiteratureQuote] = Field(default_factory=list)
    notation_mappings: dict[str, str] = Field(default_factory=dict)
    duplicate_of: str | None = None


class LiteratureQueryAnswer(StrictModel):
    answer_id: str = Field(default_factory=lambda: new_id("lit_answer"))
    query: str
    answer: str
    results: list[LiteratureQueryResult] = Field(default_factory=list)
    duplicate_results: list[LiteratureDuplicateGroup] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    used_nomenclature: dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)


class LiteratureExtract(StrictModel):
    extract_id: str = Field(default_factory=lambda: new_id("lit_extract"))
    citation_key: str
    paper_id: str = ""
    text_artifact_ref: ArtifactRef | None = None
    extracted_claims: list[ClaimRecord] = Field(default_factory=list)
    theorem_statements: list[LiteratureStatement] = Field(default_factory=list)
    algorithm_statements: list[LiteratureStatement] = Field(default_factory=list)
    lower_bound_statements: list[LiteratureStatement] = Field(default_factory=list)
    notation_mappings: dict[str, str] = Field(default_factory=dict)
    new_nomenclature_entries: list[NomenclatureEntry] = Field(default_factory=list)
    provenance_notes: str = ""
    created_at: str = Field(default_factory=utc_now)


class LeanStatement(StrictModel):
    name: str
    statement: str
    imports: list[str] = Field(default_factory=lambda: ["TCSResearch.Basic"])
    namespace: str | None = "TCSResearch"


class ProofGoal(StrictModel):
    goal_id: str = Field(default_factory=lambda: new_id("goal"))
    lean_statement: LeanStatement
    status: Literal["open", "proved", "blocked", "failed"] = "open"
    parent_goal_ids: list[str] = Field(default_factory=list)


class LeanCompilerLog(StrictModel):
    command: list[str]
    cwd: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    success: bool = False
    artifact_ref: ArtifactRef | None = None
    created_at: str = Field(default_factory=utc_now)


class ProofDAGSummary(StrictModel):
    dag_id: str = Field(default_factory=lambda: new_id("dag"))
    root_goal_id: str
    open_goal_ids: list[str] = Field(default_factory=list)
    proved_goal_ids: list[str] = Field(default_factory=list)
    blocked_goal_ids: list[str] = Field(default_factory=list)
    accepted_decomposition_ids: list[str] = Field(default_factory=list)
    rejected_decomposition_ids: list[str] = Field(default_factory=list)


class TheoremProverResult(StrictModel):
    result_id: str = Field(default_factory=lambda: new_id("leap_result"))
    status: Literal["proved", "partially_proved", "failed", "needs_human_formalization"]
    root_goal: LeanStatement
    proved_artifacts: list[ArtifactRef] = Field(default_factory=list)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    open_goals: list[ProofGoal] = Field(default_factory=list)
    accepted_claims: list[str] = Field(default_factory=list)
    failed_claims: list[str] = Field(default_factory=list)
    proof_dag_summary: str = ""
    compiler_logs: list[LeanCompilerLog] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)



class ReplicationResult(StrictModel):
    result_id: str = Field(default_factory=lambda: new_id("replication"))
    verdict: Literal["verified", "partially_verified", "refuted", "needs_human_review"]
    summary: str
    independently_reconstructed_claim_ids: list[str] = Field(default_factory=list)
    failed_claim_ids: list[str] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class ModelProfile(StrictModel):
    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 4096
    task_types: list[str] = Field(default_factory=list)
    supports_tools: bool = False
    structured_output_mode: Literal[
        "guided_json", "json_schema", "json_schema_guided_json", "json_object"
    ] = "guided_json"
    strict_json_schema: bool = True
    extra_body: dict[str, Any] = Field(default_factory=dict)


class RouterSettings(StrictModel):
    default_task: str = "routine"
    timeout_seconds: float = 120.0
    max_retries: int = 1
    profiles: dict[str, ModelProfile]


class ExperimenterPiSettings(StrictModel):
    """Configuration for the Dockerized pi coding agent used by experiments."""

    provider: str = "experimenter-vllm"
    model: str
    base_url: str = "http://host.docker.internal:8000/v1"
    api_key: str = "EMPTY"
    api: Literal[
        "openai-completions",
        "openai-responses",
        "anthropic-messages",
        "google-generative-ai",
    ] = "openai-completions"
    thinking: str = "high"
    reasoning: bool = True
    context_window: int = 128000
    max_tokens: int = 32768
    compat: dict[str, Any] = Field(
        default_factory=lambda: {
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
        }
    )
    extra_args: list[str] = Field(default_factory=list)


class ExperimenterSettings(StrictModel):
    """Docker sandbox settings for the project-level experimenter container."""

    enabled: bool = True
    image: str = "tcs-agentic-research-experimenter:latest"
    dockerfile: str = ""
    network: str = "bridge"
    memory: str = "8g"
    cpus: float = 4.0
    timeout_seconds: int = 1800
    max_output_bytes: int = 2_000_000
    container_name_prefix: str = "tcs-exp"
    add_host_gateway: bool = True
    environment: dict[str, str] = Field(default_factory=dict)
    pi: ExperimenterPiSettings


class AppConfig(StrictModel):
    router: RouterSettings
    experimenter: ExperimenterSettings | None = None


class ModelCallRecord(StrictModel):
    call_id: str = Field(default_factory=lambda: new_id("model_call"))
    task_type: str
    profile_name: str
    model: str
    latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    structured_schema: str | None = None
    structured_output_valid: bool = False
    execution_mode: Literal["real", "dry_run"] = "real"
    used_mock_output: bool = False
    failure_modes: list[str] = Field(default_factory=list)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class GraphState(TypedDict):
    workspace: str
    task_id: NotRequired[str]
    initialized: NotRequired[bool]
    iteration: NotRequired[int]
    max_iterations: NotRequired[int]
    current_proposal_id: NotRequired[str | None]
    current_proposal_path: NotRequired[str | None]
    current_report_path: NotRequired[str | None]
    last_verdict_path: NotRequired[str | None]
    possible_breakthrough: NotRequired[bool]
    confirmed_solved: NotRequired[bool]
    solved: NotRequired[bool]
    stop_reason: NotRequired[str | None]
