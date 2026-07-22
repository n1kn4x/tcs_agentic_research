from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from tcs_agentic_research.agents.experiment import ExperimentAgent
from tcs_agentic_research.agents.literature import LiteratureResearcher
from tcs_agentic_research.artifact_store import ArtifactStore
from tcs_agentic_research.cli import main
from tcs_agentic_research.engine import ResearchEngine, _methods_for_requirement
from tcs_agentic_research.pipelines.experiment import (
    _criterion_id_errors,
    _evidence_output_context,
    _protocol_output_errors,
    _review_errors,
)
from tcs_agentic_research.pipelines.literature import _preserves_required_acronyms
from tcs_agentic_research.workflow import (
    _new_contributions,
    _normalize_work_draft,
    _validate_experiment_program,
)
from tcs_agentic_research.experimenter.docker_project import _diagnostic
from tcs_agentic_research.experimenter.runner import _normalize_output_payload
from tcs_agentic_research.experimenter.errors import ExperimenterConfigurationError
from tcs_agentic_research.leap.harness import FormalProofCandidate, LEAPHarness
from tcs_agentic_research.leap.lean import LeanVerifier
from tcs_agentic_research.llm import (
    InputBudgetExceeded,
    LLMRouter,
    ModelBudgetExceeded,
    StructuredLLMError,
)
from tcs_agentic_research.schemas import (
    AnalysisSubmission,
    EvidenceRequirement,
    EvidenceStrength,
    ExperimentConclusion,
    ExperimentCriterionAssessment,
    ExperimentObservation,
    ExperimentOutput,
    ExperimentProgram,
    ExperimentProtocol,
    ExperimentProtocolReview,
    ExperimentState,
    Finding,
    FindingPolarity,
    FindingStatus,
    LeanGoalDraft,
    LeanStatement,
    ModelProfile,
    NamedDescription,
    PaperMetadata,
    PlanSubmission,
    ResearchPhase,
    ResearchQuestion,
    RouterSettings,
    WorkItem,
    WorkItemDraft,
    WorkKind,
    WorkStatus,
)


def _task(tmp_path: Path, text: str = "# Task\nAudit primary literature with exact quotes.\n") -> Path:
    (tmp_path / ArtifactStore.RESEARCH_TASK).write_text(text, encoding="utf-8")
    return tmp_path


def _router(store: ArtifactStore, *, max_input_chars: int = 30_000) -> LLMRouter:
    return LLMRouter(
        RouterSettings(
            default_profile="reasoning",
            repair_profile="reasoning",
            max_input_chars=max_input_chars,
            profiles={"reasoning": ModelProfile(model="mock")},
        ),
        store=store,
        dry_run=True,
    )


def test_all_agent_profiles_share_one_four_gpu_qwen_endpoint() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config.example.yml").read_text())
    profiles = config["router"]["profiles"]

    assert {profile["model"] for profile in profiles.values()} == {"qwen-research"}
    assert {profile["base_url"] for profile in profiles.values()} == {
        "http://localhost:8000/v1"
    }

    compose = yaml.safe_load((root / "docker-compose.vllm.yml").read_text())
    assert list(compose["services"]) == ["qwen-research"]
    service = compose["services"]["qwen-research"]
    assert "CUDA_VISIBLE_DEVICES=${QWEN_GPUS:-0,1,2,3}" in service["environment"]
    assert "${QWEN_TP:-4}" in service["command"]

    script = root / "scripts" / "launch_vllm_stack.sh"
    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0
    script_text = script.read_text()
    assert 'QWEN_GPUS="${QWEN_GPUS:-0,1,2,3}"' in script_text
    assert "ROUTINE_GPUS" not in script_text
    assert "PROOF_GPUS" not in script_text


def test_initialization_creates_only_new_core_artifacts(tmp_path: Path) -> None:
    engine = ResearchEngine(workspace=_task(tmp_path), dry_run=True)
    state = engine.initialize()

    assert state.task_sha256
    assert (tmp_path / "State.json").exists()
    assert (tmp_path / "Queue.json").exists()
    assert (tmp_path / "Events.jsonl").exists()
    assert (tmp_path / "Contributions.jsonl").exists()
    assert not (tmp_path / "Nomenclature.yml").exists()
    assert not (tmp_path / "ResearchState.json").exists()
    assert not (tmp_path / "ClaimLedger.jsonl").exists()
    assert not (tmp_path / "GraphCheckpoints.sqlite").exists()


def test_dry_run_plans_bounded_items_and_executes_one(tmp_path: Path) -> None:
    task = """# Literature test
Audit SETH lower bounds for Boolean vectors with exact source provenance.
Use the Literature subsystem and separate supported claims from gaps.
"""
    engine = ResearchEngine(workspace=_task(tmp_path, task), dry_run=True)
    status = engine.run(max_steps=1)
    queue = engine.store.load_queue()

    assert 1 <= len(queue.items) <= 4
    assert queue.items[0].kind == WorkKind.literature
    assert queue.items[0].status == WorkStatus.blocked
    assert all(item.kind != WorkKind.synthesis for item in queue.items)
    assert all(item.kind != WorkKind.proof for item in queue.items)
    assert (tmp_path / "Agenda.json").exists()
    assert (tmp_path / "Reports" / "Progress.md").exists()
    assert status["state"]["cycle"] == 1
    assert list((tmp_path / "Runs").glob("*/input.json"))
    assert list((tmp_path / "Runs").glob("*/result.json"))


def test_no_evidence_run_stops_visibly_instead_of_repeating_generic_work(
    tmp_path: Path,
) -> None:
    task = "# Literature gap\nFind an exact primary-source quote about a fictional theorem.\n"
    engine = ResearchEngine(workspace=_task(tmp_path, task), dry_run=True)

    status = engine.run(max_steps=20)

    assert status["state"]["phase"] == "needs_input"
    assert status["finding_counts"]["hypothesis"] == 0
    assert status["state"]["diversification_count"] >= 1
    progress = (tmp_path / "Reports" / "Progress.md").read_text()
    assert "Action required" in progress
    assert "All configured methods reached their attempt caps" in progress

    engine.replan()
    resumed = engine.run(max_steps=1)
    assert resumed["state"]["phase"] == "working"
    assert resumed["state"]["human_replan_count"] == 1


def test_full_pipeline_plan_splits_subsystems(tmp_path: Path) -> None:
    task = """# Integration
Use LiteratureDB for context, the Experimenter for a fixed-seed benchmark,
LEAP/Lean for one formal proof, and provide a careful analysis.
"""
    engine = ResearchEngine(workspace=_task(tmp_path, task), dry_run=True)
    engine.run(max_steps=1)
    kinds = {item.kind for item in engine.store.load_queue().items}

    assert kinds == {
        WorkKind.literature,
        WorkKind.experiment,
        WorkKind.proof,
        WorkKind.derivation,
    }
    assert WorkKind.synthesis not in kinds


def test_experiment_pipeline_accumulates_durable_dry_run_stages(tmp_path: Path) -> None:
    task = """# Experiment
Run an experimental benchmark with fixed seeds and condition-level measurements.
"""
    engine = ResearchEngine(workspace=_task(tmp_path, task), dry_run=True)

    engine.run(max_steps=8)

    state_files = sorted((tmp_path / "ExperimentStates").glob("*.json"))
    assert 1 <= len(state_files) <= 8
    states = [ExperimentState.model_validate_json(path.read_text()) for path in state_files]
    assert all(state.stage == "complete" for state in states)
    assert all(state.protocol_sha256 for state in states)
    assert all(state.program is not None for state in states)
    agenda = engine.store.load_agenda()
    assert agenda is not None
    assert all(
        requirement.attempt_count == 0
        for question in agenda.questions
        for requirement in question.requirements
    )


def test_experiment_reviews_require_exact_stable_criterion_ids() -> None:
    expected = {"P_ALIGNMENT": "aligned", "P_NULL": "null rule"}
    assessments = [
        ExperimentCriterionAssessment(
            criterion_id="P_ALIGNMENT", satisfied=True, detail="Directly aligned."
        ),
        ExperimentCriterionAssessment(
            criterion_id="P_EXTRA", satisfied=True, detail="Unexpected generic row."
        ),
    ]

    errors = _criterion_id_errors(expected, assessments)

    assert any("Missing criterion IDs: P_NULL" in error for error in errors)
    assert any("Unexpected criterion IDs: P_EXTRA" in error for error in errors)


def test_protocol_requires_a_distinct_treatment_condition() -> None:
    conditions = [
        NamedDescription(id="n8", description="Eight-bit factor code."),
        NamedDescription(id="n16", description="Sixteen-bit factor code."),
    ]
    with pytest.raises(ValueError, match="proper subset"):
        ExperimentProtocol(
            title="Invalid all-baseline comparison",
            hypothesis="The factor code expands uniform values.",
            null_outcome="No expansion is observed.",
            experimental_unit="one sampled integer",
            conditions=conditions,
            baselines=conditions,
            metrics=[NamedDescription(id="bits", description="Encoded bits.")],
            correctness_checks=[
                NamedDescription(id="roundtrip", description="Decode equals input.")
            ],
            sample_sizes=[10],
            seeds=[1],
            analysis_plan="Compare mean encoded bits to literal coding.",
            decision_rule="Classify from the signed paired difference.",
            wall_seconds=30,
            memory_mb=256,
            cpus=1,
            known_limitations=["Small pilot."],
        )


def test_failed_protocol_assessment_detail_becomes_repair_input() -> None:
    review = ExperimentProtocolReview(
        criteria=[
            ExperimentCriterionAssessment(
                criterion_id="P_BASELINES",
                satisfied=False,
                detail="Add binary_n8 as a separate baseline condition.",
            )
        ]
    )

    errors = _review_errors({"P_BASELINES": "Distinct baseline."}, review)

    assert errors == [
        "P_BASELINES: Add binary_n8 as a separate baseline condition."
    ]


def test_empirical_requirements_cannot_fall_back_to_derivation() -> None:
    methods = _methods_for_requirement(
        "Experimental measurement of compression ratio on a real dataset.",
        [WorkKind.experiment, WorkKind.derivation],
    )

    assert methods == [WorkKind.experiment]


def test_generic_mathematical_formalization_routes_to_derivation_not_lean() -> None:
    methods = _methods_for_requirement(
        "A formalization and NP-hardness proof for weighted multiset cover.",
        [WorkKind.proof, WorkKind.derivation],
    )

    assert methods == [WorkKind.derivation]


def test_interrupted_running_item_is_reopened_on_restart(tmp_path: Path) -> None:
    engine = ResearchEngine(workspace=_task(tmp_path), dry_run=True)
    engine.run(max_steps=1)
    queue = engine.store.load_queue()
    item = queue.items[0]
    item.status = WorkStatus.running
    state = engine.store.load_state()
    assert state is not None
    state.active_work_id = item.work_id
    engine.store.save_queue(queue)
    engine.store.save_state(state)

    engine.run(max_steps=0)

    recovered = next(row for row in engine.store.load_queue().items if row.work_id == item.work_id)
    assert recovered.status == WorkStatus.open
    assert engine.store.load_state().active_work_id is None  # type: ignore[union-attr]


def test_task_edit_resets_phase_to_planning_without_erasing_history(tmp_path: Path) -> None:
    engine = ResearchEngine(workspace=_task(tmp_path), dry_run=True)
    first = engine.initialize()
    first.phase = ResearchPhase.needs_input
    engine.store.save_state(first)
    (tmp_path / ArtifactStore.RESEARCH_TASK).write_text("# Changed task\nProve a Boolean lemma.\n")

    changed = engine.initialize()

    assert changed.phase == "planning"
    assert list((tmp_path / "Archive").glob("*/State.json"))
    events = engine.store.read_jsonl(ArtifactStore.EVENT_LEDGER)
    assert events[-1]["event_type"] == "task_changed"


def test_router_enforces_hard_step_call_budget(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize_layout()
    router = LLMRouter(
        RouterSettings(
            default_profile="reasoning",
            repair_profile="reasoning",
            profiles={"reasoning": ModelProfile(model="mock")},
        ),
        store=store,
    )

    def fake_post(profile: ModelProfile, body: dict[str, object]) -> dict[str, object]:
        return {
            "choices": [{"message": {"content": json.dumps({"summary": "A valid bounded summary."})}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }

    router._post = fake_post  # type: ignore[method-assign]
    with router.step_budget("test", max_calls=1):
        result = router.complete_structured(
            task_type="analysis",
            messages=[{"role": "user", "content": "summarize"}],
            schema=AnalysisSubmission,
        )
        assert result.summary.startswith("A valid")
        with pytest.raises(ModelBudgetExceeded):
            router.complete_structured(
                task_type="analysis",
                messages=[{"role": "user", "content": "again"}],
                schema=AnalysisSubmission,
            )


def test_router_rejects_oversize_context_before_http(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize_layout()
    router = LLMRouter(
        RouterSettings(
            default_profile="reasoning",
            repair_profile="reasoning",
            max_input_chars=40,
            profiles={"reasoning": ModelProfile(model="mock")},
        ),
        store=store,
    )
    with pytest.raises(InputBudgetExceeded):
        router.complete_structured(
            task_type="analysis",
            messages=[{"role": "user", "content": "x" * 100}],
            schema=AnalysisSubmission,
        )


def test_structured_repair_is_fresh_and_bounded(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize_layout()
    router = LLMRouter(
        RouterSettings(
            default_profile="reasoning",
            repair_profile="reasoning",
            repair_attempts=1,
            profiles={"reasoning": ModelProfile(model="mock")},
        ),
        store=store,
    )
    bodies: list[dict[str, object]] = []

    def fake_post(profile: ModelProfile, body: dict[str, object]) -> dict[str, object]:
        bodies.append(body)
        content = "not json" if len(bodies) == 1 else json.dumps({"summary": "Repaired conservatively."})
        return {"choices": [{"message": {"content": content}}]}

    router._post = fake_post  # type: ignore[method-assign]
    with router.step_budget("repair", max_calls=2):
        result = router.complete_structured(
            task_type="analysis",
            messages=[{"role": "user", "content": "original context"}],
            schema=AnalysisSubmission,
        )

    assert result.summary == "Repaired conservatively."
    repair_messages = bodies[1]["messages"]
    assert isinstance(repair_messages, list)
    assert len(repair_messages) == 2
    assert "original context" not in json.dumps(repair_messages)
    calls = store.read_jsonl(ArtifactStore.MODEL_LEDGER)
    assert [record["valid"] for record in calls] == [False, True]
    assert "structured_output_invalid" in calls[0]["failure"]


def test_structured_code_call_can_disable_lossy_formatter_repair(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize_layout()
    router = LLMRouter(
        RouterSettings(
            default_profile="reasoning",
            repair_profile="reasoning",
            profiles={"reasoning": ModelProfile(model="mock")},
        ),
        store=store,
    )
    call_count = 0

    def fake_post(profile: ModelProfile, body: dict[str, object]) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        return {"choices": [{"message": {"content": "not json"}}]}

    router._post = fake_post  # type: ignore[method-assign]
    with router.step_budget("no_lossy_repair", max_calls=2):
        with pytest.raises(StructuredLLMError):
            router.complete_structured(
                task_type="experiment_design",
                messages=[{"role": "user", "content": "write code"}],
                schema=ExperimentProgram,
                allow_repair=False,
            )
    assert call_count == 1


def test_no_model_tool_interface_is_emitted(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize_layout()
    router = LLMRouter(
        RouterSettings(
            default_profile="reasoning",
            repair_profile="reasoning",
            profiles={"reasoning": ModelProfile(model="mock")},
        ),
        store=store,
    )
    captured: dict[str, object] = {}

    def fake_post(profile: ModelProfile, body: dict[str, object]) -> dict[str, object]:
        captured.update(body)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "decision": "review",
                                "objective": "Review current evidence.",
                                "work_items": [],
                                "reason": "No action.",
                            }
                        )
                    }
                }
            ]
        }

    router._post = fake_post  # type: ignore[method-assign]
    router.complete_structured(
        task_type="planning",
        messages=[{"role": "user", "content": "plan"}],
        schema=PlanSubmission,
    )

    assert "tools" not in captured
    assert captured["max_tokens"] == 8192
    assert captured["response_format"]


def test_literature_acceptance_requires_explicit_acronym_anchors() -> None:
    item = WorkItem.model_construct(
        instruction="Quote the OV lower bound under SETH rather than ETH.",
        hypothesis="The OV result depends on SETH.",
    )

    assert _preserves_required_acronyms(
        item,
        "Assuming SETH, the Orthogonal Vectors (OV) bound is not stated under ETH.",
    )
    assert _preserves_required_acronyms(
        item,
        "The Orthogonal Vectors bound assumes the Strong Exponential Time Hypothesis, not ETH.",
    )
    assert not _preserves_required_acronyms(
        item,
        "Assuming SETH, Bichromatic Closest Pair requires quadratic time.",
    )


def test_deterministic_literature_extraction_has_stable_exact_support(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize_layout()
    agent = LiteratureResearcher(store, _router(store))
    text = (
        "--- page 1 ---\n\n"
        "Theorem 1. Assuming SETH, Orthogonal Vectors in logarithmic dimension has no "
        "truly subquadratic algorithm.\n\nProof. Omitted in this smoke record.\n"
    )
    text_ref = store.write_text("LiteratureDB/papers/OVSmoke/paper.txt", text)
    paper = agent.import_paper(
        PaperMetadata(
            citation_key="OVSmoke",
            title="OV Smoke",
            text_path=text_ref.path,
        )
    )

    first = agent.extract_paper(citation_key=paper.citation_key, use_llm=False)
    second = agent.extract_paper(citation_key=paper.citation_key, use_llm=False)
    first_statement = first.lower_bound_statements[0]
    second_statement = second.lower_bound_statements[0]

    assert first_statement.statement_id == second_statement.statement_id
    assert first_statement.support_id == second_statement.support_id
    assert first_statement.provenance[0].validated
    assert first_statement.provenance[0].char_start is not None
    assert len(store.read_jsonl("LiteratureDB/statements.jsonl")) == sum(
        len(group)
        for group in [
            second.theorem_statements,
            second.algorithm_statements,
            second.lower_bound_statements,
        ]
    )
    assert not store.exists("LiteratureDB/extracted_claims.jsonl")
    assert not store.exists("LiteratureDB/query_answers.jsonl")

    answer = agent.answer_query("SETH logarithmic dimension", limit=3)
    assert answer.results
    assert answer.results[0].support_id == first_statement.support_id
    assert answer.results[0].provenance[0].validated
    assert answer.results[0].provenance[0].artifact_refs[0].path == text_ref.path


def test_literature_index_rebuilds_from_three_canonical_ledgers(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize_layout()
    agent = LiteratureResearcher(store, _router(store))
    text_ref = store.write_text(
        "LiteratureDB/papers/Test/paper.txt",
        "Lemma 2. Boolean equality is reflexive.\n",
    )
    agent.import_paper(PaperMetadata(citation_key="Test", title="Test", text_path=text_ref.path))
    extract = agent.extract_paper(citation_key="Test", use_llm=False)
    support_id = extract.theorem_statements[0].support_id

    store.resolve("LiteratureDB/index.sqlite").unlink()
    rebuilt = LiteratureResearcher(store, _router(store))

    assert rebuilt.index.support_exists(support_id)
    assert store.exists("LiteratureDB/papers.jsonl")
    assert store.exists("LiteratureDB/statements.jsonl")


def test_generated_code_and_proof_contracts_are_application_bound(tmp_path: Path) -> None:
    program = ExperimentProgram(
        description="A fenced model response is normalized before syntax validation.",
        source=(
            "```python\n"
            "import json\n\n"
            "def run_experiment(mode: str) -> dict:\n"
            "    return {'schema_version': 2, 'mode': mode}\n"
            "```"
        ),
    )
    assert program.python_code.startswith("import json\n\ndef run_experiment")
    _validate_experiment_program(program)
    with pytest.raises(ValueError, match="must branch on mode"):
        _validate_experiment_program(
            ExperimentProgram(
                description="A smoke path must be explicit rather than silently running full scale.",
                source="def run_experiment(mode: str) -> dict:\n    return {}",
            )
        )
    with pytest.raises(ValueError, match="must define run_experiment"):
        _validate_experiment_program(
            ExperimentProgram(
                description="A helper without the required trusted entry function is rejected.",
                source="def helper(mode: str) -> dict:\n    return {}",
            )
        )
    with pytest.raises(ValueError, match="trusted wrapper owns the entry point"):
        _validate_experiment_program(
            ExperimentProgram(
                description="Generated source may not execute itself at module import time.",
                source=(
                    "def run_experiment(mode: str) -> dict:\n"
                    "    return {'mode': mode}\n\n"
                    "run_experiment('full')"
                ),
            )
        )
    with pytest.raises(ValueError, match="seeds field"):
        _validate_experiment_program(
            ExperimentProgram(
                description="Seed provenance must match an explicit source constant when present.",
                source=(
                    "SEEDS = [1]\n\n"
                    "def run_experiment(mode: str) -> dict:\n"
                    "    return {'mode': mode}"
                ),
                seeds=[2],
            )
        )
    _validate_experiment_program(
        ExperimentProgram(
            description="An inert command-line convenience guard is safe under trusted import.",
            source=(
                "import sys\n\n"
                "def run_experiment(mode: str) -> dict:\n"
                "    return {'mode': mode}\n\n"
                "if __name__ == '__main__':\n"
                "    sys.exit(0)"
            ),
        )
    )
    with pytest.raises(ValueError, match="sys.exit"):
        _validate_experiment_program(
            ExperimentProgram(
                description="A helper reachable by the experiment may not terminate the wrapper.",
                source=(
                    "import sys\n\n"
                    "def stop():\n"
                    "    sys.exit(1)\n\n"
                    "def run_experiment(mode: str) -> dict:\n"
                    "    stop()\n"
                    "    return {'mode': mode}"
                ),
            )
        )

    normalized_goal = LeanGoalDraft(
        name="target",
        statement="theorem other (a : Bool) : a = a := by sorry",
    )
    assert normalized_goal.name == "other"
    assert normalized_goal.statement == "∀ (a : Bool), a = a"
    precedence_goal = LeanGoalDraft(
        name="and_comm",
        statement="∀ (a b : Bool), a && b = b && a",
    )
    assert precedence_goal.statement == "∀ (a b : Bool), (a && b) = (b && a)"
    inequality_goal = LeanGoalDraft(name="neq", statement="∀ (a b : Bool), a != b")
    assert inequality_goal.statement == "∀ (a b : Bool), a != b"
    with pytest.raises(ValueError, match="only the theorem type"):
        LeanGoalDraft(name="target", statement="```lean\ntheorem other : True\n```")
    with pytest.raises(ValueError, match="must not contain Lean commands"):
        FormalProofCandidate(
            informal_proof="Attempt to replace the declaration.",
            proof="by\n  exact True.intro\ntheorem unrelated : True := by trivial",
        )

    goal = LeanStatement(name="target", statement="True")
    rendered = LEAPHarness._render_theorem(goal, "by\n  trivial")
    assert "theorem target : True := by" in rendered
    assert rendered.count("theorem") == 1
    forall_rendered = LEAPHarness._render_theorem(
        LeanStatement(name="id", statement="∀ (a : Bool), a = a"),
        "by\n  rfl",
    )
    assert "theorem id (a : Bool) : a = a := by" in forall_rendered

    store = ArtifactStore(_task(tmp_path))
    store.initialize_layout()
    LeanVerifier(store).ensure_project()
    assert (tmp_path / "LeanProject" / "TCSResearch.lean").read_text() == (
        "import TCSResearch.Basic\n"
    )


def test_plan_preserves_distinct_items_using_the_same_subsystem() -> None:
    common = {
        "question_id": "q01",
        "requirement_id": "q01-r01",
        "hypothesis": "The treatment differs from the baseline.",
        "falsification_criterion": "The measured difference is null or opposite.",
        "expected_information_gain": "Either direction narrows the stated requirement.",
        "success_criteria": ["Condition-level measurements are preserved."],
    }
    plan = PlanSubmission(
        objective="Do bounded experiment work.",
        work_items=[
            {
                **common,
                "kind": "experiment",
                "title": "Write and run a benchmark",
                "instruction": "Write and run the complete benchmark in this fresh work item.",
                "strategy": "Use a minimal discriminating fixed-seed pilot.",
            },
            {
                **common,
                "kind": "experiment",
                "title": "Run another benchmark",
                "instruction": "Run a distinct benchmark in a later bounded planning round.",
                "strategy": "Use an independent boundary-condition stress test.",
            },
            {
                **common,
                "kind": "synthesis",
                "title": "Synthesize existing evidence",
                "instruction": "Synthesize only evidence already available in this fresh step.",
                "strategy": "Generate a provenance-linked report from the evidence ledger.",
            },
        ],
    )

    assert [item.kind for item in plan.work_items] == [
        WorkKind.experiment,
        WorkKind.experiment,
        WorkKind.synthesis,
    ]
    assert plan.work_items[0].title == "Write and run a benchmark"


def test_expected_experiment_direction_is_not_a_success_criterion() -> None:
    requirement = EvidenceRequirement(
        requirement_id="q01-r01",
        description="Compare treatment and baseline costs.",
        acceptance_criteria=["Report both measured costs under the registered protocol."],
        acceptable_methods=[WorkKind.experiment],
    )
    question = ResearchQuestion(
        question_id="q01",
        question="Does treatment reduce cost relative to baseline?",
        hypotheses=["Treatment has lower cost."],
        preferred_methods=[WorkKind.experiment],
        requirements=[requirement],
    )
    draft = WorkItemDraft(
        question_id="q01",
        requirement_id="q01-r01",
        kind=WorkKind.experiment,
        title="Registered cost comparison",
        instruction="Measure both conditions with the same fixed-seed instances.",
        strategy="Use a paired fixed-seed comparison.",
        hypothesis="Treatment has lower cost.",
        falsification_criterion="Treatment is equal to or more costly than baseline.",
        expected_information_gain="Either direction resolves the bounded comparison.",
        success_criteria=[
            "The treatment must outperform the baseline.",
            "Condition-level measurements are preserved.",
        ],
    )

    normalized = _normalize_work_draft(
        draft, question=question, requirement=requirement
    )

    assert all("outperform" not in criterion for criterion in normalized.success_criteria)
    assert any("condition-level" in criterion.lower() for criterion in normalized.success_criteria)


def test_negative_and_null_results_are_first_class_novel_contributions() -> None:
    finding = Finding(
        work_id="work_test",
        question_id="q01",
        requirement_id="q01-r01",
        kind=WorkKind.experiment,
        statement="Across the registered conditions the measured difference was zero.",
        status=FindingStatus.observed,
        polarity=FindingPolarity.null,
        strength=EvidenceStrength.strong,
        source_ids=["experiment_1"],
    )

    first = _new_contributions(
        findings=[finding], existing_fingerprints=set(), result_id="result_1"
    )
    repeated = _new_contributions(
        findings=[finding],
        existing_fingerprints={first[0].fingerprint},
        result_id="result_2",
    )

    assert first[0].kind == "null_result"
    assert repeated == []


def test_experiment_v2_contract_preserves_negative_condition_level_output() -> None:
    output = ExperimentOutput(
        experiment="registered comparison",
        parameters={"seed": 7},
        aggregate_metrics={"difference": -0.25},
        observations=[
            ExperimentObservation(
                condition="treatment", sample_size=20, metrics={"mean": 0.5}
            ),
            ExperimentObservation(
                condition="baseline", sample_size=20, metrics={"mean": 0.75}
            ),
        ],
        checks=[{"name": "round trip", "passed": True, "detail": "all 40 cases"}],
        conclusion=ExperimentConclusion(
            hypothesis="The treatment mean exceeds the baseline mean.",
            outcome="contradicts",
            basis_metrics=["difference"],
            statement="The registered comparison observed a negative difference.",
        ),
        limitations=["Small synthetic sample."],
    )

    assert output.conclusion.outcome == "contradicts"
    assert len(output.observations) == 2


def test_experiment_contract_normalizes_metadata_and_nested_metrics() -> None:
    payload = {
        "schema_version": 2,
        "experiment": "nested aggregate comparison",
        "status": "completed",
        "protocol_sha256": "harmless provenance metadata",
        "parameters": {"grid": {"n": 8}, "seeds": [1, 2]},
        "aggregate_metrics": {
            "treatment": {"mean": 1.5, "outcomes": [2, 3]},
            "baseline": {"mean": 2.0},
        },
        "observations": [
            {
                "condition": "treatment",
                "sample_size": 1,
                "metrics": {"cost": {"nodes": 2}},
            }
        ],
        "checks": [
            {"check_id": "known_cases", "passed": True, "detail": "case one"},
            {"check_id": "known_cases", "passed": False, "detail": "case two"},
        ],
        "conclusion": {
            "hypothesis": "The treatment uses fewer nodes than the baseline.",
            "outcome": "model-specific rejection label",
            "basis_metrics": {"treatment": {"mean": 1.5}, "baseline": {"mean": 2.0}},
            "statement": "The bounded samples had a smaller treatment mean.",
            "hypothesis_supported": False,
        },
        "limitations": ["Small sample."],
    }

    normalized = _normalize_output_payload(payload)
    output = ExperimentOutput.model_validate(normalized)

    assert "protocol_sha256" not in normalized
    assert output.parameters == {"grid.n": 8, "seeds": [1, 2]}
    assert output.aggregate_metrics["treatment.mean"] == 1.5
    assert output.aggregate_metrics["treatment.outcomes.1"] == 3
    assert output.observations[0].metrics == {"cost.nodes": 2}
    assert len(output.checks) == 1
    assert not output.checks[0].passed
    assert output.conclusion.outcome == "inconclusive"
    assert output.conclusion.basis_metrics == ["treatment.mean", "baseline.mean"]


def test_raw_replicate_observations_are_valid_and_review_context_is_bounded() -> None:
    protocol = ExperimentProtocol(
        title="Repeated condition records",
        hypothesis="The treatment has lower measured cost than the baseline.",
        null_outcome="The paired measured costs do not differ.",
        experimental_unit="one generated instance",
        conditions=[
            NamedDescription(id="treatment", description="Treatment algorithm."),
            NamedDescription(id="baseline", description="Baseline algorithm."),
        ],
        baselines=[NamedDescription(id="baseline", description="Baseline algorithm.")],
        metrics=[NamedDescription(id="cost", description="Measured operation count.")],
        correctness_checks=[
            NamedDescription(id="known_cases", description="Known outputs are correct.")
        ],
        sample_sizes=[200],
        seeds=[1, 2],
        analysis_plan="Compare paired operation counts over all preserved instances.",
        decision_rule="Classify from the signed paired mean difference.",
        wall_seconds=30,
        memory_mb=256,
        cpus=1,
        known_limitations=["Synthetic instances."],
    )
    output = ExperimentOutput(
        experiment="raw replicate comparison",
        parameters={"seeds": [1, 2]},
        aggregate_metrics={"mean_difference": -1.0},
        observations=[
            ExperimentObservation(
                condition=condition,
                sample_size=1,
                metrics={"cost": index},
            )
            for index in range(200)
            for condition in ("treatment", "baseline")
        ],
        checks=[{"name": "known_cases", "passed": True, "detail": "computed"}],
        conclusion=ExperimentConclusion(
            hypothesis=protocol.hypothesis,
            outcome="supports",
            basis_metrics=["mean_difference"],
            statement="The bounded paired mean difference was negative.",
        ),
        limitations=["Synthetic instances."],
    )

    assert _protocol_output_errors(protocol, output, smoke=False) == []
    context = _evidence_output_context(output)
    assert context["raw_observation_count"] == 400
    assert len(context["observation_examples"]) == 2
    assert len(json.dumps(context)) < 10_000


def test_experiment_agent_fails_fast_without_configuration(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize_layout()
    with pytest.raises(ExperimenterConfigurationError):
        ExperimentAgent(store, None).status()


def test_docker_diagnostic_preserves_failure_tail() -> None:
    completed = subprocess.CompletedProcess(
        ["docker", "build"],
        1,
        stdout="start\n" + "x" * 9000,
        stderr="logs\n" + "y" * 9000 + "\nFINAL ERROR",
    )
    diagnostic = _diagnostic(completed, limit=1000)
    assert "FINAL ERROR" in diagnostic
    assert "preserving final lines" in diagnostic


def test_doctor_removes_nomenclature_only_when_explicit(tmp_path: Path) -> None:
    (tmp_path / "Nomenclature.yml").write_text("symbols: []\n")
    assert main(["doctor", "--workspace", str(tmp_path)]) == 0
    assert (tmp_path / "Nomenclature.yml").exists()
    assert main(["doctor", "--workspace", str(tmp_path), "--clean-legacy"]) == 0
    assert not (tmp_path / "Nomenclature.yml").exists()
