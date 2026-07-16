from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tcs_agentic_research.agents.experiment import ExperimentAgent
from tcs_agentic_research.agents.literature import LiteratureResearcher
from tcs_agentic_research.artifact_store import ArtifactStore
from tcs_agentic_research.cli import main
from tcs_agentic_research.engine import ResearchEngine
from tcs_agentic_research.experimenter.docker_project import _diagnostic
from tcs_agentic_research.experimenter.errors import ExperimenterConfigurationError
from tcs_agentic_research.llm import InputBudgetExceeded, LLMRouter, ModelBudgetExceeded
from tcs_agentic_research.schemas import (
    AnalysisSubmission,
    ModelProfile,
    PaperMetadata,
    PlanSubmission,
    RouterSettings,
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


def test_initialization_creates_only_new_core_artifacts(tmp_path: Path) -> None:
    engine = ResearchEngine(workspace=_task(tmp_path), dry_run=True)
    state = engine.initialize()

    assert state.task_sha256
    assert (tmp_path / "State.json").exists()
    assert (tmp_path / "Queue.json").exists()
    assert (tmp_path / "Events.jsonl").exists()
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
    assert any(item.kind == WorkKind.analysis for item in queue.items)
    assert all(item.kind != WorkKind.proof for item in queue.items)
    assert status["state"]["cycle"] == 1
    assert list((tmp_path / "Runs").glob("*/input.json"))
    assert list((tmp_path / "Runs").glob("*/result.json"))


def test_full_pipeline_plan_splits_subsystems(tmp_path: Path) -> None:
    task = """# Integration
Use LiteratureDB for context, the Experimenter for a fixed-seed benchmark,
LEAP/Lean for one formal proof, and provide a careful analysis.
"""
    engine = ResearchEngine(workspace=_task(tmp_path, task), dry_run=True)
    engine.run(max_steps=1)
    kinds = {item.kind for item in engine.store.load_queue().items}

    assert kinds == {WorkKind.literature, WorkKind.experiment, WorkKind.proof, WorkKind.analysis}


def test_task_edit_resets_phase_to_planning_without_erasing_history(tmp_path: Path) -> None:
    engine = ResearchEngine(workspace=_task(tmp_path), dry_run=True)
    first = engine.initialize()
    first.phase = "review"  # type: ignore[assignment]
    engine.store.save_state(first)
    (tmp_path / ArtifactStore.RESEARCH_TASK).write_text("# Changed task\nProve a Boolean lemma.\n")

    changed = engine.initialize()

    assert changed.phase == "planning"
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
    assert captured["max_tokens"] == 4096
    assert captured["response_format"]


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
