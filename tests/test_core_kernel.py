from __future__ import annotations

from dataclasses import dataclass, field

from tcs_agentic_research.artifact_store import ArtifactStore
from tcs_agentic_research.core.kernel import ResearchKernel
from tcs_agentic_research.core.models import (
    ActionOutcome,
    ActionProposal,
    ActionRecord,
    ActionStatus,
    EvidenceReceipt,
    EvidenceType,
    RecordDraft,
    RecordKind,
)
from tcs_agentic_research.llm import LLMRouter
from tcs_agentic_research.schemas import ModelProfile, RouterSettings


def _router(store: ArtifactStore) -> LLMRouter:
    return LLMRouter(
        RouterSettings(
            default_profile="reasoning",
            repair_profile="reasoning",
            profiles={"reasoning": ModelProfile(model="not-called")},
        ),
        store=store,
        dry_run=True,
    )


def _store(tmp_path) -> ArtifactStore:
    (tmp_path / ArtifactStore.RESEARCH_TASK).write_text("# Task\nInvestigate a durable question.\n")
    store = ArtifactStore(tmp_path)
    store.initialize_layout()
    return store


@dataclass
class ScriptedSubsystem:
    name: str
    title: str
    model_call_budget: int = 1
    description: str = "fixture"
    execute_count: int = 0
    seen_record_counts: list[int] = field(default_factory=list)
    always_yield: bool = False

    def propose(self, view):
        self.seen_record_counts.append(len(view.records))
        if self.always_yield:
            return None
        return ActionProposal(
            subsystem=self.name,
            action_type="write",
            title=self.title,
            payload={"title": self.title},
        )

    def execute(self, proposal, view, *, run_dir):
        self.execute_count += 1
        return ActionOutcome(
            summary=f"wrote {self.title}",
            records=[
                RecordDraft(
                    kind=RecordKind.analysis,
                    title=self.title,
                    summary=f"Tentative note from {self.name}.",
                    evidence=EvidenceReceipt(evidence_type=EvidenceType.model),
                )
            ],
            state_patch={"runs": self.execute_count},
        )


def test_kernel_is_fair_and_later_subsystems_build_on_prior_records(tmp_path) -> None:
    store = _store(tmp_path)
    first = ScriptedSubsystem("first", "first note")
    second = ScriptedSubsystem("second", "second note")
    kernel = ResearchKernel(store=store, router=_router(store), subsystems=[first, second])

    status = kernel.run(max_steps=2)

    assert first.execute_count == second.execute_count == 1
    assert first.seen_record_counts == [0]
    assert second.seen_record_counts == [1]
    assert status["record_counts"] == {"tentative/analysis": 2}
    assert status["note"] == "The kernel never infers scientific completion."
    assert not store.exists("Agenda.json")
    assert not store.exists("Queue.json")
    assert not store.exists("Findings.jsonl")


def test_identical_committed_action_is_not_executed_again(tmp_path) -> None:
    store = _store(tmp_path)
    actor = ScriptedSubsystem("writer", "stable note")
    kernel = ResearchKernel(store=store, router=_router(store), subsystems=[actor])

    kernel.run(max_steps=1)
    kernel.run(max_steps=1)

    assert actor.execute_count == 1
    assert len(store.read_records()) == 1
    assert any(
        action.status == ActionStatus.duplicate
        for action in store.latest_actions().values()
    )


def test_all_subsystems_yield_enters_runtime_idle_not_scientific_complete(tmp_path) -> None:
    store = _store(tmp_path)
    actors = [
        ScriptedSubsystem("one", "unused", always_yield=True),
        ScriptedSubsystem("two", "unused", always_yield=True),
    ]
    kernel = ResearchKernel(store=store, router=_router(store), subsystems=actors)

    status = kernel.run(max_steps=20)

    assert status["kernel"]["phase"] == "idle"
    assert status["kernel"]["cycle"] == 2
    assert status["recent_records"] == []


def test_interrupted_action_is_recovered_without_replaying_a_claim(tmp_path) -> None:
    store = _store(tmp_path)
    actor = ScriptedSubsystem("writer", "new note", always_yield=True)
    kernel = ResearchKernel(store=store, router=_router(store), subsystems=[actor])
    state = kernel.initialize()
    proposal = ActionProposal(
        subsystem="writer", action_type="write", title="interrupted", payload={"x": 1}
    )
    action = ActionRecord(
        cycle=1,
        task_revision=1,
        subsystem="writer",
        proposal=proposal,
        status=ActionStatus.running,
    )
    store.append_action(action)
    state.active_action_id = action.action_id
    store.save_kernel_state(state)

    recovered = kernel.initialize()

    assert recovered.active_action_id is None
    assert store.latest_actions()[action.action_id].status == ActionStatus.interrupted
    assert store.read_records() == []


def test_task_revision_preserves_prior_memory(tmp_path) -> None:
    store = _store(tmp_path)
    actor = ScriptedSubsystem("writer", "durable note")
    kernel = ResearchKernel(store=store, router=_router(store), subsystems=[actor])
    kernel.run(max_steps=1)
    (tmp_path / ArtifactStore.RESEARCH_TASK).write_text("# Revised task\nNarrow the same project.\n")

    state = kernel.initialize()

    assert state.task_revision == 2
    assert len(store.read_records()) == 1
    assert len(store.read_jsonl(ArtifactStore.TASK_VERSION_LEDGER)) == 2
