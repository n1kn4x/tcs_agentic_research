from __future__ import annotations

import shutil

import pytest

from tcs_agentic_research.artifact_store import ArtifactStore
from tcs_agentic_research.core.builtins.proof import ProofSubsystem
from tcs_agentic_research.core.models import (
    ActionProposal,
    ActionRecord,
    ResearchView,
)
from tcs_agentic_research.core.policy import EvidencePolicy
from tcs_agentic_research.llm import LLMRouter
from tcs_agentic_research.schemas import LeanStatement, ModelProfile, RouterSettings


@pytest.mark.skipif(
    shutil.which("lake") is None and shutil.which("lean") is None,
    reason="Lean is not installed",
)
def test_proof_subsystem_alone_produces_kernel_admissible_theorem(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize_layout()
    (tmp_path / ArtifactStore.RESEARCH_TASK).write_text("# Prove Boolean commutativity\n")
    router = LLMRouter(
        RouterSettings(
            default_profile="reasoning",
            repair_profile="reasoning",
            profiles={"reasoning": ModelProfile(model="must-not-be-called")},
        ),
        store=store,
        dry_run=False,
    )
    subsystem = ProofSubsystem(store, router)
    goal = LeanStatement(
        name="bool_and_comm_actor",
        statement="∀ (a b : Bool), (a && b) = (b && a)",
    )
    proposal = ActionProposal(
        subsystem="proof",
        action_type="prove",
        title="Prove Boolean and commutativity",
        payload={"goal": goal.model_dump(mode="json")},
    )
    view = ResearchView(
        task="# Prove Boolean commutativity",
        task_sha256="a" * 64,
        task_revision=1,
        cycle=1,
        subsystem="proof",
    )
    run_dir = "Runs/proof_fixture"
    store.resolve(run_dir).mkdir(parents=True)

    outcome = subsystem.execute(proposal, view, run_dir=run_dir)
    action = ActionRecord(
        cycle=1,
        task_revision=1,
        subsystem="proof",
        proposal=proposal,
    )
    admitted = EvidencePolicy(store).admit(
        action=action,
        drafts=outcome.records,
        existing_records=[],
    )

    assert not outcome.error
    assert admitted.rejected == []
    assert len(admitted.records) == 1
    assert admitted.records[0].status.value == "verified"
    assert admitted.records[0].kind.value == "formal_theorem"
