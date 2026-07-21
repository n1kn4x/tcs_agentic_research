"""The checked-in workspace tasks are executable architecture acceptance scenarios."""

from __future__ import annotations

from pathlib import Path

import pytest

from tcs_agentic_research.artifact_store import ArtifactStore
from tcs_agentic_research.engine import ResearchEngine
from tcs_agentic_research.schemas import WorkKind

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("workspace_name", "required_methods"),
    [
        ("test-exp-dpll-sat-benchmark", {WorkKind.experiment}),
        (
            "test-integration-toy-algorithm-full-pipeline",
            {WorkKind.literature, WorkKind.experiment, WorkKind.proof},
        ),
        ("test-leap-boolean-algebra-lemmas", {WorkKind.proof}),
        ("test-lit-orthogonal-vectors-lower-bounds", {WorkKind.literature}),
    ],
)
def test_workspace_task_decomposes_into_requested_subsystems(
    tmp_path: Path,
    workspace_name: str,
    required_methods: set[WorkKind],
) -> None:
    source = ROOT / "workspaces" / workspace_name / ArtifactStore.RESEARCH_TASK
    (tmp_path / ArtifactStore.RESEARCH_TASK).write_text(source.read_text())
    engine = ResearchEngine(workspace=tmp_path, dry_run=True)

    engine.run(max_steps=0)

    agenda = engine.store.load_agenda()
    assert agenda is not None
    methods = {
        method
        for question in agenda.questions
        for requirement in question.requirements
        for method in requirement.acceptable_methods
    }
    assert required_methods <= methods
    if workspace_name in {
        "test-exp-dpll-sat-benchmark",
        "test-leap-boolean-algebra-lemmas",
        "test-lit-orthogonal-vectors-lower-bounds",
    }:
        assert methods == required_methods
    if workspace_name == "test-exp-dpll-sat-benchmark":
        assert len(agenda.questions) == 1
        assert sum(len(question.requirements) for question in agenda.questions) == 1
        assert any("highly optimized SAT solvers" in value for value in agenda.constraints)


def test_user_research_objective_is_not_replaced_by_planner_meta_objective(
    tmp_path: Path,
) -> None:
    task = """# Study
## Research Objective
Determine whether a multiplicative dictionary beats literal coding on structured integers.
## Method
Run a fixed-seed experiment.
"""
    (tmp_path / ArtifactStore.RESEARCH_TASK).write_text(task)
    engine = ResearchEngine(workspace=tmp_path, dry_run=True)

    engine.run(max_steps=0)

    agenda = engine.store.load_agenda()
    assert agenda is not None
    assert agenda.objective.startswith("Determine whether a multiplicative dictionary")
    assert "Decompose" not in agenda.objective
