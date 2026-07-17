from __future__ import annotations

import pytest

from tcs_agentic_research.artifact_store import ArtifactStore
from tcs_agentic_research.leap.graph import GraphInvariantError, ProofGraph
from tcs_agentic_research.leap.models import (
    BlueprintCandidate,
    BlueprintChild,
    DecompositionReview,
)
from tcs_agentic_research.leap.sorry import find_placeholder_lines
from tcs_agentic_research.schemas import LeanStatement


def _blueprint(statement: str) -> BlueprintCandidate:
    return BlueprintCandidate(
        overview="Use one strictly smaller fixture proposition.",
        parent_strategy="Apply the child fixture.",
        children=[
            BlueprintChild(
                label="child",
                statement=statement,
                rationale="Fixture child for graph invariants.",
            )
        ],
    )


def _review() -> DecompositionReview:
    return DecompositionReview(accept=True, score=1.0, reasons=["fixture"])


def test_graph_deduplicates_propositions_and_rejects_cycles_transactionally(
    tmp_path,
) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize_layout()
    graph = ProofGraph(store)
    environment = "e" * 64
    goal_a = LeanStatement(name="a", statement="True")
    same_a = LeanStatement(name="another_hint", statement="True")
    goal_b = LeanStatement(name="b", statement="False")

    node_a = graph.register_goal(goal_a, environment_fingerprint=environment)
    reused_a = graph.register_goal(same_a, environment_fingerprint=environment)
    assert reused_a.node_id == node_a.node_id
    assert reused_a.goal.name == node_a.goal.name

    branch = graph.commit_decomposition(
        node_a.node_id,
        blueprint=_blueprint("False"),
        children=[(goal_b, True, "leap_goal_false")],
        parent_proof="by exact False.elim leap_goal_false",
        sketch_artifact_path="fixture.lean",
        review=_review(),
        environment_fingerprint=environment,
    )
    node_b = graph.get_or(branch.children[0].child_or_id)
    before_nodes = graph.node_count()
    before_branches = len(graph.decompositions(node_b.node_id))

    with pytest.raises(GraphInvariantError, match="cycle"):
        graph.commit_decomposition(
            node_b.node_id,
            blueprint=_blueprint("True"),
            children=[(goal_a, True, "leap_goal_true")],
            parent_proof="by exact leap_goal_true",
            sketch_artifact_path="cycle.lean",
            review=_review(),
            environment_fingerprint=environment,
        )

    assert graph.node_count() == before_nodes
    assert len(graph.decompositions(node_b.node_id)) == before_branches


def test_graph_propagates_verified_child_success(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize_layout()
    graph = ProofGraph(store)
    environment = "e" * 64
    parent = graph.register_goal(
        LeanStatement(name="parent", statement="True"),
        environment_fingerprint=environment,
    )
    branch = graph.commit_decomposition(
        parent.node_id,
        blueprint=_blueprint("False"),
        children=[
            (
                LeanStatement(name="child", statement="False"),
                True,
                "leap_goal_child",
            )
        ],
        parent_proof="by exact False.elim leap_goal_child",
        sketch_artifact_path="fixture.lean",
        review=_review(),
        environment_fingerprint=environment,
    )
    graph.commit_direct_proof(
        branch.children[0].child_or_id,
        proof="by contradiction",
        artifact_path="verified_fixture.lean",
    )

    assert graph.get_and(branch.node_id).status.value == "proved"
    assert graph.get_or(parent.node_id).status.value == "proved"


def test_placeholder_scan_ignores_comments_and_strings() -> None:
    code = '''
-- sorry in documentation
#check "admit in a string"
theorem ok : True := by
  /- nested /- sorry -/ comment -/
  trivial
'''
    assert find_placeholder_lines(code) == []
    assert find_placeholder_lines("theorem bad : True := by\n  sorry\n") == [2]
