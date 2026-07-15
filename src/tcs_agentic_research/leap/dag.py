"""AND-OR proof DAG data structures for LEAP."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..schemas import ArtifactRef, LeanStatement, ProofDAGSummary, ProofGoal, StrictModel, new_id


class ProofDAGNode(StrictModel):
    node_id: str = Field(default_factory=lambda: new_id("dag_node"))
    node_type: Literal["or", "and"]
    label: str
    goal: ProofGoal | None = None
    status: Literal["open", "proved", "blocked", "rejected"] = "open"
    parent_ids: list[str] = Field(default_factory=list)
    child_ids: list[str] = Field(default_factory=list)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ProofDAG(StrictModel):
    dag_id: str = Field(default_factory=lambda: new_id("proof_dag"))
    root_node_id: str
    nodes: dict[str, ProofDAGNode]
    accepted_decomposition_ids: list[str] = Field(default_factory=list)
    rejected_decomposition_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_root(cls, root: LeanStatement) -> "ProofDAG":
        goal = ProofGoal(lean_statement=root)
        node = ProofDAGNode(node_type="or", label=root.name, goal=goal)
        return cls(root_node_id=node.node_id, nodes={node.node_id: node})

    def has_open_goals(self) -> bool:
        return any(node.node_type == "or" and node.status == "open" for node in self.nodes.values())

    def select_open_goal(self) -> tuple[str, ProofGoal] | None:
        for node_id, node in self.nodes.items():
            if node.node_type == "or" and node.status == "open" and node.goal is not None:
                return node_id, node.goal
        return None

    def mark_proved(self, node_id: str, artifact: ArtifactRef | None = None, note: str = "") -> None:
        node = self.nodes[node_id]
        node.status = "proved"
        if node.goal is not None:
            node.goal.status = "proved"
        if artifact is not None:
            node.artifact_refs.append(artifact)
        if note:
            node.notes.append(note)
        self._propagate_proved()

    def mark_blocked(self, node_id: str, note: str = "") -> None:
        node = self.nodes[node_id]
        node.status = "blocked"
        if node.goal is not None:
            node.goal.status = "blocked"
        if note:
            node.notes.append(note)

    def root_proved(self) -> bool:
        return self.nodes[self.root_node_id].status == "proved"

    def add_decomposition(
        self,
        *,
        parent_node_id: str,
        sketch_ref: ArtifactRef,
        subgoals: list[LeanStatement],
        reviewer_note: str,
    ) -> str:
        if self._would_be_circular(parent_node_id, subgoals):
            decomp_id = new_id("rejected_decomp")
            self.rejected_decomposition_ids.append(decomp_id)
            self.nodes[parent_node_id].notes.append("Rejected circular decomposition: " + reviewer_note)
            return decomp_id
        and_node = ProofDAGNode(
            node_type="and",
            label=f"decomposition_for_{self.nodes[parent_node_id].label}",
            parent_ids=[parent_node_id],
            artifact_refs=[sketch_ref],
            notes=[reviewer_note],
        )
        self.nodes[and_node.node_id] = and_node
        self.nodes[parent_node_id].child_ids.append(and_node.node_id)
        parent_goal = self.nodes[parent_node_id].goal
        parent_goal_ids = [parent_goal.goal_id] if parent_goal is not None else []
        for statement in subgoals:
            child_goal = ProofGoal(lean_statement=statement, parent_goal_ids=parent_goal_ids)
            child = ProofDAGNode(
                node_type="or",
                label=statement.name,
                goal=child_goal,
                parent_ids=[and_node.node_id],
            )
            self.nodes[child.node_id] = child
            and_node.child_ids.append(child.node_id)
        self.accepted_decomposition_ids.append(and_node.node_id)
        return and_node.node_id

    def summary(self) -> ProofDAGSummary:
        open_ids = []
        proved_ids = []
        blocked_ids = []
        for node_id, node in self.nodes.items():
            if node.node_type != "or":
                continue
            if node.status == "open":
                open_ids.append(node_id)
            elif node.status == "proved":
                proved_ids.append(node_id)
            elif node.status == "blocked":
                blocked_ids.append(node_id)
        return ProofDAGSummary(
            dag_id=self.dag_id,
            root_goal_id=self.root_node_id,
            open_goal_ids=open_ids,
            proved_goal_ids=proved_ids,
            blocked_goal_ids=blocked_ids,
            accepted_decomposition_ids=self.accepted_decomposition_ids,
            rejected_decomposition_ids=self.rejected_decomposition_ids,
        )

    def _propagate_proved(self) -> None:
        changed = True
        while changed:
            changed = False
            for node_id, node in list(self.nodes.items()):
                if node.node_type == "and" and node.status == "open":
                    if node.child_ids and all(self.nodes[c].status == "proved" for c in node.child_ids):
                        node.status = "proved"
                        changed = True
                if node.node_type == "or" and node.status == "open":
                    if any(self.nodes[c].status == "proved" for c in node.child_ids):
                        node.status = "proved"
                        if node.goal:
                            node.goal.status = "proved"
                        changed = True

    def _would_be_circular(self, parent_node_id: str, subgoals: list[LeanStatement]) -> bool:
        ancestor_statements = set()
        stack = [parent_node_id]
        while stack:
            node_id = stack.pop()
            node = self.nodes[node_id]
            if node.goal:
                ancestor_statements.add(node.goal.lean_statement.statement.strip())
                ancestor_statements.add(node.goal.lean_statement.name.strip())
            stack.extend(node.parent_ids)
        for subgoal in subgoals:
            if subgoal.statement.strip() in ancestor_statements or subgoal.name.strip() in ancestor_statements:
                return True
        return False
