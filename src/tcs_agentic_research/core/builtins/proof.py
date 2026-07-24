"""Autonomous formal-proof subsystem with a Lean-only trust boundary."""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import Field

from ...agents.theorem_prover import TheoremProverAgent
from ...artifact_store import ArtifactStore
from ...leap.sorry import find_placeholder_lines
from ...llm import LLMRouter
from ...schemas import ArtifactRef, LeanStatement, StrictModel
from ..models import (
    ActionOutcome,
    ActionProposal,
    EvidenceReceipt,
    EvidenceType,
    RecordDraft,
    RecordKind,
    RecordRelation,
    ResearchView,
)


class ProofChoice(StrictModel):
    action: Literal["prove", "idle"]
    name: str = Field(default="research_goal", max_length=100)
    statement: str = Field(default="", max_length=5000)
    rationale: str = Field(default="", max_length=2000)
    parent_ids: list[str] = Field(default_factory=list, max_length=12)


class ProofSubsystem:
    name = "proof"
    description = "Selects concrete Lean propositions and records only kernel-checked theorems."

    def __init__(
        self, store: ArtifactStore, router: LLMRouter, *, prompt_dir: str | None = None
    ):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir
        self.model_call_budget = router.leap.max_model_calls_per_run + 2

    def propose(self, view: ResearchView) -> ActionProposal | None:
        if self.router.dry_run:
            return None
        choice = self.router.complete_structured(
            task_type="proof_formulation",
            schema=ProofChoice,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the autonomous formal-proof subsystem. Select the smallest useful "
                        "Lean 4 proposition suggested by the task or shared research records. Use an "
                        "existing parent record ID when formalizing it. Return only a closed proposition "
                        "type, not a theorem declaration or proof. Reject reflexivity, True, duplicate "
                        "renamings, and facts unrelated to the research objective by choosing idle."
                    ),
                },
                {"role": "user", "content": view.model_dump_json()},
            ],
        )
        if choice.action == "idle" or not choice.statement.strip():
            return None
        goal = LeanStatement(
            name=choice.name,
            statement=choice.statement,
            imports=["TCSResearch.Basic"],
            namespace="TCSResearch",
        )
        goal_hash = hashlib.sha256(goal.statement.encode("utf-8")).hexdigest()
        if _obviously_trivial(goal.statement) or goal_hash in set(
            view.subsystem_state.get("attempted", [])
        ):
            return None
        return ActionProposal(
            subsystem=self.name,
            action_type="prove",
            title=f"Prove {goal.name}: {goal.statement}"[:300],
            rationale=choice.rationale,
            payload={"goal": goal.model_dump(mode="json")},
            parent_ids=[
                record_id
                for record_id in choice.parent_ids
                if record_id in {record.record_id for record in view.records}
            ],
        )

    def execute(
        self, proposal: ActionProposal, view: ResearchView, *, run_dir: str
    ) -> ActionOutcome:
        goal = LeanStatement.model_validate(proposal.payload["goal"])
        self.store.write_json(f"{run_dir}/goal.json", goal)
        result = TheoremProverAgent(
            self.store, self.router, prompt_dir=self.prompt_dir
        ).prove(
            goal,
            context=(
                proposal.rationale
                + "\nParent records: "
                + ", ".join(proposal.parent_ids)
            ),
        )
        self.store.write_json(f"{run_dir}/proof_result.json", result)
        proved_refs = _fresh_refs(self.store, result.proved_artifacts)
        refs = _fresh_refs(self.store, [*proved_refs, *result.artifact_refs])
        placeholder_free = _placeholder_free(self.store, proved_refs)
        attempted = list(view.subsystem_state.get("attempted", []))
        goal_hash = hashlib.sha256(goal.statement.encode("utf-8")).hexdigest()
        attempted.append(goal_hash)
        state_patch = {"attempted": attempted[-100:]}
        if result.status != "proved" or not placeholder_free:
            return ActionOutcome(
                summary=f"Lean proof attempt ended with status {result.status}.",
                records=[
                    RecordDraft(
                        kind=RecordKind.obstruction,
                        title=f"Formalization blocked: {goal.name}",
                        summary=result.proof_dag_summary or f"Lean status: {result.status}",
                        body="\n".join(result.recommended_next_steps),
                        relation=(
                            RecordRelation.challenges
                            if proposal.parent_ids
                            else RecordRelation.none
                        ),
                        parent_ids=proposal.parent_ids,
                        evidence=EvidenceReceipt(
                            evidence_type=EvidenceType.system,
                            details={"goal": goal.statement, "status": result.status},
                            artifact_refs=refs,
                        ),
                    )
                ],
                state_patch=state_patch,
            )
        proved = list(view.subsystem_state.get("proved", []))
        proved.append(goal_hash)
        state_patch["proved"] = proved[-100:]
        return ActionOutcome(
            summary=f"Lean verified `{goal.name}` without placeholders.",
            records=[
                RecordDraft(
                    kind=RecordKind.formal_theorem,
                    title=f"Lean theorem {goal.name}",
                    summary=goal.statement,
                    body=(
                        "Lean accepted the proposition. The parent link expresses intended research "
                        "relevance; only the proposition itself is kernel-verified."
                    ),
                    relation=(
                        RecordRelation.extends
                        if proposal.parent_ids
                        else RecordRelation.none
                    ),
                    parent_ids=proposal.parent_ids,
                    evidence=EvidenceReceipt(
                        evidence_type=EvidenceType.lean,
                        details={
                            "accepted": True,
                            "placeholder_free": True,
                            "statement": goal.statement,
                            "theorem_name": goal.name,
                            "proof_artifact_paths": [ref.path for ref in proved_refs],
                        },
                        artifact_refs=refs,
                    ),
                )
            ],
            state_patch=state_patch,
        )


def _fresh_refs(store: ArtifactStore, refs: list[ArtifactRef]) -> list[ArtifactRef]:
    fresh: list[ArtifactRef] = []
    for ref in refs:
        if ref.path and store.exists(ref.path):
            current = store.artifact_ref(ref.path)
            if current.path not in {item.path for item in fresh}:
                fresh.append(current)
    return fresh


def _placeholder_free(store: ArtifactStore, refs: list[ArtifactRef]) -> bool:
    lean_refs = [ref for ref in refs if ref.path.endswith(".lean")]
    return bool(lean_refs) and all(
        not find_placeholder_lines(store.read_text(ref.path)) for ref in lean_refs
    )


def _obviously_trivial(statement: str) -> bool:
    compact = " ".join(statement.split())
    if compact in {"True", "∀ n : Nat, n = n", "forall n : Nat, n = n"}:
        return True
    match = re.fullmatch(r"∀\s*\((\w+)\s*:\s*[^)]+\),\s*\1\s*=\s*\1", compact)
    return match is not None
