"""LEAP theorem-proving harness.

The harness follows the LEAP discipline: direct formalization first, then blueprint
decomposition whose parent proof is placeholder-free and whose placeholders occur only in
explicit child lemmas.
"""

from __future__ import annotations

from pydantic import Field

from ..artifact_store import ArtifactStore
from ..llm import LLMRouter
from ..prompt_loader import render_prompt
from ..schemas import ArtifactRef, LeanStatement, ProofGoal, StrictModel, TheoremProverResult
from .dag import ProofDAG
from .lean import LeanVerifier
from .search import LeanSearchIndex
from .sorry import check_decomposition_placeholders, find_placeholder_lines


class FormalProofCandidate(StrictModel):
    informal_proof: str
    lean_code: str
    notes: list[str] = Field(default_factory=list)


class BlueprintCandidate(StrictModel):
    informal_blueprint: str
    formal_sketch_code: str
    proposed_lemmas: list[LeanStatement] = Field(default_factory=list)
    simplification_rationale: str = ""


class DecompositionReview(StrictModel):
    accepted: bool
    summary: str
    non_circular: bool
    parent_simplified: bool
    child_lemmas_plausible: bool
    required_changes: list[str] = Field(default_factory=list)


class LEAPHarness:
    def __init__(self, store: ArtifactStore, router: LLMRouter, *, prompt_dir: str | None = None):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir
        self.verifier = LeanVerifier(store)
        self.search = LeanSearchIndex(store)

    def prove(
        self,
        goal: LeanStatement,
        *,
        context: str = "",
        max_iterations: int = 8,
        max_revisions: int = 2,
    ) -> TheoremProverResult:
        self.verifier.ensure_project()
        retrieval_context = self.search.render_hits(goal.name + " " + goal.statement)
        context = context + "\n\nRelevant local Lean declarations:\n" + retrieval_context
        dag = ProofDAG.from_root(goal)
        compiler_logs = []
        proved_artifacts: list[ArtifactRef] = []
        root_direct_artifact: ArtifactRef | None = None
        iterations = 0

        while dag.has_open_goals() and iterations < max_iterations:
            iterations += 1
            selected = dag.select_open_goal()
            if selected is None:
                break
            node_id, proof_goal = selected

            direct_success = False
            candidate = self._direct_candidate(proof_goal.lean_statement, context)
            for revision in range(max_revisions + 1):
                rel_file = f"TCSResearch/Generated/{_safe_file_stem(proof_goal.lean_statement.name)}_direct_{revision}.lean"
                log = self.verifier.verify_code(candidate.lean_code, rel_file=rel_file)
                compiler_logs.append(log)
                code_ref = self.store.artifact_ref(f"LeanProject/{rel_file}")
                if log.success and not find_placeholder_lines(candidate.lean_code):
                    dag.mark_proved(node_id, code_ref, note="Direct formalization verified by Lean.")
                    proved_artifacts.append(code_ref)
                    if node_id == dag.root_node_id:
                        root_direct_artifact = code_ref
                    direct_success = True
                    break
                if revision < max_revisions:
                    candidate = self._revise_candidate(
                        proof_goal.lean_statement, candidate, log.stderr or log.stdout, context
                    )
            if direct_success:
                self._save_dag(dag)
                continue

            blueprint = self._blueprint_candidate(proof_goal.lean_statement, context)
            if not blueprint.proposed_lemmas:
                dag.mark_blocked(node_id, "No accepted direct proof or useful decomposition generated.")
                self._save_dag(dag)
                continue
            sketch_rel = f"TCSResearch/Generated/{_safe_file_stem(proof_goal.lean_statement.name)}_sketch.lean"
            sketch_log = self.verifier.verify_code(blueprint.formal_sketch_code, rel_file=sketch_rel)
            compiler_logs.append(sketch_log)
            child_names = [lemma.name for lemma in blueprint.proposed_lemmas]
            placeholder_check = check_decomposition_placeholders(
                blueprint.formal_sketch_code,
                parent_name=proof_goal.lean_statement.name,
                child_names=child_names,
            )
            review = self._review_decomposition(proof_goal.lean_statement, blueprint, context)
            if sketch_log.success and placeholder_check.ok and review.accepted:
                sketch_ref = self.store.artifact_ref(f"LeanProject/{sketch_rel}")
                dag.add_decomposition(
                    parent_node_id=node_id,
                    sketch_ref=sketch_ref,
                    subgoals=blueprint.proposed_lemmas,
                    reviewer_note=review.summary,
                )
            else:
                reason = "; ".join(placeholder_check.errors + review.required_changes)
                if not sketch_log.success:
                    reason = "Lean rejected sketch. " + reason
                dag.mark_blocked(node_id, reason or "Decomposition rejected.")
            self._save_dag(dag)

        self._save_dag(dag)
        summary = dag.summary()
        open_goals = [node.goal for node in dag.nodes.values() if node.goal and node.status == "open"]
        if root_direct_artifact is not None:
            status = "proved"
        elif any(log.exit_code == 127 for log in compiler_logs):
            status = "needs_human_formalization"
        elif summary.proved_goal_ids or summary.accepted_decomposition_ids or dag.root_proved():
            # Accepted decompositions remain partial until a single combined sorry-free Lean
            # artifact is extracted and verified.
            status = "partially_proved"
        else:
            status = "failed"
        return TheoremProverResult(
            status=status,
            root_goal=goal,
            proved_artifacts=proved_artifacts,
            open_goals=open_goals,
            proof_dag_summary=summary.model_dump_json(indent=2),
            compiler_logs=compiler_logs,
            recommended_next_steps=self._recommendations(status, open_goals),
        )

    def _direct_candidate(self, goal: LeanStatement, context: str) -> FormalProofCandidate:
        mock_output = FormalProofCandidate(
            informal_proof="No LLM proof was available; generated a placeholder theorem for human formalization.",
            lean_code=self._placeholder_theorem(goal),
            notes=["Dry-run mock output contains sorry and will not be accepted as proved."],
        )
        messages = [
            {"role": "system", "content": render_prompt("leap_direct_prover", override_dir=self.prompt_dir)},
            {"role": "user", "content": f"Context:\n{context}\n\nGoal:\n{goal.model_dump_json(indent=2)}"},
        ]
        return self.router.complete_structured(
            task_type="theorem_proving",
            messages=messages,
            schema=FormalProofCandidate,
            mock_output=mock_output if self.router.dry_run else None,
        )

    def _revise_candidate(
        self,
        goal: LeanStatement,
        candidate: FormalProofCandidate,
        compiler_error: str,
        context: str,
    ) -> FormalProofCandidate:
        mock_output = candidate
        messages = [
            {"role": "system", "content": render_prompt("leap_reviser", override_dir=self.prompt_dir)},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\nGoal:\n{goal.model_dump_json(indent=2)}\n\n"
                    f"Compiler error:\n{compiler_error}\n\nCurrent candidate:\n{candidate.model_dump_json(indent=2)}"
                ),
            },
        ]
        return self.router.complete_structured(
            task_type="theorem_proving",
            messages=messages,
            schema=FormalProofCandidate,
            mock_output=mock_output if self.router.dry_run else None,
        )

    def _blueprint_candidate(self, goal: LeanStatement, context: str) -> BlueprintCandidate:
        mock_output = BlueprintCandidate(
            informal_blueprint="No decomposition was generated in dry-run mock mode.",
            formal_sketch_code=self._placeholder_theorem(goal),
            proposed_lemmas=[],
            simplification_rationale="Dry-run mock output cannot introduce useful subgoals.",
        )
        messages = [
            {"role": "system", "content": render_prompt("leap_blueprint", override_dir=self.prompt_dir)},
            {"role": "user", "content": f"Context:\n{context}\n\nGoal:\n{goal.model_dump_json(indent=2)}"},
        ]
        return self.router.complete_structured(
            task_type="theorem_proving",
            messages=messages,
            schema=BlueprintCandidate,
            mock_output=mock_output if self.router.dry_run else None,
        )

    def _review_decomposition(
        self,
        goal: LeanStatement,
        blueprint: BlueprintCandidate,
        context: str,
    ) -> DecompositionReview:
        mock_output = DecompositionReview(
            accepted=False,
            summary="Dry-run mock reviewer rejects decomposition unless an LLM or human approves it.",
            non_circular=False,
            parent_simplified=False,
            child_lemmas_plausible=False,
            required_changes=["Obtain reviewer approval for proposed child lemmas."],
        )
        messages = [
            {"role": "system", "content": render_prompt("leap_decomposition_reviewer", override_dir=self.prompt_dir)},
            {
                "role": "user",
                "content": f"Context:\n{context}\nGoal:\n{goal.model_dump_json(indent=2)}\nBlueprint:\n{blueprint.model_dump_json(indent=2)}",
            },
        ]
        return self.router.complete_structured(
            task_type="theorem_proving",
            messages=messages,
            schema=DecompositionReview,
            mock_output=mock_output if self.router.dry_run else None,
        )

    def _placeholder_theorem(self, goal: LeanStatement) -> str:
        imports = "\n".join(f"import {imp}" for imp in goal.imports)
        namespace_open = f"namespace {goal.namespace}\n\n" if goal.namespace else ""
        namespace_close = f"\nend {goal.namespace}\n" if goal.namespace else ""
        return f"{imports}\n\n{namespace_open}theorem {goal.name} : {goal.statement} := by\n  sorry\n{namespace_close}"

    def _save_dag(self, dag: ProofDAG) -> ArtifactRef:
        return self.store.write_json(f"LeanProject/ProofDAGs/{dag.dag_id}.json", dag)

    def _recommendations(self, status: str, open_goals: list[ProofGoal]) -> list[str]:
        if status == "proved":
            return ["Reference the verified Lean artifact in ClaimLedger evidence."]
        if status == "needs_human_formalization":
            return ["Install Lean/Lake via elan and rerun LEAP."]
        if open_goals:
            return [f"Prove or decompose open goal `{goal.lean_statement.name}`." for goal in open_goals]
        if status == "partially_proved":
            return ["Extract a combined sorry-free Lean proof from accepted decompositions and verified child lemmas, then rerun Lean."]
        return ["Inspect compiler logs and consider decomposing the theorem into smaller lemmas."]


def _safe_file_stem(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)[:80] or "goal"
