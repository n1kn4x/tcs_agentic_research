"""A bounded Lean proof attempt: generate, compile, and optionally repair once."""

from __future__ import annotations

import re

from pydantic import Field, field_validator

from ..artifact_store import ArtifactStore
from ..llm import LLMRouter
from ..prompt_loader import render_prompt
from ..schemas import ArtifactRef, LeanStatement, StrictModel, TheoremProverResult
from .lean import LeanVerifier
from .sorry import find_placeholder_lines


class FormalProofCandidate(StrictModel):
    informal_proof: str
    proof: str = Field(min_length=2, max_length=8000)
    notes: list[str] = Field(default_factory=list, max_length=6)

    @field_validator("proof", mode="before")
    @classmethod
    def normalize_proof_term(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        text = value.strip()
        fenced = re.fullmatch(r"```(?:lean)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
        proof = fenced.group(1).strip() if fenced else text
        if not proof.startswith("by"):
            raise ValueError("proof must be one Lean proof term beginning with `by`")
        if re.search(r"(?m)^\s*(?:import|namespace|theorem|lemma|axiom|def)\b", proof):
            raise ValueError("proof must not contain Lean commands or declarations")
        return proof


class LEAPHarness:
    """One fresh proof call plus at most ``max_revisions`` compiler-guided repairs.

    Decomposition is intentionally not hidden inside an unbounded search loop.  If this attempt
    fails, the engine records the compiler error and a later plan may create smaller proof work.
    """

    def __init__(self, store: ArtifactStore, router: LLMRouter, *, prompt_dir: str | None = None):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir
        self.verifier = LeanVerifier(store)

    def prove(
        self,
        goal: LeanStatement,
        *,
        context: str = "",
        max_iterations: int = 1,
        max_revisions: int = 1,
    ) -> TheoremProverResult:
        del max_iterations  # retained as a harmless keyword for callers; search is no longer iterative.
        self.verifier.ensure_project()
        candidate = self._direct_candidate(goal, context)
        logs = []
        artifacts: list[ArtifactRef] = []

        for revision in range(max_revisions + 1):
            rel_file = f"TCSResearch/Generated/{_safe_file_stem(goal.name)}_{revision}.lean"
            code = self._render_theorem(goal, candidate.proof)
            log = self.verifier.verify_code(code, rel_file=rel_file)
            logs.append(log)
            code_ref = self.store.artifact_ref(f"LeanProject/{rel_file}")
            artifacts.append(code_ref)
            if log.artifact_ref is not None:
                artifacts.append(log.artifact_ref)
            if log.success and not find_placeholder_lines(code):
                proved_artifacts = [code_ref]
                if log.artifact_ref is not None:
                    proved_artifacts.append(log.artifact_ref)
                return TheoremProverResult(
                    status="proved",
                    root_goal=goal,
                    proved_artifacts=proved_artifacts,
                    artifact_refs=artifacts,
                    compiler_logs=logs,
                    proof_dag_summary=(
                        "The application rendered the exact requested declaration around a bounded "
                        "proof term, and Lean verified it without placeholders."
                    ),
                    recommended_next_steps=["Use the verified Lean file as proof evidence."],
                )
            if log.exit_code == 127:
                return TheoremProverResult(
                    status="needs_human_formalization",
                    root_goal=goal,
                    artifact_refs=artifacts,
                    compiler_logs=logs,
                    proof_dag_summary="Lean is unavailable; candidate was not verified.",
                    recommended_next_steps=["Install Lean/Lake via elan and retry this work item."],
                )
            if revision < max_revisions:
                candidate = self._revise_candidate(
                    goal,
                    candidate,
                    (log.stderr or log.stdout)[-6000:],
                    context,
                )

        return TheoremProverResult(
            status="failed",
            root_goal=goal,
            artifact_refs=artifacts,
            compiler_logs=logs,
            proof_dag_summary="Bounded direct attempt failed after the configured repair budget.",
            recommended_next_steps=[
                "Inspect the final compiler log.",
                "Create a separate work item for a smaller lemma or explicit decomposition.",
            ],
        )

    def _direct_candidate(self, goal: LeanStatement, context: str) -> FormalProofCandidate:
        mock = FormalProofCandidate(
            informal_proof="Dry-run candidate only; no proof is claimed.",
            proof="by\n  sorry",
            notes=["Contains sorry and cannot be accepted."],
        )
        return self.router.complete_structured(
            task_type="theorem_proving",
            messages=[
                {
                    "role": "system",
                    "content": render_prompt("leap_direct_prover", override_dir=self.prompt_dir),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context[-8000:]}\n\nGoal:\n{goal.model_dump_json(indent=2)}",
                },
            ],
            schema=FormalProofCandidate,
            mock_output=mock if self.router.dry_run else None,
        )

    def _revise_candidate(
        self,
        goal: LeanStatement,
        candidate: FormalProofCandidate,
        compiler_error: str,
        context: str,
    ) -> FormalProofCandidate:
        return self.router.complete_structured(
            task_type="proof_revision",
            messages=[
                {
                    "role": "system",
                    "content": render_prompt("leap_reviser", override_dir=self.prompt_dir),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context[-4000:]}\nGoal:\n{goal.model_dump_json(indent=2)}\n\n"
                        f"Compiler error:\n{compiler_error}\n\n"
                        f"Current candidate:\n{candidate.model_dump_json(indent=2)}"
                    ),
                },
            ],
            schema=FormalProofCandidate,
            mock_output=candidate if self.router.dry_run else None,
        )

    @staticmethod
    def _render_theorem(goal: LeanStatement, proof: str) -> str:
        """Bind model output to the exact application-owned declaration."""
        imports = "\n".join(f"import {item}" for item in goal.imports)
        opening = f"namespace {goal.namespace}\n\n" if goal.namespace else ""
        closing = f"\nend {goal.namespace}\n" if goal.namespace else ""
        binders, proposition = _leading_forall_parts(goal.statement)
        binder_text = f" {binders}" if binders else ""
        return (
            f"{imports}\n\n{opening}theorem {goal.name}{binder_text} : {proposition} := "
            f"{proof}\n{closing}"
        )


def _leading_forall_parts(statement: str) -> tuple[str, str]:
    """Move leading explicit binders into the declaration without changing its Lean type."""
    text = statement.strip()
    marker_length = 1 if text.startswith("∀") else 6 if text.startswith("forall ") else 0
    if marker_length == 0:
        return "", text
    depth = 0
    for index, character in enumerate(text[marker_length:], start=marker_length):
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth = max(0, depth - 1)
        elif character == "," and depth == 0:
            binders = text[marker_length:index].strip()
            proposition = text[index + 1 :].strip()
            if binders and proposition:
                if not binders.startswith(("(", "{", "[")):
                    binders = f"({binders})"
                return binders, proposition
            break
    return "", text


def _safe_file_stem(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)[:80] or "goal"
