"""Theorem prover agent wrapping the LEAP harness."""

from __future__ import annotations

from ..artifact_store import ArtifactStore
from ..llm import LLMRouter
from ..leap.harness import LEAPHarness
from ..schemas import LeanStatement, TheoremProverResult


class TheoremProverAgent:
    def __init__(self, store: ArtifactStore, router: LLMRouter, *, prompt_dir: str | None = None):
        self.harness = LEAPHarness(store, router, prompt_dir=prompt_dir)
        self.store = store

    def prove(self, goal: LeanStatement, *, context: str = "") -> TheoremProverResult:
        result = self.harness.prove(goal, context=context)
        ref = self.store.write_json(f"Reports/critic_summaries/{result.result_id}.json", result)
        self_ref = ref.model_copy(
            update={
                "sha256": None,
                "summary": "Theorem prover result JSON; self-reference hash omitted.",
            }
        )
        if self_ref.path not in {existing.path for existing in result.artifact_refs}:
            result.artifact_refs.append(self_ref)
        self.store.write_json(f"Reports/critic_summaries/{result.result_id}.json", result)
        return result
