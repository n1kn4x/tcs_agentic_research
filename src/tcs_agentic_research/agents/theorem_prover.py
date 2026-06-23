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
        self.store.write_json(f"Reports/critic_summaries/{result.result_id}.json", result)
        return result
