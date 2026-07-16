"""Thin adapter around the bounded Lean harness."""

from __future__ import annotations

from ..artifact_store import ArtifactStore
from ..llm import LLMRouter
from ..leap.harness import LEAPHarness
from ..schemas import LeanStatement, TheoremProverResult


class TheoremProverAgent:
    def __init__(self, store: ArtifactStore, router: LLMRouter, *, prompt_dir: str | None = None):
        self.harness = LEAPHarness(store, router, prompt_dir=prompt_dir)

    def prove(
        self,
        goal: LeanStatement,
        *,
        context: str = "",
        max_iterations: int = 1,
        max_revisions: int = 1,
    ) -> TheoremProverResult:
        return self.harness.prove(
            goal,
            context=context,
            max_iterations=max_iterations,
            max_revisions=max_revisions,
        )
