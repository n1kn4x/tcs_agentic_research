"""Application adapter for the persistent LEAP subsystem."""

from __future__ import annotations

from ..artifact_store import ArtifactStore
from ..leap.harness import LEAPHarness
from ..llm import LLMRouter
from ..schemas import LeanStatement, LeapSettings, TheoremProverResult


class TheoremProverAgent:
    def __init__(
        self,
        store: ArtifactStore,
        router: LLMRouter,
        *,
        prompt_dir: str | None = None,
        settings: LeapSettings | None = None,
    ):
        self.harness = LEAPHarness(
            store, router, prompt_dir=prompt_dir, settings=settings
        )

    def prove(self, goal: LeanStatement, *, context: str = "") -> TheoremProverResult:
        return self.harness.prove(goal, context=context)
