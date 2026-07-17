"""Public façade for the persistent LEAP subsystem."""

from __future__ import annotations

from ..artifact_store import ArtifactStore
from ..llm import LLMRouter
from ..schemas import LeanStatement, LeapSettings, TheoremProverResult
from .controller import SearchController
from .models import FormalProofCandidate
from .render import leading_forall_parts


class LEAPHarness:
    """Run or resume a persistent AND-OR proof search for one exact Lean statement."""

    def __init__(
        self,
        store: ArtifactStore,
        router: LLMRouter,
        *,
        prompt_dir: str | None = None,
        settings: LeapSettings | None = None,
    ):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir
        self.settings = settings or router.leap

    def prove(self, goal: LeanStatement, *, context: str = "") -> TheoremProverResult:
        controller = SearchController(
            self.store,
            self.router,
            prompt_dir=self.prompt_dir,
            settings=self.settings,
        )
        if self.router.operation_budget_active:
            return controller.prove(goal, context=context)
        with self.router.step_budget(
            f"leap_{goal.name}", max_calls=self.settings.max_model_calls_per_run
        ):
            return controller.prove(goal, context=context)

    @staticmethod
    def _render_theorem(goal: LeanStatement, proof: str) -> str:
        """Render a standalone exact declaration (kept as a useful public test helper)."""
        imports = "\n".join(f"import {item}" for item in goal.imports)
        opening = f"namespace {goal.namespace}\n\n" if goal.namespace else ""
        closing = f"\nend {goal.namespace}\n" if goal.namespace else ""
        binders, proposition = leading_forall_parts(goal.statement)
        binder_text = f" {binders}" if binders else ""
        return (
            f"{imports}\n\n{opening}theorem {goal.name}{binder_text} : {proposition} := "
            f"{proof}\n{closing}"
        )


__all__ = ["FormalProofCandidate", "LEAPHarness"]
