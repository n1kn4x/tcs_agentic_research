"""Public facade for the subsystem-owned research kernel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .artifact_store import ArtifactStore
from .core.builtins import (
    ExperimentSubsystem,
    LiteratureSubsystem,
    ProofSubsystem,
    TheorySubsystem,
)
from .core.kernel import ResearchKernel
from .core.subsystem import ResearchSubsystem
from .llm import LLMRouter


LEGACY_CORE_FILES = (
    "State.json",
    "Agenda.json",
    "Queue.json",
    "Findings.jsonl",
    "Contributions.jsonl",
)


class ResearchEngine:
    """Compatibility name for the v0.4 kernel facade.

    Unlike earlier versions, this class does not plan, decompose, refill a queue, construct
    deliverables, or infer completion.
    """

    def __init__(
        self,
        *,
        workspace: str | Path,
        config_path: str | Path | None = None,
        dry_run: bool = False,
        prompt_dir: str | None = None,
        subsystem_names: Sequence[str] | None = None,
        subsystems: Sequence[ResearchSubsystem] | None = None,
    ):
        self.store = ArtifactStore(workspace)
        self.store.initialize_layout()
        self.router = LLMRouter.from_config_file(
            config_path, store=self.store, dry_run=dry_run
        )
        self.prompt_dir = prompt_dir
        if subsystems is None:
            available: dict[str, ResearchSubsystem] = {
                "literature": LiteratureSubsystem(
                    self.store,
                    self.router,
                    prompt_dir=prompt_dir,
                    max_imports_per_action=self.router.core.literature_max_imports,
                ),
                "theory": TheorySubsystem(self.router),
                "proof": ProofSubsystem(self.store, self.router, prompt_dir=prompt_dir),
            }
            if self.router.experimenter is not None and self.router.experimenter.enabled:
                available["experiment"] = ExperimentSubsystem(self.store, self.router)
            selected = list(subsystem_names or available)
            unknown = set(selected) - set(available)
            if unknown:
                raise ValueError(
                    f"unknown or unavailable subsystem(s): {sorted(unknown)}; "
                    f"available: {sorted(available)}"
                )
            subsystems = [available[name] for name in selected]
        self.kernel = ResearchKernel(
            store=self.store,
            router=self.router,
            subsystems=subsystems,
            record_context_limit=self.router.core.record_context_limit,
        )

    def initialize(self):
        self._reject_legacy_workspace()
        return self.kernel.initialize()

    def run(self, *, max_steps: int = 1) -> dict[str, Any]:
        self._reject_legacy_workspace()
        return self.kernel.run(max_steps=max_steps)

    def status(self) -> dict[str, Any]:
        self._reject_legacy_workspace()
        if self.store.load_kernel_state() is None:
            self.kernel.initialize()
        return self.kernel.status()

    def records(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        records = self.store.read_records()
        if limit is not None:
            records = records[-limit:]
        return [record.model_dump(mode="json") for record in records]

    def _reject_legacy_workspace(self) -> None:
        if self.store.exists(ArtifactStore.KERNEL_STATE):
            return
        present = [path for path in LEGACY_CORE_FILES if self.store.exists(path)]
        if present:
            raise RuntimeError(
                "This workspace contains the incompatible v0.3 orchestrator state "
                f"{present}. Archive it with `tcs-research doctor --archive-legacy` or start a "
                "fresh workspace. The new kernel never interprets old agendas or queues."
            )
