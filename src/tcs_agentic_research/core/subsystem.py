"""The complete contract a research subsystem must implement."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import ActionOutcome, ActionProposal, ResearchView


@runtime_checkable
class ResearchSubsystem(Protocol):
    name: str
    description: str
    model_call_budget: int

    def propose(self, view: ResearchView) -> ActionProposal | None:
        """Choose this subsystem's next atomic action, or yield without side effects."""

    def execute(
        self, proposal: ActionProposal, view: ResearchView, *, run_dir: str
    ) -> ActionOutcome:
        """Execute one previously persisted proposal and return records plus private state."""
