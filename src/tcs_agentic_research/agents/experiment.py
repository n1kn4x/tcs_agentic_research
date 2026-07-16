"""Thin adapter for the bounded Docker experiment runner."""

from __future__ import annotations

from ..artifact_store import ArtifactStore
from ..experimenter.runner import BoundedExperimentRunner
from ..schemas import ExperimentProgram, ExperimentResult, ExperimenterSettings


class ExperimentAgent:
    def __init__(self, store: ArtifactStore, settings: ExperimenterSettings | None):
        self.store = store
        self.settings = settings

    @property
    def runner(self) -> BoundedExperimentRunner:
        return BoundedExperimentRunner(self.store, self.settings)

    def ensure_container(self) -> dict[str, object]:
        return self.runner.ensure_container()

    def status(self) -> dict[str, object]:
        return self.runner.status()

    def stop_container(self, *, remove: bool = False) -> None:
        self.runner.stop_container(remove=remove)

    def reset_container(self) -> None:
        self.runner.reset_container()

    def run_program(self, *, program: ExperimentProgram, name: str = "experiment") -> ExperimentResult:
        return self.runner.run(program=program, name=name)
