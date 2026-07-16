"""Reproducible Dockerized experiment harness."""

from __future__ import annotations

from ..artifact_store import ArtifactStore
from ..experimenter.runner import PiExperimentRunner
from ..schemas import ExperimentResult, ExperimenterSettings


class ExperimentAgent:
    """Thin adapter from research-agent tools to the experimenter subsystem.

    Experiments are executed by pi inside a project-level Docker container. Missing
    Docker/configuration raises an experimenter error when an experiment or container lifecycle
    command is requested.
    """

    def __init__(self, store: ArtifactStore, settings: ExperimenterSettings | None):
        self.store = store
        self.settings = settings

    @property
    def runner(self) -> PiExperimentRunner:
        return PiExperimentRunner(self.store, self.settings)

    def ensure_container(self) -> dict[str, object]:
        return self.runner.ensure_container()

    def status(self) -> dict[str, object]:
        return self.runner.status()

    def stop_container(self, *, remove: bool = False) -> None:
        self.runner.stop_container(remove=remove)

    def reset_container(self) -> None:
        self.runner.reset_container()

    def run_experiment(
        self,
        *,
        description: str,
        name: str = "experiment",
        supports_claim_ids: list[str] | None = None,
        timeout_seconds: int | None = None,
    ) -> ExperimentResult:
        return self.runner.run(
            description=description,
            name=name,
            supports_claim_ids=supports_claim_ids or [],
            timeout_seconds=timeout_seconds,
        )
