"""A semantic-free runtime for long-running research subsystems."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Sequence

from ..artifact_store import ArtifactStore, merge_state
from ..llm import LLMRouter
from ..schemas import utc_now
from .context import build_view
from .models import (
    ActionRecord,
    ActionStatus,
    KernelPhase,
    KernelState,
)
from .policy import EvidencePolicy
from .report import write_reports
from .subsystem import ResearchSubsystem


class ResearchKernel:
    """Persist, fairly schedule, and commit subsystem-owned atomic actions.

    This class intentionally cannot create research questions, select scientific methods, decide
    that a claim is true, or decide that a project is complete.
    """

    def __init__(
        self,
        *,
        store: ArtifactStore,
        router: LLMRouter,
        subsystems: Sequence[ResearchSubsystem],
        record_context_limit: int = 80,
    ):
        self.store = store
        self.router = router
        self.subsystems = list(subsystems)
        self.record_context_limit = record_context_limit
        self.policy = EvidencePolicy(store)
        names = [subsystem.name for subsystem in self.subsystems]
        if len(names) != len(set(names)):
            raise ValueError("research subsystem names must be unique")
        if not names:
            raise ValueError("at least one research subsystem is required")

    def initialize(self) -> KernelState:
        self.store.initialize_layout()
        if not self.store.exists(ArtifactStore.RESEARCH_TASK):
            raise RuntimeError(f"Missing `{ArtifactStore.RESEARCH_TASK}` in {self.store.root}")
        task = self.store.read_text(ArtifactStore.RESEARCH_TASK)
        if not task.strip():
            raise RuntimeError(f"`{ArtifactStore.RESEARCH_TASK}` is empty")
        digest = hashlib.sha256(task.encode("utf-8")).hexdigest()
        state = self.store.load_kernel_state()
        names = [subsystem.name for subsystem in self.subsystems]
        if state is None:
            state = KernelState(task_sha256=digest, enabled_subsystems=names)
            self.store.save_kernel_state(state)
            self.store.append_jsonl(
                ArtifactStore.TASK_VERSION_LEDGER,
                {"revision": 1, "sha256": digest, "task": task, "created_at": utc_now()},
            )
            self.store.append_event(
                "kernel_initialized", {"task_revision": 1, "subsystems": names}
            )
        else:
            if state.task_sha256 != digest:
                state.task_revision += 1
                state.task_sha256 = digest
                state.phase = KernelPhase.running
                state.consecutive_idle = 0
                self.store.append_jsonl(
                    ArtifactStore.TASK_VERSION_LEDGER,
                    {
                        "revision": state.task_revision,
                        "sha256": digest,
                        "task": task,
                        "created_at": utc_now(),
                    },
                )
                self.store.append_event(
                    "task_revised",
                    {
                        "task_revision": state.task_revision,
                        "note": "Prior records remain visible; a workspace is one continuing project.",
                    },
                )
            if state.enabled_subsystems != names:
                state.enabled_subsystems = names
                state.next_subsystem_index %= len(names)
                self.store.append_event("subsystems_changed", {"subsystems": names})
            self.store.save_kernel_state(state)
        state = self._recover_interrupted(state)
        write_reports(self.store, state)
        return state

    def run(self, *, max_steps: int = 1) -> dict[str, Any]:
        with self.store.exclusive_lock():
            state = self.initialize()
            if state.phase == KernelPhase.idle:
                state.phase = KernelPhase.running
                state.consecutive_idle = 0
                self.store.save_kernel_state(state)
            opportunities = 0
            while opportunities < max(0, max_steps):
                state = self._step(state)
                opportunities += 1
                if state.consecutive_idle >= len(self.subsystems):
                    state.phase = KernelPhase.idle
                    self.store.save_kernel_state(state)
                    self.store.append_event(
                        "all_subsystems_yielded", {"cycle": state.cycle}
                    )
                    break
            write_reports(self.store, state)
            return self.status()

    def _step(self, state: KernelState) -> KernelState:
        subsystem = self.subsystems[state.next_subsystem_index % len(self.subsystems)]
        state.next_subsystem_index = (state.next_subsystem_index + 1) % len(self.subsystems)
        state.cycle += 1
        view = build_view(
            self.store,
            state,
            subsystem=subsystem.name,
            record_limit=self.record_context_limit,
        )
        try:
            with self.router.step_budget(
                f"cycle_{state.cycle}_{subsystem.name}",
                max_calls=max(
                    1,
                    min(
                        subsystem.model_call_budget,
                        self.router.core.max_model_calls_per_action,
                    ),
                ),
            ):
                proposal = subsystem.propose(view)
                if proposal is None:
                    state.consecutive_idle += 1
                    self.store.append_event(
                        "subsystem_yielded",
                        {"cycle": state.cycle, "subsystem": subsystem.name},
                    )
                    self.store.save_kernel_state(state)
                    return state
                if proposal.subsystem != subsystem.name:
                    raise ValueError(
                        f"subsystem {subsystem.name} proposed an action owned by {proposal.subsystem}"
                    )
                action = ActionRecord(
                    cycle=state.cycle,
                    task_revision=state.task_revision,
                    subsystem=subsystem.name,
                    proposal=proposal,
                )
                action.run_dir = self.store.create_action_dir(
                    state.cycle, action.action_id, subsystem.name
                )
                self.store.write_json(f"{action.run_dir}/proposal.json", proposal)
                self.store.append_action(action)
                state.active_action_id = action.action_id
                self.store.save_kernel_state(state)

                if self._already_executed(proposal.fingerprint):
                    action.status = ActionStatus.duplicate
                    action.summary = "An identical subsystem action was already committed."
                    self.store.append_action(action)
                    state.active_action_id = None
                    state.consecutive_idle += 1
                    self.store.save_kernel_state(state)
                    return state

                action.status = ActionStatus.running
                self.store.append_action(action)
                outcome = subsystem.execute(proposal, view, run_dir=action.run_dir)
                outcome_ref = self.store.write_json(f"{action.run_dir}/outcome.json", outcome)
                if outcome_ref.path not in {ref.path for ref in outcome.artifact_refs}:
                    outcome.artifact_refs.append(outcome_ref)
                admission = self.policy.admit(
                    action=action,
                    drafts=outcome.records,
                    existing_records=self.store.read_records(),
                )
                self.store.append_records(admission.records)
                current_actor_state = self.store.load_subsystem_state(subsystem.name)
                self.store.save_subsystem_state(
                    subsystem.name, merge_state(current_actor_state, outcome.state_patch)
                )
                action.admitted_record_ids = [record.record_id for record in admission.records]
                action.rejected_drafts = admission.rejected
                action.summary = outcome.summary
                action.error = outcome.error
                if outcome.error:
                    action.status = ActionStatus.failed
                elif admission.records:
                    action.status = ActionStatus.succeeded
                else:
                    action.status = ActionStatus.no_progress
                self.store.append_action(action)
                state.active_action_id = None
                if admission.records:
                    state.consecutive_idle = 0
                else:
                    state.consecutive_idle += 1
                self.store.save_kernel_state(state)
                self.store.append_event(
                    "action_committed",
                    {
                        "cycle": state.cycle,
                        "action_id": action.action_id,
                        "subsystem": subsystem.name,
                        "status": action.status.value,
                        "record_ids": action.admitted_record_ids,
                    },
                )
                return state
        except Exception as exc:
            action_id = state.active_action_id
            if action_id:
                latest = self.store.latest_actions().get(action_id)
                if latest is not None:
                    latest.status = ActionStatus.failed
                    latest.error = f"{type(exc).__name__}: {exc}"
                    latest.summary = "Subsystem action failed before a research record was committed."
                    self.store.append_action(latest)
            state.active_action_id = None
            state.consecutive_idle += 1
            self.store.save_kernel_state(state)
            self.store.append_event(
                "subsystem_error",
                {
                    "cycle": state.cycle,
                    "subsystem": subsystem.name,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return state

    def _already_executed(self, fingerprint: str) -> bool:
        return any(
            action.proposal.fingerprint == fingerprint
            and action.status in {ActionStatus.succeeded, ActionStatus.no_progress}
            for action in self.store.latest_actions().values()
        )

    def _recover_interrupted(self, state: KernelState) -> KernelState:
        if state.active_action_id is None:
            return state
        latest = self.store.latest_actions().get(state.active_action_id)
        if latest is not None and latest.status in {
            ActionStatus.proposed,
            ActionStatus.running,
        }:
            latest.status = ActionStatus.interrupted
            latest.summary = "The process ended before the action committed an outcome."
            self.store.append_action(latest)
        self.store.append_event(
            "interrupted_action_recovered", {"action_id": state.active_action_id}
        )
        state.active_action_id = None
        state.phase = KernelPhase.running
        self.store.save_kernel_state(state)
        return state

    def status(self) -> dict[str, Any]:
        state = self.store.load_kernel_state()
        if state is None:
            state = self.initialize()
        records = self.store.read_records()
        latest_actions = list(self.store.latest_actions().values())
        return {
            "workspace": str(self.store.root),
            "kernel": state.model_dump(mode="json"),
            "record_counts": {
                f"{status}/{kind}": count
                for (status, kind), count in sorted(
                    Counter((record.status.value, record.kind.value) for record in records).items()
                )
            },
            "action_counts": dict(
                sorted(Counter(action.status.value for action in latest_actions).items())
            ),
            "recent_records": [
                {
                    "record_id": record.record_id,
                    "status": record.status.value,
                    "kind": record.kind.value,
                    "producer": record.producer,
                    "title": record.title,
                    "summary": record.summary,
                }
                for record in records[-10:]
            ],
            "note": "The kernel never infers scientific completion.",
        }
