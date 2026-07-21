"""Evidence-gap-driven orchestration for long-running, non-stagnating research.

Models propose typed plans, protocols, derivations, and reviews. Python owns scheduling, provenance,
novelty, requirement state, retries, and stopping. A cycle counts as progress only when it adds a
new evidence-backed contribution; execution, token use, and artifact churn never count by themselves.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from .artifact_store import ArtifactStore, sha256_file
from .llm import LLMRouter
from .pipelines import (
    DerivationPipeline,
    ExperimentPipeline,
    LiteraturePipeline,
    ProofPipeline,
)
from .schemas import (
    EvidenceRequirement,
    ExperimentState,
    FindingStatus,
    RequirementStatus,
    ResearchAgenda,
    ResearchAgendaDraft,
    ResearchPhase,
    ResearchQuestion,
    ResearchQuestionDraft,
    WorkItem,
    WorkItemDraft,
    WorkKind,
    WorkQueue,
    WorkResult,
    WorkStatus,
    WorkspaceState,
    utc_now,
)
from .workflow import (
    _default_plan,
    _deterministic_agenda,
    _ensure_requested_methods,
    _new_contributions,
    _next_open,
    _normalize_work_draft,
    _render_progress_report,
    _render_research_report,
    _strategy_fingerprint,
    _task_summary,
    requirement_index,
)


class ResearchEngine:
    def __init__(
        self,
        *,
        workspace: str | Path,
        config_path: str | Path | None = None,
        dry_run: bool = False,
        prompt_dir: str | None = None,
    ):
        self.store = ArtifactStore(workspace)
        self.store.initialize_layout()
        self.router = LLMRouter.from_config_file(config_path, store=self.store, dry_run=dry_run)
        self.prompt_dir = prompt_dir

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> WorkspaceState:
        if not self.store.exists(ArtifactStore.RESEARCH_TASK):
            raise RuntimeError(f"Missing `{ArtifactStore.RESEARCH_TASK}` in {self.store.root}")
        task_path = self.store.resolve(ArtifactStore.RESEARCH_TASK)
        task = self.store.read_text(ArtifactStore.RESEARCH_TASK)
        if not task.strip():
            raise RuntimeError(f"`{ArtifactStore.RESEARCH_TASK}` is empty")
        digest = sha256_file(task_path)
        state = self.store.load_state()
        if state is None:
            state = WorkspaceState(
                task_sha256=digest,
                task_summary=_task_summary(task),
                notes=[
                    "Progress is contribution-based: only novel evidence, including valid negative "
                    "or null results, resets the stagnation counter."
                ],
            )
            self.store.save_state(state)
            self.store.append_event(
                "workspace_initialized", {"task_id": state.task_id, "task_sha256": digest}
            )
            return state
        if state.task_sha256 != digest:
            self._archive_previous_task(state.task_sha256)
            state = WorkspaceState(
                task_sha256=digest,
                task_summary=_task_summary(task),
                notes=["Task changed; prior task ledgers were archived instead of mixed into this agenda."],
            )
            self.store.save_queue(WorkQueue())
            self.store.save_state(state)
            self.store.append_event("task_changed", {"new_sha256": digest})
        return state

    def _archive_previous_task(self, old_digest: str) -> None:
        archive = self.store.resolve(f"Archive/{old_digest[:16]}")
        archive.mkdir(parents=True, exist_ok=True)
        for rel in [
            ArtifactStore.RESEARCH_STATE,
            ArtifactStore.RESEARCH_AGENDA,
            ArtifactStore.WORK_QUEUE,
            ArtifactStore.FINDING_LEDGER,
            ArtifactStore.CONTRIBUTION_LEDGER,
        ]:
            source = self.store.resolve(rel)
            if source.exists():
                shutil.copy2(source, archive / Path(rel).name)
        # Cycle numbers restart for a new task. Move task-local runs/reports so new run paths cannot
        # overwrite old inputs or leak old result summaries into the new planner context.
        for directory in [*ArtifactStore.CORE_DIRECTORIES, "ExperimentStates", "ExperimentRuns"]:
            source_dir = self.store.resolve(directory)
            target_dir = archive / directory
            if source_dir.exists() and any(source_dir.iterdir()):
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.move(str(source_dir), str(target_dir))
            if directory in ArtifactStore.CORE_DIRECTORIES:
                source_dir.mkdir(parents=True, exist_ok=True)
        self.store.resolve(ArtifactStore.RESEARCH_AGENDA).unlink(missing_ok=True)
        self.store.write_text(ArtifactStore.FINDING_LEDGER, "")
        self.store.write_text(ArtifactStore.CONTRIBUTION_LEDGER, "")

    def run(self, *, max_steps: int = 1) -> dict[str, Any]:
        with self.store.exclusive_lock():
            state = self.initialize()
            state = self._recover_interrupted_work(state)
            self._ensure_agenda(state)
            state = self._require_state()
            self._write_reports(state)
            executed = 0
            while executed < max(0, max_steps):
                if state.phase in {
                    ResearchPhase.complete,
                    ResearchPhase.needs_input,
                    ResearchPhase.system_error,
                }:
                    break
                queue = self.store.load_queue()
                self._refill_queue(state, queue)
                state = self._require_state()
                if state.phase in {
                    ResearchPhase.complete,
                    ResearchPhase.needs_input,
                    ResearchPhase.system_error,
                }:
                    break
                queue = self.store.load_queue()
                item = _next_open(queue)
                if item is None:
                    break
                self._execute(state, queue, item)
                state = self._require_state()
                executed += 1
                threshold = self.router.core.max_no_progress_steps
                if state.no_progress_steps and state.no_progress_steps % threshold == 0:
                    self._diversify_after_stagnation(state, item.requirement_id)
                    state = self._require_state()
            return self.status()

    def _recover_interrupted_work(self, state: WorkspaceState) -> WorkspaceState:
        """A killed week-long process must not leave its active strategy permanently invisible."""
        queue = self.store.load_queue()
        interrupted = [item for item in queue.items if item.status == WorkStatus.running]
        if not interrupted and state.active_work_id is None:
            return state
        for item in interrupted:
            item.status = WorkStatus.open
            item.blocked_reason = "Previous process ended before committing a typed result; retrying."
            item.updated_at = utc_now()
        state.active_work_id = None
        if state.phase not in {ResearchPhase.complete, ResearchPhase.needs_input}:
            state.phase = ResearchPhase.working if interrupted else ResearchPhase.planning
        state.notes.append(
            f"Recovered {len(interrupted)} interrupted running work item(s) after process restart."
        )
        self.store.save_queue(queue)
        self.store.save_state(state)
        self.store.append_event(
            "interrupted_work_recovered",
            {"work_ids": [item.work_id for item in interrupted]},
        )
        return state

    def replan(self) -> None:
        with self.store.exclusive_lock():
            state = self.initialize()
            if state.phase == ResearchPhase.complete:
                raise RuntimeError("The agenda is complete; edit the task to start a new agenda.")
            state.phase = ResearchPhase.planning
            state.human_replan_count += 1
            queue = self.store.load_queue()
            for item in queue.items:
                if item.kind != WorkKind.experiment or item.status != WorkStatus.failed:
                    continue
                experiment_path = f"ExperimentStates/{item.work_id}.json"
                if not self.store.exists(experiment_path):
                    continue
                experiment = ExperimentState.model_validate(self.store.read_json(experiment_path))
                if not experiment.engineering_blocked:
                    continue
                experiment.engineering_blocked = False
                experiment.engineering_failures = 0
                experiment.updated_at = utc_now()
                self.store.write_json(experiment_path, experiment)
                item.status = WorkStatus.open
                item.blocked_reason = "Human replan resumed the preserved experiment stage."
                item.updated_at = utc_now()
            self.store.save_queue(queue)
            agenda = self.store.load_agenda()
            if agenda is not None:
                for _, requirement in requirement_index(agenda).values():
                    if requirement.status == RequirementStatus.blocked:
                        requirement.status = (
                            RequirementStatus.in_progress
                            if requirement.finding_ids else RequirementStatus.open
                        )
                        requirement.blocker = ""
                        requirement.updated_at = utc_now()
                self.store.save_agenda(agenda)
            state.notes.append(
                "Human requested replanning; evidence and exhausted strategies remain durable, "
                "and each method receives two additional distinct-strategy slots."
            )
            self.store.save_state(state)
            self.store.append_event(
                "human_requested_replan",
                {"cycle": state.cycle, "replan_count": state.human_replan_count},
            )

    def status(self) -> dict[str, Any]:
        state = self.store.load_state()
        queue = self.store.load_queue()
        agenda = self.store.load_agenda()
        findings = self.store.read_findings()
        contributions = self.store.read_contributions()
        counts = {status.value: 0 for status in WorkStatus}
        for item in queue.items:
            counts[item.status.value] += 1
        requirement_counts = {status.value: 0 for status in RequirementStatus}
        if agenda:
            for _, requirement in requirement_index(agenda).values():
                requirement_counts[requirement.status.value] += 1
        next_item = _next_open(queue)
        experiment_stages: dict[str, int] = {}
        experiment_state_dir = self.store.resolve("ExperimentStates")
        if experiment_state_dir.exists():
            for path in experiment_state_dir.glob("*.json"):
                experiment = ExperimentState.model_validate(self.store.read_json(path))
                experiment_stages[experiment.stage] = experiment_stages.get(experiment.stage, 0) + 1
        return {
            "workspace": str(self.store.root),
            "state": state.model_dump(mode="json") if state else None,
            "work_counts": counts,
            "requirement_counts": requirement_counts,
            "contribution_count": len(contributions),
            "next_open_work": next_item.model_dump(mode="json") if next_item else None,
            "recent_contributions": [item.model_dump(mode="json") for item in contributions[-8:]],
            "finding_counts": {
                status.value: sum(finding.status == status for finding in findings)
                for status in FindingStatus
            },
            "experiment_stages": experiment_stages,
        }

    # ------------------------------------------------------------------
    # Agenda and reports
    # ------------------------------------------------------------------
    def _ensure_agenda(self, state: WorkspaceState) -> ResearchAgenda:
        existing = self.store.load_agenda()
        if existing is not None and existing.task_sha256 == state.task_sha256:
            return existing
        task = self.store.read_text(ArtifactStore.RESEARCH_TASK)
        fallback = _deterministic_agenda(task)
        messages = [
            {
                "role": "system",
                "content": (
                    "Turn the user's research request into atomic scientific evidence requirements. "
                    "The objective field must summarize the USER'S research objective, never this "
                    "decomposition instruction. Cover every explicitly requested theorem, benchmark, "
                    "dataset comparison, and success criterion; use up to twenty-four narrow questions "
                    "when the task is broad. Every evidence need must name one independently auditable "
                    "output and be satisfiable by its methods. Never treat a requested hypothesis as "
                    "true. Choose methods precisely: literature obtains primary-source statements; "
                    "experiment produces measurements; derivation produces explicit mathematical "
                    "arguments or counterexamples; proof means Lean kernel verification and is used only "
                    "when explicitly requested and realistically expressible. Do not assign an experiment "
                    "to prove a universal theorem and do not assign literature to produce new empirical "
                    "measurements. Do not turn implementation files, configs, plots, tables, or reports "
                    "from the same experiment into separate research questions: group them as artifacts "
                    "of one scientific comparison. Deliverables must be auditable, and valid negative or "
                    "null results must be able to satisfy the corresponding evidence need."
                ),
            },
            {"role": "user", "content": _bounded_task_context(task)},
        ]
        agenda_dir = f"Runs/Agenda_{state.task_sha256[:12]}"
        self.store.resolve(agenda_dir).mkdir(parents=True, exist_ok=True)
        self.store.write_json(f"{agenda_dir}/input.json", {"messages": messages})
        error = ""
        try:
            with self.router.step_budget("agenda", max_calls=2):
                draft = self.router.complete_structured(
                    task_type="task_analysis",
                    messages=messages,
                    schema=ResearchAgendaDraft,
                    mock_output=fallback if self.router.dry_run else None,
                )
        except Exception as exc:  # deterministic decomposition is safe control fallback
            error = f"{type(exc).__name__}: {exc}"
            draft = fallback
            self.store.write_text(f"{agenda_dir}/error.log", error + "\n")
        draft = _ensure_requested_methods(draft, task)
        draft = _restrict_single_subsystem_task(draft, task)
        draft = _compact_single_experiment_task(draft, task)
        questions: list[ResearchQuestion] = []
        forced_subsystem = _single_subsystem_kind(task)
        for q_index, question in enumerate(draft.questions, 1):
            question_id = f"q{q_index:02d}"
            methods: list[WorkKind] = [
                method for method in question.preferred_methods if method != WorkKind.synthesis
            ]
            if not methods:
                methods = [WorkKind.derivation]
            requirements = [
                EvidenceRequirement(
                    requirement_id=f"{question_id}-r{r_index:02d}",
                    description=description,
                    acceptance_criteria=[
                        (
                            "Produces valid evidence for the named gap and reports supportive, "
                            f"contradictory, or null outcomes identically: {description}"
                        ),
                        "States the assumptions, scope, and limitations needed to interpret the result.",
                    ],
                    acceptable_methods=(
                        [forced_subsystem]
                        if forced_subsystem is not None
                        else _methods_for_requirement(description, methods)
                    ),
                )
                for r_index, description in enumerate(question.evidence_needed, 1)
            ]
            questions.append(
                ResearchQuestion(
                    question_id=question_id,
                    question=question.question,
                    hypotheses=question.hypotheses,
                    preferred_methods=methods,
                    requirements=requirements,
                )
            )
        agenda = ResearchAgenda(
            task_sha256=state.task_sha256,
            objective=_research_objective(task, fallback=draft.objective),
            constraints=list(dict.fromkeys([*draft.constraints, *fallback.constraints]))[:20],
            deliverables=draft.deliverables,
            questions=questions,
        )
        self.store.write_json(f"{agenda_dir}/draft.json", draft)
        self.store.save_agenda(agenda)
        self.store.append_event(
            "agenda_created",
            {
                "question_count": len(agenda.questions),
                "requirement_count": len(requirement_index(agenda)),
                "fallback_used": bool(error),
                "error": error,
            },
        )
        return agenda

    def _write_reports(self, state: WorkspaceState) -> None:
        agenda = self.store.load_agenda()
        findings = self.store.read_findings()
        contributions = self.store.read_contributions()
        self.store.write_text(
            "Reports/Progress.md",
            _render_progress_report(
                state, agenda, self.store.load_queue(), findings, contributions
            ),
        )
        self.store.write_text(
            "Reports/ResearchReport.md", _render_research_report(agenda, findings)
        )

    # ------------------------------------------------------------------
    # Gap-driven planning
    # ------------------------------------------------------------------
    def _refill_queue(self, state: WorkspaceState, queue: WorkQueue) -> None:
        """Keep a small fair portfolio ready without spending a model call on scheduling."""
        agenda = self.store.load_agenda()
        if agenda is None:
            agenda = self._ensure_agenda(state)
        if self._all_mandatory_satisfied(agenda):
            state.phase = ResearchPhase.complete
            self.store.save_state(state)
            self._write_reports(state)
            return
        self._mark_exhausted_requirements(agenda, queue)
        self.store.save_agenda(agenda)
        open_count = sum(item.status == WorkStatus.open for item in queue.items)
        slots = max(0, self.router.core.max_plan_items - open_count)
        if slots:
            plan = _default_plan(
                agenda=agenda,
                queue=queue,
                max_method_attempts=self._method_attempt_cap(state),
                limit=self.router.core.max_plan_items,
            )
            existing = {item.strategy_fingerprint for item in queue.items}
            new_items: list[WorkItem] = []
            index = requirement_index(agenda)
            for draft in plan.work_items:
                pair = index.get(draft.requirement_id)
                if pair is None:
                    continue
                question, requirement = pair
                normalized = _normalize_work_draft(
                    draft, question=question, requirement=requirement
                )
                fingerprint = _strategy_fingerprint(normalized)
                if fingerprint in existing:
                    continue
                new_items.append(self._work_item_from_draft(normalized))
                existing.add(fingerprint)
                if len(new_items) >= slots:
                    break
            if new_items:
                queue.items.extend(new_items)
                state.plan_round += 1
                state.phase = ResearchPhase.working
                self.store.save_queue(queue)
                self.store.save_state(state)
                self.store.append_event(
                    "portfolio_refilled",
                    {
                        "plan_round": state.plan_round,
                        "new_work_ids": [item.work_id for item in new_items],
                    },
                )
                self._write_reports(state)
                return
        if _next_open(queue) is not None:
            return
        unresolved = [
            requirement
            for _, requirement in requirement_index(agenda).values()
            if requirement.mandatory and requirement.status != RequirementStatus.satisfied
        ]
        state.phase = ResearchPhase.needs_input if unresolved else ResearchPhase.complete
        if unresolved:
            state.notes.append(
                "No runnable strategy remains for mandatory gaps: "
                + ", ".join(requirement.requirement_id for requirement in unresolved)
            )
            self.store.append_event(
                "agenda_exhausted",
                {"requirements": [requirement.requirement_id for requirement in unresolved]},
            )
        self.store.save_state(state)
        self._write_reports(state)

    def _work_item_from_draft(
        self,
        draft: WorkItemDraft,
        *,
        parent: WorkItem | None = None,
        prior_result_ids: list[str] | None = None,
    ) -> WorkItem:
        return WorkItem(
            question_id=draft.question_id,
            requirement_id=draft.requirement_id,
            kind=draft.kind,
            title=draft.title,
            instruction=draft.instruction,
            strategy=draft.strategy,
            hypothesis=draft.hypothesis,
            falsification_criterion=draft.falsification_criterion,
            expected_information_gain=draft.expected_information_gain,
            success_criteria=draft.success_criteria,
            strategy_fingerprint=_strategy_fingerprint(draft),
            parent_work_id=parent.work_id if parent else None,
            revision=(parent.revision + 1) if parent else 0,
            prior_result_ids=prior_result_ids or [],
        )

    # ------------------------------------------------------------------
    # Atomic execution and deterministic progress accounting
    # ------------------------------------------------------------------
    def _execute(self, state: WorkspaceState, queue: WorkQueue, item: WorkItem) -> None:
        item.status = WorkStatus.running
        item.attempts += 1
        item.updated_at = utc_now()
        state.active_work_id = item.work_id
        state.phase = ResearchPhase.working
        cycle = state.cycle + 1
        run_dir = self.store.create_run_dir(cycle, item.work_id)
        input_ref = self.store.write_json(
            f"{run_dir}/input.json",
            {
                "task_sha256": state.task_sha256,
                "work_item": item,
                "revision_context": self._revision_context(item),
            },
        )
        self.store.save_queue(queue)
        self.store.save_state(state)
        self.store.append_event(
            "work_started",
            {
                "cycle": cycle,
                "work_id": item.work_id,
                "requirement_id": item.requirement_id,
                "kind": item.kind.value,
            },
        )
        budget = (
            self.router.leap.max_model_calls_per_run
            if item.kind == WorkKind.proof
            else self.router.core.max_model_calls_per_step
        )
        try:
            with self.router.step_budget(item.work_id, max_calls=budget):
                result = self._dispatch(item, run_dir)
        except Exception as exc:  # durable result is the operation boundary
            result = WorkResult(
                work_id=item.work_id,
                outcome="failed" if item.kind != WorkKind.experiment else "partial",
                failure_class="operational",
                attempt_class="engineering",
                continue_work=False,
                summary=f"Work step failed: {type(exc).__name__}",
                artifact_refs=[input_ref],
                errors=[f"{type(exc).__name__}: {exc}"],
                next_steps=["Repair the named operation and retry without changing the hypothesis."],
            )
        if input_ref.path not in {ref.path for ref in result.artifact_refs}:
            result.artifact_refs.append(input_ref)
        for finding in result.findings:
            finding.work_id = item.work_id
            finding.question_id = item.question_id
            finding.requirement_id = item.requirement_id

        existing = {contribution.fingerprint for contribution in self.store.read_contributions()}
        contributions = _new_contributions(
            findings=result.findings,
            existing_fingerprints=existing,
            result_id=result.result_id,
        )
        novel_finding_ids = {
            finding_id for contribution in contributions for finding_id in contribution.finding_ids
        }
        result.findings = [
            finding for finding in result.findings if finding.finding_id in novel_finding_ids
        ]
        if result.requirement_satisfied and not result.findings:
            result.requirement_satisfied = False
        result.contribution_ids = [item.contribution_id for item in contributions]
        result.progress = "meaningful" if contributions else "none"

        agenda = self.store.load_agenda()
        if agenda is None:
            raise RuntimeError("Agenda.json is missing during execution commit")
        _, requirement = requirement_index(agenda)[item.requirement_id]
        if result.attempt_class == "scientific":
            requirement.attempt_count += 1
            if item.strategy_fingerprint not in requirement.attempted_strategy_fingerprints:
                requirement.attempted_strategy_fingerprints.append(item.strategy_fingerprint)
        for finding in result.findings:
            if finding.finding_id not in requirement.finding_ids:
                requirement.finding_ids.append(finding.finding_id)
            question = next(q for q in agenda.questions if q.question_id == item.question_id)
            if finding.finding_id not in question.finding_ids:
                question.finding_ids.append(finding.finding_id)
        if result.requirement_satisfied:
            requirement.status = RequirementStatus.satisfied
            requirement.blocker = ""
        elif result.findings:
            requirement.status = RequirementStatus.in_progress
        elif result.failure_class == "engineering" and not result.continue_work:
            # One implementation can be exhausted without exhausting the scientific method for
            # this gap. Keep the requirement schedulable; `_mark_exhausted_requirements` owns the
            # transition to blocked after all distinct strategy slots are consumed.
            requirement.status = (
                RequirementStatus.in_progress
                if requirement.finding_ids
                else RequirementStatus.open
            )
            requirement.blocker = "; ".join(result.errors)[-2000:]
        elif result.errors and not result.continue_work:
            requirement.blocker = "; ".join(result.errors)[-2000:]
        requirement.updated_at = utc_now()

        self.store.append_findings(result.findings)
        self.store.append_contributions(contributions)
        self.store.save_agenda(agenda)
        result_ref = self.store.write_json(f"{run_dir}/result.json", result)

        item.last_result_id = result.result_id
        item.blocked_reason = "; ".join(result.errors) if result.outcome != "done" else ""
        if result.failure_class == "operational":
            item.operational_failures += 1
        else:
            item.operational_failures = 0
        operational_retry = (
            result.failure_class == "operational"
            and item.operational_failures <= self.router.core.max_operational_retries
        )
        resume_same_work = result.continue_work or operational_retry
        item.status = WorkStatus.open if resume_same_work else WorkStatus(result.outcome)
        item.updated_at = utc_now()
        if requirement.status == RequirementStatus.satisfied:
            for pending in queue.items:
                if (
                    pending.work_id != item.work_id
                    and pending.requirement_id == requirement.requirement_id
                    and pending.status == WorkStatus.open
                ):
                    pending.status = WorkStatus.superseded
                    pending.blocked_reason = (
                        f"Requirement {requirement.requirement_id} was already satisfied by "
                        f"result {result.result_id}."
                    )
                    pending.updated_at = utc_now()
        elif not resume_same_work:
            self._schedule_recovery(queue, item, result)

        state.cycle = cycle
        state.active_work_id = None
        state.last_result_id = result.result_id
        state.contribution_count += len(contributions)
        if contributions:
            state.no_progress_steps = 0
            state.last_progress_cycle = cycle
        else:
            state.no_progress_steps += 1
        if self._all_mandatory_satisfied(agenda):
            state.phase = ResearchPhase.complete
        elif result.failure_class == "engineering" and not result.continue_work:
            state.phase = ResearchPhase.working
            state.notes.append(
                f"Blocked experiment {item.requirement_id} without stopping unrelated research: "
                + "; ".join(result.errors)[-1000:]
            )
        elif _next_open(queue) is None:
            state.phase = ResearchPhase.planning
        self.store.save_queue(queue)
        self.store.save_state(state)
        self.store.append_event(
            "work_finished",
            {
                "cycle": cycle,
                "work_id": item.work_id,
                "result_id": result.result_id,
                "outcome": result.outcome,
                "evidence_level": result.evidence_level,
                "requirement_satisfied": result.requirement_satisfied,
                "contribution_ids": result.contribution_ids,
                "progress": result.progress,
                "operational_retry": operational_retry,
                "continue_work": result.continue_work,
                "attempt_class": result.attempt_class,
                "artifact": result_ref.path,
            },
        )
        self._write_reports(state)

    def _schedule_recovery(
        self, queue: WorkQueue, item: WorkItem, result: WorkResult
    ) -> None:
        recoverable = result.failure_class in {"method", "invalid"} or (
            result.evidence_level == "preliminary" and not result.requirement_satisfied
        )
        agenda = self.store.load_agenda()
        attempted_fingerprints: set[str] = set()
        if agenda is not None:
            _, requirement = requirement_index(agenda)[item.requirement_id]
            attempted_fingerprints = set(requirement.attempted_strategy_fingerprints)
        method_items = sum(
            existing.requirement_id == item.requirement_id
            and existing.kind == item.kind
            and existing.strategy_fingerprint in attempted_fingerprints
            for existing in queue.items
        )
        if (
            not recoverable
            or item.revision >= self.router.core.max_strategy_revisions
            or method_items >= self._method_attempt_cap(self._require_state())
        ):
            return
        defects = [*result.errors, *result.next_steps]
        if not defects:
            return
        defect_text = "; ".join(defects)[:1800]
        draft = WorkItemDraft(
            question_id=item.question_id,
            requirement_id=item.requirement_id,
            kind=item.kind,
            title=f"Revision {item.revision + 1}: {item.title}"[:160],
            instruction=(
                item.instruction[:1900]
                + "\n\nRevise the preserved prior strategy rather than restarting generically. "
                + f"Correct these exact defects: {defect_text}"
            )[:4000],
            strategy=(
                item.strategy + f"; revision {item.revision + 1} addressing {defect_text[:600]}"
            )[:1200],
            hypothesis=item.hypothesis,
            falsification_criterion=item.falsification_criterion,
            expected_information_gain=item.expected_information_gain,
            success_criteria=item.success_criteria,
        )
        fingerprint = _strategy_fingerprint(draft)
        if any(existing.strategy_fingerprint == fingerprint for existing in queue.items):
            return
        recovery = self._work_item_from_draft(
            draft, parent=item, prior_result_ids=[*item.prior_result_ids, result.result_id]
        )
        position = queue.items.index(item) + 1
        queue.items.insert(position, recovery)

    def _diversify_after_stagnation(self, state: WorkspaceState, last_requirement_id: str) -> None:
        queue = self.store.load_queue()
        open_indices = [index for index, item in enumerate(queue.items) if item.status == WorkStatus.open]
        if open_indices:
            first = open_indices[0]
            alternative = next(
                (
                    index for index in open_indices
                    if queue.items[index].requirement_id != last_requirement_id
                ),
                None,
            )
            if alternative is not None:
                item = queue.items.pop(alternative)
                queue.items.insert(first, item)
                self.store.save_queue(queue)
        state.diversification_count += 1
        state.notes.append(
            f"Diversified after {state.no_progress_steps} consecutive non-contributing attempts."
        )
        self.store.save_state(state)
        self.store.append_event(
            "stagnation_diversification",
            {"streak": state.no_progress_steps, "last_requirement_id": last_requirement_id},
        )
        self._write_reports(state)

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------
    def _dispatch(self, item: WorkItem, run_dir: str) -> WorkResult:
        context = self._research_context(item)
        if item.kind == WorkKind.literature:
            return LiteraturePipeline(
                self.store, self.router, prompt_dir=self.prompt_dir
            ).run(item, run_dir, research_context=context)
        if item.kind == WorkKind.experiment:
            return ExperimentPipeline(self.store, self.router).run(
                item, run_dir, research_context=context
            )
        if item.kind == WorkKind.proof:
            return ProofPipeline(
                self.store, self.router, prompt_dir=self.prompt_dir
            ).run(item, run_dir, prior_context=context)
        if item.kind == WorkKind.derivation:
            return DerivationPipeline(self.store, self.router).run(
                item, run_dir, prior_context=context
            )
        return self._run_synthesis(item, run_dir)

    def _run_synthesis(self, item: WorkItem, run_dir: str) -> WorkResult:
        report_ref = self.store.write_text(
            f"{run_dir}/synthesis.md",
            _render_research_report(self.store.load_agenda(), self.store.read_findings()),
        )
        return WorkResult(
            work_id=item.work_id,
            outcome="done",
            summary="Regenerated the evidence-ledger research report; synthesis is not new evidence.",
            artifact_refs=[report_ref],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _research_context(self, item: WorkItem) -> dict[str, Any]:
        """Bounded cumulative evidence plus exact parent-revision artifacts."""
        findings = self.store.read_findings()
        ranked = sorted(
            findings,
            key=lambda finding: (
                finding.requirement_id == item.requirement_id,
                finding.question_id == item.question_id,
                finding.strength.value,
                finding.created_at,
            ),
            reverse=True,
        )
        evidence = [
            {
                "finding_id": finding.finding_id,
                "question_id": finding.question_id,
                "requirement_id": finding.requirement_id,
                "status": finding.status.value,
                "polarity": finding.polarity.value,
                "strength": finding.strength.value,
                "statement": finding.statement[:700],
                "scope": finding.scope[:400],
                "caveats": [value[:300] for value in finding.caveats[:2]],
                "artifacts": [ref.path for ref in finding.evidence_refs[:4]],
            }
            for finding in ranked[:8]
        ]
        reusable_code: list[dict[str, Any]] = []
        state_dir = self.store.resolve("ExperimentStates")
        if state_dir.exists():
            for path in sorted(
                state_dir.glob("*.json"), key=lambda value: value.stat().st_mtime, reverse=True
            ):
                try:
                    experiment = ExperimentState.model_validate(self.store.read_json(path))
                except Exception:
                    continue
                if experiment.program is None or experiment.execution_result is None:
                    continue
                reusable_code.append(
                    {
                        "work_id": experiment.work_id,
                        "description": experiment.program.description,
                        "source": experiment.program.python_code[:8_000],
                        "audit_defects": (
                            [
                                *experiment.final_result.errors,
                                *experiment.final_result.next_steps,
                            ][:8]
                            if experiment.final_result is not None
                            else []
                        ),
                    }
                )
                if len(reusable_code) >= 1:
                    break
        agenda = self.store.load_agenda()
        return {
            "research_objective": agenda.objective if agenda else "",
            "agenda_constraints": agenda.constraints if agenda else [],
            "accepted_prior_evidence": evidence,
            "reusable_experiment_code": reusable_code,
            "parent_revision": self._revision_context(item),
        }

    def _revision_context(self, item: WorkItem) -> dict[str, Any]:
        if not item.parent_work_id:
            return {}
        for input_path in sorted(self.store.resolve("Runs").glob("*/input.json"), reverse=True):
            try:
                payload = self.store.read_json(input_path)
            except Exception:
                continue
            work = payload.get("work_item") if isinstance(payload, dict) else None
            if not isinstance(work, dict) or work.get("work_id") != item.parent_work_id:
                continue
            run_dir = input_path.parent
            context: dict[str, Any] = {"parent_work_id": item.parent_work_id}
            result_path = run_dir / "result.json"
            if result_path.exists():
                result = self.store.read_json(result_path)
                context["result"] = {
                    key: result.get(key)
                    for key in ["summary", "errors", "next_steps", "criteria", "evidence_level"]
                }
            for name in [
                "protocol.json", "program.json", "invalid_program.json", "rejected_program.json",
                "derivation.json", "derivation_review.json", "lean_goal.json",
                "lean_goal_attempt_1.json", "lean_goal_attempt_2.json",
            ]:
                path = run_dir / name
                if path.exists():
                    text = path.read_text(encoding="utf-8")
                    context[name] = text[:12000]
            return context
        return {"parent_work_id": item.parent_work_id, "error": "parent run not found"}

    def _mark_exhausted_requirements(self, agenda: ResearchAgenda, queue: WorkQueue) -> None:
        state = self._require_state()
        cap = self._method_attempt_cap(state)
        for _, requirement in requirement_index(agenda).values():
            if requirement.status == RequirementStatus.satisfied:
                continue
            counts = {
                method: sum(
                    item.requirement_id == requirement.requirement_id
                    and item.kind == method
                    and (
                        item.strategy_fingerprint
                        in requirement.attempted_strategy_fingerprints
                        or item.status == WorkStatus.failed
                    )
                    for item in queue.items
                )
                for method in requirement.acceptable_methods
            }
            if counts and all(count >= cap for count in counts.values()):
                requirement.status = RequirementStatus.blocked
                requirement.blocker = (
                    "All configured methods reached their attempt caps: "
                    + ", ".join(f"{method.value}={count}" for method, count in counts.items())
                )
                requirement.updated_at = utc_now()

    def _method_attempt_cap(self, state: WorkspaceState) -> int:
        return (
            self.router.core.max_method_attempts_per_requirement
            + 2 * state.human_replan_count
        )

    @staticmethod
    def _all_mandatory_satisfied(agenda: ResearchAgenda) -> bool:
        mandatory = [
            requirement
            for _, requirement in requirement_index(agenda).values()
            if requirement.mandatory
        ]
        return bool(mandatory) and all(
            requirement.status == RequirementStatus.satisfied for requirement in mandatory
        )

    def _require_state(self) -> WorkspaceState:
        state = self.store.load_state()
        if state is None:
            raise RuntimeError("State.json is missing")
        return state


def _single_subsystem_kind(task: str) -> WorkKind | None:
    lowered = task.lower()
    if re.search(r"full[- ]pipeline|integration test for the full", lowered):
        return None
    if re.search(r"test case for (?:the )?literature subsystem", lowered):
        return WorkKind.literature
    if re.search(r"test case for (?:the )?(?:leap\s*/\s*lean|leap|lean) ", lowered):
        return WorkKind.proof
    if re.search(r"test case for (?:the )?experimenter subsystem", lowered):
        return WorkKind.experiment
    return None


def _restrict_single_subsystem_task(
    draft: ResearchAgendaDraft, task: str
) -> ResearchAgendaDraft:
    """Keep explicit subsystem acceptance workspaces from spawning unrelated pipelines."""
    target = _single_subsystem_kind(task)
    if target is None:
        return draft
    questions = [
        question.model_copy(update={"preferred_methods": [target]})
        for question in draft.questions
    ]
    return draft.model_copy(update={"questions": questions})


def _compact_single_experiment_task(
    draft: ResearchAgendaDraft, task: str
) -> ResearchAgendaDraft:
    """Keep an experiment-only benchmark as one cumulative experiment, not one per artifact."""
    lowered = task.lower()
    if not re.search(r"test case for (?:the )?experimenter subsystem", lowered):
        return draft
    if re.search(r"full[- ]pipeline|leap subsystem|literature subsystem", lowered):
        return draft
    question = _section_text(task, "research question") or _research_objective(
        task, fallback=draft.objective
    )
    requirement = (
        "One reproducible end-to-end benchmark implementing every requested treatment and baseline, "
        "validating them on known cases, using the frozen parameter ranges and fixed seeds, preserving "
        "condition-level JSON measurements for all requested metrics, and producing a comparison "
        "table plus a scoped limitations report under ExperimentRuns/."
    )
    compact = ResearchQuestionDraft(
        question=question[:1200],
        hypotheses=[
            "The registered solver strategies differ in measured runtime or search effort on the sampled instances.",
            "The registered comparison may be null after correctness and uncertainty are accounted for.",
        ],
        evidence_needed=[requirement],
        preferred_methods=[WorkKind.experiment],
    )
    return draft.model_copy(update={"questions": [compact]})


def _section_text(task: str, heading: str) -> str:
    match = re.search(
        rf"(?ims)^#{{1,5}}\s+{re.escape(heading)}\s*$\s*(.+?)(?=^#{{1,5}}\s|\Z)",
        task,
    )
    return re.sub(r"\s+", " ", match.group(1)).strip(" -*") if match else ""


def _research_objective(task: str, *, fallback: str) -> str:
    """Extract the user's objective so reports cannot inherit the planner's meta-instruction."""
    patterns = [
        r"(?ims)^#{1,5}\s+(?:research\s+)?objective\s*$\s*(.+?)(?=^#{1,5}\s|\Z)",
        r"(?ims)^#{1,5}\s+research question\s*$\s*(.+?)(?=^#{1,5}\s|\Z)",
    ]
    for pattern in patterns:
        match = re.search(pattern, task)
        if not match:
            continue
        text = re.sub(r"\s+", " ", match.group(1)).strip(" -*")
        if text:
            return text[:2000]
    if re.search(r"(?i)\bdecompose (?:an?|the) research task\b", fallback):
        return _task_summary(task, limit=1800)
    return fallback


def _bounded_task_context(task: str, *, limit: int = 26000) -> str:
    if len(task) <= limit:
        return task
    headings = "\n".join(
        match.group(0).strip()
        for match in re.finditer(r"(?m)^#{1,5}\s+.+$", task)
    )
    tail_budget = 4000
    head_budget = max(8000, limit - len(headings) - tail_budget - 200)
    return (
        task[:head_budget]
        + "\n\n[Full-document heading index]\n"
        + headings[: max(0, limit - head_budget - tail_budget - 100)]
        + "\n\n[Document tail]\n"
        + task[-tail_budget:]
    )[:limit]


def _methods_for_requirement(
    description: str, preferred: list[WorkKind] | tuple[WorkKind, ...]
) -> list[WorkKind]:
    """Choose methods by the evidence product, not by broad question-level preferences."""
    lowered = description.lower()
    # Formal evidence takes precedence over generic words such as "source code", "compiler", or
    # "test": a Lean snippet is not an empirical experiment.
    if re.search(
        r"\b(?:lean|leap|kernel[- ]checked|formaliz(?:e|ation)|compiler-verified)\b",
        lowered,
    ):
        return [WorkKind.proof]
    if re.search(
        r"\b(?:exact quote|verbatim quote|primary source|from the literature|literature review|published result|"
        r"citation|bibliograph)\b",
        lowered,
    ):
        return [WorkKind.literature]
    if re.search(
        r"\b(?:empirical|experiment(?:al)?|benchmark(?:ing)?|measurement|measured|observed|"
        r"statistical test|compression ratio|p-value|data ?set|plot|runtime comparison|"
        r"python script|implementation|source code|unit tests?|generator|configuration file|"
        r"csv|json results?|visualization)\b",
        lowered,
    ):
        return [WorkKind.experiment]
    if re.search(
        r"\b(?:proof|prove|deriv(?:e|ation)|lower bound|upper bound|complexity|entropy|"
        r"criterion|theorem|reduction|inequality)\b",
        lowered,
    ):
        return [WorkKind.derivation]
    unique: list[WorkKind] = []
    for method in preferred:
        if method != WorkKind.synthesis and method not in unique:
            unique.append(method)
    return unique[:4] or [WorkKind.derivation]
