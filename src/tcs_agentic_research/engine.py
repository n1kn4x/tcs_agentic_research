"""Evidence-gap-driven orchestration for long-running, non-stagnating research.

Models propose typed plans, protocols, derivations, and reviews. Python owns scheduling, provenance,
novelty, requirement state, retries, and stopping. A cycle counts as progress only when it adds a
new evidence-backed contribution; execution, token use, and artifact churn never count by themselves.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from .agents.experiment import ExperimentAgent
from .agents.literature import LiteratureResearcher
from .agents.theorem_prover import TheoremProverAgent
from .artifact_store import ArtifactStore, sha256_file
from .leap.graph import ProofGraph
from .leap.lean import LeanVerifier
from .llm import LLMRouter
from .schemas import (
    ArtifactRef,
    CriterionResult,
    DerivationReview,
    DerivationSubmission,
    EvidenceRequirement,
    EvidenceStrength,
    ExperimentConclusion,
    ExperimentCriterionAssessment,
    ExperimentEvidenceReview,
    ExperimentObservation,
    ExperimentOutput,
    ExperimentProgram,
    ExperimentProgramReview,
    ExperimentProtocol,
    ExperimentProtocolReview,
    ExperimentState,
    NamedDescription,
    Finding,
    FindingPolarity,
    FindingStatus,
    LeanGoalDraft,
    LeanStatement,
    LiteratureEvidenceReview,
    LiteraturePlan,
    PlanSubmission,
    ProofGoalReview,
    RequirementStatus,
    ResearchAgenda,
    ResearchAgendaDraft,
    ResearchPhase,
    ResearchQuestion,
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
    _candidate_is_relevant_and_extractable,
    _compact_query,
    _default_plan,
    _deterministic_agenda,
    _ensure_requested_methods,
    _existing_refs,
    _new_contributions,
    _next_open,
    _normalize_work_draft,
    _rank_candidates,
    _recent_result_context,
    _render_literature_report,
    _render_progress_report,
    _render_research_report,
    _strategy_fingerprint,
    _task_summary,
    _validate_experiment_program,
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
                item = _next_open(queue)
                if item is None:
                    if not self._plan(state, queue):
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
                    "Decompose an uncertain research request into at most twelve narrow questions. "
                    "For every question state one to four falsifiable working hypotheses and concrete "
                    "evidence needs. Never treat a hypothesis in the request as true. Constraints are "
                    "only explicit user scope/method/accounting rules, not predicted outcomes. Choose "
                    "methods precisely: literature obtains primary-source statements; experiment "
                    "produces measurements; derivation produces explicit mathematical arguments or "
                    "counterexamples; proof means Lean kernel verification and should be selected only "
                    "when the relevant concepts can realistically be expressed in the configured Lean "
                    "project; synthesis is not an evidence-producing method. Deliverables must be "
                    "auditable. Negative and null results must be able to satisfy an evidence need."
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
        questions: list[ResearchQuestion] = []
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
                        f"Directly resolves: {description}",
                        "States the assumptions, scope, and limitations needed to interpret the result.",
                    ],
                    acceptable_methods=_methods_for_requirement(description, methods),
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
            objective=draft.objective,
            constraints=draft.constraints,
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
    def _plan(self, state: WorkspaceState, queue: WorkQueue) -> bool:
        agenda = self.store.load_agenda()
        if agenda is None:
            agenda = self._ensure_agenda(state)
        if self._all_mandatory_satisfied(agenda):
            state.phase = ResearchPhase.complete
            self.store.save_state(state)
            self._write_reports(state)
            return False
        self._mark_exhausted_requirements(agenda, queue)
        self.store.save_agenda(agenda)
        available = _default_plan(
            agenda=agenda,
            queue=queue,
            max_method_attempts=self._method_attempt_cap(state),
            limit=self.router.core.max_plan_items,
        )
        if not available.work_items:
            state.phase = ResearchPhase.needs_input
            blocked = [
                requirement.requirement_id
                for _, requirement in requirement_index(agenda).values()
                if requirement.mandatory and requirement.status != RequirementStatus.satisfied
            ]
            state.notes.append("Exhausted configured strategies for mandatory gaps: " + ", ".join(blocked))
            self.store.save_state(state)
            self.store.append_event("agenda_exhausted", {"requirements": blocked})
            self._write_reports(state)
            return False

        task = self.store.read_text(ArtifactStore.RESEARCH_TASK)
        compact_requirements = [
            {
                "question_id": question.question_id,
                "question": question.question,
                "hypotheses": question.hypotheses,
                "requirement_id": requirement.requirement_id,
                "description": requirement.description,
                "acceptance_criteria": requirement.acceptance_criteria,
                "acceptable_methods": [method.value for method in requirement.acceptable_methods],
                "status": requirement.status.value,
                "attempt_count": requirement.attempt_count,
                "prior_strategy_fingerprints": requirement.attempted_strategy_fingerprints[-8:],
                "blocker": requirement.blocker[-500:],
            }
            for question in agenda.questions
            for requirement in question.requirements
            if requirement.status != RequirementStatus.satisfied
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "Propose a small portfolio of evidence-producing strategies for the supplied "
                    "atomic requirements. Every item must use an exact question_id and requirement_id, "
                    "an allowed method, a concrete strategy, a working hypothesis, an outcome that "
                    "would falsify it, expected information gain in either direction, and acceptance "
                    "criteria. Do not repeat a prior strategy with cosmetic wording. Prefer a targeted "
                    "repair when a recent attempt has one concrete defect; otherwise diversify methods "
                    "or questions. Negative, null, counterexample, and obstruction results are successes. "
                    "Synthesis is not evidence and may not be scheduled for an open gap."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": task[:6000],
                        "constraints": agenda.constraints,
                        "open_requirements": compact_requirements,
                        "recent_results": _recent_result_context(self.store, limit=12),
                        "plan_round": state.plan_round + 1,
                        "consecutive_no_progress": state.no_progress_steps,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        run_dir = self.store.create_run_dir(state.cycle + 1, f"plan_{state.plan_round + 1}")
        self.store.write_json(f"{run_dir}/input.json", {"messages": messages})
        error = ""
        try:
            with self.router.step_budget(f"plan_{state.plan_round + 1}", max_calls=2):
                plan = self.router.complete_structured(
                    task_type="planning",
                    messages=messages,
                    schema=PlanSubmission,
                    mock_output=available if self.router.dry_run else None,
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            plan = available
            self.store.write_text(f"{run_dir}/planning_error.log", error + "\n")
        self.store.write_json(f"{run_dir}/plan.json", plan)

        requirements = requirement_index(agenda)
        existing_fingerprints = {item.strategy_fingerprint for item in queue.items}
        drafts: list[WorkItemDraft] = []
        for draft in plan.work_items:
            pair = requirements.get(draft.requirement_id)
            if pair is None:
                continue
            question, requirement = pair
            if question.question_id != draft.question_id:
                continue
            if requirement.status == RequirementStatus.satisfied:
                continue
            if draft.kind not in requirement.acceptable_methods or draft.kind == WorkKind.synthesis:
                continue
            normalized = _normalize_work_draft(
                draft, question=question, requirement=requirement
            )
            fingerprint = _strategy_fingerprint(normalized)
            if fingerprint in existing_fingerprints:
                continue
            attempts = sum(
                item.requirement_id == requirement.requirement_id
                and item.kind == draft.kind
                and item.strategy_fingerprint in requirement.attempted_strategy_fingerprints
                for item in queue.items
            ) + sum(
                prior.requirement_id == requirement.requirement_id and prior.kind == draft.kind
                for prior in drafts
            )
            if attempts >= self._method_attempt_cap(state):
                continue
            drafts.append(normalized)
            existing_fingerprints.add(fingerprint)
        if not drafts:
            for draft in available.work_items:
                fingerprint = _strategy_fingerprint(draft)
                if fingerprint not in existing_fingerprints:
                    drafts.append(draft)
                    existing_fingerprints.add(fingerprint)
        new_items = [self._work_item_from_draft(draft) for draft in drafts[: self.router.core.max_plan_items]]
        state.plan_round += 1
        if new_items:
            queue.items.extend(new_items)
            state.phase = ResearchPhase.working
        else:
            state.phase = ResearchPhase.planning
        self.store.save_queue(queue)
        self.store.save_state(state)
        self.store.append_event(
            "plan_recorded",
            {
                "plan_round": state.plan_round,
                "new_work_ids": [item.work_id for item in new_items],
                "planning_error": error,
            },
        )
        self._write_reports(state)
        return bool(new_items)

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
                attempt_class=(
                    "engineering" if item.kind == WorkKind.experiment else "scientific"
                ),
                continue_work=item.kind == WorkKind.experiment,
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
        elif result.errors and not result.continue_work:
            requirement.blocker = "; ".join(result.errors)[-2000:]
        requirement.updated_at = utc_now()

        self.store.append_findings(result.findings)
        self.store.append_contributions(contributions)
        self.store.save_agenda(agenda)
        result_ref = self.store.write_json(f"{run_dir}/result.json", result)

        item.last_result_id = result.result_id
        item.blocked_reason = "; ".join(result.errors) if result.outcome != "done" else ""
        operational_retry = (
            result.failure_class == "operational"
            and item.attempts <= self.router.core.max_operational_retries
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
            state.phase = ResearchPhase.system_error
            state.notes.append(
                f"Experiment pipeline blocked before evidence for {item.requirement_id}: "
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
        if item.kind == WorkKind.literature:
            return self._run_literature(item, run_dir)
        if item.kind == WorkKind.experiment:
            return self._run_experiment(item, run_dir)
        if item.kind == WorkKind.proof:
            return self._run_proof(item, run_dir)
        if item.kind == WorkKind.derivation:
            return self._run_derivation(item, run_dir)
        return self._run_synthesis(item, run_dir)

    def _run_literature(self, item: WorkItem, run_dir: str) -> WorkResult:
        fallback_query = _compact_query(item.instruction)
        mock = LiteraturePlan(search_queries=[fallback_query], focus_questions=[fallback_query])
        messages = [
            {
                "role": "system",
                "content": (
                    "Create a targeted primary-literature plan for one evidence requirement. Return "
                    "specific theorem/definition/result queries and source titles only when confident. "
                    "Do not invent identifiers. Include queries likely to find contradictory results."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"work_item": item.model_dump(mode="json")}, ensure_ascii=False
                ),
            },
        ]
        refs: list[ArtifactRef] = [
            self.store.write_json(f"{run_dir}/literature_model_input.json", {"messages": messages})
        ]
        errors: list[str] = []
        try:
            plan = self.router.complete_structured(
                task_type="literature_planning",
                messages=messages,
                schema=LiteraturePlan,
                mock_output=mock if self.router.dry_run else None,
            )
        except Exception as exc:
            errors.append(f"planning: {type(exc).__name__}: {exc}")
            plan = mock
        refs.append(self.store.write_json(f"{run_dir}/literature_plan.json", plan))
        literature = LiteratureResearcher(self.store, self.router, prompt_dir=self.prompt_dir)
        candidates: list[Any] = []
        if not self.router.dry_run:
            for query in list(dict.fromkeys([*plan.known_source_titles, *plan.search_queries])):
                try:
                    candidates.extend(
                        literature.search_papers(
                            query, limit=self.router.core.literature_results_per_query
                        )
                    )
                except Exception as exc:
                    errors.append(f"search {query!r}: {type(exc).__name__}: {exc}")
        ranked = [
            candidate
            for candidate in _rank_candidates(
                candidates,
                preferred_titles=plan.known_source_titles,
                relevance_queries=plan.search_queries,
            )
            if _candidate_is_relevant_and_extractable(
                candidate,
                preferred_titles=plan.known_source_titles,
                relevance_queries=plan.search_queries,
            )
        ]
        imported: list[Any] = []
        attempted_imports = 0
        for candidate in ranked:
            if len(imported) >= self.router.core.literature_max_imports:
                break
            if attempted_imports >= self.router.core.literature_import_attempts:
                break
            attempted_imports += 1
            try:
                imported.append(literature.import_candidate(candidate.candidate_id, extract_text=True))
            except Exception as exc:
                errors.append(f"import {candidate.title!r}: {type(exc).__name__}: {exc}")
        try:
            extraction = literature.extract_imported_papers(
                max_papers=max(1, min(2, self.router.core.literature_max_imports)),
                only_missing=True,
                use_llm=not self.router.dry_run,
            )
        except Exception as exc:
            extraction = {"processed_count": 0, "errors": [str(exc)]}
            errors.append(f"extraction: {type(exc).__name__}: {exc}")
        refs.append(self.store.write_json(f"{run_dir}/extraction.json", extraction))
        answers = []
        for question in plan.focus_questions:
            try:
                answers.append(
                    literature.answer_query(
                        question, limit=self.router.core.literature_results_per_query
                    )
                )
            except Exception as exc:
                errors.append(f"query {question!r}: {type(exc).__name__}: {exc}")
        refs.append(
            self.store.write_json(
                f"{run_dir}/query_answers.json",
                [answer.model_dump(mode="json") for answer in answers],
            )
        )
        by_support: dict[str, Any] = {}
        for answer in answers:
            for row in answer.results:
                if not (row.provenance and row.provenance[0].validated and row.statement_text):
                    continue
                quote = row.provenance[0]
                support_id = row.support_id or _passage_support_id(
                    row.citation_key,
                    quote.char_start,
                    quote.char_end,
                    quote.quote,
                )
                by_support[support_id] = row
        accepted: dict[str, str] = {}
        review_ref: ArtifactRef | None = None
        if by_support and not self.router.dry_run:
            review_messages = [
                {
                    "role": "system",
                    "content": (
                        "Review exact primary-source statements against one evidence requirement. "
                        "Mark relevant only when the quoted statement directly supplies needed evidence; "
                        "topical overlap is unrelated. Classify supports, contradicts, or characterizes "
                        "without extending beyond the quote. Review every support_id supplied."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "requirement": item.instruction,
                            "hypothesis": item.hypothesis,
                            "statements": [
                                {
                                    "support_id": support_id,
                                    "citation_key": row.citation_key,
                                    "statement": row.statement_text,
                                }
                                for support_id, row in list(by_support.items())[:20]
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
            review = self.router.complete_structured(
                task_type="literature_review",
                messages=review_messages,
                schema=LiteratureEvidenceReview,
                allow_repair=False,
            )
            review_ref = self.store.write_json(f"{run_dir}/evidence_review.json", review)
            refs.append(review_ref)
            accepted = {
                selection.support_id: selection.relation
                for selection in review.selections
                if selection.relevant
                and selection.relation != "unrelated"
                and selection.support_id in by_support
            }
        findings: list[Finding] = []
        for support_id, relation in accepted.items():
            row = by_support[support_id]
            quote = row.provenance[0]
            findings.append(
                Finding(
                    work_id=item.work_id,
                    question_id=item.question_id,
                    requirement_id=item.requirement_id,
                    kind=WorkKind.literature,
                    statement=f"[{row.citation_key}] {row.statement_text}",
                    status=FindingStatus.supported,
                    polarity=(
                        FindingPolarity.contradicts
                        if relation == "contradicts"
                        else FindingPolarity.supports
                        if relation == "supports"
                        else FindingPolarity.characterizes
                    ),
                    strength=(
                        EvidenceStrength.strong
                        if row.support_id and row.support_level == "primary_exact"
                        else EvidenceStrength.substantive
                    ),
                    scope="Exactly the assumptions and scope stated in the quoted primary source.",
                    evidence_refs=quote.artifact_refs,
                    source_ids=[
                        value for value in [support_id, row.statement_id, row.quote_id] if value
                    ],
                )
            )
        report_ref = self.store.write_text(
            f"{run_dir}/literature_report.md",
            _render_literature_report(item, plan, candidates, imported, findings, errors),
        )
        refs.append(report_ref)
        criteria = [
            CriterionResult(
                criterion=criterion,
                satisfied=bool(findings),
                detail=(
                    f"{len(findings)} exact statement(s) passed relevance review."
                    if findings
                    else "No exact statement passed requirement-level relevance review."
                ),
            )
            for criterion in item.success_criteria
        ]
        return WorkResult(
            work_id=item.work_id,
            outcome="done" if findings else "partial" if candidates else "blocked",
            failure_class="none" if findings else "evidence_gap",
            evidence_level="substantive" if findings else "none",
            requirement_satisfied=bool(findings),
            criteria=criteria,
            summary=(
                f"Accepted {len(findings)} quote-validated, requirement-relevant source result(s)."
                if findings
                else "The bounded search produced no requirement-relevant exact source result."
            ),
            findings=findings,
            artifact_refs=_existing_refs(
                self.store,
                refs,
                [
                    "LiteratureDB/candidates.jsonl",
                    "LiteratureDB/papers.jsonl",
                    "LiteratureDB/statements.jsonl",
                    "LiteratureDB/index.sqlite",
                ],
            ),
            errors=errors,
            next_steps=[
                "Use the documented failed candidates and queries to search a distinct title, preprint, or citation trail."
            ] if not findings else [],
        )

    def _run_derivation(self, item: WorkItem, run_dir: str) -> WorkResult:
        if self.router.dry_run:
            artifact = self.store.write_text(
                f"{run_dir}/derivation.md",
                "# Dry run\n\nNo mathematical claim is generated or accepted in dry-run mode.\n",
            )
            return WorkResult(
                work_id=item.work_id,
                outcome="partial",
                failure_class="none",
                summary="Dry run validated derivation control flow without creating evidence.",
                artifact_refs=[artifact],
            )
        revision = self._revision_context(item)
        messages = [
            {
                "role": "system",
                "content": (
                    "Produce one self-contained mathematical result for the exact evidence gap. A "
                    "counterexample, obstruction, equivalence, sharp boundary, or corrected weaker "
                    "claim is as valuable as a proof. State assumptions and definitions; use labelled "
                    "steps whose dependencies are explicit; do not assume the target claim; do not "
                    "smuggle an empirical regularity into a universal theorem. Actively test small and "
                    "boundary cases in falsification_attempt. Scope the conclusion to what was shown."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "work_item": item.model_dump(mode="json"),
                        "prior_revision_context": revision,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        input_ref = self.store.write_json(f"{run_dir}/derivation_input.json", {"messages": messages})
        derivation = self.router.complete_structured(
            task_type="derivation",
            messages=messages,
            schema=DerivationSubmission,
            allow_repair=True,
        )
        draft_ref = self.store.write_json(f"{run_dir}/derivation.json", derivation)
        review_messages = [
            {
                "role": "system",
                "content": (
                    "Act as an adversarial mathematical referee. Recompute key transitions, check "
                    "quantifiers and edge cases, detect circular premises and omitted costs, and try to "
                    "construct a concrete counterexample. Evaluate every success criterion. Reject a "
                    "restatement, plausibility argument, invalid universalization, or result that does "
                    "not resolve the named gap. Accept negative/counterexample results when valid."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "work_item": item.model_dump(mode="json"),
                        "derivation": derivation.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        review = self.router.complete_structured(
            task_type="derivation_review",
            messages=review_messages,
            schema=DerivationReview,
            allow_repair=False,
        )
        review_ref = self.store.write_json(f"{run_dir}/derivation_review.json", review)
        missing_criteria = _missing_criterion_reviews(item.success_criteria, review.criteria)
        accepted = (
            review.accepted
            and review.confidence >= 0.65
            and not missing_criteria
        )
        if not accepted:
            issues = [*review.fatal_issues, *review.required_revisions]
            if review.accepted and review.confidence < 0.65:
                issues.append(f"Referee confidence {review.confidence:.2f} is below 0.65.")
            if missing_criteria:
                issues.append(
                    "Referee omitted mandatory criterion review(s): "
                    + "; ".join(missing_criteria)
                )
            return WorkResult(
                work_id=item.work_id,
                outcome="partial",
                failure_class="method",
                criteria=review.criteria,
                summary="The mathematical derivation did not pass adversarial review.",
                artifact_refs=[input_ref, draft_ref, review_ref],
                errors=issues,
                next_steps=review.required_revisions,
            )
        polarity = (
            FindingPolarity.contradicts
            if derivation.result_kind in {"counterexample", "obstruction"}
            else FindingPolarity.characterizes
            if derivation.result_kind in {"characterization", "equivalence"}
            else FindingPolarity.supports
        )
        status = (
            FindingStatus.refuted
            if derivation.result_kind == "counterexample"
            else FindingStatus.derived
        )
        finding = Finding(
            work_id=item.work_id,
            question_id=item.question_id,
            requirement_id=item.requirement_id,
            kind=WorkKind.derivation,
            statement=derivation.conclusion,
            status=status,
            polarity=polarity,
            strength=EvidenceStrength.substantive,
            scope="; ".join(derivation.assumptions),
            evidence_refs=[draft_ref, review_ref],
            source_ids=[],
            caveats=[
                "This is an explicit derivation accepted by an automated adversarial referee, not a kernel-checked proof.",
                *derivation.limitations,
            ],
        )
        return WorkResult(
            work_id=item.work_id,
            outcome="done",
            evidence_level="substantive",
            requirement_satisfied=True,
            criteria=review.criteria,
            summary=review.summary,
            findings=[finding],
            artifact_refs=[input_ref, draft_ref, review_ref],
        )

    def _run_proof(self, item: WorkItem, run_dir: str) -> WorkResult:
        verifier = LeanVerifier(
            self.store,
            timeout_seconds=self.router.leap.compiler_timeout_seconds,
            memory_mb=self.router.leap.compiler_memory_mb,
        )
        verifier.ensure_project()
        if not verifier.available():
            return WorkResult(
                work_id=item.work_id,
                outcome="blocked",
                failure_class="operational",
                summary="Lean is unavailable; no formulation call was spent.",
                errors=["Neither lake nor lean is available on PATH."],
                next_steps=["Install the configured Lean toolchain and retry unchanged work."],
            )
        mock = LeanGoalDraft(name="dry_run_goal", statement="∀ n : Nat, n = n")
        revision = self._revision_context(item)
        formulation_messages = [
            {
                "role": "system",
                "content": (
                    "Formulate one smallest nontrivial Lean proposition that directly advances the "
                    "evidence requirement. Return only a proposition type with explicit binders. Use "
                    "only TCSResearch.Basic and Lean core. Do not invent entropy/probability APIs, do "
                    "not return declarations or proofs, and do not retreat to reflexivity, True, or an "
                    "unrelated tautology. If the research concept is unavailable, encode a precise "
                    "finite/combinatorial supporting fact rather than the full theorem."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"work_item": item.model_dump(mode="json"), "prior_failures": revision},
                    ensure_ascii=False,
                ),
            },
        ]
        refs = [
            self.store.write_json(
                f"{run_dir}/proof_formulation_input.json", {"messages": formulation_messages}
            )
        ]
        goal: LeanStatement | None = None
        last_error = ""
        for attempt in range(2):
            goal_draft = self.router.complete_structured(
                task_type="proof_formulation",
                messages=formulation_messages,
                schema=LeanGoalDraft,
                mock_output=mock if self.router.dry_run else None,
            )
            refs.append(self.store.write_json(f"{run_dir}/lean_goal_attempt_{attempt + 1}.json", goal_draft))
            candidate = LeanStatement(
                name=goal_draft.name,
                statement=goal_draft.statement,
                imports=["TCSResearch.Basic"],
                namespace=goal_draft.namespace,
            )
            _, check = verifier.elaborate_statement(candidate)
            if check.log_path and self.store.exists(check.log_path):
                refs.append(self.store.artifact_ref(check.log_path))
            if check.accepted and not _obviously_trivial_goal(candidate.statement):
                goal = candidate
                break
            last_error = check.reason if not check.accepted else "Goal is a trivial or irrelevant tautology."
            formulation_messages = [
                {
                    "role": "system",
                    "content": (
                        "Repair the proposition after elaboration or relevance rejection. Preserve the "
                        "research dependency, simplify syntax, and do not use reflexivity or True."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "work_item": item.model_dump(mode="json"),
                            "rejected_goal": goal_draft.model_dump(mode="json"),
                            "error": last_error,
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
        if goal is None:
            return WorkResult(
                work_id=item.work_id,
                outcome="partial",
                failure_class="invalid",
                summary="No nontrivial elaborated Lean proposition was produced.",
                artifact_refs=refs,
                errors=[last_error or "formulation failed"],
                next_steps=["Use a smaller explicit combinatorial proposition or switch to derivation."],
            )
        goal_ref = self.store.write_json(f"{run_dir}/lean_goal.json", goal)
        refs.append(goal_ref)
        if self.router.dry_run:
            return WorkResult(
                work_id=item.work_id,
                outcome="partial",
                summary="Dry run elaborated a goal but created no proof evidence.",
                artifact_refs=refs,
            )
        review_messages = [
            {
                "role": "system",
                "content": (
                    "Review whether this exact Lean proposition is nontrivial and lies on a concrete "
                    "dependency path to the evidence requirement. Reject generic library facts, "
                    "reflexivity in disguise, omitted assumptions, and facts too weak to reuse."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"work_item": item.model_dump(mode="json"), "goal": goal.model_dump(mode="json")},
                    ensure_ascii=False,
                ),
            },
        ]
        goal_review = self.router.complete_structured(
            task_type="proof_formulation_review",
            messages=review_messages,
            schema=ProofGoalReview,
            allow_repair=False,
        )
        review_ref = self.store.write_json(f"{run_dir}/lean_goal_review.json", goal_review)
        refs.append(review_ref)
        if not goal_review.accepted:
            return WorkResult(
                work_id=item.work_id,
                outcome="partial",
                failure_class="method",
                summary="The elaborated Lean goal failed dependency/relevance review.",
                artifact_refs=refs,
                errors=goal_review.issues,
                next_steps=[goal_review.route_to_requirement],
            )
        before: set[str] = set()
        if self.store.exists(ProofGraph.DB_PATH):
            before = {node.fingerprint for node in ProofGraph(self.store).proved_nodes(limit=10000)}
        proof_result = TheoremProverAgent(
            self.store, self.router, prompt_dir=self.prompt_dir
        ).prove(goal, context=f"{item.instruction}\nDependency route: {goal_review.route_to_requirement}")
        result_ref = self.store.write_json(f"{run_dir}/lean_result.json", proof_result)
        refs.append(result_ref)
        findings: list[Finding] = []
        if proof_result.status == "proved":
            findings.append(
                Finding(
                    work_id=item.work_id,
                    question_id=item.question_id,
                    requirement_id=item.requirement_id,
                    kind=WorkKind.proof,
                    statement=f"Lean verified `{goal.name} : {goal.statement}`.",
                    status=FindingStatus.verified,
                    polarity=FindingPolarity.supports,
                    strength=EvidenceStrength.conclusive,
                    scope=goal_review.route_to_requirement,
                    evidence_refs=proof_result.proved_artifacts,
                    source_ids=[proof_result.result_id],
                )
            )
        elif self.store.exists(ProofGraph.DB_PATH):
            graph = ProofGraph(self.store)
            for node in graph.proved_nodes(limit=10000):
                if node.fingerprint in before:
                    continue
                evidence: list[ArtifactRef] = []
                if node.proof_artifact_path and self.store.exists(node.proof_artifact_path):
                    evidence.append(self.store.artifact_ref(node.proof_artifact_path))
                findings.append(
                    Finding(
                        work_id=item.work_id,
                        question_id=item.question_id,
                        requirement_id=item.requirement_id,
                        kind=WorkKind.proof,
                        statement=f"Lean verified new supporting lemma `{node.goal.name} : {node.goal.statement}`.",
                        status=FindingStatus.verified,
                        polarity=FindingPolarity.characterizes,
                        strength=EvidenceStrength.preliminary,
                        scope=goal_review.route_to_requirement,
                        evidence_refs=evidence,
                        source_ids=[node.node_id],
                        caveats=["The supporting lemma does not by itself close the parent requirement."],
                    )
                )
        root_proved = proof_result.status == "proved"
        criteria = [
            CriterionResult(
                criterion=criterion,
                satisfied=root_proved or bool(findings),
                detail=(
                    "The root is kernel checked." if root_proved
                    else f"Produced {len(findings)} new kernel-checked supporting lemma(s)."
                    if findings else proof_result.proof_dag_summary[:1200] or "No verified node."
                ),
            )
            for criterion in item.success_criteria
        ]
        return WorkResult(
            work_id=item.work_id,
            outcome="done" if root_proved else "partial",
            failure_class="none" if findings else "method",
            evidence_level="conclusive" if root_proved else "preliminary" if findings else "none",
            requirement_satisfied=root_proved,
            criteria=criteria,
            summary=f"Persistent LEAP search ended with `{proof_result.status}`.",
            findings=findings,
            artifact_refs=[*refs, *proof_result.artifact_refs],
            errors=[] if findings else [proof_result.proof_dag_summary],
            next_steps=proof_result.recommended_next_steps,
        )

    def _run_experiment(self, item: WorkItem, run_dir: str) -> WorkResult:
        """Advance one durable stage of an experiment without discarding accepted work."""
        state_path = f"ExperimentStates/{item.work_id}.json"
        if self.store.exists(state_path):
            experiment = ExperimentState.model_validate(self.store.read_json(state_path))
        else:
            experiment = ExperimentState(work_id=item.work_id)
        refs: list[ArtifactRef] = []

        def persist() -> None:
            experiment.updated_at = utc_now()
            state_ref = self.store.write_json(state_path, experiment)
            snapshot_ref = self.store.write_json(f"{run_dir}/experiment_state.json", experiment)
            refs.extend([state_ref, snapshot_ref])

        def engineering_result(
            summary: str,
            *,
            errors: list[str] | None = None,
            next_steps: list[str] | None = None,
        ) -> WorkResult:
            errors = errors or []
            if errors:
                experiment.engineering_failures += 1
                experiment.last_error = "; ".join(errors)[-4000:]
            if experiment.engineering_failures >= self.router.core.max_experiment_engineering_retries:
                experiment.engineering_blocked = True
                persist()
                return WorkResult(
                    work_id=item.work_id,
                    outcome="failed",
                    failure_class="engineering",
                    attempt_class="engineering",
                    summary=(
                        "Experiment pipeline exhausted its engineering repair budget before "
                        "producing a scientific execution."
                    ),
                    artifact_refs=refs,
                    errors=errors or [experiment.last_error],
                    next_steps=[
                        "Inspect the preserved ExperimentState and repair the pipeline or model profile."
                    ],
                )
            persist()
            return WorkResult(
                work_id=item.work_id,
                outcome="partial",
                failure_class="engineering" if errors else "none",
                attempt_class="engineering",
                continue_work=True,
                summary=summary,
                artifact_refs=refs,
                errors=errors,
                next_steps=next_steps or [],
            )

        if experiment.engineering_blocked:
            persist()
            return WorkResult(
                work_id=item.work_id,
                outcome="failed",
                failure_class="engineering",
                attempt_class="engineering",
                summary="Experiment pipeline is blocked on a preserved engineering failure.",
                artifact_refs=refs,
                errors=[experiment.last_error],
            )

        settings = self.router.experimenter
        if not self.router.dry_run and (settings is None or not settings.enabled):
            experiment.last_error = "The experimenter is not configured and enabled."
            experiment.engineering_blocked = True
            persist()
            return WorkResult(
                work_id=item.work_id,
                outcome="failed",
                failure_class="engineering",
                attempt_class="engineering",
                summary="Experiment infrastructure is not configured.",
                artifact_refs=refs,
                errors=[experiment.last_error],
            )

        max_wall = settings.timeout_seconds if settings else 600
        max_memory = _memory_to_mb(settings.memory) if settings else 4096
        max_cpus = settings.cpus if settings else 2.0
        mock_protocol = ExperimentProtocol(
            title="Dry-run bounded comparison",
            hypothesis=item.hypothesis,
            null_outcome="No measured difference is observed between treatment and baseline.",
            experimental_unit="one deterministically generated small instance",
            conditions=[
                NamedDescription(id="treatment", description="Treatment implementation."),
                NamedDescription(id="baseline", description="Strong baseline implementation."),
            ],
            baselines=[NamedDescription(id="baseline", description="Strong baseline implementation.")],
            metrics=[NamedDescription(id="measured_value", description="Primary measured value.")],
            correctness_checks=[
                NamedDescription(id="roundtrip", description="Round-trip reconstruction succeeds.")
            ],
            sample_sizes=[10],
            seeds=[0],
            analysis_plan="Compare condition measurements and preserve every observation.",
            decision_rule="Classify supports, contradicts, or null from the signed difference.",
            wall_seconds=min(30, max_wall),
            memory_mb=min(512, max_memory),
            cpus=min(1.0, max_cpus),
            known_limitations=["Small synthetic smoke-scale experiment."],
        )

        if experiment.stage in {"protocol_design", "protocol_revision"}:
            revision = experiment.stage == "protocol_revision"
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Design one bounded falsifiable experiment as the supplied typed protocol. "
                        "Use stable short ids for conditions, baselines, metrics, and checks. Every "
                        "baseline id must also be a condition id. Preserve negative and null outcomes. "
                        "Use synthetic data only when real data is not required. The requested CPU, "
                        "memory, and wall time must fit the supplied runtime limits."
                        + (
                            " Revise the preserved protocol only to correct the listed defects; do not "
                            "change the research question or expected outcome."
                            if revision
                            else ""
                        )
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "work_item": item.model_dump(mode="json"),
                            "agenda_constraints": self.store.load_agenda().constraints,  # type: ignore[union-attr]
                            "runtime_limits": {
                                "wall_seconds": max_wall,
                                "memory_mb": max_memory,
                                "cpus": max_cpus,
                            },
                            "previous_protocol": (
                                experiment.protocol.model_dump(mode="json")
                                if experiment.protocol else None
                            ),
                            "defects": experiment.last_error if revision else "",
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
            refs.append(self.store.write_json(f"{run_dir}/protocol_input.json", {"messages": messages}))
            try:
                protocol = self.router.complete_structured(
                    task_type="experiment_protocol",
                    messages=messages,
                    schema=ExperimentProtocol,
                    mock_output=mock_protocol if self.router.dry_run else None,
                    temperature=0.1,
                    max_tokens=4096,
                    allow_repair=True,
                )
            except Exception as exc:  # one failed generation is repairable engineering work
                return engineering_result(
                    "Experiment protocol generation failed and will resume from the same stage.",
                    errors=[f"{type(exc).__name__}: {exc}"],
                )
            resource_errors = _experiment_resource_errors(
                protocol,
                max_wall=max_wall,
                max_memory=max_memory,
                max_cpus=max_cpus,
            )
            experiment.protocol = protocol
            experiment.protocol_revision += int(revision)
            refs.append(self.store.write_json(f"{run_dir}/protocol.json", protocol))
            if resource_errors:
                experiment.stage = "protocol_revision"
                return engineering_result(
                    "Protocol exceeded the executable resource limits and will be revised.",
                    errors=resource_errors,
                )
            experiment.stage = "protocol_review"
            experiment.last_error = ""
            return engineering_result("Experiment protocol drafted; independent review is next.")

        if experiment.stage == "protocol_review":
            assert experiment.protocol is not None
            criteria = {
                "P_ALIGNMENT": "The protocol directly measures the stated evidence requirement.",
                "P_NULL": "The null outcome and decision rule are explicit and compatible.",
                "P_BASELINES": "The conditions include genuinely distinct strong baselines.",
                "P_CHECKS": "Correctness checks test implementation validity, not hypothesis direction.",
                "P_SAMPLING": "Seeds, sample sizes, and analysis are reproducible and feasible.",
                "P_COSTS": "Dominant costs and executable resource limits are represented.",
            }
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Audit the frozen candidate protocol. Return exactly one assessment for every "
                        "criterion_id supplied by the user, with no renamed or additional ids. Python "
                        "computes acceptance; you only assess each criterion and list concrete revisions."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "work_item": item.model_dump(mode="json"),
                            "protocol": experiment.protocol.model_dump(mode="json"),
                            "criteria": [
                                {"criterion_id": key, "text": text}
                                for key, text in criteria.items()
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
            refs.append(
                self.store.write_json(f"{run_dir}/protocol_review_input.json", {"messages": messages})
            )
            mock_review = ExperimentProtocolReview(
                criteria=[
                    ExperimentCriterionAssessment(
                        criterion_id=key,
                        satisfied=True,
                        detail="Dry-run deterministic acceptance.",
                    )
                    for key in criteria
                ]
            )
            try:
                protocol_review = self.router.complete_structured(
                    task_type="experiment_review",
                    messages=messages,
                    schema=ExperimentProtocolReview,
                    mock_output=mock_review if self.router.dry_run else None,
                    temperature=0.1,
                    max_tokens=3072,
                    allow_repair=False,
                )
            except Exception as exc:
                return engineering_result(
                    "Protocol review failed structurally and will retry unchanged.",
                    errors=[f"{type(exc).__name__}: {exc}"],
                )
            experiment.protocol_review = protocol_review
            refs.append(self.store.write_json(f"{run_dir}/protocol_review.json", protocol_review))
            coverage_errors = _experiment_criterion_errors(criteria, protocol_review.criteria)
            failed = [row.criterion_id for row in protocol_review.criteria if not row.satisfied]
            errors = [
                *coverage_errors,
                *protocol_review.issues,
                *protocol_review.required_revisions,
            ]
            if failed:
                errors.append("Failed protocol criteria: " + ", ".join(failed))
            if errors:
                experiment.stage = "protocol_revision"
                return engineering_result(
                    "Protocol review found concrete defects; the same protocol lineage will be revised.",
                    errors=errors,
                    next_steps=protocol_review.required_revisions,
                )
            protocol_payload = json.dumps(
                experiment.protocol.model_dump(mode="json"), sort_keys=True, ensure_ascii=False
            )
            experiment.protocol_sha256 = hashlib.sha256(protocol_payload.encode("utf-8")).hexdigest()
            experiment.stage = "program_design"
            experiment.last_error = ""
            return engineering_result("Protocol accepted and frozen; program implementation is next.")

        if experiment.stage in {"program_design", "program_revision"}:
            assert experiment.protocol is not None and experiment.protocol_sha256
            revision = experiment.stage == "program_revision"
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Implement the frozen protocol as Python source defining exactly "
                        "run_experiment(mode: str) -> dict. Return only the typed ExperimentProgram. "
                        "The trusted wrapper owns the entry point and writes results.json. In smoke "
                        "mode run every condition with tiny samples; in full mode use the frozen sample "
                        "sizes and seeds. Return the ExperimentOutput v2 dictionary with parameters, "
                        "aggregate_metrics, condition-level observations, implementation checks, an "
                        "outcome-neutral conclusion, and limitations. Use no network, subprocess, os, "
                        "async, or multiprocessing."
                        + (
                            " Revise the preserved source using only the exact validation, review, or "
                            "runtime defect. Do not alter the frozen protocol or optimize for result direction."
                            if revision
                            else ""
                        )
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "work_item": item.model_dump(mode="json"),
                            "frozen_protocol": experiment.protocol.model_dump(mode="json"),
                            "protocol_sha256": experiment.protocol_sha256,
                            "previous_program": (
                                experiment.program.model_dump(mode="json")
                                if experiment.program else None
                            ),
                            "defect": experiment.last_error if revision else "",
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
            refs.append(self.store.write_json(f"{run_dir}/program_input.json", {"messages": messages}))
            mock_program = _dry_experiment_program(item, experiment.protocol)
            try:
                program = self.router.complete_structured(
                    task_type="experiment_revision" if revision else "experiment_design",
                    messages=messages,
                    schema=ExperimentProgram,
                    mock_output=mock_program if self.router.dry_run else None,
                    temperature=0.1,
                    allow_repair=True,
                )
                if program.seeds != experiment.protocol.seeds:
                    raise ValueError("program seeds must exactly match the frozen protocol seeds")
                _validate_experiment_program(program)
            except Exception as exc:
                experiment.program = locals().get("program", experiment.program)
                experiment.program_revision += int(revision)
                experiment.stage = "program_revision"
                if experiment.program is not None:
                    refs.append(self.store.write_json(f"{run_dir}/invalid_program.json", experiment.program))
                return engineering_result(
                    "Program failed deterministic validation; preserved source will be repaired.",
                    errors=[f"{type(exc).__name__}: {exc}"],
                )
            experiment.program = program
            experiment.program_revision += int(revision)
            experiment.stage = "program_review"
            experiment.last_error = ""
            refs.append(self.store.write_json(f"{run_dir}/program.json", program))
            return engineering_result("Program passed deterministic validation; alignment review is next.")

        if experiment.stage == "program_review":
            assert experiment.protocol is not None and experiment.program is not None
            if self.router.dry_run:
                program_review = ExperimentProgramReview(
                    accepted=True,
                    objective_alignment="Dry-run program implements the deterministic protocol fixture.",
                )
            else:
                try:
                    program_review = self._review_experiment_program(
                        item, experiment.protocol, experiment.program
                    )
                except Exception as exc:
                    return engineering_result(
                        "Program review failed structurally and will retry unchanged.",
                        errors=[f"{type(exc).__name__}: {exc}"],
                    )
            experiment.program_review = program_review
            refs.append(self.store.write_json(f"{run_dir}/program_review.json", program_review))
            if not program_review.accepted:
                experiment.stage = "program_revision"
                return engineering_result(
                    "Program review found a protocol mismatch; preserved source will be revised.",
                    errors=program_review.issues,
                    next_steps=program_review.issues,
                )
            if self.router.dry_run:
                experiment.stage = "complete"
                persist()
                return WorkResult(
                    work_id=item.work_id,
                    outcome="partial",
                    attempt_class="engineering",
                    summary="Dry run produced a frozen protocol and reviewed program without execution.",
                    artifact_refs=refs,
                )
            experiment.stage = "smoke_execution"
            return engineering_result("Program review accepted; bounded smoke execution is next.")

        agent = ExperimentAgent(self.store, self.router.experimenter)
        if experiment.stage == "smoke_execution":
            assert experiment.protocol is not None and experiment.program is not None
            try:
                execution = agent.run_program(
                    program=experiment.program,
                    name=f"{item.title}_smoke",
                    mode="smoke",
                    timeout_seconds=min(60, experiment.protocol.wall_seconds),
                )
            except Exception as exc:
                return engineering_result(
                    "Smoke infrastructure failed; the same stage will retry.",
                    errors=[f"{type(exc).__name__}: {exc}"],
                )
            experiment.smoke_result = execution
            refs.extend(execution.artifact_refs)
            output = execution.validated_output
            smoke_errors: list[str] = []
            if not execution.success and execution.failure_class == "infrastructure":
                return engineering_result(
                    "Smoke infrastructure failed; the same stage will retry.",
                    errors=[execution.summary],
                )
            if not execution.success or output is None:
                smoke_errors.append(execution.summary)
            elif output.status != "completed":
                smoke_errors.append("Smoke execution reported capped status.")
            elif any(not check.passed for check in output.checks):
                smoke_errors.append(
                    "Smoke implementation checks failed: "
                    + ", ".join(check.name for check in output.checks if not check.passed)
                )
            if smoke_errors:
                experiment.stage = "program_revision"
                return engineering_result(
                    "Smoke execution exposed an implementation defect; source will be revised.",
                    errors=smoke_errors,
                )
            experiment.stage = "full_execution"
            experiment.last_error = ""
            return engineering_result("Smoke execution passed; full frozen execution is next.")

        if experiment.stage == "full_execution":
            assert experiment.protocol is not None and experiment.program is not None
            try:
                execution = agent.run_program(
                    program=experiment.program,
                    name=item.title,
                    mode="full",
                    timeout_seconds=experiment.protocol.wall_seconds,
                )
            except Exception as exc:
                return engineering_result(
                    "Full execution infrastructure failed; the same stage will retry.",
                    errors=[f"{type(exc).__name__}: {exc}"],
                )
            experiment.execution_result = execution
            refs.extend(execution.artifact_refs)
            if not execution.success and execution.failure_class == "infrastructure":
                return engineering_result(
                    "Full execution infrastructure failed; the same stage will retry.",
                    errors=[execution.summary],
                )
            if not execution.success or execution.validated_output is None:
                experiment.stage = "program_revision"
                return engineering_result(
                    "Full execution failed before valid measurements; source will be revised.",
                    errors=[execution.summary],
                )
            experiment.scientific_attempts += 1
            experiment.stage = "evidence_review"
            experiment.last_error = ""
            return engineering_result(
                "Full execution produced valid measurements; scientific evidence review is next."
            )

        if experiment.stage == "evidence_review":
            assert experiment.protocol is not None
            assert experiment.program is not None
            assert experiment.execution_result is not None
            execution = experiment.execution_result
            try:
                evidence_review = self._review_experiment_evidence(
                    item, experiment.protocol, experiment.program, execution
                )
            except Exception as exc:
                return engineering_result(
                    "Evidence review failed structurally and will retry the preserved measurements.",
                    errors=[f"{type(exc).__name__}: {exc}"],
                )
            expected = {
                f"W{index:02d}": criterion
                for index, criterion in enumerate(item.success_criteria, 1)
            }
            missing = _experiment_criterion_errors(expected, evidence_review.criteria)
            if evidence_review.usable == "full" and missing:
                evidence_review.usable = "preliminary"
                evidence_review.issues.extend(missing)
                evidence_review.follow_up.append("Assess every required work criterion by its exact id.")
            refs.append(self.store.write_json(f"{run_dir}/evidence_review.json", evidence_review))
            criteria_rows = _criterion_results(expected, evidence_review.criteria)
            experiment.stage = "complete"
            persist()
            if evidence_review.usable == "unusable":
                return WorkResult(
                    work_id=item.work_id,
                    outcome="partial",
                    failure_class="method",
                    attempt_class="scientific",
                    criteria=criteria_rows,
                    summary="Execution succeeded, but the measurements are scientifically unusable.",
                    artifact_refs=refs,
                    errors=evidence_review.issues,
                    next_steps=evidence_review.follow_up,
                )
            polarity = FindingPolarity(evidence_review.outcome)
            full = evidence_review.usable == "full"
            finding = Finding(
                work_id=item.work_id,
                question_id=item.question_id,
                requirement_id=item.requirement_id,
                kind=WorkKind.experiment,
                statement=evidence_review.scientific_summary,
                status=FindingStatus.observed,
                polarity=polarity,
                strength=EvidenceStrength.strong if full else EvidenceStrength.preliminary,
                scope=(
                    f"Protocol `{experiment.protocol.title}`; sample sizes "
                    f"{experiment.protocol.sample_sizes}; seeds {experiment.protocol.seeds}."
                ),
                evidence_refs=execution.artifact_refs,
                source_ids=[execution.run_id],
                caveats=[*execution.caveats, *evidence_review.caveats],
            )
            return WorkResult(
                work_id=item.work_id,
                outcome="done" if full else "partial",
                failure_class="none" if full else "method",
                attempt_class="scientific",
                evidence_level="substantive" if full else "preliminary",
                requirement_satisfied=full,
                criteria=criteria_rows,
                summary=execution.summary,
                findings=[finding],
                artifact_refs=refs,
                errors=evidence_review.issues if not full else [],
                next_steps=evidence_review.follow_up,
            )

        persist()
        return WorkResult(
            work_id=item.work_id,
            outcome="failed",
            failure_class="engineering",
            attempt_class="engineering",
            summary=f"Unknown or terminal experiment stage: {experiment.stage}",
            artifact_refs=refs,
            errors=[experiment.last_error or "invalid experiment state"],
        )

    def _review_experiment_program(
        self,
        item: WorkItem,
        protocol: ExperimentProtocol,
        program: ExperimentProgram,
    ) -> ExperimentProgramReview:
        messages = [
            {
                "role": "system",
                "content": (
                    "Compare executable code line by line to the frozen protocol. Reject omitted or "
                    "renamed conditions, proxy metrics, fake encodings, absent cost terms, biased "
                    "hypothesis assertions, broken round trips, ignored seeds/sample sizes, invalid "
                    "statistics, and workloads likely to exceed the cap. Do not redesign the protocol."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "work_item": item.model_dump(mode="json"),
                        "protocol": protocol.model_dump(mode="json"),
                        "program": program.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return self.router.complete_structured(
            task_type="experiment_review",
            messages=messages,
            schema=ExperimentProgramReview,
            allow_repair=False,
        )

    def _review_experiment_evidence(
        self,
        item: WorkItem,
        protocol: ExperimentProtocol,
        program: ExperimentProgram,
        execution: Any,
    ) -> ExperimentEvidenceReview:
        output = execution.validated_output
        messages = [
            {
                "role": "system",
                "content": (
                    "Audit completed measurements against the frozen protocol and work criteria. "
                    "Return exactly one assessment for every supplied criterion_id, without renaming "
                    "or adding ids. Recompute the conclusion from condition-level observations where possible. "
                    "Classify supports, contradicts, null, inconclusive, or characterizes without "
                    "favoring the expected direction. `full` requires every mandatory criterion and "
                    "applicable agenda constraint; `preliminary` is allowed for an interpretable pilot "
                    "that has explicit missing scope; `unusable` is for wrong metrics, invalid baselines, "
                    "genuine failed implementation correctness, leakage, or unsupported conclusions. "
                    "If a false check improperly tests the hypothesis direction, reject that check as a "
                    "methodology defect but still assess whether the preserved raw observations support "
                    "a scoped preliminary result. Never discard a sound negative or null result."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "work_item": item.model_dump(mode="json"),
                        "work_criteria": [
                            {"criterion_id": f"W{index:02d}", "text": criterion}
                            for index, criterion in enumerate(item.success_criteria, 1)
                        ],
                        "agenda_constraints": self.store.load_agenda().constraints,  # type: ignore[union-attr]
                        "protocol": protocol.model_dump(mode="json"),
                        "program": program.model_dump(mode="json"),
                        "validated_output": output.model_dump(mode="json") if output else None,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return self.router.complete_structured(
            task_type="experiment_review",
            messages=messages,
            schema=ExperimentEvidenceReview,
            allow_repair=False,
        )

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
                    and item.strategy_fingerprint in requirement.attempted_strategy_fingerprints
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


def _passage_support_id(
    citation_key: str,
    char_start: int | None,
    char_end: int | None,
    quote: str,
) -> str:
    identity = f"{citation_key}\0{char_start}\0{char_end}\0{quote}"
    return "passage_support_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


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


def _memory_to_mb(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?)b?\s*", value.lower())
    if not match:
        raise ValueError(f"unsupported experimenter memory value: {value}")
    amount = float(match.group(1))
    unit = match.group(2)
    factors = {"": 1 / (1024 * 1024), "k": 1 / 1024, "m": 1, "g": 1024, "t": 1024 * 1024}
    return max(1, int(amount * factors[unit]))


def _experiment_resource_errors(
    protocol: ExperimentProtocol,
    *,
    max_wall: int,
    max_memory: int,
    max_cpus: float,
) -> list[str]:
    errors: list[str] = []
    if protocol.wall_seconds > max_wall:
        errors.append(
            f"Protocol wall_seconds={protocol.wall_seconds} exceeds runtime limit {max_wall}."
        )
    if protocol.memory_mb > max_memory:
        errors.append(
            f"Protocol memory_mb={protocol.memory_mb} exceeds runtime limit {max_memory}."
        )
    if protocol.cpus > max_cpus:
        errors.append(f"Protocol cpus={protocol.cpus} exceeds runtime limit {max_cpus}.")
    return errors


def _experiment_criterion_errors(
    expected: dict[str, str], assessments: list[ExperimentCriterionAssessment]
) -> list[str]:
    ids = [assessment.criterion_id for assessment in assessments]
    errors: list[str] = []
    duplicates = sorted({criterion_id for criterion_id in ids if ids.count(criterion_id) > 1})
    missing = sorted(set(expected) - set(ids))
    unexpected = sorted(set(ids) - set(expected))
    if duplicates:
        errors.append("Duplicate criterion assessment ids: " + ", ".join(duplicates))
    if missing:
        errors.append("Missing criterion assessment ids: " + ", ".join(missing))
    if unexpected:
        errors.append("Unexpected criterion assessment ids: " + ", ".join(unexpected))
    return errors


def _criterion_results(
    expected: dict[str, str], assessments: list[ExperimentCriterionAssessment]
) -> list[CriterionResult]:
    by_id = {assessment.criterion_id: assessment for assessment in assessments}
    return [
        CriterionResult(
            criterion=text,
            satisfied=criterion_id in by_id and by_id[criterion_id].satisfied,
            detail=(
                by_id[criterion_id].detail
                if criterion_id in by_id
                else f"Reviewer omitted required criterion id {criterion_id}."
            ),
        )
        for criterion_id, text in expected.items()
    ]


def _methods_for_requirement(
    description: str, preferred: list[WorkKind] | tuple[WorkKind, ...]
) -> list[WorkKind]:
    lowered = description.lower()
    # Empirical gaps require observations. A failed experimental pipeline must remain visible and
    # may not be hidden by substituting a derivation that cannot produce the requested data.
    if re.search(
        r"\b(?:empirical|experiment(?:al)?|benchmark(?:ing)?|measurement|measured|"
        r"compression ratio|p-value|dataset|plot)\b",
        lowered,
    ):
        return [WorkKind.experiment]
    narrowed: list[WorkKind] = []
    if re.search(r"\b(?:literature|source|known result|published|citation)\b", lowered):
        narrowed.append(WorkKind.literature)
    if re.search(r"\b(?:proof|derive|bound|complexity|entropy|criterion|theorem)\b", lowered):
        narrowed.append(WorkKind.derivation)
        if WorkKind.proof in preferred and re.search(r"\b(?:finite|combinatorial|lean|formal)\b", lowered):
            narrowed.append(WorkKind.proof)
    narrowed.extend(preferred)
    unique: list[WorkKind] = []
    for method in narrowed:
        if method != WorkKind.synthesis and method not in unique:
            unique.append(method)
    return unique[:4]


def _missing_criterion_reviews(
    expected: list[str], reviews: list[CriterionResult]
) -> list[str]:
    """Match reviewer rows to mandatory criteria by content, not merely by list length."""
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
        "it", "of", "on", "or", "that", "the", "this", "to", "was", "with",
    }

    def terms(value: str) -> set[str]:
        return {
            token for token in re.findall(r"[a-z0-9]+", value.lower())
            if len(token) > 2 and token not in stop
        }

    reviewed = [(terms(row.criterion), row) for row in reviews]
    missing: list[str] = []
    for criterion in expected:
        wanted = terms(criterion)
        matches = [
            row
            for found, row in reviewed
            if wanted
            and found
            and len(wanted & found) / max(1, min(len(wanted), len(found))) >= 0.45
        ]
        if not matches or not any(row.satisfied for row in matches):
            missing.append(criterion)
    return missing


def _obviously_trivial_goal(statement: str) -> bool:
    normalized = re.sub(r"\s+", " ", statement.strip())
    if normalized in {"True", "∀ _ : Unit, True"}:
        return True
    body = normalized.rsplit(",", 1)[-1].strip().strip("()")
    match = re.fullmatch(r"(.+?)\s*=\s*(.+)", body)
    return bool(match and match.group(1).strip("() ") == match.group(2).strip("() "))


def _dry_experiment_program(item: WorkItem, protocol: ExperimentProtocol) -> ExperimentProgram:
    condition_ids = [condition.id for condition in protocol.conditions[:2]]
    metric_id = protocol.metrics[0].id
    output = ExperimentOutput(
        experiment="dry-run protocol implementation",
        parameters={"seed": protocol.seeds[0], "dry_run": True},
        aggregate_metrics={"difference": 0.0},
        observations=[
            ExperimentObservation(
                condition=condition_id,
                sample_size=1,
                metrics={metric_id: 0.0},
            )
            for condition_id in condition_ids
        ],
        checks=[{"name": "dry run", "passed": True, "detail": "not executed"}],
        conclusion=ExperimentConclusion(
            hypothesis=item.hypothesis,
            outcome="inconclusive",
            basis_metrics=["difference"],
            statement="Dry-run code generation provides no scientific observation.",
        ),
        limitations=["Dry run only."],
    )
    payload = json.dumps(output.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    source = (
        "import json\n\n"
        "def run_experiment(mode: str) -> dict:\n"
        f"    return json.loads({json.dumps(payload)})\n"
    )
    return ExperimentProgram(
        description=f"Dry-run implementation of {protocol.title}",
        source=source,
        seeds=protocol.seeds,
    )
