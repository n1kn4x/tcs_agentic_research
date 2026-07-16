"""Deterministic, bounded orchestration for incremental research progress.

Models propose small typed payloads.  Python chooses and executes every action.  Each work item
starts with fresh context, has a hard model-call budget, and writes a self-contained run record.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from .agents.experiment import ExperimentAgent
from .agents.literature import LiteratureResearcher
from .agents.theorem_prover import TheoremProverAgent
from .artifact_store import ArtifactStore, sha256_file
from .llm import LLMRouter
from .schemas import (
    AnalysisSubmission,
    ArtifactRef,
    ExperimentProgram,
    Finding,
    FindingStatus,
    LeanGoalDraft,
    LeanStatement,
    LiteraturePlan,
    PlanSubmission,
    ResearchPhase,
    WorkItem,
    WorkItemDraft,
    WorkKind,
    WorkQueue,
    WorkResult,
    WorkStatus,
    WorkspaceState,
    utc_now,
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
        self.router = LLMRouter.from_config_file(
            config_path,
            store=self.store,
            dry_run=dry_run,
        )
        self.prompt_dir = prompt_dir

    def initialize(self) -> WorkspaceState:
        if not self.store.exists(ArtifactStore.RESEARCH_TASK):
            raise RuntimeError(
                f"Missing `{ArtifactStore.RESEARCH_TASK}` in {self.store.root}"
            )
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
                    "Initialized without inferred notation or claims. Every work result remains "
                    "typed by its actual evidence source."
                ],
            )
            self.store.save_state(state)
            self.store.append_event(
                "workspace_initialized",
                {"task_id": state.task_id, "task_sha256": digest},
            )
            return state
        if state.task_sha256 != digest:
            old_digest = state.task_sha256
            state.task_sha256 = digest
            state.task_summary = _task_summary(task)
            state.phase = ResearchPhase.planning
            self.store.save_state(state)
            self.store.append_event(
                "task_changed",
                {"old_sha256": old_digest, "new_sha256": digest},
            )
        return state

    def run(self, *, max_steps: int = 1) -> dict[str, Any]:
        with self.store.exclusive_lock():
            state = self.initialize()
            executed = 0
            while executed < max(0, max_steps):
                queue = self.store.load_queue()
                item = _next_open(queue)
                if item is None:
                    if state.phase == ResearchPhase.review:
                        break
                    if state.plan_round >= self.router.core.max_plan_rounds:
                        state.phase = ResearchPhase.review
                        state.notes.append(
                            "Reached the configured plan-round budget; human review is required."
                        )
                        self.store.save_state(state)
                        break
                    created = self._plan(state, queue)
                    state = self._require_state()
                    if not created:
                        break
                    queue = self.store.load_queue()
                    item = _next_open(queue)
                    if item is None:
                        break
                self._execute(state, queue, item)
                state = self._require_state()
                executed += 1
            return self.status()

    def replan(self) -> None:
        with self.store.exclusive_lock():
            state = self.initialize()
            previous_round = state.plan_round
            state.plan_round = 0
            state.phase = ResearchPhase.planning
            state.notes.append("Human explicitly reset the planning-round budget.")
            self.store.save_state(state)
            self.store.append_event(
                "human_requested_replan",
                {"cycle": state.cycle, "previous_plan_round": previous_round},
            )

    def status(self) -> dict[str, Any]:
        state = self.store.load_state()
        queue = self.store.load_queue()
        findings = self.store.read_findings()
        counts = {status.value: 0 for status in WorkStatus}
        for item in queue.items:
            counts[item.status.value] += 1
        next_item = _next_open(queue)
        return {
            "workspace": str(self.store.root),
            "state": state.model_dump(mode="json") if state else None,
            "work_counts": counts,
            "next_open_work": next_item.model_dump(mode="json") if next_item else None,
            "finding_counts": {
                status.value: sum(1 for finding in findings if finding.status == status)
                for status in FindingStatus
            },
            "recent_findings": [
                finding.model_dump(mode="json") for finding in findings[-8:]
            ],
        }

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def _plan(self, state: WorkspaceState, queue: WorkQueue) -> bool:
        task = self.store.read_text(ArtifactStore.RESEARCH_TASK)
        existing = [
            {
                "kind": item.kind.value,
                "title": item.title,
                "instruction": item.instruction[:300],
                "status": item.status.value,
                "blocked_reason": item.blocked_reason[-300:],
            }
            for item in queue.items[-10:]
        ]
        findings = [
            {
                "finding_id": finding.finding_id,
                "kind": finding.kind.value,
                "status": finding.status.value,
                "statement": finding.statement[:500],
                "source_ids": finding.source_ids[:6],
            }
            for finding in self.store.read_findings()[-10:]
        ]
        recent_results = _recent_result_context(self.store, limit=6)
        mock = _default_plan(task, existing=existing)
        messages = [
            {
                "role": "system",
                "content": (
                    "Plan the next bounded research work. You have no tools. Return at most four "
                    "independent work items through the enforced response format. Each item must "
                    "be executable in one fresh step and must name observable success criteria. "
                    "Use literature for source discovery/quotes, proof for one small Lean theorem, "
                    "experiment for one bounded reproducible Python program, and analysis only "
                    "for synthesis or an informal derivation. Never claim the task is solved; "
                    "choose review when no distinct useful work remains."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": task[:8000],
                        "plan_round": state.plan_round + 1,
                        "existing_work": existing,
                        "evidence_findings": findings,
                        "recent_work_results": recent_results,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        plan_cycle = state.cycle + 1
        run_dir = self.store.create_run_dir(plan_cycle, f"plan_round_{state.plan_round + 1}")
        self.store.write_json(f"{run_dir}/input.json", {"messages": messages})
        planning_error = ""
        try:
            with self.router.step_budget(f"plan_{state.plan_round + 1}", max_calls=2):
                plan = self.router.complete_structured(
                    task_type="planning",
                    messages=messages,
                    schema=PlanSubmission,
                    mock_output=mock if self.router.dry_run else None,
                )
        except Exception as exc:  # noqa: BLE001 - deterministic scheduling remains safe
            # Planning is control flow, not scientific evidence. A conservative deterministic
            # schedule is therefore a safe fallback and keeps infrastructure failures distinct
            # from research findings.
            planning_error = f"{type(exc).__name__}: {exc}"
            plan = mock
            plan.reason = (
                "Deterministic scheduler fallback after model planning failure. "
                "No scientific claim was inferred."
            )
            self.store.write_text(f"{run_dir}/planning_error.log", planning_error + "\n")
        plan_ref = self.store.write_json(f"{run_dir}/plan.json", plan)

        # Explicit subsystem requirements in the user task are deterministic contracts, not a
        # suggestion the planner may accidentally omit. Prefer the model's item for each required
        # kind, then fill missing kinds with conservative defaults.
        required_drafts = _default_plan(task, existing=existing).work_items
        model_by_kind = {draft.kind: draft for draft in plan.work_items}
        ordered_drafts: list[WorkItemDraft] = []
        for required in required_drafts:
            ordered_drafts.append(model_by_kind.pop(required.kind, required))
        ordered_drafts.extend(model_by_kind.values())

        existing_keys = {_work_key(item.kind, item.instruction) for item in queue.items}
        new_items: list[WorkItem] = []
        for draft in ordered_drafts[: self.router.core.max_plan_items]:
            key = _work_key(draft.kind, draft.instruction)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            new_items.append(
                WorkItem(
                    kind=draft.kind,
                    title=draft.title,
                    instruction=draft.instruction,
                    success_criteria=draft.success_criteria,
                )
            )
        state.plan_round += 1
        effective_review = plan.decision == "review" and not required_drafts
        if effective_review or not new_items:
            state.phase = ResearchPhase.review
            state.notes.append(plan.reason or "Planner found no distinct bounded work.")
        else:
            queue.items.extend(new_items)
            state.phase = ResearchPhase.working
        self.store.save_queue(queue)
        self.store.save_state(state)
        self.store.append_event(
            "plan_recorded",
            {
                "plan_round": state.plan_round,
                "decision": plan.decision,
                "new_work_ids": [item.work_id for item in new_items],
                "artifact": plan_ref.path,
                "planning_error": planning_error,
            },
        )
        return bool(new_items)

    # ------------------------------------------------------------------
    # One atomic work execution
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
                "model_call_budget": self.router.core.max_model_calls_per_step,
            },
        )
        self.store.save_queue(queue)
        self.store.save_state(state)
        self.store.append_event(
            "work_started",
            {"cycle": cycle, "work_id": item.work_id, "kind": item.kind.value},
        )

        try:
            with self.router.step_budget(
                item.work_id,
                max_calls=self.router.core.max_model_calls_per_step,
            ):
                result = self._dispatch(item, run_dir)
        except Exception as exc:  # noqa: BLE001 - the durable result is the failure boundary
            result = WorkResult(
                work_id=item.work_id,
                outcome="failed",
                summary=f"Work step failed: {type(exc).__name__}",
                artifact_refs=[input_ref],
                errors=[f"{type(exc).__name__}: {exc}"],
                next_steps=["Inspect this run's input/result and create a narrower work item."],
            )

        if input_ref.path not in {ref.path for ref in result.artifact_refs}:
            result.artifact_refs.append(input_ref)
        result_ref = self.store.write_json(f"{run_dir}/result.json", result)
        self.store.append_findings(result.findings)
        item.last_result_id = result.result_id
        item.blocked_reason = "; ".join(result.errors) if result.outcome in {"blocked", "failed"} else ""
        item.status = WorkStatus(result.outcome)
        item.updated_at = utc_now()
        state.cycle = cycle
        state.active_work_id = None
        state.last_result_id = result.result_id
        if _next_open(queue) is None:
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
                "artifact": result_ref.path,
                "finding_ids": [finding.finding_id for finding in result.findings],
            },
        )

    def _dispatch(self, item: WorkItem, run_dir: str) -> WorkResult:
        if item.kind == WorkKind.literature:
            return self._run_literature(item, run_dir)
        if item.kind == WorkKind.proof:
            return self._run_proof(item, run_dir)
        if item.kind == WorkKind.experiment:
            return self._run_experiment(item, run_dir)
        return self._run_analysis(item, run_dir)

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------
    def _run_literature(self, item: WorkItem, run_dir: str) -> WorkResult:
        task = self.store.read_text(ArtifactStore.RESEARCH_TASK)
        fallback_query = _task_search_query(task)
        mock = LiteraturePlan(
            search_queries=[fallback_query],
            focus_questions=[fallback_query],
        )
        literature_messages = [
            {
                "role": "system",
                "content": (
                    "Create a small primary-literature search plan without inventing DOI/arXiv "
                    "identifiers. Return one to three search queries, up to three known source "
                    "titles if you are confident in them, and one to four precise questions. "
                    "The application, not you, will search and import actual candidates."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"task": task[:12000], "work_item": item.model_dump(mode="json")},
                    ensure_ascii=False,
                ),
            },
        ]
        model_input_ref = self.store.write_json(
            f"{run_dir}/literature_model_input.json",
            {"schema": "LiteraturePlan", "messages": literature_messages},
        )
        errors: list[str] = []
        try:
            plan = self.router.complete_structured(
                task_type="literature_planning",
                messages=literature_messages,
                schema=LiteraturePlan,
                mock_output=mock if self.router.dry_run else None,
            )
        except Exception as exc:  # noqa: BLE001 - a literal search query is safe control fallback
            error = f"literature planning: {type(exc).__name__}: {exc}"
            errors.append(error)
            self.store.write_text(f"{run_dir}/literature_planning_error.log", error + "\n")
            plan = mock
        refs: list[ArtifactRef] = [
            model_input_ref,
            self.store.write_json(f"{run_dir}/literature_plan.json", plan),
        ]
        literature = LiteratureResearcher(self.store, self.router, prompt_dir=self.prompt_dir)
        candidates = []
        if not self.router.dry_run:
            discovery_queries = list(dict.fromkeys([
                *plan.known_source_titles,
                *plan.search_queries,
            ]))
            for query in discovery_queries:
                try:
                    candidates.extend(literature.search_papers(query, limit=5))
                except Exception as exc:  # noqa: BLE001 - each search is independently useful
                    errors.append(f"search {query!r}: {type(exc).__name__}: {exc}")

        selected = _rank_candidates(
            candidates,
            preferred_titles=plan.known_source_titles,
            relevance_queries=plan.search_queries,
        )[: self.router.core.literature_max_imports]
        imported = []
        for candidate in selected:
            if candidate.status != "queued":
                continue
            try:
                imported.append(literature.import_candidate(candidate.candidate_id, extract_text=True))
            except Exception as exc:  # noqa: BLE001 - record source-level gap and continue
                errors.append(f"import {candidate.title!r}: {type(exc).__name__}: {exc}")

        try:
            extraction = literature.extract_imported_papers(max_papers=10, only_missing=True)
        except Exception as exc:  # noqa: BLE001
            extraction = {"processed_count": 0, "errors": [str(exc)]}
            errors.append(f"extraction: {type(exc).__name__}: {exc}")
        refs.append(self.store.write_json(f"{run_dir}/extraction.json", extraction))

        answers = []
        for question in plan.focus_questions:
            try:
                answers.append(
                    literature.answer_query(
                        question,
                        limit=self.router.core.literature_results_per_query,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"query {question!r}: {type(exc).__name__}: {exc}")
        refs.append(
            self.store.write_json(
                f"{run_dir}/query_answers.json",
                [answer.model_dump(mode="json") for answer in answers],
            )
        )

        findings: list[Finding] = []
        seen_supports: set[str] = set()
        for answer in answers:
            for result in answer.results:
                key = result.support_id or result.statement_id or result.result_id
                if key in seen_supports:
                    continue
                seen_supports.add(key)
                validated = bool(
                    result.support_id
                    and result.support_level == "primary_exact"
                    and result.provenance
                    and result.provenance[0].validated
                )
                findings.append(
                    Finding(
                        work_id=item.work_id,
                        kind=WorkKind.literature,
                        statement=f"[{result.citation_key}] {result.statement_text}",
                        status=(
                            FindingStatus.supported if validated else FindingStatus.hypothesis
                        ),
                        evidence_refs=list(result.provenance[0].artifact_refs)
                        if result.provenance
                        else [],
                        source_ids=[
                            value
                            for value in [result.support_id, result.statement_id, result.quote_id]
                            if value
                        ],
                        caveats=(
                            []
                            if validated
                            else [
                                "The quote span is unvalidated or the deterministic statement-span "
                                "quality gate did not accept it as a complete formal statement."
                            ]
                        ),
                    )
                )
        report_ref = self.store.write_text(
            f"{run_dir}/literature_report.md",
            _render_literature_report(item, plan, candidates, imported, findings, errors),
        )
        refs.append(report_ref)
        supported_count = sum(finding.status == FindingStatus.supported for finding in findings)
        if supported_count:
            outcome = "done"
            summary = f"Recorded {supported_count} quote-validated literature finding(s)."
        elif candidates or imported or findings:
            outcome = "partial"
            summary = "Literature scouting made progress but produced no quote-validated finding yet."
        else:
            outcome = "blocked"
            summary = "No local or external literature evidence was available in this bounded step."
        return WorkResult(
            work_id=item.work_id,
            outcome=outcome,
            summary=summary,
            findings=findings[:20],
            artifact_refs=_existing_refs(self.store, refs, [
                "LiteratureDB/candidates.jsonl",
                "LiteratureDB/papers.jsonl",
                "LiteratureDB/statements.jsonl",
                "LiteratureDB/index.sqlite",
            ]),
            errors=errors,
            next_steps=[
                "Manually provide a primary-source PDF/DOI/arXiv ID for unresolved gaps."
            ] if not supported_count else [],
        )

    def _run_analysis(self, item: WorkItem, run_dir: str) -> WorkResult:
        task = self.store.read_text(ArtifactStore.RESEARCH_TASK)
        prior = self.store.read_findings()[-12:]
        compact = [
            {
                "finding_id": finding.finding_id,
                "kind": finding.kind.value,
                "status": finding.status.value,
                "statement": finding.statement[:600],
                "source_ids": finding.source_ids[:8],
                "caveats": finding.caveats[:4],
            }
            for finding in prior
        ]
        mock = AnalysisSubmission(
            summary="Dry-run analysis recorded the available evidence without claiming a result.",
            unresolved_questions=[item.instruction],
        )
        analysis_messages = [
            {
                "role": "system",
                "content": (
                    "Perform one bounded TCS analysis. Use only the supplied findings as "
                    "evidence. Return candidate claims with the finding IDs they rely on. "
                    "Do not call tools, invent citations, claim novelty, or claim that the main "
                    "task is solved. Candidate analysis remains a hypothesis until separately "
                    "verified by literature, Lean, or experiment."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": task[:10000],
                        "work_item": item.model_dump(mode="json"),
                        "findings": compact,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        model_input_ref = self.store.write_json(
            f"{run_dir}/analysis_model_input.json",
            {"schema": "AnalysisSubmission", "messages": analysis_messages},
        )
        submission = self.router.complete_structured(
            task_type="analysis",
            messages=analysis_messages,
            schema=AnalysisSubmission,
            mock_output=mock if self.router.dry_run else None,
        )
        known_ids = {finding.finding_id for finding in prior}
        findings = [
            Finding(
                work_id=item.work_id,
                kind=WorkKind.analysis,
                statement=claim.statement,
                status=FindingStatus.hypothesis,
                source_ids=[value for value in claim.basis_finding_ids if value in known_ids],
                caveats=[
                    claim.caveat or "Model synthesis is a hypothesis, not certifying evidence."
                ],
            )
            for claim in submission.candidate_claims
        ]
        report_ref = self.store.write_text(
            f"{run_dir}/analysis.md",
            _render_analysis(item, submission),
        )
        return WorkResult(
            work_id=item.work_id,
            outcome="done",
            summary=submission.summary,
            findings=findings,
            artifact_refs=[model_input_ref, report_ref],
            next_steps=submission.suggested_next_steps,
            errors=[],
        )

    def _run_proof(self, item: WorkItem, run_dir: str) -> WorkResult:
        task = self.store.read_text(ArtifactStore.RESEARCH_TASK)
        mock = LeanGoalDraft(
            name="dry_run_goal",
            statement="∀ n : Nat, n = n",
        )
        formulation_messages = [
            {
                "role": "system",
                "content": (
                    "Translate the bounded proof work into one small Lean goal. Return only "
                    "the goal through response_format. Do not strengthen the mathematical claim."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"task": task[:8000], "work_item": item.model_dump(mode="json")},
                    ensure_ascii=False,
                ),
            },
        ]
        formulation_input_ref = self.store.write_json(
            f"{run_dir}/proof_formulation_model_input.json",
            {"schema": "LeanGoalDraft", "messages": formulation_messages},
        )
        goal_draft = self.router.complete_structured(
            task_type="proof_formulation",
            messages=formulation_messages,
            schema=LeanGoalDraft,
            mock_output=mock if self.router.dry_run else None,
        )
        goal_ref = self.store.write_json(f"{run_dir}/lean_goal.json", goal_draft)
        goal = LeanStatement(**goal_draft.model_dump())
        result = TheoremProverAgent(
            self.store,
            self.router,
            prompt_dir=self.prompt_dir,
        ).prove(
            goal,
            context=item.instruction,
            max_iterations=1,
            max_revisions=self.router.core.proof_revisions,
        )
        result_ref = self.store.write_json(f"{run_dir}/lean_result.json", result)
        if result.status == "proved":
            finding = Finding(
                work_id=item.work_id,
                kind=WorkKind.proof,
                statement=f"Lean verified `{goal.name} : {goal.statement}`.",
                status=FindingStatus.verified,
                evidence_refs=result.proved_artifacts,
                source_ids=[result.result_id],
            )
            outcome = "done"
            findings = [finding]
        else:
            outcome = "blocked" if result.status == "needs_human_formalization" else "partial"
            findings = []
        return WorkResult(
            work_id=item.work_id,
            outcome=outcome,
            summary=f"Bounded Lean attempt ended with status `{result.status}`.",
            findings=findings,
            artifact_refs=[formulation_input_ref, goal_ref, result_ref, *result.artifact_refs],
            errors=[] if result.status == "proved" else [result.proof_dag_summary],
            next_steps=result.recommended_next_steps,
        )

    def _run_experiment(self, item: WorkItem, run_dir: str) -> WorkResult:
        task = self.store.read_text(ArtifactStore.RESEARCH_TASK)
        mock = ExperimentProgram(
            description=item.instruction,
            python_code="print('dry-run: experiment was not executed')",
            seed=0,
        )
        experiment_messages = [
            {
                "role": "system",
                "content": (
                    "Design one bounded reproducible Python experiment. Return a single "
                    "self-contained program through response_format. It must use fixed seeds, "
                    "write outputs only in the current directory, avoid network access and "
                    "subprocesses, finish quickly, and print a concise factual summary. "
                    "Experiments do not prove mathematical claims."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"task": task[:8000], "work_item": item.model_dump(mode="json")},
                    ensure_ascii=False,
                ),
            },
        ]
        model_input_ref = self.store.write_json(
            f"{run_dir}/experiment_model_input.json",
            {"schema": "ExperimentProgram", "messages": experiment_messages},
        )
        program = self.router.complete_structured(
            task_type="experiment_design",
            messages=experiment_messages,
            schema=ExperimentProgram,
            mock_output=mock if self.router.dry_run else None,
        )
        _validate_experiment_program(program.python_code)
        program_ref = self.store.write_json(f"{run_dir}/program.json", program)
        if self.router.dry_run:
            return WorkResult(
                work_id=item.work_id,
                outcome="partial",
                summary="Dry-run generated an experiment program but did not execute Docker.",
                artifact_refs=[model_input_ref, program_ref],
                next_steps=["Run again without --dry-run and with experimenter configuration."],
            )
        result = ExperimentAgent(self.store, self.router.experimenter).run_program(
            program=program,
            name=item.title,
        )
        finding = Finding(
            work_id=item.work_id,
            kind=WorkKind.experiment,
            statement=result.summary,
            status=FindingStatus.observed,
            evidence_refs=result.artifact_refs,
            source_ids=[result.run_id],
            caveats=result.caveats,
        )
        return WorkResult(
            work_id=item.work_id,
            outcome="done" if result.success else "failed",
            summary=result.summary,
            findings=[finding] if result.success else [],
            artifact_refs=[model_input_ref, program_ref, *result.artifact_refs],
            errors=[] if result.success else [result.summary],
        )

    def _require_state(self) -> WorkspaceState:
        state = self.store.load_state()
        if state is None:
            raise RuntimeError("State.json is missing")
        return state


# ---------------------------------------------------------------------------
# Deterministic helpers and renderers
# ---------------------------------------------------------------------------


def _default_plan(task: str, *, existing: list[dict[str, Any]]) -> PlanSubmission:
    lowered = task.lower()
    existing_kinds = {str(item.get("kind")) for item in existing}
    drafts: list[WorkItemDraft] = []
    literature_requested = bool(
        re.search(
            r"\b(?:literature|literaturedb|citation|citations|source|sources|paper|papers|provenance)\b",
            lowered,
        )
    )
    experiment_requested = bool(
        re.search(r"\b(?:experimenter|experiment|experiments|benchmark|empirical)\b", lowered)
    )
    proof_requested = bool(
        re.search(r"\b(?:lean|leap|formalize|formalization)\b|\bformal proof\b", lowered)
    )
    if literature_requested and "literature" not in existing_kinds:
        drafts.append(
            WorkItemDraft(
                kind=WorkKind.literature,
                title="Bounded primary-source literature audit",
                instruction=(
                    "Search for primary sources relevant to the research question, import at most "
                    "a few candidates, extract exact theorem/definition/lower-bound quotes, and "
                    "record unsupported gaps explicitly."
                ),
                success_criteria=["At least one exact quote with a stable support ID, or a precise source gap."],
            )
        )
    if experiment_requested and "experiment" not in existing_kinds:
        drafts.append(
            WorkItemDraft(
                kind=WorkKind.experiment,
                title="One reproducible bounded experiment",
                instruction=(
                    "Generate and run one small Python experiment addressing the requested empirical "
                    "question, with fixed seeds, machine-readable outputs, and explicit limitations."
                ),
                success_criteria=["Program exits successfully and preserves code, seed, logs, and outputs."],
            )
        )
    if proof_requested and "proof" not in existing_kinds:
        drafts.append(
            WorkItemDraft(
                kind=WorkKind.proof,
                title="One small Lean verification target",
                instruction=(
                    "Choose one elementary supporting lemma from the task, formulate one precise Lean "
                    "statement, and attempt a compiler-checked proof without sorry or new axioms."
                ),
                success_criteria=["Lean accepts a placeholder-free proof, or compiler errors are preserved."],
            )
        )
    if "analysis" not in existing_kinds:
        drafts.append(
            WorkItemDraft(
                kind=WorkKind.analysis,
                title="Evidence-bounded synthesis and gap analysis",
                instruction=(
                    "Synthesize only the evidence currently recorded, distinguish verified facts from "
                    "hypotheses, and identify the smallest unresolved technical question."
                ),
                success_criteria=["Every candidate conclusion names its evidence basis and caveat."],
            )
        )
    return PlanSubmission(
        decision="continue" if drafts else "review",
        objective="Make auditable progress through small evidence-producing work items.",
        work_items=drafts[:4],
        reason="Deterministic dry-run plan based on subsystem requirements in the task.",
    )


def _recent_result_context(store: ArtifactStore, *, limit: int) -> list[dict[str, Any]]:
    paths = sorted(store.resolve("Runs").glob("*/result.json"))[-limit:]
    results: list[dict[str, Any]] = []
    for path in paths:
        try:
            row = store.read_json(path)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        results.append(
            {
                "work_id": str(row.get("work_id") or ""),
                "outcome": str(row.get("outcome") or ""),
                "summary": str(row.get("summary") or "")[:500],
                "errors": [str(value)[:300] for value in (row.get("errors") or [])[:2]],
                "next_steps": [
                    str(value)[:300] for value in (row.get("next_steps") or [])[:3]
                ],
                "result_artifact": store.relpath(path),
            }
        )
    return results


def _next_open(queue: WorkQueue) -> WorkItem | None:
    return next((item for item in queue.items if item.status == WorkStatus.open), None)


def _work_key(kind: WorkKind, instruction: str) -> tuple[str, str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", instruction.lower()).strip()
    return kind.value, normalized


def _task_summary(markdown: str, *, limit: int = 600) -> str:
    lines = [line.strip("# ").strip() for line in markdown.splitlines() if line.strip()]
    return " ".join(lines)[:limit]


def _compact_query(text: str) -> str:
    stop = {
        "a", "an", "and", "around", "audit", "based", "for", "from", "identify",
        "in", "is", "literature", "of", "on", "precise", "problem", "research",
        "that", "the", "this", "to", "what", "which", "with",
    }
    selected: list[str] = []
    seen: set[str] = set()
    for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9+]*", text):
        key = word.lower()
        if key in stop or key in seen or len(key) < 3:
            continue
        selected.append(word)
        seen.add(key)
        if len(selected) >= 5:
            break
    return " ".join(selected) or "theoretical computer science primary source"


def _task_search_query(task: str) -> str:
    match = re.search(
        r"(?ims)^##?\s*Research question\s*$\s*(.+?)(?=^##?\s|\Z)",
        task,
    )
    return _compact_query(match.group(1) if match else task)


def _rank_candidates(
    candidates: list[Any],
    *,
    preferred_titles: list[str] | None = None,
    relevance_queries: list[str] | None = None,
) -> list[Any]:
    unique: dict[str, Any] = {}
    for candidate in candidates:
        title_key = re.sub(r"[^a-z0-9]+", " ", candidate.title.lower()).strip()
        key = title_key or candidate.doi.lower() or candidate.arxiv_id.lower() or candidate.openalex_id.lower()
        current = unique.get(key)
        candidate_rank = (
            bool(candidate.arxiv_id or candidate.pdf_url),
            candidate.score,
            candidate.cited_by_count,
        )
        current_rank = (
            bool(current and (current.arxiv_id or current.pdf_url)),
            current.score if current else -1.0,
            current.cited_by_count if current else -1,
        )
        if current is None or candidate_rank > current_rank:
            unique[key] = candidate
    preferred_term_sets = [
        set(re.findall(r"[a-z0-9]{3,}", title.lower()))
        for title in (preferred_titles or [])
    ]
    relevance_term_sets = [
        set(re.findall(r"[a-z0-9]{3,}", query.lower()))
        for query in (relevance_queries or [])
    ]

    def title_match(candidate: Any) -> float:
        terms = set(re.findall(r"[a-z0-9]{3,}", candidate.title.lower()))
        scores = [
            len(terms & preferred) / max(1, len(terms | preferred))
            for preferred in preferred_term_sets
        ]
        return max(scores, default=0.0)

    def query_match(candidate: Any) -> float:
        terms = set(re.findall(r"[a-z0-9]{3,}", candidate.title.lower()))
        scores = [
            len(terms & query_terms) / max(1, len(query_terms))
            for query_terms in relevance_term_sets
        ]
        return max(scores, default=0.0)

    def rank_key(item: Any) -> tuple[bool, bool, float, float, float, int]:
        match = title_match(item)
        confident_title_match = match >= 0.6
        return (
            item.status == "queued",
            confident_title_match,
            match if confident_title_match else 0.0,
            query_match(item),
            item.score,
            item.cited_by_count,
        )

    return sorted(unique.values(), key=rank_key, reverse=True)


def _validate_experiment_program(code: str) -> None:
    """Reject obvious escape/network primitives before untrusted code reaches Docker."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"generated experiment is not valid Python: {exc}") from exc
    forbidden_modules = {
        "asyncio",
        "httpx",
        "multiprocessing",
        "os",
        "requests",
        "shutil",
        "socket",
        "subprocess",
        "urllib",
    }
    forbidden_calls = {"compile", "eval", "exec", "__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = {alias.name.split(".", 1)[0] for alias in node.names}
            blocked = imported & forbidden_modules
            if blocked:
                raise ValueError(f"generated experiment imports forbidden module(s): {sorted(blocked)}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in forbidden_modules:
                raise ValueError(f"generated experiment imports forbidden module: {root}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in forbidden_calls:
                raise ValueError(f"generated experiment calls forbidden builtin: {node.func.id}")


def _existing_refs(
    store: ArtifactStore,
    refs: list[ArtifactRef],
    paths: list[str],
) -> list[ArtifactRef]:
    by_path = {ref.path: ref for ref in refs}
    for path in paths:
        if store.exists(path):
            ref = store.artifact_ref(path)
            by_path.setdefault(ref.path, ref)
    return list(by_path.values())


def _render_literature_report(
    item: WorkItem,
    plan: LiteraturePlan,
    candidates: list[Any],
    imported: list[Any],
    findings: list[Finding],
    errors: list[str],
) -> str:
    lines = [
        f"# Literature work: {item.title}",
        "",
        "This report separates exactly validated quote support from unvalidated candidates.",
        "",
        "## Search queries",
        *[f"- {query}" for query in plan.search_queries],
        "",
        "## Known source-title hints",
        *[f"- {title}" for title in plan.known_source_titles],
        "",
        "## Discovery/import summary",
        f"- Candidates observed: {len(candidates)}",
        f"- Papers imported in this step: {len(imported)}",
        "",
        "## Findings",
    ]
    if not findings:
        lines.append("- No local statement matched the focus questions.")
    for finding in findings:
        lines.extend(
            [
                f"- **{finding.status.value}** `{finding.finding_id}`: {finding.statement}",
                f"  - Source IDs: {', '.join(finding.source_ids) or 'none'}",
            ]
        )
        for caveat in finding.caveats:
            lines.append(f"  - Caveat: {caveat}")
    lines.extend(["", "## Gaps and errors"])
    lines.extend([f"- {error}" for error in errors] or ["- No operational error was recorded."])
    return "\n".join(lines).rstrip() + "\n"


def _render_analysis(item: WorkItem, submission: AnalysisSubmission) -> str:
    lines = [f"# Analysis work: {item.title}", "", submission.summary, "", "## Candidate claims"]
    if not submission.candidate_claims:
        lines.append("- None. This is preferable to an unsupported claim.")
    for claim in submission.candidate_claims:
        lines.append(f"- {claim.statement}")
        lines.append(f"  - Basis: {', '.join(claim.basis_finding_ids) or 'none'}")
        if claim.caveat:
            lines.append(f"  - Caveat: {claim.caveat}")
    lines.extend(["", "## Unresolved questions"])
    lines.extend([f"- {value}" for value in submission.unresolved_questions] or ["- None recorded."])
    lines.extend(["", "## Suggested next steps"])
    lines.extend([f"- {value}" for value in submission.suggested_next_steps] or ["- Human review."])
    return "\n".join(lines).rstrip() + "\n"
