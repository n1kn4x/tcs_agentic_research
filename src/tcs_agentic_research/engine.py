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
                    "Plan the next bounded research work. You have no tools; the deterministic "
                    "application will execute the work items after your response, so lack of tools "
                    "is expected and is never a reason to choose review. Return at most four "
                    "independent work items through the enforced response format, with at most one "
                    "item of each kind. Each item must be executable in one fresh step, cannot "
                    "depend on a file another proposed item will create, and must name observable "
                    "success criteria. An experiment item must combine implementation and execution "
                    "in one self-contained program. Experiment instructions must describe output "
                    "filenames relative to the program's current directory, not canonical workspace "
                    "paths; the application imports them into ExperimentRuns. Use literature for "
                    "source discovery/quotes, proof for one small Lean theorem, experiment for one "
                    "bounded reproducible Python program, and analysis only for synthesis or an "
                    "informal derivation. "
                    "Schedule only kinds that directly advance the task's success criteria or an "
                    "evidence gap in prior results; do not add unrelated cross-checks merely to use "
                    "every subsystem. Never claim the task is solved; choose review when no distinct "
                    "useful work remains."
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
        analysis_refresh = _analysis_needs_refresh(queue, self.store.read_findings())
        if analysis_refresh and not any(
            draft.kind == WorkKind.analysis for draft in required_drafts
        ):
            required_drafts.append(_analysis_work_draft())
        remaining_model_drafts = list(plan.work_items)
        if re.search(r"(?im)^##?\s*Required subsystem", task):
            # A structured task's explicit subsystem section is a user contract. Do not let the
            # planner add unrelated executors merely to demonstrate breadth.
            allowed_kinds = {
                draft.kind for draft in _default_plan(task, existing=[]).work_items
            }
            remaining_model_drafts = [
                draft for draft in remaining_model_drafts if draft.kind in allowed_kinds
            ]
        ordered_drafts: list[WorkItemDraft] = []
        for required in required_drafts:
            match = next(
                (
                    (index, draft)
                    for index, draft in enumerate(remaining_model_drafts)
                    if draft.kind == required.kind
                ),
                None,
            )
            if match is None:
                ordered_drafts.append(required)
            else:
                index, draft = match
                ordered_drafts.append(draft)
                remaining_model_drafts.pop(index)
        ordered_drafts.extend(remaining_model_drafts)

        # A terminal failed/blocked/partial attempt must not permanently suppress a retry. Exact
        # duplicate active or successful work is skipped; bounded later rounds may retry failures.
        existing_keys = {
            _work_key(item.kind, item.instruction)
            for item in queue.items
            if item.status in {WorkStatus.open, WorkStatus.running, WorkStatus.done}
            and not (analysis_refresh and item.kind == WorkKind.analysis)
        }
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
                # Passage retrieval is useful for human scouting but is not itself a bounded
                # statement candidate. Promote only deterministic statement-extraction records.
                if not result.statement_id:
                    continue
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
        recent_results = _recent_result_context(self.store, limit=8)
        analysis_messages = [
            {
                "role": "system",
                "content": (
                    "Perform one bounded TCS analysis. Use only the supplied findings as scientific "
                    "evidence. Return candidate claims with the finding IDs they rely on. You may "
                    "accurately summarize operational success, failure, and artifact availability "
                    "from recent_work_results, but those records do not establish a scientific "
                    "claim. Do not call tools, invent citations, claim novelty, or claim that the "
                    "main task is solved. Candidate analysis remains a hypothesis until separately "
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
                        "recent_work_results": recent_results,
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
            _render_analysis(item, submission, evidence=prior),
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
                    "Translate the bounded proof work into one small Lean goal. Return only the "
                    "LeanGoalDraft JSON fields through response_format. `name` is one identifier. "
                    "`statement` is only a theorem type with every variable explicitly bound; do "
                    "not include `theorem`, `lemma`, `:=`, a proof, or a code fence. For equality "
                    "of expressions containing infix operators, add explicit parentheses so Lean "
                    "parses the intended equality (for example, `∀ (a b : Bool), (a && b) = "
                    "(b && a)`). The type must elaborate using only the requested imports: do not "
                    "refer to a task-specific helper that has not been defined. Prefer an equivalent "
                    "standard-library operation (for example `List.count`) or encode a helper/predicate "
                    "as a self-contained `let` inside the type. If the work asks to introduce a "
                    "definition, state a small direct property of that `let`; never silently replace "
                    "the requested definition with a stronger or mathematically different implication. "
                    "Prefer the smallest useful statement likely to have a direct `rfl` or `simp` proof. "
                    "Do not strengthen the mathematical claim."
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
            description=item.instruction[:1500],
            python_lines=[
                "open('dry_run.txt', 'w').write('dry-run: experiment was not executed')"
            ],
            seeds=[0],
            expected_outputs=["dry_run.txt"],
        )
        experiment_messages = [
            {
                "role": "system",
                "content": (
                    "Design and implement one bounded reproducible Python experiment. Return a "
                    "single self-contained program through response_format; it cannot rely on files "
                    "from prior work items. Put the complete source in python_lines as an array with "
                    "exactly one source line per string, preserving indentation and using no Markdown "
                    "fences. Keep it under 10,000 characters and 250 lines; favor small functions and "
                    "a text/CSV table over plotting boilerplate. List every fixed seed used by the code "
                    "in seeds, and list generated relative file paths in expected_outputs. The program "
                    "must use those fixed seeds, write outputs only in the current directory, avoid "
                    "network access and subprocesses, and finish quickly. Size the workload for well "
                    "under 60 seconds, with explicit operation or per-instance caps for potentially "
                    "exponential algorithms; preserve partial rows and label capped runs rather than "
                    "misreporting a cap as UNSAT. "
                    "Do not import asyncio, httpx, multiprocessing, os, requests, shutil, socket, "
                    "subprocess, or urllib. Use plain relative paths with open() instead. Produce the "
                    "requested machine-readable results and table/plot when applicable, and print a "
                    "concise factual summary. Include executable assertions or a tractable reference "
                    "implementation that checks result correctness and key invariants before reporting; "
                    "for algorithm benchmarks, cross-check a small subset against brute force when "
                    "feasible, and exit nonzero on disagreement. Ensure compared variants are genuinely "
                    "distinct. Prefer simple standard-library code unless "
                    "a common scientific package is materially useful. Experiments do not prove "
                    "mathematical claims."
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
            allow_repair=False,
        )
        try:
            _validate_experiment_program(program)
        except ValueError as first_error:
            if self.router.dry_run:
                raise
            self.store.write_json(f"{run_dir}/invalid_program.json", program)
            repair_messages = [
                {
                    "role": "system",
                    "content": (
                        "Repair one self-contained Python experiment after deterministic validation "
                        "failed. Preserve the experiment objective and output files, but return a "
                        "compact syntactically valid program with none of the forbidden imports or "
                        "builtins named in the error. Return complete source in python_lines with "
                        "exactly one source line per array element and preserved indentation. Do not "
                        "add Markdown fences, network access, or subprocess behavior."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "validation_error": str(first_error),
                            "invalid_program": program.model_dump(mode="json"),
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
            self.store.write_json(
                f"{run_dir}/experiment_repair_model_input.json",
                {"schema": "ExperimentProgram", "messages": repair_messages},
            )
            repaired_program = self.router.complete_structured(
                task_type="experiment_revision",
                messages=repair_messages,
                schema=ExperimentProgram,
                allow_repair=False,
            )
            # The repair owns source code only; preserve the already validated execution contract.
            program = repaired_program.model_copy(
                update={
                    "description": program.description,
                    "seeds": program.seeds,
                    "expected_outputs": program.expected_outputs,
                }
            )
            _validate_experiment_program(program)
        program_ref = self.store.write_json(f"{run_dir}/program.json", program)
        if self.router.dry_run:
            return WorkResult(
                work_id=item.work_id,
                outcome="partial",
                summary="Dry-run generated an experiment program but did not execute Docker.",
                artifact_refs=[model_input_ref, program_ref],
                next_steps=["Run again without --dry-run and with experimenter configuration."],
            )
        experiment_agent = ExperimentAgent(self.store, self.router.experimenter)
        result = experiment_agent.run_program(program=program, name=item.title)
        execution_refs = list(result.artifact_refs)
        program_refs = [program_ref]
        execution_errors = [] if result.success else [result.summary]
        if not result.success:
            runtime_repair_messages = [
                {
                    "role": "system",
                    "content": (
                        "Repair one self-contained Python experiment after a bounded execution "
                        "failed. Use the traceback or missing-output report to correct the program, "
                        "while preserving its objective, fixed seeds, metrics, and expected output "
                        "files. Return complete source in python_lines with one source line per "
                        "array element and preserved indentation. Keep the program compact and do "
                        "not add forbidden imports, network access, or subprocess behavior."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "execution_failure": result.summary[-6000:],
                            "program": program.model_dump(mode="json"),
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
            self.store.write_json(
                f"{run_dir}/experiment_runtime_repair_model_input.json",
                {"schema": "ExperimentProgram", "messages": runtime_repair_messages},
            )
            try:
                revision_output = self.router.complete_structured(
                    task_type="experiment_revision",
                    messages=runtime_repair_messages,
                    schema=ExperimentProgram,
                    allow_repair=False,
                )
                revised_program = revision_output.model_copy(
                    update={
                        "description": program.description,
                        "seeds": program.seeds,
                        "expected_outputs": program.expected_outputs,
                    }
                )
                _validate_experiment_program(revised_program)
                program_refs.append(
                    self.store.write_json(f"{run_dir}/program_revision.json", revised_program)
                )
                result = experiment_agent.run_program(
                    program=revised_program,
                    name=f"{item.title} revision",
                )
                execution_refs.extend(result.artifact_refs)
                if not result.success:
                    execution_errors.append(result.summary)
            except Exception as exc:  # noqa: BLE001 - preserve the initial execution evidence
                repair_error = f"Runtime repair failed: {type(exc).__name__}: {exc}"
                execution_errors.append(repair_error)
                return WorkResult(
                    work_id=item.work_id,
                    outcome="failed",
                    summary="Initial experiment execution failed and its bounded repair did not run.",
                    artifact_refs=[model_input_ref, *program_refs, *execution_refs],
                    errors=execution_errors,
                    next_steps=["Inspect the preserved initial execution and repair diagnostics."],
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
            artifact_refs=[model_input_ref, *program_refs, *execution_refs],
            errors=[] if result.success else execution_errors,
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
    existing_kinds = {
        str(item.get("kind"))
        for item in existing
        if str(item.get("status")) in {"open", "running", "done"}
    }
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
        drafts.append(_analysis_work_draft())
    return PlanSubmission(
        decision="continue" if drafts else "review",
        objective="Make auditable progress through small evidence-producing work items.",
        work_items=drafts[:4],
        reason="Deterministic dry-run plan based on subsystem requirements in the task.",
    )


def _analysis_work_draft() -> WorkItemDraft:
    return WorkItemDraft(
        kind=WorkKind.analysis,
        title="Evidence-bounded synthesis and gap analysis",
        instruction=(
            "Synthesize only the evidence currently recorded, distinguish verified facts from "
            "hypotheses, and identify the smallest unresolved technical question."
        ),
        success_criteria=["Every candidate conclusion names its evidence basis and caveat."],
    )


def _analysis_needs_refresh(queue: WorkQueue, findings: list[Finding]) -> bool:
    if not findings or any(
        item.kind == WorkKind.analysis and item.status in {WorkStatus.open, WorkStatus.running}
        for item in queue.items
    ):
        return False
    completed = [
        item.updated_at
        for item in queue.items
        if item.kind == WorkKind.analysis and item.status == WorkStatus.done
    ]
    latest_analysis = max(completed, default="")
    return max(finding.created_at for finding in findings) > latest_analysis


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
                "artifact_paths": [
                    str(value.get("path") or "")
                    for value in (row.get("artifact_refs") or [])[:8]
                    if isinstance(value, dict) and value.get("path")
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


def _validate_experiment_program(program: ExperimentProgram) -> None:
    """Reject incomplete output and obvious escape/network primitives before Docker."""
    code = program.python_code
    if len(code) > 20_000:
        raise ValueError("generated experiment exceeds the 20,000-character source budget")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"generated experiment is not valid Python: {exc}") from exc
    if not tree.body:
        raise ValueError("generated experiment contains no executable Python statements")
    if not program.expected_outputs:
        raise ValueError("generated experiment must declare at least one machine-readable output")
    for output in program.expected_outputs:
        if output not in code and Path(output).name not in code:
            raise ValueError(f"expected output is not referenced by the program: {output}")
    for top_node in tree.body:
        if not isinstance(top_node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = top_node.targets if isinstance(top_node, ast.Assign) else [top_node.target]
        if not any(isinstance(target, ast.Name) and target.id.lower() == "seeds" for target in targets):
            continue
        if top_node.value is None:
            continue
        try:
            declared_seeds = ast.literal_eval(top_node.value)
        except (ValueError, TypeError):
            continue
        if isinstance(declared_seeds, (list, tuple)) and list(declared_seeds) != program.seeds:
            raise ValueError("the seeds field does not match the program's SEEDS constant")
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


def _render_analysis(
    item: WorkItem,
    submission: AnalysisSubmission,
    *,
    evidence: list[Finding],
) -> str:
    lines = [
        f"# Analysis work: {item.title}",
        "",
        submission.summary,
        "",
        "## Evidence inventory",
    ]
    if not evidence:
        lines.append("- No evidence finding was available.")
    for finding in evidence:
        lines.append(
            f"- **{finding.status.value} / {finding.kind.value}** `{finding.finding_id}`: "
            f"{finding.statement}"
        )
        for caveat in finding.caveats:
            lines.append(f"  - Caveat: {caveat}")
    lines.extend(["", "## Candidate claims"])
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
