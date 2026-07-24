"""Reliable experiment campaigns built around a trusted execution harness.

Generated code implements only scientific primitives. Python owns coverage, validation rows,
timing, registered aggregation, replication comparison, persistence, and acceptance. Natural
language is reviewed semantically by models but is never parsed to drive control flow.
"""

from __future__ import annotations

import ast
import hashlib
import json
from typing import Any, Callable, Literal

from ..agents.experiment import ExperimentAgent
from ..artifact_store import ArtifactStore
from ..llm import LLMRouter, ModelBudgetExceeded, StructuredLLMError
from ..schemas import (
    ArtifactRef,
    CriterionResult,
    EvidenceStrength,
    ExperimentAnalysisSpec,
    ExperimentBlueprint,
    ExperimentCodeAudit,
    ExperimentConditionSpec,
    ExperimentDecisionClause,
    ExperimentDecisionSpec,
    ExperimentDefect,
    ExperimentDesignReview,
    ExperimentEvidenceReview,
    ExperimentImplementationPlan,
    ExperimentMechanismCheckSpec,
    ExperimentMetricSpec,
    ExperimentOutput,
    ExperimentProgram,
    ExperimentReferenceSpec,
    ExperimentResult,
    ExperimentState,
    Finding,
    FindingPolarity,
    FindingStatus,
    WorkItem,
    WorkKind,
    WorkResult,
    utc_now,
)
from ..workflow import _validate_experiment_program

# Public compatibility imports for old API callers. The v3 pipeline below never calls the legacy
# natural-language heuristic functions.
from .experiment_compat import (  # noqa: F401
    _criterion_id_errors,
    _evidence_output_context,
    _protocol_output_errors,
    _protocol_semantic_errors,
    _reconcile_evidence_review,
    _review_errors,
)


_DESIGN_CRITERIA = (
    "alignment",
    "comparators",
    "sampling",
    "reference",
    "analysis",
    "feasibility",
)


class ExperimentPipeline:
    """Advance one durable experiment through a compact, restart-safe campaign."""

    MAX_TRANSITIONS = 24
    MAX_REPAIRS_PER_CYCLE = 2
    MAX_DUPLICATE_CANDIDATES = 2

    def __init__(self, store: ArtifactStore, router: LLMRouter):
        self.store = store
        self.router = router

    def run(
        self,
        item: WorkItem,
        run_dir: str,
        *,
        research_context: dict[str, Any] | None = None,
    ) -> WorkResult:
        state_path = f"ExperimentStates/{item.work_id}.json"
        state = self._load_state(state_path, item)
        context = research_context or {}
        refs: dict[str, ArtifactRef] = {}
        initial_revisions = state.protocol_revision + state.program_revision

        def add(ref: ArtifactRef) -> None:
            refs[ref.path] = ref

        def persist(step_dir: str) -> None:
            state.updated_at = utc_now()
            add(self.store.write_json(state_path, state))
            add(self.store.write_json(f"{step_dir}/experiment_state.json", state))

        for transition in range(self.MAX_TRANSITIONS):
            step_dir = f"{run_dir}/experiment_steps/{transition + 1:02d}_{state.stage}"
            result = self._advance(item, state, step_dir, context, add, persist)
            if result is not None:
                for ref in result.artifact_refs:
                    add(ref)
                result.artifact_refs = list(refs.values())
                return result
            repairs = state.protocol_revision + state.program_revision - initial_revisions
            if repairs >= self.MAX_REPAIRS_PER_CYCLE and state.stage in {
                "repair_design",
                "repair_implementation",
            }:
                persist(step_dir)
                return WorkResult(
                    work_id=item.work_id,
                    outcome="partial",
                    failure_class="none",
                    attempt_class="engineering",
                    continue_work=True,
                    summary=(
                        "Experiment campaign paused after two repairs. The frozen design, best source, "
                        "and structured defects will resume without restarting."
                    ),
                    errors=[defect.summary for defect in state.active_defects],
                )

        state.engineering_blocked = True
        state.last_error = "Campaign exceeded its deterministic transition bound."
        persist(f"{run_dir}/experiment_steps/transition_limit")
        return self._blocked(item, state)

    def _load_state(self, path: str, item: WorkItem) -> ExperimentState:
        if not self.store.exists(path):
            return ExperimentState(work_id=item.work_id)
        state = ExperimentState.model_validate(self.store.read_json(path))
        # Old in-progress v2 states are intentionally not migrated into the new evidence contract.
        if state.stage in {
            "protocol_design", "protocol_review", "protocol_revision", "program_design",
            "program_review", "program_revision", "smoke_execution", "full_execution",
            "evidence_review",
        }:
            return ExperimentState(work_id=item.work_id)
        if state.stage == "source_audit" and state.smoke_result is None:
            # States produced by the first v3 iteration audited before smoke. Preserve their source
            # but route it through the stronger executable gate before resuming review.
            state.stage = "smoke"
        if (
            state.stage == "repair_implementation"
            and state.blueprint is not None
            and state.execution_result is not None
            and state.replication_result is not None
            and not _replication_defects(
                state.blueprint, state.execution_result, state.replication_result
            )
        ):
            # Reassess durable runs after a deterministic validator update instead of demanding a
            # no-op source rewrite. This is especially important for week-long resumable campaigns.
            state.engineering_blocked = False
            state.active_defects = []
            state.stage = "evidence_audit"
        return state

    def _advance(
        self,
        item: WorkItem,
        state: ExperimentState,
        step_dir: str,
        context: dict[str, Any],
        add: Callable[[ArtifactRef], None],
        persist: Callable[[str], None],
    ) -> WorkResult | None:
        if state.engineering_blocked:
            persist(step_dir)
            return self._blocked(item, state)

        settings = self.router.experimenter
        if not self.router.dry_run and (settings is None or not settings.enabled):
            state.engineering_blocked = True
            state.last_error = "Docker experiment execution is disabled."
            persist(step_dir)
            return self._blocked(item, state)
        max_wall = settings.timeout_seconds if settings else 600
        max_memory = _memory_to_mb(settings.memory) if settings else 4096
        max_cpus = settings.cpus if settings else 2.0

        if self.router.dry_run and state.stage == "design":
            state.blueprint = _dry_blueprint(item, max_wall, max_memory, max_cpus)
            state.protocol_sha256 = _blueprint_sha(state.blueprint)
            state.program = _dry_study_program(state.blueprint)
            result = WorkResult(
                work_id=item.work_id,
                outcome="partial",
                attempt_class="engineering",
                continue_work=False,
                summary="Dry run validated v3 campaign planning and generated no scientific claim.",
            )
            state.final_result = result
            state.stage = "complete"
            persist(step_dir)
            return result

        if state.stage in {"design", "repair_design"}:
            revision = state.stage == "repair_design"
            messages = _design_messages(
                item,
                state,
                context,
                max_wall=max_wall,
                max_memory=max_memory,
                max_cpus=max_cpus,
            )
            add(self.store.write_json(f"{step_dir}/input.json", {"messages": messages}))
            blueprint = self.router.complete_structured(
                task_type="experiment_protocol",
                messages=messages,
                schema=ExperimentBlueprint,
                temperature=0.1,
                max_tokens=6144,
                allow_repair=True,
            )
            resource_defects = _resource_defects(
                blueprint, max_wall=max_wall, max_memory=max_memory, max_cpus=max_cpus
            )
            state.blueprint = blueprint
            state.protocol_revision += int(revision)
            add(self.store.write_json(f"{step_dir}/blueprint.json", blueprint))
            if resource_defects:
                return self._repair_or_block(
                    item, state, step_dir, resource_defects, "repair_design", persist
                )
            state.design_review = None
            state.active_defects = []
            state.stage = "design_review"
            persist(step_dir)
            return None

        if state.stage == "design_review":
            assert state.blueprint is not None
            messages = _design_review_messages(item, state.blueprint, context)
            add(self.store.write_json(f"{step_dir}/input.json", {"messages": messages}))
            design_review = self.router.complete_structured(
                task_type="experiment_review",
                messages=messages,
                schema=ExperimentDesignReview,
                temperature=0.1,
                max_tokens=4096,
                allow_repair=True,
            )
            shape_errors = _design_review_shape_errors(design_review)
            if shape_errors:
                persist(step_dir)
                raise StructuredLLMError("Invalid design-review coverage: " + "; ".join(shape_errors))
            state.design_review = design_review
            add(self.store.write_json(f"{step_dir}/review.json", design_review))
            failed = [row for row in design_review.assessments if not row.satisfied]
            if failed:
                defects = design_review.defects or [
                    ExperimentDefect(
                        defect_id=row.criterion_id,
                        summary=row.detail,
                        repair=f"Revise the typed {row.criterion_id} fields to satisfy this assessment.",
                    )
                    for row in failed
                ]
                if state.protocol_revision < 1:
                    return self._repair_or_block(
                        item, state, step_dir, defects, "repair_design", persist
                    )
                # Semantic reviewers tend to discover a new optional improvement on every rewrite.
                # After one focused revision, freeze the still-valid typed design and carry the
                # dissent into the final evidence audit instead of preventing all execution.
                state.last_error = "Design-review dissent preserved: " + "; ".join(
                    defect.summary for defect in defects
                )
                state.protocol_sha256 = _blueprint_sha(state.blueprint)
                state.active_defects = []
                state.stage = "implementation"
                persist(step_dir)
                return None
            if design_review.defects:
                persist(step_dir)
                raise StructuredLLMError("Accepted design review contradicted non-empty defects")
            state.protocol_sha256 = _blueprint_sha(state.blueprint)
            state.active_defects = []
            state.stage = "implementation"
            persist(step_dir)
            return None

        if state.stage in {"implementation", "repair_implementation"}:
            assert state.blueprint is not None and state.protocol_sha256
            revision = state.stage == "repair_implementation"
            repair_plan: ExperimentImplementationPlan | None = None
            if revision and state.program is not None:
                plan_messages = _implementation_repair_plan_messages(item, state)
                add(
                    self.store.write_json(
                        f"{step_dir}/repair_plan_input.json", {"messages": plan_messages}
                    )
                )
                repair_plan = self.router.complete_structured(
                    task_type="experiment_review",
                    messages=plan_messages,
                    schema=ExperimentImplementationPlan,
                    temperature=0.1,
                    max_tokens=4096,
                    allow_repair=True,
                )
                add(self.store.write_json(f"{step_dir}/repair_plan.json", repair_plan))
            messages = _implementation_messages(
                item, state, context, revision=revision, repair_plan=repair_plan
            )
            add(self.store.write_json(f"{step_dir}/input.json", {"messages": messages}))
            try:
                source = self.router.complete_text(
                    task_type="experiment_revision" if revision else "experiment_design",
                    messages=messages,
                    temperature=0.2 if revision else 0.1,
                    max_tokens=12288,
                )
                program = ExperimentProgram(
                    description=f"Study module for {state.blueprint.title}",
                    interface="study_v1",
                    source=source,
                    seeds=state.blueprint.seeds,
                )
                _validate_experiment_program(program)
                contract_errors = _source_contract_errors(state.blueprint, program)
                if contract_errors:
                    raise ValueError("; ".join(contract_errors))
            except (ModelBudgetExceeded, StructuredLLMError):
                persist(step_dir)
                raise
            except Exception as exc:
                state.program_revision += int(revision)
                defect = ExperimentDefect(
                    defect_id="source_contract",
                    summary=f"Generated source failed structural validation: {type(exc).__name__}: {exc}"[:1200],
                    repair="Return one complete study_v1 module satisfying the frozen callable contract.",
                )
                return self._repair_or_block(
                    item, state, step_dir, [defect], "repair_implementation", persist
                )

            candidate_hash = hashlib.sha256(program.python_code.encode("utf-8")).hexdigest()
            state.candidate_hashes = [*state.candidate_hashes, candidate_hash][-40:]
            if revision:
                state.program_revision += 1
            # A byte-identical candidate is allowed through the executable gates. This preserves
            # the underlying runtime/mechanism defect instead of replacing it with a generic
            # "duplicate" message and trapping all later repairs in a no-information loop.
            state.program = program
            state.code_audit = None
            state.smoke_result = None
            state.execution_result = None
            state.replication_result = None
            state.active_defects = []
            state.stage = "smoke"
            add(self.store.write_json(f"{step_dir}/program.json", program))
            add(self.store.write_text(f"{step_dir}/implementation.py", program.python_code + "\n"))
            persist(step_dir)
            return None

        if state.stage == "source_audit":
            assert state.blueprint is not None and state.program is not None
            assert state.smoke_result is not None
            analysis_messages = _source_audit_messages(
                item, state.blueprint, state.program, state.smoke_result
            )
            add(
                self.store.write_json(
                    f"{step_dir}/analysis_input.json", {"messages": analysis_messages}
                )
            )
            # Reasoning models are better source analysts but unreliable JSON emitters under
            # response_format. Let one reason in text, then let the control profile make a compact
            # typed adjudication from the exact source and analysis. Neither sees conversational history.
            analysis_notes = self.router.complete_text(
                task_type="experiment_implementation",
                messages=analysis_messages,
                temperature=0.2,
                max_tokens=6144,
            )
            add(
                self.store.write_text(
                    f"{step_dir}/analysis_notes.md", analysis_notes.rstrip() + "\n"
                )
            )
            messages = _source_audit_adjudication_messages(
                item, state.blueprint, state.program, state.smoke_result, analysis_notes
            )
            add(self.store.write_json(f"{step_dir}/input.json", {"messages": messages}))
            audit = self.router.complete_structured(
                task_type="experiment_review",
                messages=messages,
                schema=ExperimentCodeAudit,
                temperature=0.1,
                max_tokens=6144,
                allow_repair=True,
            )
            expected = {row.id for row in state.blueprint.conditions}
            actual = set(audit.condition_implementation)
            if expected != actual:
                persist(step_dir)
                raise StructuredLLMError(
                    "Source audit condition IDs differ from blueprint: "
                    f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
                )
            state.code_audit = audit
            add(self.store.write_json(f"{step_dir}/source_audit.json", audit))
            if not audit.accepted:
                state.source_audit_failures += 1
                if state.source_audit_failures < 2:
                    return self._repair_or_block(
                        item, state, step_dir, audit.defects, "repair_implementation", persist
                    )
                # Execute the bounded full/replication gates before final classification. The
                # rejected audit remains in state and is supplied to the evidence reviewer; it can
                # prevent full acceptance, but subjective dissent cannot erase runnable measurements.
                state.last_error = "Source-audit dissent preserved: " + "; ".join(
                    defect.summary for defect in audit.defects
                )
            state.active_defects = []
            state.stage = "full"
            persist(step_dir)
            return None

        if state.stage in {"smoke", "full", "replication"}:
            assert state.blueprint is not None and state.program is not None
            mode: Literal["smoke", "full"] = "smoke" if state.stage == "smoke" else "full"
            agent = ExperimentAgent(self.store, self.router.experimenter)
            execution = agent.run_program(
                program=state.program,
                blueprint=state.blueprint,
                name=(
                    f"{item.title}_smoke" if state.stage == "smoke"
                    else f"{item.title}_{state.stage}"
                ),
                mode=mode,
                timeout_seconds=(min(60, state.blueprint.wall_seconds) if mode == "smoke" else state.blueprint.wall_seconds),
            )
            for ref in execution.artifact_refs:
                add(ref)
            if execution.failure_class == "infrastructure":
                state.infrastructure_failures += 1
                persist(step_dir)
                return WorkResult(
                    work_id=item.work_id,
                    outcome="partial",
                    failure_class="operational",
                    attempt_class="engineering",
                    summary=execution.summary,
                    errors=[execution.summary],
                    next_steps=["Restore the container infrastructure and resume the same stage."],
                )
            execution_defects = _execution_defects(state.blueprint, execution, smoke=mode == "smoke")
            if execution_defects:
                if state.stage == "smoke":
                    state.smoke_result = execution
                elif state.stage == "full":
                    state.execution_result = execution
                else:
                    state.replication_result = execution
                repair_stage = (
                    "repair_design"
                    if all(defect.defect_id == "conditions" for defect in execution_defects)
                    else "repair_implementation"
                )
                return self._repair_or_block(
                    item, state, step_dir, execution_defects, repair_stage, persist
                )
            state.infrastructure_failures = 0
            if state.stage == "smoke":
                state.smoke_result = execution
                state.stage = "source_audit"
            elif state.stage == "full":
                state.execution_result = execution
                state.stage = "replication"
            else:
                state.replication_result = execution
                assert state.execution_result is not None
                replication_defects = _replication_defects(
                    state.blueprint, state.execution_result, execution
                )
                if replication_defects:
                    return self._repair_or_block(
                        item,
                        state,
                        step_dir,
                        replication_defects,
                        "repair_implementation",
                        persist,
                    )
                state.scientific_attempts += 1
                state.stage = "evidence_audit"
            state.active_defects = []
            persist(step_dir)
            return None

        if state.stage == "evidence_audit":
            assert state.blueprint is not None
            assert state.program is not None
            assert state.execution_result is not None
            assert state.replication_result is not None
            evidence_review = self._review_evidence(item, state, context)
            add(self.store.write_json(f"{step_dir}/review.json", evidence_review))
            expected_criteria = {
                f"W{index:02d}": criterion
                for index, criterion in enumerate(item.success_criteria, 1)
            }
            id_errors = _criterion_id_errors(expected_criteria, evidence_review.criteria)
            if id_errors:
                persist(step_dir)
                raise StructuredLLMError("Evidence reviewer omitted fixed criteria: " + "; ".join(id_errors))
            all_criteria = all(row.satisfied for row in evidence_review.criteria)
            essential_design_ids = {"alignment", "analysis", "feasibility"}
            design_sound = (
                state.design_review is not None
                and all(
                    row.satisfied
                    for row in state.design_review.assessments
                    if row.criterion_id in essential_design_ids
                )
            )
            full = (
                evidence_review.usable == "full"
                and all_criteria
                and not evidence_review.issues
                and design_sound
                and state.code_audit is not None
                and state.code_audit.accepted
            )
            interpretable = evidence_review.usable in {"full", "preliminary"}
            if not interpretable:
                issues = evidence_review.issues or [evidence_review.scientific_summary]
                defects = [
                    ExperimentDefect(
                        defect_id="evidence",
                        summary=issue[:1200],
                        repair=(
                            evidence_review.follow_up[0]
                            if evidence_review.follow_up
                            else "Revise the frozen design to produce usable evidence."
                        )[:1200],
                    )
                    for issue in issues[:4]
                ]
                return self._repair_or_block(
                    item, state, step_dir, defects, "repair_design", persist
                )

            rows = _criterion_results(expected_criteria, evidence_review)
            finding = Finding(
                work_id=item.work_id,
                question_id=item.question_id,
                requirement_id=item.requirement_id,
                kind=WorkKind.experiment,
                statement=evidence_review.scientific_summary,
                status=FindingStatus.observed,
                polarity=FindingPolarity(evidence_review.outcome),
                strength=EvidenceStrength.strong if full else EvidenceStrength.preliminary,
                scope=(
                    f"Replicated registered experiment `{state.blueprint.title}` with "
                    f"{state.blueprint.sample_size} units under each of "
                    f"{len(state.blueprint.conditions)} conditions; seeds {state.blueprint.seeds}."
                ),
                evidence_refs=list(
                    {
                        ref.path: ref
                        for result in [state.execution_result, state.replication_result]
                        for ref in result.artifact_refs
                    }.values()
                ),
                source_ids=[state.execution_result.run_id, state.replication_result.run_id],
                caveats=[
                    "Empirical evidence is not a mathematical proof or asymptotic result.",
                    *state.blueprint.limitations,
                    *evidence_review.caveats,
                ],
            )
            design_errors = (
                []
                if design_sound or state.design_review is None
                else [
                    row.detail
                    for row in state.design_review.assessments
                    if not row.satisfied and row.criterion_id in essential_design_ids
                ]
            )
            audit_errors = (
                []
                if state.code_audit is None or state.code_audit.accepted
                else [defect.summary for defect in state.code_audit.defects]
            )
            audit_repairs = (
                []
                if state.code_audit is None or state.code_audit.accepted
                else [defect.repair for defect in state.code_audit.defects]
            )
            sound_measurements = (
                design_sound
                and state.code_audit is not None
                and state.code_audit.accepted
            )
            result = WorkResult(
                work_id=item.work_id,
                outcome="done" if full else "partial",
                failure_class="none" if full else "method",
                attempt_class="scientific",
                evidence_level="substantive" if full else "preliminary",
                recovery_scope=(
                    "design" if design_errors else "implementation" if audit_errors else "strategy"
                ),
                requirement_satisfied=full,
                criteria=rows,
                summary=evidence_review.scientific_summary,
                findings=[finding] if full or sound_measurements else [],
                errors=[] if full else [*evidence_review.issues, *design_errors, *audit_errors],
                next_steps=[
                    *evidence_review.follow_up,
                    *(
                        ["Revise the frozen design to resolve every preserved design-review dissent."]
                        if design_errors
                        else []
                    ),
                    *audit_repairs,
                ],
            )
            state.final_result = result
            state.active_defects = []
            state.stage = "complete"
            add(
                self.store.write_text(
                    f"{step_dir}/scientific_review.md",
                    _scientific_review_markdown(state.blueprint, evidence_review),
                )
            )
            persist(step_dir)
            return result

        if state.stage == "complete":
            persist(step_dir)
            if state.final_result is not None:
                return state.final_result.model_copy(deep=True)
            return WorkResult(
                work_id=item.work_id,
                outcome="failed",
                failure_class="engineering",
                attempt_class="engineering",
                summary="Completed campaign state has no final result.",
                errors=["Inspect or reset the campaign state."],
            )

        state.engineering_blocked = True
        state.last_error = f"Unknown experiment campaign stage: {state.stage}"
        persist(step_dir)
        return self._blocked(item, state)

    def _repair_or_block(
        self,
        item: WorkItem,
        state: ExperimentState,
        step_dir: str,
        defects: list[ExperimentDefect],
        next_stage: str,
        persist: Callable[[str], None],
        *,
        force_block: bool = False,
    ) -> WorkResult | None:
        state.active_defects = defects[:12]
        state.last_error = "; ".join(defect.summary for defect in defects)[-4000:]
        state.stage = next_stage  # type: ignore[assignment]
        revision_count = state.protocol_revision + state.program_revision
        cap = self.router.core.max_experiment_engineering_retries
        if force_block or revision_count >= cap:
            state.engineering_blocked = True
            state.last_error = (
                f"Campaign exhausted {revision_count} source/design revisions. " + state.last_error
            )[-4000:]
            persist(step_dir)
            return self._blocked(item, state)
        persist(step_dir)
        return None

    @staticmethod
    def _blocked(item: WorkItem, state: ExperimentState) -> WorkResult:
        return WorkResult(
            work_id=item.work_id,
            outcome="failed",
            failure_class="engineering",
            attempt_class="engineering",
            summary="Experiment campaign stopped at a bounded, concrete engineering defect.",
            errors=[state.last_error or "Unknown experiment defect."],
            next_steps=[defect.repair for defect in state.active_defects]
            or ["Inspect the preserved campaign and repair the named defect."],
        )

    def _review_evidence(
        self,
        item: WorkItem,
        state: ExperimentState,
        context: dict[str, Any],
    ) -> ExperimentEvidenceReview:
        assert state.blueprint is not None
        assert state.execution_result is not None
        assert state.replication_result is not None
        first = state.execution_result.validated_output
        second = state.replication_result.validated_output
        assert first is not None and second is not None
        criteria = [
            {"criterion_id": f"W{index:02d}", "text": value}
            for index, value in enumerate(item.success_criteria, 1)
        ]
        return self.router.complete_structured(
            task_type="experiment_evidence_review",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Interpret one registered, independently replicated experiment. The trusted "
                        "harness executed every condition/unit pair, generated reference comparisons, "
                        "and computed all registered aggregate operations. Assess every supplied W ID "
                        "exactly once. Recompute the decision from the supplied aggregates and frozen "
                        "decision rule. Negative and null outcomes are valid. Grant full only when the "
                        "literal work criteria and requested regimes are covered, all checks pass, both "
                        "runs agree on deterministic results, and the scoped conclusion is warranted. "
                        "Never infer asymptotic or proof claims from a bounded experiment. Issues are fatal; "
                        "limitations/caveats are nonfatal scope restrictions."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "work": {
                                "title": item.title,
                                "hypothesis": item.hypothesis,
                                "criteria": criteria,
                            },
                            "research_objective": context.get("research_objective", ""),
                            "constraints": context.get("agenda_constraints", []),
                            "deliverables": context.get("agenda_deliverables", []),
                            "blueprint": state.blueprint.model_dump(mode="json"),
                            "source_audit": (
                                state.code_audit.model_dump(mode="json") if state.code_audit else None
                            ),
                            "full_run": _evidence_output_context(first),
                            "replication": _evidence_output_context(second),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            schema=ExperimentEvidenceReview,
            temperature=0.1,
            max_tokens=6144,
            allow_repair=True,
        )


def _design_messages(
    item: WorkItem,
    state: ExperimentState,
    context: dict[str, Any],
    *,
    max_wall: int,
    max_memory: int,
    max_cpus: float,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Design one executable schema-v3 experiment. Use explicit condition roles and stable "
                "IDs. Every condition must receive the same generated units. Register only deterministic "
                "implementation metrics. Mark costs such as recursive calls role=performance and direct "
                "feature counters such as propagated literals role=mechanism. Mechanism checks may use only "
                "mechanism counters and must never assert expected performance or hypothesis direction. "
                "For wall time, either reference reserved `harness_wall_seconds` directly or register a real "
                "performance metric with source=harness_wall_seconds and deterministic=false. Every operation "
                "count or feature counter must use source=implementation. Use the typed executable decision_rule only; never request a "
                "p-value, confidence interval, or statistic not represented by an analysis ID. Paired "
                "differences are condition minus baseline. Prefer deterministic operation counts. Choose an "
                "exact independent reference for every sampled and mechanism-fixture unit whenever bounded "
                "exhaustive validation is feasible. A reference is never a single known seed or fixture. Select a "
                "scalar result_type for the primary scientific answer (for SAT use boolean); never include "
                "condition IDs, assignments, metrics, timing, or metadata in the result. The "
                "implementation will expose make_unit, run_condition, reference_result or validate_result, "
                "and make_mechanism_fixture. Register typed mechanism checks that the trusted harness "
                "evaluates from actual run_condition metrics, with a discriminating fixture for every "
                "condition. For SAT specifically, use propagated-literal counters to check unit propagation "
                "and a dedicated heuristic-selection counter to check heuristic dispatch; recursive calls and "
                "runtime are performance outcomes, never mechanism validity checks. Fixtures should create "
                "derived unit clauses after an assignment, not merely an initially unit input. A trusted "
                "harness owns loops and aggregates. For exponential pure-Python "
                "algorithms with exhaustive references, use 8-30 units and parameters that make both smoke "
                "and full validation comfortably bounded (for naive SAT, normally at most 14 variables). "
                "Cover explicitly requested regimes and fixed seeds. Preserve null/negative outcomes. "
                "Do not introduce unrequested external optimized systems. On revision, correct only the "
                "structured defects while preserving sound design decisions."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "work_item": _work_context(item),
                    "objective": context.get("research_objective", ""),
                    "constraints": context.get("agenda_constraints", []),
                    "deliverables": context.get("agenda_deliverables", []),
                    "accepted_prior_evidence": context.get("accepted_prior_evidence", []),
                    "limits": {
                        "wall_seconds": max_wall,
                        "memory_mb": max_memory,
                        "cpus": max_cpus,
                    },
                    "previous_blueprint": (
                        state.blueprint.model_dump(mode="json") if state.stage == "repair_design" and state.blueprint else None
                    ),
                    "defects": [row.model_dump(mode="json") for row in state.active_defects],
                },
                ensure_ascii=False,
            ),
        },
    ]


def _design_review_messages(
    item: WorkItem,
    blueprint: ExperimentBlueprint,
    context: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Audit this frozen design before coding. Return exactly one assessment for each enum "
                "criterion: alignment, comparators, sampling, reference, analysis, feasibility. A false "
                "assessment requires at least one structured defect with a concrete repair. Check literal "
                "task coverage, meaningful requested conditions, shared fixed-seed sampling, oracle "
                "independence, analysis direction/decision compatibility, and bounded runtime. Mechanism "
                "checks must assert direct feature counters on discriminating fixtures, not correctness "
                "outcomes, recursive-call speedups, favorable results, or hypothesis direction. The exact "
                "reference must return the same primary result_type for every sampled and fixture unit. Do "
                "not invent requirements absent from the task."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "work": _work_context(item),
                    "constraints": context.get("agenda_constraints", []),
                    "deliverables": context.get("agenda_deliverables", []),
                    "blueprint": blueprint.model_dump(mode="json"),
                },
                ensure_ascii=False,
            ),
        },
    ]


def _implementation_repair_plan_messages(
    item: WorkItem,
    state: ExperimentState,
) -> list[dict[str, str]]:
    assert state.blueprint is not None and state.program is not None
    return [
        {
            "role": "system",
            "content": (
                "Diagnose the supplied executable study failure against the complete source and frozen "
                "blueprint. Produce a concise repair plan naming exact functions, root causes, replacement "
                "logic, correctness strategy, output contract, and a fixture that demonstrates the fix. "
                "Preserve correct algorithms and IDs. The reference callable must handle every arbitrary "
                "sample and fixture; never repair it by special-casing a seed. Do not emit Python, weaken an "
                "assertion, change the blueprint, or merely restate the error."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "work_title": item.title,
                    "blueprint": state.blueprint.model_dump(mode="json"),
                    "defects": [row.model_dump(mode="json") for row in state.active_defects],
                    "complete_source": state.program.python_code,
                },
                ensure_ascii=False,
            ),
        },
    ]


def _implementation_messages(
    item: WorkItem,
    state: ExperimentState,
    context: dict[str, Any],
    *,
    revision: bool,
    repair_plan: ExperimentImplementationPlan | None,
) -> list[dict[str, str]]:
    assert state.blueprint is not None
    system = (
        "Return only one complete raw Python module, no Markdown or explanation. Implement these "
        "synchronous callables: make_unit(index, seed), run_condition(condition_id, unit) returning "
        "exactly {'result': a value of blueprint.result_type, 'metrics': {every metric whose source is implementation}}, and the "
        "same complete implementation-metric ID set must be returned by every condition (use a measured "
        "zero for a feature counter that is inactive in a baseline). "
        "blueprint's validation callable: reference_result(unit) for exact_reference or "
        "validate_result(unit, condition_id, result) for result_validator. Implement "
        "make_mechanism_fixture(check_id), returning a JSON-serializable unit with exactly the same "
        "keys/types expected from make_unit (never a class instance or partial pseudo-unit). "
        "reference_result must independently compute every arbitrary sampled or fixture unit (never "
        "special-case/reject one seed) and return only the same scientific result type as "
        "run_condition['result'], never metrics, timing, or metadata. The trusted harness invokes run_condition on fixtures and "
        "evaluates metric comparisons; generated code never returns mechanism passed bits. The harness loops over samples, "
        "times calls, compares references, and computes analyses. Implement every condition literally, "
        "use no network/subprocess/multiprocessing, never hard-code benchmark outcomes, and keep the file "
        "under 30,000 characters. Correctness fixtures must exercise actual mechanisms. For SAT "
        "experiments, use one shared recursive solver parameterized by unit-propagation and branching "
        "flags; copy assignment state at every branch; scan each clause into satisfied/conflict/unassigned "
        "literals; propagate derived unit literals to a fixpoint; select the requested branch variable; and "
        "use exhaustive truth-assignment enumeration as an independent Boolean reference for every unit. "
        "Generate each 3-SAT clause from three distinct variables before assigning signs. Never use the "
        "condition solver as its own reference and never carry propagated assignments across sibling branches."
    )
    if revision:
        system += (
            " Repair every supplied structured defect. Retain correct code, but replace defective "
            "functions. Do not weaken checks, alter IDs, or return the unchanged rejected file."
        )
        if state.candidate_hashes and state.candidate_hashes.count(state.candidate_hashes[-1]) >= 2:
            system += (
                " Prior complete-file repairs were byte-identical and did not resolve the executable "
                "defect. Independently rewrite the affected algorithm and fixture rather than copying it."
            )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "work_title": item.title,
                    "blueprint": state.blueprint.model_dump(mode="json"),
                    "blueprint_sha256": state.protocol_sha256,
                    "defects": [row.model_dump(mode="json") for row in state.active_defects],
                    "repair_plan": (
                        repair_plan.model_dump(mode="json") if repair_plan is not None else None
                    ),
                    "previous_source": (
                        state.program.python_code if revision and state.program is not None else None
                    ),
                    "constraints": context.get("agenda_constraints", []),
                },
                ensure_ascii=False,
            ),
        },
    ]


def _source_audit_messages(
    item: WorkItem,
    blueprint: ExperimentBlueprint,
    program: ExperimentProgram,
    smoke: ExperimentResult,
) -> list[dict[str, str]]:
    assert smoke.validated_output is not None
    return [
        {
            "role": "system",
            "content": (
                "Reason carefully about one complete, already smoke-executed study module. Produce a "
                "concise source audit, not JSON. Trace each condition, generator, exact reference, metric "
                "counter, and mechanism fixture. Report only concrete semantic defects with a function or "
                "expression-level witness. The trusted harness adds `harness_wall_seconds`; its absence from "
                "source is correct. Clause simplification is not unit propagation unless it assigns forced "
                "unit literals. A recursive-call metric counts recursive solver entries, not propagation-loop "
                "iterations. Do not reject harmless style, equivalent algorithms, or a valid implementation "
                "merely to find a flaw. Pay special attention to random k-SAT clauses using distinct variables, "
                "independence of the reference path, backtracking state, and whether fixtures directly exercise "
                "the requested differences. Smoke checks passing is useful evidence but not conclusive."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "work_title": item.title,
                    "blueprint": blueprint.model_dump(mode="json"),
                    "source": program.python_code,
                    "smoke_output": _evidence_output_context(smoke.validated_output),
                },
                ensure_ascii=False,
            ),
        },
    ]


def _source_audit_adjudication_messages(
    item: WorkItem,
    blueprint: ExperimentBlueprint,
    program: ExperimentProgram,
    smoke: ExperimentResult,
    analysis_notes: str,
) -> list[dict[str, str]]:
    assert smoke.validated_output is not None
    return [
        {
            "role": "system",
            "content": (
                "Adjudicate the supplied source-analysis notes against the literal source. Return the "
                "ExperimentCodeAudit contract. Include every exact condition ID once. Reject only a concrete "
                "scientific defect supported by source; discard speculative, internally contradictory, or "
                "stylistic concerns. The trusted harness owns wall timing and complete coverage. Clause "
                "simplification alone is not unit propagation, and recursive-call counters count recursive "
                "function entries. accepted=true requires all condition booleans, generator_valid, "
                "reference_independent, and metrics_valid true with no defects. Every rejection needs a "
                "structured function-level repair."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "work_title": item.title,
                    "blueprint": blueprint.model_dump(mode="json"),
                    "source": program.python_code,
                    "smoke_checks": [
                        row.model_dump(mode="json")
                        for row in smoke.validated_output.checks
                    ],
                    "analysis_notes": analysis_notes[:7000],
                },
                ensure_ascii=False,
            ),
        },
    ]


def _source_contract_errors(
    blueprint: ExperimentBlueprint,
    program: ExperimentProgram,
) -> list[str]:
    tree = ast.parse(program.python_code)
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    required = {"make_unit", "run_condition"}
    required.add(
        "reference_result" if blueprint.reference.kind == "exact_reference" else "validate_result"
    )
    if blueprint.mechanism_checks:
        required.add("make_mechanism_fixture")
    missing = sorted(required - functions)
    return ["Missing required study callables: " + ", ".join(missing)] if missing else []


def _design_review_shape_errors(review: ExperimentDesignReview) -> list[str]:
    ids = [row.criterion_id for row in review.assessments]
    errors: list[str] = []
    missing = sorted(set(_DESIGN_CRITERIA) - set(ids))
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if missing:
        errors.append("missing: " + ", ".join(missing))
    if duplicates:
        errors.append("duplicates: " + ", ".join(duplicates))
    return errors


def _resource_defects(
    blueprint: ExperimentBlueprint,
    *,
    max_wall: int,
    max_memory: int,
    max_cpus: float,
) -> list[ExperimentDefect]:
    failures: list[str] = []
    if blueprint.wall_seconds > max_wall:
        failures.append(f"wall_seconds {blueprint.wall_seconds} exceeds {max_wall}")
    if blueprint.memory_mb > max_memory:
        failures.append(f"memory_mb {blueprint.memory_mb} exceeds {max_memory}")
    if blueprint.cpus > max_cpus:
        failures.append(f"cpus {blueprint.cpus} exceeds {max_cpus}")
    defects: list[ExperimentDefect] = []
    if failures:
        defects.append(
            ExperimentDefect(
                defect_id="feasibility",
                summary="; ".join(failures),
                repair=(
                    "Reduce the declared resources to the supplied hard limits without dropping task coverage."
                ),
            )
        )
    metrics = {row.id: row for row in blueprint.metrics}
    invalid_timers = [
        row.id
        for row in blueprint.metrics
        if row.source == "harness_wall_seconds"
        and (row.value_type != "real" or row.role != "performance")
    ]
    if invalid_timers:
        defects.append(
            ExperimentDefect(
                defect_id="metrics",
                summary="Invalid harness timer declarations: " + ", ".join(invalid_timers),
                repair=(
                    "Harness wall-time metrics must have value_type=real, role=performance, "
                    "source=harness_wall_seconds, and deterministic=false."
                ),
            )
        )
    invalid_mechanisms = [
        row.id
        for row in blueprint.mechanism_checks
        if row.metric_id == "harness_wall_seconds"
        or metrics.get(row.metric_id) is None
        or metrics[row.metric_id].role != "mechanism"
    ]
    if invalid_mechanisms:
        defects.append(
            ExperimentDefect(
                defect_id="conditions",
                summary=(
                    "Mechanism checks use non-mechanism/performance metrics: "
                    + ", ".join(invalid_mechanisms)
                ),
                repair=(
                    "Register direct feature counters with role=mechanism and assert those counters; "
                    "never assert runtime, recursive-call improvement, or expected result direction."
                ),
            )
        )
    return defects


def _execution_defects(
    blueprint: ExperimentBlueprint,
    execution: ExperimentResult,
    *,
    smoke: bool,
) -> list[ExperimentDefect]:
    if not execution.success or execution.validated_output is None:
        return [
            ExperimentDefect(
                defect_id="runtime" if execution.failure_class == "program" else "source_contract",
                summary=execution.summary[-1200:],
                repair="Fix the named exception or contract mismatch in the preserved complete source.",
            )
        ]
    output = execution.validated_output
    expected_count = min(3, blueprint.sample_size) if smoke else blueprint.sample_size
    expected_conditions = {row.id for row in blueprint.conditions}
    keys = [(row.condition, str(row.unit_id)) for row in output.observations]
    required = {
        (condition, str(unit_id))
        for condition in expected_conditions
        for unit_id in range(expected_count)
    }
    defects: list[ExperimentDefect] = []
    if set(keys) != required or len(keys) != len(required):
        defects.append(
            ExperimentDefect(
                defect_id="source_contract",
                summary="Trusted output does not contain exactly one row for every condition/unit pair.",
                repair="Fix the implementation so every harness invocation returns a valid result.",
            )
        )
    failed = [row for row in output.checks if not row.passed]
    if failed:
        mechanism_names = {
            f"mechanism.{row.id}": row.id for row in blueprint.mechanism_checks
        }
        implementation_failures = [
            row
            for row in failed
            if row.name not in mechanism_names
            or any(
                validation.check_id == mechanism_names[row.name]
                and validation.reference != validation.observed
                for validation in output.validations
            )
        ]
        design_failures = [row for row in failed if row not in implementation_failures]
        mismatch_examples = [
            (
                f"{row.check_id}/{row.condition}/{row.unit_id}: "
                f"reference={row.reference!r}, observed={row.observed!r}"
            )
            for row in output.validations
            if row.reference != row.observed
        ][:4]
        if implementation_failures:
            detail = "; ".join(
                [
                    *(f"{row.name}: {row.detail}" for row in implementation_failures),
                    *mismatch_examples,
                ]
            )
            defects.append(
                ExperimentDefect(
                    defect_id="implementation",
                    summary=("Execution checks failed. " + detail)[:1200],
                    repair=(
                        "Correct the named condition, fixture, or independent reference logic using "
                        "the reported values; do not weaken the registered assertion."
                    ),
                )
            )
        if design_failures:
            detail = "; ".join(f"{row.name}: {row.detail}" for row in design_failures)
            defects.append(
                ExperimentDefect(
                    defect_id="conditions",
                    summary=(
                        "Mechanism fixture results are scientifically correct, but the registered "
                        "metric assertion is false or nondiscriminating. " + detail
                    )[:1200],
                    repair=(
                        "Revise the frozen mechanism fixture/typed comparison to directly test feature "
                        "activation without asserting sample-dependent performance or an arbitrary count."
                    ),
                )
            )
    return defects


def _replication_defects(
    blueprint: ExperimentBlueprint,
    first: ExperimentResult,
    second: ExperimentResult,
) -> list[ExperimentDefect]:
    left = first.validated_output
    right = second.validated_output
    assert left is not None and right is not None
    deterministic_metrics = {
        row.id
        for row in blueprint.metrics
        if row.deterministic and row.id != "harness_wall_seconds"
    }

    def records(output: ExperimentOutput) -> dict[tuple[str, str], tuple[Any, dict[str, Any]]]:
        return {
            (row.condition, json.dumps(row.unit_id, sort_keys=True)): (
                row.result,
                {key: value for key, value in row.metrics.items() if key in deterministic_metrics},
            )
            for row in output.observations
        }

    differences: list[str] = []
    if left.parameters.get("unit_sha256") != right.parameters.get("unit_sha256"):
        differences.append("generated units changed")
    if records(left) != records(right):
        differences.append("condition results or deterministic metrics changed")
    left_validation = [
        (row.check_id, row.condition, str(row.unit_id), row.reference, row.observed)
        for row in left.validations
    ]
    right_validation = [
        (row.check_id, row.condition, str(row.unit_id), row.reference, row.observed)
        for row in right.validations
    ]
    if left_validation != right_validation:
        differences.append("reference validations changed")
    if not differences:
        return []
    return [
        ExperimentDefect(
            defect_id="reproducibility",
            summary="Independent full replication disagreed: " + "; ".join(differences),
            repair="Remove uncontrolled randomness or mutable global state while preserving the frozen seeds.",
        )
    ]


def _criterion_results(
    expected: dict[str, str],
    review: ExperimentEvidenceReview,
) -> list[CriterionResult]:
    rows = {row.criterion_id: row for row in review.criteria}
    return [
        CriterionResult(
            criterion=text,
            satisfied=rows[criterion_id].satisfied,
            detail=rows[criterion_id].detail,
        )
        for criterion_id, text in expected.items()
    ]


def _blueprint_sha(blueprint: ExperimentBlueprint) -> str:
    payload = json.dumps(blueprint.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _work_context(item: WorkItem) -> dict[str, Any]:
    return {
        "title": item.title,
        "instruction": item.instruction,
        "hypothesis": item.hypothesis,
        "falsification_criterion": item.falsification_criterion,
        "success_criteria": item.success_criteria,
    }


def _memory_to_mb(value: str) -> int:
    units = {
        "b": 1 / (1024 * 1024), "k": 1 / 1024, "kb": 1 / 1024,
        "m": 1, "mb": 1, "g": 1024, "gb": 1024,
        "t": 1024 * 1024, "tb": 1024 * 1024,
    }
    text = value.strip().lower().replace(" ", "")
    suffix = next((unit for unit in sorted(units, key=len, reverse=True) if text.endswith(unit)), "")
    if not suffix:
        raise ValueError(f"unsupported memory value: {value}")
    amount = float(text[: -len(suffix)])
    return max(1, int(amount * units[suffix]))


def _dry_blueprint(
    item: WorkItem,
    max_wall: int,
    max_memory: int,
    max_cpus: float,
) -> ExperimentBlueprint:
    return ExperimentBlueprint(
        title="Dry-run trusted-harness comparison",
        hypothesis=item.hypothesis,
        null_outcome="The registered treatment and baseline have equal measured values.",
        experimental_unit="one deterministic integer fixture",
        result_type="integer",
        result_description="The exact integer value returned for one fixture.",
        conditions=[
            ExperimentConditionSpec(
                id="treatment",
                role="treatment",
                description="Deterministic treatment implementation.",
                implementation_requirements=["Return the fixture value."],
            ),
            ExperimentConditionSpec(
                id="baseline",
                role="baseline",
                description="Deterministic baseline implementation.",
                implementation_requirements=["Return the fixture value."],
            ),
        ],
        metrics=[
            ExperimentMetricSpec(
                id="operation_count",
                description="Counted primitive operations.",
                value_type="integer",
                role="mechanism",
            )
        ],
        analyses=[
            ExperimentAnalysisSpec(
                id="mean_difference",
                description="Treatment minus baseline mean operation count.",
                operation="paired_mean_difference",
                metric_id="operation_count",
                condition_id="treatment",
                baseline_condition_id="baseline",
            )
        ],
        reference=ExperimentReferenceSpec(
            id="exact_reference",
            kind="exact_reference",
            description="Direct fixture value used as an independent exact reference.",
        ),
        mechanism_checks=[
            ExperimentMechanismCheckSpec(
                id="treatment_dispatch",
                description="Treatment executes the registered operation on a fixed fixture.",
                condition_id="treatment",
                metric_id="operation_count",
                comparison="equal",
                threshold=1,
                fixture_description="Use integer one as a deterministic treatment fixture.",
            ),
            ExperimentMechanismCheckSpec(
                id="baseline_dispatch",
                description="Baseline executes the registered operation on a fixed fixture.",
                condition_id="baseline",
                metric_id="operation_count",
                comparison="equal",
                threshold=1,
                fixture_description="Use integer one as a deterministic baseline fixture.",
            ),
        ],
        sample_size=3,
        seeds=[0],
        generation_plan="Use the index itself as the deterministic fixture value.",
        decision_rule=ExperimentDecisionSpec(
            clauses=[
                ExperimentDecisionClause(
                    analysis_id="mean_difference",
                    comparison="absolute_at_most",
                    threshold=0.0,
                )
            ],
            combine="all",
            outcome_when_met="characterizes",
            outcome_otherwise="characterizes",
            interpretation="Characterize the trusted signed mean operation-count difference.",
        ),
        mechanism_checks_required=True,
        wall_seconds=min(30, max_wall),
        memory_mb=min(512, max_memory),
        cpus=min(1.0, max_cpus),
        limitations=["Dry run only; no scientific execution."],
    )


def _dry_study_program(blueprint: ExperimentBlueprint) -> ExperimentProgram:
    return ExperimentProgram(
        description=f"Dry study module for {blueprint.title}",
        interface="study_v1",
        seeds=blueprint.seeds,
        source=(
            "def make_unit(index, seed):\n"
            "    return {'value': index, 'seed': seed}\n\n"
            "def run_condition(condition_id, unit):\n"
            "    return {'result': unit['value'], 'metrics': {'operation_count': 1}}\n\n"
            "def reference_result(unit):\n"
            "    return unit['value']\n\n"
            "def make_mechanism_fixture(check_id):\n"
            "    return {'value': 1, 'seed': 0}\n"
        ),
    )


def _scientific_review_markdown(
    blueprint: ExperimentBlueprint,
    review: ExperimentEvidenceReview,
) -> str:
    lines = [
        f"# Independent evidence review: {blueprint.title}",
        "",
        review.scientific_summary,
        "",
        f"- Evidence grade: **{review.usable}**",
        f"- Hypothesis-relative outcome: **{review.outcome}**",
        "- Full run was independently repeated; deterministic records matched.",
        "- This is bounded empirical evidence, not a proof or asymptotic claim.",
        "",
        "## Criteria",
        "",
    ]
    lines.extend(
        f"- `{row.criterion_id}`: {'pass' if row.satisfied else 'fail'} — {row.detail}"
        for row in review.criteria
    )
    if review.caveats:
        lines.extend(["", "## Caveats", "", *[f"- {value}" for value in review.caveats]])
    if review.follow_up:
        lines.extend(["", "## Follow-up", "", *[f"- {value}" for value in review.follow_up]])
    return "\n".join(lines).rstrip() + "\n"
