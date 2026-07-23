"""Durable experiment pipeline.

One engine cycle drives protocol design, implementation, smoke execution, full execution, and
scientific review as far as the model-call/resource budget permits. Every transition is persisted,
so process interruption loses at most the current transition. Repair limits are per repeated defect,
not a large global counter.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from ..agents.experiment import ExperimentAgent
from ..artifact_store import ArtifactStore
from ..llm import LLMRouter, ModelBudgetExceeded, StructuredLLMError
from ..schemas import (
    ArtifactRef,
    CriterionResult,
    EvidenceStrength,
    ExperimentConclusion,
    ExperimentCriterionAssessment,
    ExperimentEvidenceReview,
    ExperimentImplementationAudit,
    ExperimentObservation,
    ExperimentOutput,
    ExperimentProgram,
    ExperimentProvenanceAudit,
    ExperimentProtocol,
    ExperimentProtocolReview,
    ExperimentState,
    Finding,
    FindingPolarity,
    FindingStatus,
    NamedDescription,
    WorkItem,
    WorkKind,
    WorkResult,
    utc_now,
)
from ..workflow import MAX_EXPERIMENT_SOURCE_CHARS, _validate_experiment_program


class ExperimentPipeline:
    """Run one experiment requirement without exposing engineering stages as research cycles."""

    MAX_TRANSITIONS = 32
    MAX_IDENTICAL_REPAIRS = 5
    MAX_IDENTICAL_CANDIDATES = 2

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
        refs: dict[str, ArtifactRef] = {}
        initial_protocol_revision = state.protocol_revision
        initial_program_revision = state.program_revision

        def add(ref: ArtifactRef) -> None:
            refs[ref.path] = ref

        def persist(step_dir: str) -> None:
            state.updated_at = utc_now()
            add(self.store.write_json(state_path, state))
            add(self.store.write_json(f"{step_dir}/experiment_state.json", state))

        for transition in range(self.MAX_TRANSITIONS):
            step_dir = f"{run_dir}/experiment_steps/{transition + 1:02d}_{state.stage}"
            result = self._advance(
                item, state, step_dir, persist, add, research_context or {}
            )
            if result is not None:
                for ref in result.artifact_refs:
                    add(ref)
                result.artifact_refs = list(refs.values())
                return result
            # Yield only while another repair is still needed.  If the second repair passed,
            # continue directly into review/execution instead of wasting a whole research cycle.
            repairs_exhausted = (
                state.stage == "protocol_revision"
                and state.protocol_revision - initial_protocol_revision >= 2
            ) or (
                state.stage == "program_revision"
                and state.program_revision - initial_program_revision >= 2
            )
            if repairs_exhausted:
                return WorkResult(
                    work_id=item.work_id,
                    outcome="partial",
                    failure_class="none",
                    attempt_class="engineering",
                    continue_work=True,
                    summary=(
                        "Experiment paused after two bounded repairs; the preserved next stage will "
                        "resume in a later fair scheduling cycle."
                    ),
                    artifact_refs=list(refs.values()),
                    errors=[state.last_error] if state.last_error else [],
                )

        state.engineering_blocked = True
        state.last_error = "Experiment exceeded 32 deterministic stage transitions."
        persist(f"{run_dir}/experiment_steps/limit")
        return WorkResult(
            work_id=item.work_id,
            outcome="failed",
            failure_class="engineering",
            attempt_class="engineering",
            summary="Experiment stage machine failed to converge.",
            artifact_refs=list(refs.values()),
            errors=[state.last_error],
        )

    def _load_state(self, path: str, item: WorkItem) -> ExperimentState:
        if self.store.exists(path):
            return ExperimentState.model_validate(self.store.read_json(path))
        return ExperimentState(work_id=item.work_id)

    def _advance(
        self,
        item: WorkItem,
        state: ExperimentState,
        step_dir: str,
        persist: Any,
        add: Any,
        research_context: dict[str, Any],
    ) -> WorkResult | None:
        if state.engineering_blocked:
            persist(step_dir)
            return self._blocked(item, state)

        settings = self.router.experimenter
        if not self.router.dry_run and (settings is None or not settings.enabled):
            state.engineering_blocked = True
            state.last_error = "The Docker experimenter is not configured and enabled."
            persist(step_dir)
            return self._blocked(item, state, "Experiment infrastructure is unavailable.")

        max_wall = settings.timeout_seconds if settings else 600
        max_memory = _memory_to_mb(settings.memory) if settings else 4096
        max_cpus = settings.cpus if settings else 2.0

        if state.stage in {"protocol_design", "protocol_revision"}:
            revision = state.stage == "protocol_revision"
            messages = self._protocol_messages(
                item,
                state,
                revision=revision,
                max_wall=max_wall,
                max_memory=max_memory,
                max_cpus=max_cpus,
                research_context=research_context,
            )
            add(self.store.write_json(f"{step_dir}/input.json", {"messages": messages}))
            try:
                protocol = self.router.complete_structured(
                    task_type="experiment_protocol",
                    messages=messages,
                    schema=ExperimentProtocol,
                    mock_output=(
                        _dry_protocol(item, max_wall, max_memory, max_cpus)
                        if self.router.dry_run
                        else None
                    ),
                    temperature=0.1,
                    max_tokens=4096,
                    allow_repair=True,
                )
            except (ModelBudgetExceeded, StructuredLLMError):
                persist(step_dir)
                raise
            except Exception as exc:
                state.stage = "protocol_revision"
                return self._failure(
                    item,
                    state,
                    step_dir,
                    persist,
                    f"Protocol generation failed: {type(exc).__name__}: {exc}",
                )

            resource_errors = _resource_errors(
                protocol, max_wall=max_wall, max_memory=max_memory, max_cpus=max_cpus
            )
            payload = json.dumps(protocol.model_dump(mode="json"), sort_keys=True)
            candidate_sha = hashlib.sha256(payload.encode()).hexdigest()
            if revision and candidate_sha == state.last_protocol_candidate_sha256:
                # Reassess an unchanged candidate. A prior defect may have been a malformed review,
                # an over-strict deterministic gate fixed by a software update, or an infrastructure
                # issue. If the defect is genuine, review will return it again and the ordinary stable
                # defect budget still stops the loop; blocking before review makes recovery impossible.
                state.repeated_protocol_candidates += 1
            else:
                state.repeated_protocol_candidates = 0
            state.protocol = protocol
            state.last_protocol_candidate_sha256 = candidate_sha
            state.protocol_revision += int(revision)
            add(self.store.write_json(f"{step_dir}/protocol.json", protocol))
            if resource_errors:
                state.stage = "protocol_revision"
                return self._failure(
                    item, state, step_dir, persist, "; ".join(resource_errors)
                )
            state.stage = "protocol_review"
            state.last_error = ""
            persist(step_dir)
            return None

        if state.stage == "protocol_review":
            assert state.protocol is not None
            criteria = _protocol_criteria()
            messages = self._protocol_review_messages(
                item, state.protocol, criteria, research_context=research_context
            )
            add(self.store.write_json(f"{step_dir}/input.json", {"messages": messages}))
            protocol_review = self.router.complete_structured(
                task_type="experiment_review",
                messages=messages,
                schema=ExperimentProtocolReview,
                mock_output=(
                    ExperimentProtocolReview(
                        criteria=[
                            ExperimentCriterionAssessment(
                                criterion_id=key,
                                satisfied=True,
                                detail="Dry-run deterministic acceptance.",
                            )
                            for key in criteria
                        ]
                    )
                    if self.router.dry_run
                    else None
                ),
                temperature=0.1,
                max_tokens=3072,
                allow_repair=True,
            )
            structural_errors = _protocol_review_shape_errors(criteria, protocol_review)
            if structural_errors and not self.router.dry_run:
                # Missing/duplicate gates and false bits without an actionable repair are reviewer
                # failures, not defects in the scientific protocol. Retry once with the same frozen
                # protocol rather than mutating it and starting a no-op repair loop.
                retry_messages = [
                    {
                        "role": "system",
                        "content": (
                            messages[0]["content"]
                            + " Your previous response was structurally invalid. Return exactly eight "
                            "rows, one for each supplied ID, in the supplied order. Use at most two "
                            "sentences per detail. Every false row must contain `Repair:`."
                        ),
                    },
                    messages[1],
                ]
                protocol_review = self.router.complete_structured(
                    task_type="experiment_review",
                    messages=retry_messages,
                    schema=ExperimentProtocolReview,
                    temperature=0.0,
                    max_tokens=3072,
                    allow_repair=True,
                )
                structural_errors = _protocol_review_shape_errors(
                    criteria, protocol_review
                )
            if structural_errors:
                persist(step_dir)
                raise StructuredLLMError(
                    "Protocol reviewer did not assess the fixed gates: "
                    + "; ".join(structural_errors)
                )
            state.protocol_review = protocol_review
            add(self.store.write_json(f"{step_dir}/review.json", protocol_review))
            errors = [
                *_review_errors(criteria, protocol_review),
                *_protocol_semantic_errors(item, state.protocol),
            ]
            if errors:
                state.stage = "protocol_revision"
                return self._failure(
                    item, state, step_dir, persist, "; ".join(errors)
                )
            payload = json.dumps(state.protocol.model_dump(mode="json"), sort_keys=True)
            state.protocol_sha256 = hashlib.sha256(payload.encode()).hexdigest()
            state.stage = "program_design"
            self._clear_failure(state)
            persist(step_dir)
            return None

        if state.stage in {"program_design", "program_revision"}:
            assert state.protocol is not None and state.protocol_sha256
            revision = state.stage == "program_revision"
            repair_defect = state.repair_base_error or state.last_error
            try:
                if self.router.dry_run:
                    program = _dry_program(item, state.protocol)
                else:
                    plan_messages = self._implementation_plan_messages(
                        item, state, revision=revision
                    )
                    add(
                        self.store.write_json(
                            f"{step_dir}/plan_input.json", {"messages": plan_messages}
                        )
                    )
                    implementation_plan = self.router.complete_text(
                        task_type=(
                            "experiment_debug" if revision else "experiment_implementation"
                        ),
                        messages=plan_messages,
                        temperature=0.2,
                        max_tokens=8192,
                    )
                    add(
                        self.store.write_text(
                            f"{step_dir}/implementation_plan.txt",
                            implementation_plan.rstrip() + "\n",
                        )
                    )
                    messages = self._program_messages(
                        item,
                        state,
                        revision=revision,
                        research_context=research_context,
                        implementation_plan=implementation_plan,
                    )
                    add(
                        self.store.write_json(
                            f"{step_dir}/input.json", {"messages": messages}
                        )
                    )
                    # A reasoning pass diagnoses the algorithm; the non-thinking coding profile then
                    # emits a compact complete file instead of spending its output budget thinking.
                    source = self.router.complete_text(
                        task_type="experiment_revision" if revision else "experiment_design",
                        messages=messages,
                        temperature=0.2 if revision else 0.1,
                        max_tokens=12288,
                    )
                    program = ExperimentProgram(
                        description=f"Executable implementation of {state.protocol.title}",
                        source=source,
                        seeds=state.protocol.seeds,
                    )
                candidate_sha = hashlib.sha256(program.python_code.encode()).hexdigest()
                if revision and candidate_sha == state.last_program_candidate_sha256:
                    state.repeated_program_candidates += 1
                    state.program = program
                    state.program_revision += 1
                    state.stage = "program_revision"
                    add(self.store.write_json(f"{step_dir}/invalid_program.json", program))
                    return self._program_candidate_failure(
                        item,
                        state,
                        step_dir,
                        persist,
                        (
                            "Program revision made no source change; independently reimplement the "
                            "frozen protocol. Underlying defect: "
                            + _underlying_program_defect(repair_defect)
                        ),
                        candidate_score=1,
                        force_block=(
                            state.repeated_program_candidates >= self.MAX_IDENTICAL_CANDIDATES
                        ),
                    )
                state.last_program_candidate_sha256 = candidate_sha
                state.repeated_program_candidates = 0
                _validate_experiment_program(program)
            except (ModelBudgetExceeded, StructuredLLMError):
                persist(step_dir)
                raise
            except Exception as exc:
                candidate = locals().get("program")
                if isinstance(candidate, ExperimentProgram):
                    state.program = candidate
                    add(self.store.write_json(f"{step_dir}/invalid_program.json", candidate))
                state.stage = "program_revision"
                state.program_revision += int(revision)
                return self._program_candidate_failure(
                    item,
                    state,
                    step_dir,
                    persist,
                    f"Program validation failed: {type(exc).__name__}: {exc}",
                    candidate_score=1 if isinstance(candidate, ExperimentProgram) else 0,
                )
            state.program = program
            state.provenance_audit = None
            if state.repair_base_program is None:
                state.repair_base_program = program.model_copy(deep=True)
                state.repair_base_score = 1
            state.implementation_audit = None
            state.program_revision += int(revision)
            state.stage = "program_review" if self.router.dry_run else "smoke_execution"
            # Keep the previous defect counter until smoke execution actually passes. Otherwise a
            # newly generated but still broken program resets the retry budget forever.
            add(self.store.write_json(f"{step_dir}/program.json", program))
            persist(step_dir)
            return None

        if state.stage == "program_review":
            # Deterministic syntax/safety validation, bounded smoke execution, and the final evidence
            # audit are the useful gates. A second subjective pre-execution code review caused repair
            # loops without adding evidence, so this legacy persisted stage now advances directly.
            assert state.protocol is not None and state.program is not None
            if self.router.dry_run:
                final_result = WorkResult(
                    work_id=item.work_id,
                    outcome="partial",
                    failure_class="none",
                    attempt_class="engineering",
                    summary="Dry run validated protocol and executable code generation.",
                )
                state.final_result = final_result
                state.stage = "complete"
                persist(step_dir)
                return final_result
            state.stage = "smoke_execution"
            self._clear_failure(state)
            persist(step_dir)
            return None

        agent = ExperimentAgent(self.store, self.router.experimenter)
        if state.stage == "smoke_execution":
            assert state.protocol is not None and state.program is not None
            try:
                execution = agent.run_program(
                    program=state.program,
                    name=f"{item.title}_smoke",
                    mode="smoke",
                    timeout_seconds=min(60, state.protocol.wall_seconds),
                )
            except Exception as exc:
                return self._failure(
                    item,
                    state,
                    step_dir,
                    persist,
                    f"Smoke infrastructure failed: {type(exc).__name__}: {exc}",
                )
            state.smoke_result = execution
            for ref in execution.artifact_refs:
                add(ref)
            errors = _execution_errors(execution, protocol=state.protocol, smoke=True)
            if errors:
                state.stage = (
                    "smoke_execution"
                    if execution.failure_class == "infrastructure"
                    else "program_revision"
                )
                return self._program_candidate_failure(
                    item,
                    state,
                    step_dir,
                    persist,
                    "; ".join(errors),
                    candidate_score=_execution_candidate_score(execution),
                )
            if state.repair_base_score < 1_000:
                state.repair_base_program = state.program.model_copy(deep=True)
                state.repair_base_score = 1_000
                state.repair_base_error = ""
            state.stage = "full_execution"
            # A repair may pass tiny smoke data and reproduce the same defect only at full scale.
            # Keep its signature until the full run succeeds so that repetition remains bounded.
            state.last_error = ""
            persist(step_dir)
            return None

        if state.stage == "full_execution":
            assert state.protocol is not None and state.program is not None
            try:
                execution = agent.run_program(
                    program=state.program,
                    name=item.title,
                    mode="full",
                    timeout_seconds=state.protocol.wall_seconds,
                )
            except Exception as exc:
                return self._failure(
                    item,
                    state,
                    step_dir,
                    persist,
                    f"Full execution infrastructure failed: {type(exc).__name__}: {exc}",
                )
            state.execution_result = execution
            for ref in execution.artifact_refs:
                add(ref)
            errors = _execution_errors(execution, protocol=state.protocol, smoke=False)
            if errors:
                state.stage = (
                    "full_execution"
                    if execution.failure_class == "infrastructure"
                    else "program_revision"
                )
                return self._program_candidate_failure(
                    item,
                    state,
                    step_dir,
                    persist,
                    "; ".join(errors),
                    candidate_score=_execution_candidate_score(execution) + 1_000,
                )
            state.repair_base_program = state.program.model_copy(deep=True)
            state.repair_base_score = 10_000
            state.repair_base_error = ""
            state.scientific_attempts += 1
            state.stage = "evidence_review"
            self._clear_failure(state)
            persist(step_dir)
            return None

        if state.stage == "evidence_review":
            assert state.protocol is not None
            assert state.program is not None
            assert state.execution_result is not None
            execution = state.execution_result
            if state.implementation_audit is None:
                implementation_audit = self._audit_implementation(
                    item,
                    state.protocol,
                    state.program,
                    execution,
                    research_context=research_context,
                )
                state.implementation_audit = implementation_audit
                add(
                    self.store.write_json(
                        f"{step_dir}/implementation_audit.json",
                        implementation_audit,
                    )
                )
                audit_errors = _implementation_audit_errors(
                    state.protocol, implementation_audit
                )
                if audit_errors:
                    state.stage = "program_revision"
                    state.repair_base_program = state.program.model_copy(deep=True)
                    state.repair_base_score = 10_000
                    state.repair_base_error = "; ".join(audit_errors)
                    return self._failure(
                        item,
                        state,
                        step_dir,
                        persist,
                        state.repair_base_error,
                    )
                persist(step_dir)
            if state.provenance_audit is None:
                provenance_audit = self._audit_validation_provenance(
                    item,
                    state.protocol,
                    state.program,
                    research_context=research_context,
                )
                state.provenance_audit = provenance_audit
                add(
                    self.store.write_json(
                        f"{step_dir}/provenance_audit.json", provenance_audit
                    )
                )
                provenance_errors = _provenance_audit_errors(provenance_audit)
                if provenance_errors:
                    state.stage = "program_revision"
                    state.repair_base_program = state.program.model_copy(deep=True)
                    state.repair_base_score = 10_000
                    state.repair_base_error = "; ".join(provenance_errors)
                    return self._failure(
                        item,
                        state,
                        step_dir,
                        persist,
                        state.repair_base_error,
                    )
                persist(step_dir)
            evidence_review = self._review_evidence(
                item,
                state.protocol,
                state.program,
                execution,
                research_context=research_context,
            )
            expected = {
                f"W{index:02d}": criterion
                for index, criterion in enumerate(item.success_criteria, 1)
            }
            assert execution.validated_output is not None
            missing = _reconcile_evidence_review(
                evidence_review, execution.validated_output, expected
            )
            add(self.store.write_json(f"{step_dir}/review.json", evidence_review))
            rows = _criterion_results(expected, evidence_review.criteria)
            if evidence_review.usable == "unusable":
                # The audit sees the frozen protocol, complete execution summary, source digest,
                # and a source excerpt.  Its concrete defect therefore belongs to this durable
                # implementation strategy: repair the preserved complete source in place instead
                # of throwing it away and asking a fresh strategy to rediscover the same fix.
                # Repeated/no-op and cumulative revision limits still bound an irreparable design.
                defects = list(
                    dict.fromkeys(
                        [*evidence_review.issues, *evidence_review.follow_up]
                    )
                )
                state.final_result = None
                state.stage = "program_revision"
                state.repair_base_program = state.program.model_copy(deep=True)
                state.repair_base_score = 10_000
                state.repair_base_error = (
                    "; ".join(defects)
                    or "Scientific audit found unusable measurements: "
                    + evidence_review.scientific_summary
                )
                return self._failure(
                    item,
                    state,
                    step_dir,
                    persist,
                    state.repair_base_error,
                )
            full = evidence_review.usable == "full"
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
                    f"Protocol `{state.protocol.title}`; {state.protocol.sample_size} "
                    f"independent units per condition; seeds {state.protocol.seeds}."
                ),
                evidence_refs=execution.artifact_refs,
                source_ids=[execution.run_id],
                caveats=[*execution.caveats, *evidence_review.caveats],
            )
            final_result = WorkResult(
                work_id=item.work_id,
                outcome="done" if full else "partial",
                failure_class="none" if full else "method",
                attempt_class="scientific",
                evidence_level="substantive" if full else "preliminary",
                requirement_satisfied=full,
                criteria=rows,
                summary=execution.summary,
                findings=[finding],
                errors=evidence_review.issues if not full else [],
                next_steps=evidence_review.follow_up,
            )
            state.final_result = final_result
            state.stage = "complete"
            persist(step_dir)
            return final_result

        if state.stage == "complete":
            persist(step_dir)
            if state.final_result is not None:
                return state.final_result.model_copy(deep=True)
            return WorkResult(
                work_id=item.work_id,
                outcome="failed",
                failure_class="engineering",
                attempt_class="engineering",
                summary="Experiment state completed without a durable final result.",
                errors=["Reset or inspect the preserved experiment state."],
            )

        state.engineering_blocked = True
        state.last_error = f"Unknown experiment stage: {state.stage}"
        persist(step_dir)
        return self._blocked(item, state)

    def _program_candidate_failure(
        self,
        item: WorkItem,
        state: ExperimentState,
        step_dir: str,
        persist: Any,
        error: str,
        *,
        candidate_score: int,
        force_block: bool = False,
    ) -> WorkResult | None:
        """Keep the deepest runnable source when a complete-file replacement regresses."""
        candidate = state.program
        base = state.repair_base_program
        if candidate is not None and (base is None or candidate_score > state.repair_base_score):
            state.repair_base_program = candidate.model_copy(deep=True)
            state.repair_base_score = candidate_score
            state.repair_base_error = error
        elif base is not None:
            candidate_error = error
            state.program = base.model_copy(deep=True)
            base_error = state.repair_base_error or "Repair the preserved scientific defect."
            error = (
                "Replacement candidate regressed before resolving the preserved base. "
                f"Candidate defect: {candidate_error}; Preserved base defect: {base_error}"
            )
        return self._failure(
            item,
            state,
            step_dir,
            persist,
            error,
            force_block=force_block,
        )

    def _failure(
        self,
        item: WorkItem,
        state: ExperimentState,
        step_dir: str,
        persist: Any,
        error: str,
        *,
        force_block: bool = False,
    ) -> WorkResult | None:
        signature = _defect_signature(error)
        if signature == state.last_defect_signature:
            state.repeated_defect_failures += 1
        else:
            state.last_defect_signature = signature
            state.repeated_defect_failures = 1
        # Count the current repeated defect, not every distinct issue encountered while a durable
        # experiment advances through many stages over days or weeks.
        state.engineering_failures = state.repeated_defect_failures
        state.last_error = error[-4000:]
        repair_limit = min(
            self.MAX_IDENTICAL_REPAIRS,
            self.router.core.max_experiment_engineering_retries,
        )
        total_revision_limit = self.router.core.max_experiment_engineering_retries
        if state.program_revision >= total_revision_limit:
            state.last_error = (
                f"Program strategy exhausted {state.program_revision} complete revisions. "
                f"Last defect: {state.last_error}"
            )[-4000:]
            force_block = True
        if state.stage == "protocol_revision" and state.protocol_revision >= total_revision_limit:
            state.last_error = (
                f"Protocol strategy exhausted {state.protocol_revision} complete revisions. "
                f"Last defect: {_underlying_repair_defect(state.last_error)}"
            )[-4000:]
            force_block = True
        if force_block or state.repeated_defect_failures >= repair_limit:
            state.engineering_blocked = True
            persist(step_dir)
            return self._blocked(item, state)
        persist(step_dir)
        return None

    @staticmethod
    def _clear_failure(state: ExperimentState) -> None:
        state.engineering_failures = 0
        state.repeated_defect_failures = 0
        state.last_defect_signature = ""
        state.last_error = ""

    @staticmethod
    def _blocked(
        item: WorkItem,
        state: ExperimentState,
        summary: str = "Experiment repair stopped on a repeated engineering defect.",
    ) -> WorkResult:
        return WorkResult(
            work_id=item.work_id,
            outcome="failed",
            failure_class="engineering",
            attempt_class="engineering",
            summary=summary,
            errors=[state.last_error or "unknown experiment engineering defect"],
            next_steps=["Fix the named infrastructure defect or request a human replan."],
        )

    @staticmethod
    def _protocol_messages(
        item: WorkItem,
        state: ExperimentState,
        *,
        revision: bool,
        max_wall: int,
        max_memory: int,
        max_cpus: float,
        research_context: dict[str, Any],
    ) -> list[dict[str, str]]:
        system = (
            "Design one bounded falsifiable experiment. `conditions` lists every implementation or "
            "group being measured. `baselines` designates a proper subset of those same condition IDs "
            "as comparators; overlap is required by the schema and is not a defect. Prefer the strongest "
            "scientifically valid requested comparator (often the simplest requested method). Never add "
            "a dummy, no-op, deliberately incorrect, or oracle condition merely to make a baseline look "
            "separate. Define `result_semantics` as the actual primary condition output for one unit "
            "(for example SAT/UNSAT), explicitly distinct from execution completion, a no-exception flag, "
            "and performance metrics. Include all dominant costs and requested parameter regimes. Give every statistic "
            "needed by the decision rule (p-values, intervals, effect sizes, or signed differences) a stable "
            "ID in `analysis_metrics`; use the smallest sufficient set (at most eight), and return each in "
            "aggregate_metrics. Correctness checks test implementation validity, never the expected result. "
            "For every condition, add a computed integrity check whose description explicitly names that "
            "condition ID and directly instruments its defining mechanism on a constructed fixture (for "
            "example a feature counter, disabled-feature assertion, or exact branch trace). Never infer feature presence from "
            "runtime, node-count improvement, or ordinary output differences. When implementation "
            "correctness affects evidence, validate EVERY sampled unit against an independent exhaustive "
            "oracle, reference implementation, round-trip, ground truth, or invariant; a few known fixtures "
            "and cross-condition agreement are insufficient. For generated structured objects, register "
            "and validate standard-definition invariants and exclude degenerate samples (for example, a "
            "random k-SAT clause chooses k distinct variables before signs, so it cannot contain both a "
            "variable and its negation). Keep units small enough to make full-sample validation feasible. "
            "A reproducibility check may "
            "require deterministic generated instances, decisions, and operation counts, but never "
            "identical wall-clock timings. `sample_size` is one scalar: the exact total number of "
            "independent units that full mode must execute for EACH condition. It is not the number of "
            "seeds, an input dimension, bit width, list length, or parameter value. Seeds are deterministic "
            "entropy anchors. In `unit_generation`, specify an exact index-based PRNG/hash/stream mapping "
            "that derives `sample_size` UNIQUE inputs and stable unit IDs; never cycle or truncate the seed "
            "list. Put parameter regimes explicitly in condition descriptions and the analysis plan. "
            "Use fixed seeds and a decision rule that is executable with the stated samples. Preserve "
            "negative and null outcomes."
        )
        if revision:
            system += " Revise only the concrete preserved defect; do not return the same protocol."
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "work_item": _program_work_context(item),
                        "runtime_limits": {
                            "wall_seconds": max_wall,
                            "memory_mb": max_memory,
                            "cpus": max_cpus,
                        },
                        "previous_protocol": (
                            state.protocol.model_dump(mode="json") if state.protocol else None
                        ),
                        "defect": state.last_error[:2_500] if revision else "",
                        "research_objective": research_context.get("research_objective", ""),
                        "agenda_constraints": research_context.get("agenda_constraints", []),
                        "requested_deliverables": research_context.get(
                            "agenda_deliverables", []
                        ),
                        "accepted_prior_evidence": research_context.get(
                            "accepted_prior_evidence", []
                        ),
                    },
                    ensure_ascii=False,
                ),
            },
        ]

    @staticmethod
    def _protocol_review_messages(
        item: WorkItem,
        protocol: ExperimentProtocol,
        criteria: dict[str, str],
        *,
        research_context: dict[str, Any],
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "Audit the protocol. Return every supplied criterion ID exactly once, in the supplied "
                    "order, with at most two concise sentences of detail. Do not expose internal deliberation "
                    "or argue with these instructions. In this schema baselines are designated members of "
                    "conditions, so that overlap is required, not a defect. A requested simple method can "
                    "be the valid baseline; reject dummy, "
                    "no-op, knowingly incorrect, or irrelevant controls. Do not demand an extra baseline "
                    "that the scientific comparison does not need. Reject checks that require repeated "
                    "wall-clock measurements to be identical; timing is inherently noisy. Judge the "
                    "literal description, not implications guessed from an ID. Do not search for a flaw "
                    "when a criterion is satisfied and never mark it false while saying it is valid. A "
                    "false detail must end with `Repair:` followed by one concrete imperative change. "
                    "Never infer that a determinism check includes runtime unless its literal description "
                    "names timing. A specification baseline may intentionally compute the same predicate "
                    "as the treatment through separate code; do not demand a different algorithm unless "
                    "the frozen protocol requires one. P_CONDITIONS requires each condition ID to be explicitly "
                    "named by its own direct instrumentation check on a constructed fixture, including assertions "
                    "for features that must remain disabled. P_CHECKS requires an independent oracle, reference, "
                    "round-trip, ground truth, or invariant on every sampled unit when correctness affects evidence; "
                    "known-case output agreement alone is insufficient. A check must not infer feature presence from runtime or favorable/different "
                    "node counts. P_ANALYSIS requires only the minimal stable named outputs sufficient for the "
                    "decision rule."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "work_item": _program_work_context(item),
                        "protocol": protocol.model_dump(mode="json"),
                        "agenda_constraints": research_context.get("agenda_constraints", []),
                        "requested_deliverables": research_context.get(
                            "agenda_deliverables", []
                        ),
                        "criteria": [
                            {"criterion_id": key, "text": text}
                            for key, text in criteria.items()
                        ],
                    },
                    ensure_ascii=False,
                ),
            },
        ]

    @staticmethod
    def _implementation_plan_messages(
        item: WorkItem,
        state: ExperimentState,
        *,
        revision: bool,
    ) -> list[dict[str, str]]:
        assert state.protocol is not None
        prior_program = (
            state.repair_base_program
            if revision and state.repair_base_program is not None
            else state.program
        )
        repair_defect = state.repair_base_error or state.last_error
        return [
            {
                "role": "system",
                "content": (
                    "Write a concise, concrete implementation plan for one frozen experiment. Diagnose "
                    "the exact underlying algorithm or contract defect when revising, name the affected "
                    "function and replacement logic, and explain how the repair will be tested. Plan "
                    "independent tiny correctness oracles, shared condition inputs, bounded smoke/full "
                    "branches, raw measurements, and the exact v2 return shape. Do not emit Python "
                    "source, generic component labels, or placeholder prose, and never weaken a check."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "work_title": item.title,
                        "protocol": _program_protocol_context(state.protocol),
                        "previous_source": (
                            prior_program.python_code
                            if revision
                            and prior_program is not None
                            and len(prior_program.python_code) <= MAX_EXPERIMENT_SOURCE_CHARS
                            else None
                        ),
                        "previous_source_omitted_as_oversize": bool(
                            revision
                            and prior_program is not None
                            and len(prior_program.python_code) > MAX_EXPERIMENT_SOURCE_CHARS
                        ),
                        "defect": repair_defect[:4_000] if revision else "",
                    },
                    ensure_ascii=False,
                ),
            },
        ]

    @staticmethod
    def _program_messages(
        item: WorkItem,
        state: ExperimentState,
        *,
        revision: bool,
        research_context: dict[str, Any],
        implementation_plan: str,
    ) -> list[dict[str, str]]:
        assert state.protocol is not None
        prior_program = (
            state.repair_base_program
            if revision and state.repair_base_program is not None
            else state.program
        )
        repair_defect = state.repair_base_error or state.last_error
        system = (
            "Return only compact raw Python source, with no Markdown fence, JSON wrapper, explanation, "
            "unfinished comments, dead code, or placeholders. Keep the complete source under 20,000 "
            "characters. Define run_experiment(mode: str) -> dict and implement the whole frozen protocol. "
            "Reserve the argument `mode` for the execution mode ('smoke' or 'full') and use names such "
            "as condition_id or solver_variant for experimental conditions. Near the start of that "
            "function, branch explicitly on mode (for example "
            "`sample_count = 1 if mode == 'smoke' else full_sample_count`) and actually use the selected "
            "bound. Smoke must run every condition on at most ten tiny units and finish well under 60 "
            "seconds; full mode must execute exactly protocol.sample_size independent units for each "
            "condition. The fixed seeds are entropy anchors: deterministically derive enough UNIQUE replicate "
            "IDs/seeds to reach that count instead of cycling through the seed list. Return those exact unique "
            "identifiers as the flat `parameters.unit_ids` list. The function must return "
            "this v2 shape: "
            "{'schema_version': 2, 'experiment': str, 'status': 'completed'|'capped', "
            "'parameters': {str: scalar}, 'aggregate_metrics': {str: scalar_or_short_scalar_list} "
            "containing every frozen protocol analysis_metrics ID (an interval stays under its one registered "
            "ID as [lower, upper]), 'observations': [{'condition': str, 'unit_id': str_or_int, "
            "'result': actual_primary_result, 'sample_size': 1, 'metrics': {str: scalar}}], with exactly "
            "one record per condition/unit pair. `result` must follow protocol.result_semantics and can "
            "never be a completion/no-exception flag or performance proxy. "
            "Every condition must use the exact same unique unit IDs in parameters.unit_ids and each "
            "observation must carry its matching ID. Never aggregate replicates, use a cumulative index, "
            "list length, or parameter value as sample_size. The application deterministically checks "
            "record counts and unit identity provenance. "
            "'validations': [{'check_id': str, 'condition': str, 'unit_id': str_or_int, "
            "'reference': scalar_or_flat_object, 'observed': scalar_or_flat_object, 'detail': str}], "
            "'checks': [{'name': str, 'passed': bool, 'detail': str}], "
            "'conclusion': {'hypothesis': str, 'outcome': "
            "'supports'|'contradicts'|'null'|'inconclusive'|'characterizes', "
            "'basis_metrics': [str], 'statement': str}, 'limitations': [str]}. Conclusion outcome "
            "labels refer to the frozen hypothesis, never its null hypothesis: rejecting a no-difference "
            "null supports a 'strategies differ' hypothesis. Every scalar is str, int, float, bool, or None. Parameter values may also be flat lists of "
            "scalars (for seeds or parameter grids); aggregate metrics may use lists of at most four scalar "
            "bounds/components, while observation metrics remain scalar. Do not nest dictionaries. Emit "
            "every protocol correctness-check ID exactly "
            "once as a check name, with one aggregate pass/fail decision and detail; do not suffix IDs by "
            "condition or replicate. Copy the protocol hypothesis verbatim into conclusion.hypothesis. "
            "When conditions are variants of one algorithm, prefer one small shared core parameterized "
            "by explicit feature flags so correctness fixes apply to every variant; feature flags must still "
            "produce the protocol's material differences. Validate the core on both positive and negative "
            "known cases before benchmarking. For algorithmic experiments, use an independent "
            "tiny oracle (such as exhaustive enumeration) to establish known-case answers and compare "
            "all condition outputs on the same units; never declare an UNSAT result correct merely from "
            "an expected node count. Generate named structured objects by their standard definition and "
            "compute registered validity invariants for every unit; do not silently include degenerate "
            "objects. For each correctness check that covers every sampled unit, emit one "
            "validation row for EVERY required condition/unit pair: `reference` is the independently "
            "computed oracle/reference result and `observed` is that condition's actual result. Never copy "
            "a treatment result into the reference field. The trusted gate checks complete coverage and "
            "exact equality. Never alias a treatment as a baseline, "
            "invent an unavailable external solver, hard-code measurements, or mark a check passed "
            "without computing it. Begin with imports, constants, classes, or function definitions. "
            "Use no network, subprocess, async, or multiprocessing; `os` is limited to makedirs, "
            "path.join, and read-only environment access. Available scientific packages "
            "include numpy, pandas, scipy, matplotlib, scikit-learn, statsmodels, sympy, and networkx, "
            "but prefer the standard library when sufficient. Write any explicitly requested CSV, JSON, "
            "table, or plot to a relative path in the current directory. Before returning, verify that "
            "there is no `pass`, guessed result, proxy metric, hard-coded outcome, or prose about work "
            "that the source does not perform."
        )
        if revision:
            independent = state.repeated_program_candidates > 0
            system = (
                "Return only one complete compact replacement Python source file, with no Markdown. "
                "Repair the exact preserved defect while retaining sound parts of the prior source. Do "
                "not disable checks, discard measurements, hard-code outcomes, or change the frozen "
                "protocol to make the run pass. Preserve run_experiment(mode: str) -> dict, the v2 "
                "contract, negative/null outcomes, and requested artifacts. Emit one observation per "
                "condition/unit pair with unit_id and sample_size=1 in both modes. In smoke mode, execute "
                "every condition on at most ten unique units total. Return the exact unique IDs for "
                "executed units in parameters.unit_ids and use that same ID set under every condition. "
                "Keep the complete replacement under 20,000 characters and verify its syntax before "
                "return. The returned dict must have exactly these top-level fields: schema_version=2, "
                "experiment=str, status='completed'|'capped', parameters=dict, aggregate_metrics=dict "
                "containing every frozen analysis_metrics ID with scalar values or short scalar lists for "
                "intervals, observations=list of {condition, unit_id, result, sample_size, metrics}, "
                "validations=list of {check_id, condition, unit_id, reference, observed, detail}, "
                "checks=list of {name, passed, detail}, conclusion={hypothesis, outcome, basis_metrics, "
                "statement}, and limitations=list[str]. For every full-sample correctness check, preserve "
                "one reference/observed validation row per required condition/unit pair. Valid outcomes "
                "are supports, contradicts, null, inconclusive, and characterizes, and they always refer "
                "to the frozen hypothesis rather than its null. Reserve the run_experiment argument `mode` for only "
                "'smoke' or 'full'; use a different variable for algorithm variants."
            )
            if independent:
                system += (
                    " The previous repair was byte-identical and made no progress. Do not copy it. "
                    "Independently reimplement the protocol with a smaller design from the specification."
                )
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "work_item": (
                            {"title": item.title}
                            if revision
                            else _program_work_context(item)
                        ),
                        "protocol": (
                            _program_protocol_context(state.protocol)
                            if revision
                            else state.protocol.model_dump(mode="json")
                        ),
                        "protocol_sha256": state.protocol_sha256,
                        "implementation_plan": implementation_plan[:4_000],
                        "previous_program": (
                            {
                                "description": prior_program.description,
                                "source": prior_program.python_code,
                                "seeds": prior_program.seeds,
                            }
                            if prior_program
                            and len(prior_program.python_code) <= MAX_EXPERIMENT_SOURCE_CHARS
                            else None
                        ),
                        "previous_program_omitted_as_oversize": bool(
                            prior_program
                            and len(prior_program.python_code) > MAX_EXPERIMENT_SOURCE_CHARS
                        ),
                        "defect": repair_defect[:4_000] if revision else "",
                        "agenda_constraints": research_context.get("agenda_constraints", []),
                        "requested_deliverables": research_context.get(
                            "agenda_deliverables", []
                        ),
                        "reusable_code_from_prior_completed_experiment": (
                            []
                            if revision or state.program is not None
                            else research_context.get("reusable_experiment_code", [])
                        ),
                    },
                    ensure_ascii=False,
                ),
            },
        ]

    def _audit_implementation(
        self,
        item: WorkItem,
        protocol: ExperimentProtocol,
        program: ExperimentProgram,
        execution: Any,
        *,
        research_context: dict[str, Any],
    ) -> ExperimentImplementationAudit:
        output = execution.validated_output
        return self.router.complete_structured(
            # This is a bounded source inspection, not open-ended scientific reasoning. The control
            # profile's non-thinking mode reliably emits the required complete JSON instead of spending
            # the entire output budget on hidden deliberation.
            task_type="experiment_review",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Independently audit whether the complete source actually implements every frozen "
                        "condition and whether each condition's mechanism check is discriminating. Return "
                        "every condition ID exactly once. Trace dispatch and state changes through the "
                        "actual source; do not trust check names, passed bits, prose, output agreement, or "
                        "favorable performance. `implemented` is false if any defining feature is missing, "
                        "partial, applied only initially when the protocol requires it after each transition, "
                        "or differs from the frozen algorithm. `discriminating_check` is true only when the "
                        "executed fixture directly observes the defining mechanism and would fail if that "
                        "mechanism were removed or replaced by a baseline path. SAT/UNSAT correctness and "
                        "full-sample oracle agreement do not by themselves validate unit propagation, "
                        "branching policy, preprocessing, or another internal treatment feature. For a "
                        "recursive algorithm, explicitly inspect what happens after recursive branches and "
                        "during backtracking. For unit propagation specifically, do not accept a scan of only "
                        "physically unit input clauses: verify each branch assignment reduces clauses before "
                        "new unit clauses are discovered and propagated to a fixpoint. Independently audit "
                        "validation provenance: trace every full-sample row's `observed` value back to the "
                        "actual condition result and `reference` back to the independent oracle. Mark "
                        "`validation_provenance_sound=false` if code substitutes a completion/success flag, "
                        "proxy metric, copied treatment value, hardcoded expectation, or discards the solver's "
                        "SAT/UNSAT/value result. Do not assume passing rows prove correct wiring. Audit the "
                        "implemented analysis formulas, registered IDs, decision rule, and hypothesis-relative "
                        "outcome under `analysis_implementation_sound`. Also inspect base-case/conflict logic "
                        "and state restoration rather than relying only on the generated oracle check. Name "
                        "exact functions/lines or logic in every negative detail. Do not demand an algorithm "
                        "absent from the literal protocol."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "work_title": item.title,
                            "research_objective": research_context.get(
                                "research_objective", ""
                            ),
                            "requested_deliverables": research_context.get(
                                "agenda_deliverables", []
                            ),
                            "protocol": protocol.model_dump(mode="json"),
                            "source": program.python_code,
                            "validated_output": (
                                _evidence_output_context(output) if output else None
                            ),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            schema=ExperimentImplementationAudit,
            temperature=0.1,
            max_tokens=8192,
            # A formatter cannot reconstruct a semantic source audit and may fabricate generic
            # condition IDs. Let the engine retry the fresh audit as an operational failure.
            allow_repair=False,
        )

    def _audit_validation_provenance(
        self,
        item: WorkItem,
        protocol: ExperimentProtocol,
        program: ExperimentProgram,
        *,
        research_context: dict[str, Any],
    ) -> ExperimentProvenanceAudit:
        return self.router.complete_structured(
            task_type="experiment_evidence_review",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Perform a narrow adversarial data-flow audit of generated experiment source. "
                        "Do not audit style and do not trust returned pass bits. For every registered "
                        "full-sample correctness check, trace `reference` to a genuinely independent "
                        "oracle/reference computation and trace `observed` to the ACTUAL scientific "
                        "condition result. A no-exception/completed/solved-success flag is not a SAT/UNSAT "
                        "decision or computed value. Reject if a function computes a result and then discards "
                        "it, if reference and observed share the same implementation path, if sampled units "
                        "or conditions are skipped, or if a proxy is substituted. Verify every observation "
                        "`result` follows the frozen result_semantics and that oracle-validation `observed` "
                        "is that same result. Hardcoded expected answers on explicitly constructed fixture "
                        "checks are valid; this independent-provenance requirement applies to registered "
                        "full-sample checks. Separately trace every "
                        "registered analysis metric to raw observations and verify the decision rule and "
                        "outcome are relative to the registered hypothesis (not its null). Actively inspect "
                        "base cases, conflict predicates, state restoration, exception handling, and variable "
                        "meaning. Give concrete function/expression-level reasons in `validation_detail` and "
                        "`analysis_detail` regardless of each verdict. Set a soundness bit false for any concrete "
                        "defect and put only additional fatal defects in fatal_issues; never excuse a defect "
                        "because rows happen to pass on this sample. Audit only the literal frozen scope: do "
                        "not demand multiple-testing correction, a different registered statistic, or behavior "
                        "outside the protocol's resource/parameter range."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "work_title": item.title,
                            "research_objective": research_context.get(
                                "research_objective", ""
                            ),
                            "hypothesis": protocol.hypothesis,
                            "result_semantics": protocol.result_semantics,
                            "conditions": [
                                row.model_dump(mode="json")
                                for row in protocol.conditions
                            ],
                            "correctness_checks": [
                                row.model_dump(mode="json")
                                for row in protocol.correctness_checks
                            ],
                            "metrics": [
                                row.model_dump(mode="json") for row in protocol.metrics
                            ],
                            "analysis_metrics": [
                                row.model_dump(mode="json")
                                for row in protocol.analysis_metrics
                            ],
                            "decision_rule": protocol.decision_rule,
                            "source": program.python_code,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            schema=ExperimentProvenanceAudit,
            temperature=0.1,
            max_tokens=12288,
            allow_repair=False,
        )

    def _review_evidence(
        self,
        item: WorkItem,
        protocol: ExperimentProtocol,
        program: ExperimentProgram,
        execution: Any,
        *,
        research_context: dict[str, Any],
    ) -> ExperimentEvidenceReview:
        output = execution.validated_output
        return self.router.complete_structured(
            task_type="experiment_evidence_review",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Audit measurements against the frozen protocol. Assess every supplied work "
                        "criterion ID exactly once and recompute conclusions from observations. Preserve "
                        "sound negative and null outcomes. The complete bounded source is supplied for direct "
                        "audit. The trusted wrapper invoked the validated output in full "
                        "mode; do not infer otherwise from an inert __main__ default in the source. Outcome "
                        "labels always refer to the REGISTERED HYPOTHESIS, never to its null: rejecting a "
                        "no-difference null supports a 'strategies differ' hypothesis, while a significant "
                        "effect in the opposite direction contradicts a directional hypothesis. A work "
                        "strategy label such as bounded comparison or stress test never requests final "
                        "smoke-mode evidence: smoke is an engineering gate and the frozen full run is the "
                        "scientific evidence. Inspect the actual condition dispatch, observation `result` values, correctness checks, analysis, "
                        "and conclusion. The deterministic gate has already required every registered full-sample "
                        "correctness check to preserve a reference/observed row for each required condition/unit "
                        "pair and required exact agreement; verify from source that the reference is independently "
                        "computed rather than copied from a treatment. Repeated or equal measurements across "
                        "distinct conditions are not by themselves evidence of a bug, but conditions that omit a defining frozen feature are "
                        "invalid even if ordinary known cases pass. Require a failed oracle/check or a concrete "
                        "code defect. The deterministic gate has already required one observation per "
                        "condition/unit pair and exact unique unit IDs. Significance tests are required only if the frozen protocol says "
                        "so. Full evidence requires every mandatory criterion; use preliminary for scoped "
                        "interpretable pilots and unusable for wrong metrics, invalid baselines, leakage, "
                        "or failed implementation checks. An executable specification baseline is supposed "
                        "to encode the same mathematical predicate as the treatment; separate implementation "
                        "paths plus independent known-case checks are valid property-test evidence. Never "
                        "demand Boyer-Moore, sorting, or another algorithm unless the literal frozen protocol "
                        "requires it. Distinct condition IDs remain distinct observations even when their "
                        "measured values are equal."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "work_item": {
                                "title": item.title,
                                "hypothesis": item.hypothesis,
                            },
                            "work_criteria": [
                                {"criterion_id": f"W{index:02d}", "text": criterion}
                                for index, criterion in enumerate(item.success_criteria, 1)
                            ],
                            "agenda_constraints": research_context.get(
                                "agenda_constraints", []
                            ),
                            "requested_deliverables": research_context.get(
                                "agenda_deliverables", []
                            ),
                            "protocol": protocol.model_dump(mode="json"),
                            "program": {
                                "description": program.description,
                                "seeds": program.seeds,
                                "source_sha256": hashlib.sha256(
                                    program.python_code.encode("utf-8")
                                ).hexdigest(),
                                "source": program.python_code,
                            },
                            "execution_artifacts": [
                                ref.path for ref in execution.artifact_refs[:30]
                            ],
                            "validated_output": (
                                _evidence_output_context(output) if output else None
                            ),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            schema=ExperimentEvidenceReview,
            allow_repair=False,
        )


def _protocol_semantic_errors(
    item: WorkItem, protocol: ExperimentProtocol
) -> list[str]:
    """Enforce independent anchors and keep result direction out of correctness checks."""
    requires_validation = bool(
        re.search(
            r"(?i)(?:correctness|validat(?:e|ion|ing)|known cases?|oracle)",
            " ".join([item.instruction, *item.success_criteria]),
        )
    )
    check_text = " ".join(check.description for check in protocol.correctness_checks)
    independent = bool(
        re.search(
            r"(?i)(?:known|oracle|ground truth|expected (?:answer|output|value)|exhaustive|"
            r"brute[- ]force|round[- ]?trip|invariant|analytical result|reference implementation|"
            r"edge case)",
            check_text,
        )
    )
    errors: list[str] = []
    missing_condition_integrity: list[str] = []
    for condition in protocol.conditions:
        direct_checks = [
            check.description
            for check in protocol.correctness_checks
            if _check_targets_condition(check, condition.id)
            and re.search(
                r"(?i)(?:instrument|counter|trace|dispatch|feature flag|"
                r"branch(?:ing)? (?:path|choice|variable|decision)|assign(?:ment)?|forced|"
                r"propagat(?:e|ion)|invok(?:e|ed|ation)|call(?:ed| count)|mechanism|"
                r"must (?:remain|be) (?:zero|disabled))",
                check.description,
            )
        ]
        if not direct_checks:
            missing_condition_integrity.append(condition.id)
    if missing_condition_integrity:
        errors.append(
            "P_CONDITIONS: Add a direct computed counter, trace, dispatch, or feature-flag check "
            "that explicitly names each condition (including features that must remain disabled): "
            + ", ".join(missing_condition_integrity)
        )
    if requires_validation and not independent:
        errors.append(
            "P_CHECKS: Add at least one independent known-case, exhaustive-oracle, round-trip, "
            "invariant, or ground-truth correctness check; cross-condition agreement alone can "
            "allow every implementation to be wrong."
        )
    full_sample_validation = any(
        _check_requires_full_sample_evidence(check.description)
        for check in protocol.correctness_checks
    )
    if requires_validation and not full_sample_validation:
        errors.append(
            "P_CHECKS: Validate every sampled unit against an independent oracle, reference, "
            "round-trip, ground truth, or invariant; known fixtures alone do not validate the "
            "scientific measurements."
        )
    directional_ids = [
        check.id
        for check in protocol.correctness_checks
        if re.search(
            r"(?i)(?:outperform|faster|slower|fewer|more nodes|lower (?:cost|runtime|time)|"
            r"higher (?:accuracy|rate)|reduce[sd]? (?:cost|runtime|search|nodes)|"
            r"improve[sd]?|statistically significant|better than|worse than)",
            check.description,
        )
    ]
    if directional_ids:
        errors.append(
            "P_CHECKS: Move expected performance direction out of correctness checks and into the "
            "decision rule; negative and null results must still pass implementation validation: "
            + ", ".join(directional_ids)
        )
    noisy_determinism_ids = [
        check.id
        for check in protocol.correctness_checks
        if _requires_repeated_timing(check.description)
    ]
    if noisy_determinism_ids:
        errors.append(
            "P_CHECKS: Determinism checks may compare generated inputs, outputs, and operation counts "
            "but not wall-clock timings: " + ", ".join(noisy_determinism_ids)
        )
    analysis_text = " ".join(
        f"{metric.id} {metric.description}" for metric in protocol.analysis_metrics
    )
    required_analysis_concepts = [
        (r"(?i)\b(?:confidence interval|\d+%\s*ci|ci\b)", r"(?i)\b(?:confidence interval|ci\b)", "confidence interval"),
        (r"(?i)\bp[- ]?values?\b", r"(?i)\bp[- ]?values?\b", "p-value"),
        (
            r"(?i)\beffect sizes?\b",
            r"(?i)\b(?:effect sizes?|cohen(?:'s)? d|mean difference|signed difference|"
            r"odds ratio|risk ratio)\b",
            "effect size",
        ),
    ]
    for decision_pattern, metric_pattern, label in required_analysis_concepts:
        if re.search(decision_pattern, protocol.decision_rule) and not re.search(
            metric_pattern, analysis_text
        ):
            errors.append(
                f"P_ANALYSIS: The decision rule requires a {label}, but no analysis_metrics ID "
                "names that required statistic."
            )
    generation = protocol.unit_generation
    if protocol.sample_size > len(protocol.seeds) and not re.search(
        r"(?i)\b(?:index|counter|hash|derive|prng|random stream|seedsequence|spawn)\b",
        generation,
    ):
        errors.append(
            "P_SAMPLING: sample_size exceeds the seed-anchor count; unit_generation must give an "
            "explicit index/hash/PRNG-stream derivation rather than cycling or truncating seeds."
        )
    return errors


def _check_targets_condition(check: NamedDescription, condition_id: str) -> bool:
    """Match explicit condition names across conventional `cond_`/`check_` ID prefixes."""
    haystack = f"{check.id} {check.description}".lower()
    if condition_id.lower() in haystack:
        return True
    signature = re.sub(r"^(?:cond(?:ition)?)[_.-]?", "", condition_id.lower())
    normalized = re.sub(r"^(?:cc|check)[_.-]?", "", check.id.lower())
    return bool(signature and signature in normalized)


def _requires_repeated_timing(description: str) -> bool:
    """Detect requirements for equal repeated timings, not mere deterministic timed fixtures."""
    if not re.search(r"(?i)(?:wall[- ]?clock|runtime|timing|elapsed)", description):
        return False
    if re.search(
        r"(?i)(?:exclude[sd]?|omit(?:ted)?|do not (?:compare|check|require)|not (?:used|included|"
        r"compared|required)|may vary|allowed? to vary|descriptive (?:use|purposes?) only|"
        r"not for determinism)",
        description,
    ):
        return False
    return bool(
        re.search(
            r"(?i)(?:(?:identical|exactly (?:equal|the same)|same)\s+(?:wall[- ]?clock|runtime|"
            r"timing|elapsed)|(?:wall[- ]?clock|runtime|timing|elapsed)[^.]{0,80}"
            r"(?:identical|exactly (?:equal|the same)|must match|reproducible))",
            description,
        )
    )


def _protocol_criteria() -> dict[str, str]:
    return {
        "P_ALIGNMENT": "The protocol directly measures the evidence requirement.",
        "P_NULL": "The null outcome and decision rule are explicit and compatible.",
        "P_BASELINES": (
            "At least one scientifically valid comparator is designated in baselines as a member of "
            "conditions; no dummy, knowingly incorrect, or irrelevant control is introduced."
        ),
        "P_CHECKS": "Correctness checks test implementation validity, not result direction.",
        "P_CONDITIONS": (
            "When conditions differ by implementation features, direct counters or traces on constructed "
            "fixtures validate each defining feature; output agreement and performance differences do not."
        ),
        "P_ANALYSIS": (
            "Stable analysis-metric IDs name every statistic needed to execute the decision rule."
        ),
        "P_SAMPLING": "Seeds, sample size, and analysis are reproducible and feasible.",
        "P_COSTS": "Dominant scientific costs and executable resource limits are represented.",
    }


def _provenance_audit_errors(audit: ExperimentProvenanceAudit) -> list[str]:
    errors = list(audit.fatal_issues)
    if not audit.validation_provenance_sound:
        errors.append("Validation provenance: " + audit.validation_detail)
    if not audit.analysis_implementation_sound:
        errors.append("Analysis implementation: " + audit.analysis_detail)
    return list(dict.fromkeys(errors))


def _implementation_audit_errors(
    protocol: ExperimentProtocol, audit: ExperimentImplementationAudit
) -> list[str]:
    expected = {condition.id for condition in protocol.conditions}
    ids = [row.condition_id for row in audit.conditions]
    errors: list[str] = []
    missing = sorted(expected - set(ids))
    unexpected = sorted(set(ids) - expected)
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if missing:
        errors.append("Implementation audit omitted conditions: " + ", ".join(missing))
    if unexpected:
        errors.append(
            "Implementation audit added conditions: " + ", ".join(unexpected)
        )
    if duplicates:
        errors.append(
            "Implementation audit repeated conditions: " + ", ".join(duplicates)
        )
    errors.extend(
        (
            f"Condition `{row.condition_id}` failed mechanism audit "
            f"({', '.join(reason for reason, failed in [('implementation', not row.implemented), ('discriminating check', not row.discriminating_check)] if failed)}): "
            f"{row.detail}"
        )
        for row in audit.conditions
        if not row.implemented or not row.discriminating_check
    )
    if not audit.validation_provenance_sound:
        errors.append(
            "Full-sample validation provenance is unsound: observed/reference values are not "
            "faithfully wired to independent computations."
        )
    if not audit.analysis_implementation_sound:
        errors.append(
            "The implemented statistics, decision rule, or hypothesis-relative conclusion is unsound."
        )
    errors.extend(audit.issues)
    return list(dict.fromkeys(errors))


def _protocol_review_shape_errors(
    expected: dict[str, str], review: ExperimentProtocolReview
) -> list[str]:
    errors = _criterion_id_errors(expected, review.criteria)
    malformed_false = [
        row.criterion_id
        for row in review.criteria
        if not row.satisfied and not re.search(r"(?i)\brepair\s*:", row.detail)
    ]
    if malformed_false:
        errors.append(
            "False assessments without a concrete `Repair:`: "
            + ", ".join(malformed_false)
        )
    return errors


def _review_errors(
    expected: dict[str, str], review: ExperimentProtocolReview
) -> list[str]:
    errors = _criterion_id_errors(expected, review.criteria)
    errors.extend(
        f"{row.criterion_id}: {row.detail}"
        for row in review.criteria
        if not row.satisfied
    )
    return list(dict.fromkeys(error for error in errors if error))


def _criterion_id_errors(
    expected: dict[str, str], assessments: list[ExperimentCriterionAssessment]
) -> list[str]:
    ids = [row.criterion_id for row in assessments]
    errors: list[str] = []
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    missing = sorted(set(expected) - set(ids))
    unexpected = sorted(set(ids) - set(expected))
    if duplicates:
        errors.append("Duplicate criterion IDs: " + ", ".join(duplicates))
    if missing:
        errors.append("Missing criterion IDs: " + ", ".join(missing))
    if unexpected:
        errors.append("Unexpected criterion IDs: " + ", ".join(unexpected))
    return errors


def _reconcile_evidence_review(
    review: ExperimentEvidenceReview,
    output: ExperimentOutput,
    expected: dict[str, str],
) -> list[str]:
    """Derive overall usability from stable criterion rows rather than a contradictory label."""
    missing = _criterion_id_errors(expected, review.criteria)
    unanimous = not missing and all(assessment.satisfied for assessment in review.criteria)
    if unanimous and review.usable == "unusable":
        # Unanimous criterion rows show that the measurements are interpretable, but they do not
        # erase a separate fatal scientific issue (for example a conclusion whose direction is the
        # opposite of its own aggregates). Preserve such runs as preliminary evidence, never as a
        # requirement-closing result.
        object.__setattr__(review, "usable", "preliminary")
        if not review.follow_up:
            object.__setattr__(
                review,
                "follow_up",
                ["Resolve the review's overall scientific issue without discarding the measurements."],
            )
        ExperimentEvidenceReview.model_validate(review.model_dump())
    elif review.usable == "full" and missing:
        review.usable = "preliminary"
        review.issues.extend(missing)
        review.follow_up.append("Assess every work criterion by its exact id.")
    return missing


def _criterion_results(
    expected: dict[str, str], assessments: list[ExperimentCriterionAssessment]
) -> list[CriterionResult]:
    by_id = {row.criterion_id: row for row in assessments}
    return [
        CriterionResult(
            criterion=text,
            satisfied=criterion_id in by_id and by_id[criterion_id].satisfied,
            detail=(
                by_id[criterion_id].detail
                if criterion_id in by_id
                else f"Reviewer omitted {criterion_id}."
            ),
        )
        for criterion_id, text in expected.items()
    ]


def _resource_errors(
    protocol: ExperimentProtocol,
    *,
    max_wall: int,
    max_memory: int,
    max_cpus: float,
) -> list[str]:
    errors: list[str] = []
    if protocol.wall_seconds > max_wall:
        errors.append(f"wall_seconds exceeds {max_wall}")
    if protocol.memory_mb > max_memory:
        errors.append(f"memory_mb exceeds {max_memory}")
    if protocol.cpus > max_cpus:
        errors.append(f"cpus exceeds {max_cpus}")
    return errors


def _execution_candidate_score(execution: Any) -> int:
    """Rank deterministic candidate progress without treating it as research evidence."""
    output = execution.validated_output
    if output is None:
        return 1 if execution.success else 0
    return (
        100
        + sum(check.passed for check in output.checks)
        + min(len(output.aggregate_metrics), 20)
        + min(len(output.validations), 1_000)
    )


def _execution_errors(
    execution: Any, *, protocol: ExperimentProtocol, smoke: bool
) -> list[str]:
    if not execution.success or execution.validated_output is None:
        return [execution.summary]
    output = execution.validated_output
    errors = _protocol_output_errors(protocol, output, smoke=smoke)
    if smoke and output.status != "completed":
        errors.append("Smoke execution reported capped status.")
    failed = [check.name for check in output.checks if not check.passed]
    if failed:
        errors.append("Implementation checks failed: " + ", ".join(failed))
    return list(dict.fromkeys(errors))


def _protocol_output_errors(
    protocol: ExperimentProtocol, output: ExperimentOutput, *, smoke: bool
) -> list[str]:
    """Check frozen-protocol alignment before asking a model to interpret measurements."""
    errors: list[str] = []
    expected_conditions = {condition.id for condition in protocol.conditions}
    observed_ids = [observation.condition for observation in output.observations]
    observed_conditions = set(observed_ids)
    missing_conditions = sorted(expected_conditions - observed_conditions)
    unexpected_conditions = sorted(observed_conditions - expected_conditions)
    if missing_conditions:
        errors.append("Output omitted protocol conditions: " + ", ".join(missing_conditions))
    if unexpected_conditions:
        errors.append("Output added unregistered conditions: " + ", ".join(unexpected_conditions))
    expected_metrics = {metric.id for metric in protocol.metrics}
    for observation in output.observations:
        missing_metrics = sorted(expected_metrics - set(observation.metrics))
        if missing_metrics:
            errors.append(
                f"Observation `{observation.condition}` omitted protocol metrics: "
                + ", ".join(missing_metrics)
            )
    missing_analysis = sorted(
        {metric.id for metric in protocol.analysis_metrics}
        - set(output.aggregate_metrics)
    )
    if missing_analysis:
        errors.append(
            "Output omitted frozen analysis metrics: " + ", ".join(missing_analysis)
        )
    completion_metrics = {
        metric.id
        for metric in protocol.metrics
        if re.search(
            r"(?i)boolean.*true if.*(?:completed|valid result|within (?:the )?(?:time|resource)|"
            r"found .* or proved)",
            metric.description,
        )
    }
    for metric_id in sorted(completion_metrics):
        failed_units = sum(
            observation.sample_size
            for observation in output.observations
            if observation.metrics.get(metric_id) is False
        )
        if failed_units:
            errors.append(
                f"Declared completion metric `{metric_id}` was false for {failed_units} represented "
                "units; do not treat SAT/UNSAT truth as run completion or accept timed-out units as "
                "valid completed measurements."
            )
    represented_by_condition: dict[str, int] = {}
    for condition in expected_conditions & observed_conditions:
        rows = [
            observation
            for observation in output.observations
            if observation.condition == condition
        ]
        aggregated = sum(observation.sample_size != 1 for observation in rows)
        if aggregated:
            errors.append(
                f"Condition `{condition}` has {aggregated} aggregated observation rows; emit one "
                "sample_size=1 record per independent unit."
            )
        represented = sum(observation.sample_size for observation in rows)
        represented_by_condition[condition] = represented
        if smoke and represented > 10:
            errors.append(
                f"Smoke condition `{condition}` represented {represented} units; "
                "the smoke limit is 10."
            )
        if not smoke and represented != protocol.sample_size:
            errors.append(
                f"Full condition `{condition}` represented {represented} units; the frozen "
                f"sample_size requires exactly {protocol.sample_size}."
            )
    expected_units = (
        protocol.sample_size
        if not smoke
        else (max(represented_by_condition.values()) if represented_by_condition else 0)
    )
    unit_ids = output.parameters.get("unit_ids")
    if not isinstance(unit_ids, list):
        errors.append("Output parameters must contain the exact flat `unit_ids` list.")
    else:
        stable_ids = [json.dumps(value, sort_keys=True) for value in unit_ids]
        if any(not isinstance(value, (str, int)) or isinstance(value, bool) for value in unit_ids):
            errors.append("Output unit_ids must contain only string or integer identifiers.")
        if len(unit_ids) != expected_units:
            errors.append(
                f"Output unit_ids has {len(unit_ids)} entries; execution represents "
                f"{expected_units} independent units per condition."
            )
        if len(stable_ids) != len(set(stable_ids)):
            errors.append(
                "Output unit_ids contains duplicates; repeated copies are not independent units."
            )
        expected_id_set = set(stable_ids)
        for condition in sorted(expected_conditions & observed_conditions):
            condition_ids = [
                json.dumps(observation.unit_id, sort_keys=True)
                for observation in output.observations
                if observation.condition == condition
            ]
            if len(condition_ids) != len(set(condition_ids)):
                errors.append(
                    f"Condition `{condition}` repeats observation unit IDs."
                )
            if set(condition_ids) != expected_id_set:
                errors.append(
                    f"Condition `{condition}` observation unit IDs do not exactly match "
                    "parameters.unit_ids."
                )

    expected_checks = {check.id for check in protocol.correctness_checks}
    registered_unit_ids = (
        {json.dumps(value, sort_keys=True) for value in unit_ids}
        if isinstance(unit_ids, list)
        else set()
    )
    full_sample_check_ids = {
        check.id
        for check in protocol.correctness_checks
        if _check_requires_full_sample_evidence(check.description)
    }
    validation_keys: list[tuple[str, str, str]] = []
    for validation in output.validations:
        stable_unit_id = json.dumps(validation.unit_id, sort_keys=True)
        validation_keys.append(
            (validation.check_id, validation.condition, stable_unit_id)
        )
        if validation.check_id not in expected_checks:
            errors.append(
                f"Validation row uses unregistered check `{validation.check_id}`."
            )
        if validation.check_id in full_sample_check_ids:
            if validation.condition not in expected_conditions:
                errors.append(
                    f"Full-sample validation uses unregistered condition `{validation.condition}`."
                )
            if registered_unit_ids and stable_unit_id not in registered_unit_ids:
                errors.append(
                    f"Full-sample validation uses unregistered unit ID `{validation.unit_id}`."
                )
    duplicate_validation_keys = sorted(
        {key for key in validation_keys if validation_keys.count(key) > 1}
    )
    if duplicate_validation_keys:
        errors.append(
            f"Validation evidence repeats {len(duplicate_validation_keys)} check/condition/unit rows."
        )
    for check in protocol.correctness_checks:
        if check.id not in full_sample_check_ids:
            continue
        named_targets = {
            condition.id
            for condition in protocol.conditions
            if condition.id.lower() in check.description.lower()
        }
        target_conditions = named_targets or expected_conditions
        required_keys = {
            (check.id, condition_id, stable_unit_id)
            for condition_id in target_conditions
            for stable_unit_id in registered_unit_ids
        }
        actual_keys = {
            key for key in validation_keys if key[0] == check.id
        }
        missing_keys = required_keys - actual_keys
        if missing_keys:
            errors.append(
                f"Full-sample check `{check.id}` omitted {len(missing_keys)} required "
                "condition/unit validation rows."
            )
        matching_rows = [
            row for row in output.validations if row.check_id == check.id
        ]
        mismatches = [
            row
            for row in matching_rows
            if json.dumps(row.reference, sort_keys=True)
            != json.dumps(row.observed, sort_keys=True)
        ]
        if mismatches:
            errors.append(
                f"Full-sample check `{check.id}` has {len(mismatches)} reference/observed mismatches."
            )
        if re.search(
            r"(?i)(?:oracle|reference implementation|ground truth|exhaustive)",
            check.description,
        ):
            observation_results = {
                (
                    observation.condition,
                    json.dumps(observation.unit_id, sort_keys=True),
                ): observation.result
                for observation in output.observations
            }
            proxy_rows = [
                row
                for row in matching_rows
                if json.dumps(row.observed, sort_keys=True)
                != json.dumps(
                    observation_results.get(
                        (row.condition, json.dumps(row.unit_id, sort_keys=True))
                    ),
                    sort_keys=True,
                )
            ]
            if proxy_rows:
                errors.append(
                    f"Full-sample oracle check `{check.id}` has {len(proxy_rows)} observed values "
                    "that differ from the condition observation result; do not substitute a "
                    "completion flag or proxy."
                )

    check_names = [check.name for check in output.checks]
    actual_checks = set(check_names)
    missing_checks = sorted(expected_checks - actual_checks)
    if missing_checks:
        errors.append("Output omitted protocol correctness checks: " + ", ".join(missing_checks))
    duplicate_checks = sorted(name for name in actual_checks if check_names.count(name) > 1)
    if duplicate_checks:
        errors.append("Output repeated protocol correctness checks: " + ", ".join(duplicate_checks))
    if _normalized_text(output.conclusion.hypothesis) != _normalized_text(protocol.hypothesis):
        errors.append("Output conclusion changed the frozen protocol hypothesis.")
    return errors


def _check_requires_full_sample_evidence(description: str) -> bool:
    return bool(
        re.search(
            r"(?i)(?:every|each|all)(?:\s+of\s+the\s+\d+)?\s+"
            r"(?:sampled |generated |experimental )?"
            r"(?:unit|instance|sample|observation|input|record)s?",
            description,
        )
        and re.search(
            r"(?i)(?:oracle|reference implementation|round[- ]?trip|ground truth|invariant|"
            r"exhaustive)",
            description,
        )
    )


def _underlying_program_defect(error: str) -> str:
    marker = "Underlying defect:"
    return error.rsplit(marker, 1)[-1].strip() if marker in error else error.strip()


def _underlying_repair_defect(error: str) -> str:
    marker = "Protocol repair made no semantic change. Correct the preserved defect:"
    while marker in error:
        error = error.split(marker, 1)[1]
    return error.strip()


def _defect_signature(error: str) -> str:
    """Return a stable category for bounded retries without conflating distinct defects."""
    import re

    first_line = (error.strip().splitlines() or ["unknown experiment defect"])[0].lower()
    first_line = re.sub(r"[0-9a-f]{16,}", "<id>", first_line)
    first_line = re.sub(r"\d+", "<n>", first_line)
    return re.sub(r"\s+", " ", first_line).strip()[:500]


def _evidence_output_context(output: ExperimentOutput) -> dict[str, Any]:
    """Bound raw observations into auditable summaries for the semantic review call.

    The unmodified payload remains in ``results.json``.  This context includes weighted numeric
    summaries, categorical counts, and examples so a long experiment cannot overflow later model
    calls merely by preserving more raw replicates.
    """
    by_condition: dict[str, list[ExperimentObservation]] = {}
    for observation in output.observations:
        by_condition.setdefault(observation.condition, []).append(observation)

    summaries: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for condition, rows in sorted(by_condition.items()):
        metric_names = sorted({name for row in rows for name in row.metrics})[:12]
        metrics: dict[str, Any] = {}
        for name in metric_names:
            values = [
                (row.metrics[name], row.sample_size)
                for row in rows
                if name in row.metrics and row.metrics[name] is not None
            ]
            numeric = [
                (float(value), weight)
                for value, weight in values
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            if numeric and len(numeric) == len(values):
                total_weight = sum(weight for _, weight in numeric)
                metrics[name] = {
                    "count": total_weight,
                    "mean": sum(value * weight for value, weight in numeric) / total_weight,
                    "min": min(value for value, _ in numeric),
                    "max": max(value for value, _ in numeric),
                }
            else:
                counts: dict[str, int] = {}
                for value, weight in values:
                    key = json.dumps(value, ensure_ascii=False, sort_keys=True)
                    counts[key] = counts.get(key, 0) + weight
                metrics[name] = {"counts": dict(list(sorted(counts.items()))[:8])}
        summaries.append(
            {
                "condition": condition,
                "record_count": len(rows),
                "represented_sample_size": sum(row.sample_size for row in rows),
                "metrics": metrics,
            }
        )
        examples.extend(row.model_dump(mode="json") for row in rows[:1])

    return {
        "schema_version": output.schema_version,
        "experiment": output.experiment,
        "status": output.status,
        "parameters": dict(list(output.parameters.items())[:30]),
        "aggregate_metrics": dict(list(output.aggregate_metrics.items())[:40]),
        "observation_summaries": summaries,
        "observation_examples": examples,
        "validation_summary": {
            "row_count": len(output.validations),
            "by_check_condition": {
                f"{check_id}/{condition}": {
                    "count": len(rows),
                    "mismatches": sum(
                        json.dumps(row.reference, sort_keys=True)
                        != json.dumps(row.observed, sort_keys=True)
                        for row in rows
                    ),
                }
                for (check_id, condition), rows in {
                    key: [
                        row
                        for row in output.validations
                        if (row.check_id, row.condition) == key
                    ]
                    for key in {
                        (row.check_id, row.condition)
                        for row in output.validations
                    }
                }.items()
            },
            "examples": [
                row.model_dump(mode="json") for row in output.validations[:6]
            ],
        },
        "checks": [
            {
                "name": row.name,
                "passed": row.passed,
                "detail": row.detail[:500],
            }
            for row in output.checks[:20]
        ],
        "conclusion": output.conclusion.model_dump(mode="json"),
        "limitations": [value[:300] for value in output.limitations[:10]],
        "raw_observation_count": len(output.observations),
    }


def _normalized_text(value: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _program_protocol_context(protocol: ExperimentProtocol) -> dict[str, Any]:
    """Compact the protocol without discarding semantics needed by complete-file repairs."""
    return {
        "title": protocol.title,
        "hypothesis": protocol.hypothesis,
        "null_outcome": protocol.null_outcome,
        "experimental_unit": protocol.experimental_unit,
        "result_semantics": protocol.result_semantics,
        "unit_generation": protocol.unit_generation,
        "conditions": [row.model_dump(mode="json") for row in protocol.conditions],
        "baselines": [row.model_dump(mode="json") for row in protocol.baselines],
        "metrics": [row.model_dump(mode="json") for row in protocol.metrics],
        "analysis_metrics": [
            row.model_dump(mode="json") for row in protocol.analysis_metrics
        ],
        "correctness_checks": [
            row.model_dump(mode="json") for row in protocol.correctness_checks
        ],
        "sample_size": protocol.sample_size,
        "seeds": protocol.seeds,
        "analysis_plan": protocol.analysis_plan,
        "decision_rule": protocol.decision_rule,
        "known_limitations": protocol.known_limitations,
    }


def _program_work_context(item: WorkItem) -> dict[str, Any]:
    """Give code generation only execution-relevant fields, leaving room for full repair source."""
    return {
        "title": item.title,
        "instruction": item.instruction,
        "hypothesis": item.hypothesis,
        "falsification_criterion": item.falsification_criterion,
        "success_criteria": item.success_criteria,
    }


def _memory_to_mb(value: str) -> int:
    import re

    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?)b?\s*", value.lower())
    if not match:
        raise ValueError(f"unsupported experimenter memory value: {value}")
    amount = float(match.group(1))
    factors = {"": 1 / (1024 * 1024), "k": 1 / 1024, "m": 1, "g": 1024, "t": 1024 * 1024}
    return max(1, int(amount * factors[match.group(2)]))


def _dry_protocol(
    item: WorkItem, max_wall: int, max_memory: int, max_cpus: float
) -> ExperimentProtocol:
    return ExperimentProtocol(
        title="Dry-run bounded comparison",
        hypothesis=item.hypothesis,
        null_outcome="No measured difference is observed between treatment and baseline.",
        experimental_unit="one deterministically generated small instance",
        result_semantics=(
            "The actual reconstructed output value for each condition/unit, not execution completion."
        ),
        unit_generation=(
            "For unit index i, derive a unique input from PRNG seed (seed[0] + i), and use "
            "the stable unique unit ID `unit-i`."
        ),
        conditions=[
            NamedDescription(id="treatment", description="Treatment implementation."),
            NamedDescription(id="baseline", description="Strong baseline implementation."),
        ],
        baselines=[
            NamedDescription(id="baseline", description="Strong baseline implementation.")
        ],
        metrics=[NamedDescription(id="measured_value", description="Primary measured value.")],
        analysis_metrics=[
            NamedDescription(id="signed_difference", description="Signed condition difference.")
        ],
        correctness_checks=[
            NamedDescription(
                id="roundtrip",
                description="Round-trip reconstruction succeeds for every sampled experimental unit.",
            ),
            NamedDescription(
                id="treatment_dispatch",
                description="Instrument treatment dispatch and verify its feature flag is invoked.",
            ),
            NamedDescription(
                id="baseline_dispatch",
                description="Instrument baseline dispatch and verify its feature flag remains disabled.",
            ),
        ],
        sample_size=10,
        seeds=[0],
        analysis_plan="Compare condition measurements and preserve every observation.",
        decision_rule="Classify supports, contradicts, or null from the signed difference.",
        wall_seconds=min(30, max_wall),
        memory_mb=min(512, max_memory),
        cpus=min(1.0, max_cpus),
        known_limitations=["Small synthetic smoke-scale experiment."],
    )


def _dry_program(item: WorkItem, protocol: ExperimentProtocol) -> ExperimentProgram:
    metric_id = protocol.metrics[0].id
    output = ExperimentOutput(
        experiment="dry-run protocol implementation",
        parameters={"seed": protocol.seeds[0], "dry_run": True},
        aggregate_metrics={"signed_difference": 0.0},
        observations=[
            ExperimentObservation(
                condition=condition.id,
                unit_id="unit-0",
                result="dry-run-no-result",
                sample_size=1,
                metrics={metric_id: 0.0},
            )
            for condition in protocol.conditions
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
        "    _ = mode\n"
        f"    return json.loads({json.dumps(payload)})\n"
    )
    return ExperimentProgram(
        description=f"Dry-run implementation of {protocol.title}",
        source=source,
        seeds=protocol.seeds,
    )
