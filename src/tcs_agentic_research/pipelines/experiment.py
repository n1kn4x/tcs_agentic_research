"""Durable experiment pipeline.

One engine cycle drives protocol design, implementation, smoke execution, full execution, and
scientific review as far as the model-call/resource budget permits. Every transition is persisted,
so process interruption loses at most the current transition. Repair limits are per repeated defect,
not a large global counter.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..agents.experiment import ExperimentAgent
from ..artifact_store import ArtifactStore
from ..llm import LLMRouter, ModelBudgetExceeded
from ..schemas import (
    ArtifactRef,
    CriterionResult,
    EvidenceStrength,
    ExperimentConclusion,
    ExperimentCriterionAssessment,
    ExperimentEvidenceReview,
    ExperimentObservation,
    ExperimentOutput,
    ExperimentProgram,
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
from ..workflow import _validate_experiment_program


class ExperimentPipeline:
    """Run one experiment requirement without exposing engineering stages as research cycles."""

    MAX_TRANSITIONS = 32
    MAX_IDENTICAL_REPAIRS = 2

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
            if (
                state.protocol_revision - initial_protocol_revision >= 2
                or state.program_revision - initial_program_revision >= 2
            ):
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
            except ModelBudgetExceeded:
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
                state.repeated_protocol_candidates += 1
                state.protocol_revision += 1
                state.stage = "protocol_revision"
                return self._failure(
                    item,
                    state,
                    step_dir,
                    persist,
                    "Protocol repair made no semantic change. Correct the preserved defect: "
                    + state.last_error,
                    force_block=(
                        state.repeated_protocol_candidates >= self.MAX_IDENTICAL_REPAIRS
                    ),
                )
            state.protocol = protocol
            state.last_protocol_candidate_sha256 = candidate_sha
            state.repeated_protocol_candidates = 0
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
                allow_repair=False,
            )
            state.protocol_review = protocol_review
            add(self.store.write_json(f"{step_dir}/review.json", protocol_review))
            errors = _review_errors(criteria, protocol_review)
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
            messages = self._program_messages(
                item, state, revision=revision, research_context=research_context
            )
            add(self.store.write_json(f"{step_dir}/input.json", {"messages": messages}))
            previous_source = state.program.python_code if state.program else ""
            try:
                if self.router.dry_run:
                    program = _dry_program(item, state.protocol)
                else:
                    source = self.router.complete_text(
                        task_type=(
                            "experiment_revision" if revision else "experiment_design"
                        ),
                        messages=messages,
                        temperature=0.1,
                        max_tokens=8192,
                    )
                    program = ExperimentProgram(
                        description=f"Executable implementation of {state.protocol.title}",
                        source=source,
                        seeds=state.protocol.seeds,
                    )
                _validate_experiment_program(program)
                if revision and previous_source and program.python_code == previous_source:
                    raise ValueError(
                        "program revision made no source change; apply the preserved defect"
                    )
            except ModelBudgetExceeded:
                persist(step_dir)
                raise
            except Exception as exc:
                candidate = locals().get("program")
                if isinstance(candidate, ExperimentProgram):
                    state.program = candidate
                    add(self.store.write_json(f"{step_dir}/invalid_program.json", candidate))
                state.stage = "program_revision"
                state.program_revision += int(revision)
                return self._failure(
                    item,
                    state,
                    step_dir,
                    persist,
                    f"Program validation failed: {type(exc).__name__}: {exc}",
                )
            state.program = program
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
                state.stage = "complete"
                persist(step_dir)
                return WorkResult(
                    work_id=item.work_id,
                    outcome="partial",
                    failure_class="none",
                    attempt_class="engineering",
                    summary="Dry run validated protocol and executable code generation.",
                )
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
            errors = _execution_errors(execution, smoke=True)
            if errors:
                state.stage = (
                    "smoke_execution"
                    if execution.failure_class == "infrastructure"
                    else "program_revision"
                )
                return self._failure(
                    item, state, step_dir, persist, "; ".join(errors)
                )
            state.stage = "full_execution"
            self._clear_failure(state)
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
            errors = _execution_errors(execution, smoke=False)
            if errors:
                state.stage = (
                    "full_execution"
                    if execution.failure_class == "infrastructure"
                    else "program_revision"
                )
                return self._failure(
                    item, state, step_dir, persist, "; ".join(errors)
                )
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
            missing = _criterion_id_errors(expected, evidence_review.criteria)
            if evidence_review.usable == "full" and missing:
                evidence_review.usable = "preliminary"
                evidence_review.issues.extend(missing)
                evidence_review.follow_up.append("Assess every work criterion by its exact id.")
            add(self.store.write_json(f"{step_dir}/review.json", evidence_review))
            rows = _criterion_results(expected, evidence_review.criteria)
            state.stage = "complete"
            persist(step_dir)
            if evidence_review.usable == "unusable":
                return WorkResult(
                    work_id=item.work_id,
                    outcome="partial",
                    failure_class="method",
                    attempt_class="scientific",
                    criteria=rows,
                    summary="Measurements were produced but failed scientific audit.",
                    errors=evidence_review.issues,
                    next_steps=evidence_review.follow_up,
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
                    f"Protocol `{state.protocol.title}`; sample sizes "
                    f"{state.protocol.sample_sizes}; seeds {state.protocol.seeds}."
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
                criteria=rows,
                summary=execution.summary,
                findings=[finding],
                errors=evidence_review.issues if not full else [],
                next_steps=evidence_review.follow_up,
            )

        if state.stage == "complete":
            persist(step_dir)
            return WorkResult(
                work_id=item.work_id,
                outcome="partial",
                failure_class="engineering",
                attempt_class="engineering",
                summary="Experiment state completed without a committed scientific result.",
                errors=["Reset or inspect the preserved experiment state."],
            )

        state.engineering_blocked = True
        state.last_error = f"Unknown experiment stage: {state.stage}"
        persist(step_dir)
        return self._blocked(item, state)

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
        signature = hashlib.sha256(error.encode()).hexdigest()
        if signature == state.last_defect_signature:
            state.engineering_failures += 1
        else:
            state.last_defect_signature = signature
            state.engineering_failures = 1
        state.last_error = error[-4000:]
        repair_limit = min(self.router.core.max_experiment_engineering_retries, 4)
        if force_block or state.engineering_failures >= repair_limit:
            state.engineering_blocked = True
            persist(step_dir)
            return self._blocked(item, state)
        persist(step_dir)
        return None

    @staticmethod
    def _clear_failure(state: ExperimentState) -> None:
        state.engineering_failures = 0
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
            "Design one bounded falsifiable experiment. Conditions must include each treatment and "
            "baseline as separate condition IDs. Baselines are a proper subset of conditions—for "
            "example conditions `factor_n8` and `binary_n8`, with only `binary_n8` listed in "
            "baselines. Include all encoding/model/dictionary/framing costs requested by the work "
            "item. Correctness checks test implementation validity, never the expected result. "
            "Preserve negative and null outcomes."
        )
        if revision:
            system += " Revise only the concrete preserved defect; do not return the same protocol."
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "work_item": item.model_dump(mode="json"),
                        "runtime_limits": {
                            "wall_seconds": max_wall,
                            "memory_mb": max_memory,
                            "cpus": max_cpus,
                        },
                        "previous_protocol": (
                            state.protocol.model_dump(mode="json") if state.protocol else None
                        ),
                        "defect": state.last_error if revision else "",
                        "research_objective": research_context.get("research_objective", ""),
                        "agenda_constraints": research_context.get("agenda_constraints", []),
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
                    "Audit the protocol. Return every supplied criterion ID exactly once. Put a "
                    "specific repair in the assessment detail whenever satisfied is false."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "work_item": item.model_dump(mode="json"),
                        "protocol": protocol.model_dump(mode="json"),
                        "agenda_constraints": research_context.get("agenda_constraints", []),
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
    def _program_messages(
        item: WorkItem,
        state: ExperimentState,
        *,
        revision: bool,
        research_context: dict[str, Any],
    ) -> list[dict[str, str]]:
        assert state.protocol is not None
        system = (
            "Return only raw Python source, with no Markdown fence, JSON wrapper, or explanation. "
            "Define run_experiment(mode: str) -> dict. Smoke mode must use at most one tiny unit per "
            "condition and finish well under 60 seconds; full mode uses the frozen seeds and sample "
            "sizes. The function must return exactly this v2 shape: "
            "{'schema_version': 2, 'experiment': str, 'status': 'completed'|'capped', "
            "'parameters': {str: scalar}, 'aggregate_metrics': {str: scalar}, "
            "'observations': [{'condition': str, 'sample_size': int, 'metrics': {str: scalar}}], "
            "'checks': [{'name': str, 'passed': bool, 'detail': str}], "
            "'conclusion': {'hypothesis': str, 'outcome': "
            "'supports'|'contradicts'|'null'|'inconclusive'|'characterizes', "
            "'basis_metrics': [str], 'statement': str}, 'limitations': [str]}. "
            "Every scalar is str, int, float, bool, or None; do not nest dictionaries inside parameters, "
            "aggregate_metrics, or observation metrics. Implement every condition with materially "
            "distinct logic when the protocol says it is distinct; never alias a treatment as a baseline, "
            "invent an unavailable external solver, hard-code measurements, or mark a check passed "
            "without computing it. Begin with imports, constants, classes, or function definitions. "
            "Use no network, subprocess, os, async, or multiprocessing."
        )
        if revision:
            system += " Correct only the preserved defect and change the source accordingly."
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "work_item": item.model_dump(mode="json"),
                        "protocol": state.protocol.model_dump(mode="json"),
                        "protocol_sha256": state.protocol_sha256,
                        "previous_program": (
                            {
                                "description": state.program.description,
                                "source": state.program.python_code[:12_000],
                                "seeds": state.program.seeds,
                            }
                            if state.program
                            else None
                        ),
                        "defect": state.last_error if revision else "",
                        "agenda_constraints": research_context.get("agenda_constraints", []),
                        "reusable_code_from_prior_completed_experiment": research_context.get(
                            "reusable_experiment_code", []
                        ),
                    },
                    ensure_ascii=False,
                ),
            },
        ]

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
            task_type="experiment_review",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Audit measurements against the frozen protocol. Assess every supplied work "
                        "criterion ID exactly once and recompute conclusions from observations. Preserve "
                        "sound negative and null outcomes. Full evidence requires every mandatory "
                        "criterion; use preliminary for scoped interpretable pilots and unusable for "
                        "wrong metrics, invalid baselines, leakage, or failed implementation checks."
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
                            "agenda_constraints": research_context.get(
                                "agenda_constraints", []
                            ),
                            "protocol": protocol.model_dump(mode="json"),
                            "program": {
                                "description": program.description,
                                "seeds": program.seeds,
                                "source_sha256": hashlib.sha256(
                                    program.python_code.encode("utf-8")
                                ).hexdigest(),
                                "source_excerpt": program.python_code[:5_000],
                            },
                            "validated_output": (
                                output.model_dump(mode="json") if output else None
                            ),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            schema=ExperimentEvidenceReview,
            allow_repair=False,
        )


def _protocol_criteria() -> dict[str, str]:
    return {
        "P_ALIGNMENT": "The protocol directly measures the evidence requirement.",
        "P_NULL": "The null outcome and decision rule are explicit and compatible.",
        "P_BASELINES": "Treatment and genuinely distinct strong baseline conditions are present.",
        "P_CHECKS": "Correctness checks test implementation validity, not result direction.",
        "P_SAMPLING": "Seeds, sample sizes, and analysis are reproducible and feasible.",
        "P_COSTS": "Dominant scientific costs and executable resource limits are represented.",
    }


def _review_errors(
    expected: dict[str, str], review: ExperimentProtocolReview
) -> list[str]:
    errors = _criterion_id_errors(expected, review.criteria)
    errors.extend(
        f"{row.criterion_id}: {row.detail}"
        for row in review.criteria
        if not row.satisfied
    )
    errors.extend(review.issues)
    errors.extend(review.required_revisions)
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


def _execution_errors(execution: Any, *, smoke: bool) -> list[str]:
    if not execution.success or execution.validated_output is None:
        return [execution.summary]
    output = execution.validated_output
    errors: list[str] = []
    if smoke and output.status != "completed":
        errors.append("Smoke execution reported capped status.")
    failed = [check.name for check in output.checks if not check.passed]
    if failed:
        errors.append("Implementation checks failed: " + ", ".join(failed))
    return errors


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
        conditions=[
            NamedDescription(id="treatment", description="Treatment implementation."),
            NamedDescription(id="baseline", description="Strong baseline implementation."),
        ],
        baselines=[
            NamedDescription(id="baseline", description="Strong baseline implementation.")
        ],
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


def _dry_program(item: WorkItem, protocol: ExperimentProtocol) -> ExperimentProgram:
    metric_id = protocol.metrics[0].id
    output = ExperimentOutput(
        experiment="dry-run protocol implementation",
        parameters={"seed": protocol.seeds[0], "dry_run": True},
        aggregate_metrics={"difference": 0.0},
        observations=[
            ExperimentObservation(
                condition=condition.id,
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
