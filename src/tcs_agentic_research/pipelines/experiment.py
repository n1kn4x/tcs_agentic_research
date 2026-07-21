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
from ..llm import LLMRouter, ModelBudgetExceeded, StructuredLLMError
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
    ExperimentProgramPatch,
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
            if (
                revision
                and state.program is not None
                and (
                    "source budget" in state.last_error
                    or "at most 20000 characters" in state.last_error
                    or "may use os only" in state.last_error
                )
            ):
                try:
                    _validate_experiment_program(state.program)
                except ValueError:
                    pass
                else:
                    state.stage = "smoke_execution"
                    persist(step_dir)
                    return None
            messages = self._program_messages(
                item, state, revision=revision, research_context=research_context
            )
            add(self.store.write_json(f"{step_dir}/input.json", {"messages": messages}))
            repair_defect = state.last_error
            rewrite = revision and _needs_program_rewrite(repair_defect)
            try:
                if self.router.dry_run:
                    program = _dry_program(item, state.protocol)
                elif revision and state.program is not None and not rewrite:
                    patch = self.router.complete_structured(
                        task_type="experiment_patch",
                        messages=messages,
                        schema=ExperimentProgramPatch,
                        temperature=0.1,
                        max_tokens=4096,
                        allow_repair=False,
                    )
                    add(self.store.write_json(f"{step_dir}/program_patch.json", patch))
                    source = _apply_program_patch(state.program.python_code, patch)
                    program = ExperimentProgram(
                        description=state.program.description,
                        source=source,
                        seeds=state.protocol.seeds,
                    )
                else:
                    source = self.router.complete_text(
                        task_type="experiment_revision" if rewrite else "experiment_design",
                        messages=messages,
                        temperature=0.1,
                        max_tokens=8192,
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
                    return self._failure(
                        item,
                        state,
                        step_dir,
                        persist,
                        "Program revision made no source change. The unresolved defect remains: "
                        + repair_defect,
                        force_block=(
                            state.repeated_program_candidates >= self.MAX_IDENTICAL_REPAIRS
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
            errors = _execution_errors(execution, protocol=state.protocol, smoke=False)
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
            if evidence_review.usable == "unusable":
                final_result = WorkResult(
                    work_id=item.work_id,
                    outcome="partial",
                    failure_class="method",
                    attempt_class="scientific",
                    criteria=rows,
                    summary="Measurements were produced but failed scientific audit.",
                    errors=evidence_review.issues,
                    next_steps=evidence_review.follow_up,
                )
                state.final_result = final_result
                state.stage = "complete"
                persist(step_dir)
                return final_result
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
        state.engineering_failures += 1
        state.last_error = error[-4000:]
        repair_limit = self.router.core.max_experiment_engineering_retries
        if force_block or state.engineering_failures >= repair_limit:
            state.engineering_blocked = True
            persist(step_dir)
            return self._blocked(item, state)
        persist(step_dir)
        return None

    @staticmethod
    def _clear_failure(state: ExperimentState) -> None:
        state.engineering_failures = 0
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
            "separate. Include all dominant costs and requested parameter regimes. Correctness checks "
            "test implementation validity, never the expected result. A reproducibility check may "
            "require deterministic generated instances, decisions, and operation counts, but never "
            "identical wall-clock timings. Use fixed seeds and a decision rule that is executable with "
            "the stated samples. Preserve negative and null outcomes."
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
                    "Audit the protocol. Return every supplied criterion ID exactly once. In this "
                    "schema baselines are designated members of conditions, so that overlap is required, "
                    "not a defect. A requested simple method can be the valid baseline; reject dummy, "
                    "no-op, knowingly incorrect, or irrelevant controls. Do not demand an extra baseline "
                    "that the scientific comparison does not need. Reject checks that require repeated "
                    "wall-clock measurements to be identical; timing is inherently noisy. Judge the "
                    "literal description, not implications guessed from an ID. Do not search for a flaw "
                    "when a criterion is satisfied and never mark it false while saying it is valid. A "
                    "false detail must end with `Repair:` followed by one concrete imperative change."
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
            "Return only compact raw Python source, with no Markdown fence, JSON wrapper, explanation, "
            "unfinished comments, dead code, or placeholders. Keep the complete source under 12,000 "
            "characters. Define run_experiment(mode: str) -> dict and implement the whole frozen protocol. "
            "Near the start of that function, branch explicitly on mode (for example "
            "`sample_count = 1 if mode == 'smoke' else full_sample_count`) and actually use the selected "
            "bound. Smoke must run every condition on at most one tiny unit and finish well under 60 "
            "seconds; full mode uses the frozen seeds and sample sizes. The function must return exactly "
            "this v2 shape: "
            "{'schema_version': 2, 'experiment': str, 'status': 'completed'|'capped', "
            "'parameters': {str: scalar}, 'aggregate_metrics': {str: scalar}, "
            "'observations': [{'condition': str, 'sample_size': int, 'metrics': {str: scalar}}], "
            "with exactly one aggregate observation per protocol condition (sample_size is the number "
            "of units summarized, not one row per replicate), "
            "'checks': [{'name': str, 'passed': bool, 'detail': str}], "
            "'conclusion': {'hypothesis': str, 'outcome': "
            "'supports'|'contradicts'|'null'|'inconclusive'|'characterizes', "
            "'basis_metrics': [str], 'statement': str}, 'limitations': [str]}. "
            "Every scalar is str, int, float, bool, or None. Parameter values may also be flat lists of "
            "scalars (for seeds or parameter grids); aggregate and observation metrics remain scalar. Do "
            "not nest dictionaries. Emit every protocol correctness-check ID exactly "
            "once as a check name, with one aggregate pass/fail decision and detail; do not suffix IDs by "
            "condition or replicate. Copy the protocol hypothesis verbatim into conclusion.hypothesis. "
            "When conditions are variants of one algorithm, prefer one small shared core parameterized "
            "by explicit feature flags so correctness fixes apply to every variant; feature flags must still "
            "produce the protocol's material differences. Validate the core on both positive and negative "
            "known cases before benchmarking. Never alias a treatment as a baseline, "
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
        if revision and _needs_program_rewrite(state.last_error):
            system = (
                "Return only one complete compact replacement Python source file. The prior program "
                "failed a semantic correctness or runtime check, so repair the underlying algorithm; "
                "do not merely alter, remove, or force-pass the check. Preserve the frozen protocol, "
                "run_experiment(mode: str) -> dict, the v2 contract, tiny smoke behavior, negative/null "
                "outcomes, and requested artifacts. Reuse sound helpers when useful, but a clear rewrite "
                "is preferable to layering patches over faulty state."
            )
        elif revision:
            system = (
                "Repair the preserved Python source with a small line-range patch. Return one to six "
                "replacements using the one-based line numbers in previous_program.numbered_source. "
                "start_line and end_line are inclusive; new_lines contains complete replacement lines "
                "without line-number prefixes (an empty list deletes the range). Change only what is "
                "needed for the concrete defect, but fix every part named by that defect. Do not return "
                "the whole program. Preserve run_experiment(mode: str) -> dict, the frozen protocol, the "
                "v2 output contract, negative/null outcomes, and bounded smoke behavior. Never silence a "
                "failed check or exception; repair the underlying computation."
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
                        "previous_program": (
                            (
                                {
                                    "description": state.program.description,
                                    "source_omitted": (
                                        "The prior algorithm failed semantic checks; implement a "
                                        "fresh complete replacement from the frozen protocol."
                                    ),
                                    "seeds": state.program.seeds,
                                }
                                if revision and _needs_program_rewrite(state.last_error)
                                else {
                                    "description": state.program.description,
                                    "numbered_source": _numbered_source(
                                        state.program.python_code
                                    ),
                                    "seeds": state.program.seeds,
                                }
                            )
                            if state.program
                            else None
                        ),
                        "defect": state.last_error if revision else "",
                        "agenda_constraints": research_context.get("agenda_constraints", []),
                        "reusable_code_from_prior_completed_experiment": (
                            []
                            if revision
                            else research_context.get("reusable_experiment_code", [])
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
                                "source_excerpt": program.python_code[:8_000],
                            },
                            "execution_artifacts": [
                                ref.path for ref in execution.artifact_refs[:80]
                            ],
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
        "P_BASELINES": (
            "At least one scientifically valid comparator is designated in baselines as a member of "
            "conditions; no dummy, knowingly incorrect, or irrelevant control is introduced."
        ),
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
        if not row.satisfied and not _self_accepting_assessment(row.detail)
    )
    errors.extend(review.issues)
    errors.extend(review.required_revisions)
    return list(dict.fromkeys(error for error in errors if error))


def _self_accepting_assessment(detail: str) -> bool:
    """Ignore a contradictory false bit when the same assessment explicitly accepts the design."""
    import re

    return bool(
        re.search(
            r"(?i)(?:\b(?:is|are|seems?)\s+(?:scientifically\s+)?valid\b|"
            r"\b(?:is|are|seems?)\s+satisfied\b|"
            r"\bsatisf(?:y|ies|ying)\s+the\s+(?:requirement|criterion|schema)\b|"
            r"\ball\s+(?:criteria\s+)?are\s+true\b)",
            detail,
        )
    )


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
    duplicate_conditions = sorted(
        condition for condition in observed_conditions if observed_ids.count(condition) > 1
    )
    if duplicate_conditions:
        errors.append(
            "Output must aggregate to one observation per condition; duplicate conditions: "
            + ", ".join(duplicate_conditions)
        )

    expected_metrics = {metric.id for metric in protocol.metrics}
    for observation in output.observations:
        missing_metrics = sorted(expected_metrics - set(observation.metrics))
        if missing_metrics:
            errors.append(
                f"Observation `{observation.condition}` omitted protocol metrics: "
                + ", ".join(missing_metrics)
            )
        if smoke and observation.sample_size > 1:
            errors.append(
                f"Smoke observation `{observation.condition}` used sample_size="
                f"{observation.sample_size}; the smoke limit is 1."
            )

    expected_checks = {check.id for check in protocol.correctness_checks}
    actual_checks = {check.name for check in output.checks}
    missing_checks = sorted(expected_checks - actual_checks)
    if missing_checks:
        errors.append("Output omitted protocol correctness checks: " + ", ".join(missing_checks))
    if _normalized_text(output.conclusion.hypothesis) != _normalized_text(protocol.hypothesis):
        errors.append("Output conclusion changed the frozen protocol hypothesis.")
    return errors


def _needs_program_rewrite(error: str) -> bool:
    lowered = error.lower()
    return any(
        marker in lowered
        for marker in [
            "implementation checks failed",
            "experiment program failed",
            "failed scientific audit",
            "syntax error in",
            "rewrite the",
        ]
    )


def _apply_program_patch(source: str, patch: ExperimentProgramPatch) -> str:
    lines = source.splitlines()
    ranges = sorted(
        (row.start_line, row.end_line, row.new_lines) for row in patch.replacements
    )
    if any(start < 1 or end > len(lines) for start, end, _ in ranges):
        raise ValueError("program patch line range is outside the preserved source")
    if any(left[1] >= right[0] for left, right in zip(ranges, ranges[1:])):
        raise ValueError("program patch line ranges overlap")
    for start, end, new_lines in reversed(ranges):
        lines[start - 1 : end] = new_lines
    return "\n".join(lines)


def _numbered_source(source: str) -> str:
    return "\n".join(
        f"{line_number:04d}: {line}"
        for line_number, line in enumerate(source.splitlines(), 1)
    )


def _normalized_text(value: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _program_protocol_context(protocol: ExperimentProtocol) -> dict[str, Any]:
    return {
        "title": protocol.title,
        "hypothesis": protocol.hypothesis,
        "condition_ids": [row.id for row in protocol.conditions],
        "baseline_ids": [row.id for row in protocol.baselines],
        "metric_ids": [row.id for row in protocol.metrics],
        "correctness_check_ids": [row.id for row in protocol.correctness_checks],
        "sample_sizes": protocol.sample_sizes,
        "seeds": protocol.seeds,
        "decision_rule": protocol.decision_rule,
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
