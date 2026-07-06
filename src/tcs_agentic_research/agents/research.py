"""Research execution agent and durable report writer."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from ..artifact_store import ArtifactStore
from ..llm import LLMRouter
from ..prompt_loader import render_prompt
from ..render import render_report_markdown
from ..schemas import (
    ArtifactRef,
    ClaimRecord,
    ClaimStatus,
    ClaimType,
    EvidenceRecord,
    EvidenceType,
    ExperimentPlan,
    ExperimentResult,
    LeanStatement,
    LiteratureDependency,
    ProofObligation,
    ReportOutcome,
    ResearchCritique,
    ResearchProposal,
    ResearchReport,
    ResearchState,
    TheoremProverResult,
    utc_now,
)
from .critics import ResearchCriticAgent
from .experiment import ExperimentAgent
from .literature import LiteratureResearcher
from .theorem_prover import TheoremProverAgent


@dataclass(frozen=True)
class _ResearchLoopAction:
    action_type: str
    rationale: str = ""
    query: str = ""
    proof_obligation_id: str = ""
    expected_evidence: str = ""


class ResearchAgent:
    def __init__(self, store: ArtifactStore, router: LLMRouter, *, prompt_dir: str | None = None):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir
        self.literature = LiteratureResearcher(store, router, prompt_dir=prompt_dir)
        self.experiment = ExperimentAgent(store)
        self.critic = ResearchCriticAgent(store, router, prompt_dir=prompt_dir)
        self.theorem_prover = TheoremProverAgent(store, router, prompt_dir=prompt_dir)

    def run(
        self,
        proposal: ResearchProposal,
        state: ResearchState,
        *,
        max_loop_rounds: int = 3,
    ) -> tuple[ResearchReport, str]:
        task = self.store.read_text(ArtifactStore.RESEARCH_TASK)
        literature_answers = self._initial_literature_answers(proposal)
        context = self._build_research_context(
            task=task,
            state=state,
            proposal=proposal,
            literature_answers=literature_answers,
        )
        report = self._generate_report(proposal, context)
        report = self._normalize_report(report, proposal)
        report = self._add_complexity_verification_requirements(report)

        loop_observations = self._run_subsystem_loop(
            report,
            proposal,
            context,
            max_rounds=max_loop_rounds,
        )

        final_context = self._final_critic_context(context, report, loop_observations)
        report, critique = self.critic.review(report, context=final_context)
        self._add_forced_obligations(report, critique)
        report = self.critic.enforce_evidence_statuses(report)

        iteration_dir = self.store.create_iteration_dir(state.iteration)
        critique_ref = self.store.write_json(
            f"{iteration_dir}/research_critique_{report.report_id}.json", critique
        )
        self._append_report_ref(report, critique_ref)
        report_ref = self.store.write_json(
            f"{iteration_dir}/research_report_{report.report_id}.json", report
        )
        self.store.write_text(
            f"{iteration_dir}/research_report_{report.report_id}.md", render_report_markdown(report)
        )

        self._attach_report_refs_to_claims(report, report_ref=report_ref, critique_ref=critique_ref)
        self.store.append_claims(report.claims_generated)
        return report, report_ref.path

    def _initial_literature_answers(self, proposal: ResearchProposal) -> list[dict[str, object]]:
        literature_answers: list[dict[str, object]] = []
        for query in proposal.literature_queries[:5]:
            if not query.strip():
                continue
            literature_answers.append(
                self.literature.answer_query(query, limit=3).model_dump(mode="json")
            )
        return literature_answers

    def _build_research_context(
        self,
        *,
        task: str,
        state: ResearchState,
        proposal: ResearchProposal,
        literature_answers: list[dict[str, object]],
    ) -> str:
        return json.dumps(
            {
                "research_task_md": task,
                "research_state": state.model_dump(mode="json"),
                "proposal": proposal.model_dump(mode="json"),
                # Literature context is only supplied through mapped-nomenclature answers,
                # with quote-level provenance and duplicate-result flags.
                "local_literature_answers": literature_answers,
                "recent_claims": self.store.read_jsonl(ArtifactStore.CLAIM_LEDGER, limit=30),
            },
            indent=2,
        )

    def _generate_report(self, proposal: ResearchProposal, context: str) -> ResearchReport:
        mock_output = self._mock_report(proposal)
        messages = [
            {
                "role": "system",
                "content": render_prompt("research_agent", override_dir=self.prompt_dir),
            },
            {
                "role": "user",
                "content": (
                    "Execute the selected proposal using only evidence that can be "
                    "referenced by durable artifacts. The research subagent will run "
                    "a bounded observe-plan-act loop after this draft to invoke "
                    "verification/literature subsystems for explicit obligations.\n"
                    f"Context:\n{context}"
                ),
            },
        ]
        return self.router.complete_structured(
            task_type="research_execution",
            messages=messages,
            schema=ResearchReport,
            mock_output=mock_output if self.router.dry_run else None,
        )

    def _run_subsystem_loop(
        self,
        report: ResearchReport,
        proposal: ResearchProposal,
        base_context: str,
        *,
        max_rounds: int,
    ) -> list[str]:
        """Run a private bounded tool loop before the final critic audit.

        This loop is intentionally owned by the research agent, not by the critic. It uses the
        draft report's explicit obligations and verification needs to call subsystems, mutates the
        in-memory report with observed evidence, and leaves final acceptance/downgrade decisions to
        one critic pass at the end.
        """
        observations: list[str] = []
        attempted_lean_obligation_ids: set[str] = set()
        attempted_experiment_obligation_ids: set[str] = set()
        asked_literature_queries = {q.strip() for q in proposal.literature_queries[:5] if q.strip()}

        for round_index in range(1, max(0, max_rounds) + 1):
            actions = self._plan_subsystem_actions(
                report,
                proposal,
                attempted_lean_obligation_ids=attempted_lean_obligation_ids,
                attempted_experiment_obligation_ids=attempted_experiment_obligation_ids,
                asked_literature_queries=asked_literature_queries,
            )
            if not actions:
                observations.append(
                    f"Round {round_index}: no actionable subsystem calls remained."
                )
                break

            for action in actions:
                observation, _refs = self._execute_loop_action(
                    action,
                    report,
                    context=self._loop_context(base_context, report, observations),
                )
                observations.append(f"Round {round_index}: {observation}")
                if action.action_type == "lean_proof" and action.proof_obligation_id:
                    attempted_lean_obligation_ids.add(action.proof_obligation_id)
                if action.action_type == "experiment" and action.proof_obligation_id:
                    attempted_experiment_obligation_ids.add(action.proof_obligation_id)
                if action.action_type == "literature_query" and action.query.strip():
                    asked_literature_queries.add(action.query.strip())

        return observations

    def _normalize_report(
        self, report: ResearchReport, proposal: ResearchProposal
    ) -> ResearchReport:
        if report.proposal_id != proposal.proposal_id:
            report.proposal_id = proposal.proposal_id
        for claim in report.claims_generated:
            if proposal.proposal_id not in claim.related_proposal_ids:
                claim.related_proposal_ids.append(proposal.proposal_id)
        return report

    def _add_complexity_verification_requirements(self, report: ResearchReport) -> ResearchReport:
        for estimate in report.complexity_estimates:
            if estimate.needs_derivation_review:
                message = (
                    f"Complexity estimate for {estimate.resource}={estimate.bound} "
                    "needs derivation review."
                )
                if message not in report.required_verifications:
                    report.required_verifications.append(message)
        return report

    def _loop_context(
        self,
        base_context: str,
        report: ResearchReport,
        observations: list[str],
    ) -> str:
        loop_state = {
            "current_report": report.model_dump(mode="json"),
            "private_loop_observations": observations,
        }
        return (
            base_context
            + "\n\nResearch subagent private loop state:\n"
            + json.dumps(loop_state, indent=2)
        )

    def _final_critic_context(
        self,
        base_context: str,
        report: ResearchReport,
        observations: list[str],
    ) -> str:
        return self._loop_context(base_context, report, observations)

    def _plan_subsystem_actions(
        self,
        report: ResearchReport,
        proposal: ResearchProposal,
        *,
        attempted_lean_obligation_ids: set[str],
        attempted_experiment_obligation_ids: set[str],
        asked_literature_queries: set[str],
    ) -> list[_ResearchLoopAction]:
        actions: list[_ResearchLoopAction] = []
        for obligation in report.proof_obligations:
            if obligation.suggested_tool != "lean":
                continue
            if obligation.status not in {"open", "in_progress"}:
                continue
            if obligation.obligation_id in attempted_lean_obligation_ids:
                continue
            actions.append(
                _ResearchLoopAction(
                    action_type="lean_proof",
                    proof_obligation_id=obligation.obligation_id,
                    rationale=(
                        "Open Lean proof obligation appears in the draft report; invoke "
                        "LEAP before the report can treat the claim as established."
                    ),
                    expected_evidence="Lean compiler logs, proof DAG, and any verified Lean file.",
                )
            )

        for obligation in report.proof_obligations:
            if obligation.suggested_tool != "experiment":
                continue
            if obligation.status not in {"open", "in_progress"}:
                continue
            if obligation.obligation_id in attempted_experiment_obligation_ids:
                continue
            actions.append(
                _ResearchLoopAction(
                    action_type="experiment",
                    proof_obligation_id=obligation.obligation_id,
                    rationale=(
                        "Open experimental verification obligation appears in the draft "
                        "report; generate and run an experiment through ExperimentAgent."
                    ),
                    expected_evidence="ExperimentRuns artifacts with command/config/logs/seeds.",
                )
            )

        for query in self._candidate_literature_queries(report, proposal):
            normalized = query.strip()
            if not normalized or normalized in asked_literature_queries:
                continue
            actions.append(
                _ResearchLoopAction(
                    action_type="literature_query",
                    query=normalized,
                    rationale=(
                        "A literature/novelty/citation verification need remains; query the "
                        "local LiteratureDB in canonical notation before making or "
                        "rejecting claims."
                    ),
                    expected_evidence=(
                        "Mapped LiteratureDB query answer with quote-level provenance."
                    ),
                )
            )

        return actions

    def _candidate_literature_queries(
        self, report: ResearchReport, proposal: ResearchProposal
    ) -> list[str]:
        queries: list[str] = []
        for obligation in report.proof_obligations:
            if obligation.suggested_tool == "literature" and obligation.status in {
                "open",
                "in_progress",
                "blocked",
            }:
                queries.append(obligation.statement)
        for item in report.required_verifications:
            if _looks_like_literature_need(item):
                queries.append(item)
        for item in report.unresolved_issues:
            if item.startswith("Local LiteratureDB query `"):
                continue
            if _looks_like_literature_need(item):
                queries.append(item)
        for dependency in report.literature_dependencies:
            if dependency.used_for.startswith("Research loop query:"):
                continue
            query = " ".join(
                part
                for part in [dependency.citation_key, dependency.title, dependency.used_for]
                if part
            )
            if query:
                queries.append(query)
        for claim in report.claims_generated:
            if claim.claim_type not in {ClaimType.literature, ClaimType.novelty}:
                continue
            if claim.status in {ClaimStatus.cited, ClaimStatus.proved_by_lean}:
                continue
            queries.append(claim.normalized_statement or claim.statement)
        queries.extend(proposal.literature_queries)
        return list(dict.fromkeys(query.strip() for query in queries if query.strip()))

    def _execute_loop_action(
        self,
        action: _ResearchLoopAction,
        report: ResearchReport,
        *,
        context: str,
    ) -> tuple[str, list[ArtifactRef]]:
        if action.action_type == "literature_query":
            return self._run_literature_action(action, report)
        if action.action_type == "lean_proof":
            return self._run_lean_action(action, report, context=context)
        if action.action_type == "experiment":
            return self._run_experiment_action(action, report, context=context)
        return f"Stop action recorded: {action.rationale or 'no rationale supplied'}", []

    def _run_literature_action(
        self,
        action: _ResearchLoopAction,
        report: ResearchReport,
    ) -> tuple[str, list[ArtifactRef]]:
        query = action.query.strip()
        if not query:
            return "Skipped literature query action because the query was empty.", []
        answer = self.literature.answer_query(query, limit=5)
        query_ledger_ref = self.store.artifact_ref("LiteratureDB/query_answers.jsonl")
        self._append_report_ref(report, query_ledger_ref)
        citation_keys = list(
            dict.fromkeys(result.citation_key for result in answer.results if result.citation_key)
        )
        if answer.results:
            report.evidence.append(
                EvidenceRecord(
                    evidence_type=EvidenceType.citation,
                    summary=(
                        f"Research loop LiteratureDB query `{query}` returned "
                        f"{len(answer.results)} mapped result(s). This is report-level context; "
                        "claim-local citation evidence is still required before accepting claims."
                    ),
                    artifact_refs=[query_ledger_ref],
                    citation_keys=citation_keys,
                    verifier="LiteratureResearcher",
                    confidence=0.5,
                )
            )
            existing = {(dep.citation_key, dep.used_for) for dep in report.literature_dependencies}
            used_for = f"Research loop query: {query}"
            for result in answer.results:
                if not result.citation_key or (result.citation_key, used_for) in existing:
                    continue
                locator = result.provenance[0].locator if result.provenance else result.label
                report.literature_dependencies.append(
                    LiteratureDependency(
                        citation_key=result.citation_key,
                        title=result.title,
                        used_for=used_for,
                        provenance=(
                            f"LiteratureQueryAnswer {answer.answer_id}; locator={locator}; "
                            f"result_id={result.result_id}"
                        ),
                        notation_mappings=result.notation_mappings,
                    )
                )
                existing.add((result.citation_key, used_for))
        else:
            issue = (
                f"Local LiteratureDB query `{query}` returned no results; import and extract "
                "relevant papers if this blocks the proposal."
            )
            if issue not in report.unresolved_issues:
                report.unresolved_issues.append(issue)
        return (
            f"LiteratureDB query `{query}` returned {len(answer.results)} result(s); "
            f"answer ledger: {query_ledger_ref.path}.",
            [query_ledger_ref],
        )

    def _run_lean_action(
        self,
        action: _ResearchLoopAction,
        report: ResearchReport,
        *,
        context: str,
    ) -> tuple[str, list[ArtifactRef]]:
        obligation = self._find_obligation(report, action.proof_obligation_id)
        if obligation is None:
            return (
                "Skipped LEAP action because obligation "
                f"`{action.proof_obligation_id}` was absent.",
                [],
            )
        goal = self._lean_goal_from_obligation(obligation)
        result = self.theorem_prover.prove(
            goal,
            context=(
                context
                + "\n\nSelected research-loop action:\n"
                + json.dumps(asdict(action), indent=2)
                + "\n\nCurrent report before LEAP verification:\n"
                + report.model_dump_json(indent=2)
            ),
        )
        self._record_theorem_prover_result(report, obligation, result)
        refs = _unique_refs([*result.proved_artifacts, *result.artifact_refs])
        return (
            f"LEAP returned status `{result.status}` for obligation "
            f"`{obligation.obligation_id}`; result_id={result.result_id}.",
            refs,
        )

    def _run_experiment_action(
        self,
        action: _ResearchLoopAction,
        report: ResearchReport,
        *,
        context: str,
    ) -> tuple[str, list[ArtifactRef]]:
        obligation = self._find_obligation(report, action.proof_obligation_id)
        if obligation is None:
            return (
                "Skipped experiment action because obligation "
                f"`{action.proof_obligation_id}` was absent.",
                [],
            )
        plan = self._plan_experiment(action, obligation, report, context=context)
        if not plan.should_run:
            obligation.status = "blocked"
            message = (
                f"Experiment planner declined to run obligation `{obligation.obligation_id}`: "
                f"{plan.rationale or 'no rationale supplied'}"
            )
            if message not in report.unresolved_issues:
                report.unresolved_issues.append(message)
            return message, []

        try:
            result = self._execute_experiment_plan(plan, obligation)
        except Exception as exc:  # noqa: BLE001 - record execution failure in report
            obligation.status = "blocked"
            message = (
                f"Experiment `{plan.name}` failed for obligation "
                f"`{obligation.obligation_id}`: {type(exc).__name__}: {exc}"
            )
            if message not in report.unresolved_issues:
                report.unresolved_issues.append(message)
            return message, []
        self._record_experiment_result(report, obligation, result, plan)
        refs = _unique_refs(result.artifact_refs)
        return (
            f"ExperimentAgent ran `{plan.name}` for obligation "
            f"`{obligation.obligation_id}`; {result.summary}",
            refs,
        )

    def _plan_experiment(
        self,
        action: _ResearchLoopAction,
        obligation: ProofObligation,
        report: ResearchReport,
        *,
        context: str,
    ) -> ExperimentPlan:
        mock_output = self._mock_experiment_plan(obligation)
        messages = [
            {
                "role": "system",
                "content": render_prompt("experiment_planner", override_dir=self.prompt_dir),
            },
            {
                "role": "user",
                "content": (
                    "Create an executable experiment for this research-loop action.\n\n"
                    f"Action:\n{json.dumps(asdict(action), indent=2)}\n\n"
                    f"Obligation:\n{obligation.model_dump_json(indent=2)}\n\n"
                    f"Current report:\n{report.model_dump_json(indent=2)}\n\n"
                    f"Context:\n{context}"
                ),
            },
        ]
        return self.router.complete_structured(
            task_type="experiment_planning",
            messages=messages,
            schema=ExperimentPlan,
            mock_output=mock_output if self.router.dry_run else None,
        )

    def _mock_experiment_plan(self, obligation: ProofObligation) -> ExperimentPlan:
        payload = {
            "obligation_id": obligation.obligation_id,
            "status": "dry_run_experiment_executed",
            "note": "Mock experiment exercises ExperimentAgent plumbing only.",
        }
        code = (
            "import json\n"
            f"data = {payload!r}\n"
            "print(json.dumps(data, sort_keys=True))\n"
        )
        return ExperimentPlan(
            name=_experiment_safe_name(obligation.obligation_id),
            execution_mode="python",
            code=code,
            config={"obligation_statement": obligation.statement},
            seeds=[0],
            timeout_seconds=60,
            rationale="Dry-run mock experiment plan.",
            expected_interpretation=(
                "This verifies only that the experiment subsystem can run and record artifacts."
            ),
        )

    def _execute_experiment_plan(
        self, plan: ExperimentPlan, obligation: ProofObligation
    ) -> ExperimentResult:
        config = {
            "experiment_plan": plan.model_dump(mode="json"),
            "obligation": obligation.model_dump(mode="json"),
            "user_config": plan.config,
        }
        if plan.execution_mode == "command":
            if not plan.command:
                raise RuntimeError("Experiment plan selected command mode with empty command")
            return self.experiment.run_command(
                name=plan.name,
                command=plan.command,
                config=config,
                seeds=plan.seeds,
                timeout_seconds=plan.timeout_seconds,
            )
        if not plan.code.strip():
            raise RuntimeError("Experiment plan selected python mode with empty code")
        return self.experiment.run_python(
            name=plan.name,
            code=plan.code,
            config=config,
            seeds=plan.seeds,
            timeout_seconds=plan.timeout_seconds,
        )

    def _record_experiment_result(
        self,
        report: ResearchReport,
        obligation: ProofObligation,
        result: ExperimentResult,
        plan: ExperimentPlan,
    ) -> None:
        result.supports_claim_ids = list(
            dict.fromkeys([*result.supports_claim_ids, *obligation.claim_ids])
        )
        if result.run_id not in {existing.run_id for existing in report.experimental_results}:
            report.experimental_results.append(result)
        refs = _unique_refs(result.artifact_refs)
        for ref in refs:
            if ref.path not in {existing.path for existing in obligation.artifact_refs}:
                obligation.artifact_refs.append(ref)
            self._append_report_ref(report, ref)
        succeeded = _experiment_succeeded(result)
        obligation.status = "experimentally_supported" if succeeded else "blocked"
        evidence = EvidenceRecord(
            evidence_type=EvidenceType.experiment,
            summary=(
                f"Experiment `{plan.name}` run_id={result.run_id} for obligation "
                f"`{obligation.obligation_id}`. {result.summary}"
            ),
            artifact_refs=refs,
            verifier="ExperimentAgent",
            confidence=0.4 if succeeded else 0.1,
        )
        report.evidence.append(evidence)
        for claim in report.claims_generated:
            if claim.claim_id not in obligation.claim_ids:
                continue
            claim.evidence.append(evidence.model_copy(deep=True))
            if claim.claim_type in {
                ClaimType.mathematical,
                ClaimType.algorithmic,
                ClaimType.complexity,
                ClaimType.theorem_statement,
            }:
                message = (
                    f"Experimental evidence for claim `{claim.claim_id}` is not a proof; "
                    "proof or derivation review remains required."
                )
                if message not in report.required_verifications:
                    report.required_verifications.append(message)

    def _find_obligation(
        self, report: ResearchReport, obligation_id: str
    ) -> ProofObligation | None:
        return next(
            (
                candidate
                for candidate in report.proof_obligations
                if candidate.obligation_id == obligation_id
            ),
            None,
        )

    def _add_forced_obligations(
        self, report: ResearchReport, critique: ResearchCritique
    ) -> int:
        known_ids = {obligation.obligation_id for obligation in report.proof_obligations}
        known_statements = {obligation.statement for obligation in report.proof_obligations}
        added = 0
        for obligation in critique.forced_verifications:
            if obligation.obligation_id in known_ids or obligation.statement in known_statements:
                continue
            report.proof_obligations.append(obligation)
            known_ids.add(obligation.obligation_id)
            known_statements.add(obligation.statement)
            added += 1
        return added

    def _append_report_ref(self, report: ResearchReport, ref: ArtifactRef) -> None:
        if ref.path not in {existing.path for existing in report.artifact_refs}:
            report.artifact_refs.append(ref)

    def _record_theorem_prover_result(
        self,
        report: ResearchReport,
        obligation: ProofObligation,
        result: TheoremProverResult,
    ) -> None:
        result_refs = _unique_refs([*result.proved_artifacts, *result.artifact_refs])
        for ref in result_refs:
            if ref.path not in {existing.path for existing in obligation.artifact_refs}:
                obligation.artifact_refs.append(ref)
            self._append_report_ref(report, ref)
        if result.status == "proved":
            obligation.status = "proved"
            self._drop_resolved_verification_items(report, obligation)
            for claim in report.claims_generated:
                if claim.claim_id not in obligation.claim_ids:
                    continue
                claim.evidence.append(
                    EvidenceRecord(
                        evidence_type=EvidenceType.lean_proof,
                        summary=(
                            f"LEAP verified proof obligation `{obligation.obligation_id}` "
                            f"for Lean goal `{result.root_goal.name}`."
                        ),
                        artifact_refs=result_refs,
                        verifier="LEAPHarness",
                        confidence=1.0,
                    )
                )
                claim.status = ClaimStatus.proved_by_lean
        elif result.status == "partially_proved":
            obligation.status = "in_progress"
        else:
            obligation.status = "blocked"
        if result.recommended_next_steps:
            for step in result.recommended_next_steps:
                if step not in report.required_verifications and obligation.status != "proved":
                    report.required_verifications.append(step)

    def _drop_resolved_verification_items(
        self, report: ResearchReport, obligation: ProofObligation
    ) -> None:
        markers = [obligation.obligation_id, obligation.statement, *obligation.claim_ids]
        markers = [marker for marker in markers if marker]
        report.required_verifications = [
            item
            for item in report.required_verifications
            if not any(marker in item for marker in markers)
        ]

    def _lean_goal_from_obligation(self, obligation: ProofObligation) -> LeanStatement:
        statement = obligation.statement.strip()
        for prefix in ["lean:", "Lean:", "LEAN:"]:
            if statement.startswith(prefix):
                statement = statement[len(prefix) :].strip()
                break
        name = _lean_safe_name(obligation.obligation_id)
        return LeanStatement(name=name, statement=statement)

    def _attach_report_refs_to_claims(
        self,
        report: ResearchReport,
        *,
        report_ref: ArtifactRef,
        critique_ref: ArtifactRef,
    ) -> None:
        refs = [report_ref, critique_ref]
        for claim in report.claims_generated:
            if report.report_id not in claim.related_report_ids:
                claim.related_report_ids.append(report.report_id)
            if report.proposal_id not in claim.related_proposal_ids:
                claim.related_proposal_ids.append(report.proposal_id)
            claim.evidence.append(
                EvidenceRecord(
                    evidence_type=EvidenceType.critic_review,
                    summary=(
                        "Research report and critic review committed as durable artifacts. "
                        "This audit record is not certifying proof evidence."
                    ),
                    artifact_refs=refs,
                    verifier="ResearchAgent",
                    confidence=0.0,
                )
            )
            claim.updated_at = utc_now()

    def _mock_report(self, proposal: ResearchProposal) -> ResearchReport:
        claim = ClaimRecord(
            claim_type=ClaimType.other,
            statement=(
                f"Proposal {proposal.proposal_id} has not yet produced a verified "
                "main-task solution; current progress is an auditable scoping pass."
            ),
            status=ClaimStatus.informal_argument,
            related_proposal_ids=[proposal.proposal_id],
            evidence=[
                EvidenceRecord(
                    evidence_type=EvidenceType.informal_argument,
                    summary="Dry-run mock research execution records process progress only.",
                    confidence=0.2,
                )
            ],
        )
        return ResearchReport(
            proposal_id=proposal.proposal_id,
            outcome=ReportOutcome.partially_succeeded,
            executive_summary=(
                "Dry-run mock execution completed a conservative scoping iteration. "
                "It does not claim a breakthrough."
            ),
            claims_generated=[claim],
            proof_obligations=[
                ProofObligation(
                    statement=(
                        "Formalize any central mathematical claim before treating it as "
                        "established."
                    ),
                    claim_ids=[claim.claim_id],
                    suggested_tool="lean",
                )
            ],
            unresolved_issues=[
                "Literature claims need provenance-bearing extraction.",
                "Any algorithmic improvement requires an explicit complexity derivation.",
            ],
            proposed_next_steps=[
                "Import and normalize the most relevant papers into LiteratureDB.",
                (
                    "Select a specific lemma, reduction, or algorithmic subgoal for proof "
                    "or literature review."
                ),
            ],
            required_verifications=[
                "No conjecture or informal argument may be upgraded without Lean, citation, "
                "derivation, or experimental evidence as appropriate."
            ],
        )


def _experiment_succeeded(result: ExperimentResult) -> bool:
    return bool(re.search(r"\bcode\s+0\b", result.summary.lower()))


def _experiment_safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return name[:80] or "research_loop_experiment"


def _looks_like_literature_need(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in [
            "citation",
            "cited",
            "literature",
            "novelty",
            "prior work",
            "paper",
            "provenance",
        ]
    )


def _lean_safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_']+", "_", value).strip("_")
    if not name or not re.match(r"[A-Za-z_]", name):
        name = "goal_" + name
    return name[:80]


def _unique_refs(refs: list[ArtifactRef]) -> list[ArtifactRef]:
    unique: list[ArtifactRef] = []
    seen: set[str] = set()
    for ref in refs:
        if ref.path in seen:
            continue
        seen.add(ref.path)
        unique.append(ref)
    return unique
