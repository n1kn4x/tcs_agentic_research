"""Research execution agent using native OpenAI/vLLM tool calls."""

from __future__ import annotations

import json
from typing import Any

from ..artifact_store import ArtifactStore, to_plain
from ..llm import LLMRouter
from ..prompt_loader import render_prompt
from ..prompt_serialization import compact_json_dumps
from ..render import render_report_markdown
from ..schemas import (
    ArtifactRef,
    ClaimRecord,
    ClaimStatus,
    ClaimType,
    EvidenceRecord,
    EvidenceType,
    ExperimentResult,
    LiteratureDependency,
    ObligationRun,
    ObligationRunSubmission,
    ProofObligation,
    ReportOutcome,
    ResearchCritique,
    ResearchObligation,
    ResearchProposal,
    ResearchReport,
    ResearchReportSubmission,
    ResearchState,
    utc_now,
)
from .critics import ResearchCriticAgent
from .experiment import ExperimentAgent
from .literature import LiteratureResearcher
from .theorem_prover import TheoremProverAgent
from .toolsets import artifact_refs_from_observation, research_execution_toolset


FINAL_RESEARCH_REPORT_TOOL_NAME = "submit_research_report"
FINAL_OBLIGATION_RUN_TOOL_NAME = "submit_obligation_run"


class ResearchAgent:
    def __init__(self, store: ArtifactStore, router: LLMRouter, *, prompt_dir: str | None = None):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir
        self.literature = LiteratureResearcher(store, router, prompt_dir=prompt_dir)
        self.experiment = ExperimentAgent(store, router.experimenter)
        self.critic = ResearchCriticAgent(store, router, prompt_dir=prompt_dir)
        self.theorem_prover = TheoremProverAgent(store, router, prompt_dir=prompt_dir)

    def run(
        self,
        proposal: ResearchProposal,
        state: ResearchState,
    ) -> tuple[ResearchReport, str]:
        task = self.store.read_text(ArtifactStore.RESEARCH_TASK)
        context = self._build_research_context(
            task=task,
            state=state,
            proposal=proposal,
        )

        report, trace = self._generate_report_with_tools(proposal, context)
        report = self._normalize_report(report, proposal)
        report = self._add_complexity_verification_requirements(report)

        iteration_dir = self.store.create_iteration_dir(state.iteration)
        trace_refs = self._write_research_tool_trace(iteration_dir, report=report, trace=trace)
        report = self._reconcile_tool_trace(report, trace=trace, trace_refs=trace_refs)

        final_context = self._final_critic_context(context, report, trace)
        report, critique = self.critic.review(report, context=final_context)
        self._add_forced_obligations(report, critique)
        report = self.critic.enforce_evidence_statuses(report)

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
        # Legacy broad reports are audit artifacts only. Canonical claim acceptance now flows
        # through ObligationRunValidator + CommitManager after obligation runs are validated.
        return report, report_ref.path

    def run_obligation(
        self,
        *,
        obligation: ResearchObligation,
        state: ResearchState,
    ) -> tuple[ObligationRun, str, str]:
        """Run exactly one obligation.

        This is the obligation-centered harness path.  The returned run is only an attempt;
        deterministic validation and commit happen outside the agent.
        """
        task = self.store.read_text(ArtifactStore.RESEARCH_TASK)
        context = {
            "research_task_md": task,
            "research_state": state.model_dump(mode="json"),
            "assigned_obligation": obligation.model_dump(mode="json"),
            "artifact_manifest": self.store.artifact_manifest(max_items=120),
            "instructions": (
                "Execute only the assigned obligation. Generate factual claim statements only "
                "for findings established by this obligation; never submit a meta-claim that "
                "the proposal succeeded. If the obligation cannot be fulfilled, return outcome "
                "`blocked` or `failed` with precise blockers. Reference any used tool_result_id "
                "values in the flat final submission's tool_result_ids."
            ),
        }
        run, trace = self._generate_obligation_run_with_tools(obligation, context)
        run.obligation_id = obligation.obligation_id
        run.proposal_id = obligation.proposal_id
        run = self._reconcile_obligation_tool_trace(run, trace=trace)

        iteration_dir = self.store.create_iteration_dir(state.iteration)
        trace_refs = self._write_obligation_tool_trace(iteration_dir, run=run, trace=trace)
        for ref in trace_refs:
            if ref.path not in {existing.path for existing in run.artifact_refs}:
                run.artifact_refs.append(ref)
        run_ref = self.store.write_json(f"{iteration_dir}/obligation_run_{run.run_id}.json", run)
        if run_ref.path not in {existing.path for existing in run.artifact_refs}:
            run.artifact_refs.append(run_ref)
            self.store.write_json(f"{iteration_dir}/obligation_run_{run.run_id}.json", run)
        self.store.write_text(
            f"{iteration_dir}/obligation_run_{run.run_id}.md",
            _render_obligation_run_markdown(run, obligation),
        )
        return run, run_ref.path, trace_refs[0].path

    def _build_research_context(
        self,
        *,
        task: str,
        state: ResearchState,
        proposal: ResearchProposal,
    ) -> dict[str, object]:
        return {
            "research_task_md": task,
            "research_state": state.model_dump(mode="json"),
            "proposal": proposal.model_dump(mode="json"),
            "artifact_manifest": self.store.artifact_manifest(max_items=200),
            "workspace_memory_instructions": (
                "The artifact_manifest is a compact index of durable workspace memory. "
                "Do not assume artifact contents that are not included in this prompt. "
                "Use read_artifact or read_jsonl_records when details from prior claims, "
                "literature answers, reports, or traces materially affect the research report. "
                "Use query_literature for new literature lookups rather than relying on stale summaries."
            ),
        }

    def _generate_report_with_tools(
        self, proposal: ResearchProposal, context: Any
    ) -> tuple[ResearchReport, dict[str, Any]]:
        prompt_payload = {
            "instruction": (
                "Think privately. Use native tool calls for literature checks, Lean attempts, "
                "or experiment requests when they materially affect the report. Finish only by "
                f"calling `{FINAL_RESEARCH_REPORT_TOOL_NAME}`."
            ),
            "context": context,
        }
        messages = [
            {
                "role": "system",
                "content": render_prompt("research_agent", override_dir=self.prompt_dir),
            },
            {
                "role": "user",
                "content": (
                    "Execute the selected proposal using durable, auditable evidence. "
                    "Use tool_result_ids from observations in the final flat submission when a "
                    "claim depends on a tool result.\nContext:\n"
                    + compact_json_dumps(prompt_payload)
                ),
            },
        ]
        toolset = research_execution_toolset(
            store=self.store,
            literature=self.literature,
            theorem_prover=self.theorem_prover,
            experiment=self.experiment,
        )
        submission, trace = self.router.complete_structured_with_tools(
            task_type="research_execution",
            messages=messages,
            tools=toolset.openai_tools(),
            tool_executors=toolset.executors(),
            schema=ResearchReportSubmission,
            final_tool_name=FINAL_RESEARCH_REPORT_TOOL_NAME,
            mock_output=_report_submission_from_report(self._mock_report(proposal))
            if self.router.dry_run
            else None,
        )
        return _research_report_from_submission(submission, proposal), trace

    def _generate_obligation_run_with_tools(
        self,
        obligation: ResearchObligation,
        context: Any,
    ) -> tuple[ObligationRun, dict[str, Any]]:
        prompt_payload = {
            "instruction": (
                "Think privately. Use native tool calls only when they materially affect this "
                "one obligation. Finish only by calling "
                f"`{FINAL_OBLIGATION_RUN_TOOL_NAME}`."
            ),
            "context": context,
        }
        system_prompt = (
            "You are the existing research agent, but this run is scoped to exactly one "
            "obligation. Finish with a flat ObligationRunSubmission, not nested Pydantic "
            "objects. Generate claim_statements only for factual findings established by this "
            "obligation. Do not create unrelated claims and do not state that the proposal "
            "succeeds. If evidence is missing, return `blocked` or `failed` with precise "
            "blockers. Use the complete JSON schema inserted below for the final submission:\n"
            "{{ObligationRunSubmission}}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Execute this assigned obligation with durable, auditable evidence.\nContext:\n"
                    + compact_json_dumps(prompt_payload)
                ),
            },
        ]
        toolset = research_execution_toolset(
            store=self.store,
            literature=self.literature,
            theorem_prover=self.theorem_prover,
            experiment=self.experiment,
            final_tool_name=FINAL_OBLIGATION_RUN_TOOL_NAME,
            final_schema=ObligationRunSubmission,
            final_tool_description=(
                "Commit the final flat ObligationRunSubmission for the assigned obligation. "
                "Use strings/lists for claim_statements, evidence handles, blockers, and "
                "child obligations; do not submit nested objects."
            ),
        )
        submission, trace = self.router.complete_structured_with_tools(
            task_type="research_execution",
            messages=messages,
            tools=toolset.openai_tools(),
            tool_executors=toolset.executors(),
            schema=ObligationRunSubmission,
            final_tool_name=FINAL_OBLIGATION_RUN_TOOL_NAME,
            mock_output=self._mock_obligation_submission(obligation) if self.router.dry_run else None,
        )
        return _obligation_run_from_submission(submission, obligation), trace

    def _write_research_tool_trace(
        self,
        iteration_dir: str,
        *,
        report: ResearchReport,
        trace: dict[str, Any],
    ) -> list[ArtifactRef]:
        payload = {
            "report_id": report.report_id,
            "proposal_id": report.proposal_id,
            "private_reasoning": "redacted_not_logged_or_replayed",
            "trace": trace,
        }
        json_ref = self.store.write_json(
            f"{iteration_dir}/research_tool_trace_{report.report_id}.json", payload
        )
        lines = [f"# Research Tool Trace `{report.report_id}`", ""]
        lines.extend(
            [
                "Private chain-of-thought/reasoning is intentionally not logged.",
                "Only external tool calls, arguments, observations, and finalization metadata appear here.",
                "",
            ]
        )
        for item in trace.get("tool_calls", []):
            lines.append(
                f"## Turn {item.get('turn', '?')}: `{item.get('name', '')}` "
                f"({item.get('call_id', '')})"
            )
            lines.extend(
                [
                    "",
                    "Arguments:",
                    "",
                    "```json",
                    json.dumps(item.get("arguments", {}), indent=2, sort_keys=True),
                    "```",
                    "",
                    "Observation:",
                    "",
                    "```json",
                    json.dumps(item.get("observation", {}), indent=2, sort_keys=True),
                    "```",
                    "",
                ]
            )
        if trace.get("finalization"):
            lines.extend(
                [
                    "## Finalization",
                    "",
                    "```json",
                    json.dumps(trace["finalization"], indent=2, sort_keys=True),
                    "```",
                    "",
                ]
            )
        md_ref = self.store.write_text(
            f"{iteration_dir}/research_tool_trace_{report.report_id}.md",
            "\n".join(lines).rstrip() + "\n",
        )
        return [json_ref, md_ref]

    def _write_obligation_tool_trace(
        self,
        iteration_dir: str,
        *,
        run: ObligationRun,
        trace: dict[str, Any],
    ) -> list[ArtifactRef]:
        payload = {
            "run_id": run.run_id,
            "obligation_id": run.obligation_id,
            "proposal_id": run.proposal_id,
            "private_reasoning": "redacted_not_logged_or_replayed",
            "trace": trace,
        }
        json_ref = self.store.write_json(
            f"{iteration_dir}/obligation_tool_trace_{run.run_id}.json", payload
        )
        lines = [f"# Obligation Tool Trace `{run.run_id}`", ""]
        lines.extend(
            [
                "Private chain-of-thought/reasoning is intentionally not logged.",
                "Only external tool calls, arguments, observations, and finalization metadata appear here.",
                "",
            ]
        )
        for item in trace.get("tool_calls", []):
            lines.append(
                f"## Turn {item.get('turn', '?')}: `{item.get('name', '')}` "
                f"({item.get('call_id', '')})"
            )
            lines.extend(
                [
                    "",
                    "Arguments:",
                    "",
                    "```json",
                    json.dumps(item.get("arguments", {}), indent=2, sort_keys=True),
                    "```",
                    "",
                    "Observation:",
                    "",
                    "```json",
                    json.dumps(item.get("observation", {}), indent=2, sort_keys=True),
                    "```",
                    "",
                ]
            )
        if trace.get("finalization"):
            lines.extend(
                [
                    "## Finalization",
                    "",
                    "```json",
                    json.dumps(trace["finalization"], indent=2, sort_keys=True),
                    "```",
                    "",
                ]
            )
        md_ref = self.store.write_text(
            f"{iteration_dir}/obligation_tool_trace_{run.run_id}.md",
            "\n".join(lines).rstrip() + "\n",
        )
        return [json_ref, md_ref]

    def _reconcile_tool_trace(
        self,
        report: ResearchReport,
        *,
        trace: dict[str, Any],
        trace_refs: list[ArtifactRef],
    ) -> ResearchReport:
        """Deterministically attach only real tool-produced artifacts to the report."""
        for ref in trace_refs:
            self._append_report_ref(report, ref)

        tool_results = self._tool_results_by_id(trace)
        self._attach_explicit_tool_evidence(report, tool_results)
        for tool_result in tool_results.values():
            name = str(tool_result.get("name") or "")
            observation = tool_result.get("observation") or {}
            if name == "query_literature":
                self._record_literature_observation(report, observation)
            elif name == "attempt_lean_proof":
                self._record_lean_observation(report, tool_result)
            elif name == "run_experiment":
                self._record_experiment_observation(report, observation)
        return report

    def _reconcile_obligation_tool_trace(
        self, run: ObligationRun, *, trace: dict[str, Any]
    ) -> ObligationRun:
        tool_results = self._tool_results_by_id(trace)
        for evidence in run.evidence:
            for result_id in evidence.tool_result_ids:
                tool_result = tool_results.get(result_id)
                if tool_result is None:
                    continue
                observation = tool_result.get("observation") or {}
                name = str(tool_result.get("name") or "")
                refs = artifact_refs_from_observation(observation)
                if name == "attempt_lean_proof" and evidence.evidence_type == EvidenceType.lean_proof:
                    if observation.get("proof_status") == "proved":
                        for ref in refs:
                            self._append_evidence_ref(evidence, ref)
                            if ref.path not in {existing.path for existing in run.artifact_refs}:
                                run.artifact_refs.append(ref)
                        evidence.verifier = evidence.verifier or "LEAPHarness"
                        evidence.confidence = max(evidence.confidence, 1.0)
                elif name == "query_literature" and evidence.evidence_type == EvidenceType.citation:
                    ledger_ref = _artifact_ref_from_plain(observation.get("ledger_ref"))
                    if ledger_ref is not None:
                        self._append_evidence_ref(evidence, ledger_ref)
                        if ledger_ref.path not in {existing.path for existing in run.artifact_refs}:
                            run.artifact_refs.append(ledger_ref)
                    for key in _citation_keys_from_literature_observation(observation):
                        if key not in evidence.citation_keys:
                            evidence.citation_keys.append(key)
                    evidence.verifier = evidence.verifier or "LiteratureResearcher"
                    evidence.confidence = max(evidence.confidence, 0.5)
                elif name == "run_experiment" and evidence.evidence_type == EvidenceType.experiment:
                    for ref in refs:
                        self._append_evidence_ref(evidence, ref)
                        if ref.path not in {existing.path for existing in run.artifact_refs}:
                            run.artifact_refs.append(ref)
                    evidence.verifier = evidence.verifier or "DockerPiExperimenter"
                    evidence.confidence = max(evidence.confidence, 0.7)
        return run

    def _tool_results_by_id(self, trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for item in trace.get("tool_calls", []):
            if not isinstance(item, dict):
                continue
            observation = item.get("observation")
            if not isinstance(observation, dict):
                continue
            result_id = observation.get("tool_result_id") or observation.get("answer_id")
            if not result_id:
                continue
            results[str(result_id)] = item
        return results

    def _attach_explicit_tool_evidence(
        self, report: ResearchReport, tool_results: dict[str, dict[str, Any]]
    ) -> None:
        for evidence in self._all_evidence_records(report):
            for result_id in evidence.tool_result_ids:
                tool_result = tool_results.get(result_id)
                if tool_result is None:
                    continue
                observation = tool_result.get("observation") or {}
                name = str(tool_result.get("name") or "")
                refs = artifact_refs_from_observation(observation)
                if name == "attempt_lean_proof":
                    if observation.get("proof_status") != "proved":
                        continue
                    if evidence.evidence_type != EvidenceType.lean_proof:
                        continue
                    for ref in refs:
                        self._append_evidence_ref(evidence, ref)
                        self._append_report_ref(report, ref)
                    evidence.verifier = evidence.verifier or "LEAPHarness"
                    evidence.confidence = max(evidence.confidence, 1.0)
                elif name == "query_literature":
                    if evidence.evidence_type != EvidenceType.citation:
                        continue
                    ledger_ref = _artifact_ref_from_plain(observation.get("ledger_ref"))
                    if ledger_ref is not None:
                        self._append_evidence_ref(evidence, ledger_ref)
                        self._append_report_ref(report, ledger_ref)
                    for key in _citation_keys_from_literature_observation(observation):
                        if key not in evidence.citation_keys:
                            evidence.citation_keys.append(key)
                    evidence.verifier = evidence.verifier or "LiteratureResearcher"
                    evidence.confidence = max(evidence.confidence, 0.5)
                elif name == "run_experiment":
                    if evidence.evidence_type != EvidenceType.experiment:
                        continue
                    for ref in refs:
                        self._append_evidence_ref(evidence, ref)
                        self._append_report_ref(report, ref)
                    evidence.verifier = evidence.verifier or "DockerPiExperimenter"
                    evidence.confidence = max(evidence.confidence, 0.7)

    def _record_literature_observation(self, report: ResearchReport, observation: dict[str, Any]) -> None:
        query = str(observation.get("query") or "")
        result_count = int(observation.get("result_count") or 0)
        ledger_ref = _artifact_ref_from_plain(observation.get("ledger_ref"))
        if ledger_ref is not None:
            self._append_report_ref(report, ledger_ref)
        if result_count <= 0:
            issue = f"Local LiteratureDB query `{query}` returned no results."
            if query and issue not in report.unresolved_issues:
                report.unresolved_issues.append(issue)
            return
        citation_keys = _citation_keys_from_literature_observation(observation)
        evidence = EvidenceRecord(
            evidence_type=EvidenceType.citation,
            summary=(
                f"Native research tool LiteratureDB query `{query}` returned {result_count} "
                "mapped result(s). Claim-local citation evidence is still required before "
                "accepting literature claims."
            ),
            artifact_refs=[ledger_ref] if ledger_ref is not None else [],
            citation_keys=citation_keys,
            tool_result_ids=[str(observation.get("tool_result_id"))]
            if observation.get("tool_result_id")
            else [],
            verifier="LiteratureResearcher",
            confidence=0.5,
        )
        if not _same_evidence_present(report.evidence, evidence):
            report.evidence.append(evidence)
        self._add_literature_dependencies(report, observation)

    def _add_literature_dependencies(self, report: ResearchReport, observation: dict[str, Any]) -> None:
        query = str(observation.get("query") or "")
        used_for = f"Native research tool query: {query}"
        existing = {(dep.citation_key, dep.used_for) for dep in report.literature_dependencies}
        for result in observation.get("results") or []:
            if not isinstance(result, dict):
                continue
            citation_key = str(result.get("citation_key") or "")
            if not citation_key or (citation_key, used_for) in existing:
                continue
            report.literature_dependencies.append(
                LiteratureDependency(
                    citation_key=citation_key,
                    title=str(result.get("title") or ""),
                    used_for=used_for,
                    provenance=(
                        f"LiteratureDB answer {observation.get('answer_id')}; "
                        f"label={result.get('label', '')}"
                    ),
                )
            )
            existing.add((citation_key, used_for))

    def _record_lean_observation(self, report: ResearchReport, tool_result: dict[str, Any]) -> None:
        observation = tool_result.get("observation") or {}
        if not isinstance(observation, dict):
            return
        refs = artifact_refs_from_observation(observation)
        for ref in refs:
            self._append_report_ref(report, ref)
        result_id = str(observation.get("tool_result_id") or "")
        proof_status = str(observation.get("proof_status") or "")
        statement = _tool_lean_statement(tool_result)
        matched = False
        for obligation in report.proof_obligations:
            if _statements_match(obligation.statement, statement):
                matched = True
                for ref in refs:
                    if ref.path not in {existing.path for existing in obligation.artifact_refs}:
                        obligation.artifact_refs.append(ref)
                if proof_status == "proved":
                    obligation.status = "proved"
                elif proof_status == "partially_proved":
                    obligation.status = "in_progress"
                else:
                    obligation.status = "blocked"
        if proof_status == "proved":
            evidence = EvidenceRecord(
                evidence_type=EvidenceType.lean_proof,
                summary=f"LEAP returned proved for native tool result `{result_id}`.",
                artifact_refs=refs,
                tool_result_ids=[result_id] if result_id else [],
                verifier="LEAPHarness",
                confidence=1.0,
            )
            if not _same_evidence_present(report.evidence, evidence):
                report.evidence.append(evidence)
            self._attach_matching_claim_evidence(report, statement, evidence)
        elif not matched:
            issue = (
                f"LEAP native tool result `{result_id}` for `{statement}` returned "
                f"status `{proof_status or 'unknown'}` without a matching final proof obligation."
            )
            if issue not in report.unresolved_issues:
                report.unresolved_issues.append(issue)

    def _attach_matching_claim_evidence(
        self, report: ResearchReport, statement: str, evidence: EvidenceRecord
    ) -> None:
        for claim in report.claims_generated:
            if not _statements_match(claim.statement, statement):
                continue
            claim.evidence.append(evidence.model_copy(deep=True))
            claim.status = ClaimStatus.proved_by_lean

    def _record_experiment_observation(self, report: ResearchReport, observation: dict[str, Any]) -> None:
        refs = artifact_refs_from_observation(observation)
        for ref in refs:
            self._append_report_ref(report, ref)
        result_id = str(observation.get("tool_result_id") or "")
        payload = observation.get("experiment_result")
        if not isinstance(payload, dict):
            issue = f"Experiment tool result `{result_id}` did not include a structured ExperimentResult."
            if issue not in report.unresolved_issues:
                report.unresolved_issues.append(issue)
            return
        try:
            result = ExperimentResult.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 - preserve report generation with issue marker
            issue = f"Experiment tool result `{result_id}` failed ExperimentResult validation: {exc}"
            if issue not in report.unresolved_issues:
                report.unresolved_issues.append(issue)
            return
        existing_ids = {existing.run_id for existing in report.experimental_results}
        if result.run_id not in existing_ids:
            report.experimental_results.append(result)
        evidence = EvidenceRecord(
            evidence_type=EvidenceType.experiment,
            summary=f"Dockerized pi experiment `{result.run_id}` completed: {result.summary}",
            artifact_refs=result.artifact_refs,
            tool_result_ids=[result_id] if result_id else [],
            verifier="DockerPiExperimenter",
            confidence=0.7,
        )
        if not _same_evidence_present(report.evidence, evidence):
            report.evidence.append(evidence)

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

    def _final_critic_context(
        self,
        base_context: Any,
        report: ResearchReport,
        trace: dict[str, Any],
    ) -> dict[str, object]:
        return {
            "base_context": base_context,
            "current_report_after_native_tool_loop": report.model_dump(mode="json"),
            "native_tool_trace_summary": _trace_summary(trace),
        }

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

    def _append_evidence_ref(self, evidence: EvidenceRecord, ref: ArtifactRef) -> None:
        if ref.path not in {existing.path for existing in evidence.artifact_refs}:
            evidence.artifact_refs.append(ref)

    def _all_evidence_records(self, report: ResearchReport) -> list[EvidenceRecord]:
        records = list(report.evidence)
        for claim in report.claims_generated:
            records.extend(claim.evidence)
        return records

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

    def _mock_obligation_submission(self, obligation: ResearchObligation) -> ObligationRunSubmission:
        return ObligationRunSubmission(
            outcome="fulfilled",
            summary=(
                "Dry-run obligation execution records a substantive mock argument for the assigned "
                "obligation. Real runs must provide tool-backed or derivation-backed evidence before "
                "the deterministic validator accepts generated claims."
            ),
            claim_statements=[f"Dry-run factual finding for obligation: {obligation.statement}"],
            evidence_type=EvidenceType.informal_argument,
            evidence_summary="Dry-run mock evidence for exercising the obligation harness only.",
        )

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


def _research_report_from_submission(
    submission: ResearchReportSubmission, proposal: ResearchProposal
) -> ResearchReport:
    evidence = _evidence_from_flat_submission(
        evidence_type=submission.evidence_type,
        evidence_summary=submission.evidence_summary or submission.executive_summary,
        tool_result_ids=submission.tool_result_ids,
        citation_keys=submission.citation_keys,
    )
    claims = [
        ClaimRecord(
            claim_type=submission.claim_type,
            statement=statement,
            status=_claim_status_for_evidence(submission.evidence_type, submission.claim_type),
            evidence=[evidence.model_copy(deep=True)] if evidence is not None else [],
            related_proposal_ids=[proposal.proposal_id],
        )
        for statement in _unique_nonempty(submission.claim_statements)
    ]
    proof_obligations = [
        ProofObligation(statement=statement)
        for statement in _unique_nonempty(submission.proof_obligation_statements)
    ]
    return ResearchReport(
        proposal_id=proposal.proposal_id,
        outcome=submission.outcome,
        executive_summary=submission.executive_summary,
        claims_generated=claims,
        evidence=[evidence] if evidence is not None else [],
        proof_obligations=proof_obligations,
        unresolved_issues=submission.unresolved_issues,
        proposed_next_steps=submission.proposed_next_steps,
        required_verifications=submission.required_verifications,
    )


def _report_submission_from_report(report: ResearchReport) -> ResearchReportSubmission:
    first_evidence = report.evidence[0] if report.evidence else None
    if first_evidence is None and report.claims_generated and report.claims_generated[0].evidence:
        first_evidence = report.claims_generated[0].evidence[0]
    return ResearchReportSubmission(
        outcome=report.outcome,
        executive_summary=report.executive_summary,
        claim_statements=[claim.statement for claim in report.claims_generated],
        claim_type=report.claims_generated[0].claim_type if report.claims_generated else ClaimType.other,
        evidence_type=first_evidence.evidence_type if first_evidence else EvidenceType.informal_argument,
        evidence_summary=first_evidence.summary if first_evidence else "",
        tool_result_ids=first_evidence.tool_result_ids if first_evidence else [],
        citation_keys=first_evidence.citation_keys if first_evidence else [],
        proof_obligation_statements=[obligation.statement for obligation in report.proof_obligations],
        unresolved_issues=report.unresolved_issues,
        proposed_next_steps=report.proposed_next_steps,
        required_verifications=report.required_verifications,
    )


def _obligation_run_from_submission(
    submission: ObligationRunSubmission, obligation: ResearchObligation
) -> ObligationRun:
    evidence = _evidence_from_flat_submission(
        evidence_type=submission.evidence_type,
        evidence_summary=submission.evidence_summary or submission.summary,
        tool_result_ids=submission.tool_result_ids,
        citation_keys=submission.citation_keys,
    )
    claim_type = _claim_type_for_obligation(obligation, submission.evidence_type)
    claims = [
        ClaimRecord(
            claim_type=claim_type,
            statement=statement,
            status=_claim_status_for_evidence(submission.evidence_type, claim_type),
            related_proposal_ids=[obligation.proposal_id] if obligation.proposal_id else [],
        )
        for statement in _unique_nonempty(submission.claim_statements)
    ]
    child_obligations = [
        _child_obligation_from_statement(statement, proposal_id=obligation.proposal_id)
        for statement in _unique_nonempty(submission.child_obligation_statements)
    ]
    return ObligationRun(
        obligation_id=obligation.obligation_id,
        proposal_id=obligation.proposal_id,
        outcome=submission.outcome,
        summary=submission.summary,
        claims_generated=claims,
        evidence=[evidence] if evidence is not None else [],
        child_obligations=child_obligations,
        unresolved_blockers=submission.unresolved_blockers,
    )


def _evidence_from_flat_submission(
    *,
    evidence_type: EvidenceType,
    evidence_summary: str,
    tool_result_ids: list[str],
    citation_keys: list[str],
) -> EvidenceRecord | None:
    summary = evidence_summary.strip()
    ids = _unique_nonempty(tool_result_ids)
    citations = _unique_nonempty(citation_keys)
    if not summary and not ids and not citations:
        return None
    return EvidenceRecord(
        evidence_type=evidence_type,
        summary=summary or "Flat final submission supplied evidence handles.",
        tool_result_ids=ids,
        citation_keys=citations,
        confidence=_initial_confidence_for_evidence(evidence_type),
    )


def _claim_type_for_obligation(obligation: ResearchObligation, evidence_type: EvidenceType) -> ClaimType:
    if evidence_type == EvidenceType.citation or obligation.kind == "literature":
        return ClaimType.literature
    if evidence_type == EvidenceType.experiment or obligation.kind == "experiment":
        return ClaimType.experimental
    if obligation.kind == "derivation":
        lowered = obligation.statement.lower()
        if any(word in lowered for word in ["complexity", "runtime", "resource", "bound"]):
            return ClaimType.complexity
        return ClaimType.mathematical
    if obligation.kind == "proof":
        return ClaimType.theorem_statement
    return ClaimType.mathematical


def _claim_status_for_evidence(evidence_type: EvidenceType, claim_type: ClaimType) -> ClaimStatus:
    if evidence_type == EvidenceType.lean_proof:
        return ClaimStatus.proved_by_lean
    if evidence_type == EvidenceType.citation and claim_type == ClaimType.literature:
        return ClaimStatus.cited
    if evidence_type == EvidenceType.experiment:
        return ClaimStatus.experimentally_supported
    if evidence_type == EvidenceType.counterexample:
        return ClaimStatus.refuted
    if evidence_type == EvidenceType.informal_argument:
        return ClaimStatus.informal_argument
    return ClaimStatus.needs_review


def _initial_confidence_for_evidence(evidence_type: EvidenceType) -> float:
    if evidence_type == EvidenceType.lean_proof:
        return 1.0
    if evidence_type == EvidenceType.experiment:
        return 0.7
    if evidence_type == EvidenceType.citation:
        return 0.5
    if evidence_type == EvidenceType.informal_argument:
        return 0.3
    return 0.0


def _child_obligation_from_statement(statement: str, *, proposal_id: str) -> ResearchObligation:
    kind = _classify_obligation_kind(statement)
    return ResearchObligation(
        proposal_id=proposal_id,
        statement=statement,
        kind=kind,
        required_evidence=_required_evidence_for_kind(kind),
    )


def _classify_obligation_kind(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ["citation", "literature", "paper", "theorem from"]):
        return "literature"
    if any(word in lowered for word in ["lean", "formal", "proof", "prove theorem"]):
        return "proof"
    if any(word in lowered for word in ["experiment", "simulation", "numerical", "small-instance"]):
        return "experiment"
    if any(word in lowered for word in ["complexity", "runtime", "asymptotic", "derive", "bound", "lemma"]):
        return "derivation"
    if any(word in lowered for word in ["consistent", "contradict", "conflict"]):
        return "consistency"
    return "derivation"


def _required_evidence_for_kind(kind: str) -> list[EvidenceType]:
    if kind == "literature":
        return [EvidenceType.citation]
    if kind == "proof":
        return [EvidenceType.lean_proof]
    if kind == "experiment":
        return [EvidenceType.experiment]
    if kind == "consistency":
        return [EvidenceType.external_tool]
    return [EvidenceType.informal_argument]


def _unique_nonempty(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in items if item and item.strip()))


def _render_obligation_run_markdown(
    run: ObligationRun, obligation: ResearchObligation
) -> str:
    lines = [f"# Obligation Run `{run.run_id}`", ""]
    lines.append(f"**Obligation:** `{obligation.obligation_id}`")
    if obligation.proposal_id:
        lines.append(f"**Proposal:** `{obligation.proposal_id}`")
    lines.append(f"**Outcome:** `{run.outcome}`")
    lines.extend(["", "## Obligation", obligation.statement, ""])
    lines.extend(["## Summary", run.summary, ""])
    if run.claims_generated:
        lines.append("## Claims generated")
        for claim in run.claims_generated:
            lines.append(f"- `{claim.claim_id}` [{claim.claim_type.value}/{claim.status.value}]: {claim.statement}")
        lines.append("")
    if run.evidence:
        lines.append("## Evidence")
        for evidence in run.evidence:
            tools = f" tool_results={evidence.tool_result_ids}" if evidence.tool_result_ids else ""
            cites = f" citations={evidence.citation_keys}" if evidence.citation_keys else ""
            lines.append(f"- {evidence.evidence_type.value}:{tools}{cites} {evidence.summary}")
        lines.append("")
    if run.unresolved_blockers:
        lines.append("## Unresolved blockers")
        for blocker in run.unresolved_blockers:
            lines.append(f"- {blocker}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _artifact_ref_from_plain(value: Any) -> ArtifactRef | None:
    try:
        return ArtifactRef.model_validate(value)
    except Exception:
        return None


def _citation_keys_from_literature_observation(observation: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for result in observation.get("results") or []:
        if isinstance(result, dict) and result.get("citation_key"):
            keys.append(str(result["citation_key"]))
    return list(dict.fromkeys(keys))


def _same_evidence_present(existing: list[EvidenceRecord], candidate: EvidenceRecord) -> bool:
    candidate_ids = set(candidate.tool_result_ids)
    candidate_refs = {ref.path for ref in candidate.artifact_refs}
    for evidence in existing:
        if candidate_ids and candidate_ids.intersection(evidence.tool_result_ids):
            return True
        if candidate_refs and candidate_refs == {ref.path for ref in evidence.artifact_refs}:
            return True
    return False


def _tool_lean_statement(tool_result: dict[str, Any]) -> str:
    observation = tool_result.get("observation") if isinstance(tool_result, dict) else None
    if isinstance(observation, dict):
        root_goal = observation.get("root_goal")
        if isinstance(root_goal, dict) and root_goal.get("statement"):
            return str(root_goal["statement"])
    arguments = tool_result.get("arguments") if isinstance(tool_result, dict) else None
    if isinstance(arguments, dict) and arguments.get("statement"):
        return str(arguments["statement"])
    return ""


def _statements_match(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return _normalize_statement(left) == _normalize_statement(right)


def _normalize_statement(statement: str) -> str:
    text = statement.strip()
    for prefix in ["lean:", "Lean:", "LEAN:"]:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    return " ".join(text.split())


def _trace_summary(trace: dict[str, Any]) -> dict[str, Any]:
    calls = []
    for item in trace.get("tool_calls", []):
        if not isinstance(item, dict):
            continue
        observation = item.get("observation")
        calls.append(
            {
                "turn": item.get("turn"),
                "call_id": item.get("call_id"),
                "name": item.get("name"),
                "status": item.get("status"),
                "tool_result_id": observation.get("tool_result_id")
                if isinstance(observation, dict)
                else None,
                "proof_status": observation.get("proof_status")
                if isinstance(observation, dict)
                else None,
                "result_count": observation.get("result_count")
                if isinstance(observation, dict)
                else None,
            }
        )
    return {
        "private_reasoning": "redacted_not_logged_or_replayed",
        "tool_calls": calls,
        "finalization": to_plain(trace.get("finalization")),
    }
