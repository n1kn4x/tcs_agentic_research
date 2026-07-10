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
    LiteratureDependency,
    ProofObligation,
    ReportOutcome,
    ResearchCritique,
    ResearchProposal,
    ResearchReport,
    ResearchState,
    utc_now,
)
from .critics import ResearchCriticAgent
from .experiment import ExperimentAgent
from .literature import LiteratureResearcher
from .theorem_prover import TheoremProverAgent
from .toolsets import artifact_refs_from_observation, research_execution_toolset


FINAL_RESEARCH_REPORT_TOOL_NAME = "submit_research_report"


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
    ) -> tuple[ResearchReport, str]:
        task = self.store.read_text(ArtifactStore.RESEARCH_TASK)
        context = self._build_research_context(
            task=task,
            state=state,
            proposal=proposal,
            literature_answers=self._initial_literature_answers(proposal),
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
        self.store.append_claims(report.claims_generated)
        return report, report_ref.path

    def _initial_literature_answers(self, proposal: ResearchProposal) -> list[dict[str, object]]:
        answers: list[dict[str, object]] = []
        for query in proposal.literature_queries[:5]:
            if query.strip():
                answers.append(self.literature.answer_query(query, limit=3).model_dump(mode="json"))
        return answers

    def _build_research_context(
        self,
        *,
        task: str,
        state: ResearchState,
        proposal: ResearchProposal,
        literature_answers: list[dict[str, object]],
    ) -> dict[str, object]:
        return {
            "research_task_md": task,
            "research_state": state.model_dump(mode="json"),
            "proposal": proposal.model_dump(mode="json"),
            "local_literature_answers": literature_answers,
            "recent_claims": self.store.read_jsonl(ArtifactStore.CLAIM_LEDGER, limit=30),
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
                    "Use tool_result_ids from observations in final EvidenceRecord.tool_result_ids "
                    "when a claim depends on a tool result.\nContext:\n"
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
        return self.router.complete_structured_with_tools(
            task_type="research_execution",
            messages=messages,
            tools=toolset.openai_tools(),
            tool_executors=toolset.executors(),
            schema=ResearchReport,
            final_tool_name=FINAL_RESEARCH_REPORT_TOOL_NAME,
            mock_output=self._mock_report(proposal) if self.router.dry_run else None,
        )

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
                self._record_experiment_request(report, observation)
        return report

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
                    # The current description-only experiment backend records requests only.
                    # It does not produce certifying experimental evidence.
                    continue

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

    def _record_experiment_request(self, report: ResearchReport, observation: dict[str, Any]) -> None:
        refs = artifact_refs_from_observation(observation)
        for ref in refs:
            self._append_report_ref(report, ref)
        result_id = str(observation.get("tool_result_id") or "")
        description = str(observation.get("description") or "")
        issue = (
            f"Experiment request `{result_id}` was recorded but no coding-agent experiment "
            f"backend is configured. Request: {description}"
        )
        if issue not in report.unresolved_issues:
            report.unresolved_issues.append(issue)

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
