"""Research execution agent using native OpenAI/vLLM tool calls."""

from __future__ import annotations

import json
from typing import Any

from ..artifact_store import ArtifactStore
from ..llm import LLMRouter
from ..prompt_serialization import compact_json_dumps
from ..schemas import (
    ArtifactRef,
    ClaimRecord,
    ClaimStatus,
    ClaimType,
    EvidenceRecord,
    EvidenceType,
    ObligationRun,
    ObligationRunSubmission,
    ResearchObligation,
    ResearchState,
)
from .experiment import ExperimentAgent
from .literature import LiteratureResearcher
from .theorem_prover import TheoremProverAgent
from .toolsets import artifact_refs_from_observation, research_execution_toolset


FINAL_OBLIGATION_RUN_TOOL_NAME = "submit_obligation_run"


class ResearchAgent:
    def __init__(self, store: ArtifactStore, router: LLMRouter, *, prompt_dir: str | None = None):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir
        self.literature = LiteratureResearcher(store, router, prompt_dir=prompt_dir)
        self.experiment = ExperimentAgent(store, router.experimenter)
        self.theorem_prover = TheoremProverAgent(store, router, prompt_dir=prompt_dir)

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
        literature_tools_enabled = _obligation_needs_literature_tools(obligation)
        pre_obligation_literature_extraction = (
            self._auto_extract_imported_literature() if literature_tools_enabled else None
        )
        context = {
            "research_task_md": task,
            "research_state": state.model_dump(mode="json"),
            "assigned_obligation": obligation.model_dump(mode="json"),
            "artifact_manifest": self.store.artifact_manifest(max_items=120),
            "pre_obligation_literature_extraction": pre_obligation_literature_extraction,
            "instructions": (
                "Execute only the assigned obligation. Generate factual claim statements only "
                "for findings established by this obligation; never submit a meta-claim that "
                "the proposal succeeded. If the obligation cannot be fulfilled, return outcome "
                "`blocked` or `failed` with precise blockers. Reference any used tool_result_id "
                "values in the flat final submission's tool_result_ids."
                + (
                    " For literature obligations, search/import/extract tools are available. "
                    "After importing a full-text paper, extract_paper is run automatically by "
                    "the import tools when possible; use extract_imported_papers for any "
                    "remaining papers without extracted support IDs, then query_literature."
                    if literature_tools_enabled
                    else ""
                )
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

    def _auto_extract_imported_literature(self) -> dict[str, Any]:
        """Best-effort deterministic extraction pass before a literature obligation.

        This prevents the common failure mode where papers have already been imported with text
        artifacts but no statement-level LiteratureDB support objects exist yet.
        """
        try:
            summary = self.literature.extract_imported_papers(max_papers=8, only_missing=True)
            summary["status"] = "ok"
            return summary
        except Exception as exc:  # noqa: BLE001 - do not fail the obligation prompt setup
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

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
            "You are the research agent. This run is scoped to exactly one obligation. "
            "Finish with a flat ObligationRunSubmission, not nested Pydantic "
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
        literature_tools_enabled = _obligation_needs_literature_tools(obligation)
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
            include_literature_discovery_tools=literature_tools_enabled,
            include_literature_extraction_tools=literature_tools_enabled,
            include_theorem_tools=not literature_tools_enabled,
            include_experiment_tools=not literature_tools_enabled,
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
                elif _is_literature_tool_name(name) and evidence.evidence_type == EvidenceType.citation:
                    refs = artifact_refs_from_observation(observation)
                    ledger_ref = _artifact_ref_from_plain(observation.get("ledger_ref"))
                    if ledger_ref is not None:
                        refs.append(ledger_ref)
                    for ref in refs:
                        self._append_evidence_ref(evidence, ref)
                        if ref.path not in {existing.path for existing in run.artifact_refs}:
                            run.artifact_refs.append(ref)
                    for key in _citation_keys_from_literature_observation(observation):
                        if key not in evidence.citation_keys:
                            evidence.citation_keys.append(key)
                    for support_id in _support_ids_from_literature_observation(observation):
                        if support_id not in evidence.literature_support_ids:
                            evidence.literature_support_ids.append(support_id)
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

    def _append_evidence_ref(self, evidence: EvidenceRecord, ref: ArtifactRef) -> None:
        if ref.path not in {existing.path for existing in evidence.artifact_refs}:
            evidence.artifact_refs.append(ref)

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


def _obligation_needs_literature_tools(obligation: ResearchObligation) -> bool:
    lowered = obligation.statement.lower()
    return obligation.kind == "literature" or any(
        word in lowered
        for word in [
            "arxiv",
            "citation",
            "doi",
            "extract",
            "import",
            "literature",
            "paper",
            "provenance",
            "quote",
            "source",
        ]
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
    if any(
        word in lowered
        for word in [
            "arxiv",
            "citation",
            "doi",
            "extract",
            "import",
            "literature",
            "paper",
            "provenance",
            "quote",
            "source",
            "theorem from",
        ]
    ):
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


def _is_literature_tool_name(name: str) -> bool:
    return name in {
        "query_literature",
        "search_papers",
        "import_url",
        "import_arxiv",
        "import_doi",
        "import_candidate",
        "extract_paper",
        "extract_imported_papers",
    }


def _citation_keys_from_literature_observation(observation: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for key in observation.get("citation_keys") or []:
        if key:
            keys.append(str(key))
    if observation.get("citation_key"):
        keys.append(str(observation["citation_key"]))
    paper = observation.get("paper")
    if isinstance(paper, dict) and paper.get("citation_key"):
        keys.append(str(paper["citation_key"]))
    extraction = observation.get("extraction")
    if isinstance(extraction, dict):
        for key in extraction.get("citation_keys") or []:
            if key:
                keys.append(str(key))
        if extraction.get("citation_key"):
            keys.append(str(extraction["citation_key"]))
    for item in observation.get("processed") or []:
        if isinstance(item, dict) and item.get("citation_key"):
            keys.append(str(item["citation_key"]))
    for result in observation.get("results") or []:
        if isinstance(result, dict) and result.get("citation_key"):
            keys.append(str(result["citation_key"]))
    return list(dict.fromkeys(keys))


def _support_ids_from_literature_observation(observation: dict[str, Any]) -> list[str]:
    support_ids: list[str] = []
    for support_id in observation.get("support_ids") or []:
        if support_id:
            support_ids.append(str(support_id))
    extraction = observation.get("extraction")
    if isinstance(extraction, dict):
        for support_id in extraction.get("support_ids") or []:
            if support_id:
                support_ids.append(str(support_id))
    for item in observation.get("processed") or []:
        if isinstance(item, dict):
            for support_id in item.get("support_ids") or []:
                if support_id:
                    support_ids.append(str(support_id))
    for statement in observation.get("statements") or []:
        if isinstance(statement, dict):
            for key in ["support_id", "statement_id", "quote_id"]:
                value = str(statement.get(key) or "")
                if value:
                    support_ids.append(value)
                    break
    for result in observation.get("results") or []:
        if isinstance(result, dict):
            for key in ["support_id", "statement_id"]:
                value = str(result.get(key) or "")
                if value:
                    support_ids.append(value)
                    break
            else:
                if str(result.get("kind") or "") != "text_chunk":
                    value = str(result.get("quote_id") or "")
                    if value:
                        support_ids.append(value)
    return list(dict.fromkeys(support_ids))
