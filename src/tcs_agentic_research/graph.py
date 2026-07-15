"""LangGraph orchestration for the long-running research loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .agents.critics import (
    check_solved_deterministically,
    is_claim_acceptably_supported,
    is_claim_rejected,
)
from .agents.initialization import WorkspaceInitializer
from .agents.proposal import ProposalAgent
from .agents.replication import IndependentReplicationAgent
from .agents.research import ResearchAgent
from .artifact_store import ArtifactStore
from .llm import LLMRouter
from .obligations import CommitManager, ObligationBoardManager, ObligationRunValidator
from .render import render_report_markdown, render_verdict_markdown
from .schemas import (
    GraphState,
    ObligationRun,
    ReplicationResult,
    ReportOutcome,
    ResearchReport,
    ResearchState,
)


class ResearchGraph:
    """Build and run the resumable top-level research loop.

    The graph state contains only compact references. Canonical state lives in the workspace
    artifacts managed by :class:`ArtifactStore`.
    """

    def __init__(
        self,
        *,
        workspace: str | Path,
        config_path: str | Path | None = None,
        dry_run: bool = False,
        prompt_dir: str | None = None,
        max_proposal_revisions: int = 2,
    ):
        self.store = ArtifactStore(workspace)
        self.store.initialize_layout()
        self.router = LLMRouter.from_config_file(config_path, store=self.store, dry_run=dry_run)
        self.prompt_dir = prompt_dir
        self.max_proposal_revisions = max_proposal_revisions

    def build(self):  # LangGraph is an optional runtime dependency until graph execution.
        try:
            from langgraph.graph import END, START, StateGraph
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "LangGraph is required to run the research loop. Install with `pip install -e .`."
            ) from exc

        builder = StateGraph(GraphState)
        builder.add_node("initialize_task", self._node_initialize_task)
        builder.add_node("generate_research_proposal", self._node_generate_proposal)
        builder.add_node("run_tcs_research_subagent", self._node_run_research)
        builder.add_node("update_research_state", self._node_update_state)
        builder.add_node("compute_solved_verdict", self._node_compute_solved_verdict)
        builder.add_node("independent_replication", self._node_independent_replication)

        builder.add_edge(START, "initialize_task")
        builder.add_conditional_edges(
            "initialize_task",
            self._route_after_initialize,
            {"continue": "generate_research_proposal", "end": END},
        )
        builder.add_edge("generate_research_proposal", "run_tcs_research_subagent")
        builder.add_edge("run_tcs_research_subagent", "update_research_state")
        builder.add_edge("update_research_state", "compute_solved_verdict")
        builder.add_conditional_edges(
            "compute_solved_verdict",
            self._route_after_solved_verdict,
            {
                "replicate": "independent_replication",
                "continue": "generate_research_proposal",
                "end": END,
            },
        )
        builder.add_edge("independent_replication", "compute_solved_verdict")
        return builder.compile(checkpointer=self._make_checkpointer())

    def run(
        self,
        *,
        max_iterations: int = 1,
        thread_id: str = "default",
    ) -> dict[str, Any]:
        graph = self.build()
        initial_state: GraphState = {
            "workspace": str(self.store.root),
            "max_iterations": max_iterations,
        }
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": max(20, max_iterations * 8 + 20),
        }
        return graph.invoke(initial_state, config=config)

    def _make_checkpointer(self):
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Install `langgraph-checkpoint-sqlite` to use durable graph checkpoints."
            ) from exc
        candidate = SqliteSaver.from_conn_string(str(self.store.root / "GraphCheckpoints.sqlite"))
        # Some LangGraph versions return a context manager; others return the saver directly.
        if hasattr(candidate, "__enter__") and not hasattr(candidate, "get_tuple"):
            self._checkpointer_context = candidate
            return candidate.__enter__()
        return candidate

    def _node_initialize_task(self, graph_state: GraphState) -> dict[str, Any]:
        state = WorkspaceInitializer(self.store).ensure_initialized()
        self._refresh_state_from_claim_ledger(
            state, open_obligations=state.open_proof_obligations
        )
        self.store.save_state(state)
        return {
            "initialized": True,
            "task_id": state.task_id,
            "iteration": state.iteration,
            "solved": state.solved,
            "confirmed_solved": state.confirmed_by_replication,
        }

    def _node_generate_proposal(self, graph_state: GraphState) -> dict[str, Any]:
        state = self._require_state()
        board_manager = ObligationBoardManager(self.store)
        board = board_manager.load()
        obligation = board_manager.next_open_obligation(board)
        proposal_path: str | None = None

        if obligation is None:
            proposal, _critique, proposal_path = ProposalAgent(
                self.store, self.router, prompt_dir=self.prompt_dir
            ).generate_and_review(
                state,
                max_revisions=self.max_proposal_revisions,
            )
            board_manager.ensure_obligations_from_proposal(proposal)
            board = board_manager.load()
            obligation = board_manager.next_open_obligation(board)
            if obligation is None:
                raise RuntimeError("Proposal did not produce any open obligations")
            state = self._require_state()
        else:
            # Count each obligation run as a graph iteration, even when it continues an
            # existing proposal plan.
            state.iteration += 1
            if obligation.proposal_id:
                state.current_proposal_id = obligation.proposal_id
            self.store.save_state(state)

        return {
            "iteration": state.iteration,
            "current_proposal_id": state.current_proposal_id,
            "current_proposal_path": proposal_path,
            "current_obligation_id": obligation.obligation_id,
        }

    def _node_run_research(self, graph_state: GraphState) -> dict[str, Any]:
        state = self._require_state()
        obligation_id = graph_state.get("current_obligation_id")
        if not obligation_id:
            raise RuntimeError("No current obligation id in graph state")
        board_manager = ObligationBoardManager(self.store)
        board = board_manager.load()
        obligation = board_manager.get_obligation(board, obligation_id)
        if obligation is None:
            raise RuntimeError(f"No obligation `{obligation_id}` in ObligationBoard")
        research_agent = ResearchAgent(self.store, self.router, prompt_dir=self.prompt_dir)
        run, run_path, trace_path = research_agent.run_obligation(
            obligation=obligation,
            state=state,
        )
        return {
            "current_obligation_run_path": run_path,
            "current_obligation_trace_path": trace_path,
            "current_obligation_id": run.obligation_id,
        }

    def _node_update_state(self, graph_state: GraphState) -> dict[str, Any]:
        state = self._require_state()
        run_path = graph_state.get("current_obligation_run_path")
        trace_path = graph_state.get("current_obligation_trace_path")
        if not run_path:
            raise RuntimeError("No current obligation run path in graph state")
        run = ObligationRun.model_validate(self.store.read_json(run_path))
        board_manager = ObligationBoardManager(self.store)
        board = board_manager.load()
        obligation = board_manager.get_obligation(board, run.obligation_id)
        if obligation is None:
            raise RuntimeError(f"No obligation `{run.obligation_id}` in ObligationBoard")
        trace_payload = self.store.read_json(trace_path) if trace_path else {}
        trace = trace_payload.get("trace", trace_payload) if isinstance(trace_payload, dict) else {}
        validation = ObligationRunValidator(self.store).validate(
            run=run,
            obligation=obligation,
            trace=trace,
        )
        run.validation = validation
        self.store.write_json(run_path, run)
        commit_result = CommitManager(self.store).apply_obligation_run(
            run=run,
            validation=validation,
            run_ref=self.store.artifact_ref(run_path),
        )
        state = self._require_state()
        report_path = self._write_obligation_summary_report(state, run, validation, commit_result)
        return {
            "iteration": state.iteration,
            "current_report_path": report_path,
            "current_obligation_run_path": run_path,
            "solved": state.solved,
        }

    def _node_compute_solved_verdict(self, graph_state: GraphState) -> dict[str, Any]:
        state = self._require_state()
        report_path = graph_state.get("current_report_path")
        report = ResearchReport.model_validate(self.store.read_json(report_path)) if report_path else None
        verdict = check_solved_deterministically(self.store, state, report)
        rel_dir = self.store.create_iteration_dir(state.iteration)
        verdict_ref = self.store.write_json(f"{rel_dir}/solved_verdict_{verdict.verdict_id}.json", verdict)
        self.store.write_text(
            f"{rel_dir}/solved_verdict_{verdict.verdict_id}.md", render_verdict_markdown(verdict)
        )
        state.last_verdict_ref = verdict_ref
        state.solved = verdict.confirmed_solved
        state.outcome_flags = list(dict.fromkeys(state.outcome_flags + [o.value for o in verdict.outcomes]))
        self.store.save_state(state)
        return {
            "possible_breakthrough": verdict.possible_breakthrough,
            "confirmed_solved": verdict.confirmed_solved,
            "solved": verdict.confirmed_solved,
            "last_verdict_path": verdict_ref.path,
            "stop_reason": verdict.next_action if verdict.confirmed_solved else None,
        }

    def _node_independent_replication(self, graph_state: GraphState) -> dict[str, Any]:
        state = self._require_state()
        report_path = graph_state.get("current_report_path")
        if not report_path:
            raise RuntimeError("Independent replication requires a report")
        report = ResearchReport.model_validate(self.store.read_json(report_path))
        result = IndependentReplicationAgent(self.store, self.router, prompt_dir=self.prompt_dir).verify(
            state, report
        )
        self._apply_replication_to_state(state, result)
        return {"confirmed_solved": state.confirmed_by_replication, "possible_breakthrough": False}

    def _route_after_initialize(self, graph_state: GraphState) -> str:
        if graph_state.get("confirmed_solved") or graph_state.get("solved"):
            return "end"
        return "end" if self._iteration_limit_reached(graph_state) else "continue"

    def _route_after_solved_verdict(self, graph_state: GraphState) -> str:
        if graph_state.get("confirmed_solved"):
            return "end"
        if graph_state.get("possible_breakthrough"):
            return "replicate"
        return "end" if self._iteration_limit_reached(graph_state) else "continue"

    def _iteration_limit_reached(self, graph_state: GraphState) -> bool:
        iteration = int(graph_state.get("iteration") or 0)
        max_iterations = int(graph_state.get("max_iterations") or 1)
        return max_iterations <= 0 or iteration >= max_iterations

    def _require_state(self) -> ResearchState:
        state = self.store.load_state()
        if state is None:
            raise RuntimeError(
                "ResearchState.json is missing; create "
                f"`{ArtifactStore.RESEARCH_TASK}` and run again"
            )
        return state

    def _write_obligation_summary_report(
        self,
        state: ResearchState,
        run: ObligationRun,
        validation: Any,
        commit_result: dict[str, Any],
    ) -> str:
        rel_dir = self.store.create_iteration_dir(state.iteration)
        outcome = ReportOutcome.partially_succeeded if validation.ok else ReportOutcome.needs_more_work
        summary_lines = [
            f"Obligation run `{run.run_id}` processed for obligation `{run.obligation_id}`.",
            f"Commit outcome: {commit_result.get('outcome')}",
        ]
        if validation.blocking_issues:
            summary_lines.append("Blocking issues: " + "; ".join(validation.blocking_issues))
        latest_claims = self.store.latest_claims_by_id()
        committed_claims = [
            latest_claims[claim_id]
            for claim_id in commit_result.get("accepted_claim_ids", [])
            if claim_id in latest_claims
        ]
        report = ResearchReport(
            proposal_id=state.current_proposal_id or "",
            outcome=outcome,
            executive_summary="\n".join(summary_lines),
            claims_generated=committed_claims or list(run.claims_generated),
            evidence=list(run.evidence),
            unresolved_issues=list(validation.blocking_issues),
            artifact_refs=[self.store.artifact_ref(ArtifactStore.OBLIGATION_BOARD)],
        )
        report_ref = self.store.write_json(
            f"{rel_dir}/obligation_summary_report_{report.report_id}.json", report
        )
        self.store.write_text(
            f"{rel_dir}/obligation_summary_report_{report.report_id}.md",
            render_report_markdown(report),
        )
        state.last_report_ref = report_ref
        if report_ref.path not in {ref.path for ref in state.artifact_refs}:
            state.artifact_refs.append(report_ref)
        self.store.save_state(state)
        return report_ref.path

    def _apply_report_to_state(self, state: ResearchState, report: ResearchReport, report_path: str) -> None:
        report_ref = self.store.artifact_ref(report_path)
        state.last_report_ref = report_ref
        if report_ref.path not in {ref.path for ref in state.artifact_refs}:
            state.artifact_refs.append(report_ref)
        report_open_obligations = [
            obligation.statement
            for obligation in report.proof_obligations
            if obligation.status in {"open", "in_progress", "blocked"}
        ]
        proved_or_refuted_obligations = {
            obligation.statement
            for obligation in report.proof_obligations
            if obligation.status in {"proved", "experimentally_supported", "refuted"}
        }
        merged_open_obligations = [
            obligation
            for obligation in [*state.open_proof_obligations, *report_open_obligations]
            if obligation not in proved_or_refuted_obligations
        ]
        self._refresh_state_from_claim_ledger(state, open_obligations=merged_open_obligations)
        self.store.save_state(state)

    def _refresh_state_from_claim_ledger(
        self, state: ResearchState, *, open_obligations: list[str] | None = None
    ) -> None:
        latest_claims = self.store.latest_claims_by_id()
        accepted_claim_ids = [
            claim_id
            for claim_id, claim in latest_claims.items()
            if is_claim_acceptably_supported(claim, self.store)
        ]
        # Keep ResearchState focused on accepted/proven claims. Draft findings and blocked
        # attempts live on ObligationBoard.json, not in active_claim_ids.
        state.active_claim_ids = list(accepted_claim_ids)
        state.accepted_claim_ids = accepted_claim_ids
        state.rejected_claim_ids = [
            claim_id for claim_id, claim in latest_claims.items() if is_claim_rejected(claim)
        ]
        if open_obligations is not None:
            state.open_proof_obligations = list(dict.fromkeys(open_obligations))

    def _apply_replication_to_state(self, state: ResearchState, result: ReplicationResult) -> None:
        state.artifact_refs.extend(result.artifact_refs)
        if result.verdict == "verified":
            state.confirmed_by_replication = True
            state.outcome_flags = [flag for flag in state.outcome_flags if flag != "replication_failed_or_incomplete"]
            state.notes.append("Independent replication verified the claimed breakthrough.")
        else:
            state.confirmed_by_replication = False
            if "replication_failed_or_incomplete" not in state.outcome_flags:
                state.outcome_flags.append("replication_failed_or_incomplete")
            state.notes.append(f"Independent replication did not verify: {result.summary}")
        self.store.save_state(state)
