"""LangGraph orchestration for the long-running research loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .agents.critics import SolvedCheckAgent
from .agents.initialization import InitializationAgent
from .agents.proposal import ProposalAgent
from .agents.replication import IndependentReplicationAgent
from .agents.research import ResearchAgent
from .artifact_store import ArtifactStore
from .llm import LLMRouter
from .render import render_verdict_markdown
from .schemas import (
    ClaimStatus,
    GraphState,
    ReplicationResult,
    ResearchProposal,
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
    ):
        self.store = ArtifactStore(workspace)
        self.store.initialize_layout()
        self.router = LLMRouter.from_config_file(config_path, store=self.store, dry_run=dry_run)
        self.prompt_dir = prompt_dir

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
        builder.add_node("check_is_solved", self._node_check_solved)
        builder.add_node("independent_replication", self._node_independent_replication)

        builder.add_edge(START, "initialize_task")
        builder.add_conditional_edges(
            "initialize_task",
            self._route_after_initialize,
            {"continue": "generate_research_proposal", "end": END},
        )
        builder.add_edge("generate_research_proposal", "run_tcs_research_subagent")
        builder.add_edge("run_tcs_research_subagent", "update_research_state")
        builder.add_edge("update_research_state", "check_is_solved")
        builder.add_conditional_edges(
            "check_is_solved",
            self._route_after_solved_check,
            {
                "replicate": "independent_replication",
                "continue": "generate_research_proposal",
                "end": END,
            },
        )
        builder.add_edge("independent_replication", "check_is_solved")
        return builder.compile(checkpointer=self._make_checkpointer())

    def run(
        self,
        *,
        user_seed: str = "",
        max_iterations: int = 1,
        thread_id: str = "default",
    ) -> dict[str, Any]:
        graph = self.build()
        initial_state: GraphState = {
            "workspace": str(self.store.root),
            "user_seed": user_seed,
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
        existing = self.store.load_state()
        if existing is not None and self.store.exists(ArtifactStore.RESEARCH_TASK):
            return {
                "initialized": True,
                "task_id": existing.task_id,
                "iteration": existing.iteration,
                "solved": existing.solved,
                "confirmed_solved": existing.confirmed_by_replication,
            }
        agent = InitializationAgent(self.store, self.router, prompt_dir=self.prompt_dir)
        state = agent.initialize(user_seed=graph_state.get("user_seed", ""))
        return {
            "initialized": True,
            "task_id": state.task_id,
            "iteration": state.iteration,
            "solved": state.solved,
            "confirmed_solved": state.confirmed_by_replication,
        }

    def _node_generate_proposal(self, graph_state: GraphState) -> dict[str, Any]:
        state = self._require_state()
        proposal, _critique, proposal_path = ProposalAgent(
            self.store, self.router, prompt_dir=self.prompt_dir
        ).generate_and_review(state)
        return {
            "iteration": state.iteration,
            "current_proposal_id": proposal.proposal_id,
            "current_proposal_path": proposal_path,
        }

    def _node_run_research(self, graph_state: GraphState) -> dict[str, Any]:
        state = self._require_state()
        proposal_path = graph_state.get("current_proposal_path")
        if not proposal_path:
            raise RuntimeError("No current proposal path in graph state")
        proposal = ResearchProposal.model_validate(self.store.read_json(proposal_path))
        report, report_path = ResearchAgent(self.store, self.router, prompt_dir=self.prompt_dir).run(
            proposal, state
        )
        return {"current_report_path": report_path, "current_proposal_id": report.proposal_id}

    def _node_update_state(self, graph_state: GraphState) -> dict[str, Any]:
        state = self._require_state()
        report_path = graph_state.get("current_report_path")
        if not report_path:
            raise RuntimeError("No current report path in graph state")
        report = ResearchReport.model_validate(self.store.read_json(report_path))
        self._apply_report_to_state(state, report, report_path)
        return {
            "iteration": state.iteration,
            "current_report_path": report_path,
            "solved": state.solved,
        }

    def _node_check_solved(self, graph_state: GraphState) -> dict[str, Any]:
        state = self._require_state()
        report_path = graph_state.get("current_report_path")
        report = ResearchReport.model_validate(self.store.read_json(report_path)) if report_path else None
        verdict = SolvedCheckAgent(self.store, self.router, prompt_dir=self.prompt_dir).check(
            state, report
        )
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
        iteration = int(graph_state.get("iteration") or 0)
        max_iterations = int(graph_state.get("max_iterations") or 1)
        if max_iterations <= 0 or iteration >= max_iterations:
            return "end"
        return "continue"

    def _route_after_solved_check(self, graph_state: GraphState) -> str:
        if graph_state.get("confirmed_solved"):
            return "end"
        if graph_state.get("possible_breakthrough"):
            return "replicate"
        iteration = int(graph_state.get("iteration") or 0)
        max_iterations = int(graph_state.get("max_iterations") or 1)
        if iteration >= max_iterations:
            return "end"
        return "continue"

    def _require_state(self) -> ResearchState:
        state = self.store.load_state()
        if state is None:
            raise RuntimeError("ResearchState.json is missing; run initialization first")
        return state

    def _apply_report_to_state(self, state: ResearchState, report: ResearchReport, report_path: str) -> None:
        report_ref = self.store.artifact_ref(report_path)
        state.last_report_ref = report_ref
        state.artifact_refs.append(report_ref)
        for claim in report.claims_generated:
            if claim.claim_id not in state.active_claim_ids:
                state.active_claim_ids.append(claim.claim_id)
            if claim.status in {
                ClaimStatus.proved_by_lean,
                ClaimStatus.cited,
                ClaimStatus.resource_checked,
                ClaimStatus.experimentally_supported,
                ClaimStatus.proved_informally,
            } and claim.claim_id not in state.accepted_claim_ids:
                state.accepted_claim_ids.append(claim.claim_id)
            if claim.status in {ClaimStatus.refuted, ClaimStatus.withdrawn, ClaimStatus.duplicate}:
                if claim.claim_id not in state.rejected_claim_ids:
                    state.rejected_claim_ids.append(claim.claim_id)
        state.open_proof_obligations = [
            obligation.statement
            for obligation in report.proof_obligations
            if obligation.status in {"open", "in_progress"}
        ]
        self.store.save_state(state)

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
