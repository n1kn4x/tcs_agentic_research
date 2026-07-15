"""Workspace bootstrapping from a user-authored initial research task."""

from __future__ import annotations

from ..artifact_store import ArtifactStore
from ..schemas import ArtifactKind, ProposalLedgerEntry, ResearchState


class WorkspaceInitializer:
    """Create the machine artifacts required by the research loop.

    A workspace is initialized from a single user-authored Markdown file:
    ``InitialResearchTask.md``.  No interview or LLM synthesis is involved; the task file is
    the canonical problem definition used by the rest of the system.
    """

    def __init__(self, store: ArtifactStore):
        self.store = store

    def ensure_initialized(self) -> ResearchState:
        """Return existing state or synthesize missing artifacts from the task file."""
        if not self.store.exists(ArtifactStore.RESEARCH_STATE) and not self.store.exists(
            ArtifactStore.RESEARCH_TASK
        ):
            raise RuntimeError(
                f"Missing `{ArtifactStore.RESEARCH_TASK}` in {self.store.root}. "
                "Create a workspace folder containing that Markdown file, then run again."
            )
        self.store.initialize_layout()
        state = self.store.load_state()
        if state is not None:
            task_markdown = self._read_task_markdown()
            state.task_summary = _task_summary(task_markdown)
            self.store.save_state(state)
            return state
        return self.initialize_from_task_file()

    def initialize_from_task_file(self) -> ResearchState:
        """Create ResearchState and ledger references from ``InitialResearchTask.md``."""
        self.store.initialize_layout()
        task_markdown = self._read_task_markdown()

        task_ref = self.store.artifact_ref(
            ArtifactStore.RESEARCH_TASK,
            summary="User-authored canonical research task.",
        )
        nomenclature_ref = self.store.artifact_ref(
            ArtifactStore.NOMENCLATURE,
            summary="Canonical notation table, initially empty unless edited by the user.",
        )
        board_ref = self.store.artifact_ref(
            ArtifactStore.OBLIGATION_BOARD,
            summary="Initial empty obligation board.",
        )
        claim_ledger_ref = self.store.artifact_ref(
            ArtifactStore.CLAIM_LEDGER,
            kind=ArtifactKind.jsonl,
            summary="Initial empty claim ledger.",
        )
        proposal_ledger_ref = self.store.artifact_ref(
            ArtifactStore.PROPOSAL_LEDGER,
            kind=ArtifactKind.jsonl,
            summary="Proposal ledger created during workspace bootstrap.",
        )
        model_ledger_ref = self.store.artifact_ref(
            ArtifactStore.MODEL_LEDGER,
            kind=ArtifactKind.jsonl,
            summary="Model-call ledger created during workspace bootstrap.",
        )

        state = ResearchState(
            task_summary=_task_summary(task_markdown),
            artifact_refs=[
                task_ref,
                nomenclature_ref,
                board_ref,
                claim_ledger_ref,
                proposal_ledger_ref,
                model_ledger_ref,
            ],
            notes=[
                f"Initialized deterministically from `{ArtifactStore.RESEARCH_TASK}`; "
                "scientific claims will be added only after obligation validation."
            ],
        )
        state_ref = self.store.save_state(state)
        self.store.append_proposal_event(
            ProposalLedgerEntry(
                proposal_id="workspace_initialization",
                event_type="accepted",
                reason=(
                    f"Workspace bootstrapped from user-authored `{ArtifactStore.RESEARCH_TASK}`."
                ),
                artifact_refs=[task_ref, nomenclature_ref, board_ref, state_ref],
            )
        )
        return state

    def _read_task_markdown(self) -> str:
        if not self.store.exists(ArtifactStore.RESEARCH_TASK):
            raise RuntimeError(
                f"Missing `{ArtifactStore.RESEARCH_TASK}` in {self.store.root}. "
                "Create a workspace folder containing that Markdown file, then run again."
            )
        task_markdown = self.store.read_text(ArtifactStore.RESEARCH_TASK)
        if not task_markdown.strip():
            raise RuntimeError(f"`{ArtifactStore.RESEARCH_TASK}` is empty")
        return task_markdown


def _task_summary(markdown: str, limit: int = 500) -> str:
    lines = [line.strip("# ").strip() for line in markdown.splitlines() if line.strip()]
    return " ".join(lines)[:limit]
