"""Bounded state-reader views over the persistent proof graph."""

from __future__ import annotations

from typing import Sequence

from .graph import ProofGraph
from .models import LeanDiagnostic, ProofContext
from .retrieval import LeanRetriever


class StateReader:
    def __init__(self, graph: ProofGraph, retriever: LeanRetriever):
        self.graph = graph
        self.retriever = retriever

    def build(
        self,
        node_id: str,
        *,
        user_context: str,
        informal_queries: Sequence[str] = (),
        diagnostics: Sequence[LeanDiagnostic] = (),
        remaining_nodes: int,
        remaining_seconds: int,
    ) -> ProofContext:
        node = self.graph.get_or(node_id)
        ancestors = self.graph.ancestors(node_id, limit=12)
        retrieval = self.retriever.search(
            node.goal,
            informal_queries=informal_queries,
            diagnostics=diagnostics,
            exclude_or_ids=[node_id, *[ancestor.node_id for ancestor in ancestors]],
            limit=18,
        )
        proved = [item for item in retrieval if item.proved_or_id][:10]
        library = [item for item in retrieval if not item.proved_or_id][:10]
        failures: list[str] = []
        for attempt in self.graph.recent_attempts(node_id, limit=8):
            diagnostic = " | ".join(item.message[:800] for item in attempt.diagnostics[:3])
            summary = f"{attempt.mode} #{attempt.ordinal}: {attempt.outcome}"
            if diagnostic:
                summary += f" — {diagnostic}"
            elif attempt.note:
                summary += f" — {attempt.note[:1000]}"
            failures.append(summary)
        decompositions = [
            (
                f"{item.node_id} [{item.status.value}]: "
                + ", ".join(
                    f"{child.local_name}{'' if child.required else ' (anticipatory)'}"
                    for child in item.children
                )
            )
            for item in self.graph.decompositions(node_id)
        ]
        return ProofContext(
            node_id=node.node_id,
            goal=node.goal,
            environment_fingerprint=node.environment_fingerprint,
            user_context=user_context[-6000:],
            ancestors=[ancestor.goal for ancestor in ancestors],
            proved_lemmas=proved,
            library_results=library,
            previous_failures=failures,
            existing_decompositions=decompositions,
            remaining_nodes=max(0, remaining_nodes),
            remaining_seconds=max(0, remaining_seconds),
        )


def compact_context(context: ProofContext, *, max_chars: int = 18_000) -> str:
    """Render a deterministic prompt package without dumping the complete DAG."""
    text = context.model_dump_json(indent=2)
    if len(text) <= max_chars:
        return text
    # Preserve the exact goal and the recent failure tail.  Library results are the first data to
    # compact because Lean will reject hallucinated names regardless.
    compact = context.model_copy(
        update={
            "user_context": context.user_context[-3000:],
            "library_results": context.library_results[:4],
            "proved_lemmas": context.proved_lemmas[:6],
            "ancestors": context.ancestors[:6],
            "previous_failures": context.previous_failures[:5],
            "existing_decompositions": context.existing_decompositions[:5],
        }
    ).model_dump_json(indent=2)
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 80] + "\n...[state reader context truncated deterministically]"
