"""Known-barriers and obstruction agent."""

from __future__ import annotations

from ..artifact_store import ArtifactStore
from ..llm import LLMRouter
from ..prompt_loader import render_prompt
from ..schemas import ClaimRecord, ClaimStatus, ClaimType, EvidenceRecord, EvidenceType, ObstructionResult, ResearchProposal


class ObstructionAgent:
    def __init__(self, store: ArtifactStore, router: LLMRouter, *, prompt_dir: str | None = None):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir

    def analyze(self, proposal: ResearchProposal, context: str = "") -> ObstructionResult:
        fallback = ObstructionResult(
            summary="Fallback obstruction scan did not certify absence of barriers.",
            obstruction_claims=[
                ClaimRecord(
                    claim_type=ClaimType.obstruction,
                    statement=(
                        "No lower-bound, no-go, or duplicate-literature obstruction has yet been ruled out "
                        f"for proposal {proposal.proposal_id}."
                    ),
                    status=ClaimStatus.needs_review,
                    related_proposal_ids=[proposal.proposal_id],
                    evidence=[
                        EvidenceRecord(
                            evidence_type=EvidenceType.none,
                            summary="Placeholder requiring literature and barrier review.",
                            confidence=0.0,
                        )
                    ],
                )
            ],
            recommended_changes=["Run targeted literature queries for lower bounds and duplicate algorithms."],
        )
        messages = [
            {"role": "system", "content": render_prompt("obstruction_agent", override_dir=self.prompt_dir)},
            {
                "role": "user",
                "content": (
                    "Search for lower bounds, no-go theorems, reductions, hidden assumptions, and duplicate results.\n"
                    f"Context:\n{context}\n\nProposal:\n{proposal.model_dump_json(indent=2)}"
                ),
            },
        ]
        result = self.router.complete_structured(
            task_type="obstruction_search",
            messages=messages,
            schema=ObstructionResult,
            fallback=fallback,
        )
        ref = self.store.write_json(f"Reports/critic_summaries/{result.result_id}.json", result)
        result.artifact_refs.append(ref)
        if result.obstruction_claims:
            self.store.append_claims(result.obstruction_claims)
        return result
