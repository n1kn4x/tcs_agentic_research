"""Independent replication agent."""

from __future__ import annotations

from ..artifact_store import ArtifactStore
from ..llm import LLMRouter
from ..prompt_loader import render_prompt
from ..schemas import ClaimStatus, ReplicationResult, ResearchReport, ResearchState


class IndependentReplicationAgent:
    def __init__(self, store: ArtifactStore, router: LLMRouter, *, prompt_dir: str | None = None):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir

    def verify(self, state: ResearchState, report: ResearchReport) -> ReplicationResult:
        minimized_context = {
            "task_summary": state.task_summary,
            "final_claims": [c.model_dump(mode="json") for c in report.claims_generated],
            "proof_obligations": [o.model_dump(mode="json") for o in report.proof_obligations],
            "artifact_refs": [a.model_dump(mode="json") for a in report.artifact_refs],
        }
        fallback = self._fallback_result(report)
        messages = [
            {"role": "system", "content": render_prompt("independent_replication", override_dir=self.prompt_dir)},
            {
                "role": "user",
                "content": (
                    "Independently verify the claimed breakthrough from this minimized context. "
                    "Do not rely on persuasive history; reconstruct or refute.\n"
                    f"{minimized_context}"
                ),
            },
        ]
        result = self.router.complete_structured(
            task_type="independent_replication",
            messages=messages,
            schema=ReplicationResult,
            fallback=fallback,
        )
        ref = self.store.write_json(f"Reports/critic_summaries/{result.result_id}.json", result)
        result.artifact_refs.append(ref)
        self.store.write_json(f"Reports/critic_summaries/{result.result_id}.json", result)
        return result

    def _fallback_result(self, report: ResearchReport) -> ReplicationResult:
        accepted_statuses = {
            ClaimStatus.proved_by_lean,
            ClaimStatus.cited,
            ClaimStatus.resource_checked,
        }
        verified = [c.claim_id for c in report.claims_generated if c.status in accepted_statuses]
        failed = [c.claim_id for c in report.claims_generated if c.status not in accepted_statuses]
        if failed or report.proof_obligations:
            verdict = "needs_human_review"
            summary = "Fallback replication refuses to verify claims with open obligations or non-certifying statuses."
        else:
            verdict = "partially_verified" if verified else "needs_human_review"
            summary = "Fallback replication found only ledger-certified claims; human/LLM independent reconstruction still recommended."
        return ReplicationResult(
            verdict=verdict,
            summary=summary,
            independently_reconstructed_claim_ids=verified,
            failed_claim_ids=failed,
            blocking_issues=[o.statement for o in report.proof_obligations],
        )
