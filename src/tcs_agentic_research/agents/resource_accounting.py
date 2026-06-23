"""General resource-accounting critic for TCS claims."""

from __future__ import annotations

from ..artifact_store import ArtifactStore
from ..llm import LLMRouter
from ..prompt_loader import render_prompt
from ..schemas import ComplexityEstimate, ResourceCheckResult


class ResourceAccountingAgent:
    def __init__(self, store: ArtifactStore, router: LLMRouter, *, prompt_dir: str | None = None):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir

    def check(self, estimates: list[ComplexityEstimate], context: str = "") -> ResourceCheckResult:
        fallback = self._fallback_check(estimates)
        messages = [
            {"role": "system", "content": render_prompt("resource_accountant", override_dir=self.prompt_dir)},
            {
                "role": "user",
                "content": (
                    "Check these resource estimates. Be conservative and identify hidden costs.\n"
                    f"Context:\n{context}\n\nEstimates:\n"
                    + "\n".join(e.model_dump_json() for e in estimates)
                ),
            },
        ]
        result = self.router.complete_structured(
            task_type="research_critique",
            messages=messages,
            schema=ResourceCheckResult,
            fallback=fallback,
        )
        ref = self.store.write_json(f"Reports/critic_summaries/{result.result_id}.json", result)
        result.artifact_refs.append(ref)
        self.store.write_json(f"Reports/critic_summaries/{result.result_id}.json", result)
        return result

    def _fallback_check(self, estimates: list[ComplexityEstimate]) -> ResourceCheckResult:
        issues: list[str] = []
        downgraded: list[str] = []
        accepted: list[str] = []
        for estimate in estimates:
            if estimate.needs_accounting_review:
                issues.append(
                    f"Estimate for {estimate.resource}={estimate.bound} requires derivation review in model {estimate.model}."
                )
                if estimate.claim_id:
                    downgraded.append(estimate.claim_id)
            elif estimate.claim_id:
                accepted.append(estimate.claim_id)
        return ResourceCheckResult(
            checked_claim_ids=[e.claim_id for e in estimates if e.claim_id],
            accepted_claim_ids=accepted,
            downgraded_claim_ids=downgraded,
            issues=issues,
            summary=(
                "Fallback resource checker does not certify asymptotic bounds; it marks estimates "
                "needing explicit derivations unless already reviewed."
            ),
        )
