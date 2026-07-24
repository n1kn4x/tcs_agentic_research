"""Autonomous literature discovery and exact-span retrieval subsystem."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ...agents.literature import LiteratureResearcher
from ...artifact_store import ArtifactStore
from ...llm import LLMRouter
from ...schemas import ArtifactRef, StrictModel
from ..models import (
    ActionOutcome,
    ActionProposal,
    EvidenceReceipt,
    EvidenceType,
    RecordDraft,
    RecordKind,
    RecordRelation,
    ResearchView,
)


class LiteratureChoice(StrictModel):
    action: Literal["search", "query_local", "idle"]
    query: str = Field(default="", max_length=500)
    rationale: str = Field(default="", max_length=1800)
    parent_ids: list[str] = Field(default_factory=list, max_length=12)


class LiteratureSubsystem:
    name = "literature"
    description = "Discovers sources and records exact source spans without interpreting them as truth."
    model_call_budget = 2

    def __init__(
        self,
        store: ArtifactStore,
        router: LLMRouter,
        *,
        prompt_dir: str | None = None,
        max_imports_per_action: int = 2,
    ):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir
        self.max_imports_per_action = max_imports_per_action

    def propose(self, view: ResearchView) -> ActionProposal | None:
        if self.router.dry_run:
            return None
        choice = self.router.complete_structured(
            task_type="literature_decision",
            schema=LiteratureChoice,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the autonomous literature subsystem. Choose one bounded source "
                        "action. Use search to discover/import new papers. Use query_local to retrieve "
                        "an exact span from already imported text. Search for primary sources when "
                        "possible. Do not use a quote to answer a broader question than its words. "
                        "Yield idle instead of repeating a query in subsystem_state or recent_actions."
                    ),
                },
                {"role": "user", "content": view.model_dump_json()},
            ],
        )
        if choice.action == "idle" or not choice.query.strip():
            return None
        return ActionProposal(
            subsystem=self.name,
            action_type=choice.action,
            title=f"{choice.action}: {choice.query}"[:300],
            rationale=choice.rationale,
            payload={"query": choice.query.strip()},
            parent_ids=[
                record_id
                for record_id in choice.parent_ids
                if record_id in {record.record_id for record in view.records}
            ],
        )

    def execute(
        self, proposal: ActionProposal, view: ResearchView, *, run_dir: str
    ) -> ActionOutcome:
        researcher = LiteratureResearcher(
            self.store, self.router, prompt_dir=self.prompt_dir
        )
        query = str(proposal.payload["query"])
        prior_queries = list(view.subsystem_state.get("queries", []))
        records: list[RecordDraft] = []
        errors: list[str] = []
        if proposal.action_type == "search":
            candidates = researcher.search_papers(query, limit=5)
            self.store.write_json(f"{run_dir}/candidates.json", candidates)
            for candidate in candidates[: self.max_imports_per_action]:
                try:
                    paper = researcher.import_candidate(
                        candidate.candidate_id, extract_text=True
                    )
                    refs = self._fresh_paper_refs(paper.metadata_path, paper.artifact_refs)
                    records.append(
                        RecordDraft(
                            kind=RecordKind.source,
                            title=paper.title,
                            summary=(
                                f"Imported source `{paper.citation_key}` ({paper.year or 'year unknown'}) "
                                f"by {', '.join(paper.authors[:5]) or 'unknown authors'}."
                            ),
                            body=paper.abstract,
                            relation=(
                                RecordRelation.documents
                                if proposal.parent_ids
                                else RecordRelation.none
                            ),
                            parent_ids=proposal.parent_ids,
                            evidence=EvidenceReceipt(
                                evidence_type=EvidenceType.source_metadata,
                                details={
                                    "paper_id": paper.paper_id,
                                    "citation_key": paper.citation_key,
                                    "title": paper.title,
                                    "doi": paper.doi,
                                    "arxiv_id": paper.arxiv_id,
                                    "url": paper.url,
                                },
                                artifact_refs=refs,
                            ),
                        )
                    )
                    if paper.text_path:
                        researcher.extract_paper(
                            citation_key=paper.citation_key, use_llm=False
                        )
                except Exception as exc:  # one source must not abort the bounded batch
                    errors.append(f"{candidate.title}: {type(exc).__name__}: {exc}")
        elif proposal.action_type == "query_local":
            answer = researcher.answer_query(query, limit=8)
            self.store.write_json(f"{run_dir}/query_answer.json", answer)
            for result in answer.results:
                quote = next((item for item in result.provenance if item.validated), None)
                if quote is None:
                    continue
                refs = self._fresh_refs(quote.artifact_refs)
                records.append(
                    RecordDraft(
                        kind=RecordKind.source_quote,
                        title=(result.label or result.title or f"Source span from {result.citation_key}")[:300],
                        summary=result.statement_text,
                        body=(
                            f"Exact quote ({quote.locator or 'locator unavailable'}):\n\n"
                            f"{quote.quote}"
                        ),
                        relation=(
                            RecordRelation.documents
                            if proposal.parent_ids
                            else RecordRelation.none
                        ),
                        parent_ids=proposal.parent_ids,
                        evidence=EvidenceReceipt(
                            evidence_type=EvidenceType.source_quote,
                            details={
                                "paper_id": quote.paper_id or result.paper_id,
                                "citation_key": quote.citation_key or result.citation_key,
                                "quote": quote.quote,
                                "locator": quote.locator,
                                "source_sha256": quote.source_sha256,
                                "validated": quote.validated,
                                "support_id": result.support_id,
                            },
                            artifact_refs=refs,
                        ),
                    )
                )
        else:
            return ActionOutcome(
                summary=f"Unknown literature action {proposal.action_type}",
                error="unsupported action",
            )
        prior_queries.append({"action": proposal.action_type, "query": query})
        return ActionOutcome(
            summary=(
                f"Literature {proposal.action_type} produced {len(records)} provenance-backed "
                f"record(s) and {len(errors)} bounded error(s)."
            ),
            records=records,
            state_patch={"queries": prior_queries[-50:], "last_errors": errors[-10:]},
            error="" if records or not errors else "; ".join(errors)[:5000],
            retryable=bool(errors),
        )

    def _fresh_paper_refs(
        self, metadata_path: str, refs: list[ArtifactRef]
    ) -> list[ArtifactRef]:
        paths = [ref.path for ref in refs]
        if metadata_path:
            paths.append(metadata_path)
        return self._fresh_refs([ArtifactRef(path=path) for path in dict.fromkeys(paths)])

    def _fresh_refs(self, refs: list[ArtifactRef]) -> list[ArtifactRef]:
        fresh: list[ArtifactRef] = []
        for ref in refs:
            if ref.path and self.store.exists(ref.path):
                current = self.store.artifact_ref(ref.path)
                if current.path not in {item.path for item in fresh}:
                    fresh.append(current)
        return fresh
