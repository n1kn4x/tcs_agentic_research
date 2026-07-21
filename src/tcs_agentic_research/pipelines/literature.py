"""Primary-source literature evidence pipeline."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..agents.literature import LiteratureResearcher
from ..artifact_store import ArtifactStore
from ..llm import LLMRouter
from ..schemas import (
    ArtifactRef,
    CriterionResult,
    EvidenceStrength,
    Finding,
    FindingPolarity,
    FindingStatus,
    LiteratureEvidenceReview,
    LiteraturePlan,
    WorkItem,
    WorkKind,
    WorkResult,
)
from ..workflow import (
    _candidate_is_relevant_and_extractable,
    _compact_query,
    _existing_refs,
    _rank_candidates,
    _render_literature_report,
)


class LiteraturePipeline:
    def __init__(self, store: ArtifactStore, router: LLMRouter, *, prompt_dir: str | None):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir

    def run(
        self,
        item: WorkItem,
        run_dir: str,
        *,
        research_context: dict[str, Any] | None = None,
    ) -> WorkResult:
        fallback_query = _compact_query(item.instruction)
        mock = LiteraturePlan(search_queries=[fallback_query], focus_questions=[fallback_query])
        messages = [
            {
                "role": "system",
                "content": (
                    "Create a targeted primary-literature plan for one atomic evidence requirement. "
                    "Use specific theorem, definition, result, author, and title queries. Do not invent "
                    "identifiers and include searches capable of finding contradictory evidence."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "work_item": item.model_dump(mode="json"),
                        "accepted_prior_evidence": (research_context or {}).get(
                            "accepted_prior_evidence", []
                        ),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        refs: list[ArtifactRef] = [
            self.store.write_json(f"{run_dir}/literature_input.json", {"messages": messages})
        ]
        errors: list[str] = []
        try:
            plan = self.router.complete_structured(
                task_type="literature_planning",
                messages=messages,
                schema=LiteraturePlan,
                mock_output=mock if self.router.dry_run else None,
            )
        except Exception as exc:
            errors.append(f"planning: {type(exc).__name__}: {exc}")
            plan = mock
        refs.append(self.store.write_json(f"{run_dir}/literature_plan.json", plan))
        researcher = LiteratureResearcher(
            self.store, self.router, prompt_dir=self.prompt_dir
        )
        candidates: list[Any] = []
        if not self.router.dry_run:
            for query in dict.fromkeys([*plan.known_source_titles, *plan.search_queries]):
                try:
                    candidates.extend(
                        researcher.search_papers(
                            query, limit=self.router.core.literature_results_per_query
                        )
                    )
                except Exception as exc:
                    errors.append(f"search {query!r}: {type(exc).__name__}: {exc}")
        ranked = [
            candidate
            for candidate in _rank_candidates(
                candidates,
                preferred_titles=plan.known_source_titles,
                relevance_queries=plan.search_queries,
            )
            if _candidate_is_relevant_and_extractable(
                candidate,
                preferred_titles=plan.known_source_titles,
                relevance_queries=plan.search_queries,
            )
        ]
        imported: list[Any] = []
        for candidate in ranked[: self.router.core.literature_import_attempts]:
            if len(imported) >= self.router.core.literature_max_imports:
                break
            try:
                imported.append(
                    researcher.import_candidate(candidate.candidate_id, extract_text=True)
                )
            except Exception as exc:
                errors.append(f"import {candidate.title!r}: {type(exc).__name__}: {exc}")
        try:
            extraction = researcher.extract_imported_papers(
                max_papers=max(1, min(2, self.router.core.literature_max_imports)),
                only_missing=True,
                use_llm=not self.router.dry_run,
            )
        except Exception as exc:
            extraction = {"processed_count": 0, "errors": [str(exc)]}
            errors.append(f"extraction: {type(exc).__name__}: {exc}")
        refs.append(self.store.write_json(f"{run_dir}/extraction.json", extraction))
        answers = []
        for question in plan.focus_questions:
            try:
                answers.append(
                    researcher.answer_query(
                        question, limit=self.router.core.literature_results_per_query
                    )
                )
            except Exception as exc:
                errors.append(f"query {question!r}: {type(exc).__name__}: {exc}")
        refs.append(
            self.store.write_json(
                f"{run_dir}/query_answers.json",
                [answer.model_dump(mode="json") for answer in answers],
            )
        )
        supports: dict[str, Any] = {}
        for answer in answers:
            for row in answer.results:
                if not (row.provenance and row.provenance[0].validated and row.statement_text):
                    continue
                quote = row.provenance[0]
                support_id = row.support_id or _support_id(
                    row.citation_key, quote.char_start, quote.char_end, quote.quote
                )
                supports[support_id] = row
        accepted: dict[str, str] = {}
        if supports and not self.router.dry_run:
            review = self.router.complete_structured(
                task_type="literature_review",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Review each exact primary-source statement against the atomic evidence "
                            "requirement. Topical overlap is unrelated. Do not extend beyond the quote."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "requirement": item.instruction,
                                "hypothesis": item.hypothesis,
                                "statements": [
                                    {
                                        "support_id": support_id,
                                        "citation_key": row.citation_key,
                                        "statement": row.statement_text,
                                    }
                                    for support_id, row in list(supports.items())[:20]
                                ],
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                schema=LiteratureEvidenceReview,
                allow_repair=False,
            )
            refs.append(self.store.write_json(f"{run_dir}/evidence_review.json", review))
            accepted = {
                row.support_id: row.relation
                for row in review.selections
                if row.relevant
                and row.relation != "unrelated"
                and row.support_id in supports
            }
        findings: list[Finding] = []
        for support_id, relation in accepted.items():
            row = supports[support_id]
            quote = row.provenance[0]
            findings.append(
                Finding(
                    work_id=item.work_id,
                    question_id=item.question_id,
                    requirement_id=item.requirement_id,
                    kind=WorkKind.literature,
                    statement=f"[{row.citation_key}] {row.statement_text}",
                    status=FindingStatus.supported,
                    polarity=(
                        FindingPolarity.contradicts
                        if relation == "contradicts"
                        else FindingPolarity.supports
                        if relation == "supports"
                        else FindingPolarity.characterizes
                    ),
                    strength=(
                        EvidenceStrength.strong
                        if row.support_id and row.support_level == "primary_exact"
                        else EvidenceStrength.substantive
                    ),
                    scope="Exactly the assumptions and scope in the quoted primary source.",
                    evidence_refs=quote.artifact_refs,
                    source_ids=[
                        value
                        for value in [support_id, row.statement_id, row.quote_id]
                        if value
                    ],
                )
            )
        refs.append(
            self.store.write_text(
                f"{run_dir}/literature_report.md",
                _render_literature_report(
                    item, plan, candidates, imported, findings, errors
                ),
            )
        )
        criteria = [
            CriterionResult(
                criterion=criterion,
                satisfied=bool(findings),
                detail=(
                    f"{len(findings)} exact statement(s) passed relevance review."
                    if findings
                    else "No exact statement passed requirement-level relevance review."
                ),
            )
            for criterion in item.success_criteria
        ]
        return WorkResult(
            work_id=item.work_id,
            outcome="done" if findings else "partial" if candidates else "blocked",
            failure_class="none" if findings else "evidence_gap",
            evidence_level="substantive" if findings else "none",
            requirement_satisfied=bool(findings),
            criteria=criteria,
            summary=(
                f"Accepted {len(findings)} quote-validated source result(s)."
                if findings
                else "The bounded search found no requirement-relevant exact source result."
            ),
            findings=findings,
            artifact_refs=_existing_refs(
                self.store,
                refs,
                [
                    "LiteratureDB/candidates.jsonl",
                    "LiteratureDB/papers.jsonl",
                    "LiteratureDB/statements.jsonl",
                    "LiteratureDB/index.sqlite",
                ],
            ),
            errors=errors,
            next_steps=(
                ["Search a distinct title, preprint, author, or citation trail."]
                if not findings
                else []
            ),
        )


def _support_id(
    citation_key: str, char_start: int | None, char_end: int | None, quote: str
) -> str:
    identity = f"{citation_key}\0{char_start}\0{char_end}\0{quote}"
    return "passage_support_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
