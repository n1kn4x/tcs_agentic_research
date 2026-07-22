"""Primary-source literature evidence pipeline."""

from __future__ import annotations

import hashlib
import json
import re
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
        fallback_query = _compact_query(
            f"{(research_context or {}).get('research_objective', '')} "
            f"{item.hypothesis} {item.instruction}"
        )
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
                        "research_objective": (research_context or {}).get(
                            "research_objective", ""
                        ),
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
        plan = plan.model_copy(
            update={
                "search_queries": list(
                    dict.fromkeys([fallback_query, *plan.search_queries])
                )[:4]
            }
        )
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
                relevance_queries=[fallback_query],
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
            statement_payload = [
                {
                    "support_id": support_id,
                    "citation_key": row.citation_key,
                    "statement": row.statement_text,
                }
                for support_id, row in list(supports.items())[:20]
            ]
            review_messages = [
                {
                    "role": "system",
                    "content": (
                        "Review each exact primary-source statement against the atomic evidence "
                        "requirement. Return exactly one selection for every supplied support_id; "
                        "the selections list must not be empty when statements are supplied. Topical "
                        "overlap is unrelated. Judge only the quoted statement: do not use an unstated "
                        "equivalence, reduction, paper-level context, or outside knowledge."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "requirement": item.instruction,
                            "hypothesis": item.hypothesis,
                            "statements": statement_payload,
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
            review = self.router.complete_structured(
                task_type="literature_review",
                messages=review_messages,
                schema=LiteratureEvidenceReview,
                allow_repair=False,
            )
            refs.append(self.store.write_json(f"{run_dir}/evidence_review.json", review))
            if not review.selections:
                # Empty reviews have repeatedly discarded exact definitions already present in the
                # supplied passages. One fresh, smaller retry is bounded and safer than either
                # accepting by lexical overlap or silently exhausting a valid source.
                retry_payload = statement_payload[:8]
                review = self.router.complete_structured(
                    task_type="literature_review",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "The previous audit omitted every statement. Assess each supplied "
                                "support_id exactly once as relevant or unrelated. Use only its exact "
                                "quoted text and never infer an unstated relationship."
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "requirement": item.instruction,
                                    "hypothesis": item.hypothesis,
                                    "statements": retry_payload,
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ],
                    schema=LiteratureEvidenceReview,
                    allow_repair=False,
                )
                refs.append(
                    self.store.write_json(f"{run_dir}/evidence_review_retry.json", review)
                )
            accepted = {
                selection.support_id: selection.relation
                for selection in review.selections
                if selection.relevant
                and selection.relation != "unrelated"
                and selection.support_id in supports
                and _preserves_required_acronyms(
                    item, supports[selection.support_id].statement_text
                )
            }
            rejected_for_missing_anchors = [
                selection.support_id
                for selection in review.selections
                if selection.relevant
                and selection.relation != "unrelated"
                and selection.support_id in supports
                and not _preserves_required_acronyms(
                    item, supports[selection.support_id].statement_text
                )
            ]
            if rejected_for_missing_anchors:
                errors.append(
                    "review accepted quote(s) that omit explicit requirement acronym(s): "
                    + ", ".join(rejected_for_missing_anchors)
                )
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


def _preserves_required_acronyms(item: WorkItem, statement: str) -> bool:
    """Prevent a reviewer from supplying an unstated cross-problem relationship.

    Acronyms are useful conservative entity anchors in technical requirements. If a requirement
    explicitly asks about OV under SETH, a quote about SETH hardness for another problem cannot be
    promoted by claiming an outside equivalence. Ordinary title-case words are intentionally not
    treated as anchors.
    """
    requirement_text = f"{item.instruction} {item.hypothesis}"
    anchors = set(
        re.findall(
            r"(?<![A-Za-z0-9])([A-Z][A-Z0-9]{1,9})(?![A-Za-z0-9])",
            requirement_text,
        )
    )
    return all(_statement_contains_anchor(statement, anchor) for anchor in anchors)


def _statement_contains_anchor(statement: str, anchor: str) -> bool:
    if re.search(
        rf"(?<![A-Za-z0-9]){re.escape(anchor)}(?![A-Za-z0-9])",
        statement,
    ):
        return True
    # Exact spans often spell out an acronym rather than repeating it. Accept a contiguous
    # capitalized expansion (for example, "Orthogonal Vectors" for OV) without introducing a
    # domain-specific acronym dictionary.
    words = re.findall(r"[A-Za-z][A-Za-z-]*", statement)
    width = len(anchor)
    return any(
        len(window) == width
        and all(word[0].isupper() for word in window)
        and "".join(word[0] for word in window).upper() == anchor
        for start in range(len(words) - width + 1)
        for window in [words[start : start + width]]
    )


def _support_id(
    citation_key: str, char_start: int | None, char_end: int | None, quote: str
) -> str:
    identity = f"{citation_key}\0{char_start}\0{char_end}\0{quote}"
    return "passage_support_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
