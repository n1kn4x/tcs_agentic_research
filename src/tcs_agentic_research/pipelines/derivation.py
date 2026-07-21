"""Reviewed mathematical derivation pipeline."""

from __future__ import annotations

import json
import re
from typing import Any

from ..artifact_store import ArtifactStore
from ..llm import LLMRouter
from ..schemas import (
    DerivationReview,
    DerivationSubmission,
    EvidenceStrength,
    Finding,
    FindingPolarity,
    FindingStatus,
    WorkItem,
    WorkKind,
    WorkResult,
)


class DerivationPipeline:
    def __init__(self, store: ArtifactStore, router: LLMRouter):
        self.store = store
        self.router = router

    def run(
        self, item: WorkItem, run_dir: str, *, prior_context: dict[str, Any]
    ) -> WorkResult:
        if self.router.dry_run:
            artifact = self.store.write_text(
                f"{run_dir}/derivation.md",
                "# Dry run\n\nNo mathematical claim is accepted in dry-run mode.\n",
            )
            return WorkResult(
                work_id=item.work_id,
                outcome="partial",
                summary="Dry run validated derivation control flow without creating evidence.",
                artifact_refs=[artifact],
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "Produce one self-contained mathematical result for the exact evidence gap. "
                    "Counterexamples, obstructions, sharp boundaries, and corrected weaker claims are "
                    "as valuable as positive proofs. State assumptions and definitions, use labelled "
                    "steps with explicit dependencies, actively test boundary cases, and never assume "
                    "the target claim. Incorporate all preserved reviewer corrections."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "work_item": item.model_dump(mode="json"),
                        "prior_context": prior_context,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        input_ref = self.store.write_json(
            f"{run_dir}/derivation_input.json", {"messages": messages}
        )
        derivation = self.router.complete_structured(
            task_type="derivation",
            messages=messages,
            schema=DerivationSubmission,
            allow_repair=True,
        )
        draft_ref = self.store.write_json(f"{run_dir}/derivation.json", derivation)
        review_messages = [
            {
                "role": "system",
                "content": (
                    "Act as an adversarial mathematical referee. Recompute transitions, check "
                    "quantifiers and edge cases, detect circular premises and omitted costs, and try a "
                    "concrete counterexample. Evaluate every success criterion. Reject only defects "
                    "that change mathematical validity, scope, or the requested conclusion; cosmetic "
                    "notation and immutable work-item metadata are not reasons to reject a sound result. "
                    "The work-item hypothesis is scheduling context and cannot be edited by the derivation. "
                    "Set accepted=false whenever a substantive revision is still required; accepted means "
                    "the scientific claim and argument are usable as written."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "work_item": item.model_dump(mode="json"),
                        "derivation": derivation.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        review = self.router.complete_structured(
            task_type="derivation_review",
            messages=review_messages,
            schema=DerivationReview,
            allow_repair=False,
        )
        review_ref = self.store.write_json(f"{run_dir}/derivation_review.json", review)
        missing = _missing_criterion_reviews(item.success_criteria, review)
        substantive_revisions = [
            revision
            for revision in review.required_revisions
            if not re.search(
                r"(?i)(?:immutable|work[- ]item|hypothesis field|metadata)", revision
            )
        ]
        metadata_only_rejection = (
            not review.accepted
            and bool(review.required_revisions)
            and not substantive_revisions
            and all(row.satisfied for row in review.criteria)
        )
        accepted = (
            (review.accepted or metadata_only_rejection)
            and review.confidence >= 0.65
            and not missing
            and not review.fatal_issues
            and not substantive_revisions
        )
        if not accepted:
            issues = [*review.fatal_issues, *substantive_revisions]
            if review.accepted and review.confidence < 0.65:
                issues.append(f"Referee confidence {review.confidence:.2f} is below 0.65.")
            if missing:
                issues.append("Missing or failed criterion reviews: " + "; ".join(missing))
            return WorkResult(
                work_id=item.work_id,
                outcome="partial",
                failure_class="method",
                criteria=review.criteria,
                summary="The derivation requires another targeted revision.",
                artifact_refs=[input_ref, draft_ref, review_ref],
                errors=issues,
                next_steps=substantive_revisions,
            )
        polarity = (
            FindingPolarity.contradicts
            if derivation.result_kind in {"counterexample", "obstruction"}
            else FindingPolarity.characterizes
            if derivation.result_kind in {"characterization", "equivalence"}
            else FindingPolarity.supports
        )
        finding = Finding(
            work_id=item.work_id,
            question_id=item.question_id,
            requirement_id=item.requirement_id,
            kind=WorkKind.derivation,
            statement=derivation.conclusion,
            status=(
                FindingStatus.refuted
                if derivation.result_kind == "counterexample"
                else FindingStatus.derived
            ),
            polarity=polarity,
            strength=EvidenceStrength.substantive,
            scope="; ".join(derivation.assumptions),
            evidence_refs=[draft_ref, review_ref],
            source_ids=[],
            caveats=[
                "Accepted by an automated adversarial referee; not kernel checked.",
                *derivation.limitations,
            ],
        )
        return WorkResult(
            work_id=item.work_id,
            outcome="done",
            evidence_level="substantive",
            requirement_satisfied=True,
            criteria=review.criteria,
            summary=review.summary,
            findings=[finding],
            artifact_refs=[input_ref, draft_ref, review_ref],
        )


def _missing_criterion_reviews(expected: list[str], review: DerivationReview) -> list[str]:
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
        "it", "of", "on", "or", "that", "the", "this", "to", "was", "with",
    }

    def terms(value: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", value.lower())
            if len(token) > 2 and token not in stop
        }

    rows = [(terms(row.criterion), row) for row in review.criteria]
    missing: list[str] = []
    for criterion in expected:
        wanted = terms(criterion)
        matches = [
            row
            for found, row in rows
            if wanted
            and found
            and len(wanted & found) / max(1, min(len(wanted), len(found))) >= 0.45
        ]
        if not matches or not any(row.satisfied for row in matches):
            missing.append(criterion)
    return missing
