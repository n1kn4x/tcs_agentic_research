"""Deterministic research scheduling, novelty checks, validation, and reporting."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Literal

from .artifact_store import ArtifactStore
from .schemas import (
    ArtifactRef,
    Contribution,
    EvidenceRequirement,
    Finding,
    FindingPolarity,
    FindingStatus,
    LiteraturePlan,
    PlanSubmission,
    RequirementStatus,
    ResearchAgenda,
    ResearchAgendaDraft,
    ResearchQuestion,
    ResearchQuestionDraft,
    ResearchPhase,
    WorkItem,
    WorkItemDraft,
    WorkKind,
    WorkQueue,
    WorkStatus,
    WorkspaceState,
)


def requirement_index(
    agenda: ResearchAgenda,
) -> dict[str, tuple[ResearchQuestion, EvidenceRequirement]]:
    return {
        requirement.requirement_id: (question, requirement)
        for question in agenda.questions
        for requirement in question.requirements
    }


def _default_plan(
    *,
    agenda: ResearchAgenda,
    queue: WorkQueue,
    max_method_attempts: int,
    limit: int = 4,
) -> PlanSubmission:
    """Schedule least-attempted gaps and methods without inventing a scientific claim."""
    scientific_attempts: dict[tuple[str, WorkKind], int] = {}
    scheduled_variants: dict[tuple[str, WorkKind], int] = {}
    requirements = requirement_index(agenda)
    for item in queue.items:
        requirement_pair = requirements.get(item.requirement_id)
        if requirement_pair is None:
            continue
        _, requirement = requirement_pair
        key = (item.requirement_id, item.kind)
        scheduled_variants[key] = scheduled_variants.get(key, 0) + 1
        if item.strategy_fingerprint in requirement.attempted_strategy_fingerprints:
            scientific_attempts[key] = scientific_attempts.get(key, 0) + 1
    drafts: list[WorkItemDraft] = []
    active_pairs = {
        (item.requirement_id, item.kind)
        for item in queue.items
        if item.status in {WorkStatus.open, WorkStatus.running}
    }
    candidates: list[
        tuple[int, int, int, ResearchQuestion, EvidenceRequirement, WorkKind]
    ] = []
    for q_index, question in enumerate(agenda.questions):
        for requirement in question.requirements:
            if requirement.status in {RequirementStatus.satisfied, RequirementStatus.blocked}:
                continue
            for method in requirement.acceptable_methods:
                if method == WorkKind.synthesis:
                    continue
                if (requirement.requirement_id, method) in active_pairs:
                    continue
                key = (requirement.requirement_id, method)
                attempt_count = scientific_attempts.get(key, 0)
                variant = scheduled_variants.get(key, 0)
                if attempt_count >= max_method_attempts or (
                    attempt_count == 0 and variant >= max_method_attempts
                ):
                    continue
                candidates.append(
                    (attempt_count, q_index, variant, question, requirement, method)
                )
    # First cover distinct methods (triangulation), then distinct requirements, then fill. This
    # prevents a broad agenda from spending its first batch on four near-identical experiments.
    ordered = sorted(candidates, key=lambda row: row[:2])
    used_pairs: set[tuple[str, WorkKind]] = set()
    used_methods: set[WorkKind] = set()
    for pass_number in range(3):
        for _, _, variant, question, requirement, method in ordered:
            pair = (requirement.requirement_id, method)
            if pair in used_pairs:
                continue
            if pass_number == 0 and method in used_methods:
                continue
            if pass_number == 1 and any(
                draft.requirement_id == requirement.requirement_id for draft in drafts
            ):
                continue
            drafts.append(
                _work_for_requirement(question, requirement, method, variant=variant)
            )
            used_pairs.add(pair)
            used_methods.add(method)
            if len(drafts) >= limit:
                break
        if len(drafts) >= limit:
            break
    return PlanSubmission(
        decision="continue" if drafts else "review",
        objective=f"Reduce auditable uncertainty in: {agenda.objective[:700]}",
        work_items=drafts,
        reason=(
            "Selected least-attempted methods for open atomic evidence requirements."
            if drafts
            else "Every allowed method for every open requirement is exhausted."
        ),
    )


def _work_for_requirement(
    question: ResearchQuestion,
    requirement: EvidenceRequirement,
    method: WorkKind,
    *,
    variant: int = 0,
) -> WorkItemDraft:
    hypothesis = question.hypotheses[0]
    criterion_text = "; ".join(requirement.acceptance_criteria)
    if method == WorkKind.literature:
        strategy = "Targeted primary-source search followed by exact-span relevance review"
        instruction = (
            f"Resolve `{requirement.description}` for `{question.question}`. Search primary sources "
            "using theorem names, definitions, and citation trails; import accessible full text; "
            "extract exact quote spans; and reject merely topical passages. A bounded search that "
            "finds nothing must list queries, candidates, and access failures, not claim absence."
        )
        falsification = (
            "The source states assumptions or a result incompatible with the working hypothesis, "
            "or no relevant exact span survives relevance review."
        )
        method_criteria = [
            "At least one relevant exact quote has stable source and span provenance.",
            "The quote's relation to this requirement is explicitly reviewed.",
        ]
    elif method == WorkKind.experiment:
        strategy = "Pre-registered small experiment with frozen protocol and condition-level output"
        instruction = (
            f"Test `{requirement.description}` for `{question.question}`. First specify experimental "
            "units, null outcome, distinct baselines, correctness checks, sample sizes, fixed seeds, "
            "cost accounting, analysis rule, and resource cap. Only then implement the reviewed "
            "protocol. Preserve condition-level measurements and report negative, null, capped, and "
            "positive outcomes identically."
        )
        falsification = (
            "The pre-registered decision rule is met in the opposite direction, or its uncertainty "
            "interval/diagnostic makes the expected effect indistinguishable from the null."
        )
        method_criteria = [
            "The frozen protocol is implemented rather than a proxy experiment.",
            "All correctness checks pass and raw condition-level observations are preserved.",
        ]
    elif method == WorkKind.proof:
        strategy = "One relevance-reviewed Lean proposition on the shortest formal dependency path"
        instruction = (
            f"Advance `{requirement.description}` for `{question.question}` with one exact Lean "
            "proposition. It must be nontrivial, directly reusable in the requested argument, and "
            "expressible with configured imports. Review relevance before proof search. Preserve "
            "new kernel-checked sublemmas even if the root remains open."
        )
        falsification = (
            "Lean rejects the proposition, a relevance review shows it does not advance the gap, "
            "or bounded search exposes a smaller explicit dependency."
        )
        method_criteria = [
            "The proposition elaborates and passes a nontrivial relevance review.",
            "The root or a genuinely useful new sublemma is kernel checked without placeholders.",
        ]
    else:
        strategy = "Explicit assumption-to-conclusion derivation with adversarial counterexample review"
        instruction = (
            f"Derive or refute `{requirement.description}` for `{question.question}`. State all "
            "definitions and assumptions, give dependency-labelled argument steps, test boundary "
            "cases and counterexamples, and characterize the strongest conclusion actually justified. "
            "A counterexample, obstruction, or corrected weaker theorem is a successful result."
        )
        falsification = (
            "A concrete counterexample violates the proposed claim under its stated assumptions, "
            "or an argument step fails adversarial review."
        )
        method_criteria = [
            "Every conclusion follows from explicit assumptions through checkable steps.",
            "An independent review actively attempts a counterexample and accepts the final scope.",
        ]
    variant_labels = {
        WorkKind.literature: [
            "direct keyword/title search", "citation-chain and author search",
            "preprint/version and exact theorem-name search", "adjacent survey bibliography search",
        ],
        WorkKind.experiment: [
            "bounded direct comparison", "corrected protocol replication",
            "boundary-condition stress test", "independent baseline triangulation",
        ],
        WorkKind.proof: [
            "direct finite lemma", "dependency-first formulation",
            "counterexample-guided weaker lemma", "alternate representation lemma",
        ],
        WorkKind.derivation: [
            "direct derivation", "counterexample-first analysis",
            "extremal boundary analysis", "independent alternate derivation",
        ],
        WorkKind.synthesis: ["evidence synthesis"],
    }
    label_options = variant_labels[method]
    variant_label = label_options[min(variant, len(label_options) - 1)]
    strategy = f"{strategy}; variant {variant + 1}: {variant_label}"
    instruction += f" Use the `{variant_label}` strategy and distinguish it from prior attempts."
    if method == WorkKind.experiment:
        instruction += (
            " This label selects the scientific design; it never requests smoke-mode evidence. "
            "Smoke is only an engineering gate, and accepted evidence must come from the frozen full run."
        )
    return WorkItemDraft(
        question_id=question.question_id,
        requirement_id=requirement.requirement_id,
        kind=method,
        title=f"{method.value.title()} strategy for {requirement.requirement_id}",
        instruction=instruction,
        strategy=strategy,
        hypothesis=hypothesis,
        falsification_criterion=falsification,
        expected_information_gain=(
            f"Either satisfy the gap `{requirement.description}` under `{criterion_text}`, or rule "
            "out this scoped strategy with a preserved counterexample, null result, or precise defect."
        ),
        success_criteria=[*requirement.acceptance_criteria, *method_criteria][:6],
    )


def _normalize_work_draft(
    draft: WorkItemDraft,
    *,
    question: ResearchQuestion,
    requirement: EvidenceRequirement,
) -> WorkItemDraft:
    """Preserve the proposed strategy while appending method-specific safety constraints."""
    instruction = draft.instruction.rstrip()
    neutral_criteria = [
        criterion
        for criterion in draft.success_criteria
        if not re.search(
            r"(?i)(?:\bconfirm(?:s|ed)?\b|\bsupports?\s+(?:the\s+)?hypothesis\b|"
            r"\b(?:outperform|beats?|improves?|wins?)\b|"
            r"(?:metric|ratio|bits?|cost|time|error|accuracy|mean)\s*(?:is|are|must be|[<>])"
            r"\s*(?:significantly\s+)?(?:lower|higher|better|worse|less|greater)\b)",
            criterion,
        )
    ]
    safeguards = {
        WorkKind.experiment: [
            "All correctness checks pass and condition-level observations are preserved.",
            "The frozen protocol's baselines, costs, seeds, and decision rule are implemented.",
        ],
        WorkKind.proof: [
            "The proposition elaborates, is nontrivial, and passes dependency relevance review.",
            "Every claimed proof artifact is kernel checked without placeholders.",
        ],
        WorkKind.derivation: [
            "The conclusion follows from explicit assumptions through checkable steps.",
            "An adversarial review attempts a counterexample and accepts the final scope.",
        ],
        WorkKind.literature: [
            "Every accepted statement has an exact validated primary-source quote span.",
            "A requirement-level review rejects merely topical statements.",
        ],
        WorkKind.synthesis: [],
    }
    criteria = list(
        dict.fromkeys([
            *requirement.acceptance_criteria,
            *neutral_criteria,
            *safeguards[draft.kind],
        ])
    )[:6]
    if draft.kind == WorkKind.experiment:
        instruction += (
            " The expected direction is never a correctness assertion. The run must emit the same "
            "schema and measurements when the hypothesis is contradicted or null."
        )
    elif draft.kind == WorkKind.proof:
        instruction += (
            " Do not replace the requested result with reflexivity, `True`, or another irrelevant "
            "tautology merely because it is easy to prove."
        )
    elif draft.kind == WorkKind.derivation:
        instruction += (
            " Do not restate the hypothesis as a premise or use empirical examples as a universal proof."
        )
    return draft.model_copy(
        update={
            "question_id": question.question_id,
            "requirement_id": requirement.requirement_id,
            "instruction": instruction[:4000],
            "success_criteria": criteria,
        }
    )


def _strategy_fingerprint(draft: WorkItemDraft) -> str:
    normalized = "\0".join(
        [
            draft.requirement_id,
            draft.kind.value,
            _normalize_text(draft.strategy),
            _normalize_text(draft.instruction),
        ]
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _finding_fingerprint(finding: Finding) -> str:
    identity = {
        "requirement": finding.requirement_id,
        "kind": finding.kind.value,
        "status": finding.status.value,
        "polarity": finding.polarity.value,
        "statement": _normalize_text(finding.statement),
        "sources": (
            sorted(finding.source_ids)
            if finding.kind in {WorkKind.literature, WorkKind.experiment}
            else []
        ),
        "evidence": (
            sorted((ref.sha256 or ref.path) for ref in finding.evidence_refs)
            if finding.kind in {WorkKind.literature, WorkKind.experiment}
            else []
        ),
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _contribution_kind(finding: Finding) -> Literal[
    "positive_result", "negative_result", "null_result", "characterization",
    "verified_subgoal", "source_evidence", "derived_result",
]:
    if finding.kind == WorkKind.proof and finding.status == FindingStatus.verified:
        return "verified_subgoal"
    if finding.kind == WorkKind.literature:
        return "source_evidence"
    if finding.kind == WorkKind.derivation:
        return "derived_result"
    if finding.polarity == FindingPolarity.contradicts:
        return "negative_result"
    if finding.polarity == FindingPolarity.null:
        return "null_result"
    if finding.polarity in {FindingPolarity.characterizes, FindingPolarity.inconclusive}:
        return "characterization"
    return "positive_result"


def _new_contributions(
    *,
    findings: Iterable[Finding],
    existing_fingerprints: set[str],
    result_id: str,
) -> list[Contribution]:
    contributions: list[Contribution] = []
    for finding in findings:
        if finding.status == FindingStatus.hypothesis:
            continue
        fingerprint = _finding_fingerprint(finding)
        if fingerprint in existing_fingerprints:
            continue
        existing_fingerprints.add(fingerprint)
        contributions.append(
            Contribution(
                fingerprint=fingerprint,
                work_id=finding.work_id,
                result_id=result_id,
                question_id=finding.question_id,
                requirement_id=finding.requirement_id,
                kind=_contribution_kind(finding),
                summary=finding.statement,
                finding_ids=[finding.finding_id],
            )
        )
    return contributions


def _next_open(queue: WorkQueue) -> WorkItem | None:
    """Choose fairly instead of letting one resumable pipeline starve every other gap."""
    candidates = [item for item in queue.items if item.status == WorkStatus.open]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item.attempts, item.updated_at, item.created_at))


def _task_summary(markdown: str, *, limit: int = 600) -> str:
    lines = [line.strip("# ").strip() for line in markdown.splitlines() if line.strip()]
    return " ".join(lines)[:limit]


def _deterministic_agenda(task: str) -> ResearchAgendaDraft:
    """Extract fallback questions without treating task hypotheses as true."""
    headings: list[str] = []
    for match in re.finditer(r"(?m)^#{2,5}\s+(.+?)\s*$", task):
        title = match.group(1).strip()
        if re.search(
            r"(?i)\b(?:rq\s*\d+|research question|question|theorem\s*\d+|hypothesis\s*[a-z0-9]+)\b",
            title,
        ):
            headings.append(title)
    headings.extend(
        re.sub(r"\s+", " ", line).strip()
        for line in task.splitlines()
        if line.strip().endswith("?") and len(line.strip()) >= 10
    )
    if not headings:
        headings = [_task_summary(task, limit=1000)]
    lowered = task.lower()
    methods: list[WorkKind] = []
    if re.search(r"\b(?:literature|citation|primary source|papers?)\b", lowered):
        methods.append(WorkKind.literature)
    if re.search(r"\b(?:experiment|empirical|benchmark|dataset|simulation)\b", lowered):
        methods.append(WorkKind.experiment)
    if re.search(r"\b(?:lean|leap|formaliz|kernel[- ]checked)\b", lowered):
        methods.append(WorkKind.proof)
    if re.search(r"\b(?:prove|theorem|derive|bound|complexity|analysis)\b", lowered) or not methods:
        methods.append(WorkKind.derivation)
    methods = list(dict.fromkeys(methods))[:4]
    questions = [
        ResearchQuestionDraft(
            question=title if title.endswith("?") else f"What is the answer to: {title}?",
            hypotheses=[
                f"The proposed relationship in `{title[:500]}` holds under explicitly stated assumptions."
            ],
            evidence_needed=[
                "A direct result that resolves the question under explicit assumptions, including "
                "the relevant boundary cases or independent checks."
            ],
            preferred_methods=methods,
        )
        for title in list(dict.fromkeys(headings))[:24]
    ]
    constraints = [
        re.sub(r"\s+", " ", line).strip(" *-")[:500]
        for line in task.splitlines()
        if re.search(
            r"(?i)\b(?:must|assume|excluded|include all|do not|avoid|keep|"
            r"fixed[- ]width|fixed seeds?|uniquely decodable)\b",
            line,
        )
    ][:12]
    return ResearchAgendaDraft(
        objective=_task_summary(task, limit=1800),
        constraints=constraints,
        questions=questions,
        deliverables=[
            "A requirement-by-requirement evidence matrix",
            "Positive, negative, null, and inconclusive results with identical provenance",
            "A research report that distinguishes evidence from remaining hypotheses",
        ],
    )


def _ensure_requested_methods(draft: ResearchAgendaDraft, task: str) -> ResearchAgendaDraft:
    """Add auditable gaps when a model omits an explicitly requested evidence product.

    Merely adding a method to ``preferred_methods`` is insufficient: method routing is based on the
    evidence description, so a model-generated list of literature-review questions could otherwise
    erase every requested derivation and experiment. Explicit numbered theorems are retained as
    separate derivation gaps. This is deterministic task coverage, not a scientific fallback.
    """
    lowered = task.lower()
    requested: list[WorkKind] = []
    if re.search(r"\b(?:literature|literaturedb|primary sources?|citations?|papers?)\b", lowered):
        requested.append(WorkKind.literature)
    if re.search(r"\b(?:experimenter|experiments?|empirical|benchmarks?|datasets?)\b", lowered):
        requested.append(WorkKind.experiment)
    if re.search(r"\b(?:lean|leap|kernel[- ]checked|compiler[- ]verified)\b", lowered):
        requested.append(WorkKind.proof)
    if re.search(r"\b(?:proof|prove|theorems?|derive|analysis|complexity)\b", lowered):
        requested.append(WorkKind.derivation)
    requested = list(dict.fromkeys(requested))

    questions = [question.model_copy(deep=True) for question in draft.questions]
    injected: list[ResearchQuestionDraft] = []
    theorem_titles = list(
        dict.fromkeys(
            match.group(1).strip()
            for match in re.finditer(
                r"(?im)^#{2,5}\s+((?:theorem|lemma|proposition)\s*\d*\s*:[^\n]+)$",
                task,
            )
        )
    )
    if WorkKind.derivation in requested:
        for title in theorem_titles:
            title_terms = set(re.findall(r"[a-z0-9]{4,}", title.lower())) - {
                "theorem", "lemma", "proposition",
            }
            already_covered = any(
                WorkKind.derivation
                in {
                    method
                    for description in question.evidence_needed
                    for method in _evidence_method_hints(
                        description, question.preferred_methods
                    )
                }
                and len(
                    title_terms
                    & set(
                        re.findall(
                            r"[a-z0-9]{4,}",
                            " ".join(
                                [question.question, *question.evidence_needed]
                            ).lower(),
                        )
                    )
                )
                >= max(1, len(title_terms) // 2)
                for question in questions
            )
            if already_covered:
                continue
            injected.append(
                ResearchQuestionDraft(
                    question=f"Can the requested result `{title}` be derived or refuted under explicit assumptions?",
                    hypotheses=[
                        f"The claim named `{title}` holds under the assumptions stated in the task."
                    ],
                    evidence_needed=[
                        "An explicit mathematical derivation, counterexample, or corrected theorem "
                        f"resolving `{title}` through checkable assumption-to-conclusion steps."
                    ],
                    preferred_methods=[WorkKind.derivation],
                )
            )

    present = {
        method
        for question in questions
        for description in question.evidence_needed
        for method in _evidence_method_hints(description, question.preferred_methods)
    }
    if theorem_titles:
        present.add(WorkKind.derivation)
    literature_topic = _first_literature_topic(task)
    generic = {
        WorkKind.literature: ResearchQuestionDraft(
            question=(
                f"What do primary sources establish about `{literature_topic}`?"
                if literature_topic
                else "Which primary-source statement addresses the task's central technical topic?"
            ),
            hypotheses=[
                f"A primary source states a result about `{literature_topic}` with explicit scope."
                if literature_topic
                else "A relevant primary source states the technical claim with explicit scope."
            ],
            evidence_needed=[
                "A relevant exact primary-source quote with stable span provenance and a relevance "
                + (
                    f"review specifically addressing `{literature_topic}`."
                    if literature_topic
                    else "review tied to the named technical topic."
                )
            ],
            preferred_methods=[WorkKind.literature],
        ),
        WorkKind.experiment: ResearchQuestionDraft(
            question="What happens in the reproducible empirical comparison explicitly requested by the task?",
            hypotheses=["The requested treatments and baselines may differ on the registered measurements."],
            evidence_needed=[
                "A reproducible fixed-seed experiment implementing the requested treatments and baselines, "
                "checking correctness, preserving condition-level measurements, and reporting positive, "
                "negative, or null outcomes over the requested central regimes."
            ],
            preferred_methods=[WorkKind.experiment],
        ),
        WorkKind.proof: ResearchQuestionDraft(
            question="Which explicitly requested formal proposition can be verified by the Lean kernel?",
            hypotheses=["A task-relevant proposition is expressible and provable in the configured Lean environment."],
            evidence_needed=[
                "A task-relevant Lean kernel-checked theorem with compiler artifacts and no placeholders or new axioms."
            ],
            preferred_methods=[WorkKind.proof],
        ),
        WorkKind.derivation: ResearchQuestionDraft(
            question="Which central mathematical claim requested by the task can be derived or refuted?",
            hypotheses=["A central requested claim holds under explicit assumptions."],
            evidence_needed=[
                "An explicit mathematical derivation, counterexample, or corrected theorem with checkable steps."
            ],
            preferred_methods=[WorkKind.derivation],
        ),
    }
    for method in requested:
        if method not in present and not (
            method == WorkKind.derivation and theorem_titles
        ):
            injected.append(generic[method])

    # Preserve deterministic coverage even if a model fills the schema maximum with one method.
    injected = injected[:24]
    keep = max(0, 24 - len(injected))
    return draft.model_copy(update={"questions": [*questions[:keep], *injected]})


def _evidence_method_hints(
    description: str, preferred: list[WorkKind]
) -> list[WorkKind]:
    lowered = description.lower()
    methods: list[WorkKind] = []
    explicit_proof = bool(
        re.search(r"\b(?:lean|leap|kernel[- ]checked|compiler[- ]verified)\b", lowered)
    )
    if explicit_proof:
        methods.append(WorkKind.proof)
    if re.search(
        r"\b(?:empirical|experiment(?:al)?|benchmark(?:ing)?|measurement|measured|"
        r"fixed[- ]seed|condition[- ]level|dataset)\b",
        lowered,
    ):
        methods.append(WorkKind.experiment)
    explicit_literature = bool(
        re.search(
            r"\b(?:exact quote|primary[- ]source|literature review|citation|bibliograph|published result)\b",
            lowered,
        )
    )
    if explicit_literature:
        methods.append(WorkKind.literature)
    derivation_pattern = (
        r"\b(?:derive|derivation|mathematical argument|counterexample|"
        r"complexity analysis|lower bound|upper bound|inequality)\b"
    )
    if re.search(derivation_pattern, lowered) or (
        not explicit_proof
        and not explicit_literature
        and re.search(r"\b(?:proof|theorem)\b", lowered)
    ):
        methods.append(WorkKind.derivation)
    if methods:
        return list(dict.fromkeys(methods))
    return [method for method in preferred if method != WorkKind.synthesis]


def _first_literature_topic(task: str) -> str:
    heading = re.search(r"(?im)^(#{1,4})\s+[^\n]*literature review[^\n]*$", task)
    if heading is None:
        return ""
    level = len(heading.group(1))
    following = task[heading.end() :]
    for match in re.finditer(r"(?m)^(#{1,5})\s+(.+?)\s*$", following):
        if len(match.group(1)) <= level:
            break
        topic = re.sub(r"^\d+(?:\.\d+)*\s*[:.-]?\s*", "", match.group(2)).strip()
        if topic:
            return topic[:300]
    return ""


def _compact_query(text: str) -> str:
    stop = {
        "a", "an", "and", "around", "audit", "based", "central", "conditions", "determine",
        "document", "evidence", "for", "from", "identify", "in", "is", "literature", "named",
        "of", "on", "precise", "problem", "requested", "research", "result", "review", "scheme",
        "specific", "that", "the", "theoretical", "this", "to", "under", "what", "which", "with",
        "question", "resolve",
    }
    selected: list[str] = []
    seen: set[str] = set()
    for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9+.-]*", text):
        key = word.lower()
        if key in stop or key in seen or len(key) < 3:
            continue
        selected.append(word)
        seen.add(key)
        if len(selected) >= 8:
            break
    return " ".join(selected) or "theoretical computer science primary source"


def _rank_candidates(
    candidates: list[Any],
    *,
    preferred_titles: list[str] | None = None,
    relevance_queries: list[str] | None = None,
) -> list[Any]:
    unique: dict[str, Any] = {}
    for candidate in candidates:
        title_key = _normalize_text(candidate.title)
        key = title_key or candidate.doi.lower() or candidate.arxiv_id.lower() or candidate.openalex_id.lower()
        current = unique.get(key)
        rank = (bool(candidate.arxiv_id or candidate.pdf_url), candidate.score, candidate.cited_by_count)
        current_rank = (
            bool(current and (current.arxiv_id or current.pdf_url)),
            current.score if current else -1.0,
            current.cited_by_count if current else -1,
        )
        if current is None or rank > current_rank:
            unique[key] = candidate
    targets = [*(preferred_titles or []), *(relevance_queries or [])]
    target_terms = [set(re.findall(r"[a-z0-9]{3,}", value.lower())) for value in targets]

    def overlap(candidate: Any) -> float:
        terms = set(
            re.findall(
                r"[a-z0-9]{3,}",
                (candidate.title + " " + (candidate.abstract or "")[:5000]).lower(),
            )
        )
        return max(
            (len(terms & wanted) / max(1, len(wanted)) for wanted in target_terms),
            default=0.0,
        )

    return sorted(
        unique.values(),
        key=lambda item: (
            item.status == "queued",
            bool(item.arxiv_id or item.pdf_url),
            overlap(item),
            item.cited_by_count,
        ),
        reverse=True,
    )


def _candidate_is_relevant_and_extractable(
    candidate: Any,
    *,
    preferred_titles: list[str],
    relevance_queries: list[str],
) -> bool:
    if candidate.status != "queued" or not (candidate.arxiv_id or candidate.pdf_url):
        return False
    source_terms = set(
        re.findall(
            r"[a-z0-9]{3,}",
            (candidate.title + " " + (candidate.abstract or "")[:5000]).lower(),
        )
    )
    # Model-suggested titles can be hallucinated. They help rank a candidate but cannot, by
    # themselves, establish relevance; at least one requirement-derived search query must overlap.
    for target in relevance_queries:
        terms = set(re.findall(r"[a-z0-9]{3,}", target.lower()))
        if terms and (
            len(source_terms & terms) >= 2
            or len(source_terms & terms) / len(terms) >= 0.3
        ):
            return True
    return False


def _validate_experiment_program(program: Any) -> None:
    """Reject incomplete contracts and escape/network primitives before Docker."""
    code = program.python_code
    if len(code) > 20_000:
        raise ValueError("generated experiment exceeds the 20,000-character source budget")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"generated experiment is not valid Python: {exc}") from exc
    if not tree.body:
        raise ValueError("generated experiment contains no Python statements")
    run_function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run_experiment"
        ),
        None,
    )
    if run_function is None:
        raise ValueError("generated experiment must define run_experiment(mode)")
    if isinstance(run_function, ast.AsyncFunctionDef):
        raise ValueError("run_experiment must be synchronous")
    positional = [*run_function.args.posonlyargs, *run_function.args.args]
    if len(positional) != 1 or run_function.args.vararg or run_function.args.kwarg:
        raise ValueError("run_experiment must accept exactly one positional mode argument")
    mode_used = any(
        isinstance(node, ast.Name)
        and node.id == positional[0].arg
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(run_function)
    )
    if not mode_used:
        raise ValueError(
            "run_experiment must branch on mode so smoke uses tiny per-condition samples"
        )
    meaningful_body = [
        node
        for node in run_function.body
        if not isinstance(node, ast.Pass)
        and not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]
    if not meaningful_body:
        raise ValueError("run_experiment contains only an unfinished `pass` placeholder")
    if any(not _safe_experiment_top_level(node) for node in tree.body):
        raise ValueError(
            "generated experiment may only define imports, constants, classes, and functions; "
            "the trusted wrapper owns the entry point"
        )
    for top_node in tree.body:
        if not isinstance(top_node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = top_node.targets if isinstance(top_node, ast.Assign) else [top_node.target]
        if not any(isinstance(target, ast.Name) and target.id.lower() == "seeds" for target in targets):
            continue
        if top_node.value is None:
            continue
        try:
            declared_seeds = ast.literal_eval(top_node.value)
        except (ValueError, TypeError):
            continue
        if isinstance(declared_seeds, (list, tuple)) and list(declared_seeds) != program.seeds:
            raise ValueError("the seeds field does not match the program's SEEDS constant")
    forbidden_modules = {
        "asyncio", "httpx", "multiprocessing", "requests", "shutil", "socket",
        "subprocess", "urllib",
    }
    forbidden_calls = {"compile", "eval", "exec", "__import__", "exit", "quit"}
    implementation_nodes = (
        node
        for top_node in tree.body
        if not (isinstance(top_node, ast.If) and _is_main_guard(top_node))
        for node in ast.walk(top_node)
    )
    for node in implementation_nodes:
        if isinstance(node, ast.Import):
            blocked = {alias.name.split(".", 1)[0] for alias in node.names} & forbidden_modules
            if blocked:
                raise ValueError(f"generated experiment imports forbidden module(s): {sorted(blocked)}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in forbidden_modules or root == "os":
                raise ValueError(f"generated experiment imports forbidden module: {root}")
        elif isinstance(node, ast.Attribute) and _attribute_chain(node)[:1] == ["os"]:
            chain = _attribute_chain(node)
            if chain not in (
                ["os", "path"],
                ["os", "path", "join"],
                ["os", "makedirs"],
                ["os", "environ"],
                ["os", "environ", "get"],
            ):
                raise ValueError(
                    "generated experiment may use os only for relative paths and read-only environment access"
                )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in forbidden_calls:
                raise ValueError(f"generated experiment calls forbidden builtin: {node.func.id}")
            if node.func.id == "open" and node.args and isinstance(node.args[0], ast.Constant):
                path = str(node.args[0].value)
                if Path(path).is_absolute() or ".." in Path(path).parts:
                    raise ValueError("generated experiment opens a path outside its run directory")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "sys"
            and node.func.attr == "exit"
        ):
            raise ValueError("generated experiment calls sys.exit; raise on correctness defects instead")


def _safe_experiment_top_level(node: ast.stmt) -> bool:
    if isinstance(
        node,
        (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign),
    ):
        return True
    if isinstance(node, ast.Expr):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return True
        return (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and (
                _attribute_chain(node.value.func) == ["os", "makedirs"]
                or (
                    node.value.func.attr == "mkdir"
                    and isinstance(node.value.func.value, ast.Name)
                )
            )
        )
    if isinstance(node, ast.If):
        # This guard is false when the trusted wrapper imports the implementation. It is harmless
        # boilerplate and is excluded from executable-implementation safety scanning.
        return _is_main_guard(node)
    return False


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _attribute_chain(node: ast.Attribute) -> list[str]:
    parts = [node.attr]
    value: ast.expr = node.value
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return list(reversed(parts))


def _existing_refs(
    store: ArtifactStore,
    refs: list[ArtifactRef],
    paths: list[str],
) -> list[ArtifactRef]:
    by_path = {ref.path: ref for ref in refs}
    for path in paths:
        if store.exists(path):
            ref = store.artifact_ref(path)
            by_path.setdefault(ref.path, ref)
    return list(by_path.values())


def _recent_result_context(store: ArtifactStore, *, limit: int) -> list[dict[str, Any]]:
    paths = sorted(store.resolve("Runs").glob("*/result.json"), key=lambda path: path.stat().st_mtime)
    results: list[dict[str, Any]] = []
    for path in paths[-limit:]:
        try:
            row = store.read_json(path)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        results.append(
            {
                "work_id": str(row.get("work_id") or ""),
                "outcome": str(row.get("outcome") or ""),
                "evidence_level": str(row.get("evidence_level") or "none"),
                "summary": str(row.get("summary") or "")[:700],
                "errors": [str(value)[:500] for value in (row.get("errors") or [])[:4]],
                "next_steps": [str(value)[:500] for value in (row.get("next_steps") or [])[:4]],
                "result_artifact": store.relpath(path),
            }
        )
    return results


def _render_literature_report(
    item: WorkItem,
    plan: LiteraturePlan,
    candidates: list[Any],
    imported: list[Any],
    findings: list[Finding],
    errors: list[str],
) -> str:
    lines = [
        f"# Literature work: {item.title}", "",
        "Only exact spans that passed requirement-level relevance review become findings.", "",
        "## Queries", *[f"- {query}" for query in plan.search_queries], "",
        "## Discovery", f"- Candidates inspected: {len(candidates)}",
        f"- Full-text sources imported: {len(imported)}", "", "## Accepted evidence",
    ]
    if not findings:
        lines.append("- None in this bounded search.")
    for finding in findings:
        lines.append(
            f"- **{finding.polarity.value}** `{finding.finding_id}`: {finding.statement}"
        )
        lines.append(f"  - Sources: {', '.join(finding.source_ids) or 'none'}")
    lines.extend(["", "## Search/access failures"])
    lines.extend([f"- {error}" for error in errors] or ["- None."])
    return "\n".join(lines).rstrip() + "\n"


def _render_progress_report(
    state: WorkspaceState,
    agenda: ResearchAgenda | None,
    queue: WorkQueue,
    findings: list[Finding],
    contributions: list[Contribution],
) -> str:
    lines = [
        "# Research progress", "",
        f"- Phase: **{state.phase.value}**",
        f"- Completed work cycles: {state.cycle}",
        f"- Novel research contributions: {len(contributions)}",
        f"- Last contribution cycle: {state.last_progress_cycle or 'none'}",
        f"- Consecutive attempts without a novel contribution: {state.no_progress_steps}",
        f"- Diversification events: {state.diversification_count}", "",
    ]
    if state.phase == ResearchPhase.needs_input:
        lines.extend([
            "> **Action required:** every configured strategy for at least one mandatory gap is exhausted.",
            "> This state is reached only after alternatives/revisions are tried; no silent loop remains.",
            "",
        ])
    if state.phase == ResearchPhase.system_error:
        lines.extend([
            "> **System repair required:** an experiment exhausted its engineering repair budget",
            "> before producing scientific measurements. Preserved protocol and program state remain resumable.",
            "",
        ])
    if state.phase == ResearchPhase.complete:
        lines.extend(["> **Agenda complete:** every mandatory evidence requirement is satisfied.", ""])
    if agenda is None:
        return "\n".join(lines).rstrip() + "\n"
    by_finding = {finding.finding_id: finding for finding in findings}
    work_by_requirement: dict[str, list[WorkItem]] = {}
    for item in queue.items:
        work_by_requirement.setdefault(item.requirement_id, []).append(item)
    lines.extend(["## Objective", "", agenda.objective, "", "## Evidence matrix", ""])
    for question in agenda.questions:
        satisfied = sum(req.status == RequirementStatus.satisfied for req in question.requirements)
        lines.append(f"### {question.question_id}: {question.question}")
        lines.append(f"- Coverage: **{satisfied}/{len(question.requirements)} requirements satisfied**")
        lines.append("- Working hypotheses:")
        lines.extend([f"  - {hypothesis}" for hypothesis in question.hypotheses])
        for requirement in question.requirements:
            lines.append(
                f"- `{requirement.requirement_id}` **{requirement.status.value}**: "
                f"{requirement.description} (attempts={requirement.attempt_count})"
            )
            attempts = work_by_requirement.get(requirement.requirement_id, [])
            if attempts:
                lines.append(
                    "  - Strategies: "
                    + ", ".join(
                        f"{item.kind.value}/{item.status.value}/r{item.revision}"
                        for item in attempts[-8:]
                    )
                )
            for finding_id in requirement.finding_ids[-5:]:
                finding = by_finding.get(finding_id)
                if finding is None:
                    continue
                marker = "NEGATIVE" if finding.polarity == FindingPolarity.contradicts else finding.polarity.value
                lines.append(
                    f"  - **{marker} / {finding.strength.value}** `{finding.finding_id}`: "
                    f"{finding.statement[:700]}"
                )
                if finding.evidence_refs:
                    lines.append(
                        "    - Evidence: " + ", ".join(ref.path for ref in finding.evidence_refs[:5])
                    )
            if requirement.blocker:
                lines.append(f"  - Blocker: {requirement.blocker[-700:]}")
        lines.append("")
    recent_failures = [
        item for item in queue.items
        if item.status in {WorkStatus.failed, WorkStatus.blocked, WorkStatus.partial}
        and item.blocked_reason
    ]
    lines.extend(["## Recent non-contributing attempts", ""])
    lines.extend(
        [
            f"- `{item.work_id}` ({item.requirement_id}/{item.kind.value}): "
            f"{item.blocked_reason[-700:]}"
            for item in recent_failures[-10:]
        ]
        or ["- None."]
    )
    return "\n".join(lines).rstrip() + "\n"


def _render_research_report(
    agenda: ResearchAgenda | None,
    findings: list[Finding],
) -> str:
    lines = [
        "# Research report", "",
        "This report is generated from the evidence ledger. Negative and null results are first-class "
        "results; operational failures are excluded from scientific conclusions.", "",
    ]
    if agenda is None:
        return "\n".join(lines)
    by_finding = {finding.finding_id: finding for finding in findings}
    lines.extend(["## Objective", "", agenda.objective, "", "## Results by question", ""])
    for question in agenda.questions:
        lines.append(f"### {question.question_id}: {question.question}")
        question_findings = [
            by_finding[finding_id]
            for requirement in question.requirements
            for finding_id in requirement.finding_ids
            if finding_id in by_finding
        ]
        if not question_findings:
            lines.append("- **Unresolved:** no usable evidence has been produced.")
        for finding in question_findings:
            lines.append(
                f"- **{finding.status.value}; {finding.polarity.value}; {finding.strength.value}**: "
                f"{finding.statement}"
            )
            if finding.scope:
                lines.append(f"  - Scope: {finding.scope}")
            for caveat in finding.caveats:
                lines.append(f"  - Caveat: {caveat}")
            if finding.evidence_refs:
                lines.append("  - Artifacts: " + ", ".join(ref.path for ref in finding.evidence_refs[:8]))
        open_requirements = [
            req for req in question.requirements if req.status != RequirementStatus.satisfied
        ]
        if open_requirements:
            lines.append("- Remaining evidence gaps:")
            lines.extend([f"  - `{req.requirement_id}`: {req.description}" for req in open_requirements])
        lines.append("")
    lines.extend(["## Deliverable coverage", ""])
    lines.extend([f"- {deliverable}" for deliverable in agenda.deliverables])
    return "\n".join(lines).rstrip() + "\n"


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
