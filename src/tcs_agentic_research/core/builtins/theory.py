"""Autonomous theoretical notebook subsystem.

Its output is always tentative.  It can create useful questions, arguments, and challenges, but it
has no path to self-certification.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import Field

from ...llm import LLMRouter
from ...schemas import StrictModel
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


class TheoryChoice(StrictModel):
    action: Literal["investigate", "challenge", "synthesize", "idle"]
    focus: str = Field(default="", max_length=1200)
    rationale: str = Field(default="", max_length=2000)
    parent_ids: list[str] = Field(default_factory=list, max_length=12)


class NotebookEntry(StrictModel):
    title: str = Field(min_length=3, max_length=300)
    summary: str = Field(min_length=10, max_length=2500)
    analysis: str = Field(min_length=10, max_length=20_000)
    assumptions: list[str] = Field(default_factory=list, max_length=20)
    uncertainty: list[str] = Field(min_length=1, max_length=20)
    next_questions: list[str] = Field(default_factory=list, max_length=6)


class TheorySubsystem:
    name = "theory"
    description = "Develops tentative arguments, counterexamples, questions, and syntheses."
    model_call_budget = 3

    def __init__(self, router: LLMRouter):
        self.router = router

    def propose(self, view: ResearchView) -> ActionProposal | None:
        if self.router.dry_run:
            if view.subsystem_state.get("dry_run_completed"):
                return None
            return ActionProposal(
                subsystem=self.name,
                action_type="investigate",
                title="Dry-run notebook move",
                rationale="Exercise the actor contract without asserting evidence.",
                payload={"focus": "Restate one uncertainty in the task.", "mode": "dry_run"},
            )
        choice = self.router.complete_structured(
            task_type="theory_decision",
            schema=TheoryChoice,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the autonomous theory subsystem. Choose one atomic move that adds "
                        "new information to a persistent research notebook. Use existing record IDs "
                        "when extending or challenging prior work. Prefer checking a weak point, a "
                        "small derivation, or a synthesis that exposes a concrete next question. "
                        "Yield idle rather than repeat prior work. You cannot verify your own prose."
                    ),
                },
                {"role": "user", "content": view.model_dump_json()},
            ],
        )
        if choice.action == "idle":
            return None
        return ActionProposal(
            subsystem=self.name,
            action_type=choice.action,
            title=choice.focus[:300] or f"Theory {choice.action}",
            rationale=choice.rationale,
            payload={"focus": choice.focus},
            parent_ids=[
                record_id
                for record_id in choice.parent_ids
                if record_id in {record.record_id for record in view.records}
            ],
        )

    def execute(
        self, proposal: ActionProposal, view: ResearchView, *, run_dir: str
    ) -> ActionOutcome:
        if self.router.dry_run:
            draft = RecordDraft(
                kind=RecordKind.question,
                title="Dry-run research uncertainty",
                summary="Which exact claim in the task should receive external evidence first?",
                body="This record is deliberately tentative and contains no scientific claim.",
                evidence=EvidenceReceipt(evidence_type=EvidenceType.model),
            )
            return ActionOutcome(
                summary="Created one explicitly tentative dry-run notebook record.",
                records=[draft],
                state_patch={"dry_run_completed": True},
            )
        entry = self.router.complete_structured(
            task_type="theory_investigation",
            schema=NotebookEntry,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Carry out the selected theoretical notebook move. Separate assumptions "
                        "from deductions, actively seek a counterexample, and state uncertainty. "
                        "Preserve the task's exact mathematical objects and representation; do not "
                        "silently substitute a different meaning for a technical term. If the object "
                        "is ambiguous, return a clarifying question instead of choosing a new domain. "
                        "Do not claim proof, verification, novelty, or literature support. Return a "
                        "self-contained entry that another subsystem can build on."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "proposal": proposal.model_dump(mode="json"),
                            "shared_view": view.model_dump(mode="json"),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        kind = {
            "challenge": RecordKind.counterexample,
            "synthesize": RecordKind.synthesis,
        }.get(proposal.action_type, RecordKind.analysis)
        relation = (
            {
                "challenge": RecordRelation.challenges,
                "synthesize": RecordRelation.extends,
            }.get(proposal.action_type, RecordRelation.extends)
            if proposal.parent_ids
            else RecordRelation.none
        )
        body = entry.analysis
        if entry.assumptions:
            body += "\n\nAssumptions:\n" + "\n".join(f"- {item}" for item in entry.assumptions)
        body += "\n\nUncertainty:\n" + "\n".join(f"- {item}" for item in entry.uncertainty)
        records = [
            RecordDraft(
                kind=kind,
                title=entry.title,
                summary=entry.summary,
                body=body,
                relation=relation,
                parent_ids=proposal.parent_ids,
                evidence=EvidenceReceipt(evidence_type=EvidenceType.model),
            )
        ]
        for question in entry.next_questions:
            records.append(
                RecordDraft(
                    kind=RecordKind.question,
                    title=question[:300],
                    summary=question,
                    relation=(
                        RecordRelation.extends
                        if proposal.parent_ids
                        else RecordRelation.none
                    ),
                    parent_ids=proposal.parent_ids,
                    evidence=EvidenceReceipt(evidence_type=EvidenceType.model),
                )
            )
        prior_topics = list(view.subsystem_state.get("topics", []))
        prior_topics.append(proposal.title)
        return ActionOutcome(
            summary=f"Added a tentative notebook entry with {len(entry.next_questions)} follow-up question(s).",
            records=records,
            state_patch={"topics": prior_topics[-30:], "last_action": proposal.action_type},
        )
