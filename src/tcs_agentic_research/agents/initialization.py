"""Initialization interview and artifact synthesis."""

from __future__ import annotations

from typing import Any

from ..artifact_store import ArtifactStore
from ..llm import LLMRouter
from ..prompt_loader import render_prompt
from ..schemas import (
    ArtifactKind,
    ClaimRecord,
    ClaimStatus,
    ClaimType,
    EvidenceRecord,
    EvidenceType,
    InitializationBundle,
    ProposalLedgerEntry,
    ResearchState,
)


INTERVIEW_QUESTIONS = [
    "What is the exact TCS research problem, including the computational model?",
    "What assumptions, oracle access, promises, distributions, or cryptographic hardness assumptions are allowed?",
    "What would count as a solution, and what partial outcomes would be publishable?",
    "Which papers, barriers, lower bounds, or known algorithms should be treated as essential context?",
    "What notation, definitions, theorem statements, or Lean snippets should be canonical?",
    "Which tools are desired (Lean/mathlib, SAT/SMT, Python numerics, quantum simulators, etc.)?",
]


class InitializationAgent:
    def __init__(self, store: ArtifactStore, router: LLMRouter, *, prompt_dir: str | None = None):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir

    def initialize(self, *, user_seed: str, interview_answers: dict[str, Any] | None = None) -> ResearchState:
        self.store.initialize_layout()
        fallback = self._fallback_bundle(user_seed, interview_answers or {})
        messages = [
            {"role": "system", "content": render_prompt("initialization_interviewer", override_dir=self.prompt_dir)},
            {
                "role": "user",
                "content": (
                    "Synthesize the initialization artifacts from this seed and interview answers.\n\n"
                    f"Seed:\n{user_seed}\n\nAnswers:\n{interview_answers or {}}"
                ),
            },
        ]
        bundle = self.router.complete_structured(
            task_type="initialization",
            messages=messages,
            schema=InitializationBundle,
            fallback=fallback,
        )
        return self.commit_bundle(bundle)

    def commit_bundle(self, bundle: InitializationBundle) -> ResearchState:
        task_ref = self.store.write_text(ArtifactStore.RESEARCH_TASK, bundle.research_task_markdown)
        nomenclature_payload = {
            "version": 1,
            "symbols": [entry.model_dump(mode="json") for entry in bundle.nomenclature_entries],
            "conventions": [],
            "notes": bundle.initial_state_notes,
        }
        nomenclature_ref = self.store.write_yaml(ArtifactStore.NOMENCLATURE, nomenclature_payload)
        claims = bundle.initial_claims or [
            ClaimRecord(
                claim_type=ClaimType.definition,
                statement="Research task initialized; substantive scientific claims remain unverified.",
                status=ClaimStatus.proposed,
                evidence=[
                    EvidenceRecord(
                        evidence_type=EvidenceType.none,
                        summary="Initialization placeholder, not a mathematical result.",
                        confidence=0.0,
                    )
                ],
            )
        ]
        self.store.append_claims(claims)
        state = ResearchState(
            task_summary=_task_summary(bundle.research_task_markdown),
            active_claim_ids=[claim.claim_id for claim in claims],
            artifact_refs=[
                task_ref,
                nomenclature_ref,
                self.store.artifact_ref(ArtifactStore.CLAIM_LEDGER, kind=ArtifactKind.jsonl),
                self.store.artifact_ref(ArtifactStore.PROPOSAL_LEDGER, kind=ArtifactKind.jsonl),
            ],
            notes=bundle.initial_state_notes,
        )
        state_ref = self.store.save_state(state)
        state.artifact_refs.append(state_ref)
        self.store.append_proposal_event(
            ProposalLedgerEntry(
                proposal_id="initialization",
                event_type="accepted",
                reason="Initialization created canonical task, nomenclature, and initial ledger entries.",
                artifact_refs=[task_ref, nomenclature_ref, state_ref],
            )
        )
        return state

    def _fallback_bundle(self, user_seed: str, answers: dict[str, Any]) -> InitializationBundle:
        answer_lines = "\n".join(f"- **{k}:** {v}" for k, v in answers.items()) or "- No answers supplied."
        task_md = f"""# Research Task

## Problem statement
{user_seed.strip() or "TBD: specify the exact theoretical computer science research problem."}

## Computational model and assumptions
TBD during the initialization interview. Record oracle access, randomness, quantum resources,
promise structure, cryptographic assumptions, and asymptotic conventions here.

## Success criteria
- A main-task solution requires proof-quality mathematical evidence, resource accounting, and independent replication.
- Experimental observations may suggest conjectures but are not proofs.

## Fallback publishable outcomes
- A verified obstruction or lower-bound explanation.
- A clarified formalization with useful Lean lemmas.
- A reproducible experimental counterexample search or benchmark.
- A literature synthesis resolving notation and novelty questions.

## Known barriers and literature context
TBD. Literature claims must be entered into `LiteratureDB` with provenance before use.

## User-supplied domain knowledge
{answer_lines}

## Tooling constraints
- LangGraph orchestrates resumable agent loops.
- vLLM serves local LLMs through an OpenAI-compatible API.
- LEAP/Lean is used for central formalizable claims.
- Experiments must be reproducible under `ExperimentRuns/`.

## Canonical notation
See `Nomenclature.yml`.
"""
        initial_claim = ClaimRecord(
            claim_type=ClaimType.definition,
            statement="The research task and acceptance criteria are defined by ResearchTask.md.",
            status=ClaimStatus.proposed,
            evidence=[
                EvidenceRecord(
                    evidence_type=EvidenceType.none,
                    summary="Administrative initialization claim.",
                    confidence=0.0,
                )
            ],
        )
        return InitializationBundle(
            research_task_markdown=task_md,
            initial_state_notes=["Initialized from user seed; unresolved details should be clarified."],
            initial_claims=[initial_claim],
            fallback_publishable_outcomes=[
                "verified obstruction",
                "formalized lemma library",
                "reproducible experiment/counterexample search",
                "literature and notation synthesis",
            ],
            assumptions=["TBD"],
            success_criteria=["conservative solved check and independent replication"],
        )


def _task_summary(markdown: str, limit: int = 500) -> str:
    lines = [line.strip("# ").strip() for line in markdown.splitlines() if line.strip()]
    text = " ".join(lines)
    return text[:limit]
