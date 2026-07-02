"""LLM-guided initialization interview and artifact synthesis."""

from __future__ import annotations

from collections.abc import Callable

from ..artifact_store import ArtifactStore
from ..llm import LLMRouter, StructuredLLMError
from ..prompt_loader import render_prompt
from ..schemas import (
    ArtifactKind,
    ArtifactRef,
    ClaimRecord,
    ClaimStatus,
    ClaimType,
    EvidenceRecord,
    EvidenceType,
    InitializationBundle,
    InitializationInterviewTurn,
    ProposalLedgerEntry,
    ResearchState,
)


INTERVIEW_TRANSCRIPT = "InitializationInterview.md"
_INITIAL_CONTEXT_PREFIX = "Initial context supplied before the interview:"


class InitializationAgent:
    def __init__(self, store: ArtifactStore, router: LLMRouter, *, prompt_dir: str | None = None):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir

    def initialize_interactively(
        self,
        *,
        initial_context: str = "",
        input_func: Callable[[str], str] = input,
        output_func: Callable[[str], None] = print,
        max_turns: int = 12,
    ) -> ResearchState:
        """Run an adaptive LLM interview, then synthesize canonical init artifacts."""
        self.store.initialize_layout()
        transcript: list[dict[str, str]] = []
        if initial_context.strip():
            transcript.append(
                {
                    "role": "user",
                    "content": f"{_INITIAL_CONTEXT_PREFIX}\n{initial_context.strip()}",
                }
            )

        output_func(
            "Starting LLM-guided initialization interview. "
            "Answer the questions; type /done to synthesize with the current information."
        )
        for _turn_idx in range(max_turns):
            turn = self.next_interview_turn(transcript)
            if turn.ready_to_initialize and not _has_user_answer(transcript):
                turn = self._first_question_turn(has_initial_context=bool(initial_context.strip()))

            output_func(turn.assistant_message)
            transcript.append({"role": "assistant", "content": turn.assistant_message})
            if turn.ready_to_initialize:
                break

            answer = input_func("> ").strip()
            lowered = answer.lower()
            if lowered in {"/quit", "/exit"}:
                raise KeyboardInterrupt("initialization interview aborted by user")
            if lowered == "/done":
                transcript.append(
                    {
                        "role": "user",
                        "content": "[User requested initialization with the current information.]",
                    }
                )
                output_func("Synthesizing initialization artifacts from the current conversation.")
                break
            transcript.append({"role": "user", "content": answer})
        else:
            output_func("Reached the interview turn limit; synthesizing initialization artifacts.")

        transcript_ref = self.store.write_text(
            INTERVIEW_TRANSCRIPT, _render_interview_markdown(transcript)
        )
        return self.initialize(
            initial_context=initial_context,
            conversation_transcript=transcript,
            extra_artifact_refs=[transcript_ref],
        )

    def next_interview_turn(self, transcript: list[dict[str, str]]) -> InitializationInterviewTurn:
        """Ask the LLM whether to request more information or finalize initialization."""
        mock_output = self._mock_interview_turn(transcript)
        messages = [
            {
                "role": "system",
                "content": render_prompt(
                    "initialization_interviewer", override_dir=self.prompt_dir
                ),
            },
            {
                "role": "user",
                "content": (
                    "Decide the next initialization-interview turn from this transcript.\n\n"
                    f"Transcript:\n{_format_transcript(transcript)}"
                ),
            },
        ]
        turn = self.router.complete_structured(
            task_type="initialization_interview",
            messages=messages,
            schema=InitializationInterviewTurn,
            mock_output=mock_output if self.router.dry_run else None,
            max_tokens=1200,
        )
        if not turn.assistant_message.strip():
            if self.router.dry_run:
                return mock_output
            raise StructuredLLMError("Initialization interviewer returned an empty assistant_message")
        return turn

    def initialize(
        self,
        *,
        initial_context: str = "",
        conversation_transcript: list[dict[str, str]] | None = None,
        extra_artifact_refs: list[ArtifactRef] | None = None,
    ) -> ResearchState:
        self.store.initialize_layout()
        transcript_text = _format_transcript(conversation_transcript or [])
        mock_output = self._mock_bundle(initial_context, transcript_text)
        messages = [
            {
                "role": "system",
                "content": render_prompt(
                    "initialization_synthesizer", override_dir=self.prompt_dir
                ),
            },
            {
                "role": "user",
                "content": (
                    "Synthesize the initialization artifacts from this LLM-guided interview.\n\n"
                    f"Initial context:\n{initial_context}\n\n"
                    f"Conversation transcript:\n{transcript_text}"
                ),
            },
        ]
        try:
            bundle = self.router.complete_structured(
                task_type="initialization",
                messages=messages,
                schema=InitializationBundle,
                # A generic task mock output is acceptable for dry-run demos, but it is
                # too destructive for real initialization: it can erase useful
                # interview content into a vague ResearchTask.md. In real runs, fail
                # loudly and leave the transcript for rerun/debugging.
                mock_output=mock_output if self.router.dry_run else None,
            )
        except StructuredLLMError as exc:
            raise RuntimeError(
                "Initialization synthesis failed, so no generic ResearchTask.md was committed. "
                "Inspect ModelCallLedger.jsonl for the validation/API error and rerun `tcs-research init` "
                "after fixing the model, prompt, or config. Use --dry-run only if you intentionally want "
                "a mock placeholder task."
            ) from exc
        return self.commit_bundle(bundle, extra_refs=extra_artifact_refs or [])

    def commit_bundle(
        self, bundle: InitializationBundle, *, extra_refs: list[ArtifactRef] | None = None
    ) -> ResearchState:
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
                statement=(
                    "Research task initialized; substantive scientific claims remain unverified."
                ),
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
        claims = self._sanitize_initial_claims(claims)
        self.store.append_claims(claims)
        extra_refs = extra_refs or []
        state = ResearchState(
            task_summary=_task_summary(bundle.research_task_markdown),
            active_claim_ids=[claim.claim_id for claim in claims],
            artifact_refs=[
                task_ref,
                nomenclature_ref,
                *extra_refs,
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
                reason=(
                    "Initialization created canonical task, nomenclature, "
                    "and initial ledger entries."
                ),
                artifact_refs=[task_ref, nomenclature_ref, *extra_refs, state_ref],
            )
        )
        return state

    def _sanitize_initial_claims(self, claims: list[ClaimRecord]) -> list[ClaimRecord]:
        """Initialization records task context; it must not certify science by itself."""
        for claim in claims:
            if claim.claim_type == ClaimType.literature:
                claim.status = ClaimStatus.needs_review
                if "needs_literature_ingestion" not in claim.tags:
                    claim.tags.append("needs_literature_ingestion")
            elif claim.claim_type in {
                ClaimType.mathematical,
                ClaimType.algorithmic,
                ClaimType.complexity,
                ClaimType.resource,
                ClaimType.novelty,
                ClaimType.experimental,
                ClaimType.theorem_statement,
            }:
                claim.status = ClaimStatus.proposed
            for evidence in claim.evidence:
                if evidence.evidence_type == EvidenceType.citation and not evidence.citation_keys:
                    evidence.confidence = min(evidence.confidence, 0.1)
        return claims

    def _mock_interview_turn(
        self, transcript: list[dict[str, str]]
    ) -> InitializationInterviewTurn:
        if not _has_user_answer(transcript):
            return self._first_question_turn(has_initial_context=_has_initial_context(transcript))
        return InitializationInterviewTurn(
            ready_to_initialize=True,
            assistant_message=(
                "Thanks — I have enough to create a conservative initial research task. "
                "Open details will be recorded explicitly for later clarification."
            ),
            missing_information=[
                "Any unspecified model details, assumptions, and literature context."
            ],
            relevant_information=["User supplied at least one substantive initialization answer."],
            rationale="Dry-run mock path finalizes after a minimal interactive exchange.",
        )

    def _first_question_turn(self, *, has_initial_context: bool) -> InitializationInterviewTurn:
        if has_initial_context:
            message = (
                "I read the supplied context. What should I treat as the main success criterion, "
                "and are there any assumptions, barriers, or tools that are especially important?"
            )
            missing = ["User priorities beyond the supplied context."]
        else:
            message = (
                "What theoretical computer science problem should the system work on? "
                "Please include the computational model and what would count as success if you can."
            )
            missing = ["Research problem, computational model, and success criterion."]
        return InitializationInterviewTurn(
            ready_to_initialize=False,
            assistant_message=message,
            missing_information=missing,
            relevant_information=[],
            rationale=(
                "The interview needs at least one substantive user answer before initialization."
            ),
        )

    def _mock_bundle(self, initial_context: str, transcript_text: str) -> InitializationBundle:
        user_knowledge = transcript_text.strip() or "No interview transcript supplied."
        task_md = f"""# Research Task

## Problem statement
{initial_context.strip() or "TBD: specify the exact theoretical computer science research problem."}

## Computational model and assumptions
TBD during follow-up clarification. Record oracle access, randomness, quantum resources,
promise structure, cryptographic assumptions, and asymptotic conventions here.

## Success criteria
- A main-task solution requires proof-quality mathematical evidence, explicit complexity derivations,
  and independent replication.
- Experimental observations may suggest conjectures but are not proofs.
- Unspecified user preferences from initialization should be treated as open questions,
  not assumptions.

## Fallback publishable outcomes
- A verified lower-bound or barrier explanation.
- A clarified formalization with useful Lean lemmas.
- A reproducible experimental counterexample search or benchmark.
- A literature synthesis resolving notation and novelty questions.

## Known barriers and literature context
TBD. Literature claims must be entered into `LiteratureDB` with provenance before use.

## Initialization interview transcript
{user_knowledge}

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
            initial_state_notes=[
                "Initialized from an LLM-guided user interview; unresolved details "
                "should be clarified."
            ],
            initial_claims=[initial_claim],
            fallback_publishable_outcomes=[
                "verified lower-bound or barrier explanation",
                "formalized lemma library",
                "reproducible experiment/counterexample search",
                "literature and notation synthesis",
            ],
            assumptions=["TBD"],
            success_criteria=["conservative solved check and independent replication"],
        )


def _format_transcript(transcript: list[dict[str, str]]) -> str:
    if not transcript:
        return "No conversation yet."
    blocks = []
    for message in transcript:
        role = message.get("role", "unknown").strip().title() or "Unknown"
        content = message.get("content", "").strip() or "[empty]"
        blocks.append(f"{role}:\n{content}")
    return "\n\n".join(blocks)


def _render_interview_markdown(transcript: list[dict[str, str]]) -> str:
    lines = ["# Initialization Interview", ""]
    for message in transcript:
        role = message.get("role", "unknown").strip().title() or "Unknown"
        content = message.get("content", "").strip() or "[empty]"
        lines.extend([f"## {role}", "", content, ""])
    return "\n".join(lines).rstrip() + "\n"


def _has_initial_context(transcript: list[dict[str, str]]) -> bool:
    return any(
        message.get("role") == "user"
        and message.get("content", "").startswith(_INITIAL_CONTEXT_PREFIX)
        for message in transcript
    )


def _has_user_answer(transcript: list[dict[str, str]]) -> bool:
    for message in transcript:
        if message.get("role") != "user":
            continue
        content = message.get("content", "").strip()
        if (
            not content
            or content.startswith(_INITIAL_CONTEXT_PREFIX)
            or content.startswith("[User requested")
        ):
            continue
        return True
    return False


def _task_summary(markdown: str, limit: int = 500) -> str:
    lines = [line.strip("# ").strip() for line in markdown.splitlines() if line.strip()]
    text = " ".join(lines)
    return text[:limit]
