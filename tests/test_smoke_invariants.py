from __future__ import annotations

from pathlib import Path

from tcs_agentic_research.agents.critics import (
    SolvedCheckAgent,
    is_claim_acceptably_supported,
)
from tcs_agentic_research.artifact_store import ArtifactStore
from tcs_agentic_research.leap.harness import (
    BlueprintCandidate,
    DecompositionReview,
    FormalProofCandidate,
)
from tcs_agentic_research.llm import LLMRouter
from tcs_agentic_research.prompt_loader import load_prompt
from tcs_agentic_research.schemas import (
    ArtifactRef,
    ClaimRecord,
    ClaimStatus,
    ClaimType,
    EvidenceRecord,
    EvidenceType,
    InitializationBundle,
    InitializationInterviewTurn,
    LiteratureExtract,
    ModelProfile,
    ProposalCritique,
    ReplicationResult,
    ReportOutcome,
    ResearchCritique,
    ResearchProposal,
    ResearchReport,
    ResearchState,
    RouterSettings,
    SolvedOutcome,
    SolvedVerdict,
)


PROMPT_SCHEMAS = {
    "independent_replication": ReplicationResult,
    "initialization_interviewer": InitializationInterviewTurn,
    "initialization_synthesizer": InitializationBundle,
    "leap_blueprint": BlueprintCandidate,
    "leap_decomposition_reviewer": DecompositionReview,
    "leap_direct_prover": FormalProofCandidate,
    "leap_reviser": FormalProofCandidate,
    "literature_researcher": LiteratureExtract,
    "proposal_critic": ProposalCritique,
    "proposal_generator": ResearchProposal,
    "research_agent": ResearchReport,
    "research_critic": ResearchCritique,
    "solved_checker": SolvedVerdict,
}


def _store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.initialize_layout()
    return store


def _router(store: ArtifactStore) -> LLMRouter:
    return LLMRouter(
        RouterSettings(
            profiles={
                "deep": ModelProfile(
                    model="mock",
                    task_types=[],
                )
            }
        ),
        store=store,
        dry_run=True,
    )


def test_structured_prompts_have_exact_schema_placeholders() -> None:
    for prompt_name, schema in PROMPT_SCHEMAS.items():
        assert "{{" + schema.__name__ + "}}" in load_prompt(prompt_name)


def test_latest_claim_replay_uses_newest_record(tmp_path: Path) -> None:
    store = _store(tmp_path)
    claim = ClaimRecord(
        claim_id="claim_same",
        claim_type=ClaimType.mathematical,
        statement="A",
        status=ClaimStatus.conjecture,
    )
    store.append_claims([claim])
    updated = claim.model_copy(deep=True)
    updated.status = ClaimStatus.proved_by_lean
    updated.evidence.append(
        EvidenceRecord(
            evidence_type=EvidenceType.lean_proof,
            summary="verified",
            artifact_refs=[
                store.write_text(
                    "LeanProject/TCSResearch/Test.lean", "theorem t : True := by trivial\n"
                )
            ],
            confidence=1.0,
        )
    )
    store.append_claims([updated])

    latest = store.latest_claims_by_id()

    assert latest["claim_same"].status == ClaimStatus.proved_by_lean
    assert is_claim_acceptably_supported(latest["claim_same"], store)


def test_url_only_literature_claim_is_not_accepted(tmp_path: Path) -> None:
    store = _store(tmp_path)
    claim = ClaimRecord(
        claim_type=ClaimType.literature,
        statement="Some paper proves the needed theorem.",
        status=ClaimStatus.cited,
        evidence=[
            EvidenceRecord(
                evidence_type=EvidenceType.citation,
                summary="URL-only citation",
                artifact_refs=[ArtifactRef(path="https://example.invalid/paper")],
                citation_keys=[],
            )
        ],
    )

    assert not is_claim_acceptably_supported(claim, store)


def test_solved_requires_independent_replication(tmp_path: Path) -> None:
    store = _store(tmp_path)
    proof_ref = store.write_text(
        "LeanProject/TCSResearch/Test.lean", "theorem t : True := by trivial\n"
    )
    claim = ClaimRecord(
        claim_type=ClaimType.mathematical,
        statement="True",
        status=ClaimStatus.proved_by_lean,
        evidence=[
            EvidenceRecord(
                evidence_type=EvidenceType.lean_proof,
                summary="verified",
                artifact_refs=[proof_ref],
                confidence=1.0,
            )
        ],
    )
    report = ResearchReport(
        proposal_id="proposal_x",
        outcome=ReportOutcome.succeeded,
        executive_summary="Solved.",
        claims_generated=[claim],
    )
    state = ResearchState(confirmed_by_replication=False)
    optimistic = SolvedVerdict(
        outcomes=[SolvedOutcome.solves_main_task],
        possible_breakthrough=False,
        confirmed_solved=True,
        rationale="optimistic",
    )

    checked = SolvedCheckAgent(store, _router(store))._enforce_conservatism(
        optimistic, state, report
    )

    assert not checked.confirmed_solved
    assert checked.possible_breakthrough
    assert checked.next_action == "independent_replication"
