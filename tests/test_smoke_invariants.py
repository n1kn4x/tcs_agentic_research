from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from tcs_agentic_research.agents.critics import (
    check_solved_deterministically,
    is_claim_acceptably_supported,
)
from tcs_agentic_research.agents.initialization import InitializationAgent
from tcs_agentic_research.agents.research import ResearchAgent
from tcs_agentic_research.artifact_store import ArtifactStore
from tcs_agentic_research.leap.harness import (
    BlueprintCandidate,
    DecompositionReview,
    FormalProofCandidate,
)
from tcs_agentic_research.llm import (
    LLMRouter,
    StructuredLLMError,
    _llm_json_schema,
    _prepare_structured_messages,
)
from tcs_agentic_research.prompt_loader import load_prompt
from tcs_agentic_research.prompt_serialization import compact_json_dumps
from tcs_agentic_research.schemas import (
    ArtifactRef,
    ClaimRecord,
    ClaimStatus,
    ClaimType,
    EvidenceRecord,
    EvidenceType,
    ExperimentPlan,
    InitializationBundle,
    InitializationInterviewTurn,
    LiteratureExtract,
    LiteratureSource,
    ModelProfile,
    PaperMetadata,
    ProofObligation,
    ProposalCritique,
    ProposalLoopAction,
    ReplicationResult,
    ReportOutcome,
    ResearchCritique,
    ResearchProposal,
    ResearchReport,
    ResearchState,
    RouterSettings,
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
    "experiment_planner": ExperimentPlan,
    "proposal_critic": ProposalCritique,
    "proposal_generator": ProposalLoopAction,
    "research_agent": ResearchReport,
    "research_critic": ResearchCritique,
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


def test_chat_completion_http_errors_include_vllm_response_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def post(self, url: str, **kwargs: object) -> httpx.Response:
            request = httpx.Request("POST", url)
            return httpx.Response(
                400,
                request=request,
                json={
                    "error": {
                        "message": "This model's maximum context length is 4096 tokens.",
                        "type": "BadRequestError",
                        "code": 400,
                    }
                },
            )

    monkeypatch.setattr("tcs_agentic_research.llm.httpx.Client", FakeClient)
    router = LLMRouter(RouterSettings(profiles={"deep": ModelProfile(model="mock")}))

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        router._post_chat_completion(
            ModelProfile(model="mock"),
            [{"role": "user", "content": "too much context"}],
            temperature=0.0,
            max_tokens=4096,
        )

    message = str(exc_info.value)
    assert "vLLM/OpenAI-compatible server response body" in message
    assert "maximum context length" in message
    assert '"type": "BadRequestError"' in message


def test_repo_structured_prompts_intentionally_keep_schema_placeholders() -> None:
    for prompt_name, schema in PROMPT_SCHEMAS.items():
        text = load_prompt(prompt_name)
        assert "{{" + schema.__name__ + "}}" in text
        assert "Use the complete JSON schema inserted below" in text


def test_prompt_compaction_is_self_contained_and_lossless_for_duplicate_refs() -> None:
    ref = ArtifactRef(
        path="ResearchTask.md",
        kind="markdown",
        sha256="abc123",
        summary="Core task artifact.",
        created_at="2026-01-01T00:00:00+00:00",
    )
    expected = {
        "first": ref.model_dump(mode="json"),
        "second": ref.model_dump(mode="json"),
    }

    compacted = json.loads(compact_json_dumps(expected))

    assert "$defs" in compacted
    assert compacted["payload"]["first"] == compacted["payload"]["second"]
    assert _expand_prompt_refs(compacted) == expected


def _expand_prompt_refs(compacted: dict[str, object]) -> object:
    defs = compacted.get("$defs", {})

    def expand(node: object) -> object:
        if isinstance(node, dict) and set(node) == {"$ref"}:
            return defs[str(node["$ref"])]
        if isinstance(node, dict):
            return {key: expand(value) for key, value in node.items()}
        if isinstance(node, list):
            return [expand(value) for value in node]
        return node

    return expand(compacted["payload"])


def test_schema_placeholders_are_resolved_by_name_not_by_output_schema() -> None:
    rendered = _prepare_structured_messages(
        [{"role": "system", "content": "Use {{ResearchReport}}."}],
        ExperimentPlan,
    )

    content = rendered[0]["content"]
    assert "{{ResearchReport}}" not in content
    assert "Complete JSON Schema for `ResearchReport`." in content
    assert "Complete JSON Schema for `ExperimentPlan`." not in content
    assert len(rendered) == 1


def test_structured_messages_without_placeholders_are_not_appended() -> None:
    messages = [{"role": "system", "content": "Return JSON."}]

    assert _prepare_structured_messages(messages, ResearchReport) == messages


def test_unknown_schema_placeholders_raise() -> None:
    with pytest.raises(StructuredLLMError, match="DoesNotExist"):
        _prepare_structured_messages(
            [{"role": "system", "content": "Use {{DoesNotExist}}."}],
            ResearchReport,
        )


def test_llm_schema_omits_system_owned_fields() -> None:
    schema = _llm_json_schema(ResearchReport)
    property_names = _schema_property_names(schema)

    for field_name in [
        "proposal_id",
        "report_id",
        "claim_id",
        "evidence_id",
        "obligation_id",
        "artifact_refs",
        "created_at",
        "updated_at",
        "related_proposal_ids",
        "related_report_ids",
    ]:
        assert field_name not in property_names

    report = ResearchReport.model_validate(
        {"outcome": "partially_succeeded", "executive_summary": "draft"}
    )
    assert report.proposal_id == ""
    assert report.report_id.startswith("report_")


def _schema_property_names(node: object) -> set[str]:
    names: set[str] = set()
    if isinstance(node, dict):
        properties = node.get("properties")
        if isinstance(properties, dict):
            names.update(str(key) for key in properties)
        for value in node.values():
            names.update(_schema_property_names(value))
    elif isinstance(node, list):
        for item in node:
            names.update(_schema_property_names(item))
    return names


def test_initialization_imports_declared_literature_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    imported: list[LiteratureSource] = []

    def fake_import_source(self, source: LiteratureSource) -> PaperMetadata:  # noqa: ANN001
        imported.append(source)
        ref = store.write_text("LiteratureDB/papers/example/paper.txt", "paper text")
        paper = PaperMetadata(
            citation_key="example",
            title="Example",
            source_type="url",
            text_path=ref.path,
            artifact_refs=[ref],
        )
        store.append_jsonl("LiteratureDB/papers.jsonl", paper)
        return paper

    monkeypatch.setattr(
        "tcs_agentic_research.agents.literature.LiteratureResearcher.import_source",
        fake_import_source,
    )
    bundle = InitializationBundle(
        research_task_markdown="# Task",
        literature_sources=[LiteratureSource(source="https://example.org/paper.pdf")],
    )

    state = InitializationAgent(store, _router(store)).commit_bundle(bundle)

    assert imported and imported[0].source == "https://example.org/paper.pdf"
    assert any(ref.path == "LiteratureDB/papers.jsonl" for ref in state.artifact_refs)


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


def test_research_loop_runs_experiment_obligation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    router = _router(store)
    agent = ResearchAgent(store, router)
    claim = ClaimRecord(
        claim_type=ClaimType.experimental,
        statement="A small executable experiment can run for this obligation.",
        status=ClaimStatus.conjecture,
    )
    obligation = ProofObligation(
        statement="Run a deterministic smoke experiment.",
        claim_ids=[claim.claim_id],
        suggested_tool="experiment",
    )
    report = ResearchReport(
        proposal_id="proposal_exp",
        outcome=ReportOutcome.partially_succeeded,
        executive_summary="Draft report with an experiment obligation.",
        claims_generated=[claim],
        proof_obligations=[obligation],
    )
    proposal = ResearchProposal(title="experiment", precise_goal="run experiment")

    observations = agent._run_subsystem_loop(report, proposal, "context", max_rounds=1)

    assert observations
    assert report.experimental_results
    assert list((store.root / "ExperimentRuns").glob("*"))
    assert report.proof_obligations[0].status == "experimentally_supported"
    assert report.proof_obligations[0].status != "proved"
    assert report.experimental_results[0].artifact_refs


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
    checked = check_solved_deterministically(store, state, report)

    assert not checked.confirmed_solved
    assert checked.possible_breakthrough
    assert checked.next_action == "independent_replication"
