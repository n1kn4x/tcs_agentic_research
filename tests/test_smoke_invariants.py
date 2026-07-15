from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path

import httpx
import pytest

from tcs_agentic_research.agents.critics import (
    check_solved_deterministically,
    is_claim_acceptably_supported,
)
from tcs_agentic_research.agents.initialization import WorkspaceInitializer
from tcs_agentic_research.agents.literature import LiteratureResearcher
from tcs_agentic_research.agents.proposal import ProposalAgent
from tcs_agentic_research.agents.research import ResearchAgent
from tcs_agentic_research.agents.toolsets import artifact_retrieval_toolset, literature_toolset
from tcs_agentic_research.artifact_store import ArtifactStore
from tcs_agentic_research.obligations import (
    CommitManager,
    ObligationBoardManager,
    ObligationRunValidator,
)
from tcs_agentic_research.leap.harness import (
    BlueprintCandidate,
    DecompositionReview,
    FormalProofCandidate,
)
from tcs_agentic_research.experimenter.docker_project import _diagnostic
from tcs_agentic_research.experimenter.errors import ExperimenterConfigurationError
from tcs_agentic_research.literature.pdf_text import PDFTextExtractor
from tcs_agentic_research.llm import (
    LLMRouter,
    StructuredLLMError,
    SYSTEM_OWNED_SCHEMA_FIELDS,
    _llm_json_schema,
    _prepare_structured_messages,
    openai_tool_from_schema,
)
from tcs_agentic_research.prompt_loader import load_prompt
from tcs_agentic_research.prompt_serialization import compact_json_dumps
from tcs_agentic_research import schemas as schema_module
from tcs_agentic_research.schemas import (
    ArtifactRef,
    ClaimRecord,
    ClaimStatus,
    ClaimType,
    EvidenceRecord,
    EvidenceType,
    LiteratureExtract,
    CriticDecision,
    ModelProfile,
    ObligationRun,
    PaperMetadata,
    ProposalCritique,
    ProposalSubmission,
    ProposalKind,
    ReplicationResult,
    ReportOutcome,
    ResearchCritique,
    ResearchObligation,
    ResearchProposal,
    ResearchReport,
    ResearchReportSubmission,
    ResearchState,
    RouterSettings,
    StrictModel,
    ValidationResult,
    ValidationGateStatus,
)


PROMPT_SCHEMAS = {
    "independent_replication": ReplicationResult,
    "leap_blueprint": BlueprintCandidate,
    "leap_decomposition_reviewer": DecompositionReview,
    "leap_direct_prover": FormalProofCandidate,
    "leap_reviser": FormalProofCandidate,
    "literature_researcher": LiteratureExtract,
    "proposal_critic": ProposalCritique,
    "proposal_generator": ProposalSubmission,
    "research_agent": ResearchReportSubmission,
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
        path="InitialResearchTask.md",
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
        ProposalCritique,
    )

    content = rendered[0]["content"]
    assert "{{ResearchReport}}" not in content
    assert "Complete JSON Schema for `ResearchReport`." in content
    assert "Complete JSON Schema for `ProposalCritique`." not in content
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


def test_tool_schema_can_keep_tool_argument_ids() -> None:
    class CandidateImportArgs(StrictModel):
        candidate_id: str

    stripped = openai_tool_from_schema("import_candidate", "", CandidateImportArgs)
    preserved = openai_tool_from_schema(
        "import_candidate",
        "",
        CandidateImportArgs,
        strip_system_owned_fields=False,
    )

    assert "candidate_id" not in stripped["function"]["parameters"].get("properties", {})
    assert "candidate_id" in preserved["function"]["parameters"]["properties"]


def test_structured_tool_completion_uses_openai_tool_calls(tmp_path: Path) -> None:
    class QueryArgs(StrictModel):
        query: str

    store = _store(tmp_path)
    router = LLMRouter(
        RouterSettings(
            profiles={
                "deep": ModelProfile(
                    model="mock",
                    supports_tools=True,
                    task_types=["proposal_generation"],
                )
            }
        ),
        store=store,
    )
    submit_tool = openai_tool_from_schema("submit", "", ResearchProposal)
    query_tool = openai_tool_from_schema(
        "query_literature",
        "",
        QueryArgs,
        strip_system_owned_fields=False,
    )
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_query",
                                "type": "function",
                                "function": {
                                    "name": "query_literature",
                                    "arguments": json.dumps(
                                        {"query": "barrier", "rationale": "private scratchpad"}
                                    ),
                                },
                            }
                        ]
                    }
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        },
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_submit",
                                "type": "function",
                                "function": {
                                    "name": "submit",
                                    "arguments": json.dumps(
                                        {"title": "Tool proposal", "precise_goal": "Use tools."}
                                    ),
                                },
                            }
                        ]
                    }
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
        },
    ]

    def fake_post_chat_completion(
        profile: ModelProfile,
        messages: list[dict[str, object]],
        *,
        temperature: float,
        max_tokens: int,
        json_schema: dict[str, object] | None = None,
        tools: list[dict[str, object]] | None = None,
        tool_choice: object | None = None,
    ) -> dict[str, object]:
        assert tools
        assert json_schema is None
        assert tool_choice == "auto"
        if len(responses) == 1:
            assert any(message.get("role") == "tool" for message in messages)
        return responses.pop(0)

    router._post_chat_completion = fake_post_chat_completion  # type: ignore[method-assign]
    proposal, trace = router.complete_structured_with_tools(
        task_type="proposal_generation",
        messages=[{"role": "user", "content": "make a proposal"}],
        tools=[query_tool, submit_tool],
        tool_executors={
            "query_literature": lambda args: {"status": "ok", "answer": f"saw {args['query']}"}
        },
        schema=ResearchProposal,
        final_tool_name="submit",
    )

    assert proposal.title == "Tool proposal"
    assert trace["private_reasoning"] == "redacted_not_logged_or_replayed"
    assert trace["tool_calls"][0]["name"] == "query_literature"
    assert "rationale" not in trace["tool_calls"][0]["arguments"]
    assert store.read_jsonl(ArtifactStore.MODEL_LEDGER)[-1]["completion_tokens"] == 10


def test_proposal_generation_toolset_is_read_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    agent = ProposalAgent(store, _router(store))

    names = {tool["function"]["name"] for tool in agent._proposal_toolset().openai_tools()}

    assert names == {"read_artifact", "read_jsonl_records", "submit_research_proposal"}
    assert names.isdisjoint(
        {
            "search_papers",
            "import_url",
            "import_arxiv",
            "import_doi",
            "import_candidate",
            "extract_paper",
            "extract_imported_papers",
            "query_literature",
            "run_experiment",
            "attempt_lean_proof",
        }
    )


def test_failed_proposal_revisions_convert_to_barrier_analysis(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write_text(ArtifactStore.RESEARCH_TASK, "# Task\nNo hidden oracle shortcuts.")
    state = ResearchState()
    store.save_state(state)
    router = LLMRouter(RouterSettings(profiles={"deep": ModelProfile(model="mock")}), store=store)
    agent = ProposalAgent(store, router)

    def fake_generate_proposal(**kwargs: object) -> ResearchProposal:
        return ResearchProposal(
            title="Risky positive route",
            proposal_kind=ProposalKind.positive_algorithm_attempt,
            precise_goal="Assume a disputed shortcut and solve the task.",
            assertions_used_as_assumptions=["A disputed shortcut is available."],
        )

    def fake_review_proposal(context: object, proposal: ResearchProposal) -> ProposalCritique:
        return ProposalCritique(
            decision=CriticDecision.revise,
            summary="The shortcut is unsupported and must be analyzed, not assumed.",
            consistency_with_task="The route risks violating the task model.",
            plausibility="It may become useful as a barrier analysis.",
            required_revisions=["Move the shortcut claim to a hypothesis and analyze whether it is necessary."],
        )

    agent._generate_proposal = fake_generate_proposal  # type: ignore[method-assign]
    agent._review_proposal = fake_review_proposal  # type: ignore[method-assign]

    proposal, critique, proposal_path = agent.generate_and_review(state, max_revisions=0)

    assert proposal.proposal_kind == ProposalKind.barrier_analysis
    assert "shortcut" in " ".join(proposal.critic_constraints).lower()
    assert critique.decision == CriticDecision.accept
    assert store.exists(proposal_path)
    assert store.load_state().current_proposal_id == proposal.proposal_id  # type: ignore[union-attr]


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


def test_system_owned_fields_are_never_required_for_validation() -> None:
    for _name, model in inspect.getmembers(schema_module, inspect.isclass):
        if not issubclass(model, StrictModel):
            continue
        for field_name, field in model.model_fields.items():
            assert not (
                field_name in SYSTEM_OWNED_SCHEMA_FIELDS and field.is_required()
            ), f"{model.__name__}.{field_name} is stripped from LLM payloads but required"


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


def test_workspace_initializer_bootstraps_from_initial_research_task(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write_text(
        ArtifactStore.RESEARCH_TASK,
        "# Task\n\nStudy a bounded-width branching program lower bound.\n",
    )

    state = WorkspaceInitializer(store).ensure_initialized()

    assert state.task_summary.startswith("Task Study a bounded-width")
    assert store.exists(ArtifactStore.RESEARCH_STATE)
    assert any(ref.path == ArtifactStore.RESEARCH_TASK for ref in state.artifact_refs)
    assert any(ref.path == ArtifactStore.NOMENCLATURE for ref in state.artifact_refs)
    assert any(ref.path == ArtifactStore.OBLIGATION_BOARD for ref in state.artifact_refs)
    assert store.read_jsonl(ArtifactStore.PROPOSAL_LEDGER)[-1]["proposal_id"] == (
        "workspace_initialization"
    )


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


def test_experiment_agent_requires_configuration_when_used(tmp_path: Path) -> None:
    store = _store(tmp_path)
    router = _router(store)
    agent = ResearchAgent(store, router)

    with pytest.raises(ExperimenterConfigurationError, match="no `experimenter:` block"):
        agent.experiment.run_experiment(description="Run a deterministic smoke experiment.")


def test_docker_diagnostic_preserves_tail_for_long_output() -> None:
    completed = subprocess.CompletedProcess(
        ["docker", "build"],
        1,
        stdout="build started\n" + ("x" * 9000),
        stderr="step logs\n" + ("y" * 9000) + "\nFINAL BUILD ERROR",
    )

    diagnostic = _diagnostic(completed, limit=1000)

    assert "exit_code=1" in diagnostic
    assert "preserving final lines" in diagnostic
    assert "FINAL BUILD ERROR" in diagnostic


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


def test_artifact_manifest_lists_canonical_memory_without_contents(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write_text(ArtifactStore.RESEARCH_TASK, "# Task\nImportant details.\n")
    store.append_jsonl(ArtifactStore.PROPOSAL_LEDGER, {"proposal_id": "proposal_x"})

    manifest = store.artifact_manifest(max_items=20)
    by_path = {entry["path"]: entry for entry in manifest}

    assert ArtifactStore.RESEARCH_TASK in by_path
    assert ArtifactStore.PROPOSAL_LEDGER in by_path
    assert by_path[ArtifactStore.RESEARCH_TASK]["kind"] == "markdown"
    assert "Important details" not in json.dumps(manifest)


def test_artifact_retrieval_tools_read_text_and_jsonl(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write_text("Notes.md", "abcdef")
    store.append_jsonl("Events.jsonl", {"event_id": "a", "value": 1})
    store.append_jsonl("Events.jsonl", {"event_id": "b", "value": 2})
    tools = artifact_retrieval_toolset(store=store).executors()

    text_observation = tools["read_artifact"](
        {"path": "Notes.md", "offset": 2, "max_chars": 3}
    )
    jsonl_observation = tools["read_jsonl_records"](
        {
            "path": "Events.jsonl",
            "id_field": "event_id",
            "id_value": "b",
            "limit": 5,
            "max_chars": 1000,
        }
    )

    assert text_observation["status"] == "ok"
    assert text_observation["content"] == "cde"
    assert jsonl_observation["status"] == "ok"
    assert jsonl_observation["matching_record_count"] == 1
    assert '"event_id": "b"' in jsonl_observation["content"]
    assert '"event_id": "a"' not in jsonl_observation["content"]


def test_obligation_validator_blocks_placeholder_and_unknown_tool_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    obligation = ResearchObligation(
        statement="Derive the theorem.",
        required_evidence=[EvidenceType.informal_argument],
    )
    run = ObligationRun(
        obligation_id=obligation.obligation_id,
        outcome="fulfilled",
        summary="test",
        claims_generated=[
            ClaimRecord(
                claim_type=ClaimType.mathematical,
                statement="A concrete theorem follows from the derivation.",
            )
        ],
        evidence=[
            EvidenceRecord(
                evidence_type=EvidenceType.informal_argument,
                summary="claimed derivation",
                tool_result_ids=["missing_tool_result"],
            )
        ],
    )

    validation = ObligationRunValidator(store).validate(
        run=run,
        obligation=obligation,
        trace={"tool_calls": []},
    )

    assert not validation.ok
    assert any("placeholder" in issue for issue in validation.blocking_issues)
    assert any("unknown tool_result_id" in issue for issue in validation.blocking_issues)


def test_commit_manager_accepts_claim_from_validated_obligation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    manager = ObligationBoardManager(store)
    obligation = ResearchObligation(
        statement="Give the checked derivation.",
        required_evidence=[EvidenceType.informal_argument],
    )
    board = store.load_obligation_board()
    board.obligations.append(obligation)
    manager.save(board)
    run_ref = store.write_text("Reports/iterations/iteration_0001/manual_run.json", "{}\n")
    claim = ClaimRecord(
        claim_type=ClaimType.mathematical,
        statement="A checked derivation establishes the candidate theorem.",
    )
    run = ObligationRun(
        obligation_id=obligation.obligation_id,
        outcome="fulfilled",
        summary=(
            "This derivation is intentionally long enough for the deterministic evidence gate "
            "to treat it as a substantive informal derivation in this smoke test."
        ),
        claims_generated=[claim],
        evidence=[
            EvidenceRecord(
                evidence_type=EvidenceType.informal_argument,
                summary="Substantive derivation recorded in the run artifact.",
            )
        ],
        artifact_refs=[run_ref],
    )
    validation = ValidationResult(
        ok=True,
        gate_results=[
            ValidationGateStatus(gate="scope_provenance", passed=True),
            ValidationGateStatus(gate="evidence", passed=True),
            ValidationGateStatus(gate="consistency", passed=True),
        ],
    )

    result = CommitManager(store).apply_obligation_run(
        run=run,
        validation=validation,
        run_ref=run_ref,
    )

    assert result["outcome"] == "claims_accepted"
    latest = store.latest_claims_by_id()
    assert claim.claim_id in latest
    assert is_claim_acceptably_supported(latest[claim.claim_id], store)


def test_commit_manager_blocks_obligation_on_failed_validation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    obligation = ResearchObligation(statement="Prove it.")
    board = store.load_obligation_board()
    board.obligations.append(obligation)
    store.save_obligation_board(board)
    run_ref = store.write_text("Reports/iterations/iteration_0001/blocked_run.json", "{}\n")
    run = ObligationRun(
        obligation_id=obligation.obligation_id,
        outcome="blocked",
        summary="The attempt was blocked by missing evidence.",
    )
    validation = ValidationResult(
        ok=False,
        gate_results=[ValidationGateStatus(gate="evidence", passed=False, issues=["missing"])],
        blocking_issues=["missing"],
    )

    CommitManager(store).apply_obligation_run(run=run, validation=validation, run_ref=run_ref)

    board_after = store.load_obligation_board()
    blocked_obligation = next(
        item for item in board_after.obligations if item.obligation_id == obligation.obligation_id
    )
    assert blocked_obligation.status == "blocked"
    assert not store.latest_claims_by_id()


def test_pdf_text_extractor_falls_back_to_ocr_when_embedded_text_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    pdf_ref = store.write_bytes("LiteratureDB/papers/Scanned/paper.pdf", b"%PDF-1.4\n")
    extractor = PDFTextExtractor(store)

    monkeypatch.setattr(extractor, "_extract_with_pypdf", lambda *args, **kwargs: "")
    monkeypatch.setattr(extractor, "_extract_with_pdftotext", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        extractor,
        "_extract_with_ocr",
        lambda *args, **kwargs: "\n\n--- page 1 ---\n\nOCR recovered text.\n",
    )

    text_path = extractor.extract_pdf_text(pdf_ref.path)

    assert text_path == "LiteratureDB/papers/Scanned/paper.txt"
    assert "OCR recovered text" in store.read_text(text_path)


def test_literature_extraction_indexes_support_ids_without_auto_claims(tmp_path: Path) -> None:
    store = _store(tmp_path)
    agent = LiteratureResearcher(store, _router(store))
    text = (
        "--- page 1 ---\n\n"
        "Lemma 1. The BKW algorithm for LPN has time complexity 2^O(n/log n).\n\n"
        "Proof. This is quoted only for a smoke test.\n"
    )
    text_ref = store.write_text("LiteratureDB/papers/BKWTest/paper.txt", text)
    paper = PaperMetadata(
        citation_key="BKWTest",
        title="BKW Test Paper",
        source_type="manual",
        text_path=text_ref.path,
    )
    agent.import_paper(paper)

    extract = agent.extract_paper(citation_key="BKWTest")
    assert extract.theorem_statements
    assert not store.read_claims()  # Literature extraction creates evidence objects, not claims.

    answer = agent.answer_query("BKW LPN time complexity", limit=3)
    assert answer.results
    top = answer.results[0]
    assert top.support_id
    assert top.statement_id
    assert top.provenance[0].validated

    supported = ClaimRecord(
        claim_type=ClaimType.literature,
        statement="BKWTest states a BKW LPN time bound.",
        status=ClaimStatus.cited,
        evidence=[
            EvidenceRecord(
                evidence_type=EvidenceType.citation,
                summary="Statement-level support from LiteratureDB.",
                citation_keys=["BKWTest"],
                literature_support_ids=[top.support_id],
            )
        ],
    )
    assert is_claim_acceptably_supported(supported, store)

    citation_key_only = supported.model_copy(deep=True)
    citation_key_only.evidence[0].literature_support_ids = []
    assert not is_claim_acceptably_supported(citation_key_only, store)


def test_literature_toolset_can_import_and_extract_for_literature_obligations(tmp_path: Path) -> None:
    store = _store(tmp_path)
    agent = LiteratureResearcher(store, _router(store))
    tools = literature_toolset(
        store=store,
        literature=agent,
        include_discovery_tools=True,
        include_extraction_tools=True,
        auto_extract_after_import=True,
    )
    names = {tool["function"]["name"] for tool in tools.openai_tools()}

    assert "search_papers" in names
    assert "import_arxiv" in names
    assert "import_doi" in names
    assert "import_candidate" in names
    assert "extract_paper" in names
    assert "extract_imported_papers" in names


def test_extract_imported_papers_processes_unextracted_full_text_without_llm(tmp_path: Path) -> None:
    class FailingExtractionRouter:
        dry_run = False

        def complete_structured(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("deterministic batch extraction must not call the LLM")

    store = _store(tmp_path)
    agent = LiteratureResearcher(store, FailingExtractionRouter())
    text = (
        "--- page 1 ---\n\n"
        "Theorem 2. Orthogonal Vectors has the stated conditional lower-bound role in this "
        "smoke-test literature record.\n\n"
        "Proof. This is a smoke test.\n"
    )
    text_ref = store.write_text("LiteratureDB/papers/OVSmoke/paper.txt", text)
    agent.import_paper(
        PaperMetadata(
            citation_key="OVSmoke",
            title="OV Smoke Test Paper",
            source_type="manual",
            text_path=text_ref.path,
        )
    )

    assert not store.read_jsonl("LiteratureDB/extracted_claims.jsonl")
    summary = agent.extract_imported_papers(max_papers=5, only_missing=True)

    assert summary["processed_count"] == 1
    assert summary["citation_keys"] == ["OVSmoke"]
    assert summary["support_ids"]
    assert store.read_jsonl("LiteratureDB/extracted_claims.jsonl")

    again = agent.extract_imported_papers(max_papers=5, only_missing=True)
    assert again["processed_count"] == 0
    assert any(item["reason"] == "already extracted" for item in again["skipped"])


def test_literature_registry_deduplicates_papers_by_arxiv_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    agent = LiteratureResearcher(store, _router(store))
    first = PaperMetadata(
        citation_key="GKZ2017",
        title="Learning with Errors is easy with quantum samples",
        source_type="manual",
        arxiv_id="1702.08255",
    )
    second = PaperMetadata(
        citation_key="GKZ17",
        title="Learning with Errors is easy with quantum samples",
        source_type="manual",
        arxiv_id="1702.08255",
    )

    imported_first = agent.import_paper(first)
    imported_second = agent.import_paper(second)

    assert imported_second.paper_id == imported_first.paper_id
    assert len(store.read_jsonl("LiteratureDB/papers.jsonl")) == 1
    assert agent.index.find_paper(citation_key="GKZ17").paper_id == imported_first.paper_id
