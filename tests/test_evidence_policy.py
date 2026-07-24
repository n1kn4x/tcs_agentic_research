from __future__ import annotations

from tcs_agentic_research.artifact_store import ArtifactStore
from tcs_agentic_research.core.models import (
    ActionProposal,
    ActionRecord,
    EvidenceReceipt,
    EvidenceType,
    RecordDraft,
    RecordKind,
)
from tcs_agentic_research.core.policy import EvidencePolicy


def _action() -> ActionRecord:
    return ActionRecord(
        cycle=1,
        task_revision=1,
        subsystem="fixture",
        proposal=ActionProposal(
            subsystem="fixture", action_type="test", title="test evidence"
        ),
    )


def test_model_authored_theorem_can_never_self_promote() -> None:
    result = EvidencePolicy().admit(
        action=_action(),
        drafts=[
            RecordDraft(
                kind=RecordKind.formal_theorem,
                title="Model theorem",
                summary="The model says this is proven.",
                evidence=EvidenceReceipt(
                    evidence_type=EvidenceType.model,
                    details={"accepted": True, "confidence": 1.0},
                ),
            )
        ],
        existing_records=[],
    )

    assert result.records[0].status.value == "tentative"


def test_lean_receipt_requires_hashed_placeholder_free_artifact(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize_layout()
    store.write_text("proof.lean", "theorem valid : True := by trivial\n")
    ref = store.artifact_ref("proof.lean")
    draft = RecordDraft(
        kind=RecordKind.formal_theorem,
        title="Lean theorem",
        summary="True",
        evidence=EvidenceReceipt(
            evidence_type=EvidenceType.lean,
            details={
                "accepted": True,
                "placeholder_free": True,
                "statement": "True",
                "proof_artifact_paths": [ref.path],
            },
            artifact_refs=[ref],
        ),
    )

    result = EvidencePolicy(store).admit(
        action=_action(), drafts=[draft], existing_records=[]
    )

    assert result.records[0].status.value == "verified"


def test_same_lean_proposition_under_a_new_name_is_deduplicated(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize_layout()
    ref = store.write_text("proof.lean", "theorem first_name : True := by trivial\n")
    first = RecordDraft(
        kind=RecordKind.formal_theorem,
        title="Lean theorem first_name",
        summary="True",
        evidence=EvidenceReceipt(
            evidence_type=EvidenceType.lean,
            details={
                "accepted": True,
                "placeholder_free": True,
                "statement": "True",
                "proof_artifact_paths": [ref.path],
            },
            artifact_refs=[ref],
        ),
    )
    accepted = EvidencePolicy(store).admit(
        action=_action(), drafts=[first], existing_records=[]
    )
    renamed = first.model_copy(deep=True)
    renamed.title = "Lean theorem second_name"

    duplicate = EvidencePolicy(store).admit(
        action=_action(), drafts=[renamed], existing_records=accepted.records
    )

    assert duplicate.records == []
    assert "duplicates" in duplicate.rejected[0]


def test_source_quote_requires_exact_validated_span(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize_layout()
    store.write_text("paper.txt", "Theorem 1. Exact words.\n")
    ref = store.artifact_ref("paper.txt")
    valid = RecordDraft(
        kind=RecordKind.source_quote,
        title="Theorem 1",
        summary="Exact words.",
        evidence=EvidenceReceipt(
            evidence_type=EvidenceType.source_quote,
            details={
                "citation_key": "paper",
                "quote": "Theorem 1. Exact words.",
                "source_sha256": ref.sha256,
                "validated": True,
            },
            artifact_refs=[ref],
        ),
    )
    invalid = valid.model_copy(deep=True)
    invalid.title = "Unvalidated quote"
    invalid.evidence.details["validated"] = False

    accepted = EvidencePolicy(store).admit(
        action=_action(), drafts=[valid], existing_records=[]
    )
    rejected = EvidencePolicy(store).admit(
        action=_action(), drafts=[invalid], existing_records=[]
    )

    assert accepted.records[0].status.value == "observed"
    assert rejected.records == []
    assert "not span-validated" in rejected.rejected[0]


def test_execution_is_observed_only_after_exact_replication(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize_layout()
    source = store.write_text("experiment.py", "def run_experiment(mode): return {}\n")
    first_results = store.write_json("run1/results.json", {"value": 1})
    second_results = store.write_json("run2/results.json", {"value": 1})
    receipt = EvidenceReceipt(
        evidence_type=EvidenceType.execution,
        details={
            "success": True,
            "replicated": True,
            "program_sha256": source.sha256,
            "output_sha256": first_results.sha256,
        },
        artifact_refs=[source, first_results, second_results],
    )
    draft = RecordDraft(
        kind=RecordKind.experiment,
        title="Replicated run",
        summary="The exact output repeated.",
        evidence=receipt,
    )

    result = EvidencePolicy(store).admit(
        action=_action(), drafts=[draft], existing_records=[]
    )

    assert result.records[0].status.value == "observed"
