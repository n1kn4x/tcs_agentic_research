"""Deterministic epistemic admission.

Language-model confidence, critic scores, and prose review never raise a record above tentative.
Only receipts produced by external mechanisms can do so.
"""

from __future__ import annotations

from collections.abc import Iterable
import re

from ..artifact_store import ArtifactStore
from ..leap.sorry import find_placeholder_lines
from .models import (
    ActionRecord,
    AdmissionResult,
    EvidenceReceipt,
    EvidenceType,
    RecordDraft,
    RecordStatus,
    ResearchRecord,
    record_fingerprint,
)


class EvidencePolicy:
    def __init__(self, store: ArtifactStore | None = None):
        self.store = store

    def admit(
        self,
        *,
        action: ActionRecord,
        drafts: Iterable[RecordDraft],
        existing_records: Iterable[ResearchRecord],
    ) -> AdmissionResult:
        existing = list(existing_records)
        known_ids = {record.record_id for record in existing}
        known_fingerprints = {record.fingerprint for record in existing}
        admitted: list[ResearchRecord] = []
        rejected: list[str] = []

        for index, draft in enumerate(drafts, 1):
            unknown_parents = set(draft.parent_ids) - known_ids
            if unknown_parents:
                rejected.append(
                    f"draft {index} references unknown parent records: {sorted(unknown_parents)}"
                )
                continue
            fingerprint = record_fingerprint(draft)
            if fingerprint in known_fingerprints:
                rejected.append(f"draft {index} duplicates an existing research record")
                continue
            status, defect = _status_for_receipt(draft.evidence, store=self.store)
            if defect:
                rejected.append(f"draft {index} has invalid evidence receipt: {defect}")
                continue
            record = ResearchRecord(
                task_revision=action.task_revision,
                producer=action.subsystem,
                action_id=action.action_id,
                kind=draft.kind,
                status=status,
                title=draft.title,
                summary=draft.summary,
                body=draft.body,
                relation=draft.relation,
                parent_ids=draft.parent_ids,
                evidence_type=draft.evidence.evidence_type,
                evidence_details=draft.evidence.details,
                artifact_refs=draft.evidence.artifact_refs,
                fingerprint=fingerprint,
            )
            admitted.append(record)
            known_ids.add(record.record_id)
            known_fingerprints.add(fingerprint)
        return AdmissionResult(records=admitted, rejected=rejected)


def _status_for_receipt(
    receipt: EvidenceReceipt, *, store: ArtifactStore | None
) -> tuple[RecordStatus, str]:
    details = receipt.details
    refs = receipt.artifact_refs
    if receipt.evidence_type in {EvidenceType.model, EvidenceType.system}:
        return RecordStatus.tentative, ""
    if store is None:
        return RecordStatus.tentative, "external evidence admission requires an artifact store"
    if receipt.evidence_type == EvidenceType.source_metadata:
        if not str(details.get("citation_key") or "").strip():
            return RecordStatus.tentative, "source metadata has no citation key"
        if not str(details.get("title") or "").strip():
            return RecordStatus.tentative, "source metadata has no title"
        if not any(
            ref.sha256
            and ref.path.endswith("metadata.json")
            and _hash_is_current(store, ref.path, ref.sha256)
            for ref in refs
        ):
            return RecordStatus.tentative, "source metadata has no current hashed metadata artifact"
        return RecordStatus.observed, ""
    if receipt.evidence_type == EvidenceType.source_quote:
        required = ("citation_key", "quote", "source_sha256")
        missing = [name for name in required if not str(details.get(name) or "").strip()]
        if missing:
            return RecordStatus.tentative, f"source quote is missing {missing}"
        if details.get("validated") is not True:
            return RecordStatus.tentative, "source quote was not span-validated"
        source_sha = str(details.get("source_sha256"))
        matching = [
            ref
            for ref in refs
            if ref.sha256 == source_sha
            and _hash_is_current(store, ref.path, source_sha)
        ]
        if not matching:
            return RecordStatus.tentative, "source quote hash does not match a source artifact"
        quote = str(details.get("quote"))
        if not any(quote in store.read_text(ref.path) for ref in matching):
            return RecordStatus.tentative, "exact quote is absent from the hashed source artifact"
        return RecordStatus.observed, ""
    if receipt.evidence_type == EvidenceType.lean:
        if details.get("accepted") is not True:
            return RecordStatus.tentative, "Lean did not accept the theorem"
        if details.get("placeholder_free") is not True:
            return RecordStatus.tentative, "Lean artifact was not checked for placeholders"
        if not str(details.get("statement") or "").strip():
            return RecordStatus.tentative, "Lean receipt has no proposition"
        declared_paths = set(details.get("proof_artifact_paths") or [])
        if not declared_paths:
            return RecordStatus.tentative, "Lean receipt does not identify its final proof artifact"
        lean_refs = [
            ref
            for ref in refs
            if ref.path in declared_paths
            and ref.path.endswith(".lean")
            and ref.sha256
            and _hash_is_current(store, ref.path, ref.sha256)
        ]
        if not lean_refs:
            return RecordStatus.tentative, "Lean receipt has no hashed source artifact"
        axiom = re.compile(r"(?m)^\s*axiom\b")
        if any(
            find_placeholder_lines(store.read_text(ref.path))
            or axiom.search(store.read_text(ref.path))
            for ref in lean_refs
        ):
            return RecordStatus.tentative, "Lean artifact contains a placeholder or added axiom"
        return RecordStatus.verified, ""
    if receipt.evidence_type == EvidenceType.execution:
        if details.get("success") is not True:
            return RecordStatus.tentative, "execution did not succeed"
        if details.get("replicated") is not True:
            return RecordStatus.tentative, "execution was not exactly replicated"
        if not str(details.get("program_sha256") or "").strip():
            return RecordStatus.tentative, "execution receipt has no program hash"
        if not str(details.get("output_sha256") or "").strip():
            return RecordStatus.tentative, "execution receipt has no output hash"
        program_sha = str(details.get("program_sha256"))
        programs = [
            ref
            for ref in refs
            if ref.path.endswith(".py")
            and ref.sha256 == program_sha
            and _hash_is_current(store, ref.path, program_sha)
        ]
        if not programs:
            return RecordStatus.tentative, "program hash does not match a Python artifact"
        result_refs = [
            ref
            for ref in refs
            if ref.path.endswith("/results.json")
            and ref.sha256
            and _hash_is_current(store, ref.path, ref.sha256)
        ]
        if len(result_refs) < 2:
            return RecordStatus.tentative, "execution receipt has fewer than two result artifacts"
        result_hashes = {ref.sha256 for ref in result_refs}
        if result_hashes != {str(details.get("output_sha256"))}:
            return RecordStatus.tentative, "replication result artifact hashes differ"
        return RecordStatus.observed, ""
    return RecordStatus.tentative, f"unsupported evidence type {receipt.evidence_type}"


def _hash_is_current(store: ArtifactStore, path: str, expected: str) -> bool:
    return store.exists(path) and store.artifact_ref(path).sha256 == expected
