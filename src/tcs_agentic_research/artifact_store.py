"""Filesystem-backed canonical state for the research workflow."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import yaml
from pydantic import BaseModel

from .schemas import (
    ArtifactKind,
    ArtifactRef,
    ClaimRecord,
    ProposalLedgerEntry,
    ResearchState,
    utc_now,
)


class ArtifactStore:
    """Durable artifact store rooted at a research workspace.

    The LangGraph state intentionally stays compact. Mathematical state is kept in files
    under this store and referenced by relative paths and hashes.
    """

    RESEARCH_TASK = "ResearchTask.md"
    NOMENCLATURE = "Nomenclature.yml"
    RESEARCH_STATE = "ResearchState.json"
    CLAIM_LEDGER = "ClaimLedger.jsonl"
    PROPOSAL_LEDGER = "ProposalLedger.jsonl"
    MODEL_LEDGER = "ModelCallLedger.jsonl"

    CORE_DIRECTORIES = (
        "LiteratureDB",
        "LiteratureDB/papers",
        "LeanProject",
        "LeanProject/ProofDAGs",
        "ExperimentRuns",
        "Reports",
        "Reports/critic_summaries",
        "Reports/iterations",
    )
    CORE_JSONL = (CLAIM_LEDGER, PROPOSAL_LEDGER, MODEL_LEDGER)
    LITERATURE_JSONL = (
        "LiteratureDB/papers.jsonl",
        "LiteratureDB/extracted_claims.jsonl",
        "LiteratureDB/notation_mappings.jsonl",
        "LiteratureDB/query_answers.jsonl",
        "LiteratureDB/candidates.jsonl",
    )

    def __init__(self, workspace: str | Path):
        self.root = Path(workspace).expanduser().resolve()

    def initialize_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in self.CORE_DIRECTORIES:
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        for jsonl in (*self.CORE_JSONL, *self.LITERATURE_JSONL):
            path = self.root / jsonl
            if not path.exists():
                path.write_text("", encoding="utf-8")
        nomenclature = self.root / self.NOMENCLATURE
        if not nomenclature.exists():
            self.write_yaml(
                self.NOMENCLATURE,
                {
                    "version": 1,
                    "updated_at": utc_now(),
                    "symbols": [],
                    "conventions": [],
                    "notes": ["Populate during initialization and literature ingestion."],
                },
            )

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"artifact path escapes workspace: {path}") from exc
        return resolved

    def relpath(self, path: str | Path) -> str:
        return str(Path(path).resolve().relative_to(self.root))

    def exists(self, path: str | Path) -> bool:
        return self.resolve(path).exists()

    def read_text(self, path: str | Path) -> str:
        return self.resolve(path).read_text(encoding="utf-8")

    def read_bytes(self, path: str | Path) -> bytes:
        return self.resolve(path).read_bytes()

    def write_text(self, path: str | Path, content: str) -> ArtifactRef:
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, target)
        return self.artifact_ref(path, summary=f"Wrote text artifact {path}")

    def write_bytes(self, path: str | Path, content: bytes) -> ArtifactRef:
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(content)
        os.replace(tmp, target)
        return self.artifact_ref(path, summary=f"Wrote binary artifact {path}")

    def read_json(self, path: str | Path) -> Any:
        return json.loads(self.read_text(path))

    def write_json(self, path: str | Path, payload: Any, *, indent: int = 2) -> ArtifactRef:
        plain = to_plain(payload)
        return self.write_text(path, json.dumps(plain, indent=indent, sort_keys=True) + "\n")

    def write_yaml(self, path: str | Path, payload: Any) -> ArtifactRef:
        plain = to_plain(payload)
        return self.write_text(path, yaml.safe_dump(plain, sort_keys=False, allow_unicode=True))

    def append_jsonl(self, path: str | Path, payload: Any) -> ArtifactRef:
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(to_plain(payload), sort_keys=True) + "\n")
        return self.artifact_ref(path, summary=f"Appended JSONL event to {path}")

    def read_jsonl(self, path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
        target = self.resolve(path)
        if not target.exists():
            return []
        lines = target.read_text(encoding="utf-8").splitlines()
        if limit is not None:
            lines = lines[-limit:]
        records = []
        for line in lines:
            if line.strip():
                records.append(json.loads(line))
        return records

    def append_claims(self, claims: Iterable[ClaimRecord]) -> None:
        for claim in claims:
            self.append_jsonl(self.CLAIM_LEDGER, claim)

    def read_claims(self) -> list[ClaimRecord]:
        """Replay the claim ledger as typed records, skipping malformed legacy rows."""
        claims: list[ClaimRecord] = []
        for record in self.read_jsonl(self.CLAIM_LEDGER):
            try:
                claims.append(ClaimRecord.model_validate(record))
            except Exception:
                # Workspaces are long-lived and may contain records from older schemas. Keep the
                # reducer robust; the original JSONL row remains available for audit/debugging.
                continue
        return claims

    def latest_claims_by_id(self) -> dict[str, ClaimRecord]:
        """Return the latest appended version of each claim id.

        The claim ledger is append-only: updates are represented by appending a newer full
        ``ClaimRecord`` with the same ``claim_id``. This reducer is the canonical machine view.
        """
        latest: dict[str, ClaimRecord] = {}
        for claim in self.read_claims():
            latest[claim.claim_id] = claim
        return latest

    def append_proposal_event(self, entry: ProposalLedgerEntry) -> None:
        self.append_jsonl(self.PROPOSAL_LEDGER, entry)

    def append_model_call(self, payload: Any) -> None:
        self.append_jsonl(self.MODEL_LEDGER, payload)

    def load_state(self) -> ResearchState | None:
        if not self.exists(self.RESEARCH_STATE):
            return None
        return ResearchState.model_validate(self.read_json(self.RESEARCH_STATE))

    def save_state(self, state: ResearchState) -> ArtifactRef:
        state.updated_at = utc_now()
        return self.write_json(self.RESEARCH_STATE, state)

    def create_iteration_dir(self, iteration: int) -> str:
        rel = f"Reports/iterations/iteration_{iteration:04d}"
        self.resolve(rel).mkdir(parents=True, exist_ok=True)
        return rel

    def artifact_ref(
        self,
        path: str | Path,
        *,
        kind: ArtifactKind | None = None,
        summary: str = "",
    ) -> ArtifactRef:
        target = self.resolve(path)
        if kind is None:
            kind = infer_kind(target)
        sha = sha256_file(target) if target.is_file() else None
        return ArtifactRef(path=self.relpath(target), kind=kind, sha256=sha, summary=summary)

    def snapshot_core_refs(self) -> list[ArtifactRef]:
        refs: list[ArtifactRef] = []
        for path in [
            self.RESEARCH_TASK,
            self.NOMENCLATURE,
            self.RESEARCH_STATE,
            self.CLAIM_LEDGER,
            self.PROPOSAL_LEDGER,
        ]:
            if self.exists(path):
                refs.append(self.artifact_ref(path))
        return refs


def to_plain(payload: Any) -> Any:
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    if isinstance(payload, dict):
        return {str(k): to_plain(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [to_plain(v) for v in payload]
    if isinstance(payload, tuple):
        return [to_plain(v) for v in payload]
    return payload


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_kind(path: Path) -> ArtifactKind:
    if path.is_dir():
        return ArtifactKind.directory
    suffix = path.suffix.lower()
    return {
        ".md": ArtifactKind.markdown,
        ".json": ArtifactKind.json,
        ".jsonl": ArtifactKind.jsonl,
        ".yml": ArtifactKind.yaml,
        ".yaml": ArtifactKind.yaml,
        ".lean": ArtifactKind.lean,
        ".py": ArtifactKind.python,
        ".sqlite": ArtifactKind.sqlite,
        ".log": ArtifactKind.log,
        ".pdf": ArtifactKind.other,
        ".html": ArtifactKind.other,
        ".txt": ArtifactKind.markdown,
    }.get(suffix, ArtifactKind.other)
