"""Atomic filesystem storage for the research kernel and independent subsystems.

The canonical research state is four small files: ``KernelState.json``, ``Records.jsonl``,
``Actions.jsonl``, and ``Events.jsonl``.  Subsystem-specific databases and run artifacts are opaque
to the kernel.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import yaml
from pydantic import BaseModel

from .schemas import ArtifactKind, ArtifactRef, utc_now

if TYPE_CHECKING:
    from .core.models import ActionRecord, KernelState, ResearchRecord


class ArtifactStore:
    RESEARCH_TASK = "InitialResearchTask.md"
    KERNEL_STATE = "KernelState.json"
    RECORD_LEDGER = "Records.jsonl"
    ACTION_LEDGER = "Actions.jsonl"
    EVENT_LEDGER = "Events.jsonl"
    MODEL_LEDGER = "ModelCalls.jsonl"
    TASK_VERSION_LEDGER = "TaskVersions.jsonl"

    CORE_DIRECTORIES = ("Runs", "Reports", "Subsystems")
    CORE_JSONL = (
        RECORD_LEDGER,
        ACTION_LEDGER,
        EVENT_LEDGER,
        MODEL_LEDGER,
        TASK_VERSION_LEDGER,
    )

    def __init__(self, workspace: str | Path):
        self.root = Path(workspace).expanduser().resolve()

    def initialize_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in self.CORE_DIRECTORIES:
            self.resolve(directory).mkdir(parents=True, exist_ok=True)
        for rel_path in self.CORE_JSONL:
            path = self.resolve(rel_path)
            if not path.exists():
                path.write_text("", encoding="utf-8")

    @contextmanager
    def exclusive_lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".research.lock"
        with lock_path.open("a+", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"workspace is already active: {lock_path}") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

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
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        return str(candidate.resolve().relative_to(self.root))

    def exists(self, path: str | Path) -> bool:
        return self.resolve(path).exists()

    def read_text(self, path: str | Path) -> str:
        return self.resolve(path).read_text(encoding="utf-8")

    def read_bytes(self, path: str | Path) -> bytes:
        return self.resolve(path).read_bytes()

    def write_text(self, path: str | Path, content: str) -> ArtifactRef:
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, target)
        return self.artifact_ref(target)

    def write_bytes(self, path: str | Path, content: bytes) -> ArtifactRef:
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(content)
        os.replace(tmp, target)
        return self.artifact_ref(target)

    def read_json(self, path: str | Path) -> Any:
        return json.loads(self.read_text(path))

    def write_json(self, path: str | Path, payload: Any, *, indent: int = 2) -> ArtifactRef:
        return self.write_text(
            path,
            json.dumps(to_plain(payload), indent=indent, ensure_ascii=False, sort_keys=True) + "\n",
        )

    def write_yaml(self, path: str | Path, payload: Any) -> ArtifactRef:
        return self.write_text(
            path,
            yaml.safe_dump(to_plain(payload), sort_keys=False, allow_unicode=True),
        )

    def append_jsonl(self, path: str | Path, payload: Any) -> ArtifactRef:
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(to_plain(payload), ensure_ascii=False, sort_keys=True) + "\n"
        with target.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        return self.artifact_ref(target)

    def read_jsonl(self, path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
        target = self.resolve(path)
        if not target.exists():
            return []
        lines = target.read_text(encoding="utf-8").splitlines()
        if limit is not None:
            lines = lines[-limit:]
        records: list[dict[str, Any]] = []
        for index, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL in {self.relpath(target)} at line {index}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row in {self.relpath(target)} is not an object")
            records.append(value)
        return records

    def load_kernel_state(self) -> "KernelState | None":
        from .core.models import KernelState

        if not self.exists(self.KERNEL_STATE):
            return None
        return KernelState.model_validate(self.read_json(self.KERNEL_STATE))

    def save_kernel_state(self, state: "KernelState") -> ArtifactRef:
        state.updated_at = utc_now()
        return self.write_json(self.KERNEL_STATE, state)

    def append_records(self, records: Iterable["ResearchRecord"]) -> None:
        for record in records:
            self.append_jsonl(self.RECORD_LEDGER, record)

    def read_records(self) -> list["ResearchRecord"]:
        from .core.models import ResearchRecord

        return [ResearchRecord.model_validate(row) for row in self.read_jsonl(self.RECORD_LEDGER)]

    def append_action(self, action: "ActionRecord") -> None:
        action.updated_at = utc_now()
        self.append_jsonl(self.ACTION_LEDGER, action)

    def read_action_events(self) -> list["ActionRecord"]:
        from .core.models import ActionRecord

        return [ActionRecord.model_validate(row) for row in self.read_jsonl(self.ACTION_LEDGER)]

    def latest_actions(self) -> dict[str, "ActionRecord"]:
        latest: dict[str, ActionRecord] = {}
        for action in self.read_action_events():
            latest[action.action_id] = action
        return latest

    def append_event(self, event_type: str, payload: dict[str, Any] | None = None) -> ArtifactRef:
        return self.append_jsonl(
            self.EVENT_LEDGER,
            {"event_type": event_type, "created_at": utc_now(), **(payload or {})},
        )

    def append_model_call(self, payload: Any) -> None:
        self.append_jsonl(self.MODEL_LEDGER, payload)

    def load_subsystem_state(self, name: str) -> dict[str, Any]:
        path = f"Subsystems/{name}.json"
        if not self.exists(path):
            return {}
        value = self.read_json(path)
        if not isinstance(value, dict):
            raise ValueError(f"subsystem state is not an object: {path}")
        return value

    def save_subsystem_state(self, name: str, state: dict[str, Any]) -> ArtifactRef:
        return self.write_json(f"Subsystems/{name}.json", state)

    def create_action_dir(self, cycle: int, action_id: str, subsystem: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in subsystem)[:40]
        rel = f"Runs/{cycle:06d}_{safe}_{action_id}"
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
        inferred = kind or infer_kind(target)
        digest = sha256_file(target) if target.is_file() else None
        return ArtifactRef(
            path=self.relpath(target), kind=inferred, sha256=digest, summary=summary
        )

    def manifest(self, *, max_items: int = 100) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for path in sorted(item for item in self.root.rglob("*") if item.is_file()):
            rel = self.relpath(path)
            if any(part.startswith(".") for part in Path(rel).parts):
                continue
            stat = path.stat()
            entries.append(
                {
                    "path": rel,
                    "kind": infer_kind(path).value,
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                }
            )
            if len(entries) >= max_items:
                break
        return entries


def merge_state(original: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge a subsystem-owned state patch without interpreting its fields."""
    merged = dict(original)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_state(merged[key], value)
        else:
            merged[key] = value
    return merged


def to_plain(payload: Any) -> Any:
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    if isinstance(payload, dict):
        return {str(key): to_plain(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [to_plain(value) for value in payload]
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_kind(path: Path) -> ArtifactKind:
    if path.is_dir():
        return ArtifactKind.directory
    return {
        ".md": ArtifactKind.markdown,
        ".txt": ArtifactKind.markdown,
        ".json": ArtifactKind.json,
        ".jsonl": ArtifactKind.jsonl,
        ".yml": ArtifactKind.yaml,
        ".yaml": ArtifactKind.yaml,
        ".lean": ArtifactKind.lean,
        ".py": ArtifactKind.python,
        ".sqlite": ArtifactKind.sqlite,
        ".log": ArtifactKind.log,
        ".csv": ArtifactKind.other,
    }.get(path.suffix.lower(), ArtifactKind.other)
