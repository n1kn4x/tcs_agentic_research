"""Persistent transactional AND-OR proof DAG.

SQLite is the authoritative LEAP state.  JSON and Lean files are evidence artifacts; they are not
used to reconstruct statuses.  All graph-changing decomposition checks happen in one
``BEGIN IMMEDIATE`` transaction so a rejected update cannot leave half-created nodes behind.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from contextlib import contextmanager
from typing import Iterator, Sequence

from ..artifact_store import ArtifactStore
from ..schemas import LeanStatement, utc_now
from .models import (
    AndChild,
    AndNode,
    AndStatus,
    AttemptRecord,
    BlueprintCandidate,
    DecompositionReview,
    LeanDiagnostic,
    OrNode,
    OrStatus,
    RetrievalHit,
    RunRecord,
)

_SCHEMA_VERSION = 1


class GraphInvariantError(ValueError):
    """Raised when a proposed graph mutation is cyclic, inconsistent, or non-progressing."""


class ProofGraph:
    DB_PATH = "LeanProject/LEAP/state.sqlite"

    def __init__(self, store: ArtifactStore):
        self.store = store
        self.path = store.resolve(self.DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS or_nodes (
                    id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    statement TEXT NOT NULL,
                    elaborated_statement TEXT NOT NULL,
                    imports_json TEXT NOT NULL,
                    namespace TEXT,
                    environment_fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('open','solving','proved','abandoned')),
                    proof_kind TEXT CHECK(proof_kind IN ('direct','decomposition')),
                    proof_content TEXT NOT NULL DEFAULT '',
                    proof_artifact_path TEXT NOT NULL DEFAULT '',
                    selected_and_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS and_nodes (
                    id TEXT PRIMARY KEY,
                    parent_or_id TEXT NOT NULL REFERENCES or_nodes(id),
                    fingerprint TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK(status IN ('open','solving','proved','paused','rejected')),
                    blueprint_json TEXT NOT NULL,
                    parent_proof TEXT NOT NULL,
                    sketch_artifact_path TEXT NOT NULL,
                    review_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS and_children (
                    and_id TEXT NOT NULL REFERENCES and_nodes(id) ON DELETE CASCADE,
                    child_or_id TEXT NOT NULL REFERENCES or_nodes(id),
                    required INTEGER NOT NULL CHECK(required IN (0,1)),
                    position INTEGER NOT NULL,
                    local_name TEXT NOT NULL,
                    PRIMARY KEY(and_id, child_or_id),
                    UNIQUE(and_id, position),
                    UNIQUE(and_id, local_name)
                );

                CREATE TABLE IF NOT EXISTS proof_dependencies (
                    owner_kind TEXT NOT NULL CHECK(owner_kind IN ('or','and')),
                    owner_id TEXT NOT NULL,
                    dependency_or_id TEXT NOT NULL REFERENCES or_nodes(id),
                    PRIMARY KEY(owner_kind, owner_id, dependency_or_id)
                );

                CREATE TABLE IF NOT EXISTS attempts (
                    id TEXT PRIMARY KEY,
                    or_id TEXT NOT NULL REFERENCES or_nodes(id),
                    mode TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    outcome TEXT NOT NULL,
                    candidate_sha256 TEXT NOT NULL,
                    candidate_artifact_path TEXT NOT NULL,
                    diagnostics_json TEXT NOT NULL,
                    retrieval_json TEXT NOT NULL,
                    parent_attempt_id TEXT,
                    duration_seconds REAL NOT NULL,
                    note TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(or_id, mode, ordinal)
                );

                CREATE TRIGGER IF NOT EXISTS attempts_immutable_update
                BEFORE UPDATE ON attempts BEGIN
                    SELECT RAISE(ABORT, 'LEAP attempts are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS attempts_immutable_delete
                BEFORE DELETE ON attempts BEGIN
                    SELECT RAISE(ABORT, 'LEAP attempts are immutable');
                END;

                CREATE TABLE IF NOT EXISTS proof_runs (
                    id TEXT PRIMARY KEY,
                    root_or_id TEXT NOT NULL REFERENCES or_nodes(id),
                    target_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    user_context TEXT NOT NULL,
                    final_artifact_path TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS node_leases (
                    or_id TEXT PRIMARY KEY REFERENCES or_nodes(id) ON DELETE CASCADE,
                    owner TEXT NOT NULL,
                    expires_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_and_parent ON and_nodes(parent_or_id, status);
                CREATE INDEX IF NOT EXISTS idx_child_or ON and_children(child_or_id);
                CREATE INDEX IF NOT EXISTS idx_attempt_or ON attempts(or_id, mode, ordinal);
                CREATE INDEX IF NOT EXISTS idx_or_status ON or_nodes(status);
                """
            )
            row = db.execute("SELECT value FROM metadata WHERE key = 'schema_version'").fetchone()
            if row is None:
                db.execute(
                    "INSERT INTO metadata(key, value) VALUES ('schema_version', ?)",
                    (str(_SCHEMA_VERSION),),
                )
            elif int(row["value"]) != _SCHEMA_VERSION:
                raise RuntimeError(
                    f"unsupported LEAP graph schema {row['value']}; expected {_SCHEMA_VERSION}"
                )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # ------------------------------------------------------------------
    # OR nodes and runs
    # ------------------------------------------------------------------
    def register_goal(
        self,
        goal: LeanStatement,
        *,
        environment_fingerprint: str,
        elaborated_statement: str = "",
    ) -> OrNode:
        with self._transaction() as db:
            node_id = self._ensure_or(
                db,
                goal,
                environment_fingerprint=environment_fingerprint,
                elaborated_statement=elaborated_statement,
            )
        return self.get_or(node_id)

    def _ensure_or(
        self,
        db: sqlite3.Connection,
        goal: LeanStatement,
        *,
        environment_fingerprint: str,
        elaborated_statement: str = "",
    ) -> str:
        fingerprint = statement_fingerprint(
            goal,
            environment_fingerprint,
            elaborated_statement=elaborated_statement,
        )
        existing = db.execute(
            "SELECT id FROM or_nodes WHERE fingerprint = ?", (fingerprint,)
        ).fetchone()
        if existing is not None:
            return str(existing["id"])
        node_id = "or_" + fingerprint[:24]
        canonical_name = canonical_lemma_name(goal.name, fingerprint)
        now = utc_now()
        db.execute(
            """
            INSERT INTO or_nodes(
                id, fingerprint, name, statement, elaborated_statement, imports_json, namespace,
                environment_fingerprint, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)
            """,
            (
                node_id,
                fingerprint,
                canonical_name,
                goal.statement.strip(),
                elaborated_statement.strip() or goal.statement.strip(),
                json.dumps(goal.imports, ensure_ascii=False),
                goal.namespace,
                environment_fingerprint,
                now,
                now,
            ),
        )
        return node_id

    def get_or(self, node_id: str) -> OrNode:
        with self._connect() as db:
            row = db.execute("SELECT * FROM or_nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown OR node {node_id}")
        return _or_from_row(row)

    def find_or(
        self, goal: LeanStatement, *, environment_fingerprint: str, elaborated_statement: str = ""
    ) -> OrNode | None:
        fingerprint = statement_fingerprint(
            goal, environment_fingerprint, elaborated_statement=elaborated_statement
        )
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM or_nodes WHERE fingerprint = ?", (fingerprint,)
            ).fetchone()
        return _or_from_row(row) if row is not None else None

    def create_or_resume_run(
        self, root: OrNode, target: LeanStatement, *, user_context: str
    ) -> RunRecord:
        target_dump = json.dumps(target.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256((root.node_id + "\0" + target_dump).encode()).hexdigest()
        run_id = "leap_" + digest[:20]
        now = utc_now()
        with self._transaction() as db:
            row = db.execute("SELECT id FROM proof_runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                db.execute(
                    """INSERT INTO proof_runs(
                           id, root_or_id, target_json, status, user_context, created_at, updated_at
                       ) VALUES (?, ?, ?, 'searching', ?, ?, ?)""",
                    (run_id, root.node_id, target_dump, user_context, now, now),
                )
            else:
                db.execute(
                    "UPDATE proof_runs SET user_context = ?, updated_at = ? WHERE id = ?",
                    (user_context, now, run_id),
                )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> RunRecord:
        with self._connect() as db:
            row = db.execute("SELECT * FROM proof_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown LEAP run {run_id}")
        return RunRecord(
            run_id=row["id"],
            root_or_id=row["root_or_id"],
            target=LeanStatement.model_validate(json.loads(row["target_json"])),
            status=row["status"],
            user_context=row["user_context"],
            final_artifact_path=row["final_artifact_path"],
        )

    def update_run(self, run_id: str, *, status: str, final_artifact_path: str = "") -> None:
        with self._transaction() as db:
            db.execute(
                """UPDATE proof_runs
                   SET status = ?, final_artifact_path = ?, updated_at = ? WHERE id = ?""",
                (status, final_artifact_path, utc_now(), run_id),
            )

    def acquire_lease(self, node_id: str, *, owner: str, ttl_seconds: int) -> bool:
        """Lease one OR node so concurrent workers cannot generate against stale state."""
        now = time.time()
        with self._transaction() as db:
            db.execute("DELETE FROM node_leases WHERE expires_at <= ?", (now,))
            existing = db.execute(
                "SELECT owner FROM node_leases WHERE or_id = ?", (node_id,)
            ).fetchone()
            if existing is not None and existing["owner"] != owner:
                return False
            db.execute(
                """INSERT INTO node_leases(or_id, owner, expires_at) VALUES (?, ?, ?)
                   ON CONFLICT(or_id) DO UPDATE SET owner=excluded.owner,
                       expires_at=excluded.expires_at""",
                (node_id, owner, now + max(1, ttl_seconds)),
            )
            db.execute(
                """UPDATE or_nodes SET status = 'solving', updated_at = ?
                   WHERE id = ? AND status IN ('open','solving')""",
                (utc_now(), node_id),
            )
        return True

    def release_lease(self, node_id: str, *, owner: str) -> None:
        with self._transaction() as db:
            db.execute(
                "DELETE FROM node_leases WHERE or_id = ? AND owner = ?", (node_id, owner)
            )
            db.execute(
                """UPDATE or_nodes SET status = 'open', updated_at = ?
                   WHERE id = ? AND status = 'solving'""",
                (utc_now(), node_id),
            )

    # ------------------------------------------------------------------
    # Verified commits
    # ------------------------------------------------------------------
    def commit_direct_proof(
        self,
        node_id: str,
        *,
        proof: str,
        artifact_path: str,
        dependency_or_ids: Sequence[str] = (),
    ) -> None:
        with self._transaction() as db:
            row = db.execute("SELECT status FROM or_nodes WHERE id = ?", (node_id,)).fetchone()
            if row is None:
                raise KeyError(node_id)
            for dependency_id in sorted(set(dependency_or_ids)):
                dependency = db.execute(
                    "SELECT status FROM or_nodes WHERE id = ?", (dependency_id,)
                ).fetchone()
                if dependency is None or dependency["status"] != OrStatus.proved.value:
                    raise GraphInvariantError(
                        f"direct proof dependency {dependency_id} is not already proved"
                    )
                if dependency_id == node_id:
                    raise GraphInvariantError("a proof cannot depend on its own declaration")
            db.execute(
                """UPDATE or_nodes
                   SET status = 'proved', proof_kind = 'direct', proof_content = ?,
                       proof_artifact_path = ?, selected_and_id = NULL, updated_at = ?
                   WHERE id = ?""",
                (proof, artifact_path, utc_now(), node_id),
            )
            db.execute(
                "DELETE FROM proof_dependencies WHERE owner_kind = 'or' AND owner_id = ?",
                (node_id,),
            )
            db.executemany(
                """INSERT INTO proof_dependencies(owner_kind, owner_id, dependency_or_id)
                   VALUES ('or', ?, ?)""",
                [(node_id, dependency_id) for dependency_id in sorted(set(dependency_or_ids))],
            )
            self._propagate(db)

    def commit_decomposition(
        self,
        parent_or_id: str,
        *,
        blueprint: BlueprintCandidate,
        children: Sequence[tuple[LeanStatement, bool, str]],
        parent_proof: str,
        sketch_artifact_path: str,
        review: DecompositionReview,
        environment_fingerprint: str,
        dependency_or_ids: Sequence[str] = (),
        child_elaborations: Sequence[str] = (),
    ) -> AndNode:
        if not review.accept:
            raise GraphInvariantError("a rejected review cannot be committed")
        if not children or not any(required for _, required, _ in children):
            raise GraphInvariantError("an AND node needs at least one required child")
        with self._transaction() as db:
            parent = db.execute("SELECT * FROM or_nodes WHERE id = ?", (parent_or_id,)).fetchone()
            if parent is None:
                raise KeyError(parent_or_id)
            child_rows: list[tuple[str, bool, str]] = []
            seen_child_ids: set[str] = set()
            seen_names: set[str] = set()
            for child_index, (statement, required, local_name) in enumerate(children):
                if statement.imports != json.loads(parent["imports_json"]):
                    raise GraphInvariantError("child imports must match the parent environment")
                if statement.namespace != parent["namespace"]:
                    raise GraphInvariantError("child namespace must match the parent environment")
                child_id = self._ensure_or(
                    db,
                    statement,
                    environment_fingerprint=environment_fingerprint,
                    elaborated_statement=(
                        child_elaborations[child_index]
                        if child_index < len(child_elaborations)
                        else ""
                    ),
                )
                if child_id in seen_child_ids:
                    raise GraphInvariantError("a decomposition contains duplicate child propositions")
                if local_name in seen_names:
                    raise GraphInvariantError("a decomposition contains duplicate child names")
                if child_id == parent_or_id:
                    raise GraphInvariantError("a child proposition is identical to its parent")
                if self._reachable(db, child_id, parent_or_id):
                    raise GraphInvariantError(
                        f"adding {parent_or_id} -> {child_id} would create a proof cycle"
                    )
                seen_child_ids.add(child_id)
                seen_names.add(local_name)
                child_rows.append((child_id, required, local_name))

            for dependency_id in sorted(set(dependency_or_ids)):
                dependency = db.execute(
                    "SELECT status FROM or_nodes WHERE id = ?", (dependency_id,)
                ).fetchone()
                if dependency is None or dependency["status"] != OrStatus.proved.value:
                    raise GraphInvariantError(
                        f"sketch dependency {dependency_id} is not already proved"
                    )

            identity = json.dumps(
                {
                    "parent": parent_or_id,
                    "children": child_rows,
                    "proof": normalize_lean_source(parent_proof),
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            fingerprint = hashlib.sha256(identity.encode()).hexdigest()
            existing = db.execute(
                "SELECT id FROM and_nodes WHERE fingerprint = ?", (fingerprint,)
            ).fetchone()
            if existing is not None:
                and_id = str(existing["id"])
            else:
                and_id = "and_" + fingerprint[:24]
                now = utc_now()
                db.execute(
                    """INSERT INTO and_nodes(
                           id, parent_or_id, fingerprint, status, blueprint_json, parent_proof,
                           sketch_artifact_path, review_json, created_at, updated_at
                       ) VALUES (?, ?, ?, 'open', ?, ?, ?, ?, ?, ?)""",
                    (
                        and_id,
                        parent_or_id,
                        fingerprint,
                        blueprint.model_dump_json(),
                        parent_proof,
                        sketch_artifact_path,
                        review.model_dump_json(),
                        now,
                        now,
                    ),
                )
                db.executemany(
                    """INSERT INTO and_children(
                           and_id, child_or_id, required, position, local_name
                       ) VALUES (?, ?, ?, ?, ?)""",
                    [
                        (and_id, child_id, int(required), position, local_name)
                        for position, (child_id, required, local_name) in enumerate(child_rows)
                    ],
                )
                db.executemany(
                    """INSERT INTO proof_dependencies(owner_kind, owner_id, dependency_or_id)
                       VALUES ('and', ?, ?)""",
                    [
                        (and_id, dependency_id)
                        for dependency_id in sorted(set(dependency_or_ids))
                    ],
                )
            db.execute(
                "UPDATE or_nodes SET status = 'open', updated_at = ? WHERE id = ? AND status != 'proved'",
                (utc_now(), parent_or_id),
            )
            self._propagate(db)
        return self.get_and(and_id)

    def pause_and(self, and_id: str) -> None:
        with self._transaction() as db:
            db.execute(
                "UPDATE and_nodes SET status = 'paused', updated_at = ? WHERE id = ? AND status != 'proved'",
                (utc_now(), and_id),
            )

    def activate_and(self, and_id: str) -> None:
        with self._transaction() as db:
            db.execute(
                "UPDATE and_nodes SET status = 'solving', updated_at = ? WHERE id = ? AND status IN ('open','paused')",
                (utc_now(), and_id),
            )

    def propagate(self) -> None:
        with self._transaction() as db:
            self._propagate(db)

    def _propagate(self, db: sqlite3.Connection) -> None:
        while True:
            changed = 0
            solvable = db.execute(
                """
                SELECT a.id, a.parent_or_id
                FROM and_nodes a
                WHERE a.status IN ('open','solving','paused')
                  AND EXISTS (
                      SELECT 1 FROM and_children c
                      WHERE c.and_id = a.id AND c.required = 1
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM and_children c
                      JOIN or_nodes o ON o.id = c.child_or_id
                      WHERE c.and_id = a.id AND c.required = 1 AND o.status != 'proved'
                  )
                ORDER BY a.created_at, a.id
                """
            ).fetchall()
            for row in solvable:
                changed += db.execute(
                    "UPDATE and_nodes SET status = 'proved', updated_at = ? WHERE id = ? AND status != 'proved'",
                    (utc_now(), row["id"]),
                ).rowcount
                changed += db.execute(
                    """UPDATE or_nodes
                       SET status = 'proved', proof_kind = 'decomposition', selected_and_id = ?,
                           proof_content = '', proof_artifact_path = '', updated_at = ?
                       WHERE id = ? AND status != 'proved'""",
                    (row["id"], utc_now(), row["parent_or_id"]),
                ).rowcount
            if changed == 0:
                return

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_and(self, and_id: str) -> AndNode:
        with self._connect() as db:
            row = db.execute("SELECT * FROM and_nodes WHERE id = ?", (and_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown AND node {and_id}")
            children = db.execute(
                "SELECT * FROM and_children WHERE and_id = ? ORDER BY position", (and_id,)
            ).fetchall()
        return _and_from_rows(row, children)

    def decompositions(self, parent_or_id: str, *, include_rejected: bool = False) -> list[AndNode]:
        query = "SELECT id FROM and_nodes WHERE parent_or_id = ?"
        parameters: list[object] = [parent_or_id]
        if not include_rejected:
            query += " AND status != 'rejected'"
        query += " ORDER BY created_at, id"
        with self._connect() as db:
            ids = [row["id"] for row in db.execute(query, parameters).fetchall()]
        return [self.get_and(str(and_id)) for and_id in ids]

    def dependencies(self, owner_kind: str, owner_id: str) -> list[str]:
        with self._connect() as db:
            return [
                str(row["dependency_or_id"])
                for row in db.execute(
                    """SELECT dependency_or_id FROM proof_dependencies
                       WHERE owner_kind = ? AND owner_id = ? ORDER BY dependency_or_id""",
                    (owner_kind, owner_id),
                ).fetchall()
            ]

    def ancestors(self, node_id: str, *, limit: int = 20) -> list[OrNode]:
        result: list[OrNode] = []
        seen = {node_id}
        frontier = [node_id]
        with self._connect() as db:
            while frontier and len(result) < limit:
                child = frontier.pop(0)
                rows = db.execute(
                    """SELECT DISTINCT a.parent_or_id
                       FROM and_children c JOIN and_nodes a ON a.id = c.and_id
                       WHERE c.child_or_id = ? ORDER BY a.parent_or_id""",
                    (child,),
                ).fetchall()
                for row in rows:
                    parent_id = str(row["parent_or_id"])
                    if parent_id in seen:
                        continue
                    seen.add(parent_id)
                    parent_row = db.execute(
                        "SELECT * FROM or_nodes WHERE id = ?", (parent_id,)
                    ).fetchone()
                    if parent_row is not None:
                        result.append(_or_from_row(parent_row))
                        frontier.append(parent_id)
        return result[:limit]

    def proved_nodes(self, *, exclude: Sequence[str] = (), limit: int = 200) -> list[OrNode]:
        excluded = set(exclude)
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM or_nodes WHERE status = 'proved' ORDER BY updated_at DESC, id LIMIT ?",
                (limit + len(excluded),),
            ).fetchall()
        return [_or_from_row(row) for row in rows if row["id"] not in excluded][:limit]

    def open_nodes(self, *, reachable_from: str | None = None) -> list[OrNode]:
        # For reporting, include all open descendants in deterministic DFS order.
        if reachable_from is None:
            with self._connect() as db:
                rows = db.execute(
                    "SELECT * FROM or_nodes WHERE status != 'proved' ORDER BY created_at, id"
                ).fetchall()
            return [_or_from_row(row) for row in rows]
        result: list[OrNode] = []
        seen: set[str] = set()
        frontier = [reachable_from]
        while frontier:
            node_id = frontier.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            node = self.get_or(node_id)
            if node.status != OrStatus.proved:
                result.append(node)
            decompositions = self.decompositions(node_id)
            for decomposition in reversed(decompositions):
                frontier.extend(
                    reversed(
                        [child.child_or_id for child in decomposition.children if child.required]
                    )
                )
        return result

    def checkpoint(self) -> None:
        """Merge the WAL so an artifact hash of ``state.sqlite`` covers committed state."""
        with self._connect() as db:
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def node_count(self) -> int:
        with self._connect() as db:
            return int(db.execute("SELECT count(*) AS n FROM or_nodes").fetchone()["n"])

    def proved_count(self) -> int:
        with self._connect() as db:
            return int(
                db.execute("SELECT count(*) AS n FROM or_nodes WHERE status = 'proved'").fetchone()[
                    "n"
                ]
            )

    def next_attempt_ordinal(self, or_id: str, mode: str) -> int:
        with self._connect() as db:
            row = db.execute(
                "SELECT coalesce(max(ordinal), -1) + 1 AS n FROM attempts WHERE or_id = ? AND mode = ?",
                (or_id, mode),
            ).fetchone()
        return int(row["n"])

    def candidate_seen(self, or_id: str, mode: str, candidate_sha256: str) -> bool:
        if not candidate_sha256:
            return False
        with self._connect() as db:
            row = db.execute(
                """SELECT 1 FROM attempts
                   WHERE or_id = ? AND mode = ? AND candidate_sha256 = ? LIMIT 1""",
                (or_id, mode, candidate_sha256),
            ).fetchone()
        return row is not None

    def record_attempt(self, attempt: AttemptRecord) -> None:
        with self._transaction() as db:
            # Allocate the ordinal under the same write lock as the insert.  This prevents two
            # workers recording the same node/mode from racing between MAX() and INSERT.
            ordinal = int(
                db.execute(
                    """SELECT coalesce(max(ordinal), -1) + 1 AS n FROM attempts
                       WHERE or_id = ? AND mode = ?""",
                    (attempt.or_id, attempt.mode),
                ).fetchone()["n"]
            )
            db.execute(
                """INSERT INTO attempts(
                       id, or_id, mode, ordinal, outcome, candidate_sha256,
                       candidate_artifact_path, diagnostics_json, retrieval_json,
                       parent_attempt_id, duration_seconds, note, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    attempt.attempt_id,
                    attempt.or_id,
                    attempt.mode,
                    ordinal,
                    attempt.outcome,
                    attempt.candidate_sha256,
                    attempt.candidate_artifact_path,
                    json.dumps(
                        [item.model_dump(mode="json") for item in attempt.diagnostics],
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        [item.model_dump(mode="json") for item in attempt.retrieval],
                        ensure_ascii=False,
                    ),
                    attempt.parent_attempt_id,
                    attempt.duration_seconds,
                    attempt.note,
                    utc_now(),
                ),
            )

    def recent_attempts(self, or_id: str, *, limit: int = 8) -> list[AttemptRecord]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT * FROM attempts WHERE or_id = ?
                   ORDER BY created_at DESC, ordinal DESC LIMIT ?""",
                (or_id, limit),
            ).fetchall()
        records: list[AttemptRecord] = []
        for row in rows:
            records.append(
                AttemptRecord(
                    attempt_id=row["id"],
                    or_id=row["or_id"],
                    mode=row["mode"],
                    ordinal=row["ordinal"],
                    outcome=row["outcome"],
                    candidate_sha256=row["candidate_sha256"],
                    candidate_artifact_path=row["candidate_artifact_path"],
                    diagnostics=[
                        LeanDiagnostic.model_validate(item)
                        for item in json.loads(row["diagnostics_json"])
                    ],
                    retrieval=[
                        RetrievalHit.model_validate(item)
                        for item in json.loads(row["retrieval_json"])
                    ],
                    parent_attempt_id=row["parent_attempt_id"],
                    duration_seconds=row["duration_seconds"],
                    note=row["note"],
                )
            )
        return records

    def _reachable(self, db: sqlite3.Connection, start_or_id: str, target_or_id: str) -> bool:
        if start_or_id == target_or_id:
            return True
        seen = {start_or_id}
        frontier = [start_or_id]
        while frontier:
            current = frontier.pop()
            rows = db.execute(
                """SELECT c.child_or_id
                   FROM and_nodes a JOIN and_children c ON c.and_id = a.id
                   WHERE a.parent_or_id = ? AND a.status != 'rejected'""",
                (current,),
            ).fetchall()
            for row in rows:
                child = str(row["child_or_id"])
                if child == target_or_id:
                    return True
                if child not in seen:
                    seen.add(child)
                    frontier.append(child)
        return False


def normalize_lean_source(source: str) -> str:
    """Normalize insignificant whitespace without trying to rewrite Lean syntax."""
    return re.sub(r"\s+", " ", source.strip())


def statement_fingerprint(
    goal: LeanStatement,
    environment_fingerprint: str,
    *,
    elaborated_statement: str = "",
) -> str:
    # LEAP asks Lean to pretty-print an elaborated closed proposition before registration.  Source
    # is the deterministic fallback for callers constructing graph fixtures without a compiler.
    identity = {
        "proposition": normalize_lean_source(elaborated_statement or goal.statement),
        "imports": goal.imports,
        "namespace": goal.namespace,
        "environment": environment_fingerprint,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def canonical_lemma_name(hint: str, fingerprint: str) -> str:
    # The name must depend only on proposition identity.  If it included a blueprint's human label,
    # two branches could deduplicate to one OR node while their verified parent proofs referenced
    # different names, breaking final assembly.
    del hint
    return f"leap_goal_{fingerprint[:16]}"


def _or_from_row(row: sqlite3.Row) -> OrNode:
    return OrNode(
        node_id=row["id"],
        fingerprint=row["fingerprint"],
        goal=LeanStatement(
            name=row["name"],
            statement=row["statement"],
            imports=json.loads(row["imports_json"]),
            namespace=row["namespace"],
        ),
        environment_fingerprint=row["environment_fingerprint"],
        status=OrStatus(row["status"]),
        proof_kind=row["proof_kind"],
        proof_content=row["proof_content"],
        proof_artifact_path=row["proof_artifact_path"],
        selected_and_id=row["selected_and_id"],
    )


def _and_from_rows(row: sqlite3.Row, children: Sequence[sqlite3.Row]) -> AndNode:
    return AndNode(
        node_id=row["id"],
        parent_or_id=row["parent_or_id"],
        fingerprint=row["fingerprint"],
        status=AndStatus(row["status"]),
        blueprint=BlueprintCandidate.model_validate(json.loads(row["blueprint_json"])),
        parent_proof=row["parent_proof"],
        sketch_artifact_path=row["sketch_artifact_path"],
        review=DecompositionReview.model_validate(json.loads(row["review_json"])),
        children=[
            AndChild(
                child_or_id=child["child_or_id"],
                required=bool(child["required"]),
                position=child["position"],
                local_name=child["local_name"],
            )
            for child in children
        ],
    )
