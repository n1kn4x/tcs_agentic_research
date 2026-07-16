"""Canonical SQLite index for imported literature.

The JSON/JSONL files in ``LiteratureDB`` remain the durable audit log, but this index provides
fast, stable lookup over canonical papers, aliases, passages, extracted statements, quotes, and
support IDs.  It is deliberately deterministic and has no LLM dependency.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from typing import Any, Iterable

from ..artifact_store import ArtifactStore
from ..schemas import LiteratureExtract, LiteratureQuote, LiteratureStatement, PaperMetadata


class LiteratureIndex:
    """Materialized registry/search index for LiteratureDB.

    The index is rebuildable from canonical artifacts.  Search uses SQLite FTS5 when available and
    falls back to LIKE queries on minimal SQLite builds.
    """

    INDEX_PATH = "LiteratureDB/index.sqlite"

    def __init__(self, store: ArtifactStore):
        self.store = store
        self.path = self.store.resolve(self.INDEX_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_empty(self) -> bool:
        """Return whether the index has no materialized papers/passages/statements."""
        with self._connect() as db:
            for table in ["papers", "passages", "statements"]:
                row = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                if row and int(row[0]) > 0:
                    return False
        return True

    def rebuild(self) -> None:
        """Rebuild the materialized index from current LiteratureDB artifacts."""
        if self.path.exists():
            self.path.unlink()
        self._init_schema()
        papers: dict[str, PaperMetadata] = {}
        for record in self.store.read_jsonl("LiteratureDB/papers.jsonl"):
            try:
                paper = PaperMetadata.model_validate(record)
            except Exception:
                continue
            papers[paper.paper_id] = paper
            self.upsert_paper(paper)
            if paper.text_path and self.store.exists(paper.text_path):
                self.index_paper_text(paper)
        grouped: dict[tuple[str, str], list[LiteratureStatement]] = {}
        for record in self.store.read_jsonl("LiteratureDB/statements.jsonl"):
            try:
                quote = LiteratureQuote(
                    quote_id=str(record.get("quote_id") or _stable_id("quote", str(record))),
                    citation_key=str(record.get("citation_key") or ""),
                    paper_id=str(record.get("paper_id") or ""),
                    locator=str(record.get("locator") or ""),
                    quote=str(record.get("quote") or record.get("original_statement") or ""),
                    char_start=record.get("char_start"),
                    char_end=record.get("char_end"),
                    source_sha256=str(record.get("source_sha256") or ""),
                    validated=bool(record.get("validated_exact_substring")),
                )
                statement = LiteratureStatement(
                    statement_id=str(record.get("statement_id") or _stable_id("lit_stmt", str(record))),
                    support_id=str(record.get("support_id") or ""),
                    citation_key=quote.citation_key,
                    paper_id=quote.paper_id,
                    kind=str(record.get("kind") or "other"),
                    label=str(record.get("label") or ""),
                    original_statement=str(record.get("original_statement") or record.get("statement_text") or ""),
                    statement_text=str(record.get("statement_text") or record.get("original_statement") or ""),
                    provenance=[quote],
                    confidence=float(record.get("confidence") or 0.0),
                )
            except Exception:
                continue
            grouped.setdefault((statement.paper_id, statement.citation_key), []).append(statement)
        for (paper_id, citation_key), statements in grouped.items():
            self.index_extract(
                LiteratureExtract(
                    citation_key=citation_key,
                    paper_id=paper_id,
                    theorem_statements=[
                        item for item in statements if item.kind not in {"algorithm", "lower_bound"}
                    ],
                    algorithm_statements=[item for item in statements if item.kind == "algorithm"],
                    lower_bound_statements=[item for item in statements if item.kind == "lower_bound"],
                )
            )

    def upsert_paper(self, paper: PaperMetadata, *, aliases: Iterable[str] = ()) -> None:
        """Insert/update paper metadata and all known aliases."""
        pdf_sha = self._artifact_sha(paper.pdf_path)
        text_sha = self._artifact_sha(paper.text_path)
        payload = json.dumps(paper.model_dump(mode="json"), sort_keys=True)
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO papers(
                    paper_id, canonical_key, title, authors_json, year, doi, arxiv_id,
                    url, venue, openalex_id, pdf_sha256, text_sha256, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    canonical_key=excluded.canonical_key,
                    title=excluded.title,
                    authors_json=excluded.authors_json,
                    year=excluded.year,
                    doi=excluded.doi,
                    arxiv_id=excluded.arxiv_id,
                    url=excluded.url,
                    venue=excluded.venue,
                    openalex_id=excluded.openalex_id,
                    pdf_sha256=excluded.pdf_sha256,
                    text_sha256=excluded.text_sha256,
                    metadata_json=excluded.metadata_json
                """,
                (
                    paper.paper_id,
                    paper.citation_key,
                    paper.title,
                    json.dumps(paper.authors, sort_keys=True),
                    paper.year,
                    paper.doi.lower(),
                    _normalize_arxiv_key(paper.arxiv_id),
                    paper.url,
                    paper.venue,
                    _openalex_id(paper),
                    pdf_sha,
                    text_sha,
                    payload,
                ),
            )
            for alias_type, value in self._paper_aliases(paper, pdf_sha=pdf_sha, text_sha=text_sha):
                self._insert_alias(db, paper.paper_id, alias_type, value)
            for alias in aliases:
                if alias:
                    self._insert_alias(db, paper.paper_id, "citation_key", alias)
            db.commit()

    def add_alias(self, paper_id: str, alias_type: str, alias_value: str) -> None:
        if not alias_value:
            return
        with self._connect() as db:
            self._insert_alias(db, paper_id, alias_type, alias_value)
            db.commit()

    def find_paper(
        self,
        *,
        citation_key: str | None = None,
        doi: str | None = None,
        arxiv_id: str | None = None,
        openalex_id: str | None = None,
        title: str | None = None,
        pdf_sha256: str | None = None,
        text_sha256: str | None = None,
    ) -> PaperMetadata | None:
        """Find an existing canonical paper by any strong identifier or normalized title."""
        lookups: list[tuple[str, str]] = []
        if citation_key:
            lookups.append(("citation_key", citation_key))
        if doi:
            lookups.append(("doi", doi.lower()))
        if arxiv_id:
            lookups.append(("arxiv", _normalize_arxiv_key(arxiv_id)))
        if openalex_id:
            lookups.append(("openalex", _normalize_openalex_key(openalex_id)))
        if pdf_sha256:
            lookups.append(("pdf_sha256", pdf_sha256))
        if text_sha256:
            lookups.append(("text_sha256", text_sha256))
        if title:
            lookups.append(("title", _normalize_title(title)))
        with self._connect() as db:
            for alias_type, alias_value in lookups:
                row = db.execute(
                    """
                    SELECT p.metadata_json
                    FROM paper_aliases a JOIN papers p ON p.paper_id = a.paper_id
                    WHERE a.alias_type = ? AND a.alias_value = ?
                    ORDER BY p.rowid DESC LIMIT 1
                    """,
                    (alias_type, alias_value),
                ).fetchone()
                if row:
                    return PaperMetadata.model_validate(json.loads(row[0]))
        return None

    def index_paper_text(self, paper: PaperMetadata) -> int:
        """Chunk and index the extracted text for a paper; returns number of passages."""
        if not paper.text_path or not self.store.exists(paper.text_path):
            return 0
        text = self.store.read_text(paper.text_path)
        text_sha = self._artifact_sha(paper.text_path)
        passages = list(_chunks(text))
        with self._connect() as db:
            db.execute("DELETE FROM passages WHERE paper_id = ?", (paper.paper_id,))
            if self._has_fts(db):
                db.execute("DELETE FROM passage_fts WHERE paper_id = ?", (paper.paper_id,))
            for idx, chunk in enumerate(passages, start=1):
                passage_id = _stable_id("passage", paper.paper_id, str(chunk.start), str(chunk.end), chunk.text)
                db.execute(
                    """
                    INSERT OR REPLACE INTO passages(
                        passage_id, paper_id, citation_key, title, locator, char_start,
                        char_end, text, text_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        passage_id,
                        paper.paper_id,
                        paper.citation_key,
                        paper.title,
                        chunk.locator,
                        chunk.start,
                        chunk.end,
                        chunk.text,
                        text_sha,
                    ),
                )
                if self._has_fts(db):
                    db.execute(
                        "INSERT INTO passage_fts(passage_id, paper_id, citation_key, title, locator, text) VALUES (?, ?, ?, ?, ?, ?)",
                        (passage_id, paper.paper_id, paper.citation_key, paper.title, chunk.locator, chunk.text),
                    )
            db.commit()
        return len(passages)

    def index_extract(self, extract: LiteratureExtract) -> dict[str, str]:
        """Index extracted statements/quotes and return ``statement_id -> support_id``."""
        paper = self.find_paper(citation_key=extract.citation_key)
        text_path = ""
        if extract.text_artifact_ref:
            text_path = extract.text_artifact_ref.path
        elif paper:
            text_path = paper.text_path
        text = self.store.read_text(text_path) if text_path and self.store.exists(text_path) else ""
        text_sha = self._artifact_sha(text_path) if text_path else ""
        support_ids: dict[str, str] = {}
        statements = [
            *extract.theorem_statements,
            *extract.algorithm_statements,
            *extract.lower_bound_statements,
        ]
        with self._connect() as db:
            if extract.paper_id:
                db.execute("DELETE FROM statements WHERE paper_id = ?", (extract.paper_id,))
                db.execute("DELETE FROM quotes WHERE paper_id = ?", (extract.paper_id,))
                db.execute("DELETE FROM supports WHERE paper_id = ?", (extract.paper_id,))
                if self._has_fts(db):
                    db.execute("DELETE FROM statement_fts WHERE paper_id = ?", (extract.paper_id,))
            for statement in statements:
                quote = statement.provenance[0] if statement.provenance else None
                if quote is None:
                    quote = LiteratureQuote(
                        citation_key=extract.citation_key,
                        paper_id=extract.paper_id,
                        locator=statement.label,
                        quote=statement.original_statement,
                    )
                    statement.provenance = [quote]
                validated, start, end = _validate_or_locate_quote(text, quote)
                if start is not None:
                    quote.char_start = start
                if end is not None:
                    quote.char_end = end
                quote.source_sha256 = quote.source_sha256 or text_sha
                quote.validated = validated
                quote_id = quote.quote_id
                support_id = _stable_id("lit_sup", statement.statement_id, quote_id)
                statement.support_id = support_id
                statement_quality_ok = validated and _statement_span_quality(statement)
                support_ids[statement.statement_id] = support_id
                db.execute(
                    """
                    INSERT OR REPLACE INTO quotes(
                        quote_id, paper_id, citation_key, locator, char_start, char_end,
                        quote, text_sha256, validated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        quote_id,
                        statement.paper_id or extract.paper_id,
                        statement.citation_key or extract.citation_key,
                        quote.locator,
                        quote.char_start,
                        quote.char_end,
                        quote.quote,
                        text_sha,
                        1 if validated else 0,
                    ),
                )
                db.execute(
                    """
                    INSERT OR REPLACE INTO statements(
                        statement_id, paper_id, citation_key, kind, label, title,
                        original_statement, statement_text, quote_id, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        statement.statement_id,
                        statement.paper_id or extract.paper_id,
                        statement.citation_key or extract.citation_key,
                        statement.kind,
                        statement.label,
                        statement.title,
                        statement.original_statement,
                        statement.statement_text,
                        quote_id,
                        statement.confidence,
                    ),
                )
                db.execute(
                    """
                    INSERT OR REPLACE INTO supports(
                        support_id, statement_id, quote_id, paper_id, citation_key,
                        support_level, relation, validated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        support_id,
                        statement.statement_id,
                        quote_id,
                        statement.paper_id or extract.paper_id,
                        statement.citation_key or extract.citation_key,
                        "primary_exact" if statement_quality_ok else (
                            "candidate_exact_quote" if validated else "candidate_unverified_span"
                        ),
                        "states",
                        1 if statement_quality_ok else 0,
                    ),
                )
                if self._has_fts(db):
                    searchable = "\n".join(
                        [
                            statement.kind,
                            statement.label,
                            statement.title,
                            statement.original_statement,
                            statement.statement_text,
                            quote.quote,
                        ]
                    )
                    db.execute(
                        """
                        INSERT INTO statement_fts(
                            statement_id, support_id, paper_id, citation_key, kind, label, text
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            statement.statement_id,
                            support_id,
                            statement.paper_id or extract.paper_id,
                            statement.citation_key or extract.citation_key,
                            statement.kind,
                            statement.label,
                            searchable,
                        ),
                    )
            db.commit()
        return support_ids

    def support_exists(self, support_id: str) -> bool:
        return self.support_details(support_id) is not None

    def support_details(self, support_id: str) -> dict[str, Any] | None:
        """Return statement/quote provenance for a support ID."""
        if not support_id:
            return None
        with self._connect() as db:
            row = db.execute(
                """
                SELECT sup.support_id, s.statement_id, q.quote_id,
                       COALESCE(s.paper_id, sup.paper_id, q.paper_id) AS paper_id,
                       COALESCE(s.citation_key, sup.citation_key, q.citation_key) AS citation_key,
                       s.kind, s.label, s.statement_text, q.locator, q.char_start,
                       q.char_end, q.text_sha256, sup.validated,
                       q.validated AS quote_validated, sup.support_level, sup.relation
                FROM supports sup
                LEFT JOIN statements s ON s.statement_id = sup.statement_id
                LEFT JOIN quotes q ON q.quote_id = sup.quote_id
                WHERE sup.support_id = ?
                LIMIT 1
                """,
                (support_id,),
            ).fetchone()
            return dict(row) if row else None

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Return statement/passages ranked by local index relevance."""
        query = query.strip()
        if not query:
            return []
        with self._connect() as db:
            if self._has_fts(db):
                rows = self._search_fts(db, query, limit=limit)
            else:
                rows = self._search_like(db, query, limit=limit)
        return rows[:limit]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers(
                    paper_id TEXT PRIMARY KEY,
                    canonical_key TEXT NOT NULL,
                    title TEXT NOT NULL,
                    authors_json TEXT NOT NULL DEFAULT '[]',
                    year INTEGER,
                    doi TEXT,
                    arxiv_id TEXT,
                    url TEXT,
                    venue TEXT,
                    openalex_id TEXT,
                    pdf_sha256 TEXT,
                    text_sha256 TEXT,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS paper_aliases(
                    alias_type TEXT NOT NULL,
                    alias_value TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    PRIMARY KEY(alias_type, alias_value),
                    FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
                );
                CREATE TABLE IF NOT EXISTS passages(
                    passage_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    citation_key TEXT NOT NULL,
                    title TEXT NOT NULL,
                    locator TEXT,
                    char_start INTEGER,
                    char_end INTEGER,
                    text TEXT NOT NULL,
                    text_sha256 TEXT
                );
                CREATE TABLE IF NOT EXISTS quotes(
                    quote_id TEXT PRIMARY KEY,
                    paper_id TEXT,
                    citation_key TEXT,
                    locator TEXT,
                    char_start INTEGER,
                    char_end INTEGER,
                    quote TEXT NOT NULL,
                    text_sha256 TEXT,
                    validated INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS statements(
                    statement_id TEXT PRIMARY KEY,
                    paper_id TEXT,
                    citation_key TEXT,
                    kind TEXT,
                    label TEXT,
                    title TEXT,
                    original_statement TEXT NOT NULL,
                    statement_text TEXT NOT NULL,
                    quote_id TEXT,
                    confidence REAL NOT NULL DEFAULT 0.0
                );
                CREATE TABLE IF NOT EXISTS supports(
                    support_id TEXT PRIMARY KEY,
                    statement_id TEXT,
                    quote_id TEXT,
                    paper_id TEXT,
                    citation_key TEXT,
                    support_level TEXT,
                    relation TEXT,
                    validated INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
            try:
                db.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS passage_fts USING fts5(passage_id UNINDEXED, paper_id UNINDEXED, citation_key UNINDEXED, title UNINDEXED, locator UNINDEXED, text)"
                )
                db.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS statement_fts USING fts5(statement_id UNINDEXED, support_id UNINDEXED, paper_id UNINDEXED, citation_key UNINDEXED, kind UNINDEXED, label UNINDEXED, text)"
                )
                db.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('fts5', '1')")
            except sqlite3.Error:
                # Some Python/SQLite builds omit FTS5; search methods use LIKE in that case.
                pass
            db.commit()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def _has_fts(self, db: sqlite3.Connection) -> bool:
        row = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='statement_fts'"
        ).fetchone()
        return row is not None

    def _search_fts(self, db: sqlite3.Connection, query: str, *, limit: int) -> list[dict[str, Any]]:
        match_query = _fts_query(query)
        if not match_query:
            return []
        rows: list[dict[str, Any]] = []
        stmt_rows = db.execute(
            """
            SELECT 'statement' AS result_kind, s.statement_id, sup.support_id, s.paper_id,
                   s.citation_key, p.title AS paper_title, p.year, s.kind, s.label,
                   s.statement_text, s.original_statement, s.quote_id, q.locator,
                   q.quote, q.char_start, q.char_end, q.text_sha256,
                   q.validated AS quote_validated, sup.validated AS support_validated,
                   sup.support_level, sup.relation, bm25(statement_fts) AS rank
            FROM statement_fts
            JOIN statements s ON s.statement_id = statement_fts.statement_id
            LEFT JOIN supports sup ON sup.support_id = statement_fts.support_id
            LEFT JOIN quotes q ON q.quote_id = s.quote_id
            LEFT JOIN papers p ON p.paper_id = s.paper_id
            WHERE statement_fts MATCH ?
            ORDER BY rank LIMIT ?
            """,
            (match_query, max(limit * 2, 5)),
        ).fetchall()
        for row in stmt_rows:
            rows.append(_row_to_dict(row, boost=3.0))
        passage_rows = db.execute(
            """
            SELECT 'text_chunk' AS result_kind, p.passage_id, p.paper_id, p.citation_key,
                   p.title AS paper_title, NULL AS year, p.locator, p.text AS statement_text,
                   p.text AS quote, p.char_start, p.char_end, p.text_sha256,
                   bm25(passage_fts) AS rank
            FROM passage_fts
            JOIN passages p ON p.passage_id = passage_fts.passage_id
            WHERE passage_fts MATCH ?
            ORDER BY rank LIMIT ?
            """,
            (match_query, max(limit * 2, 5)),
        ).fetchall()
        for row in passage_rows:
            rows.append(_row_to_dict(row, boost=1.0))
        rows.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return rows

    def _search_like(self, db: sqlite3.Connection, query: str, *, limit: int) -> list[dict[str, Any]]:
        terms = _terms(query)
        if not terms:
            return []
        like_clauses = " OR ".join(["LOWER(statement_text) LIKE ?" for _ in terms])
        params = [f"%{term}%" for term in terms]
        rows = []
        for row in db.execute(
            f"""
            SELECT 'statement' AS result_kind, s.statement_id, sup.support_id, s.paper_id,
                   s.citation_key, p.title AS paper_title, p.year, s.kind, s.label,
                   s.statement_text, s.original_statement, s.quote_id, q.locator,
                   q.quote, q.char_start, q.char_end, q.text_sha256,
                   q.validated AS quote_validated, sup.validated AS support_validated,
                   sup.support_level, sup.relation, 0.0 AS rank
            FROM statements s
            LEFT JOIN supports sup ON sup.statement_id = s.statement_id
            LEFT JOIN quotes q ON q.quote_id = s.quote_id
            LEFT JOIN papers p ON p.paper_id = s.paper_id
            WHERE {like_clauses}
            LIMIT ?
            """,
            [*params, max(limit * 2, 5)],
        ):
            item = _row_to_dict(row, boost=3.0)
            item["score"] = _like_score(terms, item.get("statement_text", "")) + 3.0
            rows.append(item)
        passage_like = " OR ".join(["LOWER(text) LIKE ?" for _ in terms])
        for row in db.execute(
            f"""
            SELECT 'text_chunk' AS result_kind, passage_id, paper_id, citation_key,
                   title AS paper_title, NULL AS year, locator, text AS statement_text,
                   text AS quote, char_start, char_end, text_sha256, 0.0 AS rank
            FROM passages WHERE {passage_like} LIMIT ?
            """,
            [*params, max(limit * 2, 5)],
        ):
            item = _row_to_dict(row, boost=1.0)
            item["score"] = _like_score(terms, item.get("statement_text", ""))
            rows.append(item)
        rows.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return rows

    def _insert_alias(
        self, db: sqlite3.Connection, paper_id: str, alias_type: str, alias_value: str
    ) -> None:
        value = _normalize_alias(alias_type, alias_value)
        if not value:
            return
        # Keep the first owner of an alias; duplicate imports then resolve to that paper.
        db.execute(
            "INSERT OR IGNORE INTO paper_aliases(alias_type, alias_value, paper_id) VALUES (?, ?, ?)",
            (alias_type, value, paper_id),
        )

    def _paper_aliases(
        self, paper: PaperMetadata, *, pdf_sha: str = "", text_sha: str = ""
    ) -> list[tuple[str, str]]:
        aliases = [("citation_key", paper.citation_key)]
        if paper.doi:
            aliases.append(("doi", paper.doi.lower()))
        if paper.arxiv_id:
            aliases.append(("arxiv", _normalize_arxiv_key(paper.arxiv_id)))
        openalex = _openalex_id(paper)
        if openalex:
            aliases.append(("openalex", _normalize_openalex_key(openalex)))
        title = _normalize_title(paper.title)
        if title:
            aliases.append(("title", title))
        if pdf_sha:
            aliases.append(("pdf_sha256", pdf_sha))
        if text_sha:
            aliases.append(("text_sha256", text_sha))
        return aliases

    def _artifact_sha(self, path: str) -> str:
        if not path:
            return ""
        try:
            ref = self.store.artifact_ref(path)
        except Exception:
            return ""
        return ref.sha256 or ""


def _row_to_dict(row: sqlite3.Row, *, boost: float) -> dict[str, Any]:
    payload = dict(row)
    rank = float(payload.pop("rank", 0.0) or 0.0)
    # bm25() returns smaller/better values, commonly negative. Convert to positive-ish score.
    payload["score"] = round(boost + 1.0 / (1.0 + max(rank, 0.0)), 4)
    return payload


def _validate_or_locate_quote(text: str, quote: LiteratureQuote) -> tuple[bool, int | None, int | None]:
    if not text:
        return False, quote.char_start, quote.char_end
    q = quote.quote.strip()
    if quote.char_start is not None and quote.char_end is not None:
        start, end = quote.char_start, quote.char_end
        if 0 <= start <= end <= len(text):
            candidate = text[start:end]
            if _compact_ws(candidate) == _compact_ws(q):
                return True, start, end
    if q:
        pos = text.find(q)
        if pos >= 0:
            return True, pos, pos + len(q)
        compact_q = _compact_ws(q)
        # Best effort for PDF whitespace normalization.
        compact_text = _compact_ws(text)
        if compact_q and compact_q in compact_text:
            return False, quote.char_start, quote.char_end
    return False, quote.char_start, quote.char_end


def _statement_span_quality(statement: LiteratureStatement) -> bool:
    """Conservative syntax gate for promoting a regex span beyond a quote candidate."""
    text = re.sub(r"\s+", " ", statement.statement_text).strip()
    label = re.sub(r"\s+", " ", statement.label).strip()
    if not (25 <= len(text) <= 1200) or not label:
        return False
    if not re.fullmatch(
        r"(?i)(?:theorem|lemma|corollary|proposition|algorithm|definition|"
        r"hypothesis|assumption|conjecture)(?:\s+[0-9A-Za-z]+(?:\.[0-9A-Za-z]+)*)?",
        label,
    ):
        return False
    if not text.lower().startswith(label.lower()):
        return False
    remainder = text[len(label) :].lstrip()
    if re.match(r"\.\d", remainder):  # The regex captured `Theorem 1` from `Theorem 1.3`.
        return False
    suspicious_prefix = remainder[:120].lower()
    if any(token in suspicious_prefix for token in ["figure ", "this proves", "▶"]):
        return False
    if text.endswith(":") or "following statement" in text[-120:].lower():
        return False
    return True


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _normalize_alias(alias_type: str, alias_value: str) -> str:
    if alias_type == "title":
        return _normalize_title(alias_value)
    if alias_type == "doi":
        return alias_value.strip().lower()
    if alias_type == "arxiv":
        return _normalize_arxiv_key(alias_value)
    if alias_type == "openalex":
        return _normalize_openalex_key(alias_value)
    return alias_value.strip()


def _normalize_arxiv_key(value: str) -> str:
    return value.strip().removeprefix("arXiv:").removesuffix(".pdf").lower()


def _normalize_openalex_key(value: str) -> str:
    return value.strip().rstrip("/").rsplit("/", 1)[-1].lower()


def _openalex_id(paper: PaperMetadata) -> str:
    for url in paper.source_urls:
        if "openalex.org/W" in url:
            return url.rstrip("/").rsplit("/", 1)[-1]
    return ""


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _terms(text: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z0-9_\\-]{3,}", text.lower()) if t not in _STOP]


_STOP = {"and", "are", "for", "from", "into", "that", "the", "then", "this", "with", "where"}


def _fts_query(query: str) -> str:
    terms = _terms(query)
    if not terms:
        return ""
    # OR is more robust for exploratory literature lookup than AND.
    return " OR ".join(f'"{term}"' for term in terms[:12])


def _like_score(terms: list[str], text: str) -> float:
    lowered = text.lower()
    return sum(1.0 for term in terms if term in lowered)


def _compact_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class _Chunk:
    def __init__(self, text: str, start: int, end: int, locator: str = ""):
        self.text = text
        self.start = start
        self.end = end
        self.locator = locator


def _chunks(text: str, *, size: int = 1600, overlap: int = 200) -> Iterable[_Chunk]:
    page_positions = list(re.finditer(r"--- page (\d+) ---", text, flags=re.IGNORECASE))
    if len(text) <= size:
        yield _Chunk(text=text.strip(), start=0, end=len(text), locator=_page_locator(page_positions, 0))
        return
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
            if boundary > start + size // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            yield _Chunk(text=chunk, start=start, end=end, locator=_page_locator(page_positions, start))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)


def _page_locator(page_positions: list[re.Match[str]], char_pos: int) -> str:
    page = ""
    for match in page_positions:
        if match.start() <= char_pos:
            page = match.group(1)
        else:
            break
    return f"page {page}" if page else ""
