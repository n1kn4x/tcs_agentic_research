"""Deterministic hybrid retrieval for local DAG lemmas and Lean library source."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from ..artifact_store import ArtifactStore
from ..schemas import LeanStatement
from .graph import ProofGraph
from .models import LeanDiagnostic, OrNode, RetrievalHit

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_'.]*")
_DECL_LINE_RE = re.compile(
    r"^\s*(?:private\s+|protected\s+|noncomputable\s+)*"
    r"(?:theorem|lemma|def|abbrev|structure|class|instance)\s+"
    r"([A-Za-z_][A-Za-z0-9_'.]*)\b(.*)$"
)


class LeanRetriever:
    """Search exact proved graph nodes first, then the pinned workspace's Lean source.

    This intentionally avoids an LLM-generated theorem-name search.  Results are selected and
    ranked by stable token overlap, compiler-error symbols, import/module proximity, and proof
    status.  A future semantic index can be added behind the same ``search`` contract.
    """

    def __init__(self, store: ArtifactStore, graph: ProofGraph):
        self.store = store
        self.graph = graph
        self.project_root = store.resolve("LeanProject")

    def search(
        self,
        goal: LeanStatement,
        *,
        informal_queries: Sequence[str] = (),
        diagnostics: Sequence[LeanDiagnostic] = (),
        exclude_or_ids: Sequence[str] = (),
        limit: int = 16,
    ) -> list[RetrievalHit]:
        query = " ".join(
            [goal.name, goal.statement, *informal_queries, *_diagnostic_queries(diagnostics)]
        )
        terms = _query_terms(query)
        hits = self._graph_hits(goal, terms, exclude_or_ids)
        hits.extend(self._source_hits(goal, terms, limit=max(limit * 3, 30)))
        deduplicated: dict[tuple[str, str], RetrievalHit] = {}
        for hit in hits:
            key = (hit.name, hit.statement)
            previous = deduplicated.get(key)
            if previous is None or hit.score > previous.score:
                deduplicated[key] = hit
        return sorted(
            deduplicated.values(), key=lambda item: (-item.score, item.name, item.source)
        )[:limit]

    def _graph_hits(
        self, goal: LeanStatement, terms: set[str], exclude_or_ids: Sequence[str]
    ) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for node in self.graph.proved_nodes(exclude=exclude_or_ids, limit=300):
            score = _overlap_score(terms, node.goal.name + " " + node.goal.statement)
            if _head_symbols(goal.statement) & _head_symbols(node.goal.statement):
                score += 4.0
            if score <= 0:
                continue
            hits.append(
                RetrievalHit(
                    name=node.goal.name,
                    statement=node.goal.statement,
                    source="LEAP verified DAG",
                    score=score + 20.0,
                    module="generated",
                    proved_or_id=node.node_id,
                )
            )
        return hits

    def _source_hits(
        self, goal: LeanStatement, terms: set[str], *, limit: int
    ) -> list[RetrievalHit]:
        roots = self._source_roots()
        if not roots or not terms:
            return []
        lines = self._ripgrep(roots, terms, limit=limit)
        hits: list[RetrievalHit] = []
        import_tokens = {token.lower() for item in goal.imports for token in item.split(".")}
        source_cache: dict[Path, list[str]] = {}
        for path, line_no, line in lines:
            declaration = _DECL_LINE_RE.match(line)
            if declaration is None:
                continue
            short_name, tail = declaration.groups()
            source_lines = source_cache.setdefault(
                path, path.read_text(encoding="utf-8", errors="ignore").splitlines()
            )
            name = _qualified_declaration_name(source_lines, line_no, short_name)
            statement = _declaration_type_excerpt(source_lines, line_no, tail)
            score = _overlap_score(terms, name + " " + statement)
            module = _module_name(path)
            module_tokens = {token.lower() for token in module.split(".")}
            score += 1.5 * len(import_tokens & module_tokens)
            if _head_symbols(goal.statement) & _head_symbols(statement):
                score += 3.0
            hits.append(
                RetrievalHit(
                    name=name,
                    statement=statement[:2000],
                    source=f"{path}:{line_no}",
                    score=score,
                    module=module,
                )
            )
        return hits

    def _source_roots(self) -> list[Path]:
        roots: list[Path] = []
        local = self.project_root / "TCSResearch"
        if local.exists():
            roots.append(local)
        packages = self.project_root / ".lake" / "packages"
        if packages.exists():
            for package in sorted(packages.iterdir()):
                for candidate in [package / "Mathlib", package]:
                    if candidate.exists() and candidate not in roots:
                        roots.append(candidate)
                        break
        return roots

    def _ripgrep(
        self, roots: Sequence[Path], terms: set[str], *, limit: int
    ) -> list[tuple[Path, int, str]]:
        useful = sorted(
            (term for term in terms if len(term) >= 3 and term not in _STOP_WORDS),
            key=lambda item: (-len(item), item),
        )[:12]
        if not useful:
            return []
        wanted = "|".join(re.escape(term) for term in useful)
        pattern = (
            r"^\s*(?:(?:private|protected|noncomputable)\s+)*"
            r"(?:theorem|lemma|def|abbrev|structure|class|instance)\s+.*(?:"
            + wanted
            + r")"
        )
        if shutil.which("rg") is not None:
            command = [
                "rg",
                "-i",
                "-n",
                "--no-heading",
                "--color=never",
                "--max-count",
                "2",
                "--max-filesize",
                "2M",
                "--glob",
                "*.lean",
                "--glob",
                "!**/.lake/build/**",
                "--glob",
                "!**/Generated/**",
                "--glob",
                "!**/LEAP/**",
                pattern,
                *[str(root) for root in roots],
            ]
            try:
                completed = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=30,
                )
                rows: list[tuple[Path, int, str]] = []
                for output_line in completed.stdout.splitlines():
                    parts = output_line.split(":", 2)
                    if len(parts) != 3 or not parts[1].isdigit():
                        continue
                    rows.append((Path(parts[0]), int(parts[1]), parts[2]))
                    if len(rows) >= limit:
                        break
                return rows
            except (OSError, subprocess.TimeoutExpired):
                pass
        return self._python_scan(roots, useful, limit=limit)

    @staticmethod
    def _python_scan(
        roots: Sequence[Path], terms: Sequence[str], *, limit: int
    ) -> list[tuple[Path, int, str]]:
        lowered = [term.lower() for term in terms]
        rows: list[tuple[Path, int, str]] = []
        for root in roots:
            for path in sorted(root.rglob("*.lean")):
                if "Generated" in path.parts or "LEAP" in path.parts:
                    continue
                try:
                    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
                except OSError:
                    continue
                for line_no, line in enumerate(lines, start=1):
                    if _DECL_LINE_RE.match(line) and any(term in line.lower() for term in lowered):
                        rows.append((path, line_no, line))
                        if len(rows) >= limit:
                            return rows
        return rows


def proved_support_nodes(graph: ProofGraph, hits: Sequence[RetrievalHit]) -> list[OrNode]:
    """Resolve only Lean-verified DAG hits, preserving retrieval order."""
    result: list[OrNode] = []
    seen: set[str] = set()
    for hit in hits:
        if not hit.proved_or_id or hit.proved_or_id in seen:
            continue
        node = graph.get_or(hit.proved_or_id)
        if node.status.value == "proved":
            seen.add(node.node_id)
            result.append(node)
    return result


def _query_terms(text: str) -> set[str]:
    result: set[str] = set()
    for token in _WORD_RE.findall(text):
        lower = token.lower().strip("'")
        if len(lower) >= 2 and lower not in _STOP_WORDS:
            result.add(lower)
        for part in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", token):
            if len(part) >= 3 and part.lower() not in _STOP_WORDS:
                result.add(part.lower())
    return result


def _overlap_score(terms: set[str], text: str) -> float:
    tokens = _query_terms(text)
    overlap = terms & tokens
    return float(sum(1.0 + min(len(term), 20) / 20 for term in overlap))


def _head_symbols(text: str) -> set[str]:
    return {
        token.lower()
        for token in _WORD_RE.findall(text)
        if "." in token or (token and token[0].isupper())
    }


def _diagnostic_queries(diagnostics: Sequence[LeanDiagnostic]) -> list[str]:
    queries: list[str] = []
    patterns = [
        r"unknown (?:constant|identifier)\s+['`]?([^'`\s]+)",
        r"failed to synthesize\s+([^\n]+)",
        r"application type mismatch[^\n]*",
    ]
    for diagnostic in diagnostics:
        for pattern in patterns:
            queries.extend(re.findall(pattern, diagnostic.message, flags=re.IGNORECASE))
    return queries


def _qualified_declaration_name(lines: Sequence[str], line_no: int, name: str) -> str:
    frames: list[tuple[str, list[str]]] = []
    namespace_re = re.compile(
        r"^\s*namespace\s+([A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*)"
    )
    section_re = re.compile(r"^\s*section(?:\s+[A-Za-z_][A-Za-z0-9_']*)?\s*$")
    end_re = re.compile(r"^\s*end(?:\s+[A-Za-z_][A-Za-z0-9_'.]*)?\s*$")
    for line in lines[: max(0, line_no - 1)]:
        if match := namespace_re.match(line):
            frames.append(("namespace", match.group(1).split(".")))
        elif section_re.match(line):
            frames.append(("section", []))
        elif end_re.match(line) and frames:
            frames.pop()
    prefix = [part for kind, parts in frames if kind == "namespace" for part in parts]
    return ".".join([*prefix, name]) if prefix else name


def _declaration_type_excerpt(lines: Sequence[str], line_no: int, first_tail: str) -> str:
    pieces = [first_tail.strip()]
    for line in lines[line_no : line_no + 12]:
        if _DECL_LINE_RE.match(line) or re.match(r"^\s*(?:namespace|section|end)\b", line):
            break
        pieces.append(line.strip())
        combined = " ".join(pieces)
        if ":=" in combined or re.search(r"\bwhere\b", combined):
            break
    text = " ".join(piece for piece in pieces if piece)
    text = text.split(":=", 1)[0].strip()
    return text[:2000]


def _module_name(path: Path) -> str:
    parts = list(path.with_suffix("").parts)
    for marker in ["Mathlib", "TCSResearch", "Lean", "Init"]:
        if marker in parts:
            return ".".join(parts[parts.index(marker) :])
    return path.stem


_STOP_WORDS = {
    "theorem",
    "lemma",
    "proof",
    "goal",
    "have",
    "show",
    "from",
    "with",
    "this",
    "that",
    "forall",
    "prop",
    "type",
    "true",
    "false",
    "lean",
    "exact",
    "apply",
}
