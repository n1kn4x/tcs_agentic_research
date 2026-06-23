"""Transparent Lean lemma retrieval for LEAP.

This lightweight component indexes local project declarations. It can be replaced by a richer
Mathlib search service while preserving artifact-driven behavior.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..artifact_store import ArtifactStore
from ..schemas import StrictModel

_DECL_RE = re.compile(r"^\s*(?:theorem|lemma|def)\s+([A-Za-z_][A-Za-z0-9_'.]*)\b(.*)")


class LeanSearchHit(StrictModel):
    name: str
    path: str
    line: int
    declaration: str
    score: int = 0


class LeanSearchIndex:
    def __init__(self, store: ArtifactStore):
        self.store = store

    def search(self, query: str, *, limit: int = 20) -> list[LeanSearchHit]:
        terms = [t.lower() for t in re.findall(r"[A-Za-z0-9_']+", query) if len(t) > 2]
        hits: list[LeanSearchHit] = []
        root = self.store.resolve("LeanProject")
        if not root.exists():
            return []
        for path in root.rglob("*.lean"):
            try:
                rel = self.store.relpath(path)
            except ValueError:
                continue
            for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                match = _DECL_RE.match(line)
                if not match:
                    continue
                blob = (match.group(1) + " " + match.group(2)).lower()
                score = sum(1 for term in terms if term in blob)
                if score or not terms:
                    hits.append(
                        LeanSearchHit(
                            name=match.group(1),
                            path=rel,
                            line=line_no,
                            declaration=line.strip(),
                            score=score,
                        )
                    )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    def render_hits(self, query: str, *, limit: int = 20) -> str:
        hits = self.search(query, limit=limit)
        if not hits:
            return "No local Lean declarations found."
        return "\n".join(
            f"- {hit.name} ({hit.path}:{hit.line}, score={hit.score}): {hit.declaration}"
            for hit in hits
        )
