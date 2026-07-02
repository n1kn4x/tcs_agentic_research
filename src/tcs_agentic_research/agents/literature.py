"""Filesystem-backed literature researcher.

This module intentionally separates provenance-bearing literature records from model summaries.
Claims attributed to papers are stored with citation keys and should be checked by critics before
being used as established facts.
"""

from __future__ import annotations

import json

from ..artifact_store import ArtifactStore
from ..llm import LLMRouter
from ..prompt_loader import render_prompt
from ..schemas import LiteratureExtract, PaperMetadata


class LiteratureResearcher:
    def __init__(self, store: ArtifactStore, router: LLMRouter, *, prompt_dir: str | None = None):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir

    def import_paper(self, paper: PaperMetadata) -> PaperMetadata:
        self.store.append_jsonl("LiteratureDB/papers.jsonl", paper)
        return paper

    def extract_from_text(
        self, *, citation_key: str, paper_text: str, nomenclature_yaml: str | None = None
    ) -> LiteratureExtract:
        mock_output = LiteratureExtract(
            citation_key=citation_key,
            provenance_notes="Dry-run mock extraction recorded no claims; human or LLM extraction required.",
        )
        messages = [
            {"role": "system", "content": render_prompt("literature_researcher", override_dir=self.prompt_dir)},
            {
                "role": "user",
                "content": (
                    f"Citation key: {citation_key}\n"
                    f"Nomenclature:\n{nomenclature_yaml or self.store.read_text(ArtifactStore.NOMENCLATURE)}\n\n"
                    f"Paper text/excerpt:\n{paper_text[:50000]}"
                ),
            },
        ]
        extract = self.router.complete_structured(
            task_type="literature_extraction",
            messages=messages,
            schema=LiteratureExtract,
            mock_output=mock_output if self.router.dry_run else None,
        )
        self.store.append_jsonl("LiteratureDB/extracted_claims.jsonl", extract)
        for symbol, canonical in extract.notation_mappings.items():
            self.store.append_jsonl(
                "LiteratureDB/notation_mappings.jsonl",
                {"citation_key": citation_key, "symbol": symbol, "canonical": canonical},
            )
        if extract.extracted_claims:
            self.store.append_claims(extract.extracted_claims)
        return extract

    def query_local(self, query: str, *, limit: int = 10) -> list[dict[str, object]]:
        """Simple keyword query over local literature JSONL records.

        This is deliberately transparent. A production deployment can replace it with a vector
        index while keeping JSONL provenance as canonical state.
        """
        terms = [t.lower() for t in query.split() if len(t) > 2]
        results: list[dict[str, object]] = []
        for rel in ["LiteratureDB/papers.jsonl", "LiteratureDB/extracted_claims.jsonl"]:
            for record in self.store.read_jsonl(rel):
                blob = json.dumps(record).lower()
                score = sum(1 for t in terms if t in blob)
                if score:
                    results.append({"score": score, "source": rel, "record": record})
        results.sort(key=lambda r: int(r["score"]), reverse=True)
        return results[:limit]
