from __future__ import annotations

from tcs_agentic_research.agents.literature import LiteratureResearcher
from tcs_agentic_research.artifact_store import ArtifactStore
from tcs_agentic_research.llm import LLMRouter
from tcs_agentic_research.schemas import ModelProfile, RouterSettings


def test_deterministic_extraction_preserves_exact_quote_provenance(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize_layout()
    router = LLMRouter(
        RouterSettings(
            default_profile="reasoning",
            repair_profile="reasoning",
            profiles={"reasoning": ModelProfile(model="not-called")},
        ),
        store=store,
        dry_run=True,
    )
    text = (
        "Theorem 1 (Fixture lower bound). Every fixture algorithm requires at least one step.\n"
        "Proof. The input must be inspected.\n"
    )
    ref = store.write_text("LiteratureDB/papers/fixture/paper.txt", text)
    agent = LiteratureResearcher(store, router)

    extract = agent.extract_from_text(
        citation_key="fixture",
        paper_text=text,
        paper_id="paper_fixture",
        text_artifact_path=ref.path,
        use_llm=False,
    )
    answer = agent.answer_query("fixture lower bound one step", limit=5)

    statements = [
        *extract.theorem_statements,
        *extract.lower_bound_statements,
        *extract.algorithm_statements,
    ]
    assert statements
    assert any(quote.validated for statement in statements for quote in statement.provenance)
    assert answer.results
    assert any(
        quote.source_sha256 == ref.sha256
        for result in answer.results
        for quote in result.provenance
    )
