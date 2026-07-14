"""Command line interface for the agentic TCS research system."""

from __future__ import annotations

import argparse
import json
import sys

from .agents.experiment import ExperimentAgent
from .agents.initialization import InitializationAgent
from .agents.literature import LiteratureResearcher
from .agents.theorem_prover import TheoremProverAgent
from .artifact_store import ArtifactStore
from .graph import ResearchGraph
from .llm import LLMRouter
from .schemas import LeanStatement, PaperMetadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tcs-research")
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init", help="Create or update initialization artifacts")
    _add_common(init_p)

    run_p = sub.add_parser("run", help="Run/resume the LangGraph research loop")
    _add_common(run_p)
    run_p.add_argument("--max-iterations", type=int, default=1)
    run_p.add_argument("--max-proposal-revisions", type=int, default=2)
    run_p.add_argument("--thread-id", default="default")

    status_p = sub.add_parser("status", help="Show compact workspace status")
    status_p.add_argument("--workspace", default=".")
    status_p.add_argument("--ledger-tail", type=int, default=5)

    prove_p = sub.add_parser("prove", help="Submit a Lean theorem/lemma statement to LEAP")
    _add_common(prove_p)
    prove_p.add_argument("--name", required=True)
    prove_p.add_argument(
        "--statement",
        required=True,
        help="Lean proposition/type after the colon, e.g. `∀ n : Nat, n = n`.",
    )
    prove_p.add_argument("--import", dest="imports", action="append", default=["TCSResearch.Basic"])
    prove_p.add_argument("--namespace", default="TCSResearch")

    lit_p = sub.add_parser("literature", help="Import, extract, query, and test LiteratureDB")
    lit_sub = lit_p.add_subparsers(dest="literature_command", required=True)

    lit_url = lit_sub.add_parser("import-url", help="Import a paper from URL/DOI/arXiv/PDF")
    _add_common(lit_url)
    lit_url.add_argument("--url", required=True)
    lit_url.add_argument("--citation-key")
    lit_url.add_argument("--title")
    lit_url.add_argument("--doi")
    lit_url.add_argument("--extract-text", action="store_true")

    lit_arxiv = lit_sub.add_parser("import-arxiv", help="Import an arXiv paper and PDF")
    _add_common(lit_arxiv)
    lit_arxiv.add_argument("--arxiv-id", required=True)
    lit_arxiv.add_argument("--citation-key")
    lit_arxiv.add_argument("--extract-text", action="store_true")

    lit_doi = lit_sub.add_parser("import-doi", help="Import DOI metadata and any direct PDF")
    _add_common(lit_doi)
    lit_doi.add_argument("--doi", required=True)
    lit_doi.add_argument("--citation-key")
    lit_doi.add_argument("--extract-text", action="store_true")

    lit_pdf = lit_sub.add_parser(
        "extract-pdf-text", help="Extract text from an imported or explicit PDF"
    )
    _add_common(lit_pdf)
    lit_pdf.add_argument("--citation-key")
    lit_pdf.add_argument("--paper-id")
    lit_pdf.add_argument("--pdf-path")

    lit_extract = lit_sub.add_parser(
        "extract", help="Extract theorem/algorithm statements from an imported paper"
    )
    _add_common(lit_extract)
    lit_extract.add_argument("--citation-key")
    lit_extract.add_argument("--paper-id")

    lit_query = lit_sub.add_parser(
        "query", help="Answer a local literature query with mapped notation"
    )
    _add_common(lit_query)
    lit_query.add_argument("--query", required=True)
    lit_query.add_argument("--limit", type=int, default=10)

    lit_search = lit_sub.add_parser("search", help="Search OpenAlex and queue candidates")
    _add_common(lit_search)
    lit_search.add_argument("--query", required=True)
    lit_search.add_argument("--limit", type=int, default=10)

    lit_related = lit_sub.add_parser(
        "discover-related", help="Queue papers cited by or citing an imported paper"
    )
    _add_common(lit_related)
    lit_related.add_argument("--citation-key", required=True)
    lit_related.add_argument("--direction", choices=["cited", "cited_by", "both"], default="both")
    lit_related.add_argument("--limit", type=int, default=20)

    lit_import_candidate = lit_sub.add_parser(
        "import-candidate", help="Import a queued OpenAlex candidate"
    )
    _add_common(lit_import_candidate)
    lit_import_candidate.add_argument("--candidate-id", required=True)
    lit_import_candidate.add_argument("--extract-text", action="store_true")

    lit_test = lit_sub.add_parser("test", help="Run a LiteratureDB smoke test")
    _add_common(lit_test)
    lit_test.add_argument("--query", default="sample theorem equality algorithm")
    lit_test.add_argument("--citation-key", default="literature_smoke_test")

    exp_p = sub.add_parser("experiment", help="Manage and run Dockerized pi experiments")
    exp_sub = exp_p.add_subparsers(dest="experiment_command", required=True)

    exp_start = exp_sub.add_parser("start", help="Build/start the project experimenter container")
    _add_common(exp_start)

    exp_status = exp_sub.add_parser("status", help="Show project experimenter container status")
    _add_common(exp_status)

    exp_stop = exp_sub.add_parser("stop", help="Stop the project experimenter container")
    _add_common(exp_stop)
    exp_stop.add_argument("--remove", action="store_true", help="Remove the stopped container")

    exp_reset = exp_sub.add_parser(
        "reset", help="Remove the project experimenter container and .experimenter writable state"
    )
    _add_common(exp_reset)

    exp_run = exp_sub.add_parser("run", help="Run an experiment with Dockerized pi")
    _add_common(exp_run)
    exp_run.add_argument("--description", required=True)
    exp_run.add_argument("--name", default="experiment")
    exp_run.add_argument("--supports-claim-id", action="append", default=[])
    exp_run.add_argument("--timeout-seconds", type=int)

    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            return _cmd_init(args)
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "status":
            return _cmd_status(args)
        if args.command == "prove":
            return _cmd_prove(args)
        if args.command == "literature":
            return _cmd_literature(args)
        if args.command == "experiment":
            return _cmd_experiment(args)
    except Exception as exc:  # noqa: BLE001 - CLI should surface actionable errors
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--config", help="Router config YAML; see config.example.yml")
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not call vLLM; use deterministic mock outputs"
    )
    parser.add_argument("--prompt-dir", help="Override prompt directory")


def _cmd_init(args: argparse.Namespace) -> int:
    store = ArtifactStore(args.workspace)
    store.initialize_layout()
    router = LLMRouter.from_config_file(args.config, store=store, dry_run=args.dry_run)
    state = InitializationAgent(
        store, router, prompt_dir=args.prompt_dir
    ).initialize_interactively()
    print(f"Initialized workspace: {store.root}")
    print(f"Task ID: {state.task_id}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    store = ArtifactStore(args.workspace)
    store.initialize_layout()
    if store.load_state() is None or not store.exists(ArtifactStore.RESEARCH_TASK):
        raise RuntimeError("Workspace is uninitialized; run `tcs-research init` first.")
    graph = ResearchGraph(
        workspace=args.workspace,
        config_path=args.config,
        dry_run=args.dry_run,
        prompt_dir=args.prompt_dir,
        max_proposal_revisions=int(args.max_proposal_revisions),
    )
    result = graph.run(max_iterations=args.max_iterations, thread_id=args.thread_id)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    store = ArtifactStore(args.workspace)
    store.initialize_layout()
    state = store.load_state()
    print(f"Workspace: {store.root}")
    if state is None:
        print("No ResearchState.json found. Run `tcs-research init`.")
        return 0
    print(state.model_dump_json(indent=2))
    print("\nRecent claims:")
    print(
        json.dumps(store.read_jsonl(ArtifactStore.CLAIM_LEDGER, limit=args.ledger_tail), indent=2)
    )
    print("\nRecent proposals:")
    print(
        json.dumps(
            store.read_jsonl(ArtifactStore.PROPOSAL_LEDGER, limit=args.ledger_tail), indent=2
        )
    )
    print("\nObligation board:")
    board = store.load_obligation_board()
    print(
        json.dumps(
            {
                "candidate_claims": [claim.model_dump(mode="json") for claim in board.candidate_claims[-args.ledger_tail :]],
                "open_obligations": [
                    obligation.model_dump(mode="json")
                    for obligation in board.obligations
                    if obligation.status == "open"
                ][: args.ledger_tail],
                "recent_runs": [run.model_dump(mode="json") for run in board.runs[-args.ledger_tail :]],
            },
            indent=2,
        )
    )
    return 0


def _cmd_prove(args: argparse.Namespace) -> int:
    store = ArtifactStore(args.workspace)
    store.initialize_layout()
    router = LLMRouter.from_config_file(args.config, store=store, dry_run=args.dry_run)
    result = TheoremProverAgent(store, router, prompt_dir=args.prompt_dir).prove(
        LeanStatement(
            name=args.name,
            statement=args.statement,
            imports=args.imports,
            namespace=args.namespace,
        ),
        context=(
            store.read_text(ArtifactStore.RESEARCH_TASK)
            if store.exists(ArtifactStore.RESEARCH_TASK)
            else ""
        ),
    )
    print(result.model_dump_json(indent=2))
    return 0


def _cmd_literature(args: argparse.Namespace) -> int:
    store = ArtifactStore(args.workspace)
    store.initialize_layout()
    router = LLMRouter.from_config_file(args.config, store=store, dry_run=args.dry_run)
    agent = LiteratureResearcher(store, router, prompt_dir=args.prompt_dir)

    if args.literature_command == "import-url":
        paper = agent.import_url(
            args.url,
            citation_key=args.citation_key,
            title=args.title,
            doi=args.doi,
            extract_text=args.extract_text,
        )
        print(paper.model_dump_json(indent=2))
        return 0
    if args.literature_command == "import-arxiv":
        paper = agent.import_arxiv(
            args.arxiv_id, citation_key=args.citation_key, extract_text=args.extract_text
        )
        print(paper.model_dump_json(indent=2))
        return 0
    if args.literature_command == "import-doi":
        paper = agent.import_doi(
            args.doi, citation_key=args.citation_key, extract_text=args.extract_text
        )
        print(paper.model_dump_json(indent=2))
        return 0
    if args.literature_command == "extract-pdf-text":
        text_path = agent.extract_pdf_text(
            args.pdf_path, citation_key=args.citation_key, paper_id=args.paper_id
        )
        print(json.dumps({"text_path": text_path}, indent=2))
        return 0
    if args.literature_command == "extract":
        extract = agent.extract_paper(citation_key=args.citation_key, paper_id=args.paper_id)
        print(extract.model_dump_json(indent=2))
        return 0
    if args.literature_command == "query":
        answer = agent.answer_query(args.query, limit=args.limit)
        print(answer.model_dump_json(indent=2))
        return 0
    if args.literature_command == "search":
        candidates = agent.search_papers(args.query, limit=args.limit)
        print(json.dumps([c.model_dump(mode="json") for c in candidates], indent=2))
        return 0
    if args.literature_command == "discover-related":
        candidates = agent.discover_related(
            citation_key=args.citation_key,
            direction=args.direction,
            limit=args.limit,
        )
        print(json.dumps([c.model_dump(mode="json") for c in candidates], indent=2))
        return 0
    if args.literature_command == "import-candidate":
        paper = agent.import_candidate(args.candidate_id, extract_text=args.extract_text)
        print(paper.model_dump_json(indent=2))
        return 0
    if args.literature_command == "test":
        payload = _run_literature_smoke_test(store, agent, args.citation_key, args.query)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    raise RuntimeError(f"Unknown literature subcommand: {args.literature_command}")


def _cmd_experiment(args: argparse.Namespace) -> int:
    store = ArtifactStore(args.workspace)
    store.initialize_layout()
    router = LLMRouter.from_config_file(args.config, store=store, dry_run=args.dry_run)
    agent = ExperimentAgent(store, router.experimenter)

    if args.experiment_command == "start":
        print(json.dumps(agent.ensure_container(), indent=2, sort_keys=True))
        return 0
    if args.experiment_command == "status":
        print(json.dumps(agent.status(), indent=2, sort_keys=True))
        return 0
    if args.experiment_command == "stop":
        agent.stop_container(remove=args.remove)
        print(json.dumps({"status": "stopped", "removed": bool(args.remove)}, indent=2))
        return 0
    if args.experiment_command == "reset":
        agent.reset_container()
        print(json.dumps({"status": "reset"}, indent=2))
        return 0
    if args.experiment_command == "run":
        result = agent.run_experiment(
            description=args.description,
            name=args.name,
            supports_claim_ids=args.supports_claim_id,
            timeout_seconds=args.timeout_seconds,
        )
        print(result.model_dump_json(indent=2))
        return 0
    raise RuntimeError(f"Unknown experiment subcommand: {args.experiment_command}")


def _run_literature_smoke_test(
    store: ArtifactStore, agent: LiteratureResearcher, citation_key: str, query: str
) -> dict[str, object]:
    text = """# Literature Smoke Test

Theorem 1. Let n be a natural number. In the standard model, the identity map on n
objects returns n objects and preserves equality.

Algorithm 1. Input a natural number n. Return n. The algorithm uses one step in the
unit-cost RAM model.
"""
    text_ref = store.write_text(f"LiteratureDB/papers/{citation_key}/paper.txt", text)
    paper = PaperMetadata(
        citation_key=citation_key,
        title="Literature Smoke Test",
        source_type="manual",
        text_path=text_ref.path,
        artifact_refs=[text_ref],
    )
    paper = agent.import_paper(paper)
    extract = agent.extract_from_text(
        citation_key=citation_key,
        paper_text=text,
        paper_id=paper.paper_id,
        text_artifact_path=text_ref.path,
    )
    answer = agent.answer_query(query, limit=5)
    return {
        "paper": paper.model_dump(mode="json"),
        "extract": extract.model_dump(mode="json"),
        "answer": answer.model_dump(mode="json"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
