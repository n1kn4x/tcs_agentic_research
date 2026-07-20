"""Command line interface for the bounded research engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .agents.experiment import ExperimentAgent
from .agents.literature import LiteratureResearcher
from .agents.theorem_prover import TheoremProverAgent
from .artifact_store import ArtifactStore
from .engine import ResearchEngine
from .llm import LLMRouter
from .schemas import ExperimentProgram, LeanStatement
from .workflow import _validate_experiment_program


LEGACY_FILES = (
    "Nomenclature.yml",
    "ResearchState.json",
    "ObligationBoard.json",
    "ClaimLedger.jsonl",
    "ProposalLedger.jsonl",
    "ModelCallLedger.jsonl",
    "GraphCheckpoints.sqlite",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tcs-research")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Plan and execute bounded work steps")
    _add_common(run_parser)
    run_parser.add_argument("--max-steps", type=int, default=1)

    status_parser = sub.add_parser("status", help="Show compact machine-readable progress")
    status_parser.add_argument("--workspace", default=".")

    replan_parser = sub.add_parser("replan", help="Request another bounded planning round")
    _add_common(replan_parser)

    doctor_parser = sub.add_parser("doctor", help="Inspect stale legacy artifacts")
    doctor_parser.add_argument("--workspace", default=".")
    doctor_parser.add_argument("--clean-legacy", action="store_true")

    prove_parser = sub.add_parser("prove", help="Run or resume a persistent LEAP proof search")
    _add_common(prove_parser)
    prove_parser.add_argument("--name", required=True)
    prove_parser.add_argument("--statement", required=True)
    prove_parser.add_argument("--import", dest="imports", action="append", default=[])
    prove_parser.add_argument("--namespace", default="TCSResearch")
    prove_parser.add_argument("--max-model-calls", type=int)
    prove_parser.add_argument("--max-wall-seconds", type=int)

    literature_parser = sub.add_parser("literature", help="Operate the local literature store")
    literature_sub = literature_parser.add_subparsers(dest="literature_command", required=True)
    _literature_commands(literature_sub)

    experiment_parser = sub.add_parser("experiment", help="Manage bounded Docker experiments")
    experiment_sub = experiment_parser.add_subparsers(dest="experiment_command", required=True)
    _experiment_commands(experiment_sub)

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            engine = ResearchEngine(
                workspace=args.workspace,
                config_path=args.config,
                dry_run=args.dry_run,
                prompt_dir=args.prompt_dir,
            )
            print(json.dumps(engine.run(max_steps=args.max_steps), indent=2, ensure_ascii=False))
            return 0
        if args.command == "status":
            engine = ResearchEngine(workspace=args.workspace, dry_run=True)
            print(json.dumps(engine.status(), indent=2, ensure_ascii=False))
            return 0
        if args.command == "replan":
            engine = ResearchEngine(
                workspace=args.workspace,
                config_path=args.config,
                dry_run=args.dry_run,
                prompt_dir=args.prompt_dir,
            )
            engine.replan()
            print(json.dumps(engine.status(), indent=2, ensure_ascii=False))
            return 0
        if args.command == "doctor":
            return _doctor(args)
        if args.command == "prove":
            return _prove(args)
        if args.command == "literature":
            return _literature(args)
        if args.command == "experiment":
            return _experiment(args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary prints the actionable failure
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 1


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--config", help="YAML configuration; see config.example.yml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prompt-dir")


def _literature_commands(sub: Any) -> None:
    search = sub.add_parser("search", help="Queue OpenAlex candidates")
    _add_common(search)
    search.add_argument("--query", required=True)
    search.add_argument("--limit", type=int, default=10)

    arxiv = sub.add_parser("import-arxiv", help="Import arXiv metadata/PDF")
    _add_common(arxiv)
    arxiv.add_argument("--arxiv-id", required=True)
    arxiv.add_argument("--citation-key")
    arxiv.add_argument("--extract-text", action="store_true")

    doi = sub.add_parser("import-doi", help="Import DOI metadata and available PDF")
    _add_common(doi)
    doi.add_argument("--doi", required=True)
    doi.add_argument("--citation-key")
    doi.add_argument("--extract-text", action="store_true")

    url = sub.add_parser("import-url", help="Import a URL/PDF")
    _add_common(url)
    url.add_argument("--url", required=True)
    url.add_argument("--citation-key")
    url.add_argument("--title")
    url.add_argument("--extract-text", action="store_true")

    candidate = sub.add_parser("import-candidate", help="Import a queued candidate")
    _add_common(candidate)
    candidate.add_argument("--candidate-id", required=True)
    candidate.add_argument("--extract-text", action="store_true")

    extract = sub.add_parser("extract", help="Deterministically extract exact statements")
    _add_common(extract)
    extract.add_argument("--citation-key")
    extract.add_argument("--paper-id")

    query = sub.add_parser("query", help="Query local statements/passages")
    _add_common(query)
    query.add_argument("--query", required=True)
    query.add_argument("--limit", type=int, default=10)

    rebuild = sub.add_parser("rebuild-index", help="Rebuild the materialized SQLite index")
    _add_common(rebuild)


def _experiment_commands(sub: Any) -> None:
    for name in ["start", "status", "reset"]:
        command = sub.add_parser(name)
        _add_common(command)
    stop = sub.add_parser("stop")
    _add_common(stop)
    stop.add_argument("--remove", action="store_true")
    run = sub.add_parser("run", help="Run an explicit Python script once")
    _add_common(run)
    run.add_argument("--script", required=True)
    run.add_argument("--description", required=True)
    run.add_argument("--name", default="experiment")
    run.add_argument("--seed", type=int, default=0)


def _router_and_store(args: argparse.Namespace) -> tuple[ArtifactStore, LLMRouter]:
    store = ArtifactStore(args.workspace)
    store.initialize_layout()
    router = LLMRouter.from_config_file(args.config, store=store, dry_run=args.dry_run)
    return store, router


def _prove(args: argparse.Namespace) -> int:
    store, router = _router_and_store(args)
    imports = args.imports or ["TCSResearch.Basic"]
    updates = {}
    if args.max_model_calls is not None:
        updates["max_model_calls_per_run"] = args.max_model_calls
    if args.max_wall_seconds is not None:
        updates["max_wall_seconds"] = args.max_wall_seconds
    settings = type(router.leap).model_validate(
        {**router.leap.model_dump(mode="python"), **updates}
    )
    with router.step_budget("manual_proof", max_calls=settings.max_model_calls_per_run):
        result = TheoremProverAgent(
            store, router, prompt_dir=args.prompt_dir, settings=settings
        ).prove(
            LeanStatement(
                name=args.name,
                statement=args.statement,
                imports=imports,
                namespace=args.namespace,
            ),
            context="Manual CLI proof request.",
        )
    print(result.model_dump_json(indent=2))
    return 0


def _literature(args: argparse.Namespace) -> int:
    store, router = _router_and_store(args)
    agent = LiteratureResearcher(store, router, prompt_dir=args.prompt_dir)
    command = args.literature_command
    if command == "search":
        candidates = agent.search_papers(args.query, limit=args.limit)
        print(json.dumps([item.model_dump(mode="json") for item in candidates], indent=2))
    elif command == "import-arxiv":
        paper = agent.import_arxiv(
            args.arxiv_id,
            citation_key=args.citation_key,
            extract_text=args.extract_text,
        )
        print(paper.model_dump_json(indent=2))
    elif command == "import-doi":
        paper = agent.import_doi(
            args.doi,
            citation_key=args.citation_key,
            extract_text=args.extract_text,
        )
        print(paper.model_dump_json(indent=2))
    elif command == "import-url":
        paper = agent.import_url(
            args.url,
            citation_key=args.citation_key,
            title=args.title,
            extract_text=args.extract_text,
        )
        print(paper.model_dump_json(indent=2))
    elif command == "import-candidate":
        paper = agent.import_candidate(args.candidate_id, extract_text=args.extract_text)
        print(paper.model_dump_json(indent=2))
    elif command == "extract":
        extract = agent.extract_paper(
            citation_key=args.citation_key,
            paper_id=args.paper_id,
            use_llm=False,
        )
        print(extract.model_dump_json(indent=2))
    elif command == "query":
        print(agent.answer_query(args.query, limit=args.limit).model_dump_json(indent=2))
    elif command == "rebuild-index":
        agent.index.rebuild()
        print(json.dumps({"status": "rebuilt", "path": agent.index.INDEX_PATH}, indent=2))
    return 0


def _experiment(args: argparse.Namespace) -> int:
    store, router = _router_and_store(args)
    agent = ExperimentAgent(store, router.experimenter)
    command = args.experiment_command
    if command == "start":
        payload = agent.ensure_container()
    elif command == "status":
        payload = agent.status()
    elif command == "stop":
        agent.stop_container(remove=args.remove)
        payload = {"status": "stopped", "removed": args.remove}
    elif command == "reset":
        agent.reset_container()
        payload = {"status": "reset"}
    else:
        code = Path(args.script).read_text(encoding="utf-8")
        program = ExperimentProgram(
            description=args.description,
            source=code,
            seeds=[args.seed],
        )
        _validate_experiment_program(program)
        result = agent.run_program(program=program, name=args.name)
        payload = result.model_dump(mode="json")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _doctor(args: argparse.Namespace) -> int:
    store = ArtifactStore(args.workspace)
    stale = [path for path in LEGACY_FILES if store.exists(path)]
    removed: list[str] = []
    if args.clean_legacy:
        for path in stale:
            store.resolve(path).unlink(missing_ok=True)
            removed.append(path)
        stale = [path for path in LEGACY_FILES if store.exists(path)]
    print(
        json.dumps(
            {
                "workspace": str(store.root),
                "legacy_files_present": stale,
                "removed": removed,
                "note": "The new engine never creates these files, including Nomenclature.yml.",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
