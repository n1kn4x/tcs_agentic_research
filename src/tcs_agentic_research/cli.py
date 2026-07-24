"""Command line interface for the modular research kernel and low-level services."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .agents.experiment import ExperimentAgent
from .agents.literature import LiteratureResearcher
from .agents.theorem_prover import TheoremProverAgent
from .artifact_store import ArtifactStore
from .engine import LEGACY_CORE_FILES, ResearchEngine
from .experimenter.validation import validate_experiment_program
from .llm import LLMRouter
from .schemas import ExperimentProgram, LeanStatement


LEGACY_FILES = (
    *LEGACY_CORE_FILES,
    "Nomenclature.yml",
    "ResearchState.json",
    "ObligationBoard.json",
    "ClaimLedger.jsonl",
    "ProposalLedger.jsonl",
    "ModelCallLedger.jsonl",
    "GraphCheckpoints.sqlite",
)
LEGACY_DIRECTORIES = (
    "Runs",
    "Reports",
    "ExperimentStates",
    "ExperimentRuns",
    "LiteratureDB",
    "LeanProject",
    ".experimenter",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tcs-research")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Give autonomous subsystems bounded action opportunities")
    _add_common(run_parser)
    run_parser.add_argument("--max-steps", type=int, default=1)
    run_parser.add_argument(
        "--subsystem",
        dest="subsystems",
        action="append",
        choices=["literature", "theory", "proof", "experiment"],
        help="Enable only the named subsystem(s); repeat for more than one",
    )

    status_parser = sub.add_parser("status", help="Show journal and runtime status")
    status_parser.add_argument("--workspace", default=".")
    status_parser.add_argument("--config")

    records_parser = sub.add_parser("records", help="Print immutable research records")
    records_parser.add_argument("--workspace", default=".")
    records_parser.add_argument("--config")
    records_parser.add_argument("--limit", type=int, default=50)

    doctor_parser = sub.add_parser("doctor", help="Inspect or archive incompatible v0.3 state")
    doctor_parser.add_argument("--workspace", default=".")
    doctor_parser.add_argument("--archive-legacy", action="store_true")

    prove_parser = sub.add_parser("prove", help="Run or resume one explicit LEAP proof search")
    _add_common(prove_parser)
    prove_parser.add_argument("--name", required=True)
    prove_parser.add_argument("--statement", required=True)
    prove_parser.add_argument("--import", dest="imports", action="append", default=[])
    prove_parser.add_argument("--namespace", default="TCSResearch")
    prove_parser.add_argument("--max-model-calls", type=int)
    prove_parser.add_argument("--max-wall-seconds", type=int)

    literature_parser = sub.add_parser("literature", help="Operate the local literature service")
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
                subsystem_names=args.subsystems,
            )
            print(json.dumps(engine.run(max_steps=args.max_steps), indent=2, ensure_ascii=False))
            return 0
        if args.command == "status":
            engine = ResearchEngine(
                workspace=args.workspace, config_path=args.config, dry_run=True
            )
            print(json.dumps(engine.status(), indent=2, ensure_ascii=False))
            return 0
        if args.command == "records":
            engine = ResearchEngine(
                workspace=args.workspace, config_path=args.config, dry_run=True
            )
            print(json.dumps(engine.records(limit=args.limit), indent=2, ensure_ascii=False))
            return 0
        if args.command == "doctor":
            return _doctor(args)
        if args.command == "prove":
            return _prove(args)
        if args.command == "literature":
            return _literature(args)
        if args.command == "experiment":
            return _experiment(args)
    except Exception as exc:  # CLI boundary
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 1


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--config", help="YAML configuration; see config.example.yml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prompt-dir")


def _literature_commands(sub: Any) -> None:
    search = sub.add_parser("search", help="Queue literature candidates")
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
    query = sub.add_parser("query", help="Query local exact statements/passages")
    _add_common(query)
    query.add_argument("--query", required=True)
    query.add_argument("--limit", type=int, default=10)
    rebuild = sub.add_parser("rebuild-index", help="Rebuild the SQLite literature index")
    _add_common(rebuild)


def _experiment_commands(sub: Any) -> None:
    for name in ["start", "status", "reset"]:
        command = sub.add_parser(name)
        _add_common(command)
    stop = sub.add_parser("stop")
    _add_common(stop)
    stop.add_argument("--remove", action="store_true")
    run = sub.add_parser("run", help="Run an explicit ExperimentOutput-v1 Python script")
    _add_common(run)
    run.add_argument("--script", required=True)
    run.add_argument("--description", required=True)
    run.add_argument("--name", default="experiment")
    run.add_argument("--seed", type=int, action="append", default=[])


def _router_and_store(args: argparse.Namespace) -> tuple[ArtifactStore, LLMRouter]:
    store = ArtifactStore(args.workspace)
    store.initialize_layout()
    router = LLMRouter.from_config_file(args.config, store=store, dry_run=args.dry_run)
    return store, router


def _prove(args: argparse.Namespace) -> int:
    store, router = _router_and_store(args)
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
                imports=args.imports or ["TCSResearch.Basic"],
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
    value: Any
    if command == "search":
        value = [item.model_dump(mode="json") for item in agent.search_papers(args.query, limit=args.limit)]
    elif command == "import-arxiv":
        value = agent.import_arxiv(args.arxiv_id, citation_key=args.citation_key, extract_text=args.extract_text)
    elif command == "import-doi":
        value = agent.import_doi(args.doi, citation_key=args.citation_key, extract_text=args.extract_text)
    elif command == "import-url":
        value = agent.import_url(args.url, citation_key=args.citation_key, title=args.title, extract_text=args.extract_text)
    elif command == "import-candidate":
        value = agent.import_candidate(args.candidate_id, extract_text=args.extract_text)
    elif command == "extract":
        value = agent.extract_paper(citation_key=args.citation_key, paper_id=args.paper_id, use_llm=False)
    elif command == "query":
        value = agent.answer_query(args.query, limit=args.limit)
    else:
        agent.index.rebuild()
        value = {"status": "rebuilt", "path": agent.index.INDEX_PATH}
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    print(json.dumps(value, indent=2, ensure_ascii=False))
    return 0


def _experiment(args: argparse.Namespace) -> int:
    store, router = _router_and_store(args)
    agent = ExperimentAgent(store, router.experimenter)
    command = args.experiment_command
    value: Any
    if command == "start":
        value = agent.ensure_container()
    elif command == "status":
        value = agent.status()
    elif command == "stop":
        agent.stop_container(remove=args.remove)
        value = {"status": "stopped", "removed": args.remove}
    elif command == "reset":
        agent.reset_container()
        value = {"status": "reset"}
    else:
        program = ExperimentProgram(
            description=args.description,
            source=Path(args.script).read_text(encoding="utf-8"),
            seeds=args.seed or [0],
        )
        validate_experiment_program(program)
        value = agent.run_program(program=program, name=args.name)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    print(json.dumps(value, indent=2, ensure_ascii=False))
    return 0


def _doctor(args: argparse.Namespace) -> int:
    store = ArtifactStore(args.workspace)
    present_files = [path for path in LEGACY_FILES if store.exists(path)]
    present_dirs = [path for path in LEGACY_DIRECTORIES if store.exists(path)]
    archive_path = ""
    if args.archive_legacy and (present_files or present_dirs):
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        archive = store.resolve(f"Archive/v03-{stamp}")
        archive.mkdir(parents=True, exist_ok=True)
        for rel in [*present_files, *present_dirs]:
            source = store.resolve(rel)
            if source.exists():
                target = archive / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(target))
        archive_path = store.relpath(archive)
        present_files = [path for path in LEGACY_FILES if store.exists(path)]
        present_dirs = [path for path in LEGACY_DIRECTORIES if store.exists(path)]
    print(
        json.dumps(
            {
                "workspace": str(store.root),
                "legacy_files": present_files,
                "legacy_directories": present_dirs,
                "archived_to": archive_path,
            },
            indent=2,
        )
    )
    return 0
