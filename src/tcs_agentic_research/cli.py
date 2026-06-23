"""Command line interface for the agentic TCS research system."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .agents.initialization import INTERVIEW_QUESTIONS, InitializationAgent
from .agents.theorem_prover import TheoremProverAgent
from .artifact_store import ArtifactStore
from .graph import ResearchGraph
from .llm import LLMRouter
from .schemas import LeanStatement


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tcs-research")
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init", help="Create or update initialization artifacts")
    _add_common(init_p)
    init_p.add_argument("--task", default="", help="Initial task/problem statement")
    init_p.add_argument("--task-file", help="Read initial task/problem statement from Markdown file")
    init_p.add_argument("--interactive", action="store_true", help="Ask a fixed refinement interview")

    run_p = sub.add_parser("run", help="Run/resume the LangGraph research loop")
    _add_common(run_p)
    run_p.add_argument("--task", default="", help="Seed task text if the workspace is uninitialized")
    run_p.add_argument("--task-file", help="Read seed task text if the workspace is uninitialized")
    run_p.add_argument("--max-iterations", type=int, default=1)
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
    except Exception as exc:  # noqa: BLE001 - CLI should surface actionable errors
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--config", help="Router config YAML; see config.example.yml")
    parser.add_argument("--dry-run", action="store_true", help="Do not call vLLM; use conservative fallbacks")
    parser.add_argument("--prompt-dir", help="Override prompt directory")


def _cmd_init(args: argparse.Namespace) -> int:
    store = ArtifactStore(args.workspace)
    store.initialize_layout()
    router = LLMRouter.from_config_file(args.config, store=store, dry_run=args.dry_run)
    task = _read_task(args.task, args.task_file)
    answers: dict[str, Any] = {}
    if args.interactive:
        for idx, question in enumerate(INTERVIEW_QUESTIONS, start=1):
            print(f"[{idx}/{len(INTERVIEW_QUESTIONS)}] {question}")
            answers[question] = input("> ").strip()
    state = InitializationAgent(store, router, prompt_dir=args.prompt_dir).initialize(
        user_seed=task, interview_answers=answers
    )
    print(f"Initialized workspace: {store.root}")
    print(f"Task ID: {state.task_id}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    seed = _read_task(args.task, args.task_file)
    graph = ResearchGraph(
        workspace=args.workspace,
        config_path=args.config,
        dry_run=args.dry_run,
        prompt_dir=args.prompt_dir,
    )
    result = graph.run(user_seed=seed, max_iterations=args.max_iterations, thread_id=args.thread_id)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    store = ArtifactStore(args.workspace)
    store.initialize_layout()
    state = store.load_state()
    print(f"Workspace: {store.root}")
    if state is None:
        print("No ResearchState.json found. Run `tcs-research init` or `tcs-research run --task ...`.")
        return 0
    print(state.model_dump_json(indent=2))
    print("\nRecent claims:")
    print(json.dumps(store.read_jsonl(ArtifactStore.CLAIM_LEDGER, limit=args.ledger_tail), indent=2))
    print("\nRecent proposals:")
    print(json.dumps(store.read_jsonl(ArtifactStore.PROPOSAL_LEDGER, limit=args.ledger_tail), indent=2))
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
        context=store.read_text(ArtifactStore.RESEARCH_TASK) if store.exists(ArtifactStore.RESEARCH_TASK) else "",
    )
    print(result.model_dump_json(indent=2))
    return 0


def _read_task(task: str, task_file: str | None) -> str:
    if task_file:
        return Path(task_file).read_text(encoding="utf-8")
    return task


if __name__ == "__main__":
    raise SystemExit(main())
