# Agentic TCS Research System

Artifact-driven research workflow for hard theoretical computer science problems. The system uses **LangGraph** for resumable orchestration, **vLLM** for local LLM serving, persistent files as canonical state, critic stages for scientific fidelity, and a LEAP-inspired **Lean** harness for formal verification.

This repository is a scaffold that can run conservative dry-run iterations immediately after installing dependencies, then be connected to local vLLM models for real agentic research attempts.

## Design goals

- **Correctness:** claims are typed, status-tracked, and downgraded unless supported by appropriate evidence.
- **Auditability:** every state-changing output is serialized as JSON/JSONL/YAML/Lean/code under a workspace.
- **Reproducibility:** experiments store command/config/seeds/logs under `ExperimentRuns/`.
- **Resumability:** the top-level loop is a LangGraph with SQLite checkpoints.
- **Extensibility:** prompts are editable Markdown files; agents are ordinary Python classes with Pydantic schemas.

## Canonical workspace artifacts

A research workspace contains:

```text
ResearchTask.md                 human-readable task, assumptions, criteria
Nomenclature.yml                canonical symbols and aliases
ResearchState.json              compact machine state summary
ClaimLedger.jsonl               mathematical/algorithmic/literature/resource claims
ProposalLedger.jsonl            proposal events and critic decisions
ModelCallLedger.jsonl           model routing, latency, token, validation logs
LiteratureDB/                   paper metadata, extracted claims, notation mappings
LeanProject/                    Lean/Lake project and LEAP proof DAGs
ExperimentRuns/                 reproducible runs with configs/seeds/logs
Reports/                        structured reports and derived Markdown
GraphCheckpoints.sqlite         LangGraph resumability checkpoints
```

Graph state stores references and counters only; these files are canonical.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional development tools:

```bash
pip install -e '.[dev]'
```

## vLLM serving

Start one large model and optionally a smaller routine model:

```bash
cp config.example.yml config.yml
# edit model names if needed
docker compose -f docker-compose.vllm.yml up
```

Or run vLLM directly:

```bash
vllm serve Qwen/Qwen3-32B --served-model-name deep-reasoner --port 8000
vllm serve Qwen/Qwen3-8B  --served-model-name routine-extractor --port 8001
```

The router logs model choice, latency, token usage, structured-output validity, and failure modes to `ModelCallLedger.jsonl`.

## Quick start

Dry-run mode uses conservative fallbacks and does not call vLLM:

```bash
tcs-research init --workspace workspaces/demo --dry-run --interactive \
  --task "Find improved algorithms or obstructions for a structured SAT family."

tcs-research run --workspace workspaces/demo --dry-run --max-iterations 1

tcs-research status --workspace workspaces/demo
```

With local vLLM:

```bash
tcs-research run --workspace workspaces/demo --config config.yml --max-iterations 3
```

Submit a Lean goal to LEAP:

```bash
tcs-research prove --workspace workspaces/demo --dry-run \
  --name nat_id --statement "∀ n : Nat, n = n"
```

Install Lean via `elan` for actual verification. The generated project is under `LeanProject/`.

## Top-level loop

The LangGraph implements:

```python
state = InitializeTask()
while not state.solved:
    proposal = GenerateResearchProposal(state)
    report = RunTCSResearchSubagent(proposal, state)
    state = UpdateResearchState(state, report)
    solved_verdict = CheckIsSolved(state, report)

    if solved_verdict.possible_breakthrough:
        replication = IndependentReplicationAgent.verify(state, report)
        state = UpdateResearchState(state, replication)

    if solved_verdict.confirmed_solved:
        break
```

Nodes durably write artifacts before returning. The graph is resumable through `GraphCheckpoints.sqlite` using a LangGraph `thread_id`.

## Agents

- `InitializationAgent`: interactive/refinement synthesis of `ResearchTask.md`, `Nomenclature.yml`, initial state, and ledgers.
- `ProposalAgent`: proposal generator plus proposal critic with revision/rejection logic.
- `ResearchAgent`: executes a selected proposal and writes a structured `ResearchReport`.
- `ResearchCriticAgent`: distinguishes proofs, citations, experiments, informal arguments, conjectures, refutations, and forced verification obligations.
- `LiteratureResearcher`: maintains `LiteratureDB` with paper metadata, extracted claims, theorem/algorithm statements, and notation mappings.
- `ObstructionAgent`: searches for lower bounds, no-go theorems, reductions, hidden assumptions, and duplicate literature risks.
- `ResourceAccountingAgent`: checks time/space/query/circuit/proof-size and quantum-specific resources.
- `TheoremProverAgent` / `LEAPHarness`: Lean proof search with local Lean declaration retrieval, direct formalization, revision, blueprint decomposition, AND-OR proof DAGs, and strict `sorry` discipline.
- `ExperimentAgent`: reproducible command runner for simulations, brute-force searches, and numerical checks.
- `IndependentReplicationAgent`: verifies possible breakthroughs from minimized context.

## LEAP proof discipline

A theorem is accepted as proved only if Lean verifies a final proof with no `sorry`/`admit` placeholders.

A decomposition is accepted only if:

1. the formal sketch compiles;
2. the parent theorem body is placeholder-free;
3. placeholders occur only in explicitly declared child lemmas;
4. an LLM/human reviewer accepts the child lemmas as useful and non-circular;
5. adding the decomposition preserves proof-DAG acyclicity.

Partial LEAP results are still recorded: proved lemmas, open goals, blocked goals, compiler logs, accepted/rejected decompositions, and recommended next proof steps.

## Prompts and schemas

Prompts live in `src/tcs_agentic_research/prompts/*.md` and are intended to be edited.
All state-changing agent outputs use Pydantic models in `src/tcs_agentic_research/schemas.py` and are serialized as JSON/JSONL/YAML artifacts.

## Extending the system

1. Add or modify a Pydantic schema in `schemas.py`.
2. Add an agent class under `src/tcs_agentic_research/agents/`.
3. Add editable prompts under `prompts/`.
4. Wire the agent into `graph.py` or call it from an existing node.
5. Ensure every state-changing output writes a durable artifact and references it with `ArtifactRef`.

## Safety and scientific fidelity notes

This system is not a substitute for expert judgment. It is designed to make long-running automated research attempts more auditable. It should never treat experimental evidence, unverified LLM reasoning, or unsupported literature summaries as proofs. Major claimed results require independent replication and, when feasible, Lean verification.
