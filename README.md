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
InitializationInterview.md      transcript of the adaptive initialization conversation
Nomenclature.yml                canonical symbols and aliases
ResearchState.json              compact machine state summary
ObligationBoard.json            candidate claims, linked obligations, runs, and blocked reasons
ClaimLedger.jsonl               accepted/proven mathematical/algorithmic/literature/resource claims
ProposalLedger.jsonl            proposal events and critic decisions
ModelCallLedger.jsonl           model routing, latency, token, validation logs
LiteratureDB/                   papers, discovery candidates, extracted statements/claims, query answers
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

Docker must be installed and running if you want to use the experiment subsystem. The experimenter builds and runs a project-level Docker container with internet access and shell access, and fails fast if Docker or `experimenter:` configuration is missing.

## vLLM serving

Start Qwen3.6 for the deep/agentic endpoint and optionally a smaller routine model:

```bash
cp config.example.yml config.yml
# edit model names, ports, tensor parallelism, or context lengths if needed
docker compose -f docker-compose.vllm.yml up
```

Or run vLLM directly (Qwen recommends vLLM >= 0.19.0 for Qwen3.6):

```bash
vllm serve Qwen/Qwen3.6-35B-A3B --served-model-name deep-reasoner --port 8000 \
  --tensor-parallel-size 4 --max-model-len 262144 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --language-model-only \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}'

vllm serve Qwen/Qwen3-8B --served-model-name routine-extractor --port 8001 \
  --max-model-len 32768 \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

The router config passes Qwen3.6 sampling parameters per profile: thinking + preserve-thinking for deep agentic/verifier tasks, lower-temperature thinking for precise Lean/code-style tasks, and non-thinking for routine extraction/formatting. The router logs model choice, latency, token usage, structured-output validity, dry-run mock-output usage, and failure modes to `ModelCallLedger.jsonl`.

## Quick start

Initialize a workspace with an adaptive interview. Dry-run mode uses deterministic mock outputs and does not call vLLM:

```bash
tcs-research init --workspace workspaces/demo --dry-run

tcs-research run --workspace workspaces/demo --dry-run --max-iterations 1

tcs-research status --workspace workspaces/demo
```

With local vLLM, the `init` command is an LLM-guided conversation that asks only relevant follow-up questions before writing artifacts:

```bash
tcs-research init --workspace workspaces/demo --config config.yml
tcs-research run --workspace workspaces/demo --config config.yml --max-iterations 3
```

### Subsystem LEAP
Submit a Lean goal to LEAP:

```bash
tcs-research prove --workspace workspaces/demo --dry-run \
  --name nat_id --statement "∀ n : Nat, n = n"
```
Install Lean via `elan` for actual verification. The generated project is under `LeanProject/`.

### Subsystem Experimenter

The experiment subsystem runs simulations, numerical checks, plotting jobs, data gathering, and small-instance searches through an existing coding agent (`pi`) inside Docker.

Key properties:

- one persistent Docker container per research workspace/project;
- the canonical workspace is mounted read-only at `/research` inside the container;
- portable writable experimenter state lives under `.experimenter/workspace` in the research workspace and is mounted at `/workspace` for scripts, package caches, pi sessions, and run outputs;
- copying the research workspace to another machine copies experimenter state; the Docker image is rebuilt from the bundled Dockerfile if absent on the new machine;
- completed run artifacts are imported back into `ExperimentRuns/`;
- the bundled image includes `pi`, Python, NumPy, pandas, SciPy, SymPy, matplotlib, seaborn, scikit-learn, statsmodels, NetworkX, IPython/Jupyter, git, curl, and build tools;
- no placeholder fallback is used: if the experimenter is invoked without working Docker/configuration, the command fails.

The Docker image is global to the local Docker daemon and is intentionally not stored in the workspace. The portable unit is the workspace itself: canonical artifacts, `ExperimentRuns/`, and `.experimenter/workspace/`. If you delete a workspace, its portable experimenter state is deleted with it; stopped/running Docker containers should still be cleaned up with `tcs-research experiment reset` or Docker tooling because a plain filesystem delete cannot notify the Docker daemon. The Dockerfile is available in `src/experimenter/Dockerfile` and packaged as `src/tcs_agentic_research/experimenter/Dockerfile`.

Configure the `experimenter:` block in `config.yml`, then manage it with:

```bash
# Build/start the project container
tcs-research experiment start --workspace workspaces/demo --config config.yml

# Run a one-off experiment through Dockerized pi
tcs-research experiment run --workspace workspaces/demo --config config.yml \
  --name smoke --description "Create a Python script that prints 2+2 and record the result."

# Inspect/stop/reset the project container
tcs-research experiment status --workspace workspaces/demo --config config.yml
tcs-research experiment stop --workspace workspaces/demo --config config.yml
tcs-research experiment reset --workspace workspaces/demo --config config.yml
```

### Subsystem Literature Researcher
Import and query literature with canonical notation and quote-level provenance:

```bash
tcs-research literature import-arxiv --workspace workspaces/demo \
  --arxiv-id 2401.00001 --extract-text

tcs-research literature extract --workspace workspaces/demo --citation-key arxiv_2401.00001

tcs-research literature query --workspace workspaces/demo \
  --query "lower bound for the main subproblem"

# Scholar-like discovery via OpenAlex queues candidates only:
tcs-research literature search --workspace workspaces/demo \
  --query "quantum LPN lower bound" --limit 20

tcs-research literature discover-related --workspace workspaces/demo \
  --citation-key arxiv_2401.00001 --direction cited_by --limit 20

tcs-research literature import-candidate --workspace workspaces/demo \
  --candidate-id cand_abc123 --extract-text

# deterministic smoke test (no vLLM call):
tcs-research literature test --workspace workspaces/demo --dry-run
```


## Top-level loop

The LangGraph implements:

```python
state = LoadInitializedTask()  # run `tcs-research init` first
while not state.solved:
    if no open obligation exists:
        proposal = GenerateResearchProposal(state, blocked_claims_and_failed_obligations)
        candidate_claim, obligations = CreateCandidateClaimAndObligations(proposal)

    obligation = SelectNextOpenObligation()
    run = RunTCSResearchSubagentOnOneObligation(obligation)
    validation = DeterministicObligationGates(run)  # scope/provenance, evidence, consistency
    state = CommitOnlyIfValidated(run, validation)
    solved_verdict = ComputeSolvedVerdict(state, derived_summary_report)

    if solved_verdict.possible_breakthrough:
        replication = IndependentReplicationAgent.verify(state, derived_summary_report)
        state = UpdateResearchState(state, replication)

    if solved_verdict.confirmed_solved:
        break
```

Nodes durably write artifacts before returning. Reports are derived summaries; they are not the canonical path for accepting claims. Candidate claims live on `ObligationBoard.json` until every linked obligation is fulfilled and passes deterministic gates. Only the deterministic commit manager appends accepted claims to `ClaimLedger.jsonl`. The graph is resumable through `GraphCheckpoints.sqlite` using a LangGraph `thread_id`.

## Agents

- `InitializationAgent`: LLM-guided adaptive interview and synthesis of `ResearchTask.md`, `Nomenclature.yml`, initial state, and ledgers.
- `ProposalAgent`: proposal generator using native OpenAI/vLLM tool calls plus proposal critic with revision/rejection logic. Private model reasoning is not replayed into future contexts; only committed proposal artifacts are.
- `ResearchAgent`: executes a selected proposal in a native OpenAI/vLLM tool-call loop and finishes by calling `submit_research_report`; deterministic critics/evidence gates still decide which claims are accepted.
- `ResearchCriticAgent`: distinguishes proofs, citations, experiments, informal arguments, conjectures, refutations, and forced verification obligations.
- `LiteratureResearcher`: modular literature pipeline for OpenAlex search/citation candidate discovery, arXiv/DOI/PDF import, PDF text extraction, theorem/algorithm extraction, nomenclature updates, duplicate detection, and quote-provenance query answers in mapped notation.
- `TheoremProverAgent` / `LEAPHarness`: Lean proof search with local Lean declaration retrieval, direct formalization, revision, blueprint decomposition, AND-OR proof DAGs, and strict `sorry` discipline.
- `ExperimentAgent`: Dockerized pi-backed experiment runner for simulations, brute-force searches, numerical checks, plots, and data-gathering tasks. It mounts canonical artifacts read-only and imports run artifacts into `ExperimentRuns/`.
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
Structured prompts contain schema placeholders like `{{ResearchReport}}`. At runtime
each placeholder is replaced with the full Pydantic JSON Schema. The structured-call output
schema is also sent to vLLM through `guided_json`/`response_format` when supported.
Returned JSON is validated with the same Pydantic model.

All state-changing agent outputs use Pydantic models in `src/tcs_agentic_research/schemas.py` and
are serialized as JSON/JSONL/YAML artifacts. Native tool-call agents can use different toolsets 
of the same underlying tools.

## Extending the system

1. Add or modify a Pydantic schema in `schemas.py`.
2. Add an agent class under `src/tcs_agentic_research/agents/`.
3. Add editable prompts under `prompts/`.
4. Wire the agent into `graph.py` or call it from an existing node.
5. Ensure every state-changing output writes a durable artifact and references it with `ArtifactRef`.

## Safety and scientific fidelity notes

This system is not a substitute for expert judgment. It is designed to make long-running automated research attempts more auditable. It should never treat experimental evidence, unverified LLM reasoning, or unsupported literature summaries as proofs. Major claimed results require independent replication and, when feasible, Lean verification.
