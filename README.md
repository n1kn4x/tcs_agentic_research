# Cumulative Agentic TCS Research

A small runtime for long-running research in which **subsystems choose the research actions** and a
non-semantic kernel provides durable shared memory, round-robin subsytem scheduling, crash recovery, and strict
evidence labels.

See [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Design rules

1. Literature, theory, proof, and experiment are autonomous modules behind one tiny interface.
2. Every module sees the task and prior cross-module records before choosing one atomic action.
3. Records and action transitions are append-only. Task edits preserve prior project memory.
4. Model prose is always `tentative`, even after model review.
5. Exact source spans and exactly replicated execution are `observed`.
6. Only placeholder-free, compiler-accepted Lean propositions are `verified`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp config.example.yml config.yml
```

## Workspace

A fresh workspace needs one user-authored file:

```text
InitialResearchTask.md
```

The kernel creates:

```text
KernelState.json              scheduling cursor; no scientific state
TaskVersions.jsonl            project-brief revisions
Actions.jsonl                 durable action transitions
Records.jsonl                 cumulative cross-subsystem memory
Events.jsonl                  runtime events and failures
ModelCalls.jsonl              model telemetry
Subsystems/<name>.json        opaque continuation state owned by each subsystem
Runs/...                      exact action proposals and outcomes
Reports/Status.md             deterministic runtime/status view
Reports/Research.md           deterministic epistemically grouped memory view
LiteratureDB/                 only when literature is used
LeanProject/                  only when proof is used
ExperimentRuns/               only when experiments are used
```

## Running

```bash
mkdir -p workspaces/demo
cp examples/structured_sat_task.md workspaces/demo/InitialResearchTask.md

tcs-research run --workspace workspaces/demo --config config.yml --max-steps 20
tcs-research status --workspace workspaces/demo --config config.yml
tcs-research records --workspace workspaces/demo --config config.yml --limit 20
```

`--max-steps` is the number of subsystem action opportunities.
Actions are offered round-robin. A run stops early after every enabled subsystem yields once.

Run a module in isolation:

```bash
tcs-research run --workspace workspaces/demo --config config.yml \
  --subsystem literature --max-steps 10

tcs-research run --workspace workspaces/demo --config config.yml \
  --subsystem proof --max-steps 3
```

Repeat `--subsystem` to enable a selected set. This is the preferred way to diagnose and acceptance-
test one capability;

A deterministic control-flow dry run exercises only the tentative notebook path and performs no
network, Docker, or proof work:

```bash
tcs-research run --workspace workspaces/demo --dry-run --subsystem theory --max-steps 1
```

## Evidence semantics

A record has one of three statuses, assigned only by deterministic policy:

### Tentative

Theory entries, syntheses, proposed counterexamples, model-authored relevance links, and failure
diagnoses are retained because they can guide later work. They are never presented as verified.
A critic model has no special authority.

### Observed

Literature records require hashed metadata or a span validated against imported text. An exact quote
means “this source says these words,” not “the claim is true.”

Experiment records require the same hashed program and fixed seeds to produce identical structured
output twice. The runner records raw observations and the program's interpretation separately. The
interpretation remains unverified.

### Verified

Formal theorem records require Lean acceptance, a hashed `.lean` artifact, and a placeholder scan.
Only the proposition is verified. Whether it is important or sufficient remains a research question.

## Editing a task

`InitialResearchTask.md` may be refined as the project evolves. Its new digest is appended to
`TaskVersions.jsonl`, the revision number increments, and prior records remain visible. Use a new
workspace for an unrelated project.

## Migration from v0.3

The old agenda/queue workspace format is intentionally incompatible. It encoded unreliable planner
decisions and is not migrated into the new journal.

```bash
tcs-research doctor --workspace workspaces/old
tcs-research doctor --workspace workspaces/old --archive-legacy
```

Archiving moves the old core and subsystem artifacts under `Archive/v03-<timestamp>/`; the original
`InitialResearchTask.md` remains so the workspace can start with an empty v0.4 memory.

## Direct subsystem services

The autonomous actors use the same low-level services available from the CLI.

### Lean

```bash
tcs-research prove --workspace workspaces/demo --config config.yml \
  --name count_append --statement \
  '∀ (b : Bool) (xs ys : List Bool), (xs ++ ys).count b = xs.count b + ys.count b'
```

### Literature

```bash
tcs-research literature search --workspace workspaces/demo --config config.yml \
  --query 'SETH Orthogonal Vectors logarithmic dimension'
tcs-research literature import-arxiv --workspace workspaces/demo --config config.yml \
  --arxiv-id 1811.12017 --extract-text
tcs-research literature query --workspace workspaces/demo --config config.yml \
  --query 'strongly subquadratic logarithmic dimension'
```

### Experiment

An explicit script must define `run_experiment(mode)` and return the strict `ExperimentOutput`
schema version 1 documented in `schemas.py`.

```bash
tcs-research experiment run --workspace workspaces/demo --config config.yml \
  --script experiment.py --description 'Fixed-seed bounded comparison' --seed 7
```

Generated and manual experiments run in the configured networkless Docker container with CPU,
memory, file, output, and wall-time limits.

## Model serving

The supplied configuration routes fresh OpenAI-compatible requests to one Qwen endpoint.
Start the example vLLM service with:

```bash
docker compose -f docker-compose.vllm.yml up
# or
./scripts/launch_vllm_stack.sh
```

Edit `router.profiles.*.base_url` in `config.yml` for a remote endpoint. Theory and decision calls,
experiment coding, JSON formatting, and proof search can use separate profiles while sharing one
served model.

## Development

```bash
pytest -q
ruff check src tests
mypy src/tcs_agentic_research --ignore-missing-imports
```

Tests are modular: scripted kernel actors, deterministic evidence admission, experiment containment,
literature provenance, and LEAP graph/compiler behavior. They do not assert that a language model
will invent a correct global research plan.
