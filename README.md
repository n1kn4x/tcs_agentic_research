# Bounded Agentic TCS Research

A small, inspectable research loop for theoretical computer science. The language model proposes
**typed, bounded work**; deterministic Python executes it. Models never receive a general tool loop
and never decide that their own claims are verified.

## Why this design

The previous design combined LangGraph, proposal/critic loops, obligation hydration, nested tool
calls, large schemas, replayed tool observations, and several overlapping ledgers. It was easy for
context to grow without bound and hard to tell whether a failure was scientific or merely a malformed
tool call.

The replacement follows five rules:

1. **One bounded work item at a time.** Literature, proof, experiment, and analysis are separate.
2. **No model-driven tool loop.** A model returns one small JSON object. Python performs actions.
3. **Fresh context.** Every call receives the task, one work item, and a small evidence summary—not
   conversation history or an artifact dump.
4. **Evidence determines status.** Lean output is `verified`, exact literature quotes are `supported`,
   experiment output is `observed`, and model synthesis remains a `hypothesis`.
5. **Failure is progress data.** A failed step is persisted with its input, error, and next action. It
   is not converted into a fake scientific proposal.

There is no `solved` bit. The engine eventually enters `review`; a person decides whether the task's
success criteria are met.

## Workspace contract

Only `InitialResearchTask.md` is user-authored and required. The engine creates:

```text
InitialResearchTask.md       canonical task
State.json                   small phase/cycle counters
Queue.json                   bounded work items and terminal status
Events.jsonl                 append-only lifecycle events
Findings.jsonl               evidence-typed findings
ModelCalls.jsonl             tokens, latency, input size, schema, failures
Runs/NNNN_<id>/              exact input, typed output, and step-local reports
Reports/                     optional human-facing reports
LiteratureDB/                created when literature is used
LeanProject/                 created when proof work is used
ExperimentRuns/              created when experiment work is used
```

`Nomenclature.yml`, proposal/claim ledgers, obligation boards, and graph checkpoints are not part of
the new design and are never created. Detect or remove old files with:

```bash
tcs-research doctor --workspace workspaces/demo
tcs-research doctor --workspace workspaces/demo --clean-legacy
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp config.example.yml config.yml
```

## Run

```bash
mkdir -p workspaces/demo
cp examples/structured_sat_task.md workspaces/demo/InitialResearchTask.md

# Deterministic control-flow test; no network, model, Docker, or accepted proof claims.
tcs-research run --workspace workspaces/demo --dry-run --max-steps 2

# Real bounded work.
tcs-research run --workspace workspaces/demo --config config.yml --max-steps 1
tcs-research status --workspace workspaces/demo
```

Use small invocations. Inspect the latest `Runs/` directory, then run another step. When the engine
enters `review`, either review the evidence or explicitly request another planning round:

```bash
tcs-research replan --workspace workspaces/demo --config config.yml
tcs-research run --workspace workspaces/demo --config config.yml --max-steps 1
```

## Model serving and Qwen3.6

Qwen's advertised native context length is a capacity, not a target. The application limits a request
to 30,000 characters and an output to 12,288 tokens in the example profile. The larger output
allowance is for self-contained experiment programs; control and coding calls use non-thinking
profiles, and prompts still request compact outputs. Long contexts make failures more
expensive and do not repair bad orchestration.

The default Qwen3.6 profile uses the vendor's precise-coding-style sampling (`temperature=0.6`,
`top_p=0.95`, `top_k=20`) with `presence_penalty=0`. Qwen explicitly warns that high presence
penalties can cause language mixing. Thinking is enabled only for reasoning calls and historical
thinking is not preserved because calls are fresh. Lower temperature is not a JSON validator;
`response_format` plus Pydantic validation is.

Start the provided services:

```bash
docker compose -f docker-compose.vllm.yml up
# or
./scripts/launch_vllm_stack.sh
```

The deep endpoint needs `--reasoning-parser qwen3`, but not auto tool choice or a tool-call parser: the
core sends no tools. Use a recent vLLM version with JSON-schema response-format support.

### Structured-output failure policy

- Schemas are sent once through `response_format`; they are not pasted into prompts.
- Core schemas contain a few flat strings/lists and at most four work items.
- Invalid output gets at most one fresh formatting repair on the `format` profile.
- The repair call sees only the malformed output and validation error, not the growing history.
- A second failure terminates an evidence-producing model step and is recorded. Scheduling and
  literature-query planning may fall back to conservative task-derived actions because those are
  control flow, not scientific claims. No mock or scientific result is used in a real run.

## Literature subsystem

The canonical literature data is intentionally narrow:

```text
LiteratureDB/papers.jsonl       paper metadata events
LiteratureDB/candidates.jsonl   OpenAlex discovery queue
LiteratureDB/statements.jsonl   current exact statement/quote snapshot per paper
LiteratureDB/papers/...         PDF, extracted text, metadata
LiteratureDB/index.sqlite       rebuildable search index (not canonical)
```

A literature step asks the model only for search queries and focus questions. Python then performs a
bounded OpenAlex search, imports at most the configured number of candidates, extracts statements
deterministically, and runs local retrieval. Stable statement/quote/support IDs are content-derived.
A finding is `supported` only when its quote span is found in imported text; otherwise it stays a
`hypothesis` with an explicit caveat.

Install Poppler's `pdftotext` (`poppler-utils` on Debian/Ubuntu) for substantially better PDF
layout extraction; `pypdf` remains a fallback.

Manual operations remain available:

```bash
tcs-research literature search --workspace workspaces/demo --query "SETH Orthogonal Vectors"
tcs-research literature import-arxiv --workspace workspaces/demo --arxiv-id 1811.12017 --extract-text
tcs-research literature extract --workspace workspaces/demo --citation-key arxiv_1811.12017
tcs-research literature query --workspace workspaces/demo --query "logarithmic dimension lower bound"
tcs-research literature rebuild-index --workspace workspaces/demo
```

## Lean / LEAP subsystem

Proof work uses a persistent, resumable AND-OR DAG. LEAP first tries a tiny deterministic,
compiler-checked tactic portfolio, then informal planning plus direct formalization and localized
compiler-feedback revisions, and finally a Lean-verified decomposition into shared child
propositions. Deterministic cycle/restatement checks
and a separate usefulness reviewer prevent formally valid but non-progressing branches. Attempts,
compiler output, accepted sketches, and proved nodes remain in `LeanProject/LEAP/state.sqlite` across
invocations.

```bash
tcs-research prove --workspace workspaces/demo --config config.yml \
  --name nat_id --statement "∀ n : Nat, n = n"

# Resume the same graph with a larger one-invocation budget
tcs-research prove --workspace workspaces/demo --config config.yml \
  --name nat_id --statement "∀ n : Nat, n = n" \
  --max-model-calls 256 --max-wall-seconds 86400
```

A root becomes a `verified` finding only after LEAP topologically materializes all required lemmas and
batch-compiles one self-contained, `sorry`/`admit`-free final module. See
[`src/tcs_agentic_research/leap/README.md`](src/tcs_agentic_research/leap/README.md) for architecture,
Mathlib setup, persistence, and budget details.

## Experiment subsystem

Experiment work uses one structured model call to generate one Python program, then executes exactly
that program in a resource- and time-bounded Docker container. It does **not** launch a second coding
agent with another tool loop. The research workspace is read-only and the default container network
is disabled.

You can also run a reviewed script directly:

```bash
tcs-research experiment run --workspace workspaces/demo --config config.yml \
  --script experiment.py --description "Fixed-seed small-instance check" --seed 7
```

Successful output is `observed`, never mathematical proof.

## Introspection checklist

For a surprising result, inspect in this order:

1. `State.json` and `Queue.json` — what was selected and why did it stop?
2. Latest `Runs/*/input.json` — exact bounded model/context input.
3. Latest `Runs/*/result.json` — typed outcome, errors, artifacts, next steps.
4. `ModelCalls.jsonl` — profile, input size, tokens, latency, schema, HTTP failure.
5. Evidence artifact named by the finding — exact quote, Lean file/log, or experiment output.

This separation makes an engineering failure, missing source, failed proof, and genuine mathematical
obstruction visibly different states.
