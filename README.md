# Bounded Agentic TCS Research

A small, inspectable research loop for theoretical computer science. The model proposes typed
scientific content; deterministic Python owns fair scheduling, execution, persistence, and acceptance
gates. Models never receive a general tool loop and never decide that their own claims are verified.

## Why this design

The previous design combined LangGraph, proposal/critic loops, obligation hydration, nested tool
calls, large schemas, replayed tool observations, and several overlapping ledgers. It was easy for
context to grow without bound and hard to tell whether a failure was scientific or merely a malformed
tool call.

The replacement follows seven rules:

1. **Atomic evidence gaps.** Every question is split into persisted requirements with acceptance
   criteria, allowed methods, attempt history, findings, and status.
2. **Falsifiability before execution.** Every work item states a hypothesis, strategy, falsification
   criterion, expected information gain in either direction, and non-execution success criteria.
3. **Contribution-based progress.** A cycle advances progress only when it creates novel usable
   evidence. Calls, files, imports, execution, and rewritten summaries do not count.
4. **Negative results are results.** Counterexamples, contradictions, registered null outcomes, and
   scoped obstructions are stored and reported as first-class contributions.
5. **Independent gates.** Mathematical derivations require two fresh adversarial reviews; Lean goals receive
   relevance review; experiments freeze/review a protocol before code and audit evidence afterward;
   literature statements require exact spans plus requirement-level relevance review.
6. **Persistent recovery without stage churn.** One research cycle drives an experiment through as
   many durable protocol, implementation, execution, and review stages as its bounded call budget
   permits. Exact defects are repaired in place; repeated no-op repairs stop quickly, and one blocked
   experiment never stops unrelated research.
7. **No model-driven tool loop or model-owned `solved` bit.** Python owns actions, novelty,
   requirement transitions, attempt caps, completion, and exhaustion.

A workspace enters `complete` only when every mandatory requirement is satisfied. It enters
`needs_input` only when all configured methods and revisions for unresolved mandatory gaps are
exhausted—not merely because several consecutive attempts failed.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the cycle, persistence, scheduling, and pipeline
invariants.

## Workspace contract

Only `InitialResearchTask.md` is user-authored and required. This evidence-gap schema intentionally
has no migration layer for pre-refactor `Agenda.json`/`Queue.json` workspaces; start a fresh workspace
(or retain the old one as an archive) rather than mixing incomparable progress semantics. The engine
creates:

```text
InitialResearchTask.md       canonical task
State.json                   phase/cycle, contribution and diversification counters
Agenda.json                  questions plus atomic evidence requirements and acceptance state
Queue.json                   falsifiable strategies, lineage, revisions, and status
Events.jsonl                 append-only lifecycle events
Findings.jsonl               evidence-typed findings with polarity, strength, and scope
Contributions.jsonl          novelty-deduplicated positive/negative/null research progress
ModelCalls.jsonl             tokens, latency, input size, schema, failures
Runs/NNNN_<id>/              exact input, typed output, and step-local reports
Reports/Progress.md          continuously updated evidence and failure dashboard
LiteratureDB/                created when literature is used
LeanProject/                 created when proof work is used
ExperimentStates/            durable protocol/program/execution stage for each experiment strategy
ExperimentRuns/              bounded smoke and full execution artifacts
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

Invocations may be small or long; every completed work item updates both
`Reports/Progress.md` (attempts, coverage, blockers) and `Reports/ResearchReport.md` (usable results).
There is no arbitrary planning-round limit. The configured no-progress threshold triggers a durable
diversification event; it does not halt work while another method or requirement remains. Only full
strategy exhaustion enters `needs_input`. Review the exact blockers, then request a human replan
(which grants two additional distinct-strategy slots per method while retaining all evidence and
attempt history), or revise the task to start a new archived agenda:

```bash
tcs-research replan --workspace workspaces/demo --config config.yml
tcs-research run --workspace workspaces/demo --config config.yml --max-steps 1
```

## Model serving and Qwen3.6

One shared Qwen endpoint serves every agent profile using tensor parallelism across GPUs 0-3. The
reasoning, control, coding, formatting, and proof profiles all send model name `qwen-research` to the
same OpenAI-compatible endpoint; profiles differ only in sampling, output budget, and whether Qwen
thinking is enabled. This avoids loading separate extraction and proof models.

The server defaults to `Qwen/Qwen3.6-35B-A3B`, tensor parallel size 4, and a 32,768-token model limit.
The application separately limits request input to 50,000 characters and output to at most 12,288
tokens. Reasoning and proof calls enable thinking; control, coding, and formatting calls disable it.
Historical thinking is never preserved because calls are fresh.

Start the shared endpoint with either:

```bash
docker compose -f docker-compose.vllm.yml up
# or
./scripts/launch_vllm_stack.sh
```

Both launchers default to `CUDA_VISIBLE_DEVICES=0,1,2,3`, `QWEN_TP=4`, port 8000, and served model
name `qwen-research`. Override them when needed, for example:

```bash
QWEN_PORT=18000 QWEN_MAX_MODEL_LEN=32768 REPLACE=1 ./scripts/launch_vllm_stack.sh
```

The endpoint needs `--reasoning-parser qwen3`, but not auto tool choice or a tool-call parser: the
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
bounded OpenAlex search with an arXiv fallback and a fast rate-limit circuit breaker, imports at most
the configured number of candidates, extracts statements
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

## Mathematical derivation subsystem

Not every theoretical result is practical to formalize in the minimal Lean environment. A
`derivation` work item therefore produces a structured assumption-to-conclusion argument with
labelled dependencies, an explicit falsification attempt, boundary conditions, and limitations. Two
fresh adversarial referee calls independently recompute transitions and search for counterexamples.
Rejected derivations create targeted revisions; accepted counterexamples and obstructions are
negative contributions, while accepted positive derivations are marked `derived` with an explicit
caveat that they are not kernel-checked. This is distinct from synthesis, which never creates
evidence.

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

Experiment work uses a durable state machine: protocol design/review, program design/review, smoke
execution, full execution, and evidence review. A single research cycle drives the machine until it
produces evidence, reaches its model/resource budget, or encounters a repeated concrete blocker.
Every transition persists `ExperimentStates/<work-id>.json`, so restarts resume accepted work instead
of regenerating it. A repair first produces a bounded reasoning plan, then a non-thinking coding call
emits one complete replacement file from the exact defect and prior source; fragile line-number patch
chains are not used. Two repairs per outer cycle preserve fairness. Repeated/no-op defects stop
quickly, and a configurable cumulative source-revision cap eventually retires an oscillating program
strategy so that a fresh strategy or unrelated requirement can run. Repairable programs are capped
at 20,000 characters so accepted prior source is supplied in full; oversize candidates are replaced
rather than repaired from truncated context.

The model implements `run_experiment(mode: str) -> dict`. A trusted wrapper owns the entry point,
writes `results.json`, and validates the v2 output contract. Smoke mode exercises every condition on
tiny samples before the frozen full run. Both execute in the resource- and time-bounded Docker
container; the research workspace is read-only and the default container network is disabled.

You can also run a reviewed script directly:

```bash
tcs-research experiment run --workspace workspaces/demo --config config.yml \
  --script experiment.py --description "Fixed-seed small-instance check" --seed 7
```

Before execution the container is health-checked (including stale bind mounts), and Python code is
syntax/safety checked. Smoke mode must branch from full mode and exercise every condition on tiny
samples. The explicit script passed to `experiment run` must define
`run_experiment(mode: str) -> dict`. The trusted wrapper writes the v2 `results.json` contract: scalar
parameters and aggregate metrics, condition-level observations, implementation checks, a
hypothesis/outcome/basis conclusion, and explicit limitations. This shape preserves negative and null
measurements instead of collapsing them into a pass/fail bit. A post-run reviewer performs the one
semantic alignment/methodology audit and grades evidence as `full`, `preliminary`, or `unusable`.
Only full evidence closes a requirement; sound preliminary evidence is retained for follow-up.

## Introspection checklist

For a surprising result, inspect in this order:

1. `State.json`, `Agenda.json`, and `Queue.json` — which requirement, strategy, and revision ran?
2. `Contributions.jsonl` — why this result counted as novel progress (or why it did not).
3. Latest `Runs/*/input.json` and `result.json` — exact context, criteria, errors, and next steps.
4. Protocol/derivation/goal reviews in the run directory — which independent gate accepted it?
5. `ModelCalls.jsonl` — profile, input size, tokens, latency, schema, and HTTP failure.
6. Evidence artifact named by the finding — exact quote, reviewed derivation, Lean module/log, or
   condition-level experiment output.

This separation makes an engineering failure, missing source, failed proof, and genuine mathematical
obstruction visibly different states.
