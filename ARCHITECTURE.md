# Research kernel architecture

## Non-goals

The kernel does **not**:

- decompose a task into questions or deliverables;
- maintain an agenda, requirement board, or work queue;
- decide which scientific method is appropriate;
- ask a model whether another model's prose is true;
- infer that a project is scientifically complete.

Those responsibilities caused the previous system's worst failures. A bad initial decomposition
became permanent, generic experiment schemas leaked concepts between domains, and correlated model
reviews promoted false derivations.

## One abstraction: an autonomous subsystem

A subsystem implements two methods:

```python
propose(shared_view) -> ActionProposal | None
execute(persisted_proposal, shared_view, run_dir) -> ActionOutcome
```

`propose` chooses the subsystem's next atomic move. `execute` owns all domain-specific sequencing.
The kernel only persists the proposal before execution, catches interruption, admits evidence, and
schedules the next subsystem fairly.

Built-in subsystems are independent:

- **literature** chooses searches or local exact-span queries;
- **theory** chooses investigations, challenges, or syntheses;
- **proof** chooses a concrete Lean proposition and invokes LEAP;
- **experiment** chooses a domain-specific protocol, writes one direct program, and replicates it.

A subsystem can be run and tested alone with `--subsystem`. Adding a subsystem requires no changes
to the kernel or to any other subsystem.

## Information flow

```text
InitialResearchTask.md
         |
         v
  bounded shared view <-----------------------------+
         |                                           |
         v                                           |
 subsystem proposes one action                       |
         |                                           |
         v                                           |
 Actions.jsonl: proposed -> running                   |
         |                                           |
         v                                           |
 subsystem executes domain-specific work             |
         |                                           |
         v                                           |
 deterministic evidence admission                    |
         |                                           |
         +--> Records.jsonl (immutable) --------------+
         +--> Subsystems/<name>.json (opaque private state)
         +--> Runs/<cycle>_<subsystem>_<action>/
```

Every subsystem sees compact cards for prior records, including records produced by other
subsystems. Parent record IDs make dependencies explicit. Task edits create a new task revision but
do not erase prior memory, because a workspace represents one continuing research project.

## Epistemic boundary

A record has one of three statuses, assigned only by deterministic policy:

- `tentative`: model-authored question, analysis, synthesis, challenge, or failure diagnosis;
- `observed`: exact validated source span, hashed source metadata, or exactly replicated execution;
- `verified`: placeholder-free Lean theorem accepted by the configured compiler.

There is deliberately no generic `accepted`, `supported`, or `conclusive` status. In particular:

- model confidence does nothing;
- a second model review does nothing;
- a generated report does nothing;
- an experiment program's interpretation remains unverified prose;
- a source quote establishes what the source says, not that the source is correct;
- a Lean parent link expresses intended relevance, while only the proposition is verified.

Contradictions are represented by immutable challenge/counterexample records linked to earlier
records. The kernel does not silently mutate history into a single current truth value.

## Persistence and interruption

Canonical files:

```text
InitialResearchTask.md        current project brief
KernelState.json              scheduler cursor and active action only
TaskVersions.jsonl            append-only task revisions
Actions.jsonl                 append-only action state transitions
Records.jsonl                 append-only cumulative research memory
Events.jsonl                  runtime diagnostics
ModelCalls.jsonl              model telemetry
Subsystems/<name>.json        opaque subsystem-owned continuation state
Runs/...                      exact action inputs and outputs
Reports/{Status,Research}.md  deterministic journal views
```

A proposal is persisted before side effects. If the process dies, the active action is marked
`interrupted` on restart; no claim is inferred from partial artifacts. Identical committed action
fingerprints are not re-executed. Atomic file replacement and a workspace lock protect materialized
state.

## Scheduling and stopping

Scheduling is round-robin. The kernel has no semantic priority function. One actor cannot starve
another by remaining in a resumable internal stage.

`--max-steps` counts subsystem action opportunities. A run returns early when every enabled
subsystem yields once. This is runtime `idle`, not scientific completion. A later invocation asks the
subsystems again, potentially after task edits, new records, imported sources, or human work.

## Experiments

The universal SAT-shaped experiment blueprint and multi-stage repair machine were removed.
An experiment subsystem now emits a direct, domain-specific `run_experiment(mode)` program. The
networkless runner owns process limits and `results.json`; the output contract contains only:

- protocol;
- parameters;
- raw `{unit_id, condition, values}` observations;
- summaries;
- an explicitly unverified interpretation;
- limitations.

There is no generic pass bit or expected-effect assertion. The exact program and fixed seeds run
twice. Only byte-equivalent structured results receive `observed` status. This proves reproducible
execution, not sound methodology or causality.

## Why long runs build rather than churn

- Records are immutable and content-deduplicated.
- Actions are fingerprinted and exact repeats are skipped.
- Every actor receives prior cross-subsystem records.
- Each actor owns a small persistent state rather than forcing its stages into a global schema.
- Failed work becomes a tentative obstruction record, so another actor can respond to it.
- No arbitrary attempt cap erases a promising line, and no global completion bit stops inquiry.
- Only external receipts raise epistemic status, preventing prose volume from masquerading as
  progress.

## Testing

Kernel tests use tiny scripted subsystems. They test fairness, crash recovery, deduplication, task
revision, shared-memory flow, and idle semantics without a model, network, Docker, or Lean.

Evidence-policy tests independently test each trust boundary. Literature, experiment containment,
and LEAP retain their own focused service tests. There is no end-to-end test that hopes a planner
happens to invent the right decomposition.
