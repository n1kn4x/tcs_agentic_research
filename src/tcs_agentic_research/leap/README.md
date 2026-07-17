# LEAP subsystem

LEAP is a persistent proof planner around Lean, not a one-shot proving prompt. An **OR node** is a
closed Lean proposition. It is solved by either a verified direct proof or one solved **AND node**.
An AND node is a Lean-verified decomposition whose required child OR nodes must all be solved.
Equivalent child propositions share OR nodes, so proved lemmas survive failed branches and can be
reused.

## Safety invariant

Models may propose informal plans, proof bodies, and child propositions. Only batch Lean compilation
creates formal progress:

- Direct candidates are rendered around the exact application-owned proposition and accepted only
  after placeholder-free compilation.
- A decomposition is rendered by Python. `sorry` appears only in application-owned child bodies;
  the model's parent proof must contain no placeholder and must use every required child.
- Cheap structural checks and an LLM usefulness review both run before a decomposition is committed.
- Graph commits are transactional and reject duplicate children, ancestor restatements, environment
  mismatches, and cycles.
- A final topological, self-contained, no-`sorry` module is batch-compiled before the root result is
  reported as `proved`.

## Files

```text
LeanProject/
├── lean-toolchain                 pinned Lean toolchain
├── lakefile.lean|lakefile.toml    Lake project/dependencies
├── lake-manifest.json             pinned transitive dependency selection, when present
├── TCSResearch/Generated/         candidate, sketch, and final Lean modules
└── LEAP/
    ├── state.sqlite               authoritative graph, runs, immutable attempts
    ├── artifacts/<or-id>/         model plans/candidates/reviews
    └── logs/                      exact compiler output
```

`state.sqlite` uses WAL transactions and expiring per-node worker leases. Attempt rows cannot be
updated or deleted. Calling LEAP again with the same proposition and environment resumes its graph.
A changed toolchain/manifest or local project source receives a
different proposition fingerprint, preventing stale proof reuse.

## Search order

For each open OR node the controller performs:

1. a small compiler-checked deterministic tactic portfolio (`rfl`, `simp`, `decide`);
2. bounded state reading and deterministic library/DAG retrieval;
3. informal mathematical planning;
4. direct proof generation and localized compiler-feedback revisions;
5. blueprint generation after direct failure;
6. deterministic canonical naming of proposed children;
7. formal parent-sketch generation and compiler-feedback revisions;
8. structural progress checks and decomposition review;
9. transactional graph expansion and depth-first child solving with backtracking.

Failed deterministic candidates are fingerprinted in SQLite and are not recompiled when a search is
resumed. Like model-authored candidates, they create progress only after placeholder-free Lean
compilation and final module verification.

On a resumed node, previously accepted decompositions are continued before new branches are
generated. Pausing or exhausting an invocation does not erase open nodes.

## Lake and Mathlib

LEAP creates a minimal Lean-only workspace if `LeanProject/` has no Lake project. For Mathlib or
another library, configure and pin the dependency **before** starting a run. For example:

```toml
name = "tcs_research"
version = "0.1.0"
defaultTargets = ["TCSResearch"]

[[require]]
name = "mathlib"
scope = "leanprover-community"
rev = "<pinned-tag-or-commit>"

[[lean_lib]]
name = "TCSResearch"
```

Then run `lake update`, commit `lean-toolchain` and `lake-manifest.json`, and fetch/build caches as
appropriate. LEAP never runs dependency updates during candidate verification. Retrieval searches
already-proved DAG statements first and then declaration lines in the pinned local package sources.
Fully qualified names/types are passed to formal agents; Lean still decides whether they are in scope.

## Running and resuming

```bash
tcs-research prove --workspace workspaces/demo --config config.yml \
  --name nat_id --statement "∀ n : Nat, n = n"

# Resume with a larger invocation budget
tcs-research prove --workspace workspaces/demo --config config.yml \
  --name nat_id --statement "∀ n : Nat, n = n" \
  --max-model-calls 256 --max-wall-seconds 86400
```

Important limits are under `leap:` in the YAML configuration: direct/blueprint/revision counts,
depth, graph nodes, compiler resources, reviewer threshold, model calls, and wall time. Limits bound
one invocation; they do not delete persistent progress.

## Module map

- `graph.py`: SQLite AND-OR DAG, deduplication, acyclicity, propagation, immutable attempts.
- `state.py`: bounded graph-neighborhood context.
- `retrieval.py`: deterministic graph and local Lean-source retrieval.
- `agents.py`: fresh typed model calls.
- `lean.py`: pinned Lake execution, diagnostics, resource limits, placeholder contracts.
- `render.py`: application-owned candidate/sketch rendering and final topological assembly.
- `controller.py`: resumable DFS, revisions, backtracking, budgets, and success propagation.
- `harness.py`: small public API used by the research engine.
