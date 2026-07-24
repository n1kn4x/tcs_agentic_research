# Research engine architecture

The engine is deliberately a small deterministic shell around four evidence pipelines.

## One research cycle

1. Load the persisted agenda and evidence ledger.
2. Refill a bounded portfolio deterministically from the least-attempted open requirements.
3. Select the least-attempted runnable item, so a resumable pipeline cannot starve other gaps.
4. Give the pipeline a bounded context containing accepted prior findings and one reusable previously
   executed experiment implementation (including explicit audit defects when it needs repair).
5. Run one of `pipelines/{literature,derivation,proof,experiment}.py`.
6. Commit findings, novelty-deduplicated contributions, requirement state, events, and reports
   atomically at the operation boundary.

There is no model call for queue scheduling. A model cannot set requirement status, count progress,
or stop the workspace.

## Long-running behavior

- Process interruption reopens the exact persisted work item.
- Operational/model-budget failures do not consume scientific attempts.
- A blocked experiment marks only its own requirement; unrelated work continues.
- The scheduler refills the portfolio even while another work item is resumable.
- Accepted findings are included in later research prompts, so later work builds on earlier work.
- A previously executed experiment implementation is offered to follow-up work as reusable code.
- The workspace stops only when all mandatory requirements are satisfied or every remaining gap is
  explicitly blocked/exhausted.

## Experiment pipeline

One outer research cycle advances multiple durable stages:

`blueprint -> review -> module -> smoke -> source audit -> full -> replication -> evidence audit`

The blueprint encodes conditions, result type, metrics and their owners, executable aggregate
operations, mechanism fixtures, and the decision rule as typed data. Prose is semantic context only;
Python never regex-parses it to choose a gate.

Generated code supplies scientific primitives. The trusted harness owns unit/condition loops,
coverage, timing, validation rows, typed mechanism comparisons, aggregate calculations, and decision
execution. Therefore a study cannot pass by omitting a condition, inventing validation coverage,
self-reporting a mechanism pass bit, or forging a harness-owned runtime. Full execution is repeated;
scientific results and deterministic metrics must match exactly.

Every stage is persisted under `ExperimentStates/`. Repairs receive the complete source, a structured
defect, and a separately generated typed repair plan before the coding profile emits a complete
replacement. Two repairs per outer cycle preserve scheduler fairness and a cumulative cap bounds a
bad strategy. Preliminary evidence preserves its design, source, measurements, and exact audit
defects in the follow-up campaign. Requirement closure additionally requires an accepted source
audit and an essential design review, so an evidence summarizer cannot waive an implementation flaw.

## Evidence policy

- Literature findings require validated exact named-statement or indexed-passage source spans.
- Derivations require two independent adversarial reviews; either can prevent acceptance.
- Proof findings require placeholder-free Lean verification.
- Experiments require valid condition-level output and a final methodology audit.
- Negative and null evidence follows the same acceptance path as positive evidence.
- Artifact creation and token use are never counted as research progress.
