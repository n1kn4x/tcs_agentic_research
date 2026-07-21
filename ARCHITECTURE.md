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

`protocol -> protocol review -> code -> smoke -> full run -> evidence review`

Only the protocol and final evidence receive semantic model reviews. Generated code is checked
syntactically and for unsafe operations, must explicitly use `mode`, and must pass tiny smoke
execution before a full run. This avoids subjective code-review loops.

Every stage is persisted under `ExperimentStates/`. Repairs receive exact validator/reviewer/runtime
defects and the prior candidate. Two repairs per outer cycle prevent one experiment from monopolizing
a long run. Repeated identical defects stop after a small per-defect budget.

## Evidence policy

- Literature findings require validated exact source spans.
- Derivations require an adversarial review; substantive requested revisions prevent acceptance.
- Proof findings require placeholder-free Lean verification.
- Experiments require valid condition-level output and a final methodology audit.
- Negative and null evidence follows the same acceptance path as positive evidence.
- Artifact creation and token use are never counted as research progress.
