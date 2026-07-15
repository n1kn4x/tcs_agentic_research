# Test Research: Experimenter Benchmark for Small-Instance DPLL on 3-SAT

## Purpose of this test research
This workspace is a test case for the Experimenter subsystem. The goal is to force the system to request reproducible code, run small experiments, save artifacts under `ExperimentRuns/`, and avoid treating empirical behavior as proof.

## Research question
Experimentally compare simple SAT-solving strategies on small random 3-SAT instances: naive DPLL, DPLL with unit propagation, and DPLL with a basic branching heuristic.

## Experimental scope
The experiment should use small instance sizes so that runs are fast and reproducible. Suggested parameters:
- variables `n` in a small range such as 8 to 24;
- clause-to-variable ratios around and away from the random 3-SAT threshold;
- multiple fixed random seeds per parameter setting;
- metrics such as solved/unsolved status, recursive calls, runtime, and number of propagated literals.

## Success criteria
A successful run should produce:
1. reproducible Python code implementing the SAT generators and solvers;
2. fixed random seeds and recorded experiment configuration;
3. CSV or JSON results with runtime/recursive-call statistics;
4. at least one plot or table comparing strategies;
5. a concise report explaining empirical trends and limitations.

## Required subsystem emphasis
- The Dockerized experimenter/pi subsystem should be used.
- All experiment artifacts should be imported into `ExperimentRuns/`.
- The final claims should distinguish empirical observations from mathematical proofs.

## Constraints
- Do not claim asymptotic superiority from small experiments.
- Do not benchmark external highly optimized SAT solvers; this is a toy controlled comparison.
- Keep instances small enough for deterministic CI-like smoke testing.
- Any randomized generation must use fixed seeds.

## Expected fallback outcome
A reproducible small benchmark suite and a report explaining which solver variants performed better on the sampled instances, with caveats.
