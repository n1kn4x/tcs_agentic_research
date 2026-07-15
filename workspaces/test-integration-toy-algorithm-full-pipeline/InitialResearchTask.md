# Test Research: Full-Pipeline Toy Algorithm Study

## Purpose of this test research
This workspace is an integration test for the full artifact-driven research loop. It should exercise literature context, experimenter artifacts, LEAP formalization, proposal/obligation handling, and conservative claim validation on a deliberately simple algorithmic topic.

## Research question
Study the elementary majority problem for finite Boolean lists: given a list of Boolean values, decide whether strictly more than half of the entries are `true`. Produce a small literature/context note, run a toy experiment, and formally verify one simple supporting lemma.

## Scope
The task is intentionally modest. The system should investigate:
- a precise definition of majority on finite Boolean lists;
- a simple linear-time counting algorithm;
- small empirical tests comparing the counting algorithm with a naive specification checker;
- one formalizable lemma, such as correctness of a helper count function on concatenated lists or a simple property of Boolean counts.

## Success criteria
A successful run should produce:
1. a normalized task formulation and notation notes;
2. at least one literature/context query or citation if relevant, while acknowledging that this is textbook material;
3. a reproducible small experiment over random Boolean lists with fixed seeds;
4. at least one Lean proof attempt for a simple supporting lemma;
5. a final report that separates proved facts, empirical observations, and informal context.

## Required subsystem emphasis
- Literature subsystem: gather lightweight context or identify that the result is textbook/basic.
- Experimenter subsystem: run small reproducible tests and import artifacts.
- LEAP subsystem: attempt a concrete Lean proof for a small lemma.
- Core workflow: obligations should track the separate literature, experiment, and proof tasks.

## Constraints
- Do not claim novelty.
- Do not claim that experiments prove correctness.
- Do not require sophisticated majority-circuit or threshold-circuit theory unless the proposal explicitly narrows to that literature.
- Keep the formal theorem small enough for a Lean smoke test.

## Expected fallback outcome
Even if not all subsystems fully succeed, the workspace should contain clear artifacts showing which subsystem steps succeeded and which obligations remain open.
