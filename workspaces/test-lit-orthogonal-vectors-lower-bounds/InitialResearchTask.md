# Test Research: Literature Audit for Orthogonal Vectors Lower Bounds

## Purpose of this test research
This workspace is a test case for the Literature subsystem. The goal is not to prove a new result, but to exercise literature discovery/import, theorem and claim extraction, notation normalization, quote-level provenance, and careful distinction between cited facts and unsupported claims.

## Research question
Audit the literature around SETH-based lower bounds for the Orthogonal Vectors problem. Identify the precise assumptions, reductions, dimension regimes, and running-time barriers that are actually supported by imported sources.

## Background and scope
Orthogonal Vectors (OV) is commonly used in fine-grained complexity reductions. This task should focus on:
- the definition of OV over Boolean vectors;
- the role of dimension, especially logarithmic, polylogarithmic, and polynomial dimension regimes;
- the relationship between OV lower bounds and SETH;
- reductions from or to related problems only when directly relevant;
- avoiding unsourced folklore claims unless marked as conjectural or requiring provenance.

## Success criteria
A successful run should produce:
1. imported or queued literature records for central OV/SETH lower-bound sources;
2. extracted theorem/lower-bound statements with citation keys and quote-level provenance;
3. a normalized explanation of the assumptions and parameter regimes;
4. a list of supported claims versus claims that remain unsupported or ambiguous;
5. concrete follow-up obligations for any missing citations or unclear reductions.

## Required subsystem emphasis
- LiteratureDB search/discovery/import should be used.
- Literature extraction should produce mapped statements and quote provenance where possible.
- The claim ledger must not accept URL-only or unsupported lower-bound claims.

## Constraints
- Do not claim novelty.
- Do not treat SETH, OV conjectures, or conditional lower bounds as unconditional theorems.
- Do not merge different dimension regimes without explicitly stating the regime.
- Prefer precise quoted statements over broad summaries.

## Expected fallback outcome
A structured literature map of OV lower-bound claims, even if some papers cannot be imported automatically.
