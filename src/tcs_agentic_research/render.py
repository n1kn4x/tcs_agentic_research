"""Derived human-readable renderers for structured artifacts."""

from __future__ import annotations

from .schemas import ResearchProposal, ResearchReport, SolvedVerdict


def render_proposal_markdown(proposal: ResearchProposal) -> str:
    risks = "\n".join(
        f"- **{r.severity}**: {r.risk}" + (f" Mitigation: {r.mitigation}" if r.mitigation else "")
        for r in proposal.known_risks_and_barriers
    ) or "- None recorded."
    return f"""# Research Proposal: {proposal.title}

**Proposal ID:** `{proposal.proposal_id}`  
**Kind:** `{proposal.proposal_kind.value}`

## Precise goal
{proposal.precise_goal}

## Assumptions and model
{_bullets(proposal.relevant_assumptions_and_model)}

## Expected intermediate lemmas
{_bullets(proposal.expected_intermediate_lemmas)}

## Algorithmic subgoals
{_bullets(proposal.algorithmic_subgoals)}

## Hypotheses to test
{_bullets(proposal.hypotheses_to_test)}

## Questions to answer
{_bullets(proposal.questions_to_answer)}

## Assertions used as assumptions
{_bullets(proposal.assertions_used_as_assumptions)}

## Must not assume
{_bullets(proposal.must_not_assume)}

## Critic constraints
{_bullets(proposal.critic_constraints)}

## Plausibility argument
{proposal.plausibility_argument}

## Success criteria
{_bullets(proposal.success_criteria)}

## Partial success criteria
{_bullets(proposal.partial_success_criteria)}

## Required tools
{_bullets(proposal.required_tools)}

## Resource model
{proposal.resource_model or "Not specified."}

## Literature queries
{_bullets(proposal.literature_queries)}

## Known risks and barriers
{risks}
"""


def render_report_markdown(report: ResearchReport) -> str:
    claims = "\n".join(
        f"- `{c.claim_id}` [{c.claim_type}/{c.status}]: {c.statement}" for c in report.claims_generated
    ) or "- None."
    obligations = "\n".join(
        f"- `{o.obligation_id}` [{o.status}/{o.suggested_tool}]: {o.statement}"
        for o in report.proof_obligations
    ) or "- None."
    complexities = "\n".join(
        f"- {ce.resource}: {ce.bound} in model `{ce.model}`. {ce.derivation_summary}"
        for ce in report.complexity_estimates
    ) or "- None."
    return f"""# Research Report `{report.report_id}`

**Proposal:** `{report.proposal_id}`  
**Outcome:** `{report.outcome}`

## Executive summary
{report.executive_summary}

## Claims generated
{claims}

## Proof obligations
{obligations}

## Complexity/resource estimates
{complexities}

## Literature dependencies
{_bullets([f"{d.citation_key}: {d.used_for}" for d in report.literature_dependencies])}

## Experimental results
{_bullets([f"{r.run_id}: {r.summary}" for r in report.experimental_results])}

## Unresolved issues
{_bullets(report.unresolved_issues)}

## Proposed next steps
{_bullets(report.proposed_next_steps)}

## Required verifications
{_bullets(report.required_verifications)}
"""


def render_verdict_markdown(verdict: SolvedVerdict) -> str:
    return f"""# Solved Verdict `{verdict.verdict_id}`

**Outcomes:** {", ".join(o.value for o in verdict.outcomes)}  
**Possible breakthrough:** {verdict.possible_breakthrough}  
**Confirmed solved:** {verdict.confirmed_solved}  
**Next action:** `{verdict.next_action}`

## Rationale
{verdict.rationale}

## Blocking issues
{_bullets(verdict.blocking_issues)}
"""


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- None recorded."
