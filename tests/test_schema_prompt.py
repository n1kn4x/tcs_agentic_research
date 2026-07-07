from __future__ import annotations

from tcs_agentic_research.llm import (
    _prepare_structured_messages,
    _schema_prompt,
    schema_placeholder,
)
from tcs_agentic_research.prompt_loader import load_prompt
from tcs_agentic_research.schemas import InitializationBundle, SolvedVerdict


def test_schema_prompt_includes_nested_defs_and_enum_values() -> None:
    text = _schema_prompt(InitializationBundle)

    assert "research_task_markdown" in text
    assert "Complete JSON Schema" in text
    assert '"$defs"' in text
    assert "NomenclatureEntry" in text
    assert "ClaimRecord" in text
    assert "ClaimStatus" in text
    assert "resource_checked" in text
    assert "additionalProperties" in text


def test_schema_named_placeholder_is_replaced_in_structured_messages() -> None:
    placeholder = schema_placeholder(SolvedVerdict)
    messages = [{"role": "system", "content": f"Use this schema:\n{placeholder}"}]

    rendered = _prepare_structured_messages(messages, SolvedVerdict)

    assert placeholder not in rendered[0]["content"]
    assert "SolvedVerdict" in rendered[0]["content"]
    assert "solves_main_task" in rendered[0]["content"]
    assert len(rendered) == 1


def test_structured_schema_is_appended_when_prompt_has_no_matching_placeholder() -> None:
    messages = [{"role": "system", "content": "No placeholder here."}]

    rendered = _prepare_structured_messages(messages, SolvedVerdict)

    assert len(rendered) == 2
    assert rendered[1]["role"] == "user"
    assert "SolvedVerdict" in rendered[1]["content"]
    assert "solves_main_task" in rendered[1]["content"]


def test_all_json_prompts_expose_schema_named_placeholder() -> None:
    prompt_schemas = {
        "independent_replication": "ReplicationResult",
        "initialization_interviewer": "InitializationInterviewTurn",
        "initialization_synthesizer": "InitializationBundle",
        "leap_blueprint": "BlueprintCandidate",
        "leap_decomposition_reviewer": "DecompositionReview",
        "leap_direct_prover": "FormalProofCandidate",
        "leap_reviser": "FormalProofCandidate",
        "literature_researcher": "LiteratureExtract",
        "obstruction_agent": "ObstructionResult",
        "proposal_critic": "ProposalCritique",
        "proposal_generator": "ResearchProposal",
        "research_agent": "ResearchReport",
        "research_critic": "ResearchCritique",
        "resource_accountant": "ResourceCheckResult",
        "solved_checker": "SolvedVerdict",
    }

    for name, schema_name in prompt_schemas.items():
        prompt = load_prompt(name)
        assert "{{Schema}}" not in prompt, name
        assert "{{" + schema_name + "}}" in prompt, name
