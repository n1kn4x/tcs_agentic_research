from __future__ import annotations

import pytest
from pydantic import ValidationError

from tcs_agentic_research.experimenter.validation import validate_experiment_program
from tcs_agentic_research.schemas import ExperimentOutput, ExperimentProgram


VALID_SOURCE = '''
import random

SEEDS = [7]

def run_experiment(mode):
    count = 2 if mode == "smoke" else 5
    rng = random.Random(SEEDS[0])
    observations = []
    for index in range(count):
        observations.append({
            "unit_id": index,
            "condition": "sample",
            "values": {"draw": rng.randint(0, 10)},
        })
    return {
        "schema_version": 1,
        "experiment": "fixed-seed draws",
        "status": "completed",
        "protocol": "Draw bounded pseudo-random integers from one fixed seed.",
        "parameters": {"seed": 7, "count": count},
        "observations": observations,
        "summaries": {"count": count},
        "interpretation": "This records deterministic fixture values only.",
        "limitations": ["Synthetic fixture."],
    }
'''


def test_generic_experiment_has_no_model_owned_pass_or_expected_direction() -> None:
    payload = {
        "schema_version": 1,
        "experiment": "measurement",
        "protocol": "Measure one exact bounded computational unit.",
        "observations": [
            {"unit_id": 0, "condition": "control", "values": {"value": 3}}
        ],
        "interpretation": "The observed value was three.",
        "limitations": ["One unit only."],
    }

    output = ExperimentOutput.model_validate(payload)

    assert output.observations[0].values == {"value": 3}
    with pytest.raises(ValidationError):
        ExperimentOutput.model_validate({**payload, "passed": True})


def test_static_validator_accepts_small_self_contained_program() -> None:
    validate_experiment_program(
        ExperimentProgram(description="A deterministic bounded fixture.", source=VALID_SOURCE, seeds=[7])
    )


@pytest.mark.parametrize(
    "source, message",
    [
        ("import subprocess\n" + VALID_SOURCE, "forbidden module"),
        ("open('/tmp/escape', 'w')\n" + VALID_SOURCE, "top level"),
        ("def run_experiment(mode):\n    pass\n", "placeholder"),
    ],
)
def test_static_validator_rejects_escape_and_placeholder_programs(source, message) -> None:
    program = ExperimentProgram(
        description="An invalid generated experiment fixture.", source=source, seeds=[7]
    )
    with pytest.raises(ValueError, match=message):
        validate_experiment_program(program)
