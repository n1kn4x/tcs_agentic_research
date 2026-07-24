from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from tcs_agentic_research.experimenter.study_harness import execute
from tcs_agentic_research.pipelines.experiment import (
    _execution_defects,
    _replication_defects,
)
from tcs_agentic_research.schemas import (
    ExperimentAnalysisSpec,
    ExperimentBlueprint,
    ExperimentConditionSpec,
    ExperimentDecisionClause,
    ExperimentDecisionSpec,
    ExperimentMechanismCheckSpec,
    ExperimentMetricSpec,
    ExperimentOutput,
    ExperimentReferenceSpec,
    ExperimentResult,
)


def _blueprint() -> ExperimentBlueprint:
    return ExperimentBlueprint(
        title="Trusted feature comparison",
        hypothesis="The treatment uses its registered feature.",
        null_outcome="The treatment feature counter is no larger than baseline.",
        experimental_unit="one deterministic Boolean fixture",
        result_type="boolean",
        result_description="The exact Boolean decision for the fixture.",
        conditions=[
            ExperimentConditionSpec(
                id="treatment",
                role="treatment",
                description="Treatment uses the feature before returning the decision.",
                implementation_requirements=["Increment feature_count on the feature path."],
            ),
            ExperimentConditionSpec(
                id="baseline",
                role="baseline",
                description="Baseline returns the decision without the feature.",
                implementation_requirements=["Keep feature_count equal to zero."],
            ),
        ],
        metrics=[
            ExperimentMetricSpec(
                id="feature_count",
                description="Direct count of feature invocations.",
                value_type="integer",
                role="mechanism",
            )
        ],
        analyses=[
            ExperimentAnalysisSpec(
                id="feature_difference",
                description="Treatment minus baseline mean feature count.",
                operation="paired_mean_difference",
                metric_id="feature_count",
                condition_id="treatment",
                baseline_condition_id="baseline",
            )
        ],
        reference=ExperimentReferenceSpec(
            id="exact_reference",
            kind="exact_reference",
            description="Read the independently generated expected Boolean for every unit.",
        ),
        mechanism_checks=[
            ExperimentMechanismCheckSpec(
                id="feature_discrimination",
                description="Treatment invokes the feature more often than baseline.",
                condition_id="treatment",
                baseline_condition_id="baseline",
                metric_id="feature_count",
                comparison="greater_than_baseline",
                threshold=0,
                fixture_description="A true Boolean fixture that requires the feature path.",
            )
        ],
        sample_size=3,
        seeds=[7],
        generation_plan="Map each unit index to a stable expected Boolean using seed seven.",
        decision_rule=ExperimentDecisionSpec(
            clauses=[
                ExperimentDecisionClause(
                    analysis_id="feature_difference",
                    comparison="greater_than",
                    threshold=0,
                )
            ],
            combine="all",
            outcome_when_met="supports",
            outcome_otherwise="null",
            interpretation="The treatment feature count exceeds the baseline count.",
        ),
        wall_seconds=10,
        memory_mb=128,
        cpus=1,
        limitations=["Synthetic fixtures only."],
    )


def _module(*, treatment_count: int, result_as_object: bool = False) -> SimpleNamespace:
    def make_unit(index: int, seed: int) -> dict[str, object]:
        return {"expected": index % 2 == 0, "index": index, "seed": seed}

    def run_condition(condition_id: str, unit: dict[str, object]) -> dict[str, object]:
        result: object = unit["expected"]
        if result_as_object:
            result = {"answer": result}
        return {
            "result": result,
            "metrics": {
                "feature_count": treatment_count if condition_id == "treatment" else 0
            },
        }

    def reference_result(unit: dict[str, object]) -> bool:
        return bool(unit["expected"])

    def make_mechanism_fixture(check_id: str) -> dict[str, object]:
        assert check_id == "feature_discrimination"
        return {"expected": True, "index": "fixture", "seed": 7}

    return SimpleNamespace(
        make_unit=make_unit,
        run_condition=run_condition,
        reference_result=reference_result,
        make_mechanism_fixture=make_mechanism_fixture,
    )


def test_trusted_harness_owns_coverage_aggregation_and_mechanism_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "implementation", _module(treatment_count=2))

    payload = execute(_blueprint().model_dump(mode="json"), "full")
    output = ExperimentOutput.model_validate(payload)

    assert len(output.observations) == 6
    assert output.aggregate_metrics["feature_difference"] == 2
    assert output.conclusion.outcome == "supports"
    assert all(check.passed for check in output.checks)
    assert len(output.validations) == 8  # six samples plus two fixture-condition rows


def test_trusted_mechanism_fixture_rejects_feature_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "implementation", _module(treatment_count=0))

    output = ExperimentOutput.model_validate(
        execute(_blueprint().model_dump(mode="json"), "smoke")
    )

    mechanism = next(check for check in output.checks if check.name.startswith("mechanism."))
    assert not mechanism.passed
    assert '"treatment":0' in mechanism.detail
    assert '"baseline":0' in mechanism.detail


def test_false_mechanism_assertion_with_correct_results_revises_design(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "implementation", _module(treatment_count=0))
    blueprint = _blueprint()
    output = ExperimentOutput.model_validate(
        execute(blueprint.model_dump(mode="json"), "smoke")
    )

    defects = _execution_defects(
        blueprint,
        ExperimentResult(success=True, summary="executed", validated_output=output),
        smoke=True,
    )

    assert [defect.defect_id for defect in defects] == ["conditions"]
    assert "registered metric assertion" in defects[0].summary


def test_trusted_harness_enforces_primary_result_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules, "implementation", _module(treatment_count=2, result_as_object=True)
    )

    with pytest.raises(TypeError, match="blueprint type boolean"):
        execute(_blueprint().model_dump(mode="json"), "smoke")


def test_replication_ignores_only_harness_owned_wall_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "implementation", _module(treatment_count=2))
    blueprint = _blueprint()
    first = ExperimentOutput.model_validate(execute(blueprint.model_dump(mode="json"), "full"))
    second = ExperimentOutput.model_validate(execute(blueprint.model_dump(mode="json"), "full"))

    assert _replication_defects(
        blueprint,
        ExperimentResult(success=True, summary="first", validated_output=first),
        ExperimentResult(success=True, summary="second", validated_output=second),
    ) == []

    second.observations[0].metrics["feature_count"] = 99
    assert _replication_defects(
        blueprint,
        ExperimentResult(success=True, summary="first", validated_output=first),
        ExperimentResult(success=True, summary="second", validated_output=second),
    )[0].defect_id == "reproducibility"
