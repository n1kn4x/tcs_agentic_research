"""Trusted execution harness for generated study modules.

The generated module implements scientific primitives.  This harness owns iteration, condition
coverage, timing, reference comparisons, aggregation, and the output contract.  Consequently source
code cannot silently skip a condition/unit pair or forge validation coverage.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import statistics
import time
from typing import Any


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"study value is not JSON serializable: {exc}") from exc


def _clone(value: Any) -> Any:
    return json.loads(_canonical(value))


def _check_metric_value(value: Any, value_type: str, metric_id: str) -> None:
    if value_type == "boolean":
        valid = isinstance(value, bool)
    elif value_type == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool)
    else:
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not valid:
        raise TypeError(f"metric {metric_id!r} must have type {value_type}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"metric {metric_id!r} is not finite")


def _check_result_type(value: Any, result_type: str) -> None:
    if result_type == "boolean":
        valid = isinstance(value, bool)
    elif result_type == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif result_type == "real":
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif result_type == "string":
        valid = isinstance(value, str)
    else:
        valid = value is None or isinstance(value, (str, int, float, bool, list, dict))
    if not valid:
        raise TypeError(f"scientific result must have blueprint type {result_type}")


def _condition_result(
    module: Any, condition_id: str, unit: Any, result_type: str
) -> dict[str, Any]:
    returned = module.run_condition(condition_id, _clone(unit))
    if not isinstance(returned, dict) or set(returned) != {"result", "metrics"}:
        raise TypeError("run_condition must return exactly {'result': ..., 'metrics': {...}}")
    _canonical(returned["result"])
    _check_result_type(returned["result"], result_type)
    if not isinstance(returned["metrics"], dict):
        raise TypeError("run_condition metrics must be a dictionary")
    return returned


def _analysis_value(rule: dict[str, Any], observations: list[dict[str, Any]]) -> int | float:
    metric_id = rule["metric_id"]

    def values(condition_id: str) -> dict[int, int | float | bool]:
        selected: dict[int, int | float | bool] = {}
        for row in observations:
            if row["condition"] != condition_id:
                continue
            value = row["metrics"][metric_id]
            if not isinstance(value, (int, float, bool)):
                raise TypeError(f"analysis metric {metric_id!r} is not numeric")
            selected[int(row["unit_id"])] = value
        return selected

    left = values(rule["condition_id"])
    operation = rule["operation"]
    if operation == "true_rate":
        return sum(value is True for value in left.values()) / len(left)
    numeric_left = [float(value) for _, value in sorted(left.items())]
    if operation == "mean":
        return statistics.fmean(numeric_left)
    if operation == "median":
        return statistics.median(numeric_left)
    if operation == "minimum":
        return min(numeric_left)
    if operation == "maximum":
        return max(numeric_left)
    if operation == "sum":
        return sum(numeric_left)

    right = values(rule["baseline_condition_id"])
    if set(left) != set(right):
        raise ValueError("paired analysis conditions do not have identical unit IDs")
    pairs = [(float(left[key]), float(right[key])) for key in sorted(left)]
    if operation == "paired_mean_difference":
        return statistics.fmean(a - b for a, b in pairs)
    if operation == "paired_median_difference":
        return statistics.median(a - b for a, b in pairs)
    if operation == "ratio_of_means":
        denominator = statistics.fmean(b for _, b in pairs)
        if denominator == 0:
            raise ZeroDivisionError(f"analysis {rule['id']!r} has a zero baseline mean")
        return statistics.fmean(a for a, _ in pairs) / denominator
    raise ValueError(f"unsupported trusted analysis operation: {operation}")


def _mechanism_check(module: Any, condition_id: str) -> dict[str, Any]:
    returned = module.check_condition(condition_id)
    if not isinstance(returned, dict):
        raise TypeError("check_condition must return a dictionary")
    if not isinstance(returned.get("passed"), bool):
        raise TypeError("check_condition must return a boolean passed field")
    detail = returned.get("detail", "")
    if not isinstance(detail, str):
        raise TypeError("check_condition detail must be a string")
    evidence = returned.get("evidence", {})
    if not isinstance(evidence, dict):
        raise TypeError("check_condition evidence must be a flat dictionary")
    _canonical(evidence)
    return {"passed": returned["passed"], "detail": detail, "evidence": evidence}


def execute(blueprint: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    module = importlib.import_module("implementation")
    required = ["make_unit", "run_condition"]
    required.append(
        "reference_result"
        if blueprint["reference"]["kind"] == "exact_reference"
        else "validate_result"
    )
    if blueprint["mechanism_checks"]:
        required.append("make_mechanism_fixture")
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise AttributeError("generated study module is missing callables: " + ", ".join(missing))

    count = min(3, blueprint["sample_size"]) if mode == "smoke" else blueprint["sample_size"]
    conditions = blueprint["conditions"]
    metrics = blueprint["metrics"]
    implementation_metrics = {
        row["id"]: row for row in metrics if row["source"] == "implementation"
    }
    wall_metrics = [row for row in metrics if row["source"] == "harness_wall_seconds"]
    units: list[Any] = []
    unit_hashes: list[str] = []
    for index in range(count):
        anchor = blueprint["seeds"][index % len(blueprint["seeds"])]
        unit = module.make_unit(index, anchor)
        encoded = _canonical(unit)
        units.append(json.loads(encoded))
        unit_hashes.append(hashlib.sha256(encoded.encode("utf-8")).hexdigest())

    observations: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    reference_by_unit: dict[int, Any] = {}
    if blueprint["reference"]["kind"] == "exact_reference":
        for unit_id, unit in enumerate(units):
            reference = module.reference_result(_clone(unit))
            _canonical(reference)
            _check_result_type(reference, blueprint["result_type"])
            reference_by_unit[unit_id] = reference

    for condition in conditions:
        condition_id = condition["id"]
        for unit_id, unit in enumerate(units):
            started = time.perf_counter()
            returned = _condition_result(
                module, condition_id, unit, blueprint["result_type"]
            )
            elapsed = time.perf_counter() - started
            actual_metrics = returned["metrics"]
            if set(actual_metrics) != set(implementation_metrics):
                raise ValueError(
                    f"condition {condition_id!r} returned metric IDs {sorted(actual_metrics)}; "
                    f"expected {sorted(implementation_metrics)}"
                )
            for metric_id, metric in implementation_metrics.items():
                _check_metric_value(actual_metrics[metric_id], metric["value_type"], metric_id)
            actual_metrics["harness_wall_seconds"] = elapsed
            for metric in wall_metrics:
                actual_metrics[metric["id"]] = elapsed
            observations.append(
                {
                    "condition": condition_id,
                    "unit_id": unit_id,
                    "result": returned["result"],
                    "sample_size": 1,
                    "metrics": actual_metrics,
                }
            )
            if blueprint["reference"]["kind"] == "exact_reference":
                reference = reference_by_unit[unit_id]
                observed = returned["result"]
            else:
                validation = module.validate_result(
                    _clone(unit), condition_id, _clone(returned["result"])
                )
                if isinstance(validation, bool):
                    validation = {"passed": validation, "detail": ""}
                if not isinstance(validation, dict) or not isinstance(validation.get("passed"), bool):
                    raise TypeError("validate_result must return bool or {'passed': bool, 'detail': str}")
                reference = {"valid": True}
                observed = {"valid": validation["passed"]}
            validations.append(
                {
                    "check_id": blueprint["reference"]["id"],
                    "condition": condition_id,
                    "unit_id": unit_id,
                    "reference": reference,
                    "observed": observed,
                    "detail": "trusted harness comparison",
                }
            )

    reference_passed = all(
        _canonical(row["reference"]) == _canonical(row["observed"]) for row in validations
    )
    checks = [
        {
            "name": blueprint["reference"]["id"],
            "passed": reference_passed,
            "detail": f"trusted harness validated {len(validations)} condition/unit results",
        }
    ]
    for check in blueprint["mechanism_checks"]:
        fixture = module.make_mechanism_fixture(check["id"])
        _canonical(fixture)
        condition_ids = [check["condition_id"]]
        if check.get("baseline_condition_id") is not None:
            condition_ids.append(check["baseline_condition_id"])
        fixture_runs: dict[str, dict[str, Any]] = {}
        fixture_valid = True
        for condition_id in dict.fromkeys(condition_ids):
            returned = _condition_result(
                module, condition_id, fixture, blueprint["result_type"]
            )
            actual_metrics = returned["metrics"]
            if set(actual_metrics) != set(implementation_metrics):
                raise ValueError(
                    f"mechanism fixture {check['id']!r} returned metric IDs "
                    f"{sorted(actual_metrics)}; expected {sorted(implementation_metrics)}"
                )
            for metric_id, metric in implementation_metrics.items():
                _check_metric_value(actual_metrics[metric_id], metric["value_type"], metric_id)
            fixture_runs[condition_id] = returned
            if blueprint["reference"]["kind"] == "exact_reference":
                reference = module.reference_result(_clone(fixture))
                _check_result_type(reference, blueprint["result_type"])
                observed = returned["result"]
                valid = _canonical(reference) == _canonical(observed)
            else:
                verdict = module.validate_result(
                    _clone(fixture), condition_id, _clone(returned["result"])
                )
                valid = verdict if isinstance(verdict, bool) else verdict.get("passed") is True
                reference = {"valid": True}
                observed = {"valid": valid}
            fixture_valid = fixture_valid and valid
            validations.append(
                {
                    "check_id": check["id"],
                    "condition": condition_id,
                    "unit_id": f"fixture:{check['id']}",
                    "reference": reference,
                    "observed": observed,
                    "detail": "trusted mechanism-fixture correctness comparison",
                }
            )
        target = fixture_runs[check["condition_id"]]["metrics"][check["metric_id"]]
        comparison = check["comparison"]
        threshold = check["threshold"]
        if comparison == "equal":
            mechanism_passed = float(target) == float(threshold)
        elif comparison == "greater_than":
            mechanism_passed = float(target) > float(threshold)
        elif comparison == "less_than":
            mechanism_passed = float(target) < float(threshold)
        else:
            baseline_id = check["baseline_condition_id"]
            baseline = fixture_runs[baseline_id]["metrics"][check["metric_id"]]
            mechanism_passed = (
                float(target) > float(baseline) + float(threshold)
                if comparison == "greater_than_baseline"
                else float(target) < float(baseline) - float(threshold)
            )
        evidence = {
            condition_id: fixture_runs[condition_id]["metrics"][check["metric_id"]]
            for condition_id in fixture_runs
        }
        checks.append(
            {
                "name": f"mechanism.{check['id']}",
                "passed": mechanism_passed and fixture_valid,
                "detail": (
                    f"trusted {comparison} assertion on {check['metric_id']}; "
                    f"threshold={threshold}; evidence={_canonical(evidence)}; "
                    f"fixture_results_valid={fixture_valid}"
                )[:1000],
            }
        )

    aggregates = {
        rule["id"]: _analysis_value(rule, observations) for rule in blueprint["analyses"]
    }
    decision = blueprint["decision_rule"]
    verdicts: list[bool] = []
    for clause in decision["clauses"]:
        value = float(aggregates[clause["analysis_id"]])
        threshold = float(clause["threshold"])
        comparison = clause["comparison"]
        verdicts.append(
            value < threshold if comparison == "less_than" else
            value > threshold if comparison == "greater_than" else
            abs(value) > threshold if comparison == "absolute_greater_than" else
            abs(value) <= threshold
        )
    decision_met = all(verdicts) if decision["combine"] == "all" else any(verdicts)
    outcome = decision["outcome_when_met"] if decision_met else decision["outcome_otherwise"]
    basis = [clause["analysis_id"] for clause in decision["clauses"]]
    return {
        "schema_version": 2,
        "experiment": blueprint["title"],
        "status": "completed",
        "parameters": {
            "mode": mode,
            "seeds": blueprint["seeds"],
            "unit_ids": list(range(count)),
            "unit_sha256": unit_hashes,
            "blueprint_version": blueprint["schema_version"],
        },
        "aggregate_metrics": aggregates,
        "observations": observations,
        "validations": validations,
        "checks": checks,
        "conclusion": {
            "hypothesis": blueprint["hypothesis"],
            "outcome": outcome,
            "basis_metrics": basis,
            "statement": decision["interpretation"],
        },
        "limitations": blueprint["limitations"],
    }


def main() -> None:
    with open("blueprint.json", encoding="utf-8") as handle:
        blueprint = json.load(handle)
    payload = execute(blueprint, os.environ.get("TCS_EXPERIMENT_MODE", "full"))
    with open("results.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    main()
