"""Deprecated v2 validation helpers used only by old API tests/artifacts.

The v3 campaign never calls these natural-language heuristics. All live experiment control
semantics are typed in :mod:`pipelines.experiment` and the trusted study harness.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..schemas import (
    ExperimentCriterionAssessment,
    ExperimentEvidenceReview,
    ExperimentObservation,
    ExperimentOutput,
    ExperimentProtocol,
    ExperimentProtocolReview,
    NamedDescription,
    WorkItem,
)

def _protocol_semantic_errors(
    item: WorkItem, protocol: ExperimentProtocol
) -> list[str]:
    """Enforce independent anchors and keep result direction out of correctness checks."""
    requires_validation = bool(
        re.search(
            r"(?i)(?:correctness|validat(?:e|ion|ing)|known cases?|oracle)",
            " ".join([item.instruction, *item.success_criteria]),
        )
    )
    check_text = " ".join(check.description for check in protocol.correctness_checks)
    independent = bool(
        re.search(
            r"(?i)(?:known|oracle|ground truth|expected (?:answer|output|value)|exhaustive|"
            r"brute[- ]force|round[- ]?trip|invariant|analytical result|reference implementation|"
            r"edge case)",
            check_text,
        )
    )
    errors: list[str] = []
    missing_condition_integrity: list[str] = []
    for condition in protocol.conditions:
        direct_checks = [
            check.description
            for check in protocol.correctness_checks
            if _check_targets_condition(check, condition.id)
            and re.search(
                r"(?i)(?:instrument|counter|trace|dispatch|feature flag|"
                r"branch(?:ing)? (?:path|choice|variable|decision)|assign(?:ment)?|forced|"
                r"propagat(?:e|ion)|invok(?:e|ed|ation)|call(?:ed| count)|mechanism|"
                r"must (?:remain|be) (?:zero|disabled))",
                check.description,
            )
        ]
        if not direct_checks:
            missing_condition_integrity.append(condition.id)
    if missing_condition_integrity:
        errors.append(
            "P_CONDITIONS: Add a direct computed counter, trace, dispatch, or feature-flag check "
            "that explicitly names each condition (including features that must remain disabled): "
            + ", ".join(missing_condition_integrity)
        )
    if requires_validation and not independent:
        errors.append(
            "P_CHECKS: Add at least one independent known-case, exhaustive-oracle, round-trip, "
            "invariant, or ground-truth correctness check; cross-condition agreement alone can "
            "allow every implementation to be wrong."
        )
    full_sample_validation = any(
        _check_requires_full_sample_evidence(check.description)
        for check in protocol.correctness_checks
    )
    if requires_validation and not full_sample_validation:
        errors.append(
            "P_CHECKS: Validate every sampled unit against an independent oracle, reference, "
            "round-trip, ground truth, or invariant; known fixtures alone do not validate the "
            "scientific measurements."
        )
    directional_ids = [
        check.id
        for check in protocol.correctness_checks
        if re.search(
            r"(?i)(?:outperform|faster|slower|fewer|more nodes|lower (?:cost|runtime|time)|"
            r"higher (?:accuracy|rate)|reduce[sd]? (?:cost|runtime|search|nodes)|"
            r"improve[sd]?|statistically significant|better than|worse than)",
            check.description,
        )
    ]
    if directional_ids:
        errors.append(
            "P_CHECKS: Move expected performance direction out of correctness checks and into the "
            "decision rule; negative and null results must still pass implementation validation: "
            + ", ".join(directional_ids)
        )
    noisy_determinism_ids = [
        check.id
        for check in protocol.correctness_checks
        if _requires_repeated_timing(check.description)
    ]
    if noisy_determinism_ids:
        errors.append(
            "P_CHECKS: Determinism checks may compare generated inputs, outputs, and operation counts "
            "but not wall-clock timings: " + ", ".join(noisy_determinism_ids)
        )
    analysis_text = " ".join(
        f"{metric.id} {metric.description}" for metric in protocol.analysis_metrics
    )
    required_analysis_concepts = [
        (r"(?i)\b(?:confidence interval|\d+%\s*ci|ci\b)", r"(?i)\b(?:confidence interval|ci\b)", "confidence interval"),
        (r"(?i)\bp[- ]?values?\b", r"(?i)\bp[- ]?values?\b", "p-value"),
        (
            r"(?i)\beffect sizes?\b",
            r"(?i)\b(?:effect sizes?|cohen(?:'s)? d|mean difference|signed difference|"
            r"odds ratio|risk ratio)\b",
            "effect size",
        ),
    ]
    for decision_pattern, metric_pattern, label in required_analysis_concepts:
        if re.search(decision_pattern, protocol.decision_rule) and not re.search(
            metric_pattern, analysis_text
        ):
            errors.append(
                f"P_ANALYSIS: The decision rule requires a {label}, but no analysis_metrics ID "
                "names that required statistic."
            )
    generation = protocol.unit_generation
    if protocol.sample_size > len(protocol.seeds) and not re.search(
        r"(?i)\b(?:index|counter|hash|derive|prng|random stream|seedsequence|spawn)\b",
        generation,
    ):
        errors.append(
            "P_SAMPLING: sample_size exceeds the seed-anchor count; unit_generation must give an "
            "explicit index/hash/PRNG-stream derivation rather than cycling or truncating seeds."
        )
    return errors


def _check_targets_condition(check: NamedDescription, condition_id: str) -> bool:
    """Match explicit condition names across conventional `cond_`/`check_` ID prefixes."""
    haystack = f"{check.id} {check.description}".lower()
    if condition_id.lower() in haystack:
        return True
    signature = re.sub(r"^(?:cond(?:ition)?)[_.-]?", "", condition_id.lower())
    normalized = re.sub(r"^(?:cc|check)[_.-]?", "", check.id.lower())
    return bool(signature and signature in normalized)


def _requires_repeated_timing(description: str) -> bool:
    """Detect requirements for equal repeated timings, not mere deterministic timed fixtures."""
    if not re.search(r"(?i)(?:wall[- ]?clock|runtime|timing|elapsed)", description):
        return False
    if re.search(
        r"(?i)(?:exclude[sd]?|omit(?:ted)?|do not (?:compare|check|require)|not (?:used|included|"
        r"compared|required)|may vary|allowed? to vary|descriptive (?:use|purposes?) only|"
        r"not for determinism)",
        description,
    ):
        return False
    return bool(
        re.search(
            r"(?i)(?:(?:identical|exactly (?:equal|the same)|same)\s+(?:wall[- ]?clock|runtime|"
            r"timing|elapsed)|(?:wall[- ]?clock|runtime|timing|elapsed)[^.]{0,80}"
            r"(?:identical|exactly (?:equal|the same)|must match|reproducible))",
            description,
        )
    )
def _review_errors(
    expected: dict[str, str], review: ExperimentProtocolReview
) -> list[str]:
    errors = _criterion_id_errors(expected, review.criteria)
    errors.extend(
        f"{row.criterion_id}: {row.detail}"
        for row in review.criteria
        if not row.satisfied
    )
    return list(dict.fromkeys(error for error in errors if error))


def _criterion_id_errors(
    expected: dict[str, str], assessments: list[ExperimentCriterionAssessment]
) -> list[str]:
    ids = [row.criterion_id for row in assessments]
    errors: list[str] = []
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    missing = sorted(set(expected) - set(ids))
    unexpected = sorted(set(ids) - set(expected))
    if duplicates:
        errors.append("Duplicate criterion IDs: " + ", ".join(duplicates))
    if missing:
        errors.append("Missing criterion IDs: " + ", ".join(missing))
    if unexpected:
        errors.append("Unexpected criterion IDs: " + ", ".join(unexpected))
    return errors


def _reconcile_evidence_review(
    review: ExperimentEvidenceReview,
    output: ExperimentOutput,
    expected: dict[str, str],
) -> list[str]:
    """Derive overall usability from stable criterion rows rather than a contradictory label."""
    missing = _criterion_id_errors(expected, review.criteria)
    unanimous = not missing and all(assessment.satisfied for assessment in review.criteria)
    if unanimous and review.usable == "unusable":
        # Unanimous criterion rows show that the measurements are interpretable, but they do not
        # erase a separate fatal scientific issue (for example a conclusion whose direction is the
        # opposite of its own aggregates). Preserve such runs as preliminary evidence, never as a
        # requirement-closing result.
        object.__setattr__(review, "usable", "preliminary")
        if not review.follow_up:
            object.__setattr__(
                review,
                "follow_up",
                ["Resolve the review's overall scientific issue without discarding the measurements."],
            )
        ExperimentEvidenceReview.model_validate(review.model_dump())
    elif review.usable == "full" and missing:
        review.usable = "preliminary"
        review.issues.extend(missing)
        review.follow_up.append("Assess every work criterion by its exact id.")
    return missing
def _protocol_output_errors(
    protocol: ExperimentProtocol, output: ExperimentOutput, *, smoke: bool
) -> list[str]:
    """Check frozen-protocol alignment before asking a model to interpret measurements."""
    errors: list[str] = []
    expected_conditions = {condition.id for condition in protocol.conditions}
    observed_ids = [observation.condition for observation in output.observations]
    observed_conditions = set(observed_ids)
    missing_conditions = sorted(expected_conditions - observed_conditions)
    unexpected_conditions = sorted(observed_conditions - expected_conditions)
    if missing_conditions:
        errors.append("Output omitted protocol conditions: " + ", ".join(missing_conditions))
    if unexpected_conditions:
        errors.append("Output added unregistered conditions: " + ", ".join(unexpected_conditions))
    expected_metrics = {metric.id for metric in protocol.metrics}
    for observation in output.observations:
        missing_metrics = sorted(expected_metrics - set(observation.metrics))
        if missing_metrics:
            errors.append(
                f"Observation `{observation.condition}` omitted protocol metrics: "
                + ", ".join(missing_metrics)
            )
    missing_analysis = sorted(
        {metric.id for metric in protocol.analysis_metrics}
        - set(output.aggregate_metrics)
    )
    if missing_analysis:
        errors.append(
            "Output omitted frozen analysis metrics: " + ", ".join(missing_analysis)
        )
    completion_metrics = {
        metric.id
        for metric in protocol.metrics
        if re.search(
            r"(?i)boolean.*true if.*(?:completed|valid result|within (?:the )?(?:time|resource)|"
            r"found .* or proved)",
            metric.description,
        )
    }
    for metric_id in sorted(completion_metrics):
        failed_units = sum(
            observation.sample_size
            for observation in output.observations
            if observation.metrics.get(metric_id) is False
        )
        if failed_units:
            errors.append(
                f"Declared completion metric `{metric_id}` was false for {failed_units} represented "
                "units; do not treat SAT/UNSAT truth as run completion or accept timed-out units as "
                "valid completed measurements."
            )
    represented_by_condition: dict[str, int] = {}
    for condition in expected_conditions & observed_conditions:
        rows = [
            observation
            for observation in output.observations
            if observation.condition == condition
        ]
        aggregated = sum(observation.sample_size != 1 for observation in rows)
        if aggregated:
            errors.append(
                f"Condition `{condition}` has {aggregated} aggregated observation rows; emit one "
                "sample_size=1 record per independent unit."
            )
        represented = sum(observation.sample_size for observation in rows)
        represented_by_condition[condition] = represented
        if smoke and represented > 10:
            errors.append(
                f"Smoke condition `{condition}` represented {represented} units; "
                "the smoke limit is 10."
            )
        if not smoke and represented != protocol.sample_size:
            errors.append(
                f"Full condition `{condition}` represented {represented} units; the frozen "
                f"sample_size requires exactly {protocol.sample_size}."
            )
    expected_units = (
        protocol.sample_size
        if not smoke
        else (max(represented_by_condition.values()) if represented_by_condition else 0)
    )
    unit_ids = output.parameters.get("unit_ids")
    if not isinstance(unit_ids, list):
        errors.append("Output parameters must contain the exact flat `unit_ids` list.")
    else:
        stable_ids = [json.dumps(value, sort_keys=True) for value in unit_ids]
        if any(not isinstance(value, (str, int)) or isinstance(value, bool) for value in unit_ids):
            errors.append("Output unit_ids must contain only string or integer identifiers.")
        if len(unit_ids) != expected_units:
            errors.append(
                f"Output unit_ids has {len(unit_ids)} entries; execution represents "
                f"{expected_units} independent units per condition."
            )
        if len(stable_ids) != len(set(stable_ids)):
            errors.append(
                "Output unit_ids contains duplicates; repeated copies are not independent units."
            )
        expected_id_set = set(stable_ids)
        for condition in sorted(expected_conditions & observed_conditions):
            condition_ids = [
                json.dumps(observation.unit_id, sort_keys=True)
                for observation in output.observations
                if observation.condition == condition
            ]
            if len(condition_ids) != len(set(condition_ids)):
                errors.append(
                    f"Condition `{condition}` repeats observation unit IDs."
                )
            if set(condition_ids) != expected_id_set:
                errors.append(
                    f"Condition `{condition}` observation unit IDs do not exactly match "
                    "parameters.unit_ids."
                )

    expected_checks = {check.id for check in protocol.correctness_checks}
    registered_unit_ids = (
        {json.dumps(value, sort_keys=True) for value in unit_ids}
        if isinstance(unit_ids, list)
        else set()
    )
    full_sample_check_ids = {
        check.id
        for check in protocol.correctness_checks
        if _check_requires_full_sample_evidence(check.description)
    }
    validation_keys: list[tuple[str, str, str]] = []
    for validation in output.validations:
        stable_unit_id = json.dumps(validation.unit_id, sort_keys=True)
        validation_keys.append(
            (validation.check_id, validation.condition, stable_unit_id)
        )
        if validation.check_id not in expected_checks:
            errors.append(
                f"Validation row uses unregistered check `{validation.check_id}`."
            )
        if validation.check_id in full_sample_check_ids:
            if validation.condition not in expected_conditions:
                errors.append(
                    f"Full-sample validation uses unregistered condition `{validation.condition}`."
                )
            if registered_unit_ids and stable_unit_id not in registered_unit_ids:
                errors.append(
                    f"Full-sample validation uses unregistered unit ID `{validation.unit_id}`."
                )
    duplicate_validation_keys = sorted(
        {key for key in validation_keys if validation_keys.count(key) > 1}
    )
    if duplicate_validation_keys:
        errors.append(
            f"Validation evidence repeats {len(duplicate_validation_keys)} check/condition/unit rows."
        )
    for check in protocol.correctness_checks:
        if check.id not in full_sample_check_ids:
            continue
        named_targets = {
            condition.id
            for condition in protocol.conditions
            if condition.id.lower() in check.description.lower()
        }
        target_conditions = named_targets or expected_conditions
        required_keys = {
            (check.id, condition_id, stable_unit_id)
            for condition_id in target_conditions
            for stable_unit_id in registered_unit_ids
        }
        actual_keys = {
            key for key in validation_keys if key[0] == check.id
        }
        missing_keys = required_keys - actual_keys
        if missing_keys:
            errors.append(
                f"Full-sample check `{check.id}` omitted {len(missing_keys)} required "
                "condition/unit validation rows."
            )
        matching_rows = [
            row for row in output.validations if row.check_id == check.id
        ]
        mismatches = [
            row
            for row in matching_rows
            if json.dumps(row.reference, sort_keys=True)
            != json.dumps(row.observed, sort_keys=True)
        ]
        if mismatches:
            errors.append(
                f"Full-sample check `{check.id}` has {len(mismatches)} reference/observed mismatches."
            )
        if re.search(
            r"(?i)(?:oracle|reference implementation|ground truth|exhaustive)",
            check.description,
        ):
            observation_results = {
                (
                    observation.condition,
                    json.dumps(observation.unit_id, sort_keys=True),
                ): observation.result
                for observation in output.observations
            }
            proxy_rows = [
                row
                for row in matching_rows
                if json.dumps(row.observed, sort_keys=True)
                != json.dumps(
                    observation_results.get(
                        (row.condition, json.dumps(row.unit_id, sort_keys=True))
                    ),
                    sort_keys=True,
                )
            ]
            if proxy_rows:
                errors.append(
                    f"Full-sample oracle check `{check.id}` has {len(proxy_rows)} observed values "
                    "that differ from the condition observation result; do not substitute a "
                    "completion flag or proxy."
                )

    check_names = [check.name for check in output.checks]
    actual_checks = set(check_names)
    missing_checks = sorted(expected_checks - actual_checks)
    if missing_checks:
        errors.append("Output omitted protocol correctness checks: " + ", ".join(missing_checks))
    duplicate_checks = sorted(name for name in actual_checks if check_names.count(name) > 1)
    if duplicate_checks:
        errors.append("Output repeated protocol correctness checks: " + ", ".join(duplicate_checks))
    if _normalized_text(output.conclusion.hypothesis) != _normalized_text(protocol.hypothesis):
        errors.append("Output conclusion changed the frozen protocol hypothesis.")
    return errors


def _check_requires_full_sample_evidence(description: str) -> bool:
    return bool(
        re.search(
            r"(?i)(?:every|each|all)(?:\s+of\s+the\s+\d+)?\s+"
            r"(?:sampled |generated |experimental )?"
            r"(?:unit|instance|sample|observation|input|record)s?",
            description,
        )
        and re.search(
            r"(?i)(?:oracle|reference implementation|round[- ]?trip|ground truth|invariant|"
            r"exhaustive)",
            description,
        )
    )
def _normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _evidence_output_context(output: ExperimentOutput) -> dict[str, Any]:
    """Bound raw observations into auditable summaries for the semantic review call.

    The unmodified payload remains in ``results.json``.  This context includes weighted numeric
    summaries, categorical counts, and examples so a long experiment cannot overflow later model
    calls merely by preserving more raw replicates.
    """
    by_condition: dict[str, list[ExperimentObservation]] = {}
    for observation in output.observations:
        by_condition.setdefault(observation.condition, []).append(observation)

    summaries: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for condition, rows in sorted(by_condition.items()):
        metric_names = sorted({name for row in rows for name in row.metrics})[:12]
        metrics: dict[str, Any] = {}
        for name in metric_names:
            values = [
                (row.metrics[name], row.sample_size)
                for row in rows
                if name in row.metrics and row.metrics[name] is not None
            ]
            numeric = [
                (float(value), weight)
                for value, weight in values
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            if numeric and len(numeric) == len(values):
                total_weight = sum(weight for _, weight in numeric)
                metrics[name] = {
                    "count": total_weight,
                    "mean": sum(value * weight for value, weight in numeric) / total_weight,
                    "min": min(value for value, _ in numeric),
                    "max": max(value for value, _ in numeric),
                }
            else:
                counts: dict[str, int] = {}
                for value, weight in values:
                    key = json.dumps(value, ensure_ascii=False, sort_keys=True)
                    counts[key] = counts.get(key, 0) + weight
                metrics[name] = {"counts": dict(list(sorted(counts.items()))[:8])}
        summaries.append(
            {
                "condition": condition,
                "record_count": len(rows),
                "represented_sample_size": sum(row.sample_size for row in rows),
                "metrics": metrics,
            }
        )
        examples.extend(row.model_dump(mode="json") for row in rows[:1])

    return {
        "schema_version": output.schema_version,
        "experiment": output.experiment,
        "status": output.status,
        "parameters": dict(list(output.parameters.items())[:30]),
        "aggregate_metrics": dict(list(output.aggregate_metrics.items())[:40]),
        "observation_summaries": summaries,
        "observation_examples": examples,
        "validation_summary": {
            "row_count": len(output.validations),
            "by_check_condition": {
                f"{check_id}/{condition}": {
                    "count": len(rows),
                    "mismatches": sum(
                        json.dumps(row.reference, sort_keys=True)
                        != json.dumps(row.observed, sort_keys=True)
                        for row in rows
                    ),
                }
                for (check_id, condition), rows in {
                    key: [
                        row
                        for row in output.validations
                        if (row.check_id, row.condition) == key
                    ]
                    for key in {
                        (row.check_id, row.condition)
                        for row in output.validations
                    }
                }.items()
            },
            "examples": [
                row.model_dump(mode="json") for row in output.validations[:6]
            ],
        },
        "checks": [
            {
                "name": row.name,
                "passed": row.passed,
                "detail": row.detail[:500],
            }
            for row in output.checks[:20]
        ],
        "conclusion": output.conclusion.model_dump(mode="json"),
        "limitations": [value[:300] for value in output.limitations[:10]],
        "raw_observation_count": len(output.observations),
    }
