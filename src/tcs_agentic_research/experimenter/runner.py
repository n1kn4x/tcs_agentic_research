"""Execute one model-generated Python program in the bounded Docker experimenter."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Literal

from ..artifact_store import ArtifactStore
from ..schemas import (
    ArtifactRef,
    ExperimentOutput,
    ExperimentProgram,
    ExperimentResult,
    ExperimenterSettings,
    new_id,
    utc_now,
)
from .docker_project import DockerProjectContainer


class BoundedExperimentRunner:
    """Run exactly one Python process; there is no nested coding-agent tool loop."""

    def __init__(self, store: ArtifactStore, settings: ExperimenterSettings | None):
        self.store = store
        self.settings = settings
        self.container = DockerProjectContainer(store, settings)

    def ensure_container(self) -> dict[str, Any]:
        self.container.ensure_running()
        return self.container.status()

    def status(self) -> dict[str, Any]:
        return self.container.status()

    def stop_container(self, *, remove: bool = False) -> None:
        self.container.stop(remove=remove)

    def reset_container(self) -> None:
        self.container.reset()

    def run(
        self,
        *,
        program: ExperimentProgram,
        name: str = "experiment",
        mode: Literal["smoke", "full"] = "full",
        timeout_seconds: int | None = None,
    ) -> ExperimentResult:
        assert self.settings is not None  # DockerProjectContainer already validates this.
        self.container.ensure_running()
        effective_timeout = min(
            timeout_seconds or self.settings.timeout_seconds,
            self.settings.timeout_seconds,
        )
        run_id = new_id("experiment")
        safe_name = _safe_name(name)
        rel_dir = f"ExperimentRuns/{safe_name}_{run_id}"
        canonical_dir = self.store.resolve(rel_dir)
        canonical_dir.mkdir(parents=True, exist_ok=True)
        portable_dir = self.container.workspace_state_dir / "runs" / run_id
        portable_dir.mkdir(parents=True, exist_ok=True)
        try:
            portable_dir.chmod(0o777)
        except PermissionError:
            pass

        request_ref = self.store.write_json(
            Path(rel_dir) / "request.json",
            {
                "run_id": run_id,
                "description": program.description,
                "seeds": program.seeds,
                "mode": mode,
                "expected_outputs": ["results.json"],
                "created_at": utc_now(),
                "execution_contract": {
                    "command": ["python3", "experiment.py"],
                    "network": self.settings.network,
                    "timeout_seconds": effective_timeout,
                    "research_workspace": "/research (read-only)",
                },
            },
        )
        implementation_path = portable_dir / "implementation.py"
        implementation_path.write_text(program.python_code.rstrip() + "\n", encoding="utf-8")
        # The model supplies only the scientific implementation. This trusted wrapper owns the
        # entry point and output path, eliminating a fragile generated-code contract.
        script_path = portable_dir / "experiment.py"
        script_path.write_text(
            "import json\n"
            "import os\n"
            "from implementation import run_experiment\n\n"
            "payload = run_experiment(os.environ.get('TCS_EXPERIMENT_MODE', 'full'))\n"
            "if not isinstance(payload, dict):\n"
            "    raise TypeError('run_experiment must return a dict')\n"
            "with open('results.json', 'w', encoding='utf-8') as handle:\n"
            "    json.dump(payload, handle, ensure_ascii=False, sort_keys=True)\n",
            encoding="utf-8",
        )
        # Redirect inside the bounded mount so a print loop cannot accumulate unbounded output in
        # the host-side docker client. `ulimit -f` caps each generated regular file (roughly 16 MiB).
        shell_command = (
            f"ulimit -f 32768; exec timeout {effective_timeout} "
            "python3 experiment.py > execution.stdout 2> execution.stderr"
        )
        completed = self.container.exec(
            ["sh", "-lc", shell_command],
            workdir=f"/workspace/runs/{run_id}",
            timeout=effective_timeout + 30,
            env={
                "TCS_EXPERIMENT_SEEDS": ",".join(str(seed) for seed in program.seeds),
                "TCS_EXPERIMENT_MODE": mode,
                "MPLBACKEND": "Agg",
            },
            check=False,
        )
        stdout = _read_limited(portable_dir / "execution.stdout", self.settings.max_output_bytes)
        stderr = _read_limited(portable_dir / "execution.stderr", self.settings.max_output_bytes)
        if completed.stderr:
            stderr += "\nDocker exec stderr:\n" + _limit_text(
                completed.stderr, self.settings.max_output_bytes
            )
        missing_outputs = [
            path for path in ["results.json"] if not (portable_dir / path).is_file()
        ]
        output_contract: ExperimentOutput | None = None
        contract_error = ""
        results_path = portable_dir / "results.json"
        if completed.returncode == 0 and not missing_outputs:
            try:
                raw_payload = json.loads(results_path.read_text(encoding="utf-8"))
                output_contract = ExperimentOutput.model_validate(
                    _normalize_output_payload(
                        raw_payload, fallback_experiment=program.description
                    )
                )
                # A valid output is preserved even when a check is false. The evidence reviewer,
                # which sees the protocol and measurements, decides whether this is a genuine
                # implementation failure or a wrongly encoded hypothesis-direction check. Treating
                # every false boolean as a contract failure can silently discard negative results.
            except Exception as exc:  # noqa: BLE001 - this is an untrusted generated artifact
                contract_error = f"invalid results.json contract: {type(exc).__name__}: {exc}"

        shutil.copytree(portable_dir, canonical_dir, dirs_exist_ok=True)
        stdout_ref = self.store.write_text(Path(rel_dir) / "stdout.log", stdout)
        stderr_ref = self.store.write_text(Path(rel_dir) / "stderr.log", stderr)
        result_ref = self.store.write_json(
            Path(rel_dir) / "result.json",
            {
                "run_id": run_id,
                "exit_code": completed.returncode,
                "success": completed.returncode == 0 and not missing_outputs and not contract_error,
                "missing_expected_outputs": missing_outputs,
                "contract_error": contract_error,
                "validated_output": (
                    output_contract.model_dump(mode="json") if output_contract else None
                ),
                "stdout_tail": stdout[-4000:],
                "stderr_tail": stderr[-4000:],
                "finished_at": utc_now(),
            },
        )
        refs = _artifact_refs_recursive(self.store, rel_dir)
        for ref in [request_ref, stdout_ref, stderr_ref, result_ref]:
            if ref.path not in {item.path for item in refs}:
                refs.append(ref)
        success = completed.returncode == 0 and not missing_outputs and not contract_error
        failure_class = "none"
        if success:
            assert output_contract is not None
            metrics = ", ".join(
                f"{key}={_short_value(value)}"
                for key, value in list(output_contract.aggregate_metrics.items())[:12]
            )
            passed = sum(check.passed for check in output_contract.checks)
            summary = (
                f"Experiment completed with {passed}/{len(output_contract.checks)} checks passing. "
                f"Metrics: {metrics}"
            )
        elif completed.returncode in {125, 126, 127}:
            failure_class = "infrastructure"
            summary = (
                f"Experiment infrastructure failed with exit code {completed.returncode}; "
                "the generated program was not treated as the cause."
            )
        elif completed.returncode != 0:
            failure_class = "program"
            summary = f"Experiment program failed with exit code {completed.returncode}."
        elif contract_error:
            failure_class = "contract"
            summary = contract_error
        else:
            failure_class = "contract"
            summary = "Experiment program did not create expected output(s): " + ", ".join(
                missing_outputs
            )
        output_tail = (stdout or stderr).strip()[-1500:]
        if output_tail:
            summary += " Output tail: " + output_tail
        return ExperimentResult(
            run_id=run_id,
            success=success,
            failure_class=failure_class,
            summary=summary,
            validated_output=output_contract,
            artifact_refs=refs,
            seeds=program.seeds,
            caveats=[
                "Experimental evidence is not a mathematical proof.",
                *(
                    [
                        "Reported false check(s), requiring evidence-review classification: "
                        + ", ".join(
                            check.name for check in output_contract.checks if not check.passed
                        )
                    ]
                    if output_contract is not None
                    and any(not check.passed for check in output_contract.checks)
                    else []
                ),
                *(output_contract.limitations if output_contract is not None else []),
            ],
        )


# Kept as an import alias for the thin ExperimentAgent adapter.
PiExperimentRunner = BoundedExperimentRunner


def _normalize_output_payload(
    payload: Any, *, fallback_experiment: str = "generated experiment"
) -> Any:
    """Normalize harmless representation variation without changing scientific values.

    Generated programs often include useful metadata or group aggregate metrics by condition.  The
    canonical contract stays flat and bounded, while the original unmodified ``results.json`` is
    retained as an artifact.  Unknown top-level metadata is ignored; measured leaves, checks,
    conclusions, and observations are never fabricated or force-passed.
    """
    if not isinstance(payload, dict):
        return payload
    allowed = {
        "schema_version",
        "experiment",
        "status",
        "parameters",
        "aggregate_metrics",
        "observations",
        "checks",
        "conclusion",
        "limitations",
    }
    normalized = {key: value for key, value in payload.items() if key in allowed}
    experiment = normalized.get("experiment")
    if experiment is None:
        normalized["experiment"] = fallback_experiment
    elif not isinstance(experiment, str):
        if isinstance(experiment, dict):
            normalized["experiment"] = str(
                experiment.get("title")
                or experiment.get("name")
                or experiment.get("description")
                or json.dumps(experiment, ensure_ascii=False, sort_keys=True)
            )
        else:
            normalized["experiment"] = str(experiment)
    normalized["parameters"] = _flatten_mapping(
        normalized.get("parameters"), preserve_scalar_lists=True
    )
    normalized["aggregate_metrics"] = _flatten_mapping(
        normalized.get("aggregate_metrics"), preserve_scalar_lists=False
    )
    observations = normalized.get("observations")
    if isinstance(observations, list):
        normalized["observations"] = [
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key in {"condition", "sample_size", "metrics"}
                },
                **(
                    {"condition": row["condition_id"]}
                    if "condition" not in row and "condition_id" in row
                    else {}
                ),
                "metrics": _flatten_mapping(
                    row.get("metrics"), preserve_scalar_lists=False
                ),
            }
            if isinstance(row, dict)
            else row
            for row in observations
        ]
    checks = normalized.get("checks")
    if isinstance(checks, list):
        # A generated implementation may naturally record one check result per known case. Fold
        # those records into the protocol's one aggregate decision without hiding any failure.
        grouped: dict[str, list[dict[str, Any]]] = {}
        ungrouped: list[Any] = []
        for row in checks:
            if isinstance(row, dict):
                name = row.get("name") or row.get("check_id")
                if isinstance(name, str):
                    grouped.setdefault(name, []).append(row)
                    continue
            ungrouped.append(row)
        normalized["checks"] = [
            {
                "name": name,
                "passed": all(row.get("passed") is True for row in rows),
                "detail": " | ".join(
                    str(row.get("detail") or "").strip()
                    for row in rows
                    if str(row.get("detail") or "").strip()
                )[:1000],
            }
            for name, rows in grouped.items()
        ] + ungrouped
    conclusion = normalized.get("conclusion")
    if isinstance(conclusion, dict):
        conclusion = {
            key: value
            for key, value in conclusion.items()
            if key in {"hypothesis", "outcome", "basis_metrics", "statement"}
        }
        basis = conclusion.get("basis_metrics")
        if isinstance(basis, dict):
            conclusion["basis_metrics"] = list(
                _flatten_mapping(basis, preserve_scalar_lists=False)
            )[:12]
        elif isinstance(basis, str):
            conclusion["basis_metrics"] = [basis]
        elif basis == [] and isinstance(normalized.get("aggregate_metrics"), dict):
            # Metric names are provenance pointers, not new measurements. Recovering them from the
            # already returned aggregate map preserves a negative/null outcome that would otherwise
            # be discarded solely because generated code left this descriptive list empty.
            conclusion["basis_metrics"] = list(normalized["aggregate_metrics"])[:12]
        valid_outcomes = {"supports", "contradicts", "null", "inconclusive", "characterizes"}
        if conclusion.get("outcome") not in valid_outcomes:
            conclusion["outcome"] = "inconclusive"
        normalized["conclusion"] = conclusion
    limitations = normalized.get("limitations")
    if isinstance(limitations, str):
        normalized["limitations"] = [limitations]
    elif isinstance(limitations, dict):
        normalized["limitations"] = [
            f"{key}: {value}" for key, value in limitations.items()
        ][:20]
    return normalized


def _flatten_mapping(value: Any, *, preserve_scalar_lists: bool) -> Any:
    if not isinstance(value, dict):
        return value
    flat: dict[str, Any] = {}

    def visit(prefix: str, item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)
            return
        if isinstance(item, list) and not (
            preserve_scalar_lists
            and all(child is None or isinstance(child, (str, int, float, bool)) for child in item)
        ):
            for index, child in enumerate(item):
                visit(f"{prefix}.{index}", child)
            return
        flat[prefix] = item

    for key, item in value.items():
        visit(str(key), item)
    return flat


def _short_value(value: Any, *, limit: int = 240) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return text if len(text) <= limit else text[:limit] + "..."


def _read_limited(path: Path, max_bytes: int) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        data = handle.read(max(0, max_bytes) + 1 if max_bytes > 0 else -1)
    truncated = max_bytes > 0 and len(data) > max_bytes
    if truncated:
        data = data[:max_bytes]
    text = data.decode("utf-8", errors="replace")
    return text + ("\n...[output file truncated by host importer]\n" if truncated else "")


def _limit_text(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if max_bytes <= 0 or len(encoded) <= max_bytes:
        return text
    kept = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return kept + f"\n...[truncated {len(encoded) - max_bytes} bytes]\n"


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_.-")
    return (safe or "experiment")[:80]


def _artifact_refs_recursive(
    store: ArtifactStore, rel_dir: str, *, limit: int = 250
) -> list[ArtifactRef]:
    root = store.resolve(rel_dir)
    refs: list[ArtifactRef] = [store.artifact_ref(root, summary="Experiment run directory")]
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files[:limit]:
        refs.append(store.artifact_ref(path))
    if len(files) > limit:
        manifest = root / "artifact_manifest_truncated.json"
        manifest.write_text(
            json.dumps(
                {
                    "included_file_refs": limit,
                    "omitted_file_count": len(files) - limit,
                    "all_files": [str(path.relative_to(root)) for path in files],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        refs.append(store.artifact_ref(manifest))
    unique: dict[str, ArtifactRef] = {ref.path: ref for ref in refs}
    return list(unique.values())
