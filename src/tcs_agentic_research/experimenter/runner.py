"""Execute one self-contained experiment in a bounded, networkless container."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Literal

from ..artifact_store import ArtifactStore
from ..schemas import ExperimentOutput, ExperimentProgram, ExperimentResult, ExperimenterSettings, new_id, utc_now
from .docker_project import DockerProjectContainer
from .validation import validate_experiment_program


class BoundedExperimentRunner:
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
        validate_experiment_program(program)
        if self.settings is None:
            raise RuntimeError("experimenter is not configured")
        self.container.ensure_running()
        timeout = min(timeout_seconds or self.settings.timeout_seconds, self.settings.timeout_seconds)
        run_id = new_id("experiment")
        rel_dir = f"ExperimentRuns/{_safe_name(name)}_{run_id}"
        canonical = self.store.resolve(rel_dir)
        canonical.mkdir(parents=True, exist_ok=True)
        portable = self.container.workspace_state_dir / "runs" / run_id
        portable.mkdir(parents=True, exist_ok=True)
        try:
            portable.chmod(0o777)
        except PermissionError:
            pass

        self.store.write_json(
            Path(rel_dir) / "request.json",
            {
                "run_id": run_id,
                "description": program.description,
                "seeds": program.seeds,
                "mode": mode,
                "timeout_seconds": timeout,
                "network": self.settings.network,
                "created_at": utc_now(),
            },
        )
        (portable / "implementation.py").write_text(
            program.python_code.rstrip() + "\n", encoding="utf-8"
        )
        (portable / "experiment.py").write_text(
            "import json\n"
            "import os\n"
            "from implementation import run_experiment\n\n"
            "payload = run_experiment(os.environ.get('TCS_EXPERIMENT_MODE', 'full'))\n"
            "if not isinstance(payload, dict):\n"
            "    raise TypeError('run_experiment must return a dict')\n"
            "with open('results.json', 'w', encoding='utf-8') as handle:\n"
            "    json.dump(payload, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)\n",
            encoding="utf-8",
        )
        completed = self.container.exec(
            [
                "sh",
                "-lc",
                f"ulimit -f 32768; exec timeout {timeout} python3 experiment.py "
                "> execution.stdout 2> execution.stderr",
            ],
            workdir=f"/workspace/runs/{run_id}",
            timeout=timeout + 30,
            env={
                "TCS_EXPERIMENT_SEEDS": ",".join(str(seed) for seed in program.seeds),
                "TCS_EXPERIMENT_MODE": mode,
                "MPLBACKEND": "Agg",
            },
            check=False,
        )
        stdout = _read_limited(portable / "execution.stdout", self.settings.max_output_bytes)
        stderr = _read_limited(portable / "execution.stderr", self.settings.max_output_bytes)
        if completed.stderr:
            stderr += "\nDocker exec stderr:\n" + completed.stderr[-self.settings.max_output_bytes :]

        output: ExperimentOutput | None = None
        contract_error = ""
        results_path = portable / "results.json"
        if completed.returncode == 0 and results_path.is_file():
            try:
                output = ExperimentOutput.model_validate_json(results_path.read_text(encoding="utf-8"))
                _write_standard_artifacts(portable, output)
            except Exception as exc:  # untrusted generated output
                contract_error = f"invalid results.json: {type(exc).__name__}: {exc}"
        elif completed.returncode == 0:
            contract_error = "experiment did not create results.json"

        shutil.copytree(portable, canonical, dirs_exist_ok=True)
        self.store.write_text(Path(rel_dir) / "stdout.log", stdout)
        self.store.write_text(Path(rel_dir) / "stderr.log", stderr)
        success = completed.returncode == 0 and output is not None and not contract_error
        failure_class: Literal["none", "infrastructure", "program", "contract"] = "none"
        if success:
            assert output is not None
            summary = (
                f"Experiment produced {len(output.observations)} raw observation(s) and "
                f"{len(output.summaries)} summary value(s)."
            )
        elif completed.returncode in {125, 126, 127}:
            failure_class = "infrastructure"
            summary = f"Experiment infrastructure failed with exit code {completed.returncode}."
        elif completed.returncode != 0:
            failure_class = "program"
            summary = f"Experiment program failed with exit code {completed.returncode}."
        else:
            failure_class = "contract"
            summary = contract_error
        if stderr.strip():
            summary += " Stderr tail: " + stderr.strip()[-1200:]
        self.store.write_json(
            Path(rel_dir) / "result.json",
            {
                "run_id": run_id,
                "exit_code": completed.returncode,
                "success": success,
                "failure_class": failure_class,
                "contract_error": contract_error,
                "validated_output": output,
                "finished_at": utc_now(),
            },
        )
        refs = _artifact_refs(self.store, rel_dir)
        return ExperimentResult(
            run_id=run_id,
            success=success,
            failure_class=failure_class,
            summary=summary,
            validated_output=output,
            artifact_refs=refs,
            seeds=program.seeds,
            caveats=[
                "Execution validates reproducibility of this program, not scientific design or causality.",
                *(output.limitations if output else []),
            ],
        )


PiExperimentRunner = BoundedExperimentRunner


def _write_standard_artifacts(run_dir: Path, output: ExperimentOutput) -> None:
    keys = sorted({key for observation in output.observations for key in observation.values})
    with (run_dir / "observations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["unit_id", "condition", *keys])
        writer.writeheader()
        for observation in output.observations:
            row: dict[str, Any] = {
                "unit_id": observation.unit_id,
                "condition": observation.condition,
            }
            row.update(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in observation.values.items()
                }
            )
            writer.writerow(row)
    report = [
        f"# {output.experiment}",
        "",
        "## Protocol",
        output.protocol,
        "",
        "## Execution",
        f"- Status: `{output.status}`",
        f"- Raw observations: {len(output.observations)}",
        "",
        "## Interpretation supplied by the experiment program",
        output.interpretation,
        "",
        "This interpretation is not independently verified by the execution runner.",
        "",
        "## Limitations",
        *[f"- {item}" for item in output.limitations],
        "",
    ]
    (run_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


def _artifact_refs(store: ArtifactStore, rel_dir: str) -> list:
    return [store.artifact_ref(path) for path in sorted(store.resolve(rel_dir).rglob("*")) if path.is_file()]


def _read_limited(path: Path, limit: int) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    if len(data) > limit:
        data = data[-limit:]
        prefix = f"[truncated to last {limit} bytes]\n".encode()
    else:
        prefix = b""
    return (prefix + data).decode("utf-8", errors="replace")


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value)[:80]
    return safe or "experiment"
