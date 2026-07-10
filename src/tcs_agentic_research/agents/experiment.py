"""Reproducible experiment and small-instance search harness."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from ..artifact_store import ArtifactStore
from ..schemas import ArtifactRef, ExperimentResult, new_id, utc_now


class ExperimentAgent:
    def __init__(self, store: ArtifactStore):
        self.store = store

    def record_request(self, *, description: str) -> tuple[str, list[ArtifactRef]]:
        """Record a natural-language experiment request for a future coding backend.

        This deliberately does not create an ``ExperimentResult`` and does not certify evidence.
        The native research tool can expose this as a clean description-only interface now, while
        a later coding-agent implementation can replace the executor behind the same tool name.
        """
        request_id = new_id("experiment_request")
        safe_name = "requested_experiment"
        rel_dir = f"ExperimentRuns/{safe_name}_{request_id}"
        run_dir = self.store.resolve(rel_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        request = {
            "request_id": request_id,
            "description": description,
            "status": "backend_not_configured",
            "created_at": utc_now(),
        }
        ref = self.store.write_json(Path(rel_dir) / "request.json", request)
        return request_id, [ref]

    def run_python(
        self,
        *,
        name: str,
        code: str,
        config: dict[str, Any] | None = None,
        seeds: list[int] | None = None,
        timeout_seconds: int = 3600,
    ) -> ExperimentResult:
        """Write a Python script into the run directory and execute it reproducibly."""
        run_id = new_id("run")
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:80]
        rel_dir = f"ExperimentRuns/{safe_name}_{run_id}"
        run_dir = self.store.resolve(rel_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        script = run_dir / "script.py"
        script.write_text(code, encoding="utf-8")
        result = self.run_command(
            name=f"{safe_name}_exec",
            command=["python", str(script)],
            config=config,
            seeds=seeds,
            timeout_seconds=timeout_seconds,
            existing_rel_dir=rel_dir,
            existing_run_id=run_id,
        )
        result.artifact_refs.append(self.store.artifact_ref(Path(rel_dir) / "script.py"))
        return result

    def run_command(
        self,
        *,
        name: str,
        command: list[str],
        config: dict[str, Any] | None = None,
        seeds: list[int] | None = None,
        timeout_seconds: int = 3600,
        existing_rel_dir: str | None = None,
        existing_run_id: str | None = None,
    ) -> ExperimentResult:
        """Run a deterministic command under ``ExperimentRuns/`` and capture artifacts."""
        run_id = existing_run_id or new_id("run")
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:80]
        rel_dir = existing_rel_dir or f"ExperimentRuns/{safe_name}_{run_id}"
        run_dir = self.store.resolve(rel_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(config or {}, indent=2, sort_keys=True), encoding="utf-8")
        (run_dir / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
        env = os.environ.copy()
        if seeds:
            env["TCS_EXPERIMENT_SEEDS"] = ",".join(str(s) for s in seeds)
        completed = subprocess.run(
            command,
            cwd=run_dir,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        summary = (
            f"Command exited with code {completed.returncode}. Experiments are conjecture-generating "
            "unless paired with proof or exhaustive certified search."
        )
        return ExperimentResult(
            run_id=run_id,
            summary=summary,
            artifact_refs=[
                self.store.artifact_ref(Path(rel_dir) / "config.json"),
                self.store.artifact_ref(Path(rel_dir) / "command.json"),
                self.store.artifact_ref(Path(rel_dir) / "stdout.log"),
                self.store.artifact_ref(Path(rel_dir) / "stderr.log"),
            ],
            seeds=seeds or [],
            caveats=["Experimental evidence is not a mathematical proof."],
        )
