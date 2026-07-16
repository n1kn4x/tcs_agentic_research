"""Execute one model-generated Python program in the bounded Docker experimenter."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from ..artifact_store import ArtifactStore
from ..schemas import (
    ArtifactRef,
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

    def run(self, *, program: ExperimentProgram, name: str = "experiment") -> ExperimentResult:
        assert self.settings is not None  # DockerProjectContainer already validates this.
        self.container.ensure_running()
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
                "expected_outputs": program.expected_outputs,
                "created_at": utc_now(),
                "execution_contract": {
                    "command": ["python", "experiment.py"],
                    "network": self.settings.network,
                    "timeout_seconds": self.settings.timeout_seconds,
                    "research_workspace": "/research (read-only)",
                },
            },
        )
        script_path = portable_dir / "experiment.py"
        script_path.write_text(program.python_code.rstrip() + "\n", encoding="utf-8")
        # Redirect inside the bounded mount so a print loop cannot accumulate unbounded output in
        # the host-side docker client. `ulimit -f` caps each generated regular file (roughly 16 MiB).
        shell_command = (
            f"ulimit -f 32768; exec timeout {self.settings.timeout_seconds} "
            "python experiment.py > execution.stdout 2> execution.stderr"
        )
        completed = self.container.exec(
            ["sh", "-lc", shell_command],
            workdir=f"/workspace/runs/{run_id}",
            timeout=self.settings.timeout_seconds + 30,
            env={
                "TCS_EXPERIMENT_SEEDS": ",".join(str(seed) for seed in program.seeds),
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
            path for path in program.expected_outputs if not (portable_dir / path).is_file()
        ]

        shutil.copytree(portable_dir, canonical_dir, dirs_exist_ok=True)
        stdout_ref = self.store.write_text(Path(rel_dir) / "stdout.log", stdout)
        stderr_ref = self.store.write_text(Path(rel_dir) / "stderr.log", stderr)
        result_ref = self.store.write_json(
            Path(rel_dir) / "result.json",
            {
                "run_id": run_id,
                "exit_code": completed.returncode,
                "success": completed.returncode == 0 and not missing_outputs,
                "missing_expected_outputs": missing_outputs,
                "stdout_tail": stdout[-4000:],
                "stderr_tail": stderr[-4000:],
                "finished_at": utc_now(),
            },
        )
        refs = _artifact_refs_recursive(self.store, rel_dir)
        for ref in [request_ref, stdout_ref, stderr_ref, result_ref]:
            if ref.path not in {item.path for item in refs}:
                refs.append(ref)
        success = completed.returncode == 0 and not missing_outputs
        if success:
            summary = "Experiment program completed successfully."
        elif completed.returncode != 0:
            summary = f"Experiment program failed with exit code {completed.returncode}."
        else:
            summary = "Experiment program did not create expected output(s): " + ", ".join(
                missing_outputs
            )
        output_tail = (stdout or stderr).strip()[-1500:]
        if output_tail:
            summary += " Output tail: " + output_tail
        return ExperimentResult(
            run_id=run_id,
            success=success,
            summary=summary,
            artifact_refs=refs,
            seeds=program.seeds,
            caveats=["Experimental evidence is not a mathematical proof."],
        )


# Kept as an import alias for the thin ExperimentAgent adapter.
PiExperimentRunner = BoundedExperimentRunner


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
