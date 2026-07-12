"""Run experiments through pi inside the project experimenter container."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from ..artifact_store import ArtifactStore
from ..schemas import ArtifactRef, ExperimentResult, ExperimenterSettings, new_id, utc_now
from .docker_project import DockerProjectContainer
from .errors import ExperimenterRuntimeError


class PiExperimentRunner:
    """Host-side orchestrator for a Dockerized pi coding-agent experiment."""

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
        description: str,
        name: str = "experiment",
        supports_claim_ids: list[str] | None = None,
        timeout_seconds: int | None = None,
    ) -> ExperimentResult:
        if self.settings is None:
            # DockerProjectContainer also checks this; keep type checkers happy and fail early.
            raise ExperimenterRuntimeError("Experimenter settings are missing.")
        self.container.ensure_running()
        run_id = new_id("run")
        safe_name = _safe_name(name or "experiment")
        rel_dir = f"ExperimentRuns/{safe_name}_{run_id}"
        host_run_dir = self.store.resolve(rel_dir)
        host_run_dir.mkdir(parents=True, exist_ok=True)
        container_run_dir = f"/workspace/runs/{run_id}"
        state_run_dir = self.container.workspace_state_dir / "runs" / run_id

        self.container.exec(["mkdir", "-p", container_run_dir], timeout=60, check=True)

        prompt = self._build_prompt(
            run_id=run_id,
            description=description,
            supports_claim_ids=supports_claim_ids or [],
            container_run_dir=container_run_dir,
        )
        prompt_ref = self.store.write_text(Path(rel_dir) / "prompt.md", prompt)
        request_ref = self.store.write_json(
            Path(rel_dir) / "request.json",
            {
                "run_id": run_id,
                "name": name,
                "description": description,
                "supports_claim_ids": supports_claim_ids or [],
                "created_at": utc_now(),
                "experimenter": {
                    "container_name": self.container.container_name,
                    "image": self.settings.image,
                    "research_mount": "/research (read-only)",
                    "workspace_mount": "/workspace (workspace-portable bind mount from .experimenter/workspace)",
                    "coding_agent": "pi",
                },
            },
        )
        self.container.copy_to_container(
            self.store.resolve(Path(rel_dir) / "prompt.md"),
            f"{container_run_dir}/prompt.md",
        )

        pi_command = self._pi_command(run_id=run_id, container_run_dir=container_run_dir)
        command_ref = self.store.write_json(
            Path(rel_dir) / "command.json",
            {
                "container_name": self.container.container_name,
                "workdir": container_run_dir,
                "command": pi_command,
                "timeout_seconds": timeout_seconds or self.settings.timeout_seconds,
            },
        )

        completed = self.container.exec(
            pi_command,
            workdir=container_run_dir,
            timeout=timeout_seconds or self.settings.timeout_seconds,
            env=self.settings.environment,
            check=False,
        )
        stdout_ref = self.store.write_text(
            Path(rel_dir) / "pi_events.jsonl",
            _limit_text(completed.stdout, self.settings.max_output_bytes),
        )
        stderr_ref = self.store.write_text(
            Path(rel_dir) / "pi_stderr.log",
            _limit_text(completed.stderr, self.settings.max_output_bytes),
        )

        # Always harvest the run directory before validating, so infrastructure errors retain
        # whatever files/logs pi created for debugging. /workspace is a bind mount under the
        # copied workspace, so no Docker volume export is required.
        if state_run_dir.exists():
            shutil.copytree(state_run_dir, host_run_dir, dirs_exist_ok=True)

        if completed.returncode != 0:
            raise ExperimenterRuntimeError(
                "pi experiment command failed; harvested artifacts are in "
                f"{rel_dir}. exit_code={completed.returncode}\nSTDERR:\n{completed.stderr[-4000:]}"
            )

        result_payload = self._load_result_payload(Path(rel_dir) / "experiment_result.json")
        refs = _artifact_refs_recursive(self.store, rel_dir)
        for required in [prompt_ref, request_ref, command_ref, stdout_ref, stderr_ref]:
            if required.path not in {ref.path for ref in refs}:
                refs.append(required)

        summary = str(result_payload.get("summary") or "").strip()
        if not summary:
            raise ExperimenterRuntimeError(
                f"Experiment result file exists but has no non-empty `summary`: {rel_dir}/experiment_result.json"
            )
        caveats = [str(item) for item in result_payload.get("caveats", []) if str(item).strip()]
        result_supports = [
            str(item) for item in result_payload.get("supports_claim_ids", supports_claim_ids or [])
        ]
        if "Experimental evidence is not a mathematical proof." not in caveats:
            caveats.append("Experimental evidence is not a mathematical proof.")
        return ExperimentResult(
            run_id=run_id,
            summary=summary,
            artifact_refs=refs,
            supports_claim_ids=result_supports,
            caveats=caveats,
        )

    def _build_prompt(
        self,
        *,
        run_id: str,
        description: str,
        supports_claim_ids: list[str],
        container_run_dir: str,
    ) -> str:
        return f"""You are pi, acting as the experimenter subsystem for an agentic TCS research project.

You are running inside a Docker container with shell access and internet access.

Filesystem contract:
- `/research` is a read-only mount of the canonical research workspace. Read it for context.
- `/workspace` is a writable, workspace-portable bind mount backed by `/research/.experimenter/workspace` on the host. It is copied when the research workspace is copied to another machine.
- `{container_run_dir}` is your writable run directory. Write all scripts, logs, data, plots, and summaries there.
- Do not attempt to modify `/research`; it is mounted read-only by design.
- Prefer user-level/project-local dependencies and caches under `/workspace`; system-level container mutations are not copied with the workspace.

Research context to inspect if relevant:
- `/research/ResearchTask.md`
- `/research/ResearchState.json`
- `/research/Nomenclature.yml`
- `/research/ClaimLedger.jsonl`
- `/research/LiteratureDB/`
- `/research/Reports/`

Experiment request:
{description}

Claim ids this experiment may support, if any:
{json.dumps(supports_claim_ids, indent=2)}

Requirements:
1. Create reproducible code/scripts under `{container_run_dir}`. Prefer Python for numerical experiments.
2. Run the code and inspect its outputs. Use fixed random seeds when randomness is involved.
3. Save important outputs under `{container_run_dir}`. Use text/JSON/CSV/PNG/PDF artifacts as appropriate.
4. Before finishing, write `{container_run_dir}/summary.md` with a human-readable account of what you did.
5. Before finishing, write `{container_run_dir}/experiment_result.json` exactly as JSON with at least:
   {{
     "summary": "concise factual summary of the run and its result",
     "supports_claim_ids": {json.dumps(supports_claim_ids)},
     "caveats": ["Experimental evidence is not a mathematical proof."]
   }}
   You may add extra JSON fields such as metrics, files, plots, or status.

Be conservative: distinguish successful numerical checks from proof, and report failures honestly.
"""

    def _pi_command(self, *, run_id: str, container_run_dir: str) -> list[str]:
        if self.settings is None:
            raise ExperimenterRuntimeError("Experimenter settings are missing.")
        pi = self.settings.pi
        command = [
            "pi",
            "--mode",
            "json",
            "--provider",
            pi.provider,
            "--model",
            pi.model,
            "--thinking",
            pi.thinking,
            "--tools",
            "read,write,edit,bash,grep,find,ls",
            "--no-context-files",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--approve",
            "--session-dir",
            "/workspace/home/.pi/sessions",
            "--name",
            f"experiment-{run_id}",
            *pi.extra_args,
            "-p",
            f"@{container_run_dir}/prompt.md",
        ]
        return command

    def _load_result_payload(self, rel_path: Path) -> dict[str, Any]:
        if not self.store.exists(rel_path):
            raise ExperimenterRuntimeError(
                f"Experimenter did not produce required result artifact: {rel_path}"
            )
        try:
            payload = self.store.read_json(rel_path)
        except Exception as exc:  # noqa: BLE001
            raise ExperimenterRuntimeError(
                f"Experiment result artifact is not valid JSON: {rel_path}"
            ) from exc
        if not isinstance(payload, dict):
            raise ExperimenterRuntimeError(
                f"Experiment result artifact must be a JSON object: {rel_path}"
            )
        return payload


def _limit_text(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return text
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return truncated + f"\n...[truncated {len(encoded) - max_bytes} bytes]\n"


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_.-")
    return (safe or "experiment")[:80]


def _artifact_refs_recursive(store: ArtifactStore, rel_dir: str, *, limit: int = 250) -> list[ArtifactRef]:
    root = store.resolve(rel_dir)
    refs: list[ArtifactRef] = [store.artifact_ref(rel_dir, summary="Experiment run directory")]
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files[:limit]:
        refs.append(store.artifact_ref(path))
    if len(files) > limit:
        omitted = len(files) - limit
        manifest = root / "artifact_manifest_truncated.json"
        manifest.write_text(
            json.dumps(
                {
                    "included_file_refs": limit,
                    "omitted_file_count": omitted,
                    "all_files": [str(path.relative_to(root)) for path in files],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        refs.append(store.artifact_ref(manifest))
    unique: list[ArtifactRef] = []
    seen: set[str] = set()
    for ref in refs:
        if ref.path in seen:
            continue
        seen.add(ref.path)
        unique.append(ref)
    return unique
