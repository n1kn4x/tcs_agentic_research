"""Project-level Docker container management for experiments."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..artifact_store import ArtifactStore
from ..schemas import ExperimenterSettings, utc_now
from .errors import ExperimenterConfigurationError, ExperimenterRuntimeError


class DockerProjectContainer:
    """Manage one persistent experimenter container per research workspace.

    The canonical research workspace is mounted read-only at ``/research``. Portable,
    mutable experiment state lives under ``.experimenter/workspace`` and is bind-mounted
    read-write at ``/workspace``. The core engine executes one generated Python program per
    work item; it does not run another tool-calling agent inside this container.
    """

    def __init__(self, store: ArtifactStore, settings: ExperimenterSettings | None):
        self.store = store
        self.settings = self._require_settings(settings)
        self.project_slug = _project_slug(store.root)
        self.container_name = f"{self.settings.container_name_prefix}-{self.project_slug}"
        self.experimenter_dir = self.store.resolve(".experimenter")
        self.workspace_state_dir = self.store.resolve(".experimenter/workspace")
        self.manifest_path = self.store.resolve(".experimenter/manifest.json")

    def ensure_running(self) -> None:
        self._ensure_docker_available()
        self._ensure_workspace_dirs()
        self._ensure_image()
        state = self._container_state()
        if state is not None and not self._container_has_expected_workspace_mount():
            if state == "running":
                self._docker(["stop", self.container_name], timeout=60)
            self._docker(["rm", self.container_name], timeout=60)
            state = None
        if state == "running" and self._container_healthy():
            self._write_manifest(state="running")
            return
        if state == "running":
            self._docker(["stop", self.container_name], timeout=60, check=False)
            self._docker(["rm", self.container_name], timeout=60, check=False)
            state = None
        if state is not None:
            self._docker(["start", self.container_name], timeout=60)
            if self._container_healthy():
                self._write_manifest(state="running")
                return
            self._docker(["stop", self.container_name], timeout=60, check=False)
            self._docker(["rm", self.container_name], timeout=60, check=False)
        self._docker(self._run_args(), timeout=120)
        if not self._container_healthy():
            self._docker(["rm", "-f", self.container_name], timeout=60, check=False)
            raise ExperimenterRuntimeError(
                "Experiment container started but failed its /workspace and python3 health check."
            )
        self._write_manifest(state="running")

    def status(self) -> dict[str, Any]:
        self._ensure_docker_available()
        state = self._container_state()
        return {
            "container_name": self.container_name,
            "image": self.settings.image,
            "state": state or "absent",
            "research_mount": str(self.store.root),
            "research_mount_mode": "read-only",
            "workspace_mount": str(self.workspace_state_dir),
            "workspace_mount_mode": "read-write bind mount",
            "container_workspace": "/workspace",
            "manifest": str(self.manifest_path),
        }

    def stop(self, *, remove: bool = False) -> None:
        self._ensure_docker_available()
        state = self._container_state()
        if state == "running":
            self._docker(["stop", self.container_name], timeout=60)
        if remove and self._container_state() is not None:
            self._docker(["rm", self.container_name], timeout=60)
        self._write_manifest(state="absent" if remove else "stopped")

    def reset(self) -> None:
        """Remove the project container and portable writable experimenter state.

        This does not remove the global Docker image. The image is rebuildable from the
        packaged Dockerfile or an explicitly configured Dockerfile.
        """
        self.stop(remove=True)
        if self.experimenter_dir.exists():
            shutil.rmtree(self.experimenter_dir)

    def exec(
        self,
        args: list[str],
        *,
        workdir: str = "/workspace",
        timeout: int | None = None,
        env: dict[str, str] | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        self.ensure_running()
        cmd = ["exec", "-w", workdir]
        for key, value in {**self._container_env(), **(env or {})}.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.append(self.container_name)
        cmd.extend(args)
        return self._docker(cmd, timeout=timeout or self.settings.timeout_seconds, check=check)

    def copy_to_container(self, src: Path, dest: str, *, timeout: int = 60) -> None:
        self.ensure_running()
        self._docker(["cp", str(src), f"{self.container_name}:{dest}"], timeout=timeout)

    def copy_from_container(self, src: str, dest: Path, *, timeout: int = 120) -> None:
        self.ensure_running()
        dest.mkdir(parents=True, exist_ok=True)
        self._docker(["cp", f"{self.container_name}:{src}", str(dest)], timeout=timeout)

    def _run_args(self) -> list[str]:
        args = [
            "run",
            "-d",
            "--name",
            self.container_name,
            "--network",
            self.settings.network,
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m",
            "--pids-limit",
            "128",
            "--security-opt",
            "no-new-privileges",
            "--label",
            "tcs.agentic_research.role=experimenter",
            "--label",
            f"tcs.agentic_research.workspace={self.store.root}",
            "--label",
            f"tcs.agentic_research.project_slug={self.project_slug}",
            "--mount",
            f"type=bind,src={self.store.root},dst=/research,readonly",
            "--mount",
            f"type=bind,src={self.workspace_state_dir},dst=/workspace",
        ]
        if self.settings.memory:
            args.extend(["--memory", self.settings.memory])
        if self.settings.cpus:
            args.extend(["--cpus", str(self.settings.cpus)])
        if self.settings.add_host_gateway:
            args.extend(["--add-host", "host.docker.internal:host-gateway"])
        for key, value in self._container_env().items():
            args.extend(["-e", f"{key}={value}"])
        args.extend([self.settings.image, "sleep", "infinity"])
        return args

    def _container_env(self) -> dict[str, str]:
        return {
            "HOME": "/workspace/home",
            "PIP_CACHE_DIR": "/workspace/.cache/pip",
            "PYTHONUSERBASE": "/workspace/python-user",
            "MPLBACKEND": "Agg",
            **self.settings.environment,
        }

    def _ensure_workspace_dirs(self) -> None:
        for directory in [
            self.workspace_state_dir,
            self.workspace_state_dir / "home",
            self.workspace_state_dir / "runs",
            self.workspace_state_dir / ".cache" / "pip",
            self.workspace_state_dir / "python-user",
        ]:
            directory.mkdir(parents=True, exist_ok=True)
            # The container user is fixed at uid 1000. Make the portable workspace writable
            # even when the host uid differs from 1000.
            try:
                directory.chmod(0o777)
            except PermissionError:
                pass

    def _write_manifest(self, *, state: str) -> None:
        self.experimenter_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": utc_now(),
            "container_name": self.container_name,
            "project_slug": self.project_slug,
            "image": self.settings.image,
            "state": state,
            "research_mount": {
                "host_path": str(self.store.root),
                "container_path": "/research",
                "mode": "read-only",
            },
            "workspace_mount": {
                "host_path": str(self.workspace_state_dir),
                "container_path": "/workspace",
                "mode": "read-write bind mount",
                "portable_with_workspace": True,
            },
            "notes": [
                "Copying the workspace copies generated scripts, outputs, caches, and intermediate experiment state.",
                "The Docker image is global to the Docker daemon and can be rebuilt from the packaged Dockerfile.",
            ],
        }
        self.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def _ensure_docker_available(self) -> None:
        if shutil.which("docker") is None:
            raise ExperimenterConfigurationError(
                "Experimenter requested, but `docker` is not installed or not on PATH."
            )
        completed = self._docker(["version", "--format", "{{.Server.Version}}"], check=False, timeout=20)
        if completed.returncode != 0:
            raise ExperimenterConfigurationError(
                "Experimenter requested, but Docker is not available/running: "
                f"{_diagnostic(completed)}"
            )

    def _ensure_image(self) -> None:
        inspected = self._docker(["image", "inspect", self.settings.image], check=False, timeout=30)
        if inspected.returncode == 0:
            return
        dockerfile = self._dockerfile_path()
        context = dockerfile.parent
        self._docker(
            ["build", "-t", self.settings.image, "-f", str(dockerfile), str(context)],
            timeout=max(600, self.settings.timeout_seconds),
        )

    def _dockerfile_path(self) -> Path:
        if self.settings.dockerfile:
            path = Path(self.settings.dockerfile).expanduser().resolve()
        else:
            path = Path(
                str(
                    importlib.resources.files("tcs_agentic_research.experimenter").joinpath(
                        "Dockerfile"
                    )
                )
            )
        if not path.exists():
            raise ExperimenterConfigurationError(f"Experimenter Dockerfile not found: {path}")
        return path

    def _container_state(self) -> str | None:
        completed = self._docker(
            ["inspect", "-f", "{{.State.Status}}", self.container_name],
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            return None
        return completed.stdout.strip() or None

    def _container_healthy(self) -> bool:
        """Detect stale bind mounts and incomplete images before spending a model call.

        Docker may keep a running container after its host bind-mount directory was deleted and
        recreated. Inspect still reports the same path, but every exec then fails before the
        command starts. A real exec probe is the only reliable inexpensive check.
        """
        completed = self._docker(
            [
                "exec",
                "-w",
                "/workspace",
                self.container_name,
                "sh",
                "-c",
                "test -d /workspace/runs && command -v python3 >/dev/null && python3 -c 'import json'",
            ],
            timeout=30,
            check=False,
        )
        return completed.returncode == 0

    def _container_has_expected_workspace_mount(self) -> bool:
        completed = self._docker(
            ["inspect", "-f", "{{json .Mounts}}", self.container_name],
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            return False
        try:
            mounts = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return False
        expected = str(self.workspace_state_dir)
        return any(
            isinstance(mount, dict)
            and mount.get("Destination") == "/workspace"
            and str(mount.get("Source") or "") == expected
            for mount in mounts
        )

    def _docker(
        self,
        args: list[str],
        *,
        timeout: int | None = None,
        check: bool = True,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = ["docker", *args]
        try:
            completed = subprocess.run(
                command,
                input=input_text,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExperimenterRuntimeError(
                f"Docker command timed out after {timeout}s: {' '.join(command)}"
            ) from exc
        if check and completed.returncode != 0:
            raise ExperimenterRuntimeError(
                f"Docker command failed: {' '.join(command)}\n{_diagnostic(completed)}"
            )
        return completed

    @staticmethod
    def _require_settings(settings: ExperimenterSettings | None) -> ExperimenterSettings:
        if settings is None:
            raise ExperimenterConfigurationError(
                "Experimenter requested, but no `experimenter:` block is configured."
            )
        if not settings.enabled:
            raise ExperimenterConfigurationError(
                "Experimenter requested, but `experimenter.enabled` is false."
            )
        return settings


def _project_slug(root: Path) -> str:
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", root.name).strip("-.").lower() or "workspace"
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:10]
    return f"{safe_name[:38]}-{digest}"


def _diagnostic(completed: subprocess.CompletedProcess[str], *, limit: int = 8000) -> str:
    text = f"exit_code={completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    if len(text) <= limit:
        return text
    marker = f"\n...[truncated {len(text) - limit} characters; preserving final lines]...\n"
    tail_limit = max(limit - len(marker) - 2000, limit // 2)
    head_limit = max(limit - len(marker) - tail_limit, 0)
    return text[:head_limit] + marker + text[-tail_limit:]
