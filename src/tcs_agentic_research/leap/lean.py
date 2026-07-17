"""Pinned-project Lean execution and structured verification contracts for LEAP."""

from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO, Callable

from ..artifact_store import ArtifactStore
from ..schemas import LeanCompilerLog, LeanStatement, new_id
from .models import LeanDiagnostic, VerificationResult
from .sorry import check_decomposition_placeholders, find_placeholder_lines

_DIAGNOSTIC_RE = re.compile(
    r"^(?P<path>.*?):(?P<line>\d+):(?P<column>\d+)"
    r"(?:(?:-|--)(?P<end_line>\d+):(?P<end_column>\d+))?:\s*"
    r"(?P<severity>error|warning|information)(?:\([^)]*\))?:\s*(?P<message>.*)$",
    flags=re.IGNORECASE,
)


class LeanVerifier:
    """Compile generated modules in one persistent Lake workspace.

    The first invocation builds imported local modules.  Every candidate is then checked in a fresh
    process with a wall timeout, CPU/address-space limits, bounded output, and no shell.  Final
    acceptance always uses the same batch-compiler path as ordinary candidates.
    """

    def __init__(
        self,
        store: ArtifactStore,
        *,
        timeout_seconds: int = 300,
        memory_mb: int = 16384,
        max_output_bytes: int = 1_000_000,
    ):
        self.store = store
        self.project_root = self.store.resolve("LeanProject")
        self.timeout_seconds = timeout_seconds
        self.memory_mb = memory_mb
        self.max_output_bytes = max_output_bytes
        self._built = False
        self._sandbox_supported: bool | None = None

    def ensure_project(self) -> None:
        """Create a minimal project only when the workspace does not already provide one."""
        self.project_root.mkdir(parents=True, exist_ok=True)
        (self.project_root / "TCSResearch").mkdir(parents=True, exist_ok=True)
        lakefile = self.project_root / "lakefile.lean"
        toml_lakefile = self.project_root / "lakefile.toml"
        if not lakefile.exists() and not toml_lakefile.exists():
            lakefile.write_text(
                """import Lake
open Lake DSL

package «tcs_research» where
  -- Add pinned Mathlib/domain dependencies before a run that imports them.

@[default_target]
lean_lib TCSResearch where
  roots := #[`TCSResearch]
""",
                encoding="utf-8",
            )
        toolchain = self.project_root / "lean-toolchain"
        if not toolchain.exists():
            toolchain.write_text(_installed_toolchain_spec() + "\n", encoding="utf-8")
        basic = self.project_root / "TCSResearch" / "Basic.lean"
        if not basic.exists():
            basic.write_text(
                """/- Canonical local namespace for the agentic TCS research project. -/
namespace TCSResearch

end TCSResearch
""",
                encoding="utf-8",
            )
        root_module = self.project_root / "TCSResearch.lean"
        if not root_module.exists():
            root_module.write_text("import TCSResearch.Basic\n", encoding="utf-8")
        (self.project_root / "TCSResearch" / "Generated").mkdir(parents=True, exist_ok=True)
        (self.project_root / "LEAP" / "logs").mkdir(parents=True, exist_ok=True)

    def available(self) -> bool:
        return shutil.which("lake") is not None or shutil.which("lean") is not None

    def environment_fingerprint(self) -> str:
        """Hash the toolchain and dependency-selection files that affect elaboration."""
        self.ensure_project()
        # Lake may create a manifest on the first build even for a dependency-free project.  Do
        # that before hashing so the first and all resumed invocations see the same environment.
        if shutil.which("lake") is not None and not self._built:
            self._build_once()
        records: dict[str, str] = {}
        for name in ["lean-toolchain", "lake-manifest.json", "lakefile.toml", "lakefile.lean"]:
            path = self.project_root / name
            if path.exists():
                records[name] = path.read_text(encoding="utf-8", errors="replace")
        local_source = self.project_root / "TCSResearch"
        if local_source.exists():
            for path in sorted(local_source.rglob("*.lean")):
                if "Generated" in path.parts or "LEAP" in path.parts:
                    continue
                records[f"source:{path.relative_to(self.project_root)}"] = path.read_text(
                    encoding="utf-8", errors="replace"
                )
        executable = shutil.which("lake") or shutil.which("lean")
        version = "unavailable"
        if executable:
            completed = subprocess.run(
                [executable, "--version"],
                cwd=self.project_root,
                text=True,
                capture_output=True,
                check=False,
                timeout=20,
            )
            version = (completed.stdout + completed.stderr).strip()
        records["compiler"] = version
        return hashlib.sha256(
            json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def elaborate_statement(self, goal: LeanStatement) -> tuple[str, VerificationResult]:
        """Ask Lean to elaborate a closed proposition before it enters the graph."""
        imports = "\n".join(f"import {item}" for item in goal.imports)
        opening = f"namespace {goal.namespace}\n\n" if goal.namespace else ""
        closing = f"\nend {goal.namespace}\n" if goal.namespace else ""
        code = (
            f"{imports}\n\n{opening}set_option pp.universes true in\n"
            f"#check (show Prop from ({goal.statement}))\n{closing}"
        )
        digest = hashlib.sha256(code.encode()).hexdigest()[:16]
        result = self.check_code(
            code,
            rel_file=f"TCSResearch/Generated/Statement_{digest}.lean",
            allow_placeholders=False,
        )
        # The pretty-printer output is useful when available, but it is not guaranteed to have a
        # stable transport format across Lean versions.  The environment fingerprint already
        # separates versions, and source normalization remains a deterministic fallback.
        elaborated = _check_output(result.stdout) or goal.statement.strip()
        return elaborated, result

    def verify_code(self, code: str, *, rel_file: str | None = None) -> LeanCompilerLog:
        """Compatibility-level raw compile operation used by other project code."""
        self.ensure_project()
        if rel_file is None:
            rel_file = f"TCSResearch/Generated/{new_id('proof')}.lean"
        target = self.project_root / rel_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding="utf-8")
        return self.verify_file(rel_file)

    def verify_file(self, rel_file: str | Path) -> LeanCompilerLog:
        self.ensure_project()
        rel = Path(rel_file)
        target = self.project_root / rel
        log_rel = f"LeanProject/LEAP/logs/{new_id('lean_log')}.log"
        if not self.available():
            stderr = "Neither `lake` nor `lean` was found on PATH. Install Lean via elan."
            log = LeanCompilerLog(
                command=["lake", "env", "lean", str(rel)],
                cwd=str(self.project_root),
                exit_code=127,
                stderr=stderr,
                success=False,
            )
            self.store.write_text(log_rel, stderr + "\n")
            log.artifact_ref = self.store.artifact_ref(log_rel)
            return log

        if shutil.which("lake") is not None:
            build_failure = self._build_once()
            if build_failure is not None:
                command, exit_code, stdout, stderr = build_failure
                return self._compiler_log(command, exit_code, stdout, stderr, log_rel)
            command = ["lake", "env", "lean", str(rel)]
        else:
            command = ["lean", str(target)]
        exit_code, stdout, stderr = self._run(
            command, timeout=self.timeout_seconds, sandbox=True
        )
        return self._compiler_log(command, exit_code, stdout, stderr, log_rel)

    def check_code(
        self,
        code: str,
        *,
        rel_file: str,
        allow_placeholders: bool,
    ) -> VerificationResult:
        target = self.project_root / rel_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding="utf-8")
        placeholders = find_placeholder_lines(code)
        if placeholders and not allow_placeholders:
            reason = "unauthorized Lean placeholder(s) at line(s): " + ", ".join(
                str(line) for line in placeholders
            )
            log_rel = f"LeanProject/LEAP/logs/{new_id('placeholder')}.log"
            self.store.write_text(log_rel, reason + "\n")
            return VerificationResult(
                accepted=False,
                reason=reason,
                source_path=self.store.relpath(target),
                exit_code=2,
                diagnostics=[LeanDiagnostic(severity="error", message=reason)],
                log_path=log_rel,
            )
        raw = self.verify_file(rel_file)
        diagnostics = parse_diagnostics(raw.stdout + "\n" + raw.stderr)
        errors = [item for item in diagnostics if item.severity == "error"]
        placeholder_warnings = [
            item
            for item in diagnostics
            if re.search(r"(?:declaration\s+)?uses\s+['`]?sorry|uses\s+['`]?admit", item.message, re.I)
        ]
        accepted = raw.success and not errors and (allow_placeholders or not placeholder_warnings)
        reason = (
            ""
            if accepted
            else (
                "Lean reported that the declaration uses a placeholder"
                if placeholder_warnings and not allow_placeholders
                else _failure_reason(raw, errors)
            )
        )
        return VerificationResult(
            accepted=accepted,
            reason=reason,
            source_path=self.store.relpath(target),
            exit_code=raw.exit_code,
            stdout=raw.stdout,
            stderr=raw.stderr,
            diagnostics=diagnostics,
            log_path=raw.artifact_ref.path if raw.artifact_ref else "",
        )

    def check_direct(self, code: str, *, rel_file: str, target_name: str) -> VerificationResult:
        if not re.search(
            rf"(?m)^\s*(?:theorem|lemma)\s+{re.escape(target_name)}\b", code
        ):
            return VerificationResult(
                accepted=False,
                reason=f"generated module does not declare target `{target_name}`",
                exit_code=2,
                diagnostics=[
                    LeanDiagnostic(
                        severity="error",
                        message=f"generated module does not declare target `{target_name}`",
                    )
                ],
            )
        return self.check_code(code, rel_file=rel_file, allow_placeholders=False)

    def check_sketch(
        self,
        code: str,
        *,
        rel_file: str,
        parent_name: str,
        child_names: list[str],
    ) -> VerificationResult:
        discipline = check_decomposition_placeholders(
            code, parent_name=parent_name, child_names=child_names
        )
        if not discipline.ok:
            reason = "; ".join(discipline.errors)
            return VerificationResult(
                accepted=False,
                reason=reason,
                exit_code=2,
                diagnostics=[LeanDiagnostic(severity="error", message=reason)],
            )
        result = self.check_code(code, rel_file=rel_file, allow_placeholders=True)
        child_declaration_lines = {
            line_no
            for line_no, line in enumerate(code.splitlines(), start=1)
            if any(
                re.match(rf"^\s*(?:theorem|lemma)\s+{re.escape(name)}\b", line)
                for name in child_names
            )
        }
        unauthorized_warnings = [
            diagnostic
            for diagnostic in result.diagnostics
            if re.search(r"(?:declaration\s+)?uses\s+['`]?sorry", diagnostic.message, re.I)
            and diagnostic.line not in child_declaration_lines
        ]
        if unauthorized_warnings:
            result.accepted = False
            result.reason = "Lean reported a placeholder dependency in the parent theorem"
        return result

    def _build_once(self) -> tuple[list[str], int, str, str] | None:
        if self._built:
            return None
        command = ["lake", "build"]
        exit_code, stdout, stderr = self._run(
            command, timeout=max(self.timeout_seconds, 600), sandbox=False
        )
        if exit_code != 0:
            return command, exit_code, stdout, stderr
        self._built = True
        return None

    def _run(
        self, command: list[str], *, timeout: int, sandbox: bool = False
    ) -> tuple[int, str, str]:
        effective_command = self._sandbox_command(command) if sandbox else command
        try:
            with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
                process = subprocess.Popen(
                    effective_command,
                    cwd=self.project_root,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=True,
                    env=_lean_environment(),
                    preexec_fn=_resource_limiter(
                        self.memory_mb,
                        timeout,
                        max(16 * 1024 * 1024, self.max_output_bytes * 2),
                    ),
                )
                timed_out = False
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    try:
                        os.killpg(process.pid, 9)
                    except ProcessLookupError:
                        pass
                    process.wait()
                stdout = _read_bounded_file(stdout_file, self.max_output_bytes)
                stderr = _read_bounded_file(stderr_file, self.max_output_bytes)
                if timed_out:
                    stderr += f"\nLean verification timed out after {timeout} seconds."
                    return 124, stdout, stderr
                return process.returncode, stdout, stderr
        except OSError as exc:
            return 127, "", f"failed to execute Lean: {exc}"

    def _sandbox_command(self, command: list[str]) -> list[str]:
        """Use bubblewrap when user namespaces are available.

        The project/dependency tree is visible read-only and the network namespace has no
        interfaces.  Source/log files are written by the parent before/after this process.  Hosts
        that disable unprivileged user namespaces fall back to the resource-limited subprocess.
        """
        if self._sandbox_supported is None:
            bwrap = shutil.which("bwrap")
            if bwrap is None:
                self._sandbox_supported = False
            else:
                try:
                    probe = subprocess.run(
                        [bwrap, "--die-with-parent", "--ro-bind", "/", "/", "true"],
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=10,
                    )
                    self._sandbox_supported = probe.returncode == 0
                except (OSError, subprocess.TimeoutExpired):
                    self._sandbox_supported = False
        if not self._sandbox_supported:
            return command
        assert shutil.which("bwrap") is not None
        return [
            shutil.which("bwrap") or "bwrap",
            "--die-with-parent",
            "--new-session",
            "--unshare-net",
            "--ro-bind",
            "/",
            "/",
            "--dev-bind",
            "/dev",
            "/dev",
            "--proc",
            "/proc",
            "--chdir",
            str(self.project_root),
            *command,
        ]

    def _compiler_log(
        self,
        command: list[str],
        exit_code: int,
        stdout: str,
        stderr: str,
        log_rel: str,
    ) -> LeanCompilerLog:
        self.store.write_text(log_rel, "STDOUT:\n" + stdout + "\nSTDERR:\n" + stderr)
        return LeanCompilerLog(
            command=command,
            cwd=str(self.project_root),
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            success=exit_code == 0,
            artifact_ref=self.store.artifact_ref(log_rel),
        )


def _installed_toolchain_spec() -> str:
    lean = shutil.which("lean")
    if lean is not None:
        try:
            completed = subprocess.run(
                [lean, "--version"], text=True, capture_output=True, check=False, timeout=20
            )
            match = re.search(r"Lean \(version ([^,\s)]+)", completed.stdout + completed.stderr)
            if match:
                version = match.group(1)
                if not version.startswith("v"):
                    version = "v" + version
                return "leanprover/lean4:" + version
        except (OSError, subprocess.TimeoutExpired):
            pass
    # This fallback is needed only to create an inspectable project on a host without Lean.  LEAP
    # reports `unavailable` and does not accept proofs there.
    return "leanprover/lean4:stable"


def parse_diagnostics(output: str) -> list[LeanDiagnostic]:
    diagnostics: list[LeanDiagnostic] = []
    current: LeanDiagnostic | None = None
    for line in output.splitlines():
        match = _DIAGNOSTIC_RE.match(line)
        if match:
            current = LeanDiagnostic(
                severity=match.group("severity").lower(),
                message=match.group("message").strip(),
                line=int(match.group("line")),
                column=int(match.group("column")),
                end_line=int(match.group("end_line")) if match.group("end_line") else None,
                end_column=(
                    int(match.group("end_column")) if match.group("end_column") else None
                ),
            )
            diagnostics.append(current)
        elif current is not None and line.strip() and len(current.message) < 10_000:
            current.message = (current.message + "\n" + line.rstrip())[:10_000]
    return diagnostics


def _resource_limiter(
    memory_mb: int, timeout: int, max_file_bytes: int
) -> Callable[[], None]:
    def apply() -> None:
        if memory_mb > 0:
            memory = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        # RLIMIT_CPU counts aggregate time across Lean worker threads, so allow several cores while
        # the parent still enforces the strict wall timeout.
        cpu = max(1, timeout * 4 + 5)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        # This also bounds stdout/stderr, which the parent directs to regular temporary files.
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_bytes, max_file_bytes))

    return apply


def _lean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.setdefault("LEAN_ABORT_ON_PANIC", "1")
    # Dependency synchronization belongs to explicit project setup, never candidate verification.
    environment["GIT_TERMINAL_PROMPT"] = "0"
    return environment


def _read_bounded_file(file: BinaryIO, limit: int) -> str:
    file.seek(0, os.SEEK_END)
    size = file.tell()
    if size <= limit:
        file.seek(0)
        data = file.read()
    else:
        half = max(1, limit // 2)
        file.seek(0)
        start = file.read(half)
        file.seek(-half, os.SEEK_END)
        end = file.read(half)
        data = start + b"\n...[compiler output truncated]...\n" + end
    return data.decode("utf-8", errors="replace")


def _failure_reason(log: LeanCompilerLog, errors: list[LeanDiagnostic]) -> str:
    if log.exit_code == 127:
        return "Lean is unavailable"
    if log.exit_code == 124:
        return "Lean verification timed out"
    if errors:
        return errors[0].message[:2000]
    tail = (log.stderr or log.stdout).strip()
    return tail[-2000:] or f"Lean exited with status {log.exit_code}"


def _check_output(stdout: str) -> str:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    # Strip location-prefixed diagnostics; retain the actual #check pretty-printer line(s).
    lines = [line for line in lines if not _DIAGNOSTIC_RE.match(line)]
    return " ".join(lines)[:8000]
