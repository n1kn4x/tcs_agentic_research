"""Lean/Lake verification utilities for LEAP."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..artifact_store import ArtifactStore
from ..schemas import LeanCompilerLog, new_id


class LeanVerifier:
    def __init__(self, store: ArtifactStore):
        self.store = store
        self.project_root = self.store.resolve("LeanProject")

    def ensure_project(self) -> None:
        self.project_root.mkdir(parents=True, exist_ok=True)
        (self.project_root / "TCSResearch").mkdir(parents=True, exist_ok=True)
        lakefile = self.project_root / "lakefile.lean"
        if not lakefile.exists():
            lakefile.write_text(
                """import Lake
open Lake DSL

package «tcs_research» where
  -- Add Mathlib or domain-specific dependencies here when needed.

@[default_target]
lean_lib TCSResearch where
  roots := #[`TCSResearch]
""",
                encoding="utf-8",
            )
        toolchain = self.project_root / "lean-toolchain"
        if not toolchain.exists():
            toolchain.write_text("leanprover/lean4:stable\n", encoding="utf-8")
        basic = self.project_root / "TCSResearch" / "Basic.lean"
        if not basic.exists():
            basic.write_text(
                """/- Canonical local namespace for the agentic TCS research project. -/
namespace TCSResearch

end TCSResearch
""",
                encoding="utf-8",
            )

    def verify_code(self, code: str, *, rel_file: str | None = None) -> LeanCompilerLog:
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
        logs_dir = self.project_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_rel = f"LeanProject/logs/{new_id('lean_log')}.log"
        command: list[str]
        cwd: Path
        if shutil.which("lake") is not None:
            command = ["lake", "env", "lean", str(rel)]
            cwd = self.project_root
        elif shutil.which("lean") is not None:
            command = ["lean", str(target)]
            cwd = self.project_root
        else:
            stderr = "Neither `lake` nor `lean` was found on PATH. Install Lean via elan."
            log = LeanCompilerLog(
                command=["lake", "env", "lean", str(rel)],
                cwd=str(self.project_root),
                exit_code=127,
                stdout="",
                stderr=stderr,
                success=False,
            )
            self.store.write_text(log_rel, stderr)
            log.artifact_ref = self.store.artifact_ref(log_rel)
            return log
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                text=True,
                capture_output=True,
                check=False,
                timeout=120,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = str(exc.stdout or "")
            stderr = str(exc.stderr or "") + "\nLean verification timed out after 120 seconds."
            self.store.write_text(log_rel, "STDOUT:\n" + stdout + "\nSTDERR:\n" + stderr)
            return LeanCompilerLog(
                command=command,
                cwd=str(cwd),
                exit_code=124,
                stdout=stdout,
                stderr=stderr,
                success=False,
                artifact_ref=self.store.artifact_ref(log_rel),
            )
        output = "STDOUT:\n" + completed.stdout + "\nSTDERR:\n" + completed.stderr
        self.store.write_text(log_rel, output)
        return LeanCompilerLog(
            command=command,
            cwd=str(cwd),
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            success=completed.returncode == 0,
            artifact_ref=self.store.artifact_ref(log_rel),
        )
