"""Static containment checks for generated experiment programs."""

from __future__ import annotations

import ast
from pathlib import Path

from ..schemas import ExperimentProgram

MAX_EXPERIMENT_SOURCE_CHARS = 30_000


def validate_experiment_program(program: ExperimentProgram) -> None:
    code = program.python_code
    if len(code) > MAX_EXPERIMENT_SOURCE_CHARS:
        raise ValueError("experiment source exceeds the 30,000-character limit")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"experiment is not valid Python: {exc}") from exc
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    run = functions.get("run_experiment")
    if run is None or isinstance(run, ast.AsyncFunctionDef):
        raise ValueError("experiment must define synchronous run_experiment(mode)")
    positional = [*run.args.posonlyargs, *run.args.args]
    if len(positional) != 1 or run.args.vararg or run.args.kwarg:
        raise ValueError("run_experiment must accept exactly one positional mode argument")
    meaningful = [
        node
        for node in run.body
        if not isinstance(node, ast.Pass)
        and not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]
    if not meaningful:
        raise ValueError("run_experiment contains only a placeholder")
    unsafe = next((node for node in tree.body if not _safe_top_level(node)), None)
    if unsafe is not None:
        raise ValueError(
            "experiment top level may contain only imports, constants, classes, and functions; "
            f"found {type(unsafe).__name__} at line {getattr(unsafe, 'lineno', '?')}"
        )
    forbidden_modules = {
        "asyncio", "httpx", "multiprocessing", "requests", "shutil", "socket",
        "subprocess", "urllib",
    }
    forbidden_calls = {"compile", "eval", "exec", "__import__", "exit", "quit"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            blocked = {alias.name.split(".", 1)[0] for alias in node.names} & forbidden_modules
            if blocked:
                raise ValueError(f"experiment imports forbidden module(s): {sorted(blocked)}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in forbidden_modules or root == "os":
                raise ValueError(f"experiment imports forbidden module: {root}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in forbidden_calls:
                raise ValueError(f"experiment calls forbidden builtin: {node.func.id}")
            if node.func.id == "open" and node.args and isinstance(node.args[0], ast.Constant):
                path = Path(str(node.args[0].value))
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("experiment opens a path outside its run directory")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"os", "sys"}
        ):
            raise ValueError(f"experiment calls forbidden process/filesystem API: {node.func.value.id}.{node.func.attr}")


def _safe_top_level(node: ast.stmt) -> bool:
    if isinstance(
        node,
        (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign),
    ):
        return True
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(
        node.value.value, str
    )
