"""Lean placeholder checks for LEAP decompositions."""

from __future__ import annotations

import re
from dataclasses import dataclass

_DECL_RE = re.compile(r"^\s*(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_'.]*)")
_PLACEHOLDER_RE = re.compile(r"\b(sorry|admit)\b")


@dataclass(frozen=True)
class PlaceholderCheck:
    ok: bool
    errors: list[str]
    placeholder_lines: list[int]


def find_placeholder_lines(code: str) -> list[int]:
    return [i for i, line in enumerate(code.splitlines(), start=1) if _PLACEHOLDER_RE.search(line)]


def check_decomposition_placeholders(
    code: str,
    *,
    parent_name: str,
    child_names: list[str],
) -> PlaceholderCheck:
    """Check LEAP decomposition placeholder discipline.

    Accepted sketches may contain `sorry`/`admit` only inside explicitly declared child lemma
    blocks. The parent declaration must be placeholder-free.
    """
    child_set = set(child_names)
    errors: list[str] = []
    placeholder_lines: list[int] = []
    current_decl: str | None = None
    for line_no, line in enumerate(code.splitlines(), start=1):
        match = _DECL_RE.match(line)
        if match:
            current_decl = match.group(1).split(".")[-1]
        if _PLACEHOLDER_RE.search(line):
            placeholder_lines.append(line_no)
            if current_decl == parent_name:
                errors.append(f"Placeholder in parent theorem `{parent_name}` at line {line_no}.")
            elif current_decl not in child_set:
                errors.append(
                    f"Placeholder at line {line_no} is not inside an explicit child lemma block: {line.strip()}"
                )
    for name in child_names:
        if not re.search(rf"^\s*(?:theorem|lemma)\s+{re.escape(name)}\b", code, flags=re.MULTILINE):
            errors.append(f"Child lemma `{name}` is not declared in the formal sketch.")
    return PlaceholderCheck(ok=not errors, errors=errors, placeholder_lines=placeholder_lines)
