"""Lexical placeholder checks for generated Lean modules.

A plain substring search rejects harmless comments and strings.  This scanner removes Lean's nested
block comments, line comments, strings, and character literals while preserving line numbers, then
looks for placeholder tokens in executable source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_DECL_RE = re.compile(
    r"^\s*(?:private\s+|protected\s+)?(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_'.]*)"
)
_PLACEHOLDER_RE = re.compile(r"(?<![A-Za-z0-9_'])(sorry|admit)(?![A-Za-z0-9_'])")


@dataclass(frozen=True)
class PlaceholderCheck:
    ok: bool
    errors: list[str]
    placeholder_lines: list[int]


def executable_lean_source(code: str) -> str:
    """Replace comments and literals by spaces while preserving newlines and columns."""
    output = list(code)
    index = 0
    block_depth = 0
    in_string = False
    in_char = False
    escaped = False
    while index < len(code):
        current = code[index]
        following = code[index + 1] if index + 1 < len(code) else ""
        if block_depth:
            if current == "/" and following == "-":
                output[index] = output[index + 1] = " "
                block_depth += 1
                index += 2
                continue
            if current == "-" and following == "/":
                output[index] = output[index + 1] = " "
                block_depth -= 1
                index += 2
                continue
            if current != "\n":
                output[index] = " "
            index += 1
            continue
        if in_string or in_char:
            if current != "\n":
                output[index] = " "
            if escaped:
                escaped = False
            elif current == "\\":
                escaped = True
            elif (in_string and current == '"') or (in_char and current == "'"):
                in_string = False
                in_char = False
            index += 1
            continue
        if current == "-" and following == "-":
            output[index] = output[index + 1] = " "
            index += 2
            while index < len(code) and code[index] != "\n":
                output[index] = " "
                index += 1
            continue
        if current == "/" and following == "-":
            output[index] = output[index + 1] = " "
            block_depth = 1
            index += 2
            continue
        if current == '"':
            output[index] = " "
            in_string = True
        elif current == "'":
            # Lean identifiers may end in one or more apostrophes.  Treat an apostrophe as a
            # character literal delimiter only when it is not attached to an identifier.
            previous = code[index - 1] if index else " "
            if not (previous.isalnum() or previous in "_'"):
                output[index] = " "
                in_char = True
        index += 1
    return "".join(output)


def find_placeholder_lines(code: str) -> list[int]:
    executable = executable_lean_source(code)
    return [
        line_no
        for line_no, line in enumerate(executable.splitlines(), start=1)
        if _PLACEHOLDER_RE.search(line)
    ]


def check_decomposition_placeholders(
    code: str,
    *,
    parent_name: str,
    child_names: list[str],
) -> PlaceholderCheck:
    """Allow placeholders only in the application-declared child lemma bodies."""
    executable = executable_lean_source(code)
    child_set = {name.split(".")[-1] for name in child_names}
    parent_short = parent_name.split(".")[-1]
    errors: list[str] = []
    placeholder_lines: list[int] = []
    current_decl: str | None = None
    declared_children: set[str] = set()
    for line_no, line in enumerate(executable.splitlines(), start=1):
        match = _DECL_RE.match(line)
        if match:
            current_decl = match.group(1).split(".")[-1]
            if current_decl in child_set:
                declared_children.add(current_decl)
        if _PLACEHOLDER_RE.search(line):
            placeholder_lines.append(line_no)
            if current_decl == parent_short:
                errors.append(f"placeholder in parent theorem `{parent_name}` at line {line_no}")
            elif current_decl not in child_set:
                errors.append(
                    f"placeholder at line {line_no} is outside an explicit child lemma"
                )
    for missing in sorted(child_set - declared_children):
        errors.append(f"child lemma `{missing}` is not declared in the formal sketch")
    if parent_short not in {
        match.group(1).split(".")[-1]
        for line in executable.splitlines()
        if (match := _DECL_RE.match(line)) is not None
    }:
        errors.append(f"parent theorem `{parent_name}` is not declared in the formal sketch")
    return PlaceholderCheck(
        ok=not errors,
        errors=errors,
        placeholder_lines=placeholder_lines,
    )
