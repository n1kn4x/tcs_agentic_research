"""Deterministic Lean rendering and final proof materialization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from ..schemas import LeanStatement
from .graph import ProofGraph
from .models import AndNode, OrNode, OrStatus


class MaterializationError(RuntimeError):
    pass


@dataclass(frozen=True)
class RenderedModule:
    code: str
    declaration_order: list[str]


def render_declaration(goal: LeanStatement, proof: str, *, kind: str = "theorem") -> str:
    # Keep the application-owned proposition text intact.  In particular, a model shown a leading
    # `∀` must introduce that binder in its proof; silently moving binders into the declaration
    # would change the tactic state the model was asked to solve.
    return f"{kind} {goal.name} : {goal.statement.strip()} := {proof}\n"


def render_direct_module(
    goal: LeanStatement,
    proof: str,
    *,
    support_declarations: str = "",
) -> str:
    body = support_declarations.rstrip()
    if body:
        body += "\n\n"
    body += render_declaration(goal, proof)
    return render_module(goal.imports, goal.namespace, body)


def render_sketch_module(
    parent: LeanStatement,
    children: Sequence[LeanStatement],
    parent_proof: str,
    *,
    support_declarations: str = "",
) -> str:
    pieces: list[str] = []
    if support_declarations.strip():
        pieces.append(support_declarations.strip())
    pieces.extend(render_declaration(child, "by\n  sorry", kind="theorem").rstrip() for child in children)
    pieces.append(render_declaration(parent, parent_proof).rstrip())
    return render_module(parent.imports, parent.namespace, "\n\n".join(pieces) + "\n")


def render_module(imports: Sequence[str], namespace: str | None, body: str) -> str:
    import_text = "\n".join(f"import {item}" for item in imports)
    opening = f"namespace {namespace}\n\n" if namespace else ""
    closing = f"\nend {namespace}\n" if namespace else ""
    return f"{import_text}\n\n{opening}{body.rstrip()}\n{closing}"


class ProofMaterializer:
    def __init__(self, graph: ProofGraph):
        self.graph = graph

    def support_declarations(self, node_ids: Sequence[str]) -> str:
        order = self._topological_order(node_ids)
        return "\n\n".join(self._declaration(self.graph.get_or(node_id)).rstrip() for node_id in order)

    def final_module(self, root_or_id: str, target: LeanStatement) -> RenderedModule:
        root = self.graph.get_or(root_or_id)
        if root.status != OrStatus.proved:
            raise MaterializationError(f"root OR node {root_or_id} is not proved")
        order = self._topological_order([root_or_id])
        declarations = [self._declaration(self.graph.get_or(node_id)).rstrip() for node_id in order]
        if root.goal.name != target.name:
            alias_proof = f"by\n  exact {root.goal.name}"
            declarations.append(render_declaration(target, alias_proof).rstrip())
        code = render_module(target.imports, target.namespace, "\n\n".join(declarations) + "\n")
        return RenderedModule(code=code, declaration_order=order)

    def _topological_order(self, roots: Sequence[str]) -> list[str]:
        permanent: set[str] = set()
        temporary: set[str] = set()
        order: list[str] = []

        def visit(node_id: str) -> None:
            if node_id in permanent:
                return
            if node_id in temporary:
                raise MaterializationError(f"cycle detected while materializing {node_id}")
            temporary.add(node_id)
            node = self.graph.get_or(node_id)
            if node.status != OrStatus.proved:
                raise MaterializationError(f"dependency {node_id} is not proved")
            for dependency_id in self._dependencies(node):
                visit(dependency_id)
            temporary.remove(node_id)
            permanent.add(node_id)
            order.append(node_id)

        for root in roots:
            visit(root)
        return order

    def _dependencies(self, node: OrNode) -> list[str]:
        if node.proof_kind == "direct":
            return self.graph.dependencies("or", node.node_id)
        if node.proof_kind == "decomposition" and node.selected_and_id:
            decomposition = self.graph.get_and(node.selected_and_id)
            dependencies = self.graph.dependencies("and", decomposition.node_id)
            dependencies.extend(
                child.child_or_id for child in decomposition.children if child.required
            )
            return list(dict.fromkeys(dependencies))
        raise MaterializationError(f"proved node {node.node_id} has no proof artifact")

    def _declaration(self, node: OrNode) -> str:
        if node.proof_kind == "direct":
            if not node.proof_content:
                raise MaterializationError(f"direct node {node.node_id} has an empty proof")
            return render_declaration(node.goal, node.proof_content, kind="theorem")
        if node.proof_kind == "decomposition" and node.selected_and_id:
            decomposition: AndNode = self.graph.get_and(node.selected_and_id)
            if not decomposition.parent_proof:
                raise MaterializationError(
                    f"decomposition {decomposition.node_id} has an empty parent proof"
                )
            return render_declaration(node.goal, decomposition.parent_proof, kind="theorem")
        raise MaterializationError(f"node {node.node_id} cannot be rendered")


def referenced_support_ids(proof: str, supports: Sequence[OrNode]) -> list[str]:
    """Record exact generated lemma names used by a verified proof term."""
    return [
        node.node_id
        for node in supports
        if re.search(rf"(?<![A-Za-z0-9_']){re.escape(node.goal.name)}(?![A-Za-z0-9_'])", proof)
    ]


def leading_forall_parts(statement: str) -> tuple[str, str]:
    """Move leading explicit binders into a declaration without changing its Lean type."""
    text = statement.strip()
    marker_length = 1 if text.startswith("∀") else 6 if text.startswith("forall ") else 0
    if marker_length == 0:
        return "", text
    depth = 0
    for index, character in enumerate(text[marker_length:], start=marker_length):
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth = max(0, depth - 1)
        elif character == "," and depth == 0:
            binders = text[marker_length:index].strip()
            proposition = text[index + 1 :].strip()
            if binders and proposition:
                if not binders.startswith(("(", "{", "[")):
                    binders = f"({binders})"
                return binders, proposition
            break
    return "", text
