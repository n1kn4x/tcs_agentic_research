"""Deterministic human views over the immutable research journal."""

from __future__ import annotations

from collections import Counter

from ..artifact_store import ArtifactStore
from .models import KernelState, ResearchRecord


def write_reports(store: ArtifactStore, state: KernelState) -> None:
    records = store.read_records()
    actions = store.read_action_events()
    latest_actions = {}
    for action in actions:
        latest_actions[action.action_id] = action

    status_lines = [
        "# Research status",
        "",
        f"- Task revision: **{state.task_revision}**",
        f"- Kernel cycle: **{state.cycle}**",
        f"- Runtime phase: **{state.phase.value}**",
        f"- Active action: `{state.active_action_id or 'none'}`",
        "- Scientific completion: **not inferred by the kernel**",
        "",
        "## Record counts",
        "",
    ]
    counts = Counter((record.status.value, record.kind.value) for record in records)
    if not counts:
        status_lines.append("No research records yet.")
    else:
        status_lines.extend(
            f"- `{status}/{kind}`: {count}"
            for (status, kind), count in sorted(counts.items())
        )
    status_lines.extend(["", "## Subsystems", ""])
    for name in state.enabled_subsystems:
        actor_actions = [action for action in latest_actions.values() if action.subsystem == name]
        last = max(actor_actions, key=lambda action: action.cycle) if actor_actions else None
        status_lines.append(
            f"- **{name}**: "
            + (
                f"last `{last.status.value}` — {last.summary or last.proposal.title}"
                if last
                else "not run"
            )
        )
    store.write_text("Reports/Status.md", "\n".join(status_lines).rstrip() + "\n")
    store.write_text("Reports/Research.md", render_research_report(records))


def render_research_report(records: list[ResearchRecord]) -> str:
    lines = [
        "# Research memory",
        "",
        "This is a deterministic view of the append-only record journal. `verified` means Lean-",
        "checked; `observed` means source- or execution-backed; `tentative` means model-authored",
        "analysis. The report does not claim that the project is complete.",
        "",
    ]
    if not records:
        return "\n".join(lines + ["No records yet.", ""])
    for status in ["verified", "observed", "tentative"]:
        selected = [record for record in records if record.status.value == status]
        if not selected:
            continue
        lines.extend([f"## {status.title()}", ""])
        for record in selected:
            lines.extend(
                [
                    f"### {record.title}",
                    f"- ID: `{record.record_id}`",
                    f"- Kind: `{record.kind.value}`; producer: `{record.producer}`",
                    f"- Relation: `{record.relation.value}` to "
                    + (", ".join(f"`{item}`" for item in record.parent_ids) or "no parent"),
                    f"- Evidence: `{record.evidence_type.value}`",
                    "",
                    record.summary,
                    "",
                ]
            )
            if record.body:
                lines.extend([record.body, ""])
            if record.artifact_refs:
                lines.extend(
                    [
                        "Artifacts: "
                        + ", ".join(f"`{ref.path}`" for ref in record.artifact_refs),
                        "",
                    ]
                )
    return "\n".join(lines).rstrip() + "\n"
