"""Bounded shared-memory views for autonomous subsystems."""

from __future__ import annotations

import re
from typing import Any

from ..artifact_store import ArtifactStore
from .models import KernelState, RecordCard, RecordStatus, ResearchRecord, ResearchView


def build_view(
    store: ArtifactStore,
    state: KernelState,
    *,
    subsystem: str,
    record_limit: int = 80,
    action_limit: int = 12,
) -> ResearchView:
    full_task = store.read_text(ArtifactStore.RESEARCH_TASK)
    task = _bounded_text(full_task, 16_000)
    records = store.read_records()
    selected = _select_records(records, task=task, subsystem=subsystem, limit=record_limit)
    selected = _fit_record_budget(selected, 14_000)
    actions = store.read_action_events()
    recent_actions: list[dict[str, Any]] = []
    for action in actions[-action_limit:]:
        recent_actions.append(
            {
                "action_id": action.action_id,
                "subsystem": action.subsystem,
                "action_type": action.proposal.action_type,
                "title": action.proposal.title[:200],
                "status": action.status.value,
                "summary": action.summary[:300],
                "error": action.error[:300],
            }
        )
    return ResearchView(
        task=task,
        task_sha256=state.task_sha256,
        task_revision=state.task_revision,
        cycle=state.cycle,
        subsystem=subsystem,
        records=[
            RecordCard(
                record_id=record.record_id,
                producer=record.producer,
                kind=record.kind,
                status=record.status,
                title=record.title,
                summary=record.summary,
                relation=record.relation,
                parent_ids=record.parent_ids,
                created_at=record.created_at,
            )
            for record in selected
        ],
        subsystem_state=store.load_subsystem_state(subsystem),
        recent_actions=recent_actions,
    )


def _select_records(
    records: list[ResearchRecord], *, task: str, subsystem: str, limit: int
) -> list[ResearchRecord]:
    if len(records) <= limit:
        return records
    anchors = _terms(task + " " + subsystem)
    ranked: list[tuple[tuple[int, int, int], ResearchRecord]] = []
    for index, record in enumerate(records):
        evidence_rank = {
            RecordStatus.verified: 3,
            RecordStatus.observed: 2,
            RecordStatus.tentative: 1,
        }[record.status]
        overlap = len(anchors & _terms(record.title + " " + record.summary))
        ranked.append(((evidence_rank, overlap, index), record))
    ranked_records = [
        record for _, record in sorted(ranked, key=lambda row: row[0], reverse=True)
    ]
    recent_count = min(len(records), max(1, limit // 3))
    selected: list[ResearchRecord] = list(reversed(records[-recent_count:]))
    seen = {record.record_id for record in selected}
    for record in ranked_records:
        if len(selected) >= limit:
            break
        if record.record_id not in seen:
            selected.append(record)
            seen.add(record.record_id)
    return selected


def _fit_record_budget(records: list[ResearchRecord], limit: int) -> list[ResearchRecord]:
    chosen: list[ResearchRecord] = []
    used = 0
    for record in records:
        size = len(record.title) + len(record.summary) + 200
        if chosen and used + size > limit:
            continue
        chosen.append(record)
        used += size
    return chosen


def _bounded_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    head = int(limit * 0.75)
    tail = limit - head
    return value[:head] + "\n\n...[task middle truncated by kernel]...\n\n" + value[-tail:]


def _terms(value: str) -> set[str]:
    stop = {
        "about", "after", "before", "could", "from", "have", "into", "should", "that",
        "their", "there", "these", "this", "through", "using", "what", "when", "where",
        "which", "with", "would",
    }
    return {term for term in re.findall(r"[a-z0-9]{4,}", value.lower()) if term not in stop}
