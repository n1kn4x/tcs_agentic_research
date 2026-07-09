"""Prompt-facing serialization helpers.

The artifact store intentionally keeps durable artifacts self-contained and audit-friendly, which
means repeated claim/proposal/report/artifact-ref objects can appear in multiple places.  This
module provides a model-facing serializer that removes only exact duplicate values from prompt
payloads.  It does not change canonical artifacts and it keeps prompts self-contained by including
one full definition for every emitted reference.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


_ID_FIELD_LABELS: tuple[tuple[str, str], ...] = (
    ("artifact_id", "artifact"),
    ("event_id", "event"),
    ("report_id", "report"),
    ("result_id", "result"),
    ("verdict_id", "verdict"),
    ("evidence_id", "evidence"),
    ("obligation_id", "obligation"),
    ("claim_id", "claim"),
    ("proposal_id", "proposal"),
    ("task_id", "task"),
    ("answer_id", "literature_answer"),
    ("extract_id", "extract"),
    ("paper_id", "paper"),
    ("candidate_id", "candidate"),
    ("run_id", "run"),
    ("goal_id", "goal"),
    ("statement_id", "statement"),
    ("quote_id", "quote"),
    ("call_id", "model_call"),
)

_GENERIC_DICT_MIN_CHARS = 700
_GENERIC_LIST_MIN_CHARS = 1000
_LONG_STRING_MIN_CHARS = 1000


class _Candidate(tuple):
    """Hashable candidate key with readable fields.

    A tiny tuple subclass keeps Counter keys cheap while making intent explicit.  Layout:
    ``(kind, digest, semantic_label)``.
    """

    __slots__ = ()

    def __new__(cls, kind: str, digest: str, semantic_label: str) -> "_Candidate":
        return tuple.__new__(cls, (kind, digest, semantic_label))

    @property
    def kind(self) -> str:
        return self[0]

    @property
    def digest(self) -> str:
        return self[1]

    @property
    def semantic_label(self) -> str:
        return self[2]



def compact_json_dumps(payload: Any) -> str:
    """Serialize ``payload`` as minified, self-contained JSON with exact duplicates referenced once.

    The function is intentionally conservative:
    - canonical data is first converted to plain JSON-compatible values;
    - only exact duplicate values are replaced, so no information is summarized or discarded;
    - every emitted ``{"$ref": "..."}`` has its full value in the top-level ``"$defs"`` table;
    - ID-bearing objects and artifact references are eligible even when they are relatively small;
    - generic containers and strings are deduplicated only when large enough to avoid noisy refs;
    - prompt JSON is minified to avoid spending context on pretty-printing whitespace.

    The returned string is meant for LLM prompts, not for persistence as canonical state.
    """

    plain = _to_plain(payload)
    candidates, counts = _collect_candidates(plain)
    repeated = {candidate for candidate, count in counts.items() if count > 1}
    final_payload = plain

    if repeated:
        candidate_refs = _assign_reference_ids(repeated)
        used_refs: set[str] = set()
        compact_payload = _replace_repeated_values(plain, candidate_refs, used_refs)

        if used_refs:
            definitions = {
                ref_id: candidates[candidate]
                for candidate, ref_id in sorted(
                    candidate_refs.items(), key=lambda item: item[1]
                )
                if ref_id in used_refs
            }
            final_payload = {
                "$deduplication": (
                    "This prompt JSON is self-contained. Any value of the form "
                    "{'$ref': '<id>'} is an exact replacement for the full value stored at "
                    "top-level '$defs[<id>]'. No information has been summarized or omitted; "
                    "only repeated exact text/data was moved to '$defs'."
                ),
                "$defs": definitions,
                "payload": compact_payload,
            }

    _maybe_print_prompt_compaction_stats(plain, final_payload)
    return _json_dumps(final_payload)



def _collect_candidates(value: Any) -> tuple[dict[_Candidate, Any], Counter[_Candidate]]:
    candidates: dict[_Candidate, Any] = {}
    counts: Counter[_Candidate] = Counter()

    def visit(node: Any) -> None:
        candidate = _candidate_for(node)
        if candidate is not None:
            candidates.setdefault(candidate, node)
            counts[candidate] += 1
        if isinstance(node, dict):
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return candidates, counts



def _replace_repeated_values(
    value: Any, candidate_refs: dict[_Candidate, str], used_refs: set[str]
) -> Any:
    candidate = _candidate_for(value)
    if candidate is not None and candidate in candidate_refs:
        ref_id = candidate_refs[candidate]
        used_refs.add(ref_id)
        return {"$ref": ref_id}
    if isinstance(value, dict):
        return {
            key: _replace_repeated_values(child, candidate_refs, used_refs)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_replace_repeated_values(child, candidate_refs, used_refs) for child in value]
    return value



def _assign_reference_ids(candidates: set[_Candidate]) -> dict[_Candidate, str]:
    assigned: dict[_Candidate, str] = {}
    used: set[str] = set()
    for candidate in sorted(
        candidates,
        key=lambda item: (item.semantic_label or item.kind, item.digest),
    ):
        label = candidate.semantic_label or f"{candidate.kind}:{candidate.digest[:12]}"
        base = f"{label}@{candidate.digest[:10]}"
        ref_id = base
        suffix = 2
        while ref_id in used:
            ref_id = f"{base}:{suffix}"
            suffix += 1
        used.add(ref_id)
        assigned[candidate] = ref_id
    return assigned



def _candidate_for(value: Any) -> _Candidate | None:
    if isinstance(value, dict):
        encoded = _canonical_json(value)
        semantic_label = _semantic_dict_label(value)
        if semantic_label is None and len(encoded) < _GENERIC_DICT_MIN_CHARS:
            return None
        return _Candidate("object", _digest(encoded), semantic_label or "")
    if isinstance(value, list):
        encoded = _canonical_json(value)
        if len(encoded) < _GENERIC_LIST_MIN_CHARS:
            return None
        return _Candidate("array", _digest(encoded), "")
    if isinstance(value, str) and len(value) >= _LONG_STRING_MIN_CHARS:
        encoded = _canonical_json(value)
        return _Candidate("string", _digest(encoded), "string")
    return None



def _semantic_dict_label(value: dict[str, Any]) -> str | None:
    path = value.get("path")
    if isinstance(path, str) and (
        "sha256" in value or "kind" in value or "summary" in value or "created_at" in value
    ):
        return "artifact_ref:" + _safe_label(path)
    for field_name, label in _ID_FIELD_LABELS:
        field_value = value.get(field_name)
        if isinstance(field_value, str) and field_value:
            return f"{label}:{_safe_label(field_value)}"
    return None



def _to_plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _to_plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(child) for child in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value



def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))



def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=False, ensure_ascii=False, separators=(",", ":"))



def _pretty_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=False, ensure_ascii=False, indent=2)



def _maybe_print_prompt_compaction_stats(raw_payload: Any, final_payload: Any) -> None:
    if not os.environ.get("TCS_PROMPT_COMPACT_STATS"):
        return

    raw_pretty_chars = len(_pretty_json_dumps(raw_payload))
    raw_minified_chars = len(_json_dumps(raw_payload))
    final_pretty_chars = len(_pretty_json_dumps(final_payload))
    final_minified_chars = len(_json_dumps(final_payload))
    refs_emitted = _count_prompt_refs(final_payload)
    defs_emitted = _count_prompt_defs(final_payload)

    print(
        "[prompt-compact] "
        f"raw_pretty_chars={raw_pretty_chars} "
        f"raw_minified_chars={raw_minified_chars} "
        f"dedup_pretty_chars={final_pretty_chars} "
        f"dedup_minified_chars={final_minified_chars} "
        f"pretty_saved_raw_chars={raw_pretty_chars - raw_minified_chars} "
        f"dedup_saved_pretty_chars={raw_pretty_chars - final_pretty_chars} "
        f"dedup_saved_minified_chars={raw_minified_chars - final_minified_chars} "
        f"pretty_saved_after_dedup_chars={final_pretty_chars - final_minified_chars} "
        f"total_saved_vs_raw_pretty_chars={raw_pretty_chars - final_minified_chars} "
        f"refs_emitted={refs_emitted} "
        f"defs_emitted={defs_emitted}",
        file=sys.stderr,
    )



def _count_prompt_refs(value: Any) -> int:
    if isinstance(value, dict):
        if set(value) == {"$ref"}:
            return 1
        return sum(_count_prompt_refs(child) for child in value.values())
    if isinstance(value, list):
        return sum(_count_prompt_refs(child) for child in value)
    return 0



def _count_prompt_defs(value: Any) -> int:
    if isinstance(value, dict):
        defs = value.get("$defs")
        if isinstance(defs, dict):
            return len(defs)
    return 0



def _digest(encoded: str) -> str:
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()



def _safe_label(value: str, *, limit: int = 120) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._=:/-" else "_" for ch in value.strip())
    safe = safe.strip("_") or "unnamed"
    if len(safe) <= limit:
        return safe
    digest = hashlib.sha256(safe.encode("utf-8")).hexdigest()[:8]
    return f"{safe[:limit - 9]}~{digest}"
