"""Nomenclature loading, mapping, and update utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import yaml

from ..artifact_store import ArtifactStore
from ..schemas import ArtifactRef, NomenclatureEntry, utc_now


@dataclass(frozen=True)
class LoadedNomenclature:
    raw: dict
    entries: list[NomenclatureEntry]


class NomenclatureMapper:
    """Apply and update the workspace's canonical notation table."""

    def __init__(self, store: ArtifactStore):
        self.store = store

    def load(self) -> LoadedNomenclature:
        if not self.store.exists(ArtifactStore.NOMENCLATURE):
            self.store.initialize_layout()
        raw = yaml.safe_load(self.store.read_text(ArtifactStore.NOMENCLATURE)) or {}
        entries = []
        for item in raw.get("symbols") or []:
            try:
                entries.append(NomenclatureEntry.model_validate(item))
            except Exception:
                # Keep the mapper robust to hand-edited tables.
                if isinstance(item, dict) and item.get("symbol"):
                    entries.append(
                        NomenclatureEntry(
                            symbol=str(item.get("symbol")),
                            canonical_name=str(item.get("canonical_name") or item.get("symbol")),
                            aliases=[str(a) for a in item.get("aliases") or []],
                            definition=str(item.get("definition") or ""),
                            convention=str(item.get("convention") or ""),
                        )
                    )
        return LoadedNomenclature(raw=raw, entries=entries)

    def alias_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for entry in self.load().entries:
            canonical = entry.symbol or entry.canonical_name
            if entry.symbol:
                mapping[entry.symbol] = canonical
            # canonical_name is descriptive, not automatically a textual alias; mapping it
            # aggressively would rewrite ordinary prose such as "dimension" into a symbol.
            for alias in entry.aliases:
                if alias:
                    mapping[alias] = canonical
        return mapping

    def map_text(self, text: str, extra_mappings: dict[str, str] | None = None) -> str:
        """Return text rewritten to canonical symbols where the table has explicit aliases."""
        mapped = text
        mappings = self.alias_map()
        mappings.update({k: v for k, v in (extra_mappings or {}).items() if k and v})
        for alias, canonical in sorted(mappings.items(), key=lambda kv: len(kv[0]), reverse=True):
            if not alias or alias == canonical:
                continue
            mapped = _replace_token(mapped, alias, canonical)
        return mapped

    def update_from_mappings(
        self,
        mappings: dict[str, str],
        *,
        source_refs: Iterable[ArtifactRef] = (),
        new_entries: Iterable[NomenclatureEntry] = (),
    ) -> list[NomenclatureEntry]:
        """Merge discovered notation mappings into ``Nomenclature.yml``.

        ``mappings`` is interpreted as ``paper-local symbol/alias -> canonical symbol``. If the
        canonical symbol already exists, the local symbol is added as an alias. Otherwise a new
        entry is created so later agents can inspect and refine it.
        """
        new_entries_list = list(new_entries)
        if not mappings and not new_entries_list:
            return []

        loaded = self.load()
        raw = dict(loaded.raw)
        entries = [entry.model_copy(deep=True) for entry in loaded.entries]
        refs = list(source_refs)
        changed: list[NomenclatureEntry] = []

        def find_entry(name: str) -> NomenclatureEntry | None:
            lowered = name.lower()
            for entry in entries:
                candidates = [entry.symbol, entry.canonical_name, *entry.aliases]
                if any(c.lower() == lowered for c in candidates if c):
                    return entry
            return None

        for original, canonical in mappings.items():
            original = str(original).strip()
            canonical = str(canonical).strip()
            if not original or not canonical:
                continue
            entry = find_entry(canonical)
            if entry is None:
                entry = NomenclatureEntry(
                    symbol=canonical,
                    canonical_name=canonical,
                    aliases=[] if original == canonical else [original],
                    definition="Discovered during literature ingestion; inspect/refine this entry.",
                    source_refs=refs,
                )
                entries.append(entry)
                changed.append(entry)
            else:
                if original != entry.symbol and original not in entry.aliases:
                    entry.aliases.append(original)
                for ref in refs:
                    if ref.path not in {existing.path for existing in entry.source_refs}:
                        entry.source_refs.append(ref)
                changed.append(entry)

        for candidate in new_entries_list:
            existing = find_entry(candidate.symbol) or find_entry(candidate.canonical_name)
            if existing is None:
                entry = candidate.model_copy(deep=True)
                for ref in refs:
                    if ref.path not in {existing_ref.path for existing_ref in entry.source_refs}:
                        entry.source_refs.append(ref)
                entries.append(entry)
                changed.append(entry)
            else:
                for alias in [candidate.symbol, candidate.canonical_name, *candidate.aliases]:
                    if alias and alias != existing.symbol and alias not in existing.aliases:
                        existing.aliases.append(alias)
                if candidate.definition and not existing.definition:
                    existing.definition = candidate.definition
                if candidate.convention and not existing.convention:
                    existing.convention = candidate.convention
                for ref in [*candidate.source_refs, *refs]:
                    if ref.path not in {existing_ref.path for existing_ref in existing.source_refs}:
                        existing.source_refs.append(ref)
                changed.append(existing)

        raw["version"] = raw.get("version") or 1
        raw["updated_at"] = utc_now()
        raw["symbols"] = [entry.model_dump(mode="json") for entry in entries]
        if not isinstance(raw.get("conventions"), list):
            raw["conventions"] = []
        if not isinstance(raw.get("notes"), list):
            raw["notes"] = []
        if changed:
            note = (
                "Literature ingestion updated aliases/new symbols; "
                "inspect source_refs for provenance."
            )
            if note not in raw["notes"]:
                raw["notes"].append(note)
        self.store.write_yaml(ArtifactStore.NOMENCLATURE, raw)
        return changed

    def used_mappings_for_text(self, text: str) -> dict[str, str]:
        used: dict[str, str] = {}
        for alias, canonical in self.alias_map().items():
            if alias and alias != canonical and _contains_token(text, alias):
                used[alias] = canonical
        return used


def _replace_token(text: str, needle: str, replacement: str) -> str:
    if not needle:
        return text
    # For pure word-like aliases use word boundaries. For math-ish aliases, require that the
    # match is not embedded in a longer alphanumeric/control-word token.
    if re.fullmatch(r"[A-Za-z0-9_]+", needle):
        pattern = re.compile(rf"\b{re.escape(needle)}\b")
    else:
        pattern = re.compile(rf"(?<![A-Za-z0-9_\\]){re.escape(needle)}(?![A-Za-z0-9_])")
    return pattern.sub(replacement, text)


def _contains_token(text: str, needle: str) -> bool:
    return _replace_token(text, needle, "__FOUND__") != text
