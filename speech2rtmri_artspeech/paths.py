from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _normalize_prefix(raw_path: str) -> str:
    return raw_path.replace("\\", "/").rstrip("/")


@dataclass(frozen=True)
class PathMapping:
    source: str
    target: str

    @property
    def normalized_source(self) -> str:
        return _normalize_prefix(self.source)

    @property
    def normalized_target(self) -> str:
        return _normalize_prefix(self.target)


class ConfiguredPathMapper:
    def __init__(self, mappings: Iterable[dict[str, str]] | Iterable[PathMapping]):
        built: list[PathMapping] = []
        for item in mappings:
            if isinstance(item, PathMapping):
                built.append(item)
            else:
                built.append(PathMapping(source=item["source"], target=item["target"]))
        self.mappings = built

    def map_path_string(self, raw_path: str) -> str:
        normalized = _normalize_prefix(raw_path)
        for mapping in self.mappings:
            source = mapping.normalized_source
            if normalized == source or normalized.startswith(source + "/"):
                suffix = normalized[len(source):]
                return mapping.normalized_target + suffix
        return raw_path

    def resolve_existing_path(self, raw_path: str) -> Path:
        candidates = [raw_path]
        mapped = self.map_path_string(raw_path)
        if mapped != raw_path:
            candidates.insert(0, mapped)

        for candidate in candidates:
            candidate_path = Path(candidate)
            if candidate_path.exists():
                return candidate_path
        return Path(mapped)
