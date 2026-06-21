from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .common import get_repo_root


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config is None:
        raise ValueError(f"Empty config file: {config_path}")

    base_config = config.pop("base_config", None)
    if base_config:
        base_path = (config_path.parent / base_config).resolve()
        base_payload = load_config(base_path)
        config = _deep_merge(base_payload, config)

    repo_root = get_repo_root()
    runtime_cfg = config.setdefault("runtime", {})
    output_dir = runtime_cfg.get("output_dir")
    if output_dir:
        output_dir_path = Path(output_dir)
        if not output_dir_path.is_absolute():
            runtime_cfg["output_dir"] = str((repo_root / output_dir_path).resolve())

    config["_meta"] = {
        "config_path": str(config_path),
        "repo_root": str(repo_root),
    }
    return config


def dump_config_snapshot(config: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
