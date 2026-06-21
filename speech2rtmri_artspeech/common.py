from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_random_seed(seed: int | None) -> None:
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_sample_indices(
    dataset_size: int,
    limit: int | None = None,
    selection: str = "head",
) -> list[int]:
    if dataset_size <= 0:
        return []

    if limit is None or int(limit) <= 0 or int(limit) >= dataset_size:
        return list(range(dataset_size))

    limit = int(limit)
    if selection == "head":
        return list(range(limit))
    if selection == "evenly_spaced":
        return np.linspace(0, dataset_size - 1, num=limit, dtype=int).tolist()

    raise ValueError(f"Unsupported sample selection: {selection!r}")


def setup_logging(log_path: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("speech2rtmri_artspeech")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        ensure_dir(log_path.parent)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, payload: Any, indent: int = 2) -> None:
    Path(path).write_text(json.dumps(payload, indent=indent, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
