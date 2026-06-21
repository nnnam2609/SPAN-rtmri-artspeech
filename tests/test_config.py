from __future__ import annotations

from pathlib import Path

from speech2rtmri_artspeech.common import select_sample_indices
from speech2rtmri_artspeech.config import load_config


def test_load_config_with_base() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "configs/speech2rtmri_artspeech/full_database_grouille_a100_10h.yaml")
    assert config["data"]["frame_size"] == 136
    assert config["audio"]["encoder"] == "hubert"
    assert config["audio"]["device"] == "cuda"
    assert config["splits"]["train"]["speakers"] == ["1775", "1777", "1789", "1791"]
    assert config["runtime"]["seed"] == 1337
    assert Path(config["runtime"]["output_dir"]).name == "speech2rtmri_artspeech_full_database_grouille_a100_10h"
    assert config["runtime"]["reuse_existing_manifests"] is True


def test_select_sample_indices_evenly_spaced() -> None:
    assert select_sample_indices(10, 4, "evenly_spaced") == [0, 3, 6, 9]
    assert select_sample_indices(5, 10, "evenly_spaced") == [0, 1, 2, 3, 4]
    assert select_sample_indices(5, 2, "head") == [0, 1]
