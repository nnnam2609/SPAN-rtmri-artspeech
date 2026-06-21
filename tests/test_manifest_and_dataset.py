from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import torch

from speech2rtmri_artspeech.data.dataset import ArtSpeechClipDataset
from speech2rtmri_artspeech.data.manifest import build_manifests


def _write_wav(path: Path, sr: int = 16000, duration_sec: float = 1.0) -> None:
    samples = (np.sin(2 * np.pi * 220 * np.arange(int(sr * duration_sec)) / sr) * 0.1)
    samples_i16 = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(samples_i16.tobytes())


def _write_textgrid(path: Path) -> None:
    path.write_text(
        '\n'.join([
            'File type = "ooTextFile"',
            'Object class = "TextGrid"',
            "",
            "xmin = 0",
            "xmax = 1",
            "tiers? <exists>",
            "size = 1",
            "item []:",
            "    item [1]:",
            '        class = "IntervalTier"',
            '        name = "SentenceTier"',
            "        xmin = 0",
            "        xmax = 1",
            "        intervals: size = 1",
            "        intervals [1]:",
            "            xmin = 0",
            "            xmax = 1",
            '            text = "hello"',
        ]),
        encoding="utf-8",
    )


def _make_fake_session(root: Path, speaker: str, session: str, registered: bool) -> None:
    session_dir = root / "ArtSpeech_Database_2" / speaker / session
    session_dir.mkdir(parents=True, exist_ok=True)
    _write_wav(session_dir / f"{speaker}_{session}.wav")
    _write_textgrid(session_dir / f"{speaker}_{session}_adjusted.textgrid")

    frames_dir = session_dir / ("NPY_MR_registered" if registered else "NPY_MR")
    contours_dir = session_dir / ("inference_contours_registered" if registered else "inference_contours")
    frames_dir.mkdir(parents=True, exist_ok=True)
    contours_dir.mkdir(parents=True, exist_ok=True)

    articulators = [
        "upper-lip",
        "lower-lip",
        "tongue",
        "soft-palate-midline",
        "pharynx",
        "epiglottis",
        "arytenoid-cartilage",
        "vocal-folds",
        "upper-incisor",
    ]
    for frame_id in range(1, 7):
        np.save(frames_dir / f"{frame_id:04d}.npy", np.full((136, 136), frame_id, dtype=np.uint16))
        for articulator in articulators:
            contour = np.stack([
                np.linspace(0, 10, 50),
                np.linspace(frame_id, frame_id + 5, 50),
            ], axis=1)
            if frame_id == 2 and articulator == "tongue":
                contour = contour.T
            np.save(contours_dir / f"{frame_id:04d}_{articulator}.npy", contour)


def _config(tmp_path: Path) -> dict:
    return {
        "data": {
            "iadi_root": str(tmp_path),
            "dataset_name": "ArtSpeech_Database_2",
            "path_mappings": [],
            "use_registered": True,
            "fallback_to_unregistered": True,
            "textgrid_priority": ["adjusted"],
            "required_articulators": [
                "upper-lip",
                "lower-lip",
                "tongue",
                "soft-palate-midline",
                "pharynx",
                "epiglottis",
                "arytenoid-cartilage",
                "vocal-folds",
                "upper-incisor",
            ],
            "fps": 50,
            "clip_len": 3,
            "clip_stride": 3,
            "conditioning_mode": "first_frame",
            "sync_shift_frames": 0,
            "pixel_spacing_mm": 1.62,
            "frame_size": 136,
            "normalization": {
                "lower_percentile": 1.0,
                "upper_percentile": 99.0,
                "cache_dir": str(tmp_path / "shared_norm"),
                "max_sampled_frames": 2,
            },
        },
        "splits": {
            "train": {"speakers": ["1775"], "sessions_by_speaker": {"1775": ["S10"]}},
            "val": {"speakers": ["1775"], "sessions_by_speaker": {"1775": ["S11"]}},
            "test": {"speakers": ["1775"], "sessions_by_speaker": {"1775": ["S12"]}},
        },
        "audio": {"encoder": "mfcc", "embedding_dim": 13, "pooling": False},
        "runtime": {
            "output_dir": str(tmp_path / "outputs"),
            "reuse_existing_manifests": False,
        },
    }


def test_manifest_builder_and_dataset(tmp_path: Path) -> None:
    _make_fake_session(tmp_path, "1775", "S10", registered=False)
    _make_fake_session(tmp_path, "1775", "S11", registered=True)
    _make_fake_session(tmp_path, "1775", "S12", registered=True)

    config = _config(tmp_path)
    logger = __import__("logging").getLogger("test")
    summary = build_manifests(config, logger)
    assert summary["splits"]["train"]["num_clips"] >= 1
    manifest_path = Path(config["runtime"]["output_dir"]) / "manifests" / "train.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["speaker"] == "1775"
    assert rows[0]["clip_len"] == 3
    stats_path = Path(rows[0]["normalization_stats_path"])
    stats_payload = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats_path.parent.parent == tmp_path / "shared_norm"
    assert stats_payload["num_frames_total"] == 6
    assert stats_payload["num_frames_used"] == 2

    audio_cache_dir = Path(config["runtime"]["output_dir"]) / "audio_cache" / "mfcc" / "1775"
    audio_cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": torch.randn(20, 13),
            "times_sec": torch.linspace(0.0, 1.0, 20),
            "metadata": {"encoder": "mfcc"},
        },
        audio_cache_dir / "S10.pt",
    )

    dataset = ArtSpeechClipDataset(manifest_path, config, Path(config["runtime"]["output_dir"]) / "audio_cache", return_mode="dict")
    sample = dataset[0]
    assert sample["video"].shape == (1, 3, 136, 136)
    assert sample["conditioning_frame"].shape == (1, 136, 136)
    assert sample["audio_embeddings"].shape[-1] == 13
    assert sample["video"].dtype == torch.float32
    assert sample["target_contours"]["tongue"].shape == (3, 50, 2)

    dataset_no_contours = ArtSpeechClipDataset(
        manifest_path,
        config,
        Path(config["runtime"]["output_dir"]) / "audio_cache",
        return_mode="dict",
        load_contours=False,
    )
    sample_no_contours = dataset_no_contours[0]
    assert sample_no_contours["target_contours"] == {}
    assert sample_no_contours["metadata"]["target_contours_loaded"] is False

    second_config = _config(tmp_path)
    second_config["runtime"]["output_dir"] = str(tmp_path / "outputs_second")
    summary_second = build_manifests(second_config, logger)
    assert summary_second["splits"]["train"]["num_clips"] >= 1
    second_manifest_path = Path(second_config["runtime"]["output_dir"]) / "manifests" / "train.jsonl"
    second_rows = [json.loads(line) for line in second_manifest_path.read_text(encoding="utf-8").splitlines()]
    assert second_rows[0]["normalization_stats_path"] == rows[0]["normalization_stats_path"]
