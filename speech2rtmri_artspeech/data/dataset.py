from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..common import read_jsonl


def _load_norm_stats(stats_path: str | Path) -> tuple[float, float]:
    payload = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    return float(payload["lower_value"]), float(payload["upper_value"])


def normalize_contour_array(contour: np.ndarray) -> np.ndarray:
    contour = np.asarray(contour, dtype=np.float32)
    contour = np.squeeze(contour)
    if contour.ndim != 2:
        raise ValueError(f"Expected 2D contour array after squeeze, got shape={tuple(contour.shape)}")
    if contour.shape[-1] == 2:
        return contour
    if contour.shape[0] == 2:
        return contour.T
    raise ValueError(f"Unsupported contour layout shape={tuple(contour.shape)}")


class ArtSpeechClipDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        config: dict[str, Any],
        audio_cache_root: str | Path,
        return_mode: str = "dict",
        load_contours: bool = True,
    ):
        if return_mode not in {"dict", "trainer_tuple"}:
            raise ValueError(f"Unsupported return_mode: {return_mode}")

        self.manifest_path = Path(manifest_path)
        self.rows = read_jsonl(manifest_path)
        self.config = config
        self.data_cfg = config["data"]
        self.audio_cfg = config["audio"]
        self.audio_cache_root = Path(audio_cache_root)
        self.return_mode = return_mode
        self.load_contours = load_contours
        self.repo_root = Path(config.get("_meta", {}).get("repo_root", Path(__file__).resolve().parents[2]))
        self._session_audio_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        low, high = _load_norm_stats(self._resolve_row_path(row["normalization_stats_path"]))

        video = self._load_video(row, low, high)
        conditioning_frame = self._load_conditioning_frame(row, low, high)
        audio_embeddings = self._slice_audio_embeddings(row)
        contour_targets = self._load_contours(row) if self.load_contours else {}

        metadata = {
            "speaker": row["speaker"],
            "session": row["session"],
            "split": row["split"],
            "registered": bool(row["registered"]),
            "frame_ids": list(row["frame_ids"]),
            "conditioning_frame": int(row["conditioning_frame"]),
            "wav_path": row["wav_path"],
            "frames_dir": str(self._resolve_row_path(row["frames_dir"])),
            "contours_dir": str(self._resolve_row_path(row["contours_dir"])),
            "selected_textgrid": row["selected_textgrid"],
            "orig_h": int(row["orig_h"]),
            "orig_w": int(row["orig_w"]),
            "scale_x": 1.0,
            "scale_y": 1.0,
            "audio_start_sec": float(row["audio_start_sec"]),
            "audio_end_sec": float(row["audio_end_sec"]),
            "target_contours_loaded": bool(self.load_contours),
        }

        if self.return_mode == "trainer_tuple":
            return video, audio_embeddings, conditioning_frame.unsqueeze(1)

        return {
            "video": video,
            "conditioning_frame": conditioning_frame,
            "audio_embeddings": audio_embeddings,
            "target_contours": contour_targets,
            "metadata": metadata,
        }

    def _resolve_row_path(self, raw_path: str | Path) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        manifest_relative = (self.manifest_path.parent / path).resolve()
        if manifest_relative.exists():
            return manifest_relative
        return (self.repo_root / path).resolve()

    def _load_video(self, row: dict[str, Any], low: float, high: float) -> torch.Tensor:
        frames = [
            self._load_frame(self._resolve_row_path(row["frames_dir"]) / f"{frame_id:04d}.npy", low, high)
            for frame_id in row["frame_ids"]
        ]
        video = torch.stack(frames, dim=1)
        return video

    def _load_conditioning_frame(self, row: dict[str, Any], low: float, high: float) -> torch.Tensor:
        path = self._resolve_row_path(row["frames_dir"]) / f"{int(row['conditioning_frame']):04d}.npy"
        return self._load_frame(path, low, high)

    def _load_frame(self, path: Path, low: float, high: float) -> torch.Tensor:
        frame = np.load(path).astype(np.float32)
        if frame.ndim != 2:
            frame = np.squeeze(frame)
        frame = np.clip((frame - low) / max(high - low, 1e-6), 0.0, 1.0)
        tensor = torch.from_numpy(frame).unsqueeze(0)

        target_size = int(self.data_cfg["frame_size"])
        if tensor.shape[-1] != target_size or tensor.shape[-2] != target_size:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return tensor

    def _slice_audio_embeddings(self, row: dict[str, Any]) -> torch.Tensor:
        cache = self._load_session_audio_cache(row)
        embeddings = cache["embeddings"]
        times_sec = cache["times_sec"]
        fps = float(row["fps"])
        sync_shift_frames = float(self.data_cfg.get("sync_shift_frames", 0))
        clip_times = torch.tensor(
            [((int(fid) - 1) + 0.5 + sync_shift_frames) / fps for fid in row["frame_ids"]],
            dtype=torch.float32,
        )
        indices = []
        for clip_time in clip_times:
            idx = int(torch.argmin(torch.abs(times_sec - clip_time)).item())
            indices.append(idx)
        sliced = embeddings[indices]

        if bool(self.audio_cfg.get("pooling", False)):
            sliced = sliced.mean(dim=0, keepdim=True)

        return sliced

    def _load_session_audio_cache(self, row: dict[str, Any]) -> dict[str, Any]:
        key = (row["speaker"], row["session"])
        if key not in self._session_audio_cache:
            cache_path = (
                self.audio_cache_root
                / self.audio_cfg["encoder"]
                / row["speaker"]
                / f"{row['session']}.pt"
            )
            self._session_audio_cache[key] = torch.load(cache_path, map_location="cpu")
        return self._session_audio_cache[key]

    def _load_contours(self, row: dict[str, Any]) -> dict[str, torch.Tensor]:
        targets: dict[str, list[torch.Tensor]] = {}
        for articulator in self.data_cfg["required_articulators"]:
            per_frame = []
            for frame_id in row["frame_ids"]:
                contour_path = self._resolve_row_path(row["contours_dir"]) / f"{int(frame_id):04d}_{articulator}.npy"
                contour = np.load(contour_path)
                contour_shape = tuple(np.asarray(contour).shape)
                try:
                    contour = normalize_contour_array(contour)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid contour shape at {contour_path}: shape={contour_shape}"
                    ) from exc
                per_frame.append(torch.from_numpy(contour))
            targets[articulator] = torch.stack(per_frame, dim=0)
        return targets
