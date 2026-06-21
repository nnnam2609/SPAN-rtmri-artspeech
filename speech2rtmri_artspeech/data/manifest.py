from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..common import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from ..paths import ConfiguredPathMapper


def _speaker_sessions(split_cfg: dict[str, Any], speaker_dir: Path) -> list[str]:
    requested = split_cfg.get("sessions_by_speaker", {}).get(speaker_dir.name)
    if requested:
        return list(requested)
    return sorted(
        path.name for path in speaker_dir.iterdir()
        if path.is_dir() and path.name.startswith("S")
    )


def _basename_candidates(speaker: str, session: str, priority: list[str]) -> list[str]:
    suffix_map = {
        "adjusted": f"{speaker}_{session}_adjusted.textgrid",
        "DENOISED": f"{speaker}_{session}_DENOISED.textgrid",
        "original": f"{speaker}_{session}_original.textgrid",
        "mocap": f"{speaker}_{session}_mocap.textgrid",
    }
    return [suffix_map[item] for item in priority if item in suffix_map]


def _compute_histogram_percentiles(
    frame_paths: list[Path],
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[float, float]:
    histogram = np.zeros(65536, dtype=np.int64)
    total = 0
    for frame_path in frame_paths:
        frame = np.load(frame_path)
        if frame.ndim != 2:
            frame = np.squeeze(frame)
        if frame.dtype != np.uint16:
            frame = frame.astype(np.uint16)
        histogram += np.bincount(frame.ravel(), minlength=65536)
        total += int(frame.size)

    if total == 0:
        raise ValueError("No frames found while computing normalization statistics")

    cumulative = np.cumsum(histogram)
    low_rank = total * lower_percentile / 100.0
    high_rank = total * upper_percentile / 100.0
    low_value = int(np.searchsorted(cumulative, low_rank, side="left"))
    high_value = int(np.searchsorted(cumulative, high_rank, side="left"))
    if high_value <= low_value:
        high_value = low_value + 1
    return float(low_value), float(high_value)


def _sample_frame_paths(frame_paths: list[Path], max_sampled_frames: int | None) -> list[Path]:
    if not frame_paths:
        return []
    if max_sampled_frames is None or max_sampled_frames <= 0 or len(frame_paths) <= max_sampled_frames:
        return frame_paths

    indices = np.linspace(0, len(frame_paths) - 1, num=max_sampled_frames, dtype=int)
    unique_indices = sorted(set(int(index) for index in indices))
    return [frame_paths[index] for index in unique_indices]


@dataclass
class SessionMaterial:
    speaker: str
    session: str
    split: str
    session_dir: Path
    wav_path: Path
    textgrid_path: Path
    frames_dir: Path
    contours_dir: Path
    registered: bool
    orig_h: int
    orig_w: int
    normalization_stats_path: Path
    num_frames: int


class ManifestBuilder:
    def __init__(self, config: dict[str, Any], logger):
        self.config = config
        self.logger = logger
        self.data_cfg = config["data"]
        self.runtime_cfg = config["runtime"]
        self.repo_root = Path(config.get("_meta", {}).get("repo_root", Path(__file__).resolve().parents[2]))
        self.mapper = ConfiguredPathMapper(self.data_cfg.get("path_mappings", []))

        mapped_root = self.mapper.resolve_existing_path(self.data_cfg["iadi_root"])
        self.iadi_root = mapped_root
        self.dataset_root = mapped_root / self.data_cfg["dataset_name"]
        self.output_root = Path(self.runtime_cfg["output_dir"])
        self.manifest_dir = ensure_dir(self.output_root / "manifests")
        normalization_cfg = self.data_cfg["normalization"]
        norm_cache_dir = normalization_cfg.get("cache_dir")
        if norm_cache_dir:
            raw_norm_dir = Path(norm_cache_dir)
            if not raw_norm_dir.is_absolute():
                raw_norm_dir = (self.repo_root / raw_norm_dir).resolve()
            self.norm_dir = ensure_dir(raw_norm_dir)
        else:
            self.norm_dir = ensure_dir(self.output_root / "normalization_stats")
        self.required_articulators = set(self.data_cfg["required_articulators"])
        self.require_contours_for_manifest = bool(self.data_cfg.get("require_contours_for_manifest", False))
        self.lower_percentile = float(normalization_cfg["lower_percentile"])
        self.upper_percentile = float(normalization_cfg["upper_percentile"])
        max_sampled_frames = normalization_cfg.get("max_sampled_frames")
        self.max_sampled_frames = None if max_sampled_frames in (None, "", 0) else int(max_sampled_frames)
        self.reuse_existing_manifests = bool(self.runtime_cfg.get("reuse_existing_manifests", False))
        self.skipped_sessions: dict[str, Counter] = defaultdict(Counter)

    def build(self) -> dict[str, Any]:
        if self.reuse_existing_manifests:
            existing_summary = self._load_existing_summary()
            if existing_summary is not None:
                self.logger.info("Reusing existing manifests from %s", self.manifest_dir)
                return existing_summary

        summary: dict[str, Any] = {
            "dataset_root": str(self.dataset_root),
            "splits": {},
            "skipped_sessions": {},
            "normalization_cache_dir": str(self.norm_dir),
            "max_sampled_frames": self.max_sampled_frames,
            "require_contours_for_manifest": self.require_contours_for_manifest,
        }

        for split_name in ("train", "val", "test"):
            split_start = time.perf_counter()
            rows = self._build_split(split_name)
            output_path = self.manifest_dir / f"{split_name}.jsonl"
            write_jsonl(output_path, rows)
            summary["splits"][split_name] = {
                "manifest_path": str(output_path),
                "num_clips": len(rows),
                "num_sessions_with_clips": len({(row["speaker"], row["session"]) for row in rows}),
                "elapsed_seconds": round(time.perf_counter() - split_start, 3),
            }

        for split_name, counter in self.skipped_sessions.items():
            summary["skipped_sessions"][split_name] = dict(counter)

        write_json(self.manifest_dir / "manifest_summary.json", summary)
        return summary

    def _load_existing_summary(self) -> dict[str, Any] | None:
        manifest_paths = [self.manifest_dir / f"{split_name}.jsonl" for split_name in ("train", "val", "test")]
        if not all(path.exists() for path in manifest_paths):
            return None

        summary_path = self.manifest_dir / "manifest_summary.json"
        if summary_path.exists():
            return read_json(summary_path)

        summary: dict[str, Any] = {
            "dataset_root": str(self.dataset_root),
            "splits": {},
            "skipped_sessions": {},
            "normalization_cache_dir": str(self.norm_dir),
            "max_sampled_frames": self.max_sampled_frames,
            "require_contours_for_manifest": self.require_contours_for_manifest,
        }
        for split_name, manifest_path in zip(("train", "val", "test"), manifest_paths):
            rows = read_jsonl(manifest_path)
            summary["splits"][split_name] = {
                "manifest_path": str(manifest_path),
                "num_clips": len(rows),
                "num_sessions_with_clips": len({(row["speaker"], row["session"]) for row in rows}),
                "elapsed_seconds": 0.0,
            }
        write_json(summary_path, summary)
        return summary

    def _build_split(self, split_name: str) -> list[dict[str, Any]]:
        split_cfg = self.config["splits"][split_name]
        rows: list[dict[str, Any]] = []
        max_clips = split_cfg.get("max_clips")

        for speaker in split_cfg["speakers"]:
            speaker_dir = self.dataset_root / speaker
            if not speaker_dir.exists():
                self._skip(split_name, f"missing_speaker:{speaker}")
                continue

            sessions = _speaker_sessions(split_cfg, speaker_dir)
            self.logger.info("Split %s speaker %s: scanning %d session(s)", split_name, speaker, len(sessions))
            for session_index, session in enumerate(sessions, start=1):
                self.logger.info("Split %s speaker %s: session %s (%d/%d)", split_name, speaker, session, session_index, len(sessions))
                session_material = self._resolve_session(split_name, speaker, session)
                if session_material is None:
                    continue
                session_rows = self._build_session_rows(session_material)
                if not session_rows:
                    self._skip(split_name, f"no_valid_clips:{speaker}/{session}")
                    continue

                rows.extend(session_rows)
                self.logger.info(
                    "Split %s speaker %s session %s: added %d clip(s) from %d frame(s)",
                    split_name,
                    speaker,
                    session,
                    len(session_rows),
                    session_material.num_frames,
                )
                if max_clips is not None and len(rows) >= max_clips:
                    return rows[:max_clips]

        return rows

    def _resolve_session(self, split_name: str, speaker: str, session: str) -> SessionMaterial | None:
        session_dir = self.dataset_root / speaker / session
        if not session_dir.exists():
            self._skip(split_name, f"missing_session:{speaker}/{session}")
            return None

        wav_path = session_dir / f"{speaker}_{session}.wav"
        if not wav_path.exists():
            self._skip(split_name, f"missing_wav:{speaker}/{session}")
            return None

        textgrid_path = self._select_textgrid(session_dir, speaker, session)
        if textgrid_path is None:
            self._skip(split_name, f"missing_textgrid:{speaker}/{session}")
            return None

        frames_dir, contours_dir, registered = self._select_modalities(session_dir)
        if frames_dir is None or contours_dir is None:
            self._skip(split_name, f"missing_modalities:{speaker}/{session}")
            return None

        frame_paths = sorted(frames_dir.glob("*.npy"))
        if not frame_paths:
            self._skip(split_name, f"no_frames:{speaker}/{session}")
            return None

        sample = np.load(frame_paths[0])
        if sample.ndim != 2:
            sample = np.squeeze(sample)
        if sample.ndim != 2:
            self._skip(split_name, f"bad_frame_shape:{speaker}/{session}")
            return None
        orig_h, orig_w = map(int, sample.shape)

        stats_path = self._write_normalization_stats(speaker, session, frames_dir, frame_paths)
        return SessionMaterial(
            speaker=speaker,
            session=session,
            split=split_name,
            session_dir=session_dir,
            wav_path=wav_path,
            textgrid_path=textgrid_path,
            frames_dir=frames_dir,
            contours_dir=contours_dir,
            registered=registered,
            orig_h=orig_h,
            orig_w=orig_w,
            normalization_stats_path=stats_path,
            num_frames=len(frame_paths),
        )

    def _select_textgrid(self, session_dir: Path, speaker: str, session: str) -> Path | None:
        for candidate in _basename_candidates(speaker, session, self.data_cfg["textgrid_priority"]):
            path = session_dir / candidate
            if path.exists():
                return path
        return None

    def _select_modalities(self, session_dir: Path) -> tuple[Path | None, Path | None, bool]:
        use_registered = bool(self.data_cfg["use_registered"])
        fallback = bool(self.data_cfg["fallback_to_unregistered"])

        registered_frames = session_dir / "NPY_MR_registered"
        registered_contours = session_dir / "inference_contours_registered"
        unregistered_frames = session_dir / "NPY_MR"
        unregistered_contours = session_dir / "inference_contours"

        if use_registered and registered_frames.exists() and registered_contours.exists():
            return registered_frames, registered_contours, True

        if use_registered and not fallback:
            return None, None, False

        if unregistered_frames.exists() and unregistered_contours.exists():
            return unregistered_frames, unregistered_contours, False

        return None, None, False

    def _write_normalization_stats(self, speaker: str, session: str, frames_dir: Path, frame_paths: list[Path]) -> Path:
        stats_path = self.norm_dir / speaker / f"{session}.json"
        if stats_path.exists():
            cached = read_json(stats_path)
            if (
                cached.get("frames_dir") == str(frames_dir)
                and float(cached.get("lower_percentile", -1)) == self.lower_percentile
                and float(cached.get("upper_percentile", -1)) == self.upper_percentile
            ):
                self.logger.info("Reusing normalization stats for %s/%s from %s", speaker, session, stats_path)
                return stats_path

            self.logger.warning("Refreshing stale normalization stats for %s/%s at %s", speaker, session, stats_path)

        sampled_paths = _sample_frame_paths(frame_paths, self.max_sampled_frames)
        self.logger.info(
            "Computing normalization stats for %s/%s using %d/%d frame(s)",
            speaker,
            session,
            len(sampled_paths),
            len(frame_paths),
        )
        lower, upper = _compute_histogram_percentiles(sampled_paths, self.lower_percentile, self.upper_percentile)
        ensure_dir(stats_path.parent)
        payload = {
            "speaker": speaker,
            "session": session,
            "frames_dir": str(frames_dir),
            "lower_percentile": self.lower_percentile,
            "upper_percentile": self.upper_percentile,
            "lower_value": lower,
            "upper_value": upper,
            "num_frames_total": len(frame_paths),
            "num_frames_used": len(sampled_paths),
            "sampling_strategy": "all_frames" if len(sampled_paths) == len(frame_paths) else "evenly_spaced",
        }
        write_json(stats_path, payload)
        return stats_path

    def _build_session_rows(self, session_material: SessionMaterial) -> list[dict[str, Any]]:
        frame_ids = {int(path.stem) for path in session_material.frames_dir.glob("*.npy")}
        if self.require_contours_for_manifest:
            contour_index = self._index_contours(session_material.contours_dir)
            eligible_frames = sorted(
                fid for fid in frame_ids if self.required_articulators.issubset(contour_index.get(fid, set()))
            )
        else:
            eligible_frames = sorted(frame_ids)

        if not eligible_frames:
            return []

        eligible_set = set(eligible_frames)
        clip_len = int(self.data_cfg["clip_len"])
        clip_stride = int(self.data_cfg["clip_stride"])
        conditioning_mode = self.data_cfg["conditioning_mode"]
        fps = float(self.data_cfg["fps"])
        sync_shift_frames = float(self.data_cfg.get("sync_shift_frames", 0))

        rows: list[dict[str, Any]] = []
        min_frame = min(eligible_frames)
        max_frame = max(eligible_frames)
        for start_frame in range(min_frame, max_frame - clip_len + 2, clip_stride):
            clip_frame_ids = list(range(start_frame, start_frame + clip_len))
            if not all(frame_id in eligible_set for frame_id in clip_frame_ids):
                continue

            conditioning_frame = start_frame if conditioning_mode == "first_frame" else start_frame - 1
            if conditioning_frame not in frame_ids:
                continue

            end_frame = clip_frame_ids[-1]
            audio_start_sec = (start_frame - 1 + sync_shift_frames) / fps
            audio_end_sec = (end_frame + sync_shift_frames) / fps
            rows.append({
                "speaker": session_material.speaker,
                "session": session_material.session,
                "wav_path": str(session_material.wav_path),
                "frames_dir": str(session_material.frames_dir),
                "contours_dir": str(session_material.contours_dir),
                "registered": session_material.registered,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "clip_len": clip_len,
                "fps": fps,
                "split": session_material.split,
                "conditioning_mode": conditioning_mode,
                "conditioning_frame": conditioning_frame,
                "frame_ids": clip_frame_ids,
                "audio_start_sec": audio_start_sec,
                "audio_end_sec": audio_end_sec,
                "selected_textgrid": str(session_material.textgrid_path),
                "pixel_spacing_mm": float(self.data_cfg["pixel_spacing_mm"]),
                "orig_h": session_material.orig_h,
                "orig_w": session_material.orig_w,
                "normalization_stats_path": str(session_material.normalization_stats_path),
                "contours_required_for_manifest": self.require_contours_for_manifest,
            })
        return rows

    def _index_contours(self, contours_dir: Path) -> dict[int, set[str]]:
        index: dict[int, set[str]] = defaultdict(set)
        for contour_path in contours_dir.glob("*.npy"):
            stem = contour_path.stem
            if "_" not in stem:
                continue
            frame_raw, articulator = stem.split("_", 1)
            if frame_raw.isdigit():
                index[int(frame_raw)].add(articulator)
        return index

    def _skip(self, split_name: str, reason: str) -> None:
        self.skipped_sessions[split_name][reason] += 1
        self.logger.warning("Skipping session: %s", reason)


def build_manifests(config: dict[str, Any], logger) -> dict[str, Any]:
    builder = ManifestBuilder(config, logger)
    return builder.build()
