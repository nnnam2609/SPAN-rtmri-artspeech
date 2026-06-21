from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import soundfile as sf
import torch
from PIL import Image


def tensor_frame_to_uint8(frame: torch.Tensor) -> np.ndarray:
    frame = frame.detach().cpu()
    if frame.ndim == 3 and frame.shape[0] == 1:
        frame = frame.repeat(3, 1, 1)
    if frame.ndim == 2:
        frame = frame.unsqueeze(0).repeat(3, 1, 1)
    if frame.ndim != 3:
        raise ValueError(f"Expected frame tensor [C, H, W], got {tuple(frame.shape)}")
    array = frame.clamp(0.0, 1.0).mul(255.0).byte().permute(1, 2, 0).numpy()
    return array


def save_video(video: torch.Tensor, path: str | Path, fps: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [tensor_frame_to_uint8(video[:, frame_idx]) for frame_idx in range(video.shape[1])]
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)


def save_side_by_side_video(target: torch.Tensor, prediction: torch.Tensor, path: str | Path, fps: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame_idx in range(target.shape[1]):
        left = tensor_frame_to_uint8(target[:, frame_idx])
        right = tensor_frame_to_uint8(prediction[:, frame_idx])
        frames.append(np.concatenate([left, right], axis=1))
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)


def save_summary_image(target: torch.Tensor, prediction: torch.Tensor, path: str | Path, max_frames: int = 4) -> None:
    tiles = []
    num_frames = min(target.shape[1], max_frames)
    for frame_idx in range(num_frames):
        top = Image.fromarray(tensor_frame_to_uint8(target[:, frame_idx]))
        bottom = Image.fromarray(tensor_frame_to_uint8(prediction[:, frame_idx]))
        canvas = Image.new("RGB", (top.width, top.height + bottom.height))
        canvas.paste(top, (0, 0))
        canvas.paste(bottom, (0, top.height))
        tiles.append(canvas)

    if not tiles:
        return

    grid = Image.new("RGB", (sum(tile.width for tile in tiles), tiles[0].height))
    x_offset = 0
    for tile in tiles:
        grid.paste(tile, (x_offset, 0))
        x_offset += tile.width
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def save_audio_segment(
    wav_path: str | Path,
    start_sec: float,
    end_sec: float,
    output_path: str | Path,
) -> None:
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    start_idx = max(int(start_sec * sr), 0)
    end_idx = min(int(end_sec * sr), len(audio))
    clipped = audio[start_idx:end_idx]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, clipped, sr)
