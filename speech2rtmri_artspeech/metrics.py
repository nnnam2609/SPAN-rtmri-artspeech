from __future__ import annotations

from typing import Any

import numpy as np
import torch


def compute_psnr(target: torch.Tensor, prediction: torch.Tensor, eps: float = 1e-8) -> float:
    mse = torch.mean((target - prediction) ** 2).item()
    if mse <= eps:
        return float("inf")
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def compute_ssim(target: torch.Tensor, prediction: torch.Tensor) -> float:
    try:
        from torchmetrics.functional import structural_similarity_index_measure
    except ImportError as exc:
        raise ImportError("torchmetrics is required for SSIM computation") from exc

    if target.ndim == 3:
        target = target.unsqueeze(0)
        prediction = prediction.unsqueeze(0)
    return float(structural_similarity_index_measure(prediction, target, data_range=1.0).item())


def compute_video_metrics(target_video: torch.Tensor, pred_video: torch.Tensor) -> dict[str, Any]:
    target_video = target_video.detach().cpu()
    pred_video = pred_video.detach().cpu()
    frame_metrics = []
    for frame_index in range(target_video.shape[1]):
        target = target_video[:, frame_index]
        pred = pred_video[:, frame_index]
        frame_metrics.append({
            "frame_index": frame_index,
            "psnr": compute_psnr(target, pred),
            "ssim": compute_ssim(target, pred),
        })

    psnr_values = [item["psnr"] for item in frame_metrics if np.isfinite(item["psnr"])]
    ssim_values = [item["ssim"] for item in frame_metrics]
    return {
        "psnr_mean": float(np.mean(psnr_values)) if psnr_values else float("inf"),
        "ssim_mean": float(np.mean(ssim_values)) if ssim_values else float("nan"),
        "lpips": None,
        "fvd": None,
        "frame_metrics": frame_metrics,
    }


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _median_distance_mm(a: np.ndarray, b: np.ndarray, pixel_spacing_mm: float) -> float:
    distances = np.linalg.norm(a - b, axis=-1) * pixel_spacing_mm
    return float(np.median(distances))


def _temporal_jitter(contours: np.ndarray) -> float:
    if contours.shape[0] < 3:
        return 0.0
    velocities = np.diff(contours, axis=0)
    accelerations = np.diff(velocities, axis=0)
    return float(np.mean(np.linalg.norm(accelerations, axis=-1)))


def compute_contour_metrics(
    target_contours: dict[str, np.ndarray],
    pred_contours: dict[str, np.ndarray],
    pixel_spacing_mm: float = 1.62,
) -> dict[str, Any]:
    per_articulator = {}
    all_rmse_px = []
    all_rmse_mm = []
    all_median_mm = []
    all_p2cp = []
    all_jitter = []

    try:
        from vt_tools.metrics import p2cp_rms
    except ImportError:
        p2cp_rms = None

    for articulator, target in target_contours.items():
        if articulator not in pred_contours:
            continue
        pred = pred_contours[articulator]
        rmse_px = _rmse(target, pred)
        rmse_mm = rmse_px * pixel_spacing_mm
        median_mm = _median_distance_mm(target, pred, pixel_spacing_mm)
        jitter = _temporal_jitter(pred)
        p2cp_value = None
        if p2cp_rms is not None:
            values = [
                float(p2cp_rms(target_frame, pred_frame))
                for target_frame, pred_frame in zip(target, pred)
            ]
            p2cp_value = float(np.mean(values))
            all_p2cp.append(p2cp_value)

        per_articulator[articulator] = {
            "rmse_px": rmse_px,
            "rmse_mm": rmse_mm,
            "median_mm": median_mm,
            "p2cp_rms_px": p2cp_value,
            "temporal_jitter_px": jitter,
        }
        all_rmse_px.append(rmse_px)
        all_rmse_mm.append(rmse_mm)
        all_median_mm.append(median_mm)
        all_jitter.append(jitter)

    return {
        "rmse_px_mean": float(np.mean(all_rmse_px)) if all_rmse_px else None,
        "rmse_mm_mean": float(np.mean(all_rmse_mm)) if all_rmse_mm else None,
        "median_mm_mean": float(np.mean(all_median_mm)) if all_median_mm else None,
        "p2cp_rms_px_mean": float(np.mean(all_p2cp)) if all_p2cp else None,
        "temporal_jitter_px_mean": float(np.mean(all_jitter)) if all_jitter else None,
        "per_articulator": per_articulator,
    }
