from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .common import ensure_dir, select_sample_indices, set_random_seed, setup_logging, timestamp, write_json
from .config import load_config
from .data.dataset import ArtSpeechClipDataset
from .metrics import compute_video_metrics
from .models.diffusion import ImagenVideoDiffusionWrapper
from .reporting import generate_eval_report


def _evaluation_indices(dataset_size: int, eval_cfg: dict) -> list[int]:
    return select_sample_indices(
        dataset_size=dataset_size,
        limit=eval_cfg.get("max_eval_samples"),
        selection=str(eval_cfg.get("sample_selection", "evenly_spaced")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ArtSpeech DB2 diffusion checkpoints")
    parser.add_argument("--config", required=True, help="Path to the YAML config")
    parser.add_argument("--checkpoint", required=True, help="Trainer checkpoint path")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split to evaluate")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    args = parser.parse_args()

    config = load_config(args.config)
    set_random_seed(config.get("runtime", {}).get("seed"))
    output_root = Path(config["runtime"]["output_dir"])
    run_id = f"eval_{timestamp()}"
    eval_root = ensure_dir(output_root / "eval" / run_id)
    logger = setup_logging(eval_root / "evaluate.log")

    manifest_path = output_root / "manifests" / f"{args.split}.jsonl"
    dataset = ArtSpeechClipDataset(
        manifest_path,
        config,
        output_root / "audio_cache",
        return_mode="dict",
        load_contours=False,
    )
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    wrapper = ImagenVideoDiffusionWrapper(config, logger, device)
    wrapper.load(args.checkpoint)
    eval_cfg = config.get("eval", {})
    eval_indices = _evaluation_indices(len(dataset), eval_cfg)
    logger.info(
        "Evaluating split=%s on %s/%s sample(s)",
        args.split,
        len(eval_indices),
        len(dataset),
    )

    per_sample_rows = []
    psnr_values = []
    ssim_values = []

    for index in eval_indices:
        item = dataset[index]
        prediction = wrapper.sample(
            text_embeds=item["audio_embeddings"].unsqueeze(0),
            cond_video_frames=item["conditioning_frame"].unsqueeze(0).unsqueeze(2),
            video_frames=item["video"].shape[1],
            cond_scale=float(config["eval"].get("cond_scale", 1.0)),
            num_sample_steps=eval_cfg.get("sample_num_steps"),
        )[0].detach().cpu()
        target = item["video"].detach().cpu()
        metrics = compute_video_metrics(target, prediction)
        metadata = item["metadata"]
        per_sample_rows.append({
            "speaker": metadata["speaker"],
            "session": metadata["session"],
            "start_frame": metadata["frame_ids"][0],
            "end_frame": metadata["frame_ids"][-1],
            "psnr_mean": metrics["psnr_mean"],
            "ssim_mean": metrics["ssim_mean"],
            "contour_metrics_mode": "video_only",
        })
        psnr_values.append(metrics["psnr_mean"])
        ssim_values.append(metrics["ssim_mean"])

    csv_path = eval_root / "per_sample_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_sample_rows[0].keys()) if per_sample_rows else [])
        if per_sample_rows:
            writer.writeheader()
            writer.writerows(per_sample_rows)

    summary = {
        "split": args.split,
        "num_samples": len(per_sample_rows),
        "num_available_samples": len(dataset),
        "psnr_mean": float(np.mean(psnr_values)) if psnr_values else None,
        "ssim_mean": float(np.mean(ssim_values)) if ssim_values else None,
        "lpips": None,
        "fvd": None,
        "contour_metrics_mode": "video_only",
        "checkpoint": str(args.checkpoint),
        "sample_selection": eval_cfg.get("sample_selection", "all"),
        "max_eval_samples": eval_cfg.get("max_eval_samples"),
    }
    write_json(eval_root / "metrics_summary.json", summary)
    generate_eval_report(eval_root, per_sample_rows=per_sample_rows, summary=summary, logger=logger)
    logger.info("Evaluation complete. output_dir=%s", eval_root)


if __name__ == "__main__":
    main()
