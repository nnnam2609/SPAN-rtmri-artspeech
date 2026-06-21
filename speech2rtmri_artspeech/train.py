from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import torch

from .common import ensure_dir, select_sample_indices, set_random_seed, setup_logging, timestamp, write_json
from .config import dump_config_snapshot, load_config
from .data.dataset import ArtSpeechClipDataset
from .media import save_side_by_side_video, save_video
from .metrics import compute_video_metrics
from .models.diffusion import ImagenVideoDiffusionWrapper
from .reporting import generate_training_report


def _save_training_sample(
    wrapper,
    sample_item,
    sample_dir: Path,
    fps: int,
    cond_scale: float,
    sample_num_steps: int | None,
) -> dict:
    sample_dir.mkdir(parents=True, exist_ok=True)
    prediction = wrapper.sample(
        text_embeds=sample_item["audio_embeddings"].unsqueeze(0),
        cond_video_frames=sample_item["conditioning_frame"].unsqueeze(0).unsqueeze(2),
        video_frames=sample_item["video"].shape[1],
        cond_scale=cond_scale,
        num_sample_steps=sample_num_steps,
    )[0].detach().cpu()
    target = sample_item["video"].detach().cpu()

    save_video(target, sample_dir / "gt_video.mp4", fps=fps)
    save_video(prediction, sample_dir / "pred_video.mp4", fps=fps)
    save_side_by_side_video(target, prediction, sample_dir / "side_by_side.mp4", fps=fps)
    metrics = compute_video_metrics(target, prediction)
    write_json(sample_dir / "metrics.json", metrics)
    return metrics


def _visual_benchmark(
    wrapper,
    benchmark_items: list[dict],
    sample_root: Path,
    fps: int,
    cond_scale: float,
    sample_num_steps: int | None,
) -> dict:
    sample_root.mkdir(parents=True, exist_ok=True)
    sample_metrics = []
    for item in benchmark_items:
        metadata = item["metadata"]
        sample_id = (
            f"{metadata['speaker']}_{metadata['session']}"
            f"_{metadata['frame_ids'][0]:04d}_{metadata['frame_ids'][-1]:04d}"
        )
        metrics = _save_training_sample(
            wrapper=wrapper,
            sample_item=item,
            sample_dir=sample_root / sample_id,
            fps=fps,
            cond_scale=cond_scale,
            sample_num_steps=sample_num_steps,
        )
        sample_metrics.append(
            {
                "sample_id": sample_id,
                "psnr_mean": metrics["psnr_mean"],
                "ssim_mean": metrics["ssim_mean"],
            }
        )

    benchmark_summary = {
        "num_samples": len(sample_metrics),
        "psnr_mean": sum(item["psnr_mean"] for item in sample_metrics) / max(len(sample_metrics), 1),
        "ssim_mean": sum(item["ssim_mean"] for item in sample_metrics) / max(len(sample_metrics), 1),
        "samples": sample_metrics,
    }
    write_json(sample_root / "benchmark_summary.json", benchmark_summary)
    return benchmark_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ArtSpeech DB2 diffusion model")
    parser.add_argument("--config", required=True, help="Path to the YAML config")
    parser.add_argument("--checkpoint", default=None, help="Optional trainer checkpoint to resume from")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    args = parser.parse_args()

    config = load_config(args.config)
    runtime_cfg = config.setdefault("runtime", {})
    set_random_seed(runtime_cfg.get("seed"))
    output_root = Path(config["runtime"]["output_dir"])
    run_id = f"train_{timestamp()}"
    run_dir = ensure_dir(output_root / "train" / run_id)
    logger = setup_logging(run_dir / "train.log")
    dump_config_snapshot(config, run_dir / "config_snapshot.yaml")
    train_cfg = config["train"]

    manifest_dir = output_root / "manifests"
    audio_cache_root = output_root / "audio_cache"
    train_dataset = ArtSpeechClipDataset(
        manifest_dir / "train.jsonl",
        config,
        audio_cache_root,
        return_mode="trainer_tuple",
        load_contours=False,
    )
    valid_manifest = manifest_dir / "val.jsonl"
    valid_dataset = None
    if valid_manifest.exists():
        valid_dataset = ArtSpeechClipDataset(
            valid_manifest,
            config,
            audio_cache_root,
            return_mode="trainer_tuple",
            load_contours=False,
        )

    benchmark_source_manifest = valid_manifest if valid_manifest.exists() else manifest_dir / "train.jsonl"
    benchmark_dataset = ArtSpeechClipDataset(
        benchmark_source_manifest,
        config,
        audio_cache_root,
        return_mode="dict",
        load_contours=False,
    )
    benchmark_indices = select_sample_indices(
        len(benchmark_dataset),
        int(train_cfg.get("visual_benchmark_count", 4)),
        "evenly_spaced",
    )
    benchmark_items = [benchmark_dataset[index] for index in benchmark_indices]

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    logger.info("Training on device=%s", device)
    micro_batch_size = int(train_cfg["batch_size"])
    gradient_accumulation_steps = int(train_cfg["gradient_accumulation_steps"])
    effective_batch_size = micro_batch_size * gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch_size)
    requested_max_steps = train_cfg.get("max_train_steps", 0)
    if requested_max_steps is None or int(requested_max_steps) <= 0:
        max_steps = steps_per_epoch * max(1, int(train_cfg.get("epochs", 1)))
    else:
        max_steps = int(requested_max_steps)
    logger.info(
        "train_clips=%s valid_clips=%s effective_batch_size=%s steps_per_epoch=%s requested_max_steps=%s resolved_max_steps=%s",
        len(train_dataset),
        len(valid_dataset) if valid_dataset is not None else 0,
        effective_batch_size,
        steps_per_epoch,
        requested_max_steps,
        max_steps,
    )
    logger.info(
        "visual_benchmark_source=%s visual_benchmark_indices=%s",
        benchmark_source_manifest.name,
        benchmark_indices,
    )

    wrapper = ImagenVideoDiffusionWrapper(config, logger, device)
    wrapper.register_datasets(
        train_dataset=train_dataset,
        train_batch_size=effective_batch_size,
        valid_dataset=valid_dataset,
        valid_batch_size=effective_batch_size,
    )
    resume_checkpoint = args.checkpoint or train_cfg.get("resume_checkpoint")
    resumed_steps = 0
    if resume_checkpoint:
        wrapper.load(resume_checkpoint)
        resumed_steps = wrapper.num_steps_taken(unet_number=1)
        logger.info("Resumed from checkpoint=%s at step=%s", resume_checkpoint, resumed_steps)
    if resumed_steps >= max_steps:
        logger.info("Checkpoint already at or beyond requested max_steps=%s, skipping training loop", max_steps)

    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    samples_dir = ensure_dir(run_dir / "samples")
    history_path = run_dir / "history.csv"
    best_checkpoint_path = checkpoints_dir / "best.pt"
    best_valid_checkpoint_path = checkpoints_dir / "best_valid_loss.pt"
    best_visual_checkpoint_path = checkpoints_dir / "best_visual.pt"
    last_checkpoint_path = checkpoints_dir / "last.pt"

    best_valid_loss = float("inf")
    best_visual_ssim = float("-inf")
    best_visual_psnr = float("-inf")
    valid_every = int(train_cfg["valid_every"])
    sample_every = int(train_cfg["sample_every"])
    checkpoint_every = int(train_cfg["checkpoint_every"])
    cond_scale = float(train_cfg.get("cond_scale", 1.0))
    ignore_time = bool(train_cfg.get("ignore_time", False))
    sample_num_steps = train_cfg.get("sample_num_steps")
    best_checkpoint_metric = str(train_cfg.get("best_checkpoint_metric", "visual_ssim"))
    history_fieldnames = ["step", "train_loss", "valid_loss", "sample_psnr", "sample_ssim"]

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_fieldnames)
        writer.writeheader()

        for step in range(resumed_steps + 1, max_steps + 1):
            train_loss = wrapper.train_step(
                unet_number=1,
                max_batch_size=micro_batch_size,
                ignore_time=ignore_time,
            )
            row = {"step": step, "train_loss": train_loss, "valid_loss": "", "sample_psnr": "", "sample_ssim": ""}

            if step % valid_every == 0:
                try:
                    valid_loss = wrapper.valid_step(
                        unet_number=1,
                        max_batch_size=micro_batch_size,
                        ignore_time=ignore_time,
                    )
                    row["valid_loss"] = valid_loss
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        wrapper.save(best_valid_checkpoint_path)
                        if best_checkpoint_metric == "valid_loss":
                            wrapper.save(best_checkpoint_path)
                except Exception as exc:  # pragma: no cover - runtime fallback
                    logger.warning("Validation step failed at step %s: %s", step, exc)

            if step % sample_every == 0 or step == max_steps:
                benchmark_metrics = _visual_benchmark(
                    wrapper,
                    benchmark_items,
                    samples_dir / f"step_{step:06d}",
                    fps=int(config["data"]["fps"]),
                    cond_scale=cond_scale,
                    sample_num_steps=sample_num_steps,
                )
                row["sample_psnr"] = benchmark_metrics["psnr_mean"]
                row["sample_ssim"] = benchmark_metrics["ssim_mean"]
                if benchmark_metrics["ssim_mean"] > best_visual_ssim:
                    best_visual_ssim = benchmark_metrics["ssim_mean"]
                    wrapper.save(best_visual_checkpoint_path)
                    if best_checkpoint_metric == "visual_ssim":
                        wrapper.save(best_checkpoint_path)
                if benchmark_metrics["psnr_mean"] > best_visual_psnr:
                    best_visual_psnr = benchmark_metrics["psnr_mean"]
                    if best_checkpoint_metric == "visual_psnr":
                        wrapper.save(best_checkpoint_path)

            if step % checkpoint_every == 0 or step == max_steps:
                wrapper.save(checkpoints_dir / f"step_{step:06d}.pt")
                wrapper.save(last_checkpoint_path)

            writer.writerow(row)
            handle.flush()
            logger.info(
                "step=%s/%s train_loss=%.6f valid_loss=%s sample_psnr=%s sample_ssim=%s",
                step,
                max_steps,
                train_loss,
                row["valid_loss"],
                row["sample_psnr"],
                row["sample_ssim"],
            )

    if not best_checkpoint_path.exists():
        for candidate in (best_visual_checkpoint_path, best_valid_checkpoint_path, last_checkpoint_path):
            if candidate.exists():
                shutil.copy2(candidate, best_checkpoint_path)
                break

    generate_training_report(run_dir, logger=logger)
    logger.info("Training complete. run_dir=%s", run_dir)


if __name__ == "__main__":
    main()
