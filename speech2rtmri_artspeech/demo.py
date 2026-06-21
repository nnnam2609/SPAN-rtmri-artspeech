from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .common import ensure_dir, select_sample_indices, set_random_seed, setup_logging, timestamp, write_json
from .config import load_config
from .data.dataset import ArtSpeechClipDataset
from .media import save_audio_segment, save_side_by_side_video, save_summary_image, save_video
from .metrics import compute_video_metrics
from .models.diffusion import ImagenVideoDiffusionWrapper
from .reporting import generate_demo_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run demo sampling for ArtSpeech DB2")
    parser.add_argument("--config", required=True, help="Path to the YAML config")
    parser.add_argument("--checkpoint", required=True, help="Trainer checkpoint path")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split to sample from")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to render")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    args = parser.parse_args()

    config = load_config(args.config)
    set_random_seed(config.get("runtime", {}).get("seed"))
    output_root = Path(config["runtime"]["output_dir"])
    run_id = f"demo_{timestamp()}"
    demo_root = ensure_dir(output_root / "demo" / run_id)
    logger = setup_logging(demo_root / "demo.log")

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
    demo_rows = []
    demo_cfg = config.get("demo", {})
    demo_indices = select_sample_indices(
        len(dataset),
        min(args.num_samples, int(demo_cfg.get("max_demo_samples", args.num_samples))),
        str(demo_cfg.get("sample_selection", "head")),
    )

    for index in demo_indices:
        item = dataset[index]
        prediction = wrapper.sample(
            text_embeds=item["audio_embeddings"].unsqueeze(0),
            cond_video_frames=item["conditioning_frame"].unsqueeze(0).unsqueeze(2),
            video_frames=item["video"].shape[1],
            cond_scale=float(demo_cfg.get("cond_scale", 1.0)),
            num_sample_steps=demo_cfg.get("sample_num_steps"),
        )[0].detach().cpu()
        target = item["video"].detach().cpu()

        metadata = item["metadata"]
        sample_id = (
            f"{metadata['speaker']}_{metadata['session']}"
            f"_{metadata['frame_ids'][0]:04d}_{metadata['frame_ids'][-1]:04d}"
        )
        sample_dir = ensure_dir(demo_root / sample_id)

        save_audio_segment(
            metadata["wav_path"],
            metadata["audio_start_sec"],
            metadata["audio_end_sec"],
            sample_dir / "input_audio.wav",
        )
        save_video(target, sample_dir / "gt_video.mp4", fps=int(config["data"]["fps"]))
        save_video(prediction, sample_dir / "pred_video.mp4", fps=int(config["data"]["fps"]))
        save_side_by_side_video(target, prediction, sample_dir / "side_by_side.mp4", fps=int(config["data"]["fps"]))
        save_summary_image(target, prediction, sample_dir / "summary.png")

        metrics = compute_video_metrics(target, prediction)
        metrics["contour_metrics"] = None
        metrics["overlay_gt_pred_contours_mp4"] = None
        write_json(sample_dir / "metrics.json", metrics)
        demo_rows.append(
            {
                "sample_id": sample_id,
                "speaker": metadata["speaker"],
                "session": metadata["session"],
                "start_frame": metadata["frame_ids"][0],
                "end_frame": metadata["frame_ids"][-1],
                "psnr_mean": metrics["psnr_mean"],
                "ssim_mean": metrics["ssim_mean"],
            }
        )

    generate_demo_report(demo_root, sample_rows=demo_rows, logger=logger)
    logger.info("Demo complete. output_dir=%s", demo_root)


if __name__ == "__main__":
    main()
