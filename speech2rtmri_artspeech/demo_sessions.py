from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

from .common import ensure_dir, set_random_seed, setup_logging, timestamp, write_json
from .config import load_config
from .data.dataset import ArtSpeechClipDataset
from .media import tensor_frame_to_uint8
from .models.diffusion import ImagenVideoDiffusionWrapper


def _parse_sessions(raw: str) -> list[str]:
    sessions = [item.strip() for item in raw.split(",") if item.strip()]
    if not sessions:
        raise ValueError("No session was provided")
    return sessions


def _load_rows(manifest_path: Path) -> list[dict[str, Any]]:
    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _session_clip_indices(
    rows: list[dict[str, Any]],
    speaker: str,
    requested_sessions: list[str],
) -> dict[str, list[int]]:
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    wanted = set(requested_sessions)
    for index, row in enumerate(rows):
        if row["speaker"] != speaker:
            continue
        session = row["session"]
        if session not in wanted:
            continue
        grouped[session].append((index, int(row["frame_ids"][0])))

    result: dict[str, list[int]] = {}
    for session in requested_sessions:
        clips = grouped.get(session, [])
        clips.sort(key=lambda item: item[1])
        result[session] = [index for index, _ in clips]
    return result


def _open_video_writer(path: Path, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one stitched demo video per hold-out session")
    parser.add_argument("--config", required=True, help="Path to the YAML config")
    parser.add_argument("--checkpoint", required=True, help="Trainer checkpoint path")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split to sample from")
    parser.add_argument("--speaker", required=True, help="Speaker id to render (e.g., 1804)")
    parser.add_argument("--sessions", required=True, help="Comma separated session list (e.g., S10,S11,S12)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    args = parser.parse_args()

    config = load_config(args.config)
    set_random_seed(config.get("runtime", {}).get("seed"))
    output_root = Path(config["runtime"]["output_dir"])
    run_id = f"demo_sessions_{args.speaker}_{timestamp()}"
    demo_root = ensure_dir(output_root / "demo_sessions" / run_id)
    logger = setup_logging(demo_root / "demo_sessions.log")

    requested_sessions = _parse_sessions(args.sessions)
    manifest_path = output_root / "manifests" / f"{args.split}.jsonl"
    rows = _load_rows(manifest_path)
    session_to_indices = _session_clip_indices(rows, args.speaker, requested_sessions)

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

    demo_cfg = config.get("demo", {})
    fps = int(config["data"]["fps"])
    run_summary = {
        "speaker": args.speaker,
        "split": args.split,
        "sessions": [],
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
    }

    for session in requested_sessions:
        indices = session_to_indices.get(session, [])
        if not indices:
            logger.warning("No clips found for speaker=%s session=%s in split=%s", args.speaker, session, args.split)
            continue

        session_dir = ensure_dir(demo_root / f"{args.speaker}_{session}")
        gt_writer = _open_video_writer(session_dir / "gt_session.mp4", fps=fps)
        pred_writer = _open_video_writer(session_dir / "pred_session.mp4", fps=fps)
        side_writer = _open_video_writer(session_dir / "side_by_side_session.mp4", fps=fps)

        seen_frame_ids: set[int] = set()
        written_frames = 0
        source_clips = 0
        logger.info(
            "Rendering speaker=%s session=%s with %d clip(s)",
            args.speaker,
            session,
            len(indices),
        )

        for index in indices:
            item = dataset[index]
            prediction = wrapper.sample(
                text_embeds=item["audio_embeddings"].unsqueeze(0),
                cond_video_frames=item["conditioning_frame"].unsqueeze(0).unsqueeze(2),
                video_frames=item["video"].shape[1],
                cond_scale=float(demo_cfg.get("cond_scale", 1.0)),
                num_sample_steps=demo_cfg.get("sample_num_steps"),
            )[0].detach().cpu()
            target = item["video"].detach().cpu()

            source_clips += 1
            frame_ids = item["metadata"]["frame_ids"]
            for frame_index, frame_id in enumerate(frame_ids):
                frame_id = int(frame_id)
                if frame_id in seen_frame_ids:
                    continue
                seen_frame_ids.add(frame_id)

                gt_frame = tensor_frame_to_uint8(target[:, frame_index])
                pred_frame = tensor_frame_to_uint8(prediction[:, frame_index])
                side_frame = np.concatenate([gt_frame, pred_frame], axis=1)

                gt_writer.append_data(gt_frame)
                pred_writer.append_data(pred_frame)
                side_writer.append_data(side_frame)
                written_frames += 1

        gt_writer.close()
        pred_writer.close()
        side_writer.close()

        session_summary = {
            "speaker": args.speaker,
            "session": session,
            "num_clips": len(indices),
            "num_source_clips_processed": source_clips,
            "num_unique_frames_written": written_frames,
            "gt_video": str(session_dir / "gt_session.mp4"),
            "pred_video": str(session_dir / "pred_session.mp4"),
            "side_by_side_video": str(session_dir / "side_by_side_session.mp4"),
        }
        write_json(session_dir / "session_summary.json", session_summary)
        run_summary["sessions"].append(session_summary)
        logger.info(
            "Completed speaker=%s session=%s unique_frames=%d",
            args.speaker,
            session,
            written_frames,
        )

    write_json(demo_root / "run_summary.json", run_summary)
    logger.info("Session demo complete. output_dir=%s", demo_root)


if __name__ == "__main__":
    main()
