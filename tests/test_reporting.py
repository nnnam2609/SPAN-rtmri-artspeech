from __future__ import annotations

import csv
import json
from pathlib import Path

from speech2rtmri_artspeech.reporting import (
    generate_demo_report,
    generate_eval_report,
    generate_phase_report,
    generate_training_report,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_reporting_generates_figures(tmp_path: Path) -> None:
    train_run = tmp_path / "train" / "train_20260416_010000"
    _write_csv(
        train_run / "history.csv",
        ["step", "train_loss", "valid_loss", "sample_psnr", "sample_ssim"],
        [
            {"step": 1, "train_loss": 1.0, "valid_loss": "", "sample_psnr": "", "sample_ssim": ""},
            {"step": 2, "train_loss": 0.8, "valid_loss": 0.9, "sample_psnr": 8.0, "sample_ssim": 0.01},
            {"step": 3, "train_loss": 0.6, "valid_loss": 0.7, "sample_psnr": 8.5, "sample_ssim": 0.02},
        ],
    )
    train_report = generate_training_report(train_run)
    assert train_report["status"] == "ok"
    assert (train_run / "reports" / "figures" / "training_curves.png").exists()

    eval_run = tmp_path / "eval" / "eval_20260416_010000"
    _write_csv(
        eval_run / "per_sample_metrics.csv",
        ["speaker", "session", "start_frame", "end_frame", "psnr_mean", "ssim_mean", "contour_metrics_mode"],
        [
            {"speaker": "1777", "session": "S13", "start_frame": 1, "end_frame": 10, "psnr_mean": 7.8, "ssim_mean": 0.01, "contour_metrics_mode": "video_only"},
            {"speaker": "1777", "session": "S14", "start_frame": 11, "end_frame": 20, "psnr_mean": 8.1, "ssim_mean": 0.02, "contour_metrics_mode": "video_only"},
        ],
    )
    (eval_run / "metrics_summary.json").write_text(
        json.dumps({"psnr_mean": 7.95, "ssim_mean": 0.015}) + "\n",
        encoding="utf-8",
    )
    eval_report = generate_eval_report(eval_run)
    assert eval_report["status"] == "ok"
    assert (eval_run / "reports" / "figures" / "evaluation_metrics.png").exists()

    demo_run = tmp_path / "demo" / "demo_20260416_010000"
    sample_dir = demo_run / "1777_S13_0001_0010"
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "metrics.json").write_text(
        json.dumps({"psnr_mean": 8.2, "ssim_mean": 0.03}) + "\n",
        encoding="utf-8",
    )
    demo_report = generate_demo_report(demo_run)
    assert demo_report["status"] == "ok"
    assert (demo_run / "reports" / "figures" / "demo_metrics.png").exists()

    trace_path = tmp_path / "logs" / "phase_trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"phase": "build_manifest", "event": "start", "ts": "2026-04-16T01:00:00+02:00"}),
                json.dumps({"phase": "build_manifest", "event": "end", "ts": "2026-04-16T01:02:00+02:00"}),
                json.dumps({"phase": "train", "event": "start", "ts": "2026-04-16T01:02:00+02:00"}),
                json.dumps({"phase": "train", "event": "end", "ts": "2026-04-16T01:12:30+02:00"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    phase_report = generate_phase_report(trace_path, tmp_path / "logs" / "report")
    assert phase_report["status"] == "ok"
    assert (tmp_path / "logs" / "report" / "phase_durations.png").exists()
