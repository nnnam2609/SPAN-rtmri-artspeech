from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .common import ensure_dir, read_json, read_jsonl, write_json

_WHITE = (255, 255, 255)
_BLACK = (24, 24, 24)
_GRAY = (140, 140, 140)
_GRID = (228, 228, 228)
_BLUE = (52, 101, 164)
_RED = (204, 70, 70)
_GREEN = (70, 146, 96)
_ORANGE = (217, 125, 39)


def _coerce_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _series_bounds(series_values: list[list[float | None]]) -> tuple[float, float]:
    values = [value for series in series_values for value in series if value is not None]
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if high <= low:
        high = low + 1.0
    padding = (high - low) * 0.1
    return low - padding, high + padding


def _draw_line_panels(
    panels: list[dict[str, Any]],
    output_path: Path,
    size: tuple[int, int] = (1440, 960),
) -> None:
    if not panels:
        return

    image = Image.new("RGB", size, _WHITE)
    draw = ImageDraw.Draw(image)
    width, height = size
    top_margin = 56
    bottom_margin = 40
    side_margin = 70
    panel_gap = 36
    panel_height = max((height - top_margin - bottom_margin - panel_gap * (len(panels) - 1)) // len(panels), 220)

    draw.text((side_margin, 18), "speech2rtmri_artspeech report", fill=_BLACK)

    for panel_index, panel in enumerate(panels):
        panel_top = top_margin + panel_index * (panel_height + panel_gap)
        plot_left = side_margin
        plot_top = panel_top + 26
        plot_right = width - side_margin
        plot_bottom = panel_top + panel_height
        draw.text((plot_left, panel_top), panel["title"], fill=_BLACK)
        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=_GRAY, width=1)

        min_y, max_y = _series_bounds([series["values"] for series in panel["series"]])
        for tick_index in range(5):
            y = plot_top + (plot_bottom - plot_top) * tick_index / 4.0
            value = max_y - (max_y - min_y) * tick_index / 4.0
            draw.line((plot_left, y, plot_right, y), fill=_GRID, width=1)
            draw.text((12, y - 8), f"{value:.3f}", fill=_GRAY)

        max_len = max((len(series["values"]) for series in panel["series"]), default=0)
        if max_len <= 1:
            max_len = 2
        for series_index, series in enumerate(panel["series"]):
            last_point = None
            color = series["color"]
            values = series["values"]
            for value_index, value in enumerate(values):
                if value is None:
                    last_point = None
                    continue
                x = plot_left + (plot_right - plot_left) * value_index / (max_len - 1)
                y_ratio = (value - min_y) / max(max_y - min_y, 1e-8)
                y = plot_bottom - y_ratio * (plot_bottom - plot_top)
                if last_point is not None:
                    draw.line((last_point[0], last_point[1], x, y), fill=color, width=3)
                r = 3
                draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=color)
                last_point = (x, y)
            legend_x = plot_right - 220
            legend_y = plot_top + 8 + series_index * 18
            draw.rectangle((legend_x, legend_y + 4, legend_x + 10, legend_y + 14), fill=color)
            draw.text((legend_x + 16, legend_y), series["label"], fill=_BLACK)

    ensure_dir(output_path.parent)
    image.save(output_path)


def _draw_bar_chart(title: str, rows: list[dict[str, Any]], output_path: Path, value_key: str, label_key: str) -> None:
    if not rows:
        return

    width, height = 1200, 720
    image = Image.new("RGB", (width, height), _WHITE)
    draw = ImageDraw.Draw(image)

    draw.text((60, 24), title, fill=_BLACK)
    plot_left = 220
    plot_top = 80
    plot_right = width - 60
    plot_bottom = height - 60
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=_GRAY, width=1)

    max_value = max(float(row[value_key]) for row in rows)
    for row_index, row in enumerate(rows):
        top = plot_top + row_index * max((plot_bottom - plot_top) // max(len(rows), 1), 22)
        bottom = min(top + 16, plot_bottom)
        ratio = float(row[value_key]) / max(max_value, 1e-8)
        right = plot_left + ratio * (plot_right - plot_left)
        draw.rectangle((plot_left, top, right, bottom), fill=_BLUE)
        draw.text((20, top - 2), str(row[label_key]), fill=_BLACK)
        draw.text((right + 8, top - 2), f"{float(row[value_key]):.2f}s", fill=_BLACK)

    ensure_dir(output_path.parent)
    image.save(output_path)


def generate_training_report(run_dir: str | Path, logger=None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    history_path = run_dir / "history.csv"
    rows = _read_csv_rows(history_path)
    if not rows:
        summary = {"status": "missing_history", "history_path": str(history_path)}
        write_json(run_dir / "reports" / "training_summary.json", summary)
        return summary

    steps = [int(row["step"]) for row in rows]
    train_loss = [_coerce_float(row["train_loss"]) for row in rows]
    valid_loss = [_coerce_float(row["valid_loss"]) for row in rows]
    sample_psnr = [_coerce_float(row["sample_psnr"]) for row in rows]
    sample_ssim = [_coerce_float(row["sample_ssim"]) for row in rows]

    figures_dir = ensure_dir(run_dir / "reports" / "figures")
    _draw_line_panels(
        [
            {
                "title": "Train vs validation loss",
                "series": [
                    {"label": "train_loss", "values": train_loss, "color": _BLUE},
                    {"label": "valid_loss", "values": valid_loss, "color": _RED},
                ],
            },
            {
                "title": "Sample video metrics during training",
                "series": [
                    {"label": "sample_psnr", "values": sample_psnr, "color": _GREEN},
                    {"label": "sample_ssim", "values": sample_ssim, "color": _ORANGE},
                ],
            },
        ],
        figures_dir / "training_curves.png",
    )

    summary = {
        "status": "ok",
        "num_steps": len(steps),
        "last_step": steps[-1],
        "last_train_loss": next((value for value in reversed(train_loss) if value is not None), None),
        "best_valid_loss": min((value for value in valid_loss if value is not None), default=None),
        "best_sample_psnr": max((value for value in sample_psnr if value is not None), default=None),
        "best_sample_ssim": max((value for value in sample_ssim if value is not None), default=None),
        "history_path": str(history_path),
        "figure_path": str(figures_dir / "training_curves.png"),
        "best_visual_checkpoint_path": str(run_dir / "checkpoints" / "best_visual.pt"),
        "best_valid_checkpoint_path": str(run_dir / "checkpoints" / "best_valid_loss.pt"),
        "best_checkpoint_path": str(run_dir / "checkpoints" / "best.pt"),
    }
    write_json(run_dir / "reports" / "training_summary.json", summary)
    if logger is not None:
        logger.info("Saved training report under %s", run_dir / "reports")
    return summary


def generate_eval_report(
    eval_root: str | Path,
    per_sample_rows: list[dict[str, Any]] | None = None,
    summary: dict[str, Any] | None = None,
    logger=None,
) -> dict[str, Any]:
    eval_root = Path(eval_root)
    if per_sample_rows is None:
        per_sample_rows = _read_csv_rows(eval_root / "per_sample_metrics.csv")
    if summary is None:
        summary_path = eval_root / "metrics_summary.json"
        summary = read_json(summary_path) if summary_path.exists() else {}

    figures_dir = ensure_dir(eval_root / "reports" / "figures")
    psnr_values = [_coerce_float(row.get("psnr_mean")) for row in per_sample_rows]
    ssim_values = [_coerce_float(row.get("ssim_mean")) for row in per_sample_rows]
    labels = [
        f"{row.get('speaker', '?')}_{row.get('session', '?')}_{row.get('start_frame', '?')}"
        for row in per_sample_rows
    ]
    _draw_line_panels(
        [
            {
                "title": "Per-sample PSNR on evaluation split",
                "series": [{"label": "psnr_mean", "values": psnr_values, "color": _GREEN}],
            },
            {
                "title": "Per-sample SSIM on evaluation split",
                "series": [{"label": "ssim_mean", "values": ssim_values, "color": _ORANGE}],
            },
        ],
        figures_dir / "evaluation_metrics.png",
        size=(1440, 900),
    )
    if labels:
        top_psnr = sorted(
            (
                {"label": label, "value": value}
                for label, value in zip(labels, psnr_values)
                if value is not None
            ),
            key=lambda item: float(item["value"]),
            reverse=True,
        )[:10]
        _draw_bar_chart("Top PSNR evaluation samples", top_psnr, figures_dir / "evaluation_top_psnr.png", "value", "label")

    report = {
        "status": "ok" if per_sample_rows else "empty",
        "num_rows": len(per_sample_rows),
        "psnr_mean": summary.get("psnr_mean"),
        "ssim_mean": summary.get("ssim_mean"),
        "figure_path": str(figures_dir / "evaluation_metrics.png"),
    }
    write_json(eval_root / "reports" / "evaluation_report.json", report)
    if logger is not None:
        logger.info("Saved evaluation report under %s", eval_root / "reports")
    return report


def generate_demo_report(
    demo_root: str | Path,
    sample_rows: list[dict[str, Any]] | None = None,
    logger=None,
) -> dict[str, Any]:
    demo_root = Path(demo_root)
    if sample_rows is None:
        sample_rows = []
        for metrics_path in sorted(demo_root.glob("*/metrics.json")):
            metrics = read_json(metrics_path)
            sample_rows.append(
                {
                    "sample_id": metrics_path.parent.name,
                    "psnr_mean": metrics.get("psnr_mean"),
                    "ssim_mean": metrics.get("ssim_mean"),
                }
            )

    figures_dir = ensure_dir(demo_root / "reports" / "figures")
    _draw_line_panels(
        [
            {
                "title": "Demo sample PSNR",
                "series": [{"label": "psnr_mean", "values": [_coerce_float(row.get("psnr_mean")) for row in sample_rows], "color": _GREEN}],
            },
            {
                "title": "Demo sample SSIM",
                "series": [{"label": "ssim_mean", "values": [_coerce_float(row.get("ssim_mean")) for row in sample_rows], "color": _ORANGE}],
            },
        ],
        figures_dir / "demo_metrics.png",
        size=(1440, 900),
    )

    report = {
        "status": "ok" if sample_rows else "empty",
        "num_samples": len(sample_rows),
        "psnr_mean": _mean([_coerce_float(row.get("psnr_mean")) for row in sample_rows]),
        "ssim_mean": _mean([_coerce_float(row.get("ssim_mean")) for row in sample_rows]),
        "figure_path": str(figures_dir / "demo_metrics.png"),
    }
    write_json(demo_root / "reports" / "demo_report.json", report)
    if logger is not None:
        logger.info("Saved demo report under %s", demo_root / "reports")
    return report


def _mean(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return float(sum(usable) / len(usable))


def generate_phase_report(phase_trace_path: str | Path, output_dir: str | Path, logger=None) -> dict[str, Any]:
    phase_trace_path = Path(phase_trace_path)
    output_dir = ensure_dir(output_dir)
    if not phase_trace_path.exists():
        summary = {"status": "missing_trace", "phase_trace_path": str(phase_trace_path)}
        write_json(output_dir / "phase_summary.json", summary)
        return summary

    events = read_jsonl(phase_trace_path)
    starts: dict[str, datetime] = {}
    rows: list[dict[str, Any]] = []
    for event in events:
        phase = str(event["phase"])
        stamp = datetime.fromisoformat(str(event["ts"]))
        event_type = str(event["event"])
        if event_type == "start":
            starts[phase] = stamp
            continue
        started_at = starts.get(phase)
        if started_at is None:
            continue
        duration_sec = (stamp - started_at).total_seconds()
        rows.append(
            {
                "phase": phase,
                "seconds": round(duration_sec, 3),
                "status": event_type,
                "started_at": started_at.isoformat(),
                "finished_at": stamp.isoformat(),
            }
        )

    if rows:
        _draw_bar_chart("Pipeline phase durations", rows, output_dir / "phase_durations.png", "seconds", "phase")

    summary = {
        "status": "ok" if rows else "empty",
        "num_phases": len(rows),
        "phase_trace_path": str(phase_trace_path),
        "phases": rows,
        "figure_path": str(output_dir / "phase_durations.png") if rows else None,
    }
    write_json(output_dir / "phase_summary.json", summary)
    if logger is not None:
        logger.info("Saved phase report under %s", output_dir)
    return summary
