from __future__ import annotations

import argparse
from pathlib import Path

from .reporting import (
    generate_demo_report,
    generate_eval_report,
    generate_phase_report,
    generate_training_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reports and figures for speech2rtmri_artspeech runs")
    parser.add_argument("--train-run", default=None, help="Path to a train/train_* run directory")
    parser.add_argument("--eval-run", default=None, help="Path to an eval/eval_* run directory")
    parser.add_argument("--demo-run", default=None, help="Path to a demo/demo_* run directory")
    parser.add_argument("--phase-trace", default=None, help="Path to a phase_trace.jsonl file")
    parser.add_argument("--output-dir", default=None, help="Output directory for phase reports")
    args = parser.parse_args()

    if not any((args.train_run, args.eval_run, args.demo_run, args.phase_trace)):
        parser.error("At least one of --train-run, --eval-run, --demo-run, or --phase-trace is required")

    if args.train_run:
        generate_training_report(Path(args.train_run))
    if args.eval_run:
        generate_eval_report(Path(args.eval_run))
    if args.demo_run:
        generate_demo_report(Path(args.demo_run))
    if args.phase_trace:
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.phase_trace).parent
        generate_phase_report(Path(args.phase_trace), output_dir)


if __name__ == "__main__":
    main()
