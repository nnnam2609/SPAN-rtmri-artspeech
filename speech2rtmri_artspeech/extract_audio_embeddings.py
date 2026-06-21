from __future__ import annotations

import argparse
from pathlib import Path

from .common import setup_logging
from .config import load_config
from .data.audio import extract_audio_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract session-level audio embeddings")
    parser.add_argument("--config", required=True, help="Path to the YAML config")
    parser.add_argument("--force", action="store_true", help="Recompute embeddings even if cache exists")
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = Path(config["runtime"]["output_dir"])
    logger = setup_logging(output_root / "audio_cache" / "extract_audio_embeddings.log")
    saved_paths = extract_audio_embeddings(config, logger, force=args.force)
    logger.info("Saved %d audio cache file(s)", len(saved_paths))


if __name__ == "__main__":
    main()
