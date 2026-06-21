from __future__ import annotations

import argparse
from pathlib import Path

from .common import setup_logging
from .config import load_config
from .data.manifest import build_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ArtSpeech DB2 manifests")
    parser.add_argument("--config", required=True, help="Path to the YAML config")
    parser.add_argument("--force", action="store_true", help="Rebuild manifests even if cached manifests already exist")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.force:
        config.setdefault("runtime", {})["reuse_existing_manifests"] = False
    output_root = Path(config["runtime"]["output_dir"])
    logger = setup_logging(output_root / "manifests" / "build_manifest.log")
    summary = build_manifests(config, logger)
    logger.info("Manifest build complete: %s", summary)


if __name__ == "__main__":
    main()
