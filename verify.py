#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from jplawdb_mirror import Config, VerificationError, verify_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a generated jplawdb mirror")
    parser.add_argument("--config", default="config.yaml", help="YAML configuration file")
    parser.add_argument("--output", help="Output directory; defaults to config.output_dir")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    try:
        config = Config.from_file(Path(args.config))
        output = Path(args.output) if args.output else config.output_dir
        report = verify_output(output, config)
    except (VerificationError, ValueError) as exc:
        logging.error("%s", exc)
        raise SystemExit(1) from exc
    print(f"OK: {report.file_count} files, {report.total_bytes} bytes, {report.html_links_checked} local HTML links")


if __name__ == "__main__":
    main()
