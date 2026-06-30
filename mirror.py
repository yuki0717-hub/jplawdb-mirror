#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from jplawdb_mirror import Config, MirrorError, build_mirror, discover_mirror


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a verified jplawdb mirror")
    parser.add_argument("--config", default="config.yaml", help="YAML configuration file")
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Discover and validate the source without downloading the full mirror",
    )
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    config = Config.from_file(Path(args.config))
    if args.plan:
        plan = await discover_mirror(config)
        print(json.dumps({"targets": len(plan.targets), "metrics": dict(sorted(plan.metrics.items()))}, ensure_ascii=False, indent=2))
        return 0
    result = await build_mirror(config)
    print(json.dumps({"output": str(result.output_dir), "files": result.file_count, "bytes": result.total_bytes, "metrics": dict(sorted(result.metrics.items()))}, ensure_ascii=False, indent=2))
    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        raise SystemExit(asyncio.run(async_main()))
    except (MirrorError, ValueError) as exc:
        logging.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
