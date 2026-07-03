"""Command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .runner import classify_all


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async vLLM-Omni audio classification pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Print first compiled request and exit")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N manifest rows")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    return asyncio.run(classify_all(config_path, dry_run=args.dry_run, limit=args.limit))


if __name__ == "__main__":
    raise SystemExit(main())
