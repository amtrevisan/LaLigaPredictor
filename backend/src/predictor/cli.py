"""Command-line entry point: ``predictor run``."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from predictor.config import DEFAULT_EXPORT_DIR, LEAGUES, LEAGUES_BY_ID
from predictor.pipeline import RunConfig, default_now, run


def _parse_now(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="predictor", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="refresh data, evaluate, predict and export JSON")
    run_cmd.add_argument("--out", type=Path, default=DEFAULT_EXPORT_DIR, help="export directory")
    run_cmd.add_argument(
        "--now", type=_parse_now, default=None, help="pretend the run happens at this UTC time"
    )
    run_cmd.add_argument(
        "--no-media", action="store_true", help="skip TheSportsDB lookups (use cached media only)"
    )
    run_cmd.add_argument(
        "--league", action="append", choices=sorted(LEAGUES_BY_ID), help="restrict to a league"
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    leagues = tuple(LEAGUES_BY_ID[i] for i in args.league) if args.league else LEAGUES
    result = run(
        RunConfig(
            now=args.now or default_now(),
            export_dir=args.out,
            fetch_media=not args.no_media,
            leagues=leagues,
        )
    )
    print(json.dumps({"changed": result["changed"], "blendWeight": result["blendWeight"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
