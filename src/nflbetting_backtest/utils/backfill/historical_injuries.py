"""Rebuild data/history_injuries.parquet from nflreadpy for recent seasons."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import polars as pl

from nflbetting_backtest.utils.injuries import (
    fetch_nflreadpy_injuries,
    map_positions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons", default="2022-2024",
        help="range or comma-list (default 2022-2024)",
    )
    parser.add_argument(
        "--out", default="data/history_injuries.parquet",
    )
    parser.add_argument(
        "--merge-existing", action="store_true",
        help="also pull rows for older seasons from the on-disk file",
    )
    args = parser.parse_args()

    def _parse(spec):
        if "-" in spec and "," not in spec:
            a, b = spec.split("-")
            return list(range(int(a), int(b) + 1))
        return [int(x) for x in spec.split(",")]

    seasons = _parse(args.seasons)
    logger.info("processing seasons: %s", seasons)

    df = fetch_nflreadpy_injuries(seasons)
    logger.info("nflreadpy returned %d rows", len(df))

    if args.merge_existing and Path(args.out).exists():
        existing = pd.read_parquet(args.out)
        existing = existing[~existing["season"].isin(seasons)]
        existing["week"] = pd.NA
        existing["season"] = existing["season"].astype("Int64")
        df = pd.concat([df, existing], ignore_index=True)
        logger.info("after merge: %d rows total", len(df))

    df = map_positions(df)
    n_changed = (df["position"] != df["position"]).sum()
    logger.info("position remapping: %d / %d rows changed", n_changed, len(df))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(df).write_parquet(out_path)
    logger.info("wrote %d rows to %s", len(df), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())