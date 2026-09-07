"""Collect 2025 NFL injury reports via nflreadpy."""

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
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument(
        "--out", default="data/injuries/history_injuries_2025.parquet",
    )
    args = parser.parse_args()

    raw = fetch_nflreadpy_injuries([args.season])
    keep = ["season", "week", "team", "full_name", "position",
            "report_status"]
    if "date_modified" in raw.columns:
        keep.append("date_modified")
    raw = raw[keep].copy()
    raw = map_positions(raw)
    logger.info("position distribution after mapping:")
    for pos, cnt in raw["position"].value_counts().head(15).items():
        logger.info("  %-5s %d", pos, cnt)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(raw).write_parquet(out_path)
    logger.info("wrote %d rows to %s", len(raw), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())