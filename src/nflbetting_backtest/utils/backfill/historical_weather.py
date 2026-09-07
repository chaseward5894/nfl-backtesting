"""Backfill historical weather (default 2022-2024) via Open-Meteo Archive."""

import argparse
import logging
import sys
import time
from pathlib import Path

import nflreadpy as nfl
import pandas as pd
import polars as pl

from nflbetting_backtest.utils.weather import (
    ARCHIVE_URL,  # noqa: F401
    CACHE_COLUMNS,
    PRE_KICKOFF_WINDOW_HOURS,
    REQUEST_DELAY_SEC,
    TEAM_LOCAL_TZ,
    WMO_CODE_TO_MAIN,
    c_to_f,
    fetch_archive_hourly,
    kmh_to_mph,
    mm_to_in,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _add_kickoff_utc(schedule: pd.DataFrame) -> pd.DataFrame:
    sched = schedule.copy()
    naive = pd.to_datetime(
        sched["gameday"].astype(str) + " "
        + sched["gametime"].fillna("13:00").astype(str),
        errors="coerce",
    )
    out = []
    for t, tz in zip(naive, sched["home_team"].map(TEAM_LOCAL_TZ)):
        if pd.isna(t) or pd.isna(tz):
            out.append(pd.NaT)
        else:
            out.append(t.tz_localize(tz).tz_convert("UTC").tz_localize(None))
    sched["kickoff_utc"] = out
    return sched


def _load_stadiums(stadiums_csv: Path) -> pd.DataFrame:
    sd = pd.read_csv(stadiums_csv)
    sd.loc[sd["team"] == "LAR", "team_alias"] = sd.loc[sd["team"] == "LAR", "team"]
    la_row = sd[sd["team"] == "LAR"].copy()
    if len(la_row):
        la_row["team"] = "LA"
        sd = pd.concat([sd, la_row], ignore_index=True)
    sd = sd[sd["team"].isin(list(TEAM_LOCAL_TZ.keys()))].copy()
    sd = sd.sort_values("team").drop_duplicates(subset=["team"], keep="first")
    sd["tz"] = sd["team"].map(TEAM_LOCAL_TZ)
    return sd


def fetch_for_game(game, stadium_row, season: int) -> pd.DataFrame:
    kickoff = game["kickoff_utc"]
    local_date = pd.to_datetime(kickoff).date()
    start = (local_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = (local_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = fetch_archive_hourly(
        float(stadium_row["lat"]), float(stadium_row["long"]),
        start, end, stadium_row["tz"],
    )
    if raw.empty:
        return pd.DataFrame()
    raw["timestamp_local"] = raw["time"].dt.tz_localize(
        stadium_row["tz"], ambiguous="NaT", nonexistent="shift_forward",
    )
    raw = raw.dropna(subset=["timestamp_local"])
    kickoff_local = pd.to_datetime(kickoff).tz_localize(stadium_row["tz"])
    delta_h = (raw["timestamp_local"] - kickoff_local).dt.total_seconds() / 3600
    mask = (delta_h < 0) & (delta_h >= -PRE_KICKOFF_WINDOW_HOURS)
    raw = raw.loc[mask].copy()
    if raw.empty:
        return pd.DataFrame()
    raw["temp_f"] = raw["temp_c"].apply(
        lambda x: c_to_f(x) if pd.notna(x) else None
    )
    raw["wind_mph"] = raw["wind_kmh"].apply(
        lambda x: kmh_to_mph(x) if pd.notna(x) else None
    )
    raw["precip_in"] = raw["precip_mm"].apply(
        lambda x: mm_to_in(x) if pd.notna(x) else None
    )
    raw["weather_main"] = raw["weather_code"].apply(
        lambda c: WMO_CODE_TO_MAIN.get(int(c), "") if pd.notna(c) else None
    )
    away, home = game["away_team"], game["home_team"]
    week = int(game["week"])
    location_id = f"{season}_{week:02d}_{away}_{home}"
    rows = pd.DataFrame({
        "location_id": location_id,
        "timestamp": raw["timestamp_local"].dt.tz_convert("UTC").dt.tz_localize(None),
        "temp": raw["temp_f"].astype("float64"),
        "feels_like": pd.Series([None] * len(raw), dtype="float64"),
        "humidity": pd.Series([None] * len(raw), dtype="float64"),
        "wind_speed": raw["wind_mph"].astype("float64"),
        "wind_gust": pd.Series([None] * len(raw), dtype="float64"),
        "wind_deg": pd.Series([None] * len(raw), dtype="float64"),
        "clouds": pd.Series([None] * len(raw), dtype="float64"),
        "visibility": pd.Series([None] * len(raw), dtype="float64"),
        "pop": pd.Series([None] * len(raw), dtype="float64"),
        "rain_1h": raw["precip_in"].astype("float64"),
        "snow_1h": pd.Series([None] * len(raw), dtype="float64"),
        "weather_main": raw["weather_main"],
        "weather_description": pd.Series([None] * len(raw), dtype="object"),
        "cached_at": pd.Timestamp.now("UTC"),
    })[CACHE_COLUMNS]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--existing-cache", type=Path,
        default=Path(__file__).resolve().parent.parent.parent.parent
        / "data" / "weather" / "weather_cache.parquet",
    )
    parser.add_argument(
        "--stadiums", type=Path,
        default=Path(__file__).resolve().parent.parent.parent.parent
        / "data" / "stadiums.csv",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent.parent.parent.parent
        / "data" / "weather" / "weather_cache_historical.parquet",
    )
    parser.add_argument("--seasons", default="2022-2024")
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--skip-cached", action="store_true")
    args = parser.parse_args()

    def _parse(spec):
        if "-" in spec and "," not in spec:
            a, b = spec.split("-")
            return list(range(int(a), int(b) + 1))
        return [int(x) for x in spec.split(",")]

    seasons = _parse(args.seasons)
    stadiums = _load_stadiums(args.stadiums)
    stadium_by_team = {row["team"]: row for _, row in stadiums.iterrows()}

    sched = nfl.load_schedules(seasons=seasons).to_pandas()
    sched = _add_kickoff_utc(sched)
    sched = sched.dropna(subset=["kickoff_utc"]).reset_index(drop=True)

    existing = pd.DataFrame(columns=CACHE_COLUMNS)
    if args.existing_cache.exists():
        existing = pl.read_parquet(args.existing_cache).to_pandas()
    existing_ids = (
        set(existing["location_id"].unique()) if len(existing) else set()
    )

    if args.max_games > 0:
        sched = sched.head(args.max_games)

    new_rows = []
    n_total = len(sched)
    n_skipped_dome = n_skipped_cached = n_failed = 0
    t_start = time.time()
    for idx, game in sched.iterrows():
        home = game["home_team"]
        stadium = stadium_by_team.get(home)
        if stadium is None:
            n_skipped_dome += 1
            continue
        roof = (stadium.get("roof_type") or "").lower()
        if "dome" in roof or "indoor" in roof or "retractable" in roof:
            n_skipped_dome += 1
            continue
        week = int(game["week"])
        away = game["away_team"]
        season = int(game["season"])
        lid = f"{season}_{week:02d}_{away}_{home}"
        if args.skip_cached and lid in existing_ids:
            n_skipped_cached += 1
            continue
        time.sleep(REQUEST_DELAY_SEC)
        rows = fetch_for_game(game, stadium, season=season)
        if rows.empty:
            n_failed += 1
            continue
        new_rows.append(rows)
        if (idx + 1) % 20 == 0 or idx + 1 == n_total:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed if elapsed else 0
            eta = (n_total - idx - 1) / rate if rate else 0
            logger.info(
                "progress %d/%d (skipped_dome=%d skipped_cached=%d failed=%d "
                "rate=%.1f/s eta=%.0fs)",
                idx + 1, n_total, n_skipped_dome, n_skipped_cached,
                n_failed, rate, eta,
            )

    if not new_rows:
        logger.warning("no new rows fetched")
        return 1
    new_df = pd.concat(new_rows, ignore_index=True)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["location_id", "timestamp"], keep="last",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(combined).write_parquet(args.out)
    logger.info(
        "wrote %d rows (%d new) to %s",
        len(combined), len(new_df), args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())