"""Shared injury helpers."""

import nflreadpy as nfl
import pandas as pd


# Raw nflreadpy position -> library vocabulary (matches review2026/baseline).
POSITION_MAP = {
    "QB": "QB",
    "RB": "RB", "FB": "RB",
    "WR": "WR", "TE": "TE",
    "OT": "LT", "T": "LT",
    "OG": "LG", "OL": "LG", "G": "LG",
    "C": "C",
    "DT": "DT", "NT": "DT", "DL": "DT",
    "DE": "DE",
    "OLB": "OLB", "ILB": "MLB", "MLB": "MLB", "LB": "OLB",
    "CB": "CB", "DB": "CB",
    "S": "SS", "FS": "FS", "SS": "SS",
    "LS": "C",
}


def map_positions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["position"] = df["position"].map(POSITION_MAP).fillna(df["position"])
    return df


def fetch_nflreadpy_injuries(seasons: list[int]) -> pd.DataFrame:
    """Pull raw injury data from nflreadpy for the given seasons."""
    raw = nfl.load_injuries(seasons=seasons).to_pandas()
    raw["date_modified"] = (
        pd.to_datetime(raw["date_modified"], errors="coerce")
        .dt.strftime("%Y-%m-%d")
    )
    keep = [
        "season", "week", "team", "full_name", "position",
        "report_status", "date_modified",
    ]
    raw = raw[keep].copy()
    raw["season"] = raw["season"].astype("Int64")
    raw["week"] = pd.to_numeric(raw["week"], errors="coerce").astype("Int64")
    return raw