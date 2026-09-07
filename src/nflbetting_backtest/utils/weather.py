"""Shared weather (Open-Meteo) helpers."""

import time
from typing import Optional
import numpy as np
import pandas as pd
import requests


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = "temperature_2m,wind_speed_10m,precipitation,weather_code"
PRE_KICKOFF_WINDOW_HOURS = 6
REQUEST_DELAY_SEC = 0.4
MAX_RETRIES = 3

CACHE_COLUMNS = [
    "location_id", "timestamp", "temp", "feels_like", "humidity",
    "wind_speed", "wind_gust", "wind_deg", "clouds", "visibility",
    "pop", "rain_1h", "snow_1h", "weather_main", "weather_description",
    "cached_at",
]

TEAM_LOCAL_TZ = {
    "ARI": "America/Phoenix", "ATL": "America/New_York",
    "BAL": "America/New_York", "BUF": "America/New_York",
    "CAR": "America/New_York", "CHI": "America/Chicago",
    "CIN": "America/New_York", "CLE": "America/New_York",
    "DAL": "America/Chicago", "DEN": "America/Denver",
    "DET": "America/New_York", "GB": "America/Chicago",
    "HOU": "America/Chicago", "IND": "America/Indiana/Indianapolis",
    "JAX": "America/New_York", "KC": "America/Chicago",
    "LA": "America/Los_Angeles", "LAC": "America/Los_Angeles",
    "LV": "America/Los_Angeles", "MIA": "America/New_York",
    "MIN": "America/Chicago", "NE": "America/New_York",
    "NO": "America/Chicago", "NYG": "America/New_York",
    "NYJ": "America/New_York", "PHI": "America/New_York",
    "PIT": "America/New_York", "SEA": "America/Los_Angeles",
    "SF": "America/Los_Angeles", "TB": "America/New_York",
    "TEN": "America/Chicago", "WAS": "America/New_York",
}

WMO_CODE_TO_MAIN = {
    0: "Clear", 1: "Mostly Clear", 2: "Partly Cloudy", 3: "Overcast",
    45: "Fog", 48: "Fog",
    51: "Drizzle", 53: "Drizzle", 55: "Drizzle",
    61: "Rain", 63: "Rain", 65: "Heavy Rain",
    71: "Snow", 73: "Snow", 75: "Heavy Snow", 77: "Snow",
    80: "Rain Showers", 81: "Rain Showers", 82: "Heavy Showers",
    85: "Snow Showers", 86: "Snow Showers",
    95: "Thunderstorm", 96: "Thunderstorm", 99: "Thunderstorm",
}


def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def kmh_to_mph(kmh: float) -> float:
    return kmh * 0.621371


def mm_to_in(mm: float) -> float:
    return mm / 25.4


def fetch_archive_hourly(
    lat: float, lon: float, start_date: str, end_date: str, tz: str
) -> pd.DataFrame:
    """Pull hourly observations for a 1-2 day window. Empty df on failure."""
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": HOURLY_VARS, "timezone": tz,
    }
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(ARCHIVE_URL, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    else:
        return pd.DataFrame()
    h = data.get("hourly") or {}
    times = h.get("time") or []
    if not times:
        return pd.DataFrame()
    return pd.DataFrame({
        "time": pd.to_datetime(times),
        "temp_c": h.get("temperature_2m", [None] * len(times)),
        "wind_kmh": h.get("wind_speed_10m", [None] * len(times)),
        "precip_mm": h.get("precipitation", [None] * len(times)),
        "weather_code": h.get("weather_code", [None] * len(times)),
    })