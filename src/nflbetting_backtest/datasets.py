"""Bundled evaluation datasets."""

from dataclasses import dataclass
from pathlib import Path
import polars as pl

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


@dataclass(frozen=True)
class Dataset:
    """A bundle of features + schedule parquets for one evaluation slice."""

    name: str
    features: pl.DataFrame
    schedule: pl.DataFrame
    features_source: str = "bundled"

    @property
    def feature_names(self) -> list[str]:
        """Model input columns, derived from features.columns."""
        id_cols = {
            "game_id",
            "team_id",
            "location",
            "week",
            "season",
            "team_score",
            "opponent_score",
            "margin",
        }
        return [c for c in self.features.columns if c not in id_cols]

    @property
    def season_range(self) -> tuple[int, int]:
        return (
            int(self.features["season"].min()),
            int(self.features["season"].max()),
        )


def _load_dataset(name: str) -> Dataset:
    # Bundled folders are lowercase (data/dataset_full/, data/dataset_2025/);
    # the public name keeps the original casing (e.g. "DATASET_FULL").
    folder = _DATA_DIR / name.lower()
    features = pl.read_parquet(folder / "features.parquet")
    schedule = pl.read_parquet(folder / "schedule.parquet")
    return Dataset(
        name=name,
        features=features,
        schedule=schedule,
        features_source=f"bundled:{folder / 'features.parquet'}",
    )


def load_dataset_with_features(name: str, features_path: Path) -> Dataset:
    """Build a Dataset whose features parquet is at ``features_path``.

    The schedule is still read from the bundled folder (the
    framework relies on the schedule's spread_line / season /
    week columns). Only the features frame is replaced; this
    is the entry point for the framework's ``--features-source``
    CLI flag.
    """
    folder = _DATA_DIR / name.lower()
    schedule = pl.read_parquet(folder / "schedule.parquet")
    features = pl.read_parquet(str(features_path))
    return Dataset(
        name=name,
        features=features,
        schedule=schedule,
        features_source=f"override:{features_path}",
    )


DATASET_FULL: Dataset = _load_dataset("DATASET_FULL")
DATASET_2025: Dataset = _load_dataset("DATASET_2025")
