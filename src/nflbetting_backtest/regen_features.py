"""
nflbt-regen-features: regenerate per-season features from the library.

The backtesting framework ships with bundled feature parquets
under ``data/dataset_full/`` and ``data/dataset_2025/`` that
were computed once and frozen. Those parquets do NOT reflect
the new library knobs (shrinkage, momentum window, per-season
HFA). To exercise the new knobs in the framework's
walk-forward, the user must regenerate the feature parquet
from the library with their config.yaml.

This module provides both:

- :func:`regenerate_features_for_dataset`: the library-side
  helper that calls ``create_model_dataset`` for each season
  in the schedule range and returns a single concatenated
  DataFrame with the same schema as the bundled parquet.
- :func:`main`: the ``nflbt-regen-features`` CLI entry
  point. Writes one parquet per dataset under the user's
  chosen output directory; the user then points
  ``nflbt-run --features-source`` at the regenerated path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from nflbetting.config import AppConfig, load_config
from nflbetting.pipelines.model import create_model_dataset


def _season_from_game_id(df: pl.DataFrame) -> pl.DataFrame:
    """Add a ``season`` int64 column derived from ``game_id``.

    The bundled feature parquet carries ``season`` as a top-
    level column. The library's ``create_model_dataset``
    omits it (the season is encoded in the ``game_id``
    prefix). This helper restores it so the regenerated
    parquet has the schema the framework's
    ``walk_forward_validation`` expects.
    """
    return df.with_columns(
        pl.col("game_id").str.slice(0, 4).cast(pl.Int64).alias("season")
    )


def regenerate_features_for_dataset(
    cfg: AppConfig,
    schedule: pl.DataFrame,
    *,
    seasons: list[int] | None = None,
) -> pl.DataFrame:
    """Regenerate the per-season features frame for one
    dataset's schedule range.

    Args:
        cfg: App config. The library's feature pipeline reads
            ``cfg.start_season``, ``cfg.model.{shrinkage_prior,
            momentum_min/max_window, momentum_dispersion_threshold,
            elo_hfa_mode, elo_hfa_fixed, decay_halflife_weeks}``
            and ``cfg.data_dir``. The walk-forward
            ``training_weeks`` / ``test_weeks`` are NOT used here.
        schedule: The dataset's schedule frame (used to
            determine the season range to regenerate). The
            schedule's ``season`` and ``week`` columns drive
            the union of seasons.
        seasons: Optional explicit list of seasons to
            regenerate. When ``None``, every season in
            ``schedule['season']`` is regenerated, sorted.

    Returns:
        Concatenated features frame with the bundled parquet
        schema (``game_id, team_id, location, week, season,
        R_avg, R_opp_avg, Eff_off, Eff_def, Mmtum_off,
        Mmtum_def, Rest_Travel_Fatigue, Weather_Impact,
        Injury_Impact, margin``). The two-row-per-game shape
        matches the bundled parquet. The library's
        ``create_model_dataset`` returns two extra columns
        (``team_score``, ``opponent_score``) which the bundled
        parquet does not carry; they are dropped here so the
        regenerated parquet has the same schema as the
        bundled artifact.

    Raises:
        ValueError: If the schedule has no seasons.
    """
    if seasons is None:
        seasons = sorted(s for s in schedule["season"].unique().to_list())
    if not seasons:
        raise ValueError("schedule has no seasons to regenerate")

    parts: list[pl.DataFrame] = []
    skipped: list[int] = []
    for season in seasons:
        try:
            season_df = create_model_dataset(cfg, season)
        except Exception as e:
            # Future seasons with no schedule (e.g. 2026 in
            # late 2025) cannot be regenerated; skip them
            # rather than failing the entire regen.
            print(
                f"  skipping season={season}: {e}",
                file=sys.stderr,
            )
            skipped.append(season)
            continue
        if season_df.is_empty():
            continue
        season_df = _season_from_game_id(season_df)
        parts.append(season_df)
    if not parts:
        raise ValueError(
            f"create_model_dataset returned no rows for seasons={seasons} "
            f"(skipped={skipped})"
        )
    out = pl.concat(parts).sort(["season", "week", "game_id", "team_id"])
    # Drop the library-only extras to match the bundled schema.
    drop_cols = [c for c in ("team_score", "opponent_score") if c in out.columns]
    if drop_cols:
        out = out.drop(drop_cols)
    return out


def _write_features(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(path))


def parse_args(argv=None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    parser = argparse.ArgumentParser(
        prog="nflbt-regen-features",
        description=(
            "Regenerate per-season features parquet from the "
            "library for one or both bundled datasets. The "
            "regenerated parquet has the same schema as the "
            "bundled artifact and can be passed to "
            "nflbt-run via --features-source."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent.parent
        / "NFL-Model-UPDATED"
        / "config.yaml",
        help="Path to nflbetting config.yaml (default: "
        "../NFL-Model-UPDATED/config.yaml).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Each dataset gets a "
        "<dataset_name>/features.parquet subdirectory. "
        "Default: <out>/regenerated_features/.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["DATASET_FULL", "DATASET_2025"],
        choices=["DATASET_FULL", "DATASET_2025"],
        help="Which bundled datasets to regenerate (default: " "both).",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=None,
        help="Explicit season list to regenerate. When omitted, "
        "the schedule's season range is used.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.config.exists():
        print(
            f"error: config not found: {args.config}",
            file=sys.stderr,
        )
        return 1
    cfg = load_config(str(args.config))
    # Normalize data_dir to the library's data directory so
    # stadiums.csv and history_*.parquet resolve correctly
    # regardless of cwd.
    raw_data_dir = Path(cfg.data_dir)
    if not raw_data_dir.is_absolute():
        # The library lives at <repo>/NFL-Model-UPDATED/ and the
        # default config.yaml points at ./data. Resolve against
        # the library's repo root (one level above this file's
        # package parent), not the cwd.
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        cfg.data_dir = str(repo_root / "NFL-Model-UPDATED" / raw_data_dir)

    from nflbetting_backtest.datasets import DATASET_FULL, DATASET_2025

    by_name = {"DATASET_FULL": DATASET_FULL, "DATASET_2025": DATASET_2025}
    out_root = args.out_dir or (
        Path(__file__).resolve().parent.parent.parent.parent / "regenerated_features"
    )

    for dataset_name in args.datasets:
        dataset = by_name[dataset_name]
        print(
            f"regenerating {dataset_name} features "
            f"({len(dataset.schedule['season'].unique())} seasons)...",
            file=sys.stderr,
        )
        df = regenerate_features_for_dataset(
            cfg, dataset.schedule, seasons=args.seasons
        )
        out_path = out_root / dataset_name.lower() / "features.parquet"
        _write_features(df, out_path)
        print(
            f"  -> {out_path} ({df.height} rows, {df.width} cols)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
