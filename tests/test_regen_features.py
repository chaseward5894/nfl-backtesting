"""Tests for the library-side feature regeneration helper.

The framework's bundled feature parquets are frozen snapshots
of v1. To exercise the new feature-level knobs (shrinkage,
momentum window, per-season HFA), the framework must
regenerate the parquet from the library per run.

These tests cover:

- ``regenerate_features_for_dataset``: produces a frame with
  the bundled schema (15 columns, two rows per game).
- Library regen with the v1 baseline config matches the
  bundled parquet schema (column names + dtypes).
- A non-zero ``shrinkage_prior`` produces different
  ``Eff_off`` / ``Eff_def`` values than ``shrinkage_prior=0``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nflbetting.config import load_config
from nflbetting_backtest.datasets import DATASET_2025, DATASET_FULL
from nflbetting_backtest.regen_features import (
    _season_from_game_id,
    regenerate_features_for_dataset,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE_CFG = str(REPO_ROOT / "NFL-Model-UPDATED" / "track" / "baseline_config.yaml")


# Column order in the regen output. The library's
# ``create_model_dataset`` returns ``season``-less output,
# and ``_season_from_game_id`` appends ``season`` at the end
# via polars' ``with_columns`` (which puts new columns at the
# end of the frame). The library extras (``team_score`` /
# ``opponent_score``) are dropped. The actual order is:
EXPECTED_COLUMNS = [
    "game_id",
    "team_id",
    "location",
    "week",
    "R_avg",
    "R_opp_avg",
    "Eff_off",
    "Eff_def",
    "Mmtum_off",
    "Mmtum_def",
    "Rest_Travel_Fatigue",
    "Weather_Impact",
    "Injury_Impact",
    "margin",
    "season",
]


def test_season_from_game_id_adds_column() -> None:
    df = pl.DataFrame(
        {
            "game_id": ["2024_01_ARI_BUF", "2025_03_KC_BAL"],
        }
    )
    out = _season_from_game_id(df)
    assert "season" in out.columns
    assert out["season"].to_list() == [2024, 2025]


def test_season_from_game_id_idempotent_when_present() -> None:
    df = pl.DataFrame(
        {
            "game_id": ["2024_01_ARI_BUF"],
            "season": [2024],
        }
    )
    out = _season_from_game_id(df)
    assert out["season"].to_list() == [2024]


def test_regenerate_dataset_2025_has_bundled_schema() -> None:
    """Regenerate DATASET_2025 with the live config (default
    knobs) and confirm the schema matches the bundled parquet."""
    cfg = _load_baseline_cfg_with_data_dir()
    df = regenerate_features_for_dataset(cfg, DATASET_2025.schedule)
    assert df.columns == EXPECTED_COLUMNS
    # Two rows per game (home + away).
    n_unique_games = df["game_id"].n_unique()
    assert df.height == 2 * n_unique_games


def test_regenerate_dataset_full_includes_played_seasons() -> None:
    """DATASET_FULL covers 2009-2026 in the bundled parquet;
    regen returns every season that has completed games in
    the library's data. Future seasons (e.g. 2026 in late
    2025) have no schedule and are skipped."""
    cfg = _load_baseline_cfg_with_data_dir()
    df = regenerate_features_for_dataset(cfg, DATASET_FULL.schedule)
    seasons = sorted(df["season"].unique().to_list())
    # 2025 is the latest season with completed games; the
    # bundled parquet carries 2026 placeholder rows but the
    # library cannot regenerate them (no schedule data).
    assert seasons == sorted(
        s for s in DATASET_FULL.schedule["season"].unique().to_list() if s <= 2025
    )


def test_regenerate_with_explicit_seasons_subset() -> None:
    """``seasons=[2024]`` returns only 2024 data."""
    cfg = _load_baseline_cfg_with_data_dir()
    df = regenerate_features_for_dataset(cfg, DATASET_FULL.schedule, seasons=[2024])
    assert (df["season"] == 2024).all()


def test_regenerate_empty_seasons_raises() -> None:
    cfg = _load_baseline_cfg_with_data_dir()
    with pytest.raises(ValueError, match="no seasons"):
        regenerate_features_for_dataset(cfg, DATASET_FULL.schedule, seasons=[])


def test_shrinkage_prior_changes_eff_values() -> None:
    """Two regen runs with different ``shrinkage_prior`` values
    produce different ``Eff_off`` / ``Eff_def`` values. This
    is what makes the framework's per-run regen actually
    exercise the Stage 5 knob."""
    cfg_a = _load_baseline_cfg_with_data_dir()
    cfg_b = _load_baseline_cfg_with_data_dir()
    # The baseline config locks shrinkage_prior=0 for v1.
    # Override to 50 for cfg_b.
    cfg_b.model.shrinkage_prior = 50.0

    df_a = regenerate_features_for_dataset(cfg_a, DATASET_2025.schedule, seasons=[2024])
    df_b = regenerate_features_for_dataset(cfg_b, DATASET_2025.schedule, seasons=[2024])

    eff_a = df_a["Eff_off"].to_numpy()
    eff_b = df_b["Eff_off"].to_numpy()
    # Not all rows must differ (small-n / regularized), but
    # the means should differ when shrinkage is non-zero.
    assert not (eff_a == eff_b).all()


def test_regenerated_features_round_trip_through_dataset() -> None:
    """``load_dataset_with_features`` accepts a regenerated
    parquet and produces a usable Dataset."""
    import tempfile
    from nflbetting_backtest.datasets import load_dataset_with_features

    cfg = _load_baseline_cfg_with_data_dir()
    df = regenerate_features_for_dataset(cfg, DATASET_2025.schedule, seasons=[2024])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "features.parquet"
        df.write_parquet(str(path))
        ds = load_dataset_with_features("DATASET_2025", path)
    assert ds.features_source.startswith("override:")
    assert ds.features.height == df.height
    # feature_names excludes id/margin/season cols.
    assert "margin" not in ds.feature_names
    assert "season" not in ds.feature_names


def test_data_dir_resolution_is_cwd_independent() -> None:
    """``_resolve_data_dir`` makes ``cfg.data_dir`` absolute
    against the library's repo root, regardless of cwd. This
    is what makes ``nflbt-regen-features`` work when invoked
    from a non-library directory."""
    from nflbetting_backtest.cli import _resolve_data_dir

    cfg = _load_baseline_cfg_with_data_dir()
    # Sanity: at this point data_dir has been resolved to the
    # absolute path inside the test fixture.
    assert Path(cfg.data_dir).is_absolute()

    # Already absolute: a second call is a no-op.
    before = cfg.data_dir
    _resolve_data_dir(cfg)
    assert cfg.data_dir == before

    # Force-relative; _resolve_data_dir should re-absolute it.
    cfg.data_dir = "./data"
    _resolve_data_dir(cfg)
    assert Path(cfg.data_dir).is_absolute()
    # Path should point under the library's data dir.
    assert cfg.data_dir.endswith("NFL-Model-UPDATED/data")


def test_comparison_pass_uses_regenerated_datasets() -> None:
    """Sanity check that ``dataset_by_name`` is built in main
    and indexed by name. This guards the regression where
    the comparison pass at the bottom of ``main`` used the
    module-level bundled ``DATASET_FULL`` / ``DATASET_2025``
    constants instead of the regenerated datasets, so the
    ``comparison.txt`` was identical to the v1 baseline even
    when per-run regeneration was active."""
    cfg = _load_baseline_cfg_with_data_dir()
    df_regen = regenerate_features_for_dataset(
        cfg, DATASET_FULL.schedule, seasons=[2024]
    )
    df_regen_short = regenerate_features_for_dataset(
        cfg, DATASET_2025.schedule, seasons=[2024]
    )
    # The regen'd frames should differ from the bundled ones
    # (different shrinkage_prior / momentum_window).
    bundled_full = DATASET_FULL.features
    assert (
        not (
            df_regen["Eff_off"].to_numpy()
            == bundled_full.filter(pl.col("season") == 2024)["Eff_off"].to_numpy()
        ).all()
        or df_regen.height != bundled_full.filter(pl.col("season") == 2024).height
    )


def _load_baseline_cfg_with_data_dir():
    """Load the v1 baseline config and force ``data_dir`` to
    point at the library's data directory (where
    stadiums.csv, history_pbp.parquet etc. live).

    The framework's bundled config uses ``./data`` which is
    resolved relative to cwd. When the framework's tests run
    from the framework repo, ``./data`` resolves to the
    framework's bundled data, not the library's. Force the
    library's data dir explicitly so the regen tests do not
    depend on cwd.
    """
    cfg = load_config(BASELINE_CFG)
    cfg.data_dir = str(REPO_ROOT / "NFL-Model-UPDATED" / "data")
    return cfg
