"""Regeneration must cover each dataset's full feature history.

Bug: ``_resolve_dataset`` regenerated from the dataset's *schedule*
seasons, so DATASET_2025 (2025-only schedule) got a 2025-only feature
parquet and had no training history.
"""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from nflbetting_backtest import cli
from nflbetting_backtest.datasets import DATASET_2025

_FEATURE_COLS = [
    "R_avg",
    "R_opp_avg",
    "Eff_off",
    "Eff_def",
    "Mmtum_off",
    "Mmtum_def",
    "Rest_Travel_Fatigue",
    "Weather_Impact",
    "Injury_Impact",
]


def test_resolve_dataset_regenerates_full_feature_history(
    monkeypatch, tmp_path
) -> None:
    captured: dict = {}

    def fake_regen(cfg, schedule, *, seasons=None):
        captured["seasons"] = seasons
        return pl.DataFrame(
            {
                "game_id": ["2025_01_A_B"],
                "team_id": ["A"],
                "location": ["home"],
                "week": [1],
                **{c: [0.0] for c in _FEATURE_COLS},
                "margin": [0.0],
                "season": [2025],
            }
        )

    monkeypatch.setattr(cli, "regenerate_features_for_dataset", fake_regen)

    cli._resolve_dataset(
        "DATASET_2025",
        cfg=SimpleNamespace(),
        run_dir=tmp_path,
        use_regenerated=True,
    )

    expected = sorted(DATASET_2025.features["season"].unique().to_list())
    assert captured["seasons"] == expected
    assert 2009 in captured["seasons"]  # full history, not just 2025