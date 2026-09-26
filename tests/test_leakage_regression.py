"""
Leakage regression tests (W4 of ``implementation_plan.md``).

The safety net against reintroducing target leakage: a feature must not
encode the game it is predicting. The pre-fix features showed this
directly (post-game Elo at r=0.46, own-game efficiency at r=0.40); the
leakage-free features stay below ``MAX_ABS_CORR``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from nflbetting.config import load_config
from nflbetting.model.rating_model.defs import ModelConfig
from nflbetting.pipelines.features import (
    adjust_cumulative_efficiency,
    calculate_cumulative_efficiency,
)
from nflbetting_backtest import (
    DATASET_FULL,
    walk_forward_validation,
)
from nflbetting_backtest.regen_features import regenerate_features_for_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE_CFG = (
    REPO_ROOT / "NFL-Model-UPDATED" / "track" / "baseline_config.yaml"
)

FEATURE_COLS = [
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

# Legitimate team-strength signal peaks at |r| ~ 0.27; the leaky
# features sat at 0.40-0.46, so this separates the two.
MAX_ABS_CORR = 0.35


def _cfg():
    cfg = load_config(str(BASELINE_CFG))
    cfg.data_dir = str(REPO_ROOT / "NFL-Model-UPDATED" / "data")
    return cfg


def _abs_correlations(df: pl.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in FEATURE_COLS:
        sub = df.select([col, "margin"]).drop_nulls()
        if sub.height < 2:
            continue
        x = sub[col].to_numpy().astype(float)
        y = sub["margin"].to_numpy().astype(float)
        out[col] = abs(float(np.corrcoef(x, y)[0, 1]))
    return out


def _assert_bounded(corr: dict[str, float]) -> None:
    assert corr, "no feature/margin pairs were compared"
    offenders = {k: round(v, 3) for k, v in corr.items() if v > MAX_ABS_CORR}
    assert not offenders, f"features correlate with own margin: {offenders}"


def test_bundled_features_are_leakage_free() -> None:
    _assert_bounded(_abs_correlations(DATASET_FULL.features))


def test_regenerated_features_are_leakage_free() -> None:
    cfg = _cfg()
    features = regenerate_features_for_dataset(
        cfg, DATASET_FULL.schedule, seasons=[2024]
    )
    _assert_bounded(_abs_correlations(features))


def test_efficiency_as_of_is_invariant_to_future_weeks() -> None:
    cfg = _cfg()
    target_week = 5
    efficiencies = calculate_cumulative_efficiency(cfg, 2024, 2024, 19)

    _, adjusted_full = adjust_cumulative_efficiency(
        cfg, 2024, efficiencies, 2024, target_week
    )
    truncated = efficiencies.filter(pl.col("week") < target_week)
    _, adjusted_truncated = adjust_cumulative_efficiency(
        cfg, 2024, truncated, 2024, target_week
    )

    keys = ["week", "game_id", "team", "adj_ypp_off", "adj_epa_def"]
    left = (
        adjusted_full.filter(pl.col("week") < target_week)
        .select(keys)
        .sort(keys)
    )
    right = adjusted_truncated.select(keys).sort(keys)
    assert left.equals(right)


def test_baseline_walk_forward_is_leakage_free_golden() -> None:
    model_cfg = ModelConfig(
        model_type="linear",
        alpha=1.0,
        l1_ratio=0.5,
        scale_features=True,
        test_split=0.2,
        cv_splits=5,
        feature_names=DATASET_FULL.feature_names,
    )
    predictions = walk_forward_validation(
        DATASET_FULL.features,
        DATASET_FULL.schedule,
        model_cfg,
        training_weeks=60,
        test_weeks=1,
    )
    assert predictions, "walk-forward produced no predictions"

    with_result = [p for p in predictions if p.actual_margin is not None]
    su = np.mean(
        [
            (p.predicted_margin > 0) == (p.actual_margin > 0)
            for p in with_result
        ]
    )
    mae = np.mean(
        [abs(p.predicted_margin - p.actual_margin) for p in with_result]
    )

    wins = losses = 0
    for p in predictions:
        if p.spread_line is None or p.actual_margin is None:
            continue
        market = p.spread_line
        edge = p.predicted_margin - market
        if abs(p.actual_margin - market) < 0.25 or abs(edge) < 0.01:
            continue
        correct = (
            p.actual_margin > market if edge > 0 else p.actual_margin < market
        )
        wins += int(correct)
        losses += int(not correct)
    ats = wins / (wins + losses)

    # Leakage-free baseline: near chance ATS, ~10.7 MAE, ~61.6% SU.
    assert 0.48 <= ats <= 0.52, f"ATS outside leakage-free band: {ats:.3f}"
    assert 0.60 <= su <= 0.63, f"SU outside leakage-free band: {su:.3f}"
    assert 10.4 <= mae <= 11.0, f"MAE outside leakage-free band: {mae:.3f}"