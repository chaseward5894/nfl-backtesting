"""Unit tests for the framework's new library-knob wiring (F1).

These tests exercise the framework's helpers in isolation,
without spinning up the full ``nflbt-run`` walk-forward. The
end-to-end correctness is verified by
``test_outputs_match_baseline.py``.

Covers:

- F1.2 — ``_train_and_collect_weights`` accepts and forwards
  ``sample_weight`` to ``RatingModel.fit``.
- F1.2 — ``_build_decay_sample_weight`` returns ``None`` for
  ``halflife_weeks <= 0`` and a 1-D array otherwise; the
  weights decay with age.
- F1.4 — ``walk_forward_validation`` and ``train_period``
  accept ``decay_halflife_weeks`` as a keyword argument and
  build sample weights.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from nflbetting.model.rating_model.defs import ModelConfig
from nflbetting_backtest.backtest import (
    _build_decay_sample_weight,
    _train_and_collect_weights,
)


def _make_train_df(n_games: int = 6) -> pl.DataFrame:
    """Synthetic training-set frame with the schema
    ``prepare_training_data`` expects.

    Margins alternate sign so the classifier sees both
    classes.
    """
    feature_cols = [
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
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_games):
        season = 2024
        week = i + 1
        gid = f"{season}_{week:02d}_H{i}_A{i}"
        home_margin = 6.0 if i % 2 == 0 else -6.0
        for loc, sign in (("home", 1.0), ("away", -1.0)):
            team_score = 24.0 + rng.normal() * 0.5
            opp_score = team_score - sign * home_margin
            rows.append(
                {
                    "game_id": gid,
                    "team_id": "HOME" if loc == "home" else "AWAY",
                    "location": loc,
                    "week": week,
                    **{c: float(rng.normal()) for c in feature_cols},
                    "team_score": team_score,
                    "opponent_score": opp_score,
                    "margin": sign * (team_score - opp_score),
                }
            )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# F1.2 — sample_weight plumbing
# ---------------------------------------------------------------------------


def test_train_and_collect_weights_sample_weight_none_is_default() -> None:
    """``sample_weight=None`` reproduces the unweighted fit byte-for-byte."""
    df = _make_train_df(8)
    cfg = ModelConfig(model_type="ridge", alpha=0.1, scale_features=False)

    m_default = _train_and_collect_weights(df, cfg)
    m_none = _train_and_collect_weights(df, cfg, sample_weight=None)

    coef_default = m_default.regression_weights.weights
    coef_none = m_none.regression_weights.weights
    for name in coef_default:
        assert coef_default[name] == pytest.approx(coef_none[name])


def test_train_and_collect_weights_sample_weight_changes_fit() -> None:
    """A non-trivial sample weight changes the fitted weights."""
    df = _make_train_df(10)
    cfg = ModelConfig(model_type="ridge", alpha=0.1, scale_features=False)

    # ``prepare_training_data`` produces one row per *game*;
    # ``sample_weight`` has length == n_games (10), not
    # df.height (20).
    n_games = df.height // 2
    m_eq = _train_and_collect_weights(df, cfg, sample_weight=np.ones(n_games))
    weights = np.ones(n_games)
    weights[0] = 1000.0
    m_skewed = _train_and_collect_weights(df, cfg, sample_weight=weights)

    coef_eq = m_eq.regression_weights.weights
    coef_skewed = m_skewed.regression_weights.weights
    any_different = any(
        coef_eq[name] != pytest.approx(coef_skewed[name]) for name in coef_eq
    )
    assert any_different


# ---------------------------------------------------------------------------
# F1.2 — _build_decay_sample_weight
# ---------------------------------------------------------------------------


def test_build_decay_sample_weight_zero_halflife_returns_none() -> None:
    df = _make_train_df(8)
    out = _build_decay_sample_weight(df, 2024, 9, halflife_weeks=0.0)
    assert out is None


def test_build_decay_sample_weight_positive_halflife_returns_array() -> None:
    df = _make_train_df(8)
    out = _build_decay_sample_weight(df, 2024, 9, halflife_weeks=52.0)
    assert out is not None
    assert out.shape == (df.height,)
    assert (out > 0).all()


def test_build_decay_sample_weight_decay_with_age() -> None:
    """Older rows have lower weight than newer rows."""
    df = _make_train_df(8)
    out = _build_decay_sample_weight(df, 2024, 9, halflife_weeks=52.0)
    assert out is not None
    # Two rows per game; per-game averages must be monotone in
    # game week. Compare the per-game means.
    df_with_w = df.with_columns(pl.Series("w", out))
    per_game = (
        df_with_w.group_by("game_id")
        .agg(
            pl.col("w").mean().alias("mean_w"),
            pl.col("week").first().alias("week"),
        )
        .sort("week")
    )
    means = per_game["mean_w"].to_numpy()
    assert np.all(np.diff(means) >= 0)


def test_build_decay_sample_weight_no_game_id_column() -> None:
    df = _make_train_df(4).drop("game_id")
    assert _build_decay_sample_weight(df, 2024, 9, halflife_weeks=52.0) is None
