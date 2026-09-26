"""Tests for the same-game paired significance comparison (W19)."""

from __future__ import annotations

import pytest

from nflbetting_backtest import paired_significance
from nflbetting_backtest.metrics import Prediction
from nflbetting_backtest.report import write_comparison_report
from nflbetting_backtest.metrics import aggregate_metrics


def _pred(game_id, predicted_margin, actual_margin, spread=0.0):
    return Prediction(
        game_id=game_id,
        season=2025,
        week=1,
        home_team="H",
        away_team="A",
        predicted_margin=predicted_margin,
        predicted_win_probability=0.5,
        actual_margin=actual_margin,
        spread_line=spread,
    )


# spread = 0; ATS correctness == sign(predicted) == sign(actual)
FULL = [
    _pred("g1", 3.0, 3.0),   # correct
    _pred("g2", -3.0, 3.0),  # wrong
    _pred("g3", -3.0, 3.0),  # wrong
    _pred("g4", -3.0, 3.0),  # wrong
]
SHORT = [
    _pred("g1", 3.0, 3.0),   # correct
    _pred("g2", 3.0, 3.0),   # correct
    _pred("g3", 3.0, 3.0),   # correct
    _pred("g4", 3.0, 3.0),   # correct
]


def test_paired_significance_counts_and_tests() -> None:
    sig = paired_significance(FULL, SHORT)

    ats = sig["ats"]
    assert ats["n"] == 4
    assert ats["full_wins"] == 1
    assert ats["short_wins"] == 4
    assert ats["b01_full_wrong_short_right"] == 3
    assert ats["b10_full_right_short_wrong"] == 0
    assert ats["mcnemar_p"] == pytest.approx(0.25)


def test_paired_error_differences() -> None:
    sig = paired_significance(FULL, SHORT)

    # Full errors: 0, -6, -6, -6. Short errors: all 0.
    assert sig["mae"]["diff"] == pytest.approx(-4.5)
    assert sig["mae"]["n"] == 4
    assert sig["rmse"]["diff"] == pytest.approx(-27 ** 0.5)


def test_paired_uses_only_common_games() -> None:
    full = FULL + [_pred("g5", 3.0, 3.0)]
    short = SHORT

    sig = paired_significance(full, short)

    assert sig["ats"]["n"] == 4
    assert sig["common_games"] == 4


def test_comparison_report_includes_paired_section(tmp_path) -> None:
    path = tmp_path / "comparison.txt"
    report = aggregate_metrics(FULL)

    write_comparison_report(
        full_report=report,
        short_report=aggregate_metrics(SHORT),
        full_n=len(FULL),
        short_n=len(SHORT),
        path=path,
        paired_predictions=(FULL, SHORT),
    )

    text = path.read_text()
    assert "Paired significance" in text
    assert "McNemar" in text