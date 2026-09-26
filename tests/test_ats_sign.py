"""Sign-convention regression test for ATS grading (W6).

nflverse ``spread_line`` is the market's expected home margin (positive =
home favored), inverted relative to sportsbook display. The model's
``predicted_margin`` is also home-perspective, so the edge is
``predicted_margin - spread_line``; using ``+`` grades the opposite side.
"""

from __future__ import annotations

from nflbetting_backtest.metrics import Prediction, _ats


def _pred(game_id, predicted_margin, spread_line, actual_margin):
    return Prediction(
        game_id=game_id,
        season=2024,
        week=1,
        home_team="H",
        away_team="A",
        predicted_margin=predicted_margin,
        predicted_win_probability=0.5,
        actual_margin=actual_margin,
        spread_line=spread_line,
    )


def _correct_with_sign(predictions, sign):
    correct = 0
    for p in predictions:
        market = p.spread_line
        edge = (
            p.predicted_margin - market
            if sign == "-"
            else p.predicted_margin + market
        )
        if abs(p.actual_margin - market) < 0.25 or abs(edge) < 0.01:
            continue
        if edge > 0:
            correct += p.actual_margin > market
        else:
            correct += p.actual_margin < market
    return correct


CASES = [
    _pred("home_cover", 10.0, 7.0, 10.0),
    _pred("away_cover", 0.0, 7.0, -3.0),
    _pred("home_no_cover", 10.0, 7.0, 3.0),
    _pred("push", 5.0, 7.0, 7.0),
    _pred("no_edge", 7.0, 7.0, 0.0),
]


def test_ats_grades_on_the_home_margin_convention() -> None:
    report = _ats(CASES)
    assert report.corret_items == 2
    assert report.total_items == 3  # one push and one no-edge excluded


def test_sportsbook_sign_would_grade_the_opposite_side() -> None:
    assert _correct_with_sign(CASES, "+") == 1
    assert _correct_with_sign(CASES, "-") == 2