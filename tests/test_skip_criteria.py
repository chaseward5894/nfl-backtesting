"""Tests for W7 skip-criteria resolution and the comparison filter section."""

from __future__ import annotations

from types import SimpleNamespace

from nflbetting_backtest import aggregate_metrics
from nflbetting_backtest.cli import _effective_skip_criteria
from nflbetting_backtest.metrics import Prediction
from nflbetting_backtest.report import write_comparison_report


def _args(filter_edge=None, filter_prob=None, filter_spread=None):
    return SimpleNamespace(
        filter_edge=filter_edge,
        filter_prob=filter_prob,
        filter_spread=filter_spread,
    )


def _criteria(enabled=True):
    return SimpleNamespace(
        enabled=enabled, min_edge=3.0, min_prob=0.6, max_spread=14.0
    )


def test_explicit_flags_override_config() -> None:
    args = _args(filter_edge=5.0)
    cfg = SimpleNamespace(skip_criteria=_criteria())
    assert _effective_skip_criteria(args, cfg) == (5.0, None, None)


def test_config_used_when_no_flags() -> None:
    args = _args()
    cfg = SimpleNamespace(skip_criteria=_criteria())
    assert _effective_skip_criteria(args, cfg) == (3.0, 0.6, 14.0)


def test_disabled_config_returns_no_criteria() -> None:
    args = _args()
    cfg = SimpleNamespace(skip_criteria=_criteria(enabled=False))
    assert _effective_skip_criteria(args, cfg) == (None, None, None)


def _predictions():
    return [
        Prediction("g1", 2024, 1, "H", "A", 7.0, 0.60, 10.0, 24, 14, 3.0),
        Prediction("g2", 2024, 1, "H", "A", -3.0, 0.55, -7.0, 14, 21, 3.0),
    ]


def test_comparison_report_includes_filter_section(tmp_path) -> None:
    all_report = aggregate_metrics(_predictions())
    picks_report = aggregate_metrics(_predictions()[:1])
    path = tmp_path / "comparison.txt"

    write_comparison_report(
        full_report=all_report,
        short_report=all_report,
        full_n=2,
        short_n=2,
        path=path,
        filter_sections=[
            {
                "dataset": "DATASET_FULL",
                "all_report": all_report,
                "filtered_report": picks_report,
                "n_all": 2,
                "n_filtered": 1,
            }
        ],
    )

    text = path.read_text()
    assert "All games vs publishable picks" in text
    assert "DATASET_FULL" in text