"""Tests for W5: chunk-failure accounting and coverage reporting."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl
import pytest

from nflbetting_backtest import (
    Prediction,
    WalkForwardDiagnostics,
    write_coverage_txt,
)
from nflbetting_backtest.backtest import (
    _neutralize_missing_features,
    _predict_chunk,
    walk_forward_validation,
)


def _features() -> pl.DataFrame:
    rows = []
    for game_id, home, away in [("g1", "A", "B"), ("g2", "C", "D")]:
        for team, location in [(home, "home"), (away, "away")]:
            rows.append(
                {
                    "game_id": game_id,
                    "team_id": team,
                    "location": location,
                    "R_avg": 1500.0,
                    "Eff_off": None if game_id == "g2" else 0.5,
                }
            )
    return pl.DataFrame(rows)


def _schedule() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "season": [2024, 2024, 2024],
            "week": [2, 2, 2],
            "home_team": ["A", "C", "E"],
            "away_team": ["B", "D", "F"],
            "home_score": [24, 17, 20],
            "away_score": [17, 20, 20],
            "spread_line": [-3.0, 1.5, 0.0],
        }
    )


def test_neutralize_missing_features_zeroes_the_difference() -> None:
    home = {"A": 1.0, "B": None, "C": 3.0}
    away = {"A": 2.0, "B": 5.0, "C": None}

    imputed = _neutralize_missing_features(home, away, ["A", "B", "C"])

    assert imputed == 2
    assert home["B"] == away["B"] == 5.0
    assert home["C"] == away["C"] == 3.0
    assert home["A"] - away["A"] == -1.0


def test_predict_chunk_counts_missing_features_and_imputations(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "nflbetting_backtest.backtest.predict_outcome",
        lambda *args, **kwargs: (0.0, 0, 0.5),
    )
    model_cfg = SimpleNamespace(feature_names=["R_avg", "Eff_off"])

    predictions, missing, imputed = _predict_chunk(
        _features(), _schedule(), None, model_cfg
    )

    assert len(predictions) == 2
    assert missing == 1  # g3 has no feature row
    assert imputed == 1  # g2's Eff_off is missing on both sides


def _wf_features() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": [f"2024_{w:02d}" for w in (1, 2, 3)],
            "season": [2024, 2024, 2024],
            "week": [1, 2, 3],
        }
    )


def _wf_schedule() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": [f"2024_{w:02d}" for w in (1, 2, 3)],
            "season": [2024, 2024, 2024],
            "week": [1, 2, 3],
            "home_team": ["A", "A", "A"],
            "away_team": ["B", "B", "B"],
            "home_score": [20, 20, 20],
            "away_score": [17, 17, 17],
            "spread_line": [-3.0, -3.0, -3.0],
        }
    )


def _raise_on_train(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("train failed")

    monkeypatch.setattr(
        "nflbetting_backtest.backtest._train_and_collect_weights", _boom
    )


def test_strict_reraises_a_chunk_failure(monkeypatch) -> None:
    _raise_on_train(monkeypatch)
    model_cfg = SimpleNamespace(feature_names=["R_avg"])

    with pytest.raises(RuntimeError, match="train failed"):
        walk_forward_validation(
            _wf_features(), _wf_schedule(), model_cfg, strict=True
        )


def test_non_strict_records_skipped_chunks_and_coverage(monkeypatch) -> None:
    _raise_on_train(monkeypatch)
    model_cfg = SimpleNamespace(feature_names=["R_avg"])
    diagnostics = WalkForwardDiagnostics()

    predictions = walk_forward_validation(
        _wf_features(),
        _wf_schedule(),
        model_cfg,
        diagnostics=diagnostics,
    )

    assert predictions == []
    reasons = {skip.reason for skip in diagnostics.skipped_chunks}
    assert "no_training_data" in reasons
    assert any("train failed" in reason for reason in reasons)

    coverage = {record.season: record for record in diagnostics.coverage}
    assert coverage[2024].predicted == 0
    assert coverage[2024].scheduled == 3
    assert coverage[2024].coverage_pct == 0.0


def test_populate_and_write_coverage_report(tmp_path) -> None:
    diagnostics = WalkForwardDiagnostics()
    predictions = [
        Prediction("g1", 2024, 2, "A", "B", 0.0, 0.5),
        Prediction("g2", 2024, 2, "C", "D", 0.0, 0.5),
    ]

    diagnostics.populate(
        predictions=predictions,
        schedule=_schedule(),
        skipped_chunks=[],
        season_missing={2024: 1},
        season_imputed={2024: 2},
    )

    coverage = diagnostics.coverage[0]
    assert coverage.season == 2024
    assert coverage.scheduled == 3
    assert coverage.scheduled_with_odds == 3
    assert coverage.predicted == 2
    assert coverage.coverage_pct == pytest.approx(2 / 3)
    assert diagnostics.imputed_feature_cells == 2

    path = tmp_path / "DATASET_FULL.coverage.txt"
    write_coverage_txt(diagnostics, path, label="DATASET_FULL")
    text = path.read_text()
    assert "dataset: DATASET_FULL" in text
    assert "predicted: 2" in text
    assert "2024" in text