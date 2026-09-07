"""Report writers: metrics.txt, predictions.xlsx, predictions.jsonl."""

import io
import json
from pathlib import Path
from typing import Iterable, Optional
import polars as pl

from .metrics import MetricsReport, MetricsBlock, MarketComparison, Prediction


# ---------------------------------------------------------------------------
# Text metrics (matches review2026/baseline/metrics.txt byte-for-byte)
# ---------------------------------------------------------------------------

def _write_block(label: str, block: MetricsBlock, fh) -> None:
    ats = block.ats
    spread = block.spread
    su = block.su
    ll = block.log_loss
    if label:
        fh.write(f"{label}\n")
    fh.write(
        f"  ATS:  {ats.p_value:.1%} "
        f"[{ats.ci_lower:.1%}, {ats.ci_upper:.1%}]  "
        f"{ats.corret_items}/{ats.total_items}\n"
    )
    fh.write(
        f"  MAE:  {spread.mae:.2f} "
        f"[{spread.mae_ci_lower:.2f}, {spread.mae_ci_upper:.2f}]\n"
    )
    fh.write(
        f"  SU:   {su.p_value:.1%} "
        f"[{su.ci_lower:.1%}, {su.ci_upper:.1%}]  "
        f"{su.corret_items}/{su.total_items}\n"
    )
    fh.write(
        f"  LL:   {ll.p_value:.4f} "
        f"[{ll.ci_lower:.4f}, {ll.ci_upper:.4f}]  "
        f"n={ll.total_items}\n"
    )


def _format_edge_range(lo: float, hi: float) -> str:
    hi_s = "inf" if hi == float("inf") else f"{hi:.1f}"
    return f"[{lo:.1f}, {hi_s})"


def _write_edge_row(per_edge, fh) -> None:
    ats = per_edge.ats
    hi = "inf" if per_edge.max_edge == float("inf") else f"{per_edge.max_edge:.1f}"
    range_s = f"[{per_edge.min_edge:.1f}, {hi})"
    # Mark low-n rows (<30) with an asterisk (matches baseline convention)
    flag = "*" if 0 < ats.total_items < 30 else " "
    fh.write(
        f"{flag}{per_edge.label:<6s} {range_s:<14s} "
        f"ATS:  {ats.p_value:.1%} "
        f"[{ats.ci_lower:.1%}, {ats.ci_upper:.1%}]  "
        f"{ats.corret_items}/{ats.total_items}\n"
    )


def _write_market(cmp: MarketComparison, fh) -> None:
    fh.write(f"  Games compared: {cmp.total_items}\n")
    fh.write(
        f"  MAE:   model={cmp.mae_model:.2f}  "
        f"market={cmp.mae_market:.2f}  "
        f"diff={cmp.mae_diff:+.2f}   "
        f"({'model better' if cmp.mae_diff < 0 else 'market better'})\n"
    )
    fh.write(
        f"  RMSE:  model={cmp.rmse_model:.2f}  "
        f"market={cmp.rmse_market:.2f}  "
        f"diff={cmp.rmse_diff:+.2f}   "
        f"({'model better' if cmp.rmse_diff < 0 else 'market better'})\n"
    )
    fh.write(
        f"  MedAE: model={cmp.median_ae_model:.2f}  "
        f"market={cmp.median_ae_market:.2f}  "
        f"diff={cmp.median_ae_diff:+.2f}   "
        f"({'model better' if cmp.median_ae_diff < 0 else 'market better'})\n"
    )
    fh.write(
        f"  Bias:  model={cmp.bias_model:+.2f}  "
        f"market={cmp.bias_market:+.2f}  "
        f"diff={cmp.bias_diff:+.2f}\n"
    )


def write_metrics_txt(report: MetricsReport, path: Path) -> None:
    """Reproduce metrics.txt format exactly (same sections, same labels).

    When the predictions span a single season, emit "by week" with
    `wk{season}` labels (matching metrics_2025.txt). Otherwise emit
    "by season and timing" with all/early/late (matching metrics.txt).
    """
    path = Path(path)
    with path.open("w") as fh:
        fh.write("# metric value [ci_lower, ci_upper]  correct/total\n\n")
        fh.write("aggregate\n")
        _write_block("  ", report.aggregate, fh)
        fh.write("\nby season timing\n")
        for label, block in report.by_season_timing.items():
            _write_block(f"  {label}", block, fh)
        fh.write("\nby season and timing\n")
        for season, timing in report.by_season_and_timing.items():
            fh.write(f"  {season}\n")
            for timing_label, block in timing.items():
                _write_block(f"    {timing_label}", block, fh)
        fh.write("\nby edge size\n")
        for per_edge in report.by_edge_size.values():
            _write_edge_row(per_edge, fh)
        fh.write("\nvs market (model vs market spread error on same games)\n")
        _write_market(report.market, fh)
        fh.write("\nvs market by season\n")
        for season in sorted(report.market_by_season):
            cmp = report.market_by_season[season]
            fh.write(f"  {season}\n")
            _write_market(cmp, fh)


# ---------------------------------------------------------------------------
# Predictions → jsonl
# ---------------------------------------------------------------------------

def write_predictions_jsonl(predictions: Iterable[Prediction], fh) -> int:
    """Write one JSON object per line to fh (file path or file-like)."""
    n = 0
    if isinstance(fh, (str, Path)):
        with open(fh, "w") as f:
            return write_predictions_jsonl(predictions, f)
    for p in predictions:
        fh.write(json.dumps({
            "game_id": p.game_id,
            "season": p.season,
            "week": p.week,
            "home_team": p.home_team,
            "away_team": p.away_team,
            "predicted_margin": p.predicted_margin,
            "predicted_win_probability": p.predicted_win_probability,
            "actual_margin": p.actual_margin,
            "actual_home_score": p.actual_home_score,
            "actual_away_score": p.actual_away_score,
            "spread_line": p.spread_line,
        }) + "\n")
        n += 1
    return n


def write_predictions_pkl(predictions: Iterable[Prediction], path: Path) -> int:
    """Pickle the list of Prediction objects (compatible with pickle.load)."""
    import pickle
    preds = list(predictions)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(preds, f)
    return len(preds)


# ---------------------------------------------------------------------------
# Predictions → xlsx (mirrors review2026/baseline/baseline.xlsx)
# ---------------------------------------------------------------------------

def _predictions_to_rows(predictions: Iterable[Prediction]) -> list[dict]:
    rows = []
    for p in predictions:
        spread = p.spread_line
        edge = (p.predicted_margin - spread) if spread is not None else None
        rows.append({
            "season": p.season,
            "week": p.week,
            "game_id": p.game_id,
            "home_team": p.home_team,
            "away_team": p.away_team,
            "predicted_margin": p.predicted_margin,
            "predicted_win_probability": p.predicted_win_probability,
            "spread_line": spread,
            "market_implied_margin": spread,
            "edge": edge,
            "actual_margin": p.actual_margin,
            "actual_home_score": p.actual_home_score,
            "actual_away_score": p.actual_away_score,
            "home_covered": (
                p.actual_margin > spread
                if spread is not None and p.actual_margin is not None
                else None
            ),
            "is_push": (
                abs(p.actual_margin - spread) < 0.25
                if spread is not None and p.actual_margin is not None
                else None
            ),
            "no_edge": abs(edge) < 0.01 if edge is not None else None,
        })
    return sorted(rows, key=lambda r: (r["season"], r["week"], r["game_id"]))


def _metrics_block_to_rows(block: MetricsBlock) -> list[dict]:
    return [
        {"metric": "ATS", "value": block.ats.p_value,
         "ci_lower": block.ats.ci_lower, "ci_upper": block.ats.ci_upper,
         "n_correct": block.ats.corret_items, "n_total": block.ats.total_items,
         "n_games_with_odds": block.ats.games_with_odds},
        {"metric": "MAE", "value": block.spread.mae,
         "ci_lower": block.spread.mae_ci_lower, "ci_upper": block.spread.mae_ci_upper,
         "n_correct": None, "n_total": None, "n_games_with_odds": None},
        {"metric": "RMSE", "value": block.spread.rmse,
         "ci_lower": block.spread.rmse_ci_lower, "ci_upper": block.spread.rmse_ci_upper,
         "n_correct": None, "n_total": None, "n_games_with_odds": None},
        {"metric": "MedAE", "value": block.spread.median_ae,
         "ci_lower": block.spread.median_ae_ci_lower, "ci_upper": block.spread.median_ae_ci_upper,
         "n_correct": None, "n_total": None, "n_games_with_odds": None},
        {"metric": "Bias", "value": block.spread.bias,
         "ci_lower": block.spread.bias_ci_lower, "ci_upper": block.spread.bias_ci_upper,
         "n_correct": None, "n_total": None, "n_games_with_odds": None},
        {"metric": "SU", "value": block.su.p_value,
         "ci_lower": block.su.ci_lower, "ci_upper": block.su.ci_upper,
         "n_correct": block.su.corret_items, "n_total": block.su.total_items,
         "n_games_with_odds": None},
        {"metric": "LL", "value": block.log_loss.p_value,
         "ci_lower": block.log_loss.ci_lower, "ci_upper": block.log_loss.ci_upper,
         "n_correct": None, "n_total": block.log_loss.total_items,
         "n_games_with_odds": None},
    ]


def _by_season_timing_to_rows(by_season_timing: dict) -> list[dict]:
    rows = []
    for label, block in by_season_timing.items():
        for r in _metrics_block_to_rows(block):
            r = dict(r)
            r["timing"] = label
            rows.append(r)
    return rows


def _by_season_to_rows(by_season: dict) -> list[dict]:
    rows = []
    for season, block in by_season.items():
        for r in _metrics_block_to_rows(block):
            r = dict(r)
            r["season"] = season
            rows.append(r)
    return rows


def _by_season_and_timing_to_rows(by_season_and_timing: dict) -> list[dict]:
    rows = []
    for season, timing_dict in by_season_and_timing.items():
        for timing_label, block in timing_dict.items():
            for r in _metrics_block_to_rows(block):
                r = dict(r)
                r["season"] = season
                r["timing"] = timing_label
                rows.append(r)
    return rows


def _by_edge_size_to_rows(by_edge_size: dict) -> list[dict]:
    rows = []
    for label, per_edge in by_edge_size.items():
        hi = "inf" if per_edge.max_edge == float("inf") else f"{per_edge.max_edge:.1f}"
        rng = f"[{per_edge.min_edge:.1f}, {hi})"
        rows.append({
            "label": label,
            "range": rng,
            "min_edge": per_edge.min_edge,
            "max_edge": per_edge.max_edge,
            "ats_value": per_edge.ats.p_value,
            "ats_ci_lower": per_edge.ats.ci_lower,
            "ats_ci_upper": per_edge.ats.ci_upper,
            "n_correct": per_edge.ats.corret_items,
            "n_total": per_edge.ats.total_items,
            "n_games_with_odds": per_edge.ats.games_with_odds,
        })
    return rows


def _market_to_rows(cmp: MarketComparison) -> list[dict]:
    return [
        {"metric": "MAE", "model": cmp.mae_model, "market": cmp.mae_market,
         "diff_model_minus_market": cmp.mae_diff},
        {"metric": "RMSE", "model": cmp.rmse_model, "market": cmp.rmse_market,
         "diff_model_minus_market": cmp.rmse_diff},
        {"metric": "MedAE", "model": cmp.median_ae_model,
         "market": cmp.median_ae_market,
         "diff_model_minus_market": cmp.median_ae_diff},
        {"metric": "Bias", "model": cmp.bias_model, "market": cmp.bias_market,
         "diff_model_minus_market": cmp.bias_diff},
    ]


def _market_by_season_to_rows(market_by_season: dict) -> list[dict]:
    rows = []
    for season, cmp in market_by_season.items():
        for entry in _market_to_rows(cmp):
            entry = dict(entry)
            entry["season"] = season
            entry["n_games"] = cmp.total_items
            rows.append(entry)
    return rows


def write_predictions_xlsx(
    predictions: Iterable[Prediction],
    report: MetricsReport,
    path: Path,
    *,
    features: Optional[pl.DataFrame] = None,
) -> int:
    """Write 8-sheet workbook matching review2026/baseline/baseline.xlsx."""
    import openpyxl
    from openpyxl import Workbook
    import polars as pl  # noqa

    preds = list(predictions)
    pred_rows = _predictions_to_rows(preds)

    # Merge per-game features if provided
    if features is not None:
        feat_cols = [
            "R_avg", "R_opp_avg", "Eff_off", "Eff_def",
            "Mmtum_off", "Mmtum_def",
            "Rest_Travel_Fatigue", "Weather_Impact", "Injury_Impact",
        ]
        keep = ["game_id"] + feat_cols
        home = (
            features.filter(pl.col("location") == "home")
            .select(keep)
            .rename({c: f"home_{c}" for c in feat_cols})
        )
        away = (
            features.filter(pl.col("location") == "away")
            .select(keep)
            .rename({c: f"away_{c}" for c in feat_cols})
        )
        merged = home.join(away, on="game_id", how="full", coalesce=True)
        feat_map = {r["game_id"]: r for r in merged.to_dicts()}
        for row in pred_rows:
            extra = feat_map.get(row["game_id"], {})
            row.update({k: v for k, v in extra.items() if k != "game_id"})

    sheets = [
        ("predictions", pred_rows),
        ("aggregate", _metrics_block_to_rows(report.aggregate)),
        ("by_season_timing", _by_season_timing_to_rows(report.by_season_timing)),
        ("by_season", _by_season_to_rows(report.by_season)),
        ("by_season_and_timing", _by_season_and_timing_to_rows(report.by_season_and_timing)),
        ("by_edge_size", _by_edge_size_to_rows(report.by_edge_size)),
        ("vs_market", _market_to_rows(report.market)),
        ("vs_market_by_season", _market_by_season_to_rows(report.market_by_season)),
    ]

    wb = Workbook()
    # Remove default sheet
    wb.remove(wb.active)
    for sheet_name, rows in sheets:
        ws = wb.create_sheet(sheet_name)
        if rows:
            cols = list(rows[0].keys())
            ws.append(cols)
            for r in rows:
                ws.append([r.get(c) for c in cols])
    wb.save(path)
    return len(preds)