"""Report writers: metrics.txt, predictions.xlsx, predictions.pkl, comparison.txt."""

import io
import json
from pathlib import Path
from typing import Iterable, Optional
import polars as pl

from .metrics import MetricsReport, MetricsBlock, MarketComparison, Prediction


# ---------------------------------------------------------------------------
# Text metrics (matches the v1 baseline format byte-for-byte)
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
# Predictions → pkl
# ---------------------------------------------------------------------------

def write_predictions_pkl(predictions: Iterable[Prediction], path: Path) -> int:
    """Pickle the list of Prediction objects (compatible with pickle.load)."""
    import pickle
    preds = list(predictions)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(preds, f)
    return len(preds)


# ---------------------------------------------------------------------------
# Predictions → xlsx (mirrors the v1 baseline workbook)
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
    """Write 8-sheet workbook matching the v1 baseline workbook."""
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


# ---------------------------------------------------------------------------
# Comparison: performance (full vs 2025) + error estimates (model vs market)
# ---------------------------------------------------------------------------

def _fmt_signed(x: float, digits: int = 4) -> str:
    return f"{x:+.{digits}f}"


def write_comparison_report(
    full_report: MetricsReport,
    short_report: MetricsReport,
    full_n: int,
    short_n: int,
    path: Path,
    *,
    full_label: str = "full-features",
    short_label: str = "2025-features",
    filter_sections: Optional[list] = None,
) -> None:
    """Two-block side-by-side comparison on the same 2025 games.

    Section 1: performance metrics comparing the two feature sets:
        full vs short, with delta = short - full, verdict
        (higher is better for ATS, SU).

    Section 2: error estimates comparing the model against the market,
        once per dataset (full / short). Each row is one of MAE, RMSE,
        MedAE, Bias, with delta = model - market and a verdict
        (lower is better for absolute errors; for signed Bias,
        closer to zero is better).
    """
    a = full_report.aggregate
    b = short_report.aggregate
    fa = full_report.market
    sb = short_report.market
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _perf_line(label, av, bv, fmt, *, lower_is_better):
        delta = bv - av
        better = (
            "short better" if (delta < 0) == lower_is_better
            else "full better"
        )
        return f"  {label:<18}  {fmt(av):>10}  {fmt(bv):>10}  " \
               f"{_fmt_signed(delta):>10}   {better}"

    def _err_line(label, model_value, market_value, delta_value, *, lower_is_better):
        better = (
            "model better" if (delta_value < 0) == lower_is_better
            else "market better"
        )
        return (
            f"  {label:<12}  {model_value:>10.4f}  {market_value:>10.4f}  "
            f"{_fmt_signed(delta_value):>10}   {better}"
        )

    with path.open("w") as fh:
        fh.write(
            f"# Comparison on {full_n} games (both models scored on the "
            f"same 2025 schedule).\n"
        )
        fh.write("# Generated by nflbt-run.\n\n")

        # ---- Section 1: performance ----
        fh.write("## Performance: {} vs {}\n".format(full_label, short_label))
        fh.write("# Higher is better for ATS, SU.\n")
        fh.write("  %-18s  %10s  %10s  %10s   %s\n" % (
            "metric", "full", "short", "delta", "verdict",
        ))
        fh.write("  " + "-" * 18 + "  " + "-" * 10 + "  " + "-" * 10
                 + "  " + "-" * 10 + "   -------\n")
        fh.write(_perf_line(
            "ATS pct", a.ats.p_value, b.ats.p_value,
            lambda x: f"{x:.1%}", lower_is_better=False,
        ) + "\n")
        fh.write(_perf_line(
            "SU pct", a.su.p_value, b.su.p_value,
            lambda x: f"{x:.1%}", lower_is_better=False,
        ) + "\n")
        fh.write("\n")

        # ---- Section 2: error estimates vs market (per dataset) ----
        fh.write("## Error estimates: model vs market (per dataset)\n")
        fh.write("# For each error metric: delta = model - market.\n")
        fh.write("#   MAE / RMSE / MedAE: lower (closer to market) is better.\n")
        fh.write("#   Bias:                signed; 0 is best.\n\n")

        fh.write(f"### {full_label}  (n={full_n}, "
                 f"games={fa.total_items})\n")
        fh.write("  %-12s  %10s  %10s  %10s   %s\n" % (
            "metric", "model", "market", "delta", "verdict",
        ))
        fh.write("  " + "-" * 12 + "  " + "-" * 10 + "  " + "-" * 10
                 + "  " + "-" * 10 + "   -------\n")
        fh.write(_err_line("MAE",   fa.mae_model,  fa.mae_market,
                          fa.mae_diff, lower_is_better=True)  + "\n")
        fh.write(_err_line("RMSE",  fa.rmse_model, fa.rmse_market,
                          fa.rmse_diff, lower_is_better=True)  + "\n")
        fh.write(_err_line("MedAE", fa.median_ae_model, fa.median_ae_market,
                          fa.median_ae_diff, lower_is_better=True)  + "\n")
        fh.write(_err_line("Bias",  fa.bias_model, fa.bias_market,
                          fa.bias_diff, lower_is_better=False) + "\n")
        fh.write("\n")

        fh.write(f"### {short_label}  (n={short_n}, "
                 f"games={sb.total_items})\n")
        fh.write("  %-12s  %10s  %10s  %10s   %s\n" % (
            "metric", "model", "market", "delta", "verdict",
        ))
        fh.write("  " + "-" * 12 + "  " + "-" * 10 + "  " + "-" * 10
                 + "  " + "-" * 10 + "   -------\n")
        fh.write(_err_line("MAE",   sb.mae_model,  sb.mae_market,
                          sb.mae_diff, lower_is_better=True)  + "\n")
        fh.write(_err_line("RMSE",  sb.rmse_model, sb.rmse_market,
                          sb.rmse_diff, lower_is_better=True)  + "\n")
        fh.write(_err_line("MedAE", sb.median_ae_model, sb.median_ae_market,
                          sb.median_ae_diff, lower_is_better=True)  + "\n")
        fh.write(_err_line("Bias",  sb.bias_model, sb.bias_market,
                          sb.bias_diff, lower_is_better=False) + "\n")
        fh.write("\n")

        if filter_sections:
            fh.write("## All games vs publishable picks (per dataset)\n")
            fh.write("# ATS uses the home-margin convention (see metrics.txt).\n")
            for section in filter_sections:
                all_ats = section["all_report"].aggregate.ats
                picks_ats = section["filtered_report"].aggregate.ats
                delta = picks_ats.p_value - all_ats.p_value
                fh.write(
                    f"  {section['dataset']:<14} "
                    f"all={all_ats.p_value:6.1%} (n={section['n_all']:4d})  "
                    f"picks={picks_ats.p_value:6.1%} "
                    f"(n={section['n_filtered']:4d})  "
                    f"delta={delta:+5.1%}\n"
                )
            fh.write("\n")


# ---------------------------------------------------------------------------
# Walk-forward coverage (W5)
# ---------------------------------------------------------------------------

def write_coverage_txt(
    diagnostics, path: Path, *, label: Optional[str] = None
) -> None:
    """Write the per-season coverage report for a walk-forward run.

    ``diagnostics`` is a ``WalkForwardDiagnostics``; typed loosely to
    avoid a report → backtest import cycle.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write("# walk-forward coverage\n\n")
        if label:
            fh.write(f"dataset: {label}\n")
        fh.write(f"predicted: {diagnostics.predicted}\n")
        fh.write(
            f"imputed_feature_cells: {diagnostics.imputed_feature_cells}\n"
        )
        fh.write(f"skipped_chunks: {len(diagnostics.skipped_chunks)}\n\n")
        fh.write("per season\n")
        fh.write(
            "  season  scheduled  with_odds  predicted  coverage  "
            "missing_features  skipped_games  imputed_cells\n"
        )
        for coverage in diagnostics.coverage:
            fh.write(
                f"  {coverage.season:>6}  {coverage.scheduled:>9}  "
                f"{coverage.scheduled_with_odds:>9}  "
                f"{coverage.predicted:>9}  {coverage.coverage_pct:>8.1%}  "
                f"{coverage.skipped_missing_features:>16}  "
                f"{coverage.skipped_chunk_games:>13}  "
                f"{coverage.imputed_feature_cells:>13}\n"
            )
        if diagnostics.skipped_chunks:
            fh.write("\nskipped chunks\n")
            for skip in diagnostics.skipped_chunks:
                fh.write(
                    f"  {skip.season} weeks={skip.weeks}: {skip.reason}\n"
                )