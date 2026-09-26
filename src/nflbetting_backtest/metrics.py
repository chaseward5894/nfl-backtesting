"""Metrics dataclasses + aggregate_metrics. Reproduces metrics.txt format."""

from dataclasses import dataclass, field
from typing import Iterable, Optional
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Dataclasses (matching the v1 baseline metrics format)
# ---------------------------------------------------------------------------

@dataclass
class ATSAccuracy:
    p_value: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    interval_size: float = 0.95
    corret_items: int = 0
    total_items: int = 0
    games_with_odds: int = 0


@dataclass
class SpreadError:
    mae: float = 0.0
    mae_ci_lower: float = 0.0
    mae_ci_upper: float = 0.0
    rmse: float = 0.0
    rmse_ci_lower: float = 0.0
    rmse_ci_upper: float = 0.0
    median_ae: float = 0.0
    median_ae_ci_lower: float = 0.0
    median_ae_ci_upper: float = 0.0
    bias: float = 0.0
    bias_ci_lower: float = 0.0
    bias_ci_upper: float = 0.0
    interval_size: float = 0.95


@dataclass
class SUAccuracy:
    p_value: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    interval_size: float = 0.95
    corret_items: int = 0
    total_items: int = 0


@dataclass
class LogLoss:
    p_value: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    interval_size: float = 0.95
    total_items: int = 0


@dataclass
class ATSPerEdge:
    label: str = ""
    min_edge: float = 0.0
    max_edge: float = 0.0
    ats: ATSAccuracy = field(default_factory=ATSAccuracy)


@dataclass
class MarketComparison:
    mae_model: float = 0.0
    mae_market: float = 0.0
    mae_diff: float = 0.0
    rmse_model: float = 0.0
    rmse_market: float = 0.0
    rmse_diff: float = 0.0
    median_ae_model: float = 0.0
    median_ae_market: float = 0.0
    median_ae_diff: float = 0.0
    bias_model: float = 0.0
    bias_market: float = 0.0
    bias_diff: float = 0.0
    total_items: int = 0
    interval_size: float = 0.95


@dataclass
class Prediction:
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    predicted_margin: float
    predicted_win_probability: float
    actual_margin: Optional[float] = None
    actual_home_score: Optional[int] = None
    actual_away_score: Optional[int] = None
    spread_line: Optional[float] = None


@dataclass
class MetricsBlock:
    ats: ATSAccuracy
    spread: SpreadError
    su: SUAccuracy
    log_loss: LogLoss


@dataclass
class MetricsReport:
    aggregate: MetricsBlock
    by_season_timing: dict
    by_season: dict
    by_season_and_timing: dict
    by_edge_size: dict
    market: MarketComparison
    market_by_season: dict


# ---------------------------------------------------------------------------
# Helpers (Wilson CI, t-value)
# ---------------------------------------------------------------------------

def _wilson(p: float, n: int, alpha: float = 0.05) -> tuple:
    if n <= 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def _t(n: int, alpha: float = 0.05) -> float:
    return stats.t.ppf(1 - alpha / 2, n - 1)


# ---------------------------------------------------------------------------
# Per-metric computations (verbatim port from baseline)
# ---------------------------------------------------------------------------

def _ats(predictions, alpha=0.05) -> ATSAccuracy:
    valid = [p for p in predictions
             if p.spread_line is not None and p.actual_margin is not None]
    if not valid:
        return ATSAccuracy(interval_size=1 - alpha)
    correct = pushes = no_edge = 0
    for p in valid:
        market = p.spread_line
        edge = p.predicted_margin + market
        if abs(p.actual_margin - market) < 0.25:
            pushes += 1
            continue
        if abs(edge) < 0.01:
            no_edge += 1
            continue
        if edge > 0:
            if p.actual_margin > market:
                correct += 1
        else:
            if p.actual_margin < market:
                correct += 1
    total = len(valid) - pushes - no_edge
    pv = correct / total if total > 0 else 0.0
    lo, hi = _wilson(pv, total, alpha)
    return ATSAccuracy(pv, lo, hi, 1 - alpha, correct, total, len(valid))


def _spread(predictions, alpha=0.05) -> SpreadError:
    valid = [p for p in predictions if p.actual_margin is not None]
    if not valid:
        return SpreadError(interval_size=1 - alpha)
    errs = np.array([p.predicted_margin - p.actual_margin for p in valid])
    abs_errs = np.abs(errs)
    n = len(errs)
    t = _t(n, alpha)
    mae = float(np.mean(abs_errs))
    mae_se = float(np.std(abs_errs, ddof=1) / np.sqrt(n))
    sq = errs ** 2
    rmse = float(np.sqrt(np.mean(sq)))
    mse = float(np.mean(sq))
    mse_var = float(np.var(sq, ddof=1) / n)
    rmse_se = float(np.sqrt(mse_var / (4 * mse))) if mse > 0 else 0.0
    rng = np.random.default_rng(42)
    medians = [np.median(rng.choice(abs_errs, n, replace=True)) for _ in range(1000)]
    med_lo = float(np.percentile(medians, 2.5))
    med_hi = float(np.percentile(medians, 97.5))
    bias = float(np.mean(errs))
    bias_se = float(np.std(errs, ddof=1) / np.sqrt(n))
    return SpreadError(
        mae, mae - t * mae_se, mae + t * mae_se,
        rmse, rmse - t * rmse_se, rmse + t * rmse_se,
        float(np.median(abs_errs)), med_lo, med_hi,
        bias, bias - t * bias_se, bias + t * bias_se,
        1 - alpha,
    )


def _su(predictions, alpha=0.05) -> SUAccuracy:
    valid = [p for p in predictions if p.actual_margin is not None]
    if not valid:
        return SUAccuracy(interval_size=1 - alpha)
    correct = sum(
        1 for p in valid
        if (p.predicted_margin > 0) == (p.actual_margin > 0)
    )
    pv = correct / len(valid)
    lo, hi = _wilson(pv, len(valid), alpha)
    return SUAccuracy(pv, lo, hi, 1 - alpha, correct, len(valid))


def _log_loss(predictions, alpha=0.05, eps=1e-15) -> LogLoss:
    valid = [p for p in predictions
             if p.actual_margin is not None and p.predicted_win_probability is not None]
    if not valid:
        return LogLoss(interval_size=1 - alpha)
    losses = []
    for p in valid:
        prob = float(np.clip(p.predicted_win_probability, eps, 1 - eps))
        y = 1.0 if p.actual_margin > 0 else 0.0
        losses.append(-(y * np.log(prob) + (1 - y) * np.log(1 - prob)))
    arr = np.asarray(losses)
    n = len(arr)
    mean = float(np.mean(arr))
    t = _t(n, alpha)
    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    return LogLoss(mean, mean - t * se, mean + t * se, 1 - alpha, n)


def _block(predictions) -> MetricsBlock:
    return MetricsBlock(
        ats=_ats(predictions),
        spread=_spread(predictions),
        su=_su(predictions),
        log_loss=_log_loss(predictions),
    )


def _market(predictions) -> MarketComparison:
    valid = [p for p in predictions
             if p.spread_line is not None and p.actual_margin is not None]
    if not valid:
        return MarketComparison()
    model_err = np.array([p.predicted_margin - p.actual_margin for p in valid])
    market_err = np.array([p.spread_line - p.actual_margin for p in valid])
    return MarketComparison(
        mae_model=float(np.mean(np.abs(model_err))),
        mae_market=float(np.mean(np.abs(market_err))),
        mae_diff=float(np.mean(np.abs(model_err)) - np.mean(np.abs(market_err))),
        rmse_model=float(np.sqrt(np.mean(model_err ** 2))),
        rmse_market=float(np.sqrt(np.mean(market_err ** 2))),
        rmse_diff=float(np.sqrt(np.mean(model_err ** 2)) - np.sqrt(np.mean(market_err ** 2))),
        median_ae_model=float(np.median(np.abs(model_err))),
        median_ae_market=float(np.median(np.abs(market_err))),
        median_ae_diff=float(np.median(np.abs(model_err)) - np.median(np.abs(market_err))),
        bias_model=float(np.mean(model_err)),
        bias_market=float(np.mean(market_err)),
        bias_diff=float(np.mean(model_err) - np.mean(market_err)),
        total_items=len(valid),
        interval_size=0.95,
    )


# ---------------------------------------------------------------------------
# Sections (aggregate, by_season_timing, by_season, by_edge_size)
# ---------------------------------------------------------------------------

def _by_season_timing(predictions, early_weeks=6):
    early = [p for p in predictions if p.week <= early_weeks]
    late = [p for p in predictions if p.week > early_weeks]
    return {"early_season": _block(early), "late_season": _block(late)}


def _by_season(predictions):
    seasons = sorted({p.season for p in predictions if p.season is not None})
    return {s: _block([p for p in predictions if p.season == s]) for s in seasons}


def _by_season_and_timing(predictions, early_weeks=6):
    seasons = sorted({p.season for p in predictions if p.season is not None})
    out = {}
    for s in seasons:
        sp = [p for p in predictions if p.season == s]
        out[s] = {
            "all": _block(sp),
            "early": _block([p for p in sp if p.week <= early_weeks]),
            "late": _block([p for p in sp if p.week > early_weeks]),
        }
    return out


def _by_edge_size(predictions):
    bins = [
        (0.0, 1.0, "tiny"),
        (1.0, 3.0, "small"),
        (3.0, 7.0, "medium"),
        (7.0, 14.0, "large"),
        (14.0, float("inf"), "huge"),
    ]
    out = {}
    for lo, hi, label in bins:
        subset = [p for p in predictions
                  if p.spread_line is not None
                  and lo <= abs(p.predicted_margin - p.spread_line) < hi]
        out[label] = ATSPerEdge(label, lo, hi, _ats(subset))
    return out


def _market_by_season(predictions):
    seasons = sorted({p.season for p in predictions if p.season is not None})
    return {s: _market([p for p in predictions if p.season == s]) for s in seasons}


# ---------------------------------------------------------------------------
# Public aggregate
# ---------------------------------------------------------------------------

def aggregate_metrics(predictions: Iterable[Prediction]) -> MetricsReport:
    """Compute the full MetricsReport (matches metrics.txt sections)."""
    preds = list(predictions)
    return MetricsReport(
        aggregate=_block(preds),
        by_season_timing=_by_season_timing(preds),
        by_season=_by_season(preds),
        by_season_and_timing=_by_season_and_timing(preds),
        by_edge_size=_by_edge_size(preds),
        market=_market(preds),
        market_by_season=_market_by_season(preds),
    )
