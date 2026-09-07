"""Walk-forward validation + explicit-period training."""

from typing import Optional
import polars as pl
from nflbetting.model.rating_model import (
    RatingModel,
    RatingModelWeights,
    prepare_training_data,
    predict_outcome,
)
from nflbetting.model.rating_model.defs import ModelConfig
from nflbetting.model.model_lifecycle import TrainingPeriod
from nflbetting.utils.cutoffs import filter_as_of

from .metrics import Prediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_period(
    df: pl.DataFrame, start: tuple[int, int], end: tuple[int, int]
) -> pl.DataFrame:
    start_s, start_w = start
    end_s, end_w = end
    if start_s == end_s:
        return df.filter(
            (pl.col("season") == start_s)
            & (pl.col("week") >= start_w)
            & (pl.col("week") <= end_w)
        )
    return df.filter(
        ((pl.col("season") == start_s) & (pl.col("week") >= start_w))
        | ((pl.col("season") > start_s) & (pl.col("season") < end_s))
        | ((pl.col("season") == end_s) & (pl.col("week") <= end_w))
    ).sort(["season", "week", "game_id"])


def _trim_to_last_n_weeks(
    df: pl.DataFrame, season: int, week: int, n: int
) -> pl.DataFrame:
    """Keep rows where (season, week) is strictly before the test chunk's
    first week, then trim to the most recent n distinct (season, week) pairs.

    The strict-less-than step is delegated to
    ``nflbetting.utils.cutoffs.filter_as_of`` (single chokepoint
    shared with the production feature pipeline; see Stage 2 of
    ``track/roadmap.md``). The trim-to-last-n-weeks step is
    framework-specific and stays here.
    """
    train = filter_as_of(df, target_season=season, target_week=week)
    if train.is_empty():
        return train
    train = train.with_columns(
        (pl.col("season") * 100 + pl.col("week")).alias("_time_index")
    )
    unique_times = sorted(train["_time_index"].unique().to_list())
    if len(unique_times) > n:
        cutoff = unique_times[-n]
        train = train.filter(pl.col("_time_index") >= cutoff)
    return train.drop("_time_index").sort(["season", "week", "game_id"])


def _train_and_collect_weights(
    train_df: pl.DataFrame, model_cfg: ModelConfig
) -> RatingModelWeights:
    X, y = prepare_training_data(train_df, model_cfg)
    if len(X) == 0:
        raise ValueError("no training data available")
    model = RatingModel(model_cfg)
    model.fit(X, y)
    reg_w, class_w = model.get_weights()
    # Real metrics would require held-out data; for walk-forward we
    # score on the next chunk, so leave these blank. nflbetting's
    # RatingModelWeights requires the fields so we fill with zeros.
    from nflbetting.model.rating_model.defs import (
        RegressionMetrics, ClassificationMetrics,
    )
    return RatingModelWeights(
        season=0,
        regression_weights=reg_w,
        regression_metrics=RegressionMetrics(0.0, 0.0),
        classification_weights=class_w,
        classification_metrics=ClassificationMetrics(),
        alpha=model_cfg.alpha,
        model_type=model_cfg.model_type,
    )


def _predict_chunk(
    features: pl.DataFrame,
    schedule: pl.DataFrame,
    weights: RatingModelWeights,
    model_cfg: ModelConfig,
) -> list[Prediction]:
    """Predict all games in the test chunk, returning Prediction objects."""
    preds: list[Prediction] = []
    schedule_games = schedule.select(["game_id", "season", "week",
                                       "home_team", "away_team",
                                       "home_score", "away_score",
                                       "spread_line"]).to_dicts()
    feature_groups = {g[0][0]: g[1] for g in features.partition_by("game_id", as_dict=True).items()}
    for game_row in schedule_games:
        gid = game_row["game_id"]
        if gid not in feature_groups:
            continue
        game_features = feature_groups[gid]
        home_mask = (pl.col("team_id") == game_row["home_team"]) & (pl.col("location") == "home")
        away_mask = (pl.col("team_id") == game_row["away_team"]) & (pl.col("location") == "away")
        try:
            home_row = game_features.filter(home_mask).to_dicts()[0]
            away_row = game_features.filter(away_mask).to_dicts()[0]
        except IndexError:
            continue
        predicted_margin, _, win_prob = predict_outcome(home_row, away_row, weights, model_cfg)
        home_score = game_row.get("home_score")
        away_score = game_row.get("away_score")
        if home_score is not None and away_score is not None \
                and not (isinstance(home_score, float) and np_isnan(home_score)) \
                and not (isinstance(away_score, float) and np_isnan(away_score)):
            actual_home_score = int(home_score)
            actual_away_score = int(away_score)
            actual_margin = actual_home_score - actual_away_score
        else:
            actual_home_score = actual_away_score = actual_margin = None
        spread = game_row.get("spread_line")
        if spread is not None and (isinstance(spread, float) and np_isnan(spread)):
            spread = None
        preds.append(Prediction(
            game_id=gid,
            season=int(game_row["season"]),
            week=int(game_row["week"]),
            home_team=game_row["home_team"],
            away_team=game_row["away_team"],
            predicted_margin=float(predicted_margin),
            predicted_win_probability=float(win_prob),
            actual_margin=actual_margin,
            actual_home_score=actual_home_score,
            actual_away_score=actual_away_score,
            spread_line=spread,
        ))
    return preds


def np_isnan(x) -> bool:
    """Polars-friendly isnan (no numpy import)."""
    try:
        return x != x  # NaN != NaN
    except TypeError:
        return False


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def walk_forward_validation(
    features: pl.DataFrame,
    schedule: pl.DataFrame,
    model_cfg: ModelConfig,
    *,
    training_weeks: int = 60,
    test_weeks: int = 1,
) -> list[Prediction]:
    """Rolling walk-forward: train on the last N weeks, predict next chunk."""
    features = features.sort(["season", "week", "game_id"])
    all_preds: list[Prediction] = []
    seasons = sorted(features["season"].unique().to_list())
    for season in seasons:
        sched_season = schedule.filter(pl.col("season") == season)
        if sched_season.is_empty():
            continue
        weeks = sorted(sched_season["week"].unique().to_list())
        for i in range(0, len(weeks), test_weeks):
            chunk = weeks[i:i + test_weeks]
            if not chunk:
                continue
            test_start_w = chunk[0]
            test_end_w = chunk[-1]
            train = _trim_to_last_n_weeks(features, season, test_start_w, training_weeks)
            if train.is_empty():
                continue
            test_features = features.filter(
                (pl.col("season") == season)
                & (pl.col("week") >= test_start_w)
                & (pl.col("week") <= test_end_w)
            )
            if "margin" in test_features.columns:
                test_features = test_features.drop("margin")
            if test_features.is_empty():
                continue
            test_sched = sched_season.filter(
                (pl.col("week") >= test_start_w) & (pl.col("week") <= test_end_w)
            )
            try:
                weights = _train_and_collect_weights(train, model_cfg)
                all_preds.extend(_predict_chunk(test_features, test_sched, weights, model_cfg))
            except Exception:
                continue
    return all_preds


def train_period(
    features: pl.DataFrame,
    schedule: pl.DataFrame,
    period: TrainingPeriod,
    model_cfg: ModelConfig,
) -> list[Prediction]:
    """Train on period.train_*, predict on period.test_*."""
    train = _filter_period(features, period.train_start, period.train_end)
    test = _filter_period(features, period.test_start, period.test_end)
    if "margin" in test.columns:
        test = test.drop("margin")
    test_sched = schedule.filter(
        (pl.col("season") == period.test_start[0])
        & (pl.col("week") >= period.test_start[1])
        & (pl.col("week") <= period.test_end[1])
    )
    model = _train_and_collect_weights(train, model_cfg)
    return _predict_chunk(test, test_sched, model, model_cfg)
