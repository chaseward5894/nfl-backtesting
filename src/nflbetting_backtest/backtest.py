"""Walk-forward validation + explicit-period training."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
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

logger = logging.getLogger(__name__)


@dataclass
class ChunkSkip:
    """A walk-forward chunk that produced no predictions."""

    season: int
    weeks: list[int]
    reason: str


@dataclass
class SeasonCoverage:
    """Per-season prediction coverage for one walk-forward run."""

    season: int
    scheduled: int
    scheduled_with_odds: int
    predicted: int
    skipped_missing_features: int
    skipped_chunk_games: int
    imputed_feature_cells: int

    @property
    def coverage_pct(self) -> float:
        if self.scheduled == 0:
            return 1.0
        return self.predicted / self.scheduled


@dataclass
class WalkForwardDiagnostics:
    """Coverage and skip accounting for a walk-forward run.

    Populated by :func:`walk_forward_validation` when passed as
    ``diagnostics``, so callers can see and report what was dropped
    instead of losing it to a bare ``except``.
    """

    predicted: int = 0
    imputed_feature_cells: int = 0
    skipped_chunks: list[ChunkSkip] = field(default_factory=list)
    coverage: list[SeasonCoverage] = field(default_factory=list)

    def populate(
        self,
        *,
        predictions: list[Prediction],
        schedule: pl.DataFrame,
        skipped_chunks: list[ChunkSkip],
        season_missing: dict[int, int],
        season_imputed: dict[int, int],
    ) -> None:
        scheduled = {
            int(row["season"]): int(row["len"])
            for row in schedule.group_by("season")
            .len()
            .iter_rows(named=True)
        }
        with_odds = {
            int(row["season"]): int(row["len"])
            for row in schedule.filter(pl.col("spread_line").is_not_null())
            .group_by("season")
            .len()
            .iter_rows(named=True)
        }
        predicted: dict[int, int] = {}
        for prediction in predictions:
            predicted[prediction.season] = (
                predicted.get(prediction.season, 0) + 1
            )
        skipped_games: dict[int, int] = {}
        for skip in skipped_chunks:
            games = schedule.filter(
                (pl.col("season") == skip.season)
                & (pl.col("week").is_in(skip.weeks))
            ).height
            skipped_games[skip.season] = (
                skipped_games.get(skip.season, 0) + games
            )

        self.predicted = len(predictions)
        self.skipped_chunks = list(skipped_chunks)
        self.imputed_feature_cells = sum(season_imputed.values())
        self.coverage = [
            SeasonCoverage(
                season=season,
                scheduled=scheduled[season],
                scheduled_with_odds=with_odds.get(season, 0),
                predicted=predicted.get(season, 0),
                skipped_missing_features=season_missing.get(season, 0),
                skipped_chunk_games=skipped_games.get(season, 0),
                imputed_feature_cells=season_imputed.get(season, 0),
            )
            for season in sorted(scheduled)
        ]


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
    train_df: pl.DataFrame,
    model_cfg: ModelConfig,
    sample_weight: Optional[np.ndarray] = None,
) -> RatingModelWeights:
    X, y = prepare_training_data(train_df, model_cfg)
    if len(X) == 0:
        raise ValueError("no training data available")
    model = RatingModel(model_cfg)
    model.fit(X, y, sample_weight=sample_weight)
    reg_w, class_w = model.get_weights()
    # Real metrics would require held-out data; for walk-forward we
    # score on the next chunk, so leave these blank. nflbetting's
    # RatingModelWeights requires the fields so we fill with zeros.
    from nflbetting.model.rating_model.defs import (
        RegressionMetrics,
        ClassificationMetrics,
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


def _build_decay_sample_weight(
    train_df: pl.DataFrame,
    test_start_season: int,
    test_start_week: int,
    halflife_weeks: float,
) -> Optional[np.ndarray]:
    """Build per-row sample weights for Stage 9 decay-weighted
    seasons.

    Returns ``None`` when ``halflife_weeks <= 0`` (no decay; the
    pre-Stage-9 behavior is unchanged). Otherwise returns a
    1-D ``np.ndarray`` of weights, one per row in
    ``train_df``, computed by
    ``nflbetting.model.weighting.decay_weights`` so the framework
    and the library's ``train_period_model`` produce identical
    fits given the same config.
    """
    if halflife_weeks <= 0:
        return None
    if "game_id" not in train_df.columns:
        return None
    from nflbetting.model.weighting import decay_weights

    game_id_str = train_df["game_id"].cast(pl.Utf8)
    game_year = game_id_str.str.slice(0, 4).cast(pl.Int64)
    game_week = game_id_str.str.slice(5, 2).cast(pl.Int64)
    game_week_index = game_year * 100 + game_week
    cutoff_index = test_start_season * 100 + test_start_week
    age_indices = (cutoff_index - game_week_index).to_numpy()
    age_weeks = np.maximum(age_indices.astype(float) * 18.0, 0.0)
    return decay_weights(age_weeks, halflife_weeks)


def _is_missing(value) -> bool:
    if value is None:
        return True
    return isinstance(value, float) and value != value


def _neutralize_missing_features(
    home_row: dict,
    away_row: dict,
    feature_names: list[str],
) -> int:
    """Make a feature with a missing side contribute nothing.

    ``predict_outcome`` subtracts the two rows, which raises on
    ``None``. Replacing both sides with the present value (or 0.0)
    makes that feature contribute nothing to the prediction.

    Returns the number of feature cells that had a missing side.
    """
    imputed = 0
    for name in feature_names:
        home_value = home_row.get(name)
        away_value = away_row.get(name)
        if _is_missing(home_value) or _is_missing(away_value):
            if not _is_missing(home_value):
                fallback = home_value
            elif not _is_missing(away_value):
                fallback = away_value
            else:
                fallback = 0.0
            home_row[name] = fallback
            away_row[name] = fallback
            imputed += 1
    return imputed


def _predict_chunk(
    features: pl.DataFrame,
    schedule: pl.DataFrame,
    weights: RatingModelWeights,
    model_cfg: ModelConfig,
) -> tuple[list[Prediction], int, int]:
    """Predict a test chunk.

    Returns ``(predictions, missing_feature_games, imputed_cells)``.
    ``missing_feature_games`` counts schedule games skipped because no
    feature row existed; ``imputed_cells`` counts feature cells whose
    home or away side was missing and was neutralized.
    """
    preds: list[Prediction] = []
    missing_feature_games = 0
    imputed_cells = 0
    schedule_games = schedule.select(
        [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "spread_line",
        ]
    ).to_dicts()
    feature_groups = {
        g[0][0]: g[1] for g in features.partition_by("game_id", as_dict=True).items()
    }
    for game_row in schedule_games:
        gid = game_row["game_id"]
        if gid not in feature_groups:
            missing_feature_games += 1
            continue
        game_features = feature_groups[gid]
        home_mask = (pl.col("team_id") == game_row["home_team"]) & (
            pl.col("location") == "home"
        )
        away_mask = (pl.col("team_id") == game_row["away_team"]) & (
            pl.col("location") == "away"
        )
        try:
            home_row = game_features.filter(home_mask).to_dicts()[0]
            away_row = game_features.filter(away_mask).to_dicts()[0]
        except IndexError:
            missing_feature_games += 1
            continue
        imputed_cells += _neutralize_missing_features(
            home_row, away_row, list(model_cfg.feature_names)
        )
        predicted_margin, _, win_prob = predict_outcome(
            home_row, away_row, weights, model_cfg
        )
        home_score = game_row.get("home_score")
        away_score = game_row.get("away_score")
        if (
            home_score is not None
            and away_score is not None
            and not (isinstance(home_score, float) and np_isnan(home_score))
            and not (isinstance(away_score, float) and np_isnan(away_score))
        ):
            actual_home_score = int(home_score)
            actual_away_score = int(away_score)
            actual_margin = actual_home_score - actual_away_score
        else:
            actual_home_score = actual_away_score = actual_margin = None
        spread = game_row.get("spread_line")
        if spread is not None and (isinstance(spread, float) and np_isnan(spread)):
            spread = None
        preds.append(
            Prediction(
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
            )
        )
    return preds, missing_feature_games, imputed_cells


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
    decay_halflife_weeks: float = 0.0,
    strict: bool = False,
    diagnostics: Optional[WalkForwardDiagnostics] = None,
) -> list[Prediction]:
    """Rolling walk-forward: train on the last N weeks, predict next chunk.

    ``decay_halflife_weeks`` enables Stage 9 decay-weighted
    seasons. ``0`` (default) reproduces the pre-Stage-9
    behavior byte-for-byte; positive values build per-row
    sample weights via
    ``nflbetting.model.weighting.decay_weights`` and forward
    them to ``RatingModel.fit``.

    A chunk that raises is recorded (never silently dropped) and the
    run continues. ``strict=True`` re-raises instead. When
    ``diagnostics`` is supplied it is populated with per-season
    coverage and the skipped chunks.
    """
    features = features.sort(["season", "week", "game_id"])
    all_preds: list[Prediction] = []
    skipped_chunks: list[ChunkSkip] = []
    season_missing: dict[int, int] = {}
    season_imputed: dict[int, int] = {}
    seasons = sorted(features["season"].unique().to_list())
    for season in seasons:
        sched_season = schedule.filter(pl.col("season") == season)
        if sched_season.is_empty():
            continue
        weeks = sorted(sched_season["week"].unique().to_list())
        for i in range(0, len(weeks), test_weeks):
            chunk = weeks[i : i + test_weeks]
            if not chunk:
                continue
            test_start_w = chunk[0]
            test_end_w = chunk[-1]
            train = _trim_to_last_n_weeks(
                features, season, test_start_w, training_weeks
            )
            if train.is_empty():
                skipped_chunks.append(
                    ChunkSkip(
                        int(season), [int(w) for w in chunk], "no_training_data"
                    )
                )
                continue
            test_features = features.filter(
                (pl.col("season") == season)
                & (pl.col("week") >= test_start_w)
                & (pl.col("week") <= test_end_w)
            )
            if "margin" in test_features.columns:
                test_features = test_features.drop("margin")
            if test_features.is_empty():
                skipped_chunks.append(
                    ChunkSkip(
                        int(season), [int(w) for w in chunk], "no_test_features"
                    )
                )
                continue
            test_sched = sched_season.filter(
                (pl.col("week") >= test_start_w) & (pl.col("week") <= test_end_w)
            )
            sample_weight = _build_decay_sample_weight(
                train,
                season,
                test_start_w,
                decay_halflife_weeks,
            )
            try:
                weights = _train_and_collect_weights(
                    train,
                    model_cfg,
                    sample_weight=sample_weight,
                )
                chunk_preds, missing, imputed = _predict_chunk(
                    test_features, test_sched, weights, model_cfg
                )
            except Exception as exc:
                if strict:
                    raise
                logger.warning(
                    "walk-forward chunk skipped: season=%s weeks=%s: %s",
                    season,
                    chunk,
                    exc,
                )
                skipped_chunks.append(
                    ChunkSkip(int(season), [int(w) for w in chunk], repr(exc))
                )
                continue
            all_preds.extend(chunk_preds)
            season_missing[int(season)] = (
                season_missing.get(int(season), 0) + missing
            )
            season_imputed[int(season)] = (
                season_imputed.get(int(season), 0) + imputed
            )
    if diagnostics is not None:
        diagnostics.populate(
            predictions=all_preds,
            schedule=schedule,
            skipped_chunks=skipped_chunks,
            season_missing=season_missing,
            season_imputed=season_imputed,
        )
    return all_preds


def train_period(
    features: pl.DataFrame,
    schedule: pl.DataFrame,
    period: TrainingPeriod,
    model_cfg: ModelConfig,
    *,
    decay_halflife_weeks: float = 0.0,
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
    sample_weight = _build_decay_sample_weight(
        train,
        period.test_start[0],
        period.test_start[1],
        decay_halflife_weeks,
    )
    model = _train_and_collect_weights(train, model_cfg, sample_weight=sample_weight)
    predictions, _, _ = _predict_chunk(test, test_sched, model, model_cfg)
    return predictions
