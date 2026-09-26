"""nflbetting-backtest: backtesting for the nflbetting rating model."""

__version__ = "0.1.0"

from .datasets import Dataset, DATASET_FULL, DATASET_2025
from .metrics import (
    Prediction,
    MetricsReport,
    aggregate_metrics,
    ats_outcomes,
    paired_significance,
)
from .backtest import (
    ChunkSkip,
    SeasonCoverage,
    TrainingPeriod,
    WalkForwardDiagnostics,
    train_period,
    walk_forward_validation,
)
from .report import (
    write_comparison_report,
    write_coverage_txt,
    write_metrics_txt,
    write_predictions_pkl,
    write_predictions_xlsx,
)
from .run_card import build_run_card, write_run_card

__all__ = [
    "Dataset",
    "DATASET_FULL",
    "DATASET_2025",
    "Prediction",
    "MetricsReport",
    "aggregate_metrics",
    "ats_outcomes",
    "paired_significance",
    "walk_forward_validation",
    "train_period",
    "TrainingPeriod",
    "ChunkSkip",
    "SeasonCoverage",
    "WalkForwardDiagnostics",
    "write_metrics_txt",
    "write_predictions_xlsx",
    "write_predictions_pkl",
    "write_comparison_report",
    "write_coverage_txt",
    "build_run_card",
    "write_run_card",
]