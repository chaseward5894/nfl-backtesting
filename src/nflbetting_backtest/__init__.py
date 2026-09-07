"""nflbetting-backtest: backtesting for the nflbetting rating model."""

__version__ = "0.1.0"

from .datasets import Dataset, DATASET_FULL, DATASET_2025
from .metrics import Prediction, MetricsReport, aggregate_metrics
from .backtest import (
    walk_forward_validation,
    train_period,
    TrainingPeriod,
)
from .report import (
    write_metrics_txt,
    write_predictions_xlsx,
    write_predictions_pkl,
    write_comparison_report,
)
from .run_card import build_run_card, write_run_card

__all__ = [
    "Dataset",
    "DATASET_FULL",
    "DATASET_2025",
    "Prediction",
    "MetricsReport",
    "aggregate_metrics",
    "walk_forward_validation",
    "train_period",
    "TrainingPeriod",
    "write_metrics_txt",
    "write_predictions_xlsx",
    "write_predictions_pkl",
    "write_comparison_report",
    "build_run_card",
    "write_run_card",
]