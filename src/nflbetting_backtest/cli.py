"""nflbt-run: walk-forward validation entry point."""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

from .backtest import walk_forward_validation
from .datasets import DATASET_FULL, DATASET_2025, Dataset
from .metrics import aggregate_metrics
from .report import (
    write_metrics_txt,
    write_predictions_xlsx,
    write_predictions_jsonl,
    write_predictions_pkl,
)
from .run_card import build_run_card, copy_config, write_run_card


CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "NFL-Model-UPDATED" / "config.yaml"
)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _resolve_dataset(name_or_path: str) -> Dataset:
    """'DATASET_FULL' / 'DATASET_2025' / a directory with parquet files."""
    if name_or_path == "DATASET_FULL":
        return DATASET_FULL
    if name_or_path == "DATASET_2025":
        return DATASET_2025
    p = Path(name_or_path)
    if p.is_dir():
        return Dataset(
            name=p.name,
            features=pl.read_parquet(p / "features.parquet"),
            schedule=pl.read_parquet(p / "schedule.parquet"),
        )
    raise SystemExit(f"Unknown dataset: {name_or_path}")


def _build_model_config(cfg, dataset: Dataset):
    from nflbetting.model.rating_model.defs import ModelConfig
    return ModelConfig(
        model_type=cfg.model.regularization,
        alpha=cfg.model.alpha,
        l1_ratio=cfg.model.l1_ratio,
        scale_features=cfg.model.scale_features,
        test_split=cfg.model.test_split,
        cv_splits=cfg.model.cv_splits,
        feature_names=dataset.feature_names,
    )


def _make_run_dir(out_root: Path, dataset_name: str) -> tuple[Path, datetime]:
    """Create a timestamped run subdirectory: out_root/<dataset>_YYYYMMDDTHHMMSS."""
    now = datetime.now()
    slug = now.strftime("%Y%m%dT%H%M%S")
    safe_name = dataset_name.replace("/", "_")
    run_dir = out_root / f"{safe_name}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, now


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="nflbt-run",
        description="Walk-forward validation of the nflbetting model.",
    )
    parser.add_argument(
        "dataset",
        help="DATASET_FULL, DATASET_2025, or a directory with features.parquet+schedule.parquet",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports"),
        help="Root directory; each run gets a timestamped subdirectory inside.",
    )
    parser.add_argument(
        "--training-weeks", type=int, default=None,
        help="Override cfg.model.training_weeks.",
    )
    parser.add_argument(
        "--test-weeks", type=int, default=None,
        help="Override cfg.model.test_weeks.",
    )
    parser.add_argument(
        "--config", type=Path, default=CONFIG_PATH,
        help="Path to nflbetting config.yaml.",
    )
    args = parser.parse_args(argv)

    from nflbetting.config import load_config

    cfg = load_config(args.config)
    dataset = _resolve_dataset(args.dataset)
    model_cfg = _build_model_config(cfg, dataset)

    training_weeks = args.training_weeks or cfg.model.training_weeks
    test_weeks = args.test_weeks or cfg.model.test_weeks

    preds = walk_forward_validation(
        dataset.features, dataset.schedule, model_cfg,
        training_weeks=training_weeks,
        test_weeks=test_weeks,
    )

    out_root = args.out.resolve()
    run_dir, run_ts = _make_run_dir(out_root, dataset.name)

    # Write outputs
    write_metrics_txt(aggregate_metrics(preds), run_dir / "metrics.txt")
    write_predictions_xlsx(
        preds, aggregate_metrics(preds), run_dir / "predictions.xlsx",
        features=dataset.features,
    )
    write_predictions_jsonl(preds, run_dir / "predictions.jsonl")
    write_predictions_pkl(preds, run_dir / "predictions.pkl")
    copy_config(args.config, run_dir / "config.yaml.copy")

    # Run card (built AFTER all writes so n_predictions is accurate)
    report = aggregate_metrics(preds)
    card = build_run_card(
        args=args,
        dataset_name=dataset.name,
        cfg=cfg,
        model_cfg=model_cfg,
        training_weeks=training_weeks,
        test_weeks=test_weeks,
        feature_names=dataset.feature_names,
        n_predictions=len(preds),
        out_dir=run_dir,
        repo_root=REPO_ROOT,
        timestamp=run_ts,
    )
    write_run_card(card, run_dir / "run_card.json")

    print(
        f"wrote {len(preds)} predictions + metrics + xlsx + run_card to {run_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())