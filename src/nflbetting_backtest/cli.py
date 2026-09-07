"""nflbt-run: walk-forward validation for both bundled datasets, one directory per run."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .backtest import walk_forward_validation
from .datasets import DATASET_FULL, DATASET_2025, Dataset
from .metrics import aggregate_metrics
from .report import (
    write_metrics_txt,
    write_predictions_xlsx,
    write_predictions_pkl,
    write_comparison_report,
)
from .run_card import build_run_card, copy_config, write_run_card


CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "NFL-Model-UPDATED" / "config.yaml"
)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _resolve_dataset(name: str) -> Dataset:
    if name == "DATASET_FULL":
        return DATASET_FULL
    if name == "DATASET_2025":
        return DATASET_2025
    raise SystemExit(f"Unknown dataset: {name}")


def _build_model_config(cfg, dataset: Dataset, feature_names: list[str]):
    from nflbetting.model.rating_model.defs import ModelConfig
    return ModelConfig(
        model_type=cfg.model.regularization,
        alpha=cfg.model.alpha,
        l1_ratio=cfg.model.l1_ratio,
        scale_features=cfg.model.scale_features,
        test_split=cfg.model.test_split,
        cv_splits=cfg.model.cv_splits,
        feature_names=feature_names,
    )


def _make_run_dir(out_root: Path) -> tuple[Path, datetime]:
    """out_root/run_<YYYYMMDDTHHMMSS>/ — one directory per CLI invocation."""
    now = datetime.now()
    slug = now.strftime("%Y%m%dT%H%M%S")
    run_dir = out_root / f"run_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, now


def _run_one_dataset(
    *,
    dataset: Dataset,
    cfg,
    args: argparse.Namespace,
    training_weeks: int,
    test_weeks: int,
    run_dir: Path,
):
    """Train + predict + write outputs for one dataset. Returns (preds, report)."""
    model_cfg = _build_model_config(cfg, dataset, dataset.feature_names)

    preds = walk_forward_validation(
        dataset.features, dataset.schedule, model_cfg,
        training_weeks=training_weeks,
        test_weeks=test_weeks,
    )

    report = aggregate_metrics(preds)

    write_metrics_txt(report, run_dir / f"{dataset.name}.metrics.txt")
    write_predictions_xlsx(
        preds, report, run_dir / f"{dataset.name}.predictions.xlsx",
        features=dataset.features,
    )
    write_predictions_pkl(preds, run_dir / f"{dataset.name}.predictions.pkl")
    copy_config(args.config, run_dir / f"{dataset.name}.config.yaml.copy")

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
        timestamp=datetime.now(),
    )
    write_run_card(card, run_dir / f"{dataset.name}.run_card.yaml")

    print(
        f"{dataset.name}: wrote {len(preds)} predictions -> "
        f"{run_dir}/{dataset.name}.*",
        file=sys.stderr,
    )
    return preds, report


def _run_pass_on_2025(
    *,
    features_source: Dataset,
    feature_names: list[str],
    cfg,
    args: argparse.Namespace,
    training_weeks: int,
    test_weeks: int,
    label: str,
):
    """One walk-forward pass restricted to DATASET_2025.schedule.

    `features_source` provides the historical context (so the rolling
    window has prior weeks to train on); `feature_names` controls
    which columns the model sees (e.g. zeroing out injury/weather).
    """
    model_cfg = _build_model_config(cfg, features_source, feature_names)
    preds = walk_forward_validation(
        features_source.features, DATASET_2025.schedule, model_cfg,
        training_weeks=training_weeks,
        test_weeks=test_weeks,
    )
    report = aggregate_metrics(preds)
    print(
        f"2025-only pass ({label}): wrote {len(preds)} predictions",
        file=sys.stderr,
    )
    return preds, report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="nflbt-run",
        description=(
            "Walk-forward validation of the nflbetting model on both "
            "bundled datasets (DATASET_FULL and DATASET_2025). No "
            "positional arguments required."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports"),
        help="Output directory. Each run gets one timestamped "
             "subdirectory; all files inside are flat.",
    )
    parser.add_argument(
        "--config", type=Path, default=CONFIG_PATH,
        help="Path to nflbetting config.yaml.",
    )
    args = parser.parse_args(argv)

    from nflbetting.config import load_config

    cfg = load_config(args.config)
    training_weeks = cfg.model.training_weeks
    test_weeks = cfg.model.test_weeks

    run_dir, _run_ts = _make_run_dir(args.out.resolve())

    # 1. The two main per-dataset walk-forwards (full history vs 2025).
    results: dict[str, tuple] = {}
    for dataset_name in ("DATASET_FULL", "DATASET_2025"):
        dataset = _resolve_dataset(dataset_name)
        results[dataset_name] = _run_one_dataset(
            dataset=dataset,
            cfg=cfg,
            args=args,
            training_weeks=training_weeks,
            test_weeks=test_weeks,
            run_dir=run_dir,
        )

    # 2. The comparison report: both models scored on the SAME 2025
    # games. Re-run a walk-forward restricted to DATASET_2025.schedule:
    #   a) using DATASET_FULL.features (zeroed impact cols) so the model
    #      sees only the historical baseline signal;
    #   b) using DATASET_2025.features (real overlays) so the model
    #      also sees injury / weather impact.
    # Both predict the same 2025 games; we compare each against market.
    full_feature_names = [
        c for c in DATASET_FULL.feature_names
    ]
    short_feature_names = [
        c for c in DATASET_2025.feature_names
    ]

    full_on_2025_preds, full_on_2025_report = _run_pass_on_2025(
        features_source=DATASET_FULL,
        feature_names=full_feature_names,
        cfg=cfg,
        args=args,
        training_weeks=training_weeks,
        test_weeks=test_weeks,
        label="full-features on 2025 schedule",
    )
    short_on_2025_preds, short_on_2025_report = _run_pass_on_2025(
        features_source=DATASET_2025,
        feature_names=short_feature_names,
        cfg=cfg,
        args=args,
        training_weeks=training_weeks,
        test_weeks=test_weeks,
        label="2025-features on 2025 schedule",
    )

    write_comparison_report(
        full_report=full_on_2025_report,
        short_report=short_on_2025_report,
        full_n=len(full_on_2025_preds),
        short_n=len(short_on_2025_preds),
        path=run_dir / "comparison.txt",
        full_label="full-features",
        short_label="2025-features",
    )
    print(f"comparison -> {run_dir / 'comparison.txt'}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())