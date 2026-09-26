"""nflbt-run: walk-forward validation for both bundled datasets, one directory per run."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .backtest import WalkForwardDiagnostics, walk_forward_validation
from .datasets import (
    DATASET_FULL,
    DATASET_2025,
    Dataset,
    load_dataset_with_features,
)
from .metrics import aggregate_metrics
from .regen_features import regenerate_features_for_dataset
from .report import (
    write_metrics_txt,
    write_predictions_xlsx,
    write_predictions_pkl,
    write_comparison_report,
    write_coverage_txt,
)
from .run_card import build_run_card, copy_config, write_run_card

CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "NFL-Model-UPDATED"
    / "config.yaml"
)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LIBRARY_ROOT = REPO_ROOT / "NFL-Model-UPDATED"


def _resolve_data_dir(cfg) -> None:
    """Force ``cfg.data_dir`` to an absolute path under
    ``LIBRARY_ROOT``.

    The library's feature pipeline reads files like
    ``stadiums.csv`` and ``history_pbp.parquet`` relative to
    ``cfg.data_dir``. When the framework's CLI is invoked from
    a different cwd (e.g. an installed ``nflbt-run`` in a venv
    bin/), ``./data`` resolves to the wrong directory. We
    normalize ``cfg.data_dir`` to ``<LIBRARY_ROOT>/<cfg.data_dir>``
    so the pipeline always finds the library's data files
    regardless of where the framework was invoked from.

    Mutates ``cfg`` in place; returns ``None``. The
    normalization is idempotent: a path that is already
    absolute is returned unchanged.
    """
    raw = Path(cfg.data_dir)
    if raw.is_absolute():
        return
    cfg.data_dir = str(LIBRARY_ROOT / raw)


def _resolve_dataset(
    name: str,
    *,
    cfg=None,
    run_dir: Path | None = None,
    use_regenerated: bool = False,
) -> Dataset:
    """Return a Dataset, optionally with library-regenerated features.

    - ``use_regenerated=True`` calls
      :func:`regenerate_features_for_dataset` for the dataset's
      schedule range and writes the parquet under
      ``run_dir/features/<dataset>/features.parquet``. The
      returned Dataset wraps that parquet.
    - ``use_regenerated=False`` returns the bundled Dataset.
    """
    if not use_regenerated:
        if name == "DATASET_FULL":
            return DATASET_FULL
        if name == "DATASET_2025":
            return DATASET_2025
        raise SystemExit(f"Unknown dataset: {name}")

    # Regenerate from the library using the supplied config.
    if cfg is None or run_dir is None:
        raise ValueError("regenerated features require cfg and run_dir")
    base = DATASET_FULL if name == "DATASET_FULL" else DATASET_2025
    print(
        f"regenerating features for {name} from library "
        f"({len(base.schedule['season'].unique())} seasons)...",
        file=sys.stderr,
    )
    df = regenerate_features_for_dataset(cfg, base.schedule)
    out_dir = run_dir / "features" / name.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features.parquet"
    df.write_parquet(str(out_path))
    print(
        f"  -> {out_path} ({df.height} rows, {df.width} cols)",
        file=sys.stderr,
    )
    return load_dataset_with_features(name, out_path)


def _build_model_config(cfg, dataset: Dataset, feature_names: list[str]):
    """Build the framework ``ModelConfig`` from the app config.

    The framework does not own feature engineering (Stage 5/6/7
    settings flow through the *bundled feature parquets* that
    the library pre-computes). The framework only needs to
    record them in the run card (audit trail). What the
    framework *does* own is the fit path, which now accepts
    ``sample_weight`` for Stage 9; that is handled in
    ``backtest._train_and_collect_weights`` rather than here.
    """
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

    diagnostics = WalkForwardDiagnostics()
    preds = walk_forward_validation(
        dataset.features,
        dataset.schedule,
        model_cfg,
        training_weeks=training_weeks,
        test_weeks=test_weeks,
        decay_halflife_weeks=cfg.model.decay_halflife_weeks,
        strict=args.strict,
        diagnostics=diagnostics,
    )

    report = aggregate_metrics(preds)

    write_metrics_txt(report, run_dir / f"{dataset.name}.metrics.txt")
    write_coverage_txt(
        diagnostics,
        run_dir / f"{dataset.name}.coverage.txt",
        label=dataset.name,
    )
    write_predictions_xlsx(
        preds,
        report,
        run_dir / f"{dataset.name}.predictions.xlsx",
        features=dataset.features,
    )
    write_predictions_pkl(preds, run_dir / f"{dataset.name}.predictions.pkl")
    copy_config(args.config, run_dir / f"{dataset.name}.config.yaml.copy")

    # Stage 10: write a filtered metrics report + pkl + xlsx
    # when the caller passes any --filter-* flag. Filtering
    # happens via the library's
    # ``nflbetting.predictions.filter_by_edge`` predicate so the
    # same predicate applies to the live ``refresh.py`` path.
    if (
        args.filter_edge is not None
        or args.filter_prob is not None
        or args.filter_spread is not None
    ):
        from nflbetting.predictions import filter_by_edge
        import polars as _pl

        preds_df = _pl.DataFrame(
            [
                {
                    "game_id": p.game_id,
                    "predicted_margin": p.predicted_margin,
                    "predicted_win_probability": p.predicted_win_probability,
                    "spread_line": p.spread_line,
                }
                for p in preds
            ]
        )
        filtered_df = filter_by_edge(
            preds_df,
            min_edge=args.filter_edge,
            min_prob=args.filter_prob,
            max_spread=args.filter_spread,
        )
        keep_ids = set(filtered_df["game_id"].to_list())
        filtered_preds = [p for p in preds if p.game_id in keep_ids]
        filtered_report = aggregate_metrics(filtered_preds)
        write_metrics_txt(
            filtered_report,
            run_dir / f"{dataset.name}.metrics_filtered.txt",
        )
        write_predictions_xlsx(
            filtered_preds,
            filtered_report,
            run_dir / f"{dataset.name}.predictions_filtered.xlsx",
            features=dataset.features,
        )
        write_predictions_pkl(
            filtered_preds,
            run_dir / f"{dataset.name}.predictions_filtered.pkl",
        )
        print(
            f"{dataset.name}: filter kept {len(filtered_preds)}/"
            f"{len(preds)} predictions -> "
            f"{run_dir}/{dataset.name}.*filtered.*",
            file=sys.stderr,
        )

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
        features_source.features,
        DATASET_2025.schedule,
        model_cfg,
        training_weeks=training_weeks,
        test_weeks=test_weeks,
        decay_halflife_weeks=cfg.model.decay_halflife_weeks,
        strict=args.strict,
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
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to nflbetting config.yaml.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise on a failed walk-forward chunk instead of recording "
        "it and continuing (default: record and continue).",
    )
    parser.add_argument(
        "--filter-edge",
        type=float,
        default=None,
        help="Stage 10 live-skip criteria: minimum |model_spread - "
        "market_spread|, in points. When set (with or without "
        "--filter-prob / --filter-spread), the framework writes "
        "a second metrics_filtered.txt / predictions_filtered.* "
        "set per dataset using only the rows that pass the "
        "filter. Default: no filter.",
    )
    parser.add_argument(
        "--filter-prob",
        type=float,
        default=None,
        help="Stage 10 live-skip criteria: minimum model confidence "
        "max(p, 1-p). Default: no filter.",
    )
    parser.add_argument(
        "--filter-spread",
        type=float,
        default=None,
        help="Stage 10 live-skip criteria: maximum |market_spread| "
        "in points. Default: no filter.",
    )
    regen_group = parser.add_mutually_exclusive_group()
    regen_group.add_argument(
        "--regenerate-features",
        dest="regenerate_features",
        action="store_true",
        default=None,
        help="Force per-run library feature regeneration "
        "(overrides cfg.regenerate_features). Default behavior "
        "follows cfg.regenerate_features (default True).",
    )
    regen_group.add_argument(
        "--no-regenerate-features",
        dest="regenerate_features",
        action="store_false",
        default=None,
        help="Use the bundled feature parquet (v1 baseline). "
        "Required for byte-for-byte reproduction of the v1 "
        "baseline; ignored if --features-source is also set.",
    )
    parser.add_argument(
        "--features-source",
        type=Path,
        default=None,
        help="Path to a pre-regenerated features parquet. When "
        "set, skips both library regeneration and the "
        "bundled parquet; takes precedence over "
        "--regenerate-features / --no-regenerate-features.",
    )
    args = parser.parse_args(argv)

    from nflbetting.config import load_config

    cfg = load_config(args.config)
    training_weeks = cfg.model.training_weeks
    test_weeks = cfg.model.test_weeks

    # Resolve the feature-source policy. Explicit CLI flags
    # take precedence over cfg.regenerate_features; an explicit
    # --features-source overrides both.
    if args.features_source is not None:
        regenerate_features = False
        features_source_override = args.features_source
    elif args.regenerate_features is not None:
        regenerate_features = args.regenerate_features
        features_source_override = None
    else:
        regenerate_features = bool(getattr(cfg, "regenerate_features", True))
        features_source_override = None

    run_dir, _run_ts = _make_run_dir(args.out.resolve())

    # 1. The two main per-dataset walk-forwards (full history vs 2025).
    results: dict[str, tuple] = {}
    dataset_by_name: dict[str, Dataset] = {}
    # Normalize data_dir to the library's data directory so the
    # regeneration path (and any future feature pipeline calls
    # by the framework) always find stadiums.csv / history_*.parquet
    # regardless of where the framework was invoked from.
    _resolve_data_dir(cfg)
    for dataset_name in ("DATASET_FULL", "DATASET_2025"):
        if features_source_override is not None:
            dataset = load_dataset_with_features(dataset_name, features_source_override)
        else:
            dataset = _resolve_dataset(
                dataset_name,
                cfg=cfg,
                run_dir=run_dir,
                use_regenerated=regenerate_features,
            )
        dataset_by_name[dataset_name] = dataset
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
    # When the framework regenerated features per run, both
    # ``dataset_by_name`` entries point at the regenerated parquet,
    # so the comparison exercises the live config rather than the
    # bundled v1 artifact.
    full_dataset = dataset_by_name["DATASET_FULL"]
    short_dataset = dataset_by_name["DATASET_2025"]
    full_feature_names = list(full_dataset.feature_names)
    short_feature_names = list(short_dataset.feature_names)

    full_on_2025_preds, full_on_2025_report = _run_pass_on_2025(
        features_source=full_dataset,
        feature_names=full_feature_names,
        cfg=cfg,
        args=args,
        training_weeks=training_weeks,
        test_weeks=test_weeks,
        label="full-features on 2025 schedule",
    )
    short_on_2025_preds, short_on_2025_report = _run_pass_on_2025(
        features_source=short_dataset,
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
