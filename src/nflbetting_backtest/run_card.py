"""Run metadata: write a YAML card describing one nflbt-run invocation."""

import argparse
import datetime
import hashlib
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _short_git_sha(cwd: Path) -> Optional[str]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return f"{sha}{'+dirty' if dirty else ''}" if sha else None
    except Exception:
        return None


def _file_sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_run_card(
    *,
    args: argparse.Namespace,
    dataset_name: str,
    cfg,  # AppConfig (from nflbetting.config.load_config)
    model_cfg,  # ModelConfig passed to walk_forward_validation
    training_weeks: int,
    test_weeks: int,
    feature_names: list[str],
    n_predictions: int,
    out_dir: Path,
    repo_root: Path,
    timestamp: datetime.datetime | None = None,
) -> dict[str, Any]:
    """Build a structured metadata dict for one run."""
    config_path = Path(args.config).resolve()
    ts = timestamp or datetime.datetime.now()
    card: dict[str, Any] = {
        "timestamp_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timestamp_slug": ts.strftime("%Y%m%dT%H%M%S"),
        "dataset": dataset_name,
        "out_dir": str(out_dir.resolve()),
        "command": "nflbt-run " + " ".join(sys.argv[1:]),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "executable": sys.executable,
        },
        "git": {
            "repo_root": str(repo_root.resolve()),
            "nfl_model_sha": _short_git_sha(repo_root / "NFL-Model-UPDATED"),
            "nfl_model_backtest_sha": _short_git_sha(repo_root),
        },
        "config": {
            "source_path": str(config_path),
            "source_sha256": _file_sha256(config_path),
            "values": {
                "regularization": cfg.model.regularization,
                "alpha": cfg.model.alpha,
                "l1_ratio": cfg.model.l1_ratio,
                "scale_features": cfg.model.scale_features,
                "test_split": cfg.model.test_split,
                "cv_splits": cfg.model.cv_splits,
                "elo_k_factor": cfg.model.elo_k_factor,
                "elo_start_rating": cfg.model.elo_start_rating,
                "elo_home_advantage": cfg.model.elo_home_advantage,
                "elo_hfa_mode": getattr(cfg.model, "elo_hfa_mode", "fixed"),
                "elo_hfa_fixed": getattr(
                    cfg.model,
                    "elo_hfa_fixed",
                    cfg.model.elo_home_advantage,
                ),
                "shrinkage_prior": getattr(cfg.model, "shrinkage_prior", 0.0),
                "momentum_min_window": getattr(cfg.model, "momentum_min_window", 3),
                "momentum_max_window": getattr(cfg.model, "momentum_max_window", 3),
                "momentum_dispersion_threshold": getattr(
                    cfg.model, "momentum_dispersion_threshold", 1.5
                ),
                "decay_halflife_weeks": getattr(cfg.model, "decay_halflife_weeks", 0.0),
                "training_weeks": cfg.model.training_weeks,
                "test_weeks": cfg.model.test_weeks,
                "training_weeks_used": training_weeks,
                "test_weeks_used": test_weeks,
                "feature_names": feature_names,
                "n_features": len(feature_names),
            },
        },
        "outputs": {
            "metrics_txt": "metrics.txt",
            "predictions_xlsx": "predictions.xlsx",
            "predictions_pkl": "predictions.pkl",
            "config_copy": "config.yaml.copy",
            "run_card": "run_card.yaml",
        },
        "n_predictions": n_predictions,
    }
    return card


def write_run_card(card: dict[str, Any], path: Path) -> None:
    """Write the run card as YAML (order-preserving, human-readable)."""
    try:
        import yaml
    except ImportError as e:
        raise SystemExit(
            "PyYAML is required for run_card.yaml. pip install pyyaml."
        ) from e
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(
            card,
            f,
            sort_keys=False,
            allow_unicode=True,
            width=100,
        )


def copy_config(source: Path, dest: Path) -> str:
    """Copy the config used for the run; return its sha256."""
    import shutil

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return _file_sha256(dest) or ""
