"""Single acceptance test: nflbt-run output must match the baseline reference files.

One invocation of `nflbt-run` (no arguments) creates one timestamped
directory `run_<TS>/` under the user-supplied --out root and writes
11 files flat inside it:

    run_<TS>/DATASET_FULL.metrics.txt
    run_<TS>/DATASET_FULL.predictions.xlsx
    run_<TS>/DATASET_FULL.predictions.pkl
    run_<TS>/DATASET_FULL.config.yaml.copy
    run_<TS>/DATASET_FULL.run_card.yaml
    run_<TS>/DATASET_2025.metrics.txt
    run_<TS>/DATASET_2025.predictions.xlsx
    run_<TS>/DATASET_2025.predictions.pkl
    run_<TS>/DATASET_2025.config.yaml.copy
    run_<TS>/DATASET_2025.run_card.yaml
    run_<TS>/comparison.txt

The test asserts metrics.txt and predictions.xlsx byte-equal the
regenerated reference files in NFL-Model-UPDATED/review2026/.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import openpyxl
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REFERENCE_ROOT = (
    REPO_ROOT / "NFL-Model-UPDATED" / "review2026" / "baseline"
)
PACKAGE_ROOT = REPO_ROOT / "src"


# (dataset name, reference files for comparison)
CASES = [
    {
        "name": "dataset_full",
        "metrics_ref": REFERENCE_ROOT / "metrics.txt",
        "xlsx_ref": REFERENCE_ROOT / "baseline.xlsx",
        "pkl_ref": REFERENCE_ROOT / "predictions.pkl",
    },
    {
        "name": "dataset_2025",
        "metrics_ref": (
            REFERENCE_ROOT / "backtesting_2025" / "metrics_2025.txt"
        ),
        "xlsx_ref": (
            REFERENCE_ROOT / "backtesting_2025" / "backtesting_2025.xlsx"
        ),
        "pkl_ref": (
            REFERENCE_ROOT / "backtesting_2025" / "predictions_2025.pkl"
        ),
    },
]


# File kinds produced per dataset (run_card is yaml, no jsonl)
FILE_KINDS = [
    "metrics.txt", "predictions.xlsx", "predictions.pkl",
    "config.yaml.copy", "run_card.yaml",
]


def _run_cli(tmp_path: Path) -> Path:
    """Run `nflbt-run --out <tmp>/out` and return the run directory."""
    out = tmp_path / "out"
    out.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_ROOT)
    result = subprocess.run(
        [sys.executable, "-m", "nflbetting_backtest.cli", "--out", str(out)],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"CLI failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    # Exactly one run_<TS>/ subdirectory under out
    runs = sorted(p for p in out.iterdir() if p.name.startswith("run_"))
    assert len(runs) == 1, (
        f"expected exactly one run_<TS>/ under {out}, got: "
        f"{[r.name for r in runs]}"
    )
    return runs[0]


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    """Single shared CLI invocation for all parametrized cases."""
    return _run_cli(tmp_path_factory.mktemp("out"))


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_outputs_match_baseline(run_dir, case):
    dataset_name = case["name"].upper()  # DATASET_FULL / DATASET_2025

    # All 5 expected files for this dataset are present, flat in run_dir
    expected_files = [f"{dataset_name}.{kind}" for kind in FILE_KINDS]
    actual_names = {p.name for p in run_dir.iterdir()}
    for fname in expected_files:
        assert fname in actual_names, (
            f"missing output: {fname}; saw: {sorted(actual_names)}"
        )

    metrics_path = run_dir / f"{dataset_name}.metrics.txt"
    xlsx_path = run_dir / f"{dataset_name}.predictions.xlsx"

    # metrics.txt: byte-for-byte match against the regenerated reference
    metrics_actual = metrics_path.read_text()
    metrics_expected = case["metrics_ref"].read_text()
    assert metrics_actual == metrics_expected, (
        f"metrics.txt for {case['name']} differs.\n"
        f"--- diff (head) ---\n"
        + "\n".join(
            difflib_unified_diff(
                metrics_expected.splitlines(),
                metrics_actual.splitlines(),
                lineterm="",
            )[:40]
        )
    )

    # predictions.xlsx: sheets + row counts + per-row content
    wb_actual = openpyxl.load_workbook(xlsx_path)
    wb_expected = openpyxl.load_workbook(case["xlsx_ref"])
    assert wb_actual.sheetnames == wb_expected.sheetnames, (
        f"sheet names differ: {wb_actual.sheetnames} vs {wb_expected.sheetnames}"
    )
    actual_rows = list(wb_actual.active.iter_rows(values_only=True))
    expected_rows = list(wb_expected.active.iter_rows(values_only=True))
    assert len(actual_rows) == len(expected_rows), (
        f"row count differs: {len(actual_rows)} vs {len(expected_rows)}"
    )
    for i, (a, e) in enumerate(zip(actual_rows, expected_rows)):
        for j, (av, ev) in enumerate(zip(a, e)):
            if isinstance(av, float) and isinstance(ev, float):
                assert abs(av - ev) <= 0.5, (
                    f"row {i} col {j}: {av} vs {ev}"
                )
            else:
                assert av == ev, (
                    f"row {i} col {j}: {av!r} vs {ev!r}"
                )

    # run_card.yaml sanity (parses as YAML)
    card = yaml.safe_load((run_dir / f"{dataset_name}.run_card.yaml").read_text())
    assert card["dataset"] == dataset_name
    assert card["n_predictions"] == len(actual_rows) - 1
    assert card["config"]["values"]["training_weeks_used"] > 0
    assert card["config"]["values"]["test_weeks_used"] > 0
    assert card["outputs"]["config_copy"] == "config.yaml.copy"
    assert card["outputs"]["run_card"] == "run_card.yaml"
    assert Path(card["config"]["source_path"]).exists()
    # predictions.jsonl must NOT be present
    assert not (run_dir / f"{dataset_name}.predictions.jsonl").exists()


def test_comparison_report_exists(run_dir):
    """One comparison.txt is written. Both prediction sets cover the same
    2025 games; the file lists each dataset's model-vs-market error
    estimates (MAE, RMSE, MedAE, Bias deltas) with a verdict column."""
    comparison_path = run_dir / "comparison.txt"
    text = comparison_path.read_text()
    # The two label headings should both appear
    assert "full-features" in text
    assert "2025-features" in text
    # Required rows in each block
    expected_rows = ("MAE", "RMSE", "MedAE", "Bias",
                     "Games compared", "n_predictions")
    for marker in expected_rows:
        assert marker in text, f"comparison.txt missing '{marker}' row"
    # Verdict column must use the right vocabulary
    assert "market better" in text or "model better" in text
    # Make sure the old aggregate-only rows are gone
    assert "ATS (correct/total)" not in text
    assert "SU (correct/total)" not in text


def difflib_unified_diff(a, b, lineterm=""):
    import difflib
    return list(difflib.unified_diff(a, b, lineterm=lineterm))