# nflbetting-backtest

Backtesting for the [`nflbetting`](https://github.com/chaseward5894/NFL-Model)
rating model.

## Install

```bash
pip install -e .
```

This installs `nflbetting` from `../NFL-Model-UPDATED` (editable local
path) plus the bundled parquets in `data/`. No API keys required to
run the core evaluation — only the optional data-recovery scripts
under `utils/` need Open-Meteo / nflreadpy access.

## Quick start

```bash
nflbt-run
```

A single invocation with no arguments runs walk-forward validation on
both bundled datasets (`DATASET_FULL` and `DATASET_2025`) and writes
one timestamped directory under `./reports/` with flat files (no
per-file timestamps):

```
reports/run_<TS>/
├── DATASET_FULL.metrics.txt
├── DATASET_FULL.coverage.txt
├── DATASET_FULL.predictions.xlsx
├── DATASET_FULL.predictions.pkl
├── DATASET_FULL.config.yaml.copy
├── DATASET_FULL.run_card.yaml
├── DATASET_2025.metrics.txt
├── DATASET_2025.coverage.txt
├── DATASET_2025.predictions.xlsx
├── DATASET_2025.predictions.pkl
├── DATASET_2025.config.yaml.copy
├── DATASET_2025.run_card.yaml
└── comparison.txt
```

When the live skip criteria are enabled (see below), each dataset also
writes `metrics_filtered.txt` and `predictions_filtered.{xlsx,pkl}`
covering only the publishable picks.

Each per-dataset file contains:

- `metrics.txt` — six-section human-readable report (aggregate,
  by season timing, by season and timing, by edge size, vs market,
  vs market by season)
- `coverage.txt` — walk-forward coverage: scheduled/predicted games per
  season, imputed feature cells, and any skipped chunks with reasons
- `predictions.xlsx` — eight-sheet workbook (predictions + 7 metric
  views)
- `predictions.pkl` — pickle of the `Prediction` list
- `config.yaml.copy` — copy of the config used for the run
- `run_card.yaml` — run metadata (timestamp, command, host, git SHA,
  config SHA256, resolved model hyperparameters, n_predictions, ...)

The cross-dataset file at the top of the run directory is:

- `comparison.txt` — side-by-side performance (ATS, SU) and error
  (MAE, RMSE, MedAE, Bias, LogLoss) comparison of DATASET_FULL vs
  DATASET_2025, with delta and verdict per metric, plus an
  all-games vs publishable ATS section per dataset

Default settings are read from `NFL-Model-UPDATED/config.yaml`.
Override the config file with `--config PATH`, or the output root
with `--out PATH`.

## Bundled datasets

| Name | Seasons | Injury/weather features |
|---|---|---|
| `DATASET_FULL` | 2009–2026 | zeroed (core Elo + efficiency + momentum + rest/travel only) |
| `DATASET_2025` | 2009–2025 features, 2025 schedule | real for 2022–2025, zeroed for older |

Each is loaded from `data/{name}/features.parquet` + `schedule.parquet`
on import.

## Python API

```python
from nflbetting_backtest import (
    DATASET_FULL, DATASET_2025,
    walk_forward_validation,
    aggregate_metrics,
    write_metrics_txt,
    write_predictions_xlsx,
)
```

## Data-recovery utilities

To refresh the raw parquets, use the library's scripts (e.g.
`NFL-Model-UPDATED/scripts/backfill_injuries.py`,
`NFL-Model-UPDATED/scripts/backfill_weather.py`) or fetch from nflreadpy /
Open-Meteo. Re-adding framework-side entry points would require restoring
`src/nflbetting_backtest/utils/backfill/` and the console scripts.

## Test

`tests/test_outputs_match_baseline.py` runs the CLI on both bundled datasets
and diffs `metrics.txt` / `predictions.xlsx` against the reference files in
`reports/baseline/` (`DATASET_FULL.*`, `DATASET_2025.*`). The backtester is
the source of truth for backtesting; `NFL-Model-UPDATED/review2026/` is dead
information and is not used.

Regenerate the references in `reports/baseline/` from a known-good baseline
run before running the test:

```bash
nflbt-run --no-regenerate-features --out reports
# copy the relevant run_<TS>/ outputs into reports/baseline/ as
# DATASET_FULL.* / DATASET_2025.*
```

```bash
pytest tests/
```
