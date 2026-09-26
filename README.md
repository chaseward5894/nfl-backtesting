# nflbetting-backtest

Backtesting for the [`nflbetting`](https://github.com/chaseward5894/NFL-Model)
rating model. Ships two pre-built historical datasets and runs a
walk-forward validation against each.

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

Default settings are read from `NFL-Model-UPDATED/config.yaml`
(`regularization: linear`, `test_weeks: 1`, `training_weeks: 60`).
Override the config file with `--config PATH`, or the output root
with `--out PATH`.

### Live skip criteria (two ATS reports)

`config.yaml` has a `skip_criteria` block (`enabled`, `min_edge`,
`min_prob`, `max_spread`). When enabled, `nflbt-run` also emits the
filtered report/predictions per dataset and `comparison.txt` shows the
all-games vs publishable ATS gap; the rows kept are exactly those the
live sheet marks `Publishable`. Explicit `--filter-edge` / `--filter-prob`
/ `--filter-spread` flags override the config. See
`NFL-Model-UPDATED/track/live_skip_criteria.md` for the sign convention.

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

```bash
nflbt-backfill-historical-injuries   # nflreadpy -> data/history_injuries.parquet
nflbt-backfill-historical-weather    # Open-Meteo -> data/weather/weather_cache_historical.parquet
nflbt-backfill-2025-injuries         # nflreadpy -> data/injuries/history_injuries_2025.parquet
nflbt-backfill-2025-weather          # Open-Meteo -> data/weather/weather_cache_2025.parquet
```

Each script's `--help` lists its flags.

## Layout

```
nfl-model-backtesting/
├── pyproject.toml
├── README.md
├── PLAN.md / IMPLEMENTATION.md
├── src/nflbetting_backtest/
│   ├── datasets.py            # DATASET_FULL, DATASET_2025
│   ├── backtest.py            # walk_forward_validation, train_period
│   ├── metrics.py             # aggregate_metrics + dataclasses
│   ├── report.py              # write_metrics_txt, write_predictions_xlsx
│   ├── cli.py                 # nflbt-run entry point
│   └── utils/
│       ├── weather.py, injuries.py
│       └── backfill/{historical_weather,historical_injuries,
│                      season_2025_weather,season_2025_injuries}.py
├── data/{dataset_full,dataset_2025}/{features,schedule}.parquet
└── tests/test_outputs_match_baseline.py
```

## Test

The single acceptance test runs the CLI on both bundled datasets and
compares every output against the corresponding reference file in
`NFL-Model-UPDATED/review2026/`:

```bash
pytest tests/
```
