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
nflbt-run DATASET_FULL --out reports
nflbt-run DATASET_2025 --out reports
```

Each invocation creates a timestamped subdirectory
(`reports/DATASET_FULL_20261201T123045/`) containing:

- `metrics.txt` — six-section human-readable report (aggregate,
  by season timing, by season and timing, by edge size, vs market,
  vs market by season)
- `predictions.xlsx` — eight-sheet workbook (predictions + 7 metric
  views)
- `predictions.jsonl` — newline-delimited JSON, one prediction per line
- `predictions.pkl` — pickle of the `Prediction` list
- `config.yaml.copy` — copy of the config used for the run
- `run_card.json` — run metadata (timestamp, command, host, git SHA,
  config SHA256, resolved model hyperparameters, n_predictions, ...)

Default settings are read from `NFL-Model-UPDATED/config.yaml`
(`regularization: linear`, `test_weeks: 1`, `training_weeks: 60`).
Override with `--training-weeks N`, `--test-weeks N`, or `--config PATH`.

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
