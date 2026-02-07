# BTC Data Pipeline and Backtest (Simple Guide)

## Overview
This project builds a merged BTC daily dataset from multiple CSV sources, creates in-sample/out-of-sample splits, and runs a backtest driven by prediction signals. It also exports feature/label CSVs for downstream modeling.

## Data Files (CSV) Meaning
- **feature_is.csv**: In-sample features. One row per day. Includes `time` plus all prefixed source columns (e.g., `BTCUSD__open`, `ETHBTC__close`, etc.).
- **feature_oos.csv**: Out-of-sample features. Same columns as in-sample.
- **label_is.csv**: In-sample labels. Columns: `time`, `label`. Label is next-day open-to-open return: $O_{t+1} / O_t - 1$.
- **label_oos.csv**: Out-of-sample labels. Same definition as in-sample.
- **pred_is.csv / pred_oos.csv**: Prediction signals (one row per day). Columns: `time`, `signal`. Used by the backtest. If missing, the backtest generates random signals and saves them.

## Key Python Modules
- **src/data_pipeline.py**
  - Purpose: Build merged dataset and export IS/OOS feature and label CSVs.
  - Entrypoint: `python -m src.data_pipeline`

- **src/backtest_pipeline.py**
  - Purpose: Run backtests using `pred_is.csv` and `pred_oos.csv` (or generate random predictions if missing). Produces metrics, trades, and report images.
  - Entrypoint: `python -m src.backtest_pipeline`

- **src/pipeline_utils.py**
  - Shared utilities: load/merge data, load or generate predictions, and build report plots.

- **src/backtest_exec.py**
  - Trading simulation logic. Current rule: if signal exceeds historical 90% quantile, enter long at next open and exit after 1 day; if below 10% quantile, enter short for 1 day.

- **src/quantiles.py**
  - Expanding quantile computation (no leakage). Minimum history is 30 days.

- **src/eval_decile_ic.py**
  - Decile performance and rolling IC calculations.

- **src/splits.py**
  - Date split definitions and slicing helpers. OOS is 2024-09-01 to 2026-01-31.

- **src/io_data.py**
  - CSV discovery, loading, and JSON/Parquet saving helpers.

- **src/preprocess.py**
  - Merge sources, create labels, and run basic sanity checks.

- **src/config.py**
  - Global constants (columns, dates, min history, quantile levels).

## How to Run
1) **Generate IS/OOS feature and label CSVs**
- Run: `python -m src.data_pipeline`
- Outputs: `outputs/feature_is.csv`, `outputs/feature_oos.csv`, `outputs/label_is.csv`, `outputs/label_oos.csv`

2) **Run backtest with prediction signals**
- Place prediction files at:
  - `outputs/pred_is.csv`
  - `outputs/pred_oos.csv`
- If missing, the pipeline generates random signals and saves them.
- Run: `python -m src.backtest_pipeline`
- Outputs:
  - `outputs/backtest/is_report.png`, `outputs/backtest/oos_report.png`
  - `outputs/backtest/*_trades.csv`, `*_daily_returns.csv`, `*_metrics.json`
  - `outputs/backtest/*_decile_summary.csv`, `*_ic_series.csv`, `*_ic_summary.json`

## Notes
- All decisions use next-day open price for entry/exit.
- Quantiles and IC are computed within each segment to avoid leakage.
- The backtest raises an error if OOS has no trades.

## Trading Rules (Current)
- Signal thresholds use expanding quantiles computed only from past data within each segment.
- If signal exceeds historical 90% quantile: enter long at next-day open, hold for 1 day, exit next open.
- If signal falls below historical 10% quantile: enter short at next-day open, hold for 1 day, exit next open.
- Minimum history for quantiles is 30 days; during warm-up no trades are opened.

## Metrics Used
**Trade-level:**
- `avg_trade_return`, `total_trade_return` (compounded)
- `long_avg`, `long_total`, `short_avg`, `short_total`
- `avg_holding_days`, `win_rate`, `payoff_ratio`

**Daily-level:**
- `mean_daily`, `std_daily`, `sharpe` (annualized, 365)
- `max_drawdown`
