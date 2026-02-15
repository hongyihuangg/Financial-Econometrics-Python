# BTC Quantitative Trading Backtest Framework

This project provides a complete pipeline for Bitcoin daily trading strategy backtesting, including data processing, signal evaluation, and trading simulation.

## 1. Data Description

The project uses daily feature and label data split into In-Sample (IS) and Out-of-Sample (OOS) periods.

### **File Locations**
All processed data files are located in the `outputs/` directory:
- **Features**: 
  - `outputs/feature_is.csv` (Training data)
  - `outputs/feature_oos.csv` (Testing data)
- **Labels**: 
  - `outputs/label_is.csv` (Training labels)
  - `outputs/label_oos.csv` (Testing labels)
  - *Note: Labels represent the **Next Day Intraday Return** ($Close_{t+1} / Open_{t+1} - 1$).*

### **Time Split**
- **In-Sample (IS)**: `2015-09-01` to `2024-08-30`
- **Out-of-Sample (OOS)**: `2024-09-01` to `2026-01-31`

---

## 2. How to Run the Backtest

The framework is designed to automatically batch process multiple model predictions.

### **Step 1: Prepare Prediction Files**
Place your model's prediction results in the **`outputs/pred_result/`** directory.

**File Naming Convention:**
- For **In-Sample** results: `is_results_<MODEL_NAME>.csv` or `<MODEL_NAME>_pred_is.csv`
- For **Out-of-Sample** results: `oos_results_<MODEL_NAME>.csv` or `<MODEL_NAME>_pred_oos.csv`

**File Format:**
- Must be a CSV file.
- Must contain a **Time column** (e.g., `time`, `date`, `datetime`).
- Must contain a **Signal column** (e.g., `pred`, `signal`, `forecast_ret`).
- *Note: The framework automatically detects these columns.*

### **Step 2: Run the Backtest Command**
Execute the following command in the root directory:

```bash
python -m src.batch_backtest
```

### **Step 3: Check Results**
Results are generated in **`outputs/batch_backtest/<MODEL_NAME>/`**:
- **`*_report.png`**: A comprehensive chart showing:
  - **Rolling IC**: Information Coefficient over time.
  - **Decile Returns**: Average return of 10 signal groups.
  - **Cumulative Return**: Strategy performance curve.
  - **Prediction Distribution**: Histogram of signal values.
- **`*_metrics.json`**: Key metrics (Sharpe Ratio, Win Rate, Max Drawdown).
- **`*_trades.csv`**: Detailed log of every trade executed.

### **Configuration**
To modify backtest parameters (e.g., quantile thresholds, minimum history window), edit **`src/config.py`**:
- `QS`: Quantile thresholds (default: `0.1` for short, `0.9` for long).
- `MIN_HISTORY`: Minimum days required to compute expanding quantiles (default: `30`).

---

## 3. Backtest Methodology

The backtest simulates a realistic **Intraday Trend Following / Reversal Strategy**.

1.  **Signal Processing**:
    - **Winsorization**: Signals are clipped at the 1% and 99% percentiles to remove extreme outliers.
    - **Expanding Quantiles**: Signal thresholds are calculated using an **expanding window** (only using past data) to prevent look-ahead bias.

2.  **Trading Logic**:
    - **Long Entry**: If today's signal > Historical 90% Quantile.
    - **Short Entry**: If today's signal < Historical 10% Quantile.
    - **No Trade**: If signal is between the 10% and 90% quantiles.

3.  **Execution (Intraday)**:
    - **Open**: Enter position at **Next Day's Open Price** ($Open_{t+1}$).
    - **Close**: Exit position at **Next Day's Close Price** ($Close_{t+1}$).
    - **Holding Period**: 1 Day (Intraday). No overnight risk after the trading day.

4.  **Performance Metrics**:
    - **IC (Information Coefficient)**: Rank correlation between Signal($t$) and Return($t+1$).
    - **Decile Analysis**: Partitions signals into 10 equal-sized bins based on rank and calculates the mean return for each bin.
    - **Cumulative Return**: Calculated using **Simple Interest** (sum of daily returns).
