
import logging
from pathlib import Path
import re
from typing import cast
import pandas as pd
import numpy as np

from .backtest_exec import simulate_strategy
from .eval_decile_ic import assign_deciles_time_series, decile_performance, rolling_ic
from .io_data import save_json
from .metrics import daily_level_metrics, merge_metrics, trade_level_metrics
from .pipeline_utils import plot_backtest_report
from .quantiles import compute_expanding_quantiles
from .config import QS, MIN_HISTORY, SIGNAL_COL, LABEL_COL, OPEN_COL, CLOSE_COL, TIME_COL
from .preprocess import winsorize_series

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PRED_DIR = Path("outputs/pred_result")
OUTPUT_BASE = Path("outputs/batch_backtest")
LABEL_IS_PATH = Path("data/label_is.csv")
LABEL_OOS_PATH = Path("data/label_oos.csv")

def get_model_files():
    """Scan prediction directory and group files by model."""
    files = list(PRED_DIR.glob("*.csv"))
    models = {}

    for p in files:
        name = p.name
        segment = None
        model = None
        
        # Pattern 1: is_results_<MODEL>.csv
        if name.startswith("is_results_"):
            segment = "is"
            model = name.replace("is_results_", "").replace(".csv", "")
        elif name.startswith("oos_results_"):
            segment = "oos"
            model = name.replace("oos_results_", "").replace(".csv", "")
        # Pattern 2: <MODEL>_pred_is.csv
        elif name.endswith("_pred_is.csv"):
            segment = "is"
            model = name.replace("_pred_is.csv", "").replace("pred_", "") # Handle edge case if needed
            # Actually pattern is ridge_pred_is.csv -> model=ridge
            model = name.replace("_pred_is.csv", "")
        elif name.endswith("_pred_oos.csv"):
            segment = "oos"
            model = name.replace("_pred_oos.csv", "")
            
        if model and segment:
            if model not in models:
                models[model] = {}
            models[model][segment] = p
            
    return models

def load_labels(path):
    df = cast(pd.DataFrame, pd.read_csv(path))
    # Ensure time column is datetime
    if TIME_COL not in df.columns:
        # Check if 'date' or similar exists
        for col in df.columns:
            if col.lower() in ['date', 'datetime', 'timestamp']:
                df = df.rename(columns={col: TIME_COL})
                break
    
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL]).dt.normalize()
    else:
        raise ValueError(f"Time column '{TIME_COL}' not found in {path}")
        
    return df

def load_prices():
    """Load raw prices for backtesting."""
    try:
        # Assuming run from workspace root
        path = Path("data/raw/BTCUSD_MAs.csv")
        if not path.exists():
            # Try alternative path if run from src
            path = Path("../data/BTCUSD_MAs.csv")
            
        df = pd.read_csv(path)
        # Ensure time column
        if TIME_COL not in df.columns:
            # Check for common date column names
            for col in df.columns:
                if str(col).lower() in ['date', 'datetime', 'timestamp', 'time']:
                    df = df.rename(columns={col: TIME_COL})
                    break
        df[TIME_COL] = pd.to_datetime(df[TIME_COL]).dt.normalize()
        
        # Check for open/close columns
        # Config says: BTCUSD__open, BTCUSD__close
        # But data might have different names, check io_data.py or file content
        # Tool output showed: BTCUSD__open, BTCUSD__close
        
        # Rename columns to match config expectations
        # Config expects BTCUSD__open, BTCUSD__close
        # Data has open, close
        rename_map = {}
        for c in df.columns:
            if c.lower() == "open":
                rename_map[c] = OPEN_COL
            elif c.lower() == "close":
                rename_map[c] = CLOSE_COL
        
        if rename_map:
            df = df.rename(columns=rename_map)
            
        return df[[TIME_COL, OPEN_COL, CLOSE_COL]]
    except Exception as e:
        logger.error(f"Failed to load price data: {e}")
        return None

def run_backtest_for_model(model, files, df_prices):
    logger.info(f"Processing model: {model}")
    model_out_dir = OUTPUT_BASE / model
    model_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process IS and OOS
    for segment in ["is", "oos"]:
        if segment not in files:
            continue
            
        file_path = files[segment]
        logger.info(f"  Segment: {segment}, File: {file_path}")
        
        # Load labels
        label_path = LABEL_IS_PATH if segment == "is" else LABEL_OOS_PATH
        if not label_path.exists():
            logger.warning(f"Label file not found: {label_path}")
            continue
            
        try:
            # We use df_prices instead of synthetic prices
            # But we still need to filter df_prices to match the segment
            
            df_label = load_labels(label_path)
            # Filter prices to match label range
            # Merge labels with prices
            df_seg = df_label.merge(df_prices, on=TIME_COL, how="inner")
            
            # Load predictions
            df_pred = cast(pd.DataFrame, pd.read_csv(file_path))
            # ... (Time parsing logic) ...
            if TIME_COL not in df_pred.columns:
                # Check for common date column names
                for col in df_pred.columns:
                    if str(col).lower() in ['date', 'datetime', 'timestamp']:
                        df_pred = df_pred.rename(columns={col: TIME_COL})
                        break
                
                # If still not found, check if first column is unnamed and contains dates
                if TIME_COL not in df_pred.columns:
                    first_col = df_pred.columns[0]
                    # Check if unnamed or empty string
                    if str(first_col).startswith("Unnamed") or str(first_col) == "":
                        # Check if values look like dates
                        try:
                            pd.to_datetime(df_pred[first_col].iloc[0])
                            logger.info(f"Inferring time column from unnamed column '{first_col}'")
                            df_pred = df_pred.rename(columns={first_col: TIME_COL})
                        except:
                            pass
            
            if TIME_COL not in df_pred.columns:
                logger.error(f"Time column not found in {file_path}. Columns: {df_pred.columns}")
                continue

            df_pred[TIME_COL] = pd.to_datetime(df_pred[TIME_COL]).dt.normalize()
            
            # ... (Signal column logic) ...
            # Rename signal column if needed
            # Expected names: 'signal' or 'forecast_ret'
            # Check for common signal names
            signal_candidates = [SIGNAL_COL, 'forecast_ret', 'pred', 'prediction', 'y_pred']
            found_signal = False
            for cand in signal_candidates:
                if cand in df_pred.columns:
                    df_pred = df_pred.rename(columns={cand: SIGNAL_COL})
                    found_signal = True
                    break
            
            if not found_signal:
                # Try to find a float column that is not time or actual_ret or label
                candidates = [c for c in df_pred.columns if c not in [TIME_COL, 'actual_ret', 'label', 'Unnamed: 0']]
                # Filter for numeric columns
                numeric_candidates = []
                for c in candidates:
                    if pd.api.types.is_numeric_dtype(df_pred[c]):
                        numeric_candidates.append(c)
                
                if len(numeric_candidates) == 1:
                    logger.info(f"Inferring signal column as '{numeric_candidates[0]}'")
                    df_pred = df_pred.rename(columns={numeric_candidates[0]: SIGNAL_COL})
                elif len(numeric_candidates) > 1:
                     # Heuristic: pick the one with 'ret' or 'sig' in name
                    best_cand = None
                    for c in numeric_candidates:
                        if 'ret' in c.lower() or 'sig' in c.lower():
                            best_cand = c
                            break
                    if best_cand:
                        logger.info(f"Inferring signal column as '{best_cand}'")
                        df_pred = df_pred.rename(columns={best_cand: SIGNAL_COL})
                    else:
                        logger.error(f"Ambiguous signal columns {numeric_candidates} in {file_path}")
                        continue
                else:
                    logger.error(f"Could not identify signal column in {file_path}")
                    continue
            
            # Merge predictions
            df_seg = df_seg.merge(df_pred[[TIME_COL, SIGNAL_COL]], on=TIME_COL, how="inner")
            
            if df_seg.empty:
                logger.warning(f"No overlapping data for {model} {segment}")
                continue
                
            # Run Backtest Logic (copied from backtest_pipeline.py)
            
            # 0. Winsorize Signal
            df_seg[SIGNAL_COL] = winsorize_series(df_seg[SIGNAL_COL])
            
            # 1. Compute Quantiles
            q_df = compute_expanding_quantiles(
                df_seg[SIGNAL_COL], qs=QS, min_history=MIN_HISTORY
            )
            
            # 2. Decile Analysis
            # NOTE: For Decile Analysis to match Strategy (Next Day Intraday),
            # we should technically use Next Day Return as the target.
            # Label in label_is.csv is currently: Open[t+1]/Open[t] - 1 (based on previous logic)
            # OR whatever the user provided in label_is.csv.
            # If we want Decile plot to match Strategy, we should calculate the target return:
            # Target = Close[t+1] / Open[t+1] - 1
            # We can calculate this from df_prices.
            
            df_seg["target_ret"] = df_seg[CLOSE_COL].shift(-1) / df_seg[OPEN_COL].shift(-1) - 1.0
            
            deciles = assign_deciles_time_series(
                df_seg[SIGNAL_COL], n_bins=10, min_history=MIN_HISTORY
            )
            decile_summary = decile_performance(
                df_seg.assign(decile=deciles), "decile", "target_ret" # Use new target
            )
            decile_summary.to_csv(
                model_out_dir / f"{segment}_decile_summary.csv", index=False
            )
            
            # 3. Rolling IC
            # Use target_ret for IC as well to be consistent
            ic_series, ic_summary = rolling_ic(
                df_seg, SIGNAL_COL, "target_ret", window=MIN_HISTORY, method="spearman"
            )
            ic_series.to_csv(model_out_dir / f"{segment}_ic_series.csv")
            save_json(ic_summary, model_out_dir / f"{segment}_ic_summary.json")
            
            # 4. Simulate Strategy
            trades_df, daily_ret = simulate_strategy(
                df_seg,
                time_col=TIME_COL,
                open_col=OPEN_COL,
                close_col=CLOSE_COL, # Pass close col
                signal_col=SIGNAL_COL,
                q_df=q_df,
                min_history=MIN_HISTORY,
                fee_bps=0.0,
            )
            
            trades_df.to_csv(model_out_dir / f"{segment}_trades.csv", index=False)
            daily_ret.to_csv(model_out_dir / f"{segment}_daily_returns.csv")
            
            metrics = merge_metrics(
                trade_level_metrics(trades_df), daily_level_metrics(daily_ret)
            )
            save_json(metrics, model_out_dir / f"{segment}_metrics.json")
            
            report_path = model_out_dir / f"{segment}_report.png"
            plot_backtest_report(
                segment, ic_series, decile_summary, daily_ret, metrics, report_path,
                predictions=df_seg[SIGNAL_COL] # Pass predictions
            )
            logger.info(f"Saved report: {report_path}")
            
        except Exception as e:
            logger.error(f"Error processing {model} {segment}: {e}", exc_info=True)

def main():
    df_prices = load_prices()
    if df_prices is None:
        return

    models = get_model_files()
    logger.info(f"Found {len(models)} models: {list(models.keys())}")
    
    for model, files in models.items():
        run_backtest_for_model(model, files, df_prices)

if __name__ == "__main__":
    main()
