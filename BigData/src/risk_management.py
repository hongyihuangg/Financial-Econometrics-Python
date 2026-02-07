"""
src/risk_management.py
Debug Version: Checks for empty data and column mismatches.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURATION ---
OUTPUT_DIR = Path("outputs")
FEATURE_OOS = OUTPUT_DIR / "feature_oos.csv"
PRED_OOS = OUTPUT_DIR / "pred_oos.csv"
RISK_REPORT = OUTPUT_DIR / "risk_report.png"

# Settings
STOP_LOSS_PCT = 0.01
VIX_THRESHOLD = 25.0

def apply_risk_management():
    print("\n--- DEBUG: Starting Risk Management ---")
    
    # 1. Load Data
    if not FEATURE_OOS.exists():
        print(f"CRITICAL ERROR: {FEATURE_OOS} not found!")
        return
    if not PRED_OOS.exists():
        print(f"CRITICAL ERROR: {PRED_OOS} not found!")
        return
        
    df_feat = pd.read_csv(FEATURE_OOS)
    df_pred = pd.read_csv(PRED_OOS)
    
    print(f"Loaded Features: {df_feat.shape} rows.")
    print(f"Loaded Predictions: {df_pred.shape} rows.")
    
    # Check Time Format
    print(f"Feature Time Sample: {df_feat['time'].iloc[0]}")
    print(f"Pred Time Sample:    {df_pred['time'].iloc[0]}")

    # 2. Merge
    # We convert to datetime to ensure they match even if string formats differ
    df_feat['time'] = pd.to_datetime(df_feat['time'])
    df_pred['time'] = pd.to_datetime(df_pred['time'])
    
    df = pd.merge(df_feat, df_pred, on='time')
    print(f"Merged DataFrame: {df.shape} rows.")
    
    if df.empty:
        print("CRITICAL ERROR: Merge resulted in 0 rows. Check your dates!")
        return

    # 3. Define Columns (Auto-detect if possible)
    # We look for typical columns from your dataset
    col_open = 'BTCUSD__open'
    col_close = 'BTCUSD__close'
    col_vix = 'TVC_VIX_1D__open'
    
    # Fallback checks
    if col_open not in df.columns:
        print(f"ERROR: Column '{col_open}' not found. Available columns:")
        print(df.columns.tolist())
        return

    # 4. Strategy Logic
    print("Calculating returns...")
    
    # Raw Strategy (Simple Long/Short logic)
    df['ret_raw'] = 0.0
    
    # Long logic (Signal > 0)
    # Check if 'signal' exists
    if 'signal' not in df.columns:
         print("ERROR: 'signal' column missing from predictions.")
         return
         
    # Basic returns
    df['ret_raw'] = np.where(df['signal'] > 0, 
                             (df[col_close] / df[col_open]) - 1, 
                             0)
    
    # 5. VIX Filter
    df['ret_vix'] = df['ret_raw']
    # If VIX column exists, filter
    if col_vix in df.columns:
        mask_high_risk = df[col_vix] > VIX_THRESHOLD
        df.loc[mask_high_risk, 'ret_vix'] = 0.0
        print(f"VIX Filter: Excluded {mask_high_risk.sum()} days.")
    else:
        print(f"WARNING: VIX column '{col_vix}' not found. Skipping VIX filter.")

    # 6. Stop Loss (Simulated)
    # Simple approximation: If Daily Low is < 2% below Open, we assume we hit stop.
    df['ret_sl'] = df['ret_raw']
    col_low = 'BTCUSD__low'
    
    if col_low in df.columns:
        # Long Stop Loss
        # If Low < Open * 0.98, we limit loss to -2%
        stop_price = df[col_open] * (1 - STOP_LOSS_PCT)
        hit_stop = (df[col_low] < stop_price) & (df['signal'] > 0)
        df.loc[hit_stop, 'ret_sl'] = -STOP_LOSS_PCT
        print(f"Stop Loss: Triggered {hit_stop.sum()} times.")
    
    # 7. Plotting
    print("Generating Plot...")
    df['cum_raw'] = (1 + df['ret_raw']).cumprod()
    df['cum_vix'] = (1 + df['ret_vix']).cumprod()
    df['cum_sl'] = (1 + df['ret_sl']).cumprod()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df['cum_raw'], label='Raw Strategy', color='gray', alpha=0.6)
    plt.plot(df['time'], df['cum_vix'], label='With VIX Filter', color='blue')
    plt.plot(df['time'], df['cum_sl'], label='With Stop Loss', color='green')
    
    plt.title('Risk Management Analysis')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(RISK_REPORT)
    print(f"SUCCESS: Plot saved to {RISK_REPORT}")

if __name__ == "__main__":
    apply_risk_management()