"""
src/train_ridge_bayes.py
Ridge Regression Pipeline with BAYESIAN OPTIMIZATION (Optuna):
1. Loads Data & Fixes Labels (Shift -1)
2. Engineers Features
3. Optimizes Alpha using Optuna (TPE Sampler)
4. Saves Best Model Predictions for Backtest
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import logging

from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path

# --- CONFIGURATION ---
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
PRED_RESULT_DIR = OUTPUT_DIR / "pred_result"
PRED_RESULT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Inputs
FEATURE_IS = DATA_DIR / "feature_is.csv"
FEATURE_OOS = DATA_DIR / "feature_oos.csv"
LABEL_IS = DATA_DIR / "label_is.csv"
LABEL_OOS = DATA_DIR / "label_oos.csv"

# Outputs
REPORT_FILE = OUTPUT_DIR / "oos_report_ridge_bayes.png"
FORECAST_PLOT_FILE = PLOTS_DIR / "ridge_bayes_forecast.png"

# Setup Logging for Optuna
optuna.logging.set_verbosity(optuna.logging.INFO)

def keep_close_only(df):
    df = df.copy()
    drop_cols = [c for c in df.columns if c.endswith(("_open", "_high", "_low"))]
    return df.drop(columns=drop_cols, errors="ignore")

def build_features(df):
    # Same feature logic as original
    df = df.copy()
    tradfi_cols = ["IG_NASDAQ_1D__close", "TVC_VIX_1D__close", "TVC_US10Y_1D__close", "TVC_US02Y_1D__close", "ECONOMICS_USCBBS_1D__close", "FRED_M2SL_1D__close"]
    present_tradfi = [c for c in tradfi_cols if c in df.columns]
    df[present_tradfi] = df[present_tradfi].ffill()
    df = df.ffill()
    
    col_btc_close = "BTCUSD__close"
    
    # Returns / Macro
    if "IG_NASDAQ_1D__close" in df.columns: df["NASDAQ_ret"] = df["IG_NASDAQ_1D__close"].pct_change()
    if "TVC_VIX_1D__close" in df.columns: df["VIX_ret"] = df["TVC_VIX_1D__close"].pct_change()
    
    # Volume
    vol_col = next((c for c in ["BTCUSD__volume", "BTCUSD_volume", "volume"] if c in df.columns), None)
    if vol_col:
        df["VOL_REL_MA"] = (df[vol_col] / df[vol_col].rolling(20).mean()) - 1.0
    elif "TVC_VIX_1D__close" in df.columns:
        df["VOL_REL_MA"] = df["TVC_VIX_1D__close"].pct_change()

    # Term Spread
    if "TVC_US10Y_1D__close" in df.columns and "TVC_US02Y_1D__close" in df.columns:
        df["TERM_10Y_2Y"] = df["TVC_US10Y_1D__close"] - df["TVC_US02Y_1D__close"]
    
    # Technicals
    ma_cols = ["BTCUSD__MA", "BTCUSD__MA.1", "BTCUSD__MA.2", "BTCUSD__MA.3", "BTCUSD__MA.4"]
    present_mas = [c for c in ma_cols if c in df.columns]
    if col_btc_close in df.columns and present_mas:
        for ma in present_mas:
            df[f"DIST_{ma}"] = (df[col_btc_close] / df[ma]) - 1.0
        if len(present_mas) >= 2:
            df["TREND_SLOPE"] = (df[present_mas[0]] - df[present_mas[-1]]) / df[col_btc_close]

    # RSI
    if col_btc_close in df.columns:
        delta = df[col_btc_close].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 0.001)
        df['RSI_14'] = (100 - (100 / (1 + rs))) / 100.0 - 0.5 

    drop_cols = [col_btc_close] + present_tradfi + present_mas + ["ECONOMICS_USCBBS_1D__close", "FRED_M2SL_1D__close", "TVC_US03MY_1D__close"]
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

def prepare_data():
    print("Loading data...")
    df_is = pd.read_csv(FEATURE_IS)
    df_oos = pd.read_csv(FEATURE_OOS)
    lbl_is = pd.read_csv(LABEL_IS)
    lbl_oos = pd.read_csv(LABEL_OOS)
    
    for d in [df_is, df_oos, lbl_is, lbl_oos]:
        d['time'] = pd.to_datetime(d['time'])

    df_is = pd.merge(df_is, lbl_is, on='time', how='inner')
    df_oos = pd.merge(df_oos, lbl_oos, on='time', how='inner')

    df_is['split'] = 'train'
    df_oos['split'] = 'test'
    full_df = pd.concat([df_is, df_oos], axis=0).sort_values('time').reset_index(drop=True)
    full_df = keep_close_only(full_df)
    
    # --- FIX DATA LEAKAGE (Critical) ---
    print("Applying Shift(-1) to fix detected Data Leakage...")
    full_df['label'] = full_df['label'].shift(-1)
    
    full_df = build_features(full_df)
    full_df = full_df.dropna(subset=['label'])
    
    meta_cols = ['time', 'split', 'label', 'BTCUSD__close']
    feature_cols = [c for c in full_df.columns if c not in meta_cols]
    full_df[feature_cols] = full_df[feature_cols].ffill().fillna(0)
    
    train_df = full_df[full_df['split'] == 'train'].drop(columns=['split']).copy()
    test_df = full_df[full_df['split'] == 'test'].drop(columns=['split']).copy()
    return train_df, test_df

# --- OPTUNA OBJECTIVE FUNCTION ---
def objective(trial, X, y):
    # Suggest an alpha value. Log=True explores 0.001 and 0.1 as equally "distant" as 1 and 100
    alpha = trial.suggest_float('alpha', 1e-3, 1e3, log=True)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=alpha))
    ])
    
    # Use TimeSeriesSplit to respect temporal order
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Score using Negative MSE (closest proxy to "minimize error")
    scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    
    # Optuna minimizes by default, but cross_val_score returns negative values for error.
    # We return the MEAN MSE (positive) to minimize.
    return -scores.mean()

def run_bayesian_optimization():
    print(f"\n--- Running Pipeline: RIDGE with BAYESIAN OPTIMIZATION ---")
    train_data, oos_features = prepare_data()
    
    X_train = train_data.drop(columns=['time', 'label'])
    y_train = train_data['label']
    X_oos = oos_features.drop(columns=['time', 'label'], errors='ignore')
    
    # 1. Create Study
    print("Starting Optuna Study (50 Trials)...")
    study = optuna.create_study(direction='minimize') # Minimize MSE
    
    # 2. Optimize
    # We pass X_train and y_train to the objective function using a lambda
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)
    
    print(f"\nBest Alpha Found: {study.best_params['alpha']:.6f}")
    print(f"Best Mean CV MSE: {study.best_value:.6f}")
    
    # 3. Train Final Model with Best Params
    best_alpha = study.best_params['alpha']
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=best_alpha))
    ])
    final_pipeline.fit(X_train, y_train)
    
    # 4. Predict
    pred_is = final_pipeline.predict(X_train)
    pred_oos = final_pipeline.predict(X_oos)
    
    # 5. Save Results (Overwrite ridge_pred_is.csv so backtester picks it up)
    is_file = PRED_RESULT_DIR / "ridge_bayes_pred_is.csv"
    oos_file = PRED_RESULT_DIR / "ridge_bayes_pred_oos.csv"
    
    # Note: I am naming it ridge_bayes so you can compare it side-by-side if you want
    pd.DataFrame({'time': train_data['time'], 'signal': pred_is}).to_csv(is_file, index=False)
    pd.DataFrame({'time': oos_features['time'], 'signal': pred_oos}).to_csv(oos_file, index=False)
    
    print(f"Predictions saved to:\n - {is_file}\n - {oos_file}")
    
    # 6. Quick OOS R2 Check
    oos_r2 = r2_score(oos_features['label'], pred_oos)
    print(f"OOS R2: {oos_r2:.5f}")

if __name__ == "__main__":
    run_bayesian_optimization()