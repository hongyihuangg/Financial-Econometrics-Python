"""
src/train_ridge_bayes.py
Ridge Regression Pipeline with BAYESIAN OPTIMIZATION (Optuna):
1. Loads Data & Fixes Labels (Shift -1)
2. Engineers Features (Weekends Removed, RSI, Volume, Macro Pct, Intermediate MA Slopes)
3. Optimizes Alpha using Optuna (TPE Sampler, Seed 42)
4. Trains Final Model & Generates Coefficient Plot
5. Saves Best Model Predictions for Backtest
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
from sklearn.metrics import r2_score
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
COEFF_PLOT_FILE = PLOTS_DIR / "ridge_bayes_coefficients.png" 

# Setup Logging for Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def keep_close_only(df):
    df = df.copy()
    drop_cols = [c for c in df.columns if c.endswith(("_open", "_high", "_low"))]
    return df.drop(columns=drop_cols, errors="ignore")

def build_features(df):
    print("--- Engineering Features (Merged Logic) ---")
    df = df.copy()
    
    # 1. Forward Fill TradFi
    tradfi_cols = [
        "IG_NASDAQ_1D__close", "TVC_VIX_1D__close", 
        "TVC_US10Y_1D__close", "TVC_US02Y_1D__close", "TVC_US03MY_1D__close",
        "ECONOMICS_USCBBS_1D__close", "FRED_M2SL_1D__close"
    ]
    present_tradfi = [c for c in tradfi_cols if c in df.columns]
    df[present_tradfi] = df[present_tradfi].ffill()
    df = df.ffill()
    
    col_btc_close = "BTCUSD__close"
    
    # 2. Returns & Macro Changes
    if "IG_NASDAQ_1D__close" in df.columns: 
        df["NASDAQ_ret"] = df["IG_NASDAQ_1D__close"].pct_change()
    if "TVC_VIX_1D__close" in df.columns: 
        df["VIX_ret"] = df["TVC_VIX_1D__close"].pct_change()
    if "ECONOMICS_USCBBS_1D__close" in df.columns:
        df["USCBBS_change"] = df["ECONOMICS_USCBBS_1D__close"].pct_change()
    if "FRED_M2SL_1D__close" in df.columns:
        df["M2SL_change"] = df["FRED_M2SL_1D__close"].pct_change()
    
    # 3. Volume
    vol_col = next((c for c in ["BTCUSD__volume", "BTCUSD_volume", "volume"] if c in df.columns), None)
    if vol_col:
        df["VOL_REL_MA"] = (df[vol_col] / df[vol_col].rolling(20).mean()) - 1.0
    elif "TVC_VIX_1D__close" in df.columns:
        df["VOL_REL_MA"] = df["TVC_VIX_1D__close"].pct_change()

    # 4. Term Spreads
    if "TVC_US10Y_1D__close" in df.columns and "TVC_US02Y_1D__close" in df.columns:
        df["TERM_10Y_2Y"] = df["TVC_US10Y_1D__close"] - df["TVC_US02Y_1D__close"]
    if "TVC_US10Y_1D__close" in df.columns and "TVC_US03MY_1D__close" in df.columns:
        df["TERM_10Y_3M"] = df["TVC_US10Y_1D__close"] - df["TVC_US03MY_1D__close"]
    
    # 5. Technicals (Distances & Intermediate Slopes)
    ma_cols = ["BTCUSD__MA", "BTCUSD__MA.1", "BTCUSD__MA.2", "BTCUSD__MA.3", "BTCUSD__MA.4"]
    present_mas = [c for c in ma_cols if c in df.columns]
    
    if col_btc_close in df.columns and present_mas:
        # Distance to MA
        for ma in present_mas:
            df[f"DIST_{ma}"] = (df[col_btc_close] / df[ma]) - 1.0
            
        # Intermediate Slopes (MA1 vs MA2, MA2 vs MA3...)
        for i in range(len(present_mas) - 1):
            short_ma = present_mas[i]
            long_ma = present_mas[i + 1]
            df[f"SLOPE_{short_ma}_minus_{long_ma}"] = (df[short_ma] - df[long_ma]) / df[col_btc_close]

        # Mega Slope
        if len(present_mas) >= 2:
            df["TREND_SLOPE_MEGA"] = (df[present_mas[0]] - df[present_mas[-1]]) / df[col_btc_close]

    # 6. RSI
    if col_btc_close in df.columns:
        delta = df[col_btc_close].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 0.001)
        df['RSI_14'] = (100 - (100 / (1 + rs))) / 100.0 - 0.5 

    # 7. Cleanup Raw Columns
    drop_cols = [col_btc_close] + present_tradfi + present_mas 
    drop_cols += ["ETHBTC_close", "BTCDOMclose", "CRYPTOCAP_USDT_D_1D_close"] # User requested drops
    
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df

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
    
    # --- Drop Weekends ---
    print("Removing weekends to reduce forward-fill drag...")
    full_df = full_df[full_df['time'].dt.dayofweek < 5].reset_index(drop=True)
    
    full_df = keep_close_only(full_df)
    
    # --- FIX DATA LEAKAGE ---
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

def plot_coefficients(pipeline, feature_names):
    """Generates and saves a bar chart of the top 20 Ridge coefficients."""
    model = pipeline.named_steps['model']
    
    coefs = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    })
    
    coefs = coefs.sort_values('Abs_Coefficient', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4' if c > 0 else '#d62728' for c in coefs['Coefficient']]
    sns.barplot(x='Coefficient', y='Feature', data=coefs, hue='Feature', palette=colors, legend=False)
    
    plt.title('Top 20 Ridge Coefficients (Bayesian Optimized)', fontsize=14)
    plt.xlabel('Coefficient Value (Impact Direction)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(COEFF_PLOT_FILE, dpi=300)
    print(f"Coefficient Plot saved to {COEFF_PLOT_FILE}")
    plt.close()

def objective(trial, X, y):
    alpha = trial.suggest_float('alpha', 1e-3, 1e3, log=True)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=alpha))
    ])
    
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    return -scores.mean()

def run_bayesian_optimization():
    print(f"\n--- Running Pipeline: RIDGE with BAYESIAN OPTIMIZATION ---")
    train_data, oos_features = prepare_data()
    
    X_train = train_data.drop(columns=['time', 'label'])
    y_train = train_data['label']
    X_oos = oos_features.drop(columns=['time', 'label'], errors='ignore')
    
    print("Starting Optuna Study (50 Trials, Seed=42)...")
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)
    
    print(f"\nBest Alpha Found: {study.best_params['alpha']:.6f}")
    
    best_alpha = study.best_params['alpha']
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=best_alpha))
    ])
    final_pipeline.fit(X_train, y_train)
    
    plot_coefficients(final_pipeline, X_train.columns)
    
    pred_is = final_pipeline.predict(X_train)
    pred_oos = final_pipeline.predict(X_oos)
    
    is_file = PRED_RESULT_DIR / "ridge_bayes_pred_is.csv"
    oos_file = PRED_RESULT_DIR / "ridge_bayes_pred_oos.csv"
    
    pd.DataFrame({'time': train_data['time'], 'signal': pred_is}).to_csv(is_file, index=False)
    pd.DataFrame({'time': oos_features['time'], 'signal': pred_oos}).to_csv(oos_file, index=False)
    
    print(f"Predictions saved to:\n - {is_file}\n - {oos_file}")
    
    oos_r2 = r2_score(oos_features['label'], pred_oos)
    print(f"OOS R2: {oos_r2:.5f}")

if __name__ == "__main__":
    run_bayesian_optimization()