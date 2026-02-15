"""
src/train_ridge.py
Ridge Regression Pipeline:
1. Loads Data (Features from 'data/', Labels from 'data/')
2. Engineers Features (RSI, Volume, Weekend Fill)
3. Trains Ridge (Standardized)
4. Saves to 'outputs/pred_result/' for Batch Backtest
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from pathlib import Path

# --- CONFIGURATION ---
DATA_DIR = Path("data")       # Inputs are now in 'data/'
OUTPUT_DIR = Path("outputs")  
PRED_RESULT_DIR = OUTPUT_DIR / "pred_result" # New folder for backtest predictions
PRED_RESULT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Inputs (Read from DATA_DIR)
FEATURE_IS = DATA_DIR / "feature_is.csv"
FEATURE_OOS = DATA_DIR / "feature_oos.csv"
LABEL_IS = DATA_DIR / "label_is.csv"
LABEL_OOS = DATA_DIR / "label_oos.csv"

# Report Output
REPORT_FILE = OUTPUT_DIR / "oos_report.png" 
FORECAST_PLOT_FILE = PLOTS_DIR / "ridge_forecast_vs_actual.png"

def keep_close_only(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = [c for c in df.columns if c.endswith(("_open", "_high", "_low"))]
    return df.drop(columns=drop_cols, errors="ignore")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("--- Engineering Features ---")
    df = df.copy()
    
    # --- 1. TARGETED FORWARD FILL (Weekend Handling) ---
    tradfi_cols = [
        "IG_NASDAQ_1D__close", "TVC_VIX_1D__close", 
        "TVC_US10Y_1D__close", "TVC_US02Y_1D__close",
        "ECONOMICS_USCBBS_1D__close", "FRED_M2SL_1D__close"
    ]
    present_tradfi = [c for c in tradfi_cols if c in df.columns]
    df[present_tradfi] = df[present_tradfi].ffill()
    
    # General ffill
    df = df.ffill()
    
    col_btc_close = "BTCUSD__close"
    col_nasdaq = "IG_NASDAQ_1D__close"
    col_vix = "TVC_VIX_1D__close"
    col_us10y = "TVC_US10Y_1D__close"
    col_us02y = "TVC_US02Y_1D__close"
    
    # Returns
    if col_nasdaq in df.columns:
        df["NASDAQ_ret"] = df[col_nasdaq].pct_change()
    if col_vix in df.columns:
        df["VIX_ret"] = df[col_vix].pct_change()
    for col in ["ECONOMICS_USCBBS_1D__close", "FRED_M2SL_1D__close"]:
        if col in df.columns:
            df[f"{col}_pct"] = df[col].pct_change()
    
    # --- 2. Volume Logic (FAIL-SAFE) ---
    vol_col = next((c for c in ["BTCUSD__volume", "BTCUSD_volume", "volume"] if c in df.columns), None)
    
    if vol_col:
        df["VOL_REL_MA"] = (df[vol_col] / df[vol_col].rolling(20).mean()) - 1.0
        print(f"Successfully added Volume Indicator from: {vol_col}")
    else:
        print("Warning: Volume not found. Using VIX return as activity proxy.")
        if "TVC_VIX_1D__close" in df.columns:
            df["VOL_REL_MA"] = df["TVC_VIX_1D__close"].pct_change()

    # Term Spreads
    if col_us10y in df.columns and col_us02y in df.columns:
        df["TERM_10Y_2Y"] = df[col_us10y] - df[col_us02y]
    
    # Technicals
    ma_cols = ["BTCUSD__MA", "BTCUSD__MA.1", "BTCUSD__MA.2", "BTCUSD__MA.3", "BTCUSD__MA.4"]
    present_mas = [c for c in ma_cols if c in df.columns]
    
    if col_btc_close in df.columns and present_mas:
        for ma in present_mas:
            df[f"DIST_{ma}"] = (df[col_btc_close] / df[ma]) - 1.0
        
        if len(present_mas) >= 2:
            short_ma = present_mas[0]
            long_ma = present_mas[-1]
            df["TREND_SLOPE"] = (df[short_ma] - df[long_ma]) / df[col_btc_close]

    # --- RSI CALCULATION ---
    if col_btc_close in df.columns:
        delta = df[col_btc_close].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 0.001)
        df['RSI_14'] = (100 - (100 / (1 + rs))) / 100.0 - 0.5 
    # -----------------------

    # Drop Raw Levels
    drop_cols = [col_btc_close, col_nasdaq, col_vix, col_us10y, col_us02y] + present_mas
    drop_cols += ["ECONOMICS_USCBBS_1D__close", "FRED_M2SL_1D__close", "TVC_US03MY_1D__close"]
    
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return df

def prepare_data():
    print("Loading data from data/ folder...")
    # Load Features
    df_is = pd.read_csv(FEATURE_IS)
    df_oos = pd.read_csv(FEATURE_OOS)
    # Load NEW Labels (Intraday Return)
    lbl_is = pd.read_csv(LABEL_IS)
    lbl_oos = pd.read_csv(LABEL_OOS)
    
    # Time Conversion
    for d in [df_is, df_oos, lbl_is, lbl_oos]:
        d['time'] = pd.to_datetime(d['time'])

    # Merge Features + Labels on Time
    # (Left join to keep feature rows, but strictly we need labels for training)
    df_is = pd.merge(df_is, lbl_is, on='time', how='inner')
    df_oos = pd.merge(df_oos, lbl_oos, on='time', how='inner')

    df_is['split'] = 'train'
    df_oos['split'] = 'test'
    full_df = pd.concat([df_is, df_oos], axis=0).sort_values('time').reset_index(drop=True)
    
    full_df = keep_close_only(full_df)
    
    # --- FIX DATA LEAKAGE HERE ---
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

def calculate_oos_metrics(y_true, y_pred, train_mean):
    oos_r2 = r2_score(y_true, y_pred)
    y_bench = np.full_like(y_true, train_mean)
    bench_r2 = r2_score(y_true, y_bench)
    
    print(f"\n--- Statistical Performance ---")
    print(f"Model OOS R2:     {oos_r2:.5f}")
    print(f"Benchmark OOS R2: {bench_r2:.5f}")

def get_feature_importance(model, feature_names):
    if hasattr(model, 'named_steps'):
        regressor = model.named_steps['model']
    else:
        regressor = model
    
    if hasattr(regressor, 'coef_'):
        importances = regressor.coef_
        feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feat_imp['abs_importance'] = feat_imp['importance'].abs()
        feat_imp = feat_imp.sort_values('abs_importance', ascending=False).head(20)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feat_imp, palette='coolwarm')
        plt.title('Top 20 Features (Ridge Coefficients)')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "importance_ridge.png")
        plt.close()

def generate_full_report(df):
    print("Generating OOS Report...")
    df = df.copy().dropna()
    
    df['ic_rolling'] = df['label'].rolling(60).corr(df['signal'])
    df['decile'] = pd.qcut(df['signal'], 10, labels=False, duplicates='drop') + 1
    decile_returns = df.groupby('decile')['label'].mean()
    
    df['position'] = np.sign(df['signal'])
    df['strat_ret'] = df['position'] * df['label']
    df['cum_ret'] = (1 + df['strat_ret']).cumprod()
    
    total_ret = df['cum_ret'].iloc[-1] - 1
    sharpe = df['strat_ret'].mean() / df['strat_ret'].std() * np.sqrt(365)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0,0].plot(df['time'], df['ic_rolling'])
    axes[0,0].axhline(0, color='k', linewidth=0.5)
    axes[0,0].set_title("OOS IC (Rolling 60d)")
    
    axes[0,1].bar(decile_returns.index, decile_returns.values)
    axes[0,1].set_title("OOS Decile Mean Returns")
    
    axes[1,0].plot(df['time'], df['cum_ret'])
    axes[1,0].set_title("OOS Cumulative Return")
    
    text_str = f"Total Return: {total_ret:.2%}\nSharpe Ratio: {sharpe:.2f}\n"
    text_str += f"Model: Ridge Regression\n(Standardized)"
    axes[1,1].text(0.1, 0.5, text_str, fontsize=12, fontfamily='monospace')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(REPORT_FILE)
    print(f"Full Report saved to {REPORT_FILE}")
    plt.close()

def plot_forecast_vs_actual():
    # Only try to plot if we have predictions in memory (or passed to this func)
    # Since we are changing where we save files, this function relies on the DataFrame passed to generate_full_report
    # But for compatibility with the old script structure, we'll just skip loading from file if not needed.
    pass 

def run_training():
    print(f"\n--- Running Pipeline: RIDGE REGRESSION ---")
    train_data, oos_features = prepare_data()
    
    X_train = train_data.drop(columns=['time', 'label'])
    y_train = train_data['label']
    X_oos = oos_features.drop(columns=['time', 'label'], errors='ignore')
    feature_names = X_train.columns
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge())
    ])
    
    param_grid = {
        'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }

    print("Tuning hyperparameters...")
    grid = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    print(f"Best Params: {grid.best_params_}") 
    
    pred_is = best_model.predict(X_train)
    pred_oos = best_model.predict(X_oos)
    
    # --- NEW SAVING LOGIC (outputs/pred_result) ---
    is_file = PRED_RESULT_DIR / "ridge_pred_is.csv"
    oos_file = PRED_RESULT_DIR / "ridge_pred_oos.csv"
    
    pd.DataFrame({'time': train_data['time'], 'signal': pred_is}).to_csv(is_file, index=False)
    pd.DataFrame({'time': oos_features['time'], 'signal': pred_oos}).to_csv(oos_file, index=False)
    
    print(f"Success! Saved predictions to:\n - {is_file}\n - {oos_file}")
    
    # NOTE: We DO NOT overwrite label files in data/ anymore.
    
    get_feature_importance(best_model, feature_names)
    calculate_oos_metrics(oos_features['label'], pred_oos, y_train.mean())

    report_df = pd.DataFrame({
        'time': oos_features['time'],
        'label': oos_features['label'],
        'signal': pred_oos
    })
    generate_full_report(report_df)

if __name__ == "__main__":
    run_training()