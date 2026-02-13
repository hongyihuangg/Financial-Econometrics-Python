"""
src/train_rf.py
Random Forest Pipeline:
1. Train (Theory-Aligned Parameters + RSI)
2. Predict (Fix Stale Labels)
3. Report (Generate Forecast Plot + 4-Panel OOS Report)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from pathlib import Path

# --- CONFIGURATION ---
OUTPUT_DIR = Path("outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_IS = OUTPUT_DIR / "feature_is.csv"
FEATURE_OOS = OUTPUT_DIR / "feature_oos.csv"
LABEL_IS = OUTPUT_DIR / "label_is.csv"
LABEL_OOS = OUTPUT_DIR / "label_oos.csv"
REPORT_FILE = OUTPUT_DIR / "oos_report.png" 

# --- PLOT CONFIG ---
PRED_OOS_FILE = OUTPUT_DIR / "pred_oos.csv"
LABEL_OOS_FILE = OUTPUT_DIR / "label_oos.csv"
# Changing name and folder for RF specific plot
FORECAST_PLOT_FILE = PLOTS_DIR / "rf_forecast_vs_actual.png"

def keep_close_only(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = [c for c in df.columns if c.endswith(("_open", "_high", "_low"))]
    return df.drop(columns=drop_cols, errors="ignore")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("--- Engineering Features (Robust Volume + Weekend Fill) ---")
    df = df.copy()
    
    # 1. Targeted Forward Fill (TradFi only) - Fixes the Weekend Gap
    tradfi_cols = ["IG_NASDAQ_1D__close", "TVC_VIX_1D__close", "TVC_US10Y_1D__close", 
                   "TVC_US02Y_1D__close", "ECONOMICS_USCBBS_1D__close", "FRED_M2SL_1D__close"]
    present_tradfi = [c for c in tradfi_cols if c in df.columns]
    df[present_tradfi] = df[present_tradfi].ffill()

    # 2. Volume Logic (FAIL-SAFE) - Adds the missing Volume Feature
    vol_col = next((c for c in ["BTCUSD__volume", "BTCUSD_volume", "volume"] if c in df.columns), None)
    
    if vol_col:
        # Calculate Volume Relative to 20-day Average
        df["VOL_REL_MA"] = (df[vol_col] / df[vol_col].rolling(20).mean()) - 1.0
        print(f"Successfully added Volume Indicator from: {vol_col}")
    else:
        print("Warning: Volume not found. Using VIX return as activity proxy.")
        if "TVC_VIX_1D__close" in df.columns:
            df["VOL_REL_MA"] = df["TVC_VIX_1D__close"].pct_change()
    
    # 3. Standard Indicators (Your original logic starts here)
    col_btc_close = "BTCUSD__close"
    if "IG_NASDAQ_1D__close" in df.columns:
        df["NASDAQ_ret"] = df["IG_NASDAQ_1D__close"].pct_change()
    
    if "TVC_VIX_1D__close" in df.columns:
        df["VIX_ret"] = df["TVC_VIX_1D__close"].pct_change()

    for col in ["ECONOMICS_USCBBS_1D__close", "FRED_M2SL_1D__close"]:
        if col in df.columns:
            df[f"{col}_pct"] = df[col].pct_change()

    if "TVC_US10Y_1D__close" in df.columns and "TVC_US02Y_1D__close" in df.columns:
        df["TERM_10Y_2Y"] = df["TVC_US10Y_1D__close"] - df["TVC_US02Y_1D__close"]
    
    ma_cols = ["BTCUSD__MA", "BTCUSD__MA.1", "BTCUSD__MA.2", "BTCUSD__MA.3", "BTCUSD__MA.4"]
    present_mas = [c for c in ma_cols if c in df.columns]
    
    if col_btc_close in df.columns and present_mas:
        for ma in present_mas:
            df[f"DIST_{ma}"] = (df[col_btc_close] / df[ma]) - 1.0
        
        if len(present_mas) >= 2:
            short_ma = present_mas[0]
            long_ma = present_mas[-1]
            df["TREND_SLOPE"] = (df[short_ma] - df[long_ma]) / df[col_btc_close]

    # RSI
    if col_btc_close in df.columns:
        delta = df[col_btc_close].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 0.001)
        df['RSI_14'] = (100 - (100 / (1 + rs))) / 100.0 - 0.5 

    df = df.ffill().fillna(0)
    
    # Drop raw levels so model doesn't overfit price level
    # Added vol_col to this list so raw volume doesn't leak in
    drop_raw = [col_btc_close, vol_col] + present_tradfi
    df = df.drop(columns=[c for c in drop_raw if c in df.columns], errors="ignore")
    
    return df

def prepare_data():
    print("Loading data...")
    df_is = pd.read_csv(FEATURE_IS)
    df_oos = pd.read_csv(FEATURE_OOS)
    
    df_is['time'] = pd.to_datetime(df_is['time'])
    df_oos['time'] = pd.to_datetime(df_oos['time'])

    df_is['split'] = 'train'
    df_oos['split'] = 'test'
    full_df = pd.concat([df_is, df_oos], axis=0).sort_values('time').reset_index(drop=True)
    
    full_df = keep_close_only(full_df)
    full_df['label'] = full_df['BTCUSD__close'].pct_change().shift(-1) # Forward Label
    
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
    if oos_r2 > bench_r2:
        print("RESULT: Model BEATS Benchmark!")
    else:
        print("RESULT: Model UNDERPERFORMS Benchmark.")

def get_feature_importance(model, feature_names):
    if hasattr(model, 'named_steps'):
        regressor = model.named_steps['model']
    else:
        regressor = model
        
    if hasattr(regressor, 'feature_importances_'):
        importances = regressor.feature_importances_
        feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feat_imp = feat_imp.sort_values('importance', ascending=False).head(20)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feat_imp, palette='viridis')
        plt.title('Top 20 Features (Random Forest)')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "importance_rf.png")
        plt.close()

# --- 4-PANEL OOS REPORT FUNCTION ---
def generate_full_report(df):
    """Generates the 4-panel OOS Report (IC, Deciles, Cumulative Return, Metrics)."""
    print("Generating OOS Report...")
    df = df.copy().dropna()
    
    # 1. OOS IC (Rolling)
    df['ic_rolling'] = df['label'].rolling(60).corr(df['signal'])
    
    # 2. Deciles
    df['decile'] = pd.qcut(df['signal'], 10, labels=False, duplicates='drop') + 1
    decile_returns = df.groupby('decile')['label'].mean()
    
    # 3. Cumulative Return (Strategy vs Hold)
    df['position'] = np.sign(df['signal']) # -1 or +1
    df['strat_ret'] = df['position'] * df['label']
    df['cum_ret'] = (1 + df['strat_ret']).cumprod()
    
    # Metrics
    total_ret = df['cum_ret'].iloc[-1] - 1
    sharpe = df['strat_ret'].mean() / df['strat_ret'].std() * np.sqrt(365)
    
    # --- PLOTTING ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Top Left: IC
    axes[0,0].plot(df['time'], df['ic_rolling'])
    axes[0,0].axhline(0, color='k', linewidth=0.5)
    axes[0,0].set_title("OOS IC (Rolling 60d)")
    
    # Top Right: Deciles
    axes[0,1].bar(decile_returns.index, decile_returns.values)
    axes[0,1].set_title("OOS Decile Mean Returns")
    axes[0,1].set_xlabel("Decile")
    axes[0,1].set_ylabel("Mean Return")
    
    # Bottom Left: Cumulative
    axes[1,0].plot(df['time'], df['cum_ret'])
    axes[1,0].set_title("OOS Cumulative Return (Signal Sign)")
    
    # Bottom Right: Text Stats
    text_str = f"Total Return: {total_ret:.2%}\nSharpe Ratio: {sharpe:.2f}\n"
    text_str += f"Model: Random Forest\n(Theory + RSI)"
    axes[1,1].text(0.1, 0.5, text_str, fontsize=12, fontfamily='monospace')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(REPORT_FILE)
    print(f"Full Report saved to {REPORT_FILE}")
    plt.close()

# --- FORECAST VS ACTUAL PLOT FUNCTION ---
def plot_forecast_vs_actual():
    # 1. Load Data
    print("Loading OOS data for Plotting...")
    if not PRED_OOS_FILE.exists() or not LABEL_OOS_FILE.exists():
        print("Predictions or Labels not found. Skipping plot.")
        return

    df_pred = pd.read_csv(PRED_OOS_FILE)
    df_label = pd.read_csv(LABEL_OOS_FILE)
    
    # 2. Merge on Time
    df_pred['time'] = pd.to_datetime(df_pred['time'])
    df_label['time'] = pd.to_datetime(df_label['time'])
    
    # Merge using inner join to ensure alignment
    df = pd.merge(df_label, df_pred, on='time', how='inner', suffixes=('_actual', '_pred'))
    df = df.sort_values('time')
    
    print(f"Plotting {len(df)} days of OOS data...")

    # 3. Plot
    plt.figure(figsize=(14, 7))
    
    # Plot Actual Returns (Blue, jagged)
    plt.plot(df['time'], df['label'], 
             label='Actual OOS Return', 
             color='#1f77b4', 
             marker='o', 
             markersize=3, 
             linestyle='-', 
             linewidth=1, 
             alpha=0.6)
    
    # Plot RF Forecast (Green, smooth)
    plt.plot(df['time'], df['signal'], 
             label='Random Forest Forecast', 
             color='#2ca02c',  # Green for RF
             marker='.', 
             markersize=4, 
             linestyle='-', 
             linewidth=2)

    # Style
    plt.title("Random Forest OOS Returns: Forecast vs Actual", fontsize=14)
    plt.ylabel("Return", fontsize=12)
    plt.xlabel("Time", fontsize=12)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # Ensure plots dir exists
    FORECAST_PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FORECAST_PLOT_FILE, dpi=300)
    print(f"Forecast Plot saved to {FORECAST_PLOT_FILE}")
    # plt.show() 

def run_training():
    print(f"\n--- Running Pipeline: RANDOM FOREST ---")
    train_data, oos_features = prepare_data()
    
    drop_cols = ['time', 'label']
    X_train = train_data.drop(columns=drop_cols)
    y_train = train_data['label']
    X_oos = oos_features.drop(columns=drop_cols, errors='ignore')
    feature_names = X_train.columns
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    # 1. Pipeline: No StandardScaler (Per Lecture 3 Notes)
    pipeline = Pipeline([
        ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    
    # 2. Expanded Param Grid: Tests Theory vs. Reality
    # - Theory: min_samples_leaf=5-20, max_features=0.33
    # - Reality: min_samples_leaf=50, max_features='sqrt' (Robust)
    param_grid = {
        'model__n_estimators': [200],
        'model__max_depth': [3, 5, 10, None], 
        'model__min_samples_leaf': [10, 50],
        'model__max_features': ['sqrt', 0.33] 
    }

    print("Tuning hyperparameters (Testing Theory vs Robustness)...")
    grid = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    print(f"Best Params: {grid.best_params_}") 
    
    pred_is = best_model.predict(X_train)
    pred_oos = best_model.predict(X_oos)
    
    pd.DataFrame({'time': train_data['time'], 'signal': pred_is}).to_csv(OUTPUT_DIR / "pred_is.csv", index=False)
    pd.DataFrame({'time': oos_features['time'], 'signal': pred_oos}).to_csv(OUTPUT_DIR / "pred_oos.csv", index=False)
    
    # Fix Stale Labels
    train_data[['time', 'label']].to_csv(LABEL_IS, index=False)
    oos_features[['time', 'label']].to_csv(LABEL_OOS, index=False)
    
    print(f"Success! Predictions saved.")
    
    get_feature_importance(best_model, feature_names)
    
    train_mean = y_train.mean()
    calculate_oos_metrics(oos_features['label'], pred_oos, train_mean)

    # --- REPORTING ---
    # 1. Generate Forecast vs Actual Plot
    plot_forecast_vs_actual()

    # 2. Generate Full 4-Panel Report
    report_df = pd.DataFrame({
        'time': oos_features['time'],
        'label': oos_features['label'],
        'signal': pred_oos
    })
    generate_full_report(report_df)

if __name__ == "__main__":
    run_training()
