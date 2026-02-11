"""
src/train_rf.py
Random Forest Pipeline:
1. Train (Robust / "Financial" Parameters)
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
FORECAST_PLOT_FILE = PLOTS_DIR / "rf_forecast_vs_actual.png"

def keep_close_only(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = [c for c in df.columns if c.endswith(("_open", "_high", "_low"))]
    return df.drop(columns=drop_cols, errors="ignore")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("--- Engineering Features ---")
    df = df.copy()
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
        df['RSI_14'] = 100 - (100 / (1 + rs))
        df['RSI_14'] = (df['RSI_14'] / 100.0) - 0.5 
    # -----------------------

    drop_cols = [col_btc_close, col_nasdaq, col_vix, col_us10y, col_us02y] + present_mas
    drop_cols += ["ECONOMICS_USCBBS_1D__close", "FRED_M2SL_1D__close", "TVC_US03MY_1D__close"]
    
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
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
    full_df['label'] = full_df['BTCUSD__close'].pct_change().shift(-1)
    
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

def generate_full_report(df):
    """Generates the 4-panel OOS Report."""
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
    axes[0,1].set_xlabel("Decile")
    axes[0,1].set_ylabel("Mean Return")
    
    axes[1,0].plot(df['time'], df['cum_ret'])
    axes[1,0].set_title("OOS Cumulative Return (Signal Sign)")
    
    text_str = f"Total Return: {total_ret:.2%}\nSharpe Ratio: {sharpe:.2f}\n"
    text_str += f"Model: Random Forest\n(Robust Financial Params)"
    axes[1,1].text(0.1, 0.5, text_str, fontsize=12, fontfamily='monospace')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(REPORT_FILE)
    print(f"Full Report saved to {REPORT_FILE}")
    plt.close()

def plot_forecast_vs_actual():
    print("Loading OOS data for Plotting...")
    if not PRED_OOS_FILE.exists() or not LABEL_OOS_FILE.exists():
        print("Predictions or Labels not found. Skipping plot.")
        return

    df_pred = pd.read_csv(PRED_OOS_FILE)
    df_label = pd.read_csv(LABEL_OOS_FILE)
    
    df_pred['time'] = pd.to_datetime(df_pred['time'])
    df_label['time'] = pd.to_datetime(df_label['time'])
    
    df = pd.merge(df_label, df_pred, on='time', how='inner', suffixes=('_actual', '_pred'))
    df = df.sort_values('time')
    
    plt.figure(figsize=(14, 7))
    plt.plot(df['time'], df['label'], label='Actual OOS Return', color='#1f77b4', marker='o', markersize=3, linestyle='-', linewidth=1, alpha=0.6)
    plt.plot(df['time'], df['signal'], label='Random Forest Forecast', color='#2ca02c', marker='.', markersize=4, linestyle='-', linewidth=2)
    plt.title("Random Forest OOS Returns: Forecast vs Actual", fontsize=14)
    plt.ylabel("Return", fontsize=12)
    plt.xlabel("Time", fontsize=12)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    FORECAST_PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FORECAST_PLOT_FILE, dpi=300)
    print(f"Forecast Plot saved to {FORECAST_PLOT_FILE}")

def run_training():
    print(f"\n--- Running Pipeline: RANDOM FOREST ---")
    train_data, oos_features = prepare_data()
    
    drop_cols = ['time', 'label']
    X_train = train_data.drop(columns=drop_cols)
    y_train = train_data['label']
    X_oos = oos_features.drop(columns=drop_cols, errors='ignore')
    feature_names = X_train.columns
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    # 1. Pipeline: No StandardScaler (Correct per Lecture 3)
    pipeline = Pipeline([
        ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    
    # 2. "Financial" / Robust Hyperparameters
    # We removed the "textbook" options (10, None) because they overfit noise.
    param_grid = {
        'model__n_estimators': [200],
        
        # Keep trees simple: 3 or 5 levels max. 
        # Prevents "memorizing" specific days.
        'model__max_depth': [3, 5], 
        
        # Require 50 days of data per leaf.
        # "Don't trade it unless you've seen it 50 times."
        'model__min_samples_leaf': [50],
        
        # 'sqrt' is standard for removing noise in RF
        'model__max_features': ['sqrt'] 
    }

    print("Tuning hyperparameters (Robust Mode)...")
    grid = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    print(f"Best Params: {grid.best_params_}") 
    
    pred_is = best_model.predict(X_train)
    pred_oos = best_model.predict(X_oos)
    
    pd.DataFrame({'time': train_data['time'], 'signal': pred_is}).to_csv(OUTPUT_DIR / "pred_is.csv", index=False)
    pd.DataFrame({'time': oos_features['time'], 'signal': pred_oos}).to_csv(OUTPUT_DIR / "pred_oos.csv", index=False)
    
    train_data[['time', 'label']].to_csv(LABEL_IS, index=False)
    oos_features[['time', 'label']].to_csv(LABEL_OOS, index=False)
    
    print(f"Success! Predictions saved.")
    
    get_feature_importance(best_model, feature_names)
    
    train_mean = y_train.mean()
    calculate_oos_metrics(oos_features['label'], pred_oos, train_mean)

    plot_forecast_vs_actual()

    report_df = pd.DataFrame({
        'time': oos_features['time'],
        'label': oos_features['label'],
        'signal': pred_oos
    })
    generate_full_report(report_df)

if __name__ == "__main__":
    run_training()