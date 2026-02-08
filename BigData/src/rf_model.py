"""
src/train_rf.py
Random Forest Pipeline with Data Leakage Fixes and Statistical Metrics.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
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

def keep_close_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drops Open/High/Low columns to reduce noise."""
    df = df.copy()
    drop_cols = [c for c in df.columns if c.endswith(("_open", "_high", "_low"))]
    return df.drop(columns=drop_cols, errors="ignore")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers financial features (Returns, Volatility, Trends)."""
    print("--- Engineering Features ---")
    df = df.copy()
    
    # 1. Forward Fill Macro Data
    df = df.ffill()
    
    # 2. Setup Column Names
    col_btc_close = "BTCUSD__close"
    col_nasdaq = "IG_NASDAQ_1D__close"
    col_vix = "TVC_VIX_1D__close"
    col_us10y = "TVC_US10Y_1D__close"
    col_us02y = "TVC_US02Y_1D__close"
    
    # 3. Calculate Returns
    if col_nasdaq in df.columns:
        df["NASDAQ_ret"] = df[col_nasdaq].pct_change()
    if col_vix in df.columns:
        df["VIX_ret"] = df[col_vix].pct_change()
        
    for col in ["ECONOMICS_USCBBS_1D__close", "FRED_M2SL_1D__close"]:
        if col in df.columns:
            df[f"{col}_pct"] = df[col].pct_change()

    # 4. Term Spreads
    if col_us10y in df.columns and col_us02y in df.columns:
        df["TERM_10Y_2Y"] = df[col_us10y] - df[col_us02y]
    
    # 5. Technicals: Distance to MA
    ma_cols = ["BTCUSD__MA", "BTCUSD__MA.1", "BTCUSD__MA.2", "BTCUSD__MA.3", "BTCUSD__MA.4"]
    present_mas = [c for c in ma_cols if c in df.columns]
    
    if col_btc_close in df.columns and present_mas:
        for ma in present_mas:
            df[f"DIST_{ma}"] = (df[col_btc_close] / df[ma]) - 1.0
        
        # Trend Slope
        if len(present_mas) >= 2:
            short_ma = present_mas[0]
            long_ma = present_mas[-1]
            df["TREND_SLOPE"] = (df[short_ma] - df[long_ma]) / df[col_btc_close]

    # --- 5b. NEW CODE: Relative Strength Index (RSI) ---
    # We add this HERE because we need 'col_btc_close' before it gets dropped in Step 6.
    if col_btc_close in df.columns:
        # Calculate daily price changes
        delta = df[col_btc_close].diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        # Calculate RS and RSI
        # (Replace 0s with a tiny number to avoid division by zero errors)
        rs = gain / loss.replace(0, 0.001)
        df['RSI_14'] = 100 - (100 / (1 + rs))
    # ---------------------------------------------------

    # 6. Drop Raw Levels (Non-Stationary)
    drop_cols = [col_btc_close, col_nasdaq, col_vix, col_us10y, col_us02y] + present_mas
    drop_cols += ["ECONOMICS_USCBBS_1D__close", "FRED_M2SL_1D__close", "TVC_US03MY_1D__close"]
    
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return df

def prepare_data():
    """Loads data, calculates Forward Label, then builds features."""
    print("Loading data...")
    df_is = pd.read_csv(FEATURE_IS)
    df_oos = pd.read_csv(FEATURE_OOS)
    
    df_is['time'] = pd.to_datetime(df_is['time'])
    df_oos['time'] = pd.to_datetime(df_oos['time'])

    # Combine
    df_is['split'] = 'train'
    df_oos['split'] = 'test'
    full_df = pd.concat([df_is, df_oos], axis=0).sort_values('time').reset_index(drop=True)
    
    # 1. Clean Data
    full_df = keep_close_only(full_df)

    # --- CRITICAL FIX: CALCULATE LABEL FIRST ---
    # Predict Return from Close(T) to Close(T+1)
    full_df['label'] = full_df['BTCUSD__close'].pct_change().shift(-1)

    # 2. Engineer Features
    full_df = build_features(full_df)
    
    # 3. Clean NaNs
    full_df = full_df.dropna(subset=['label'])
    
    meta_cols = ['time', 'split', 'label', 'BTCUSD__close']
    feature_cols = [c for c in full_df.columns if c not in meta_cols]
    full_df[feature_cols] = full_df[feature_cols].ffill().fillna(0)
    
    # 4. Split
    train_df = full_df[full_df['split'] == 'train'].drop(columns=['split']).copy()
    test_df = full_df[full_df['split'] == 'test'].drop(columns=['split']).copy()

    if len(train_df) == 0:
        raise ValueError("Training Data is empty!")

    return train_df, test_df

def calculate_oos_metrics(y_true, y_pred, train_mean):
    """Calculates OOS R2 vs a Historical Mean Benchmark."""
    # 1. Model R2
    oos_r2 = r2_score(y_true, y_pred)
    
    # 2. Benchmark (Predicting the historical mean)
    y_bench = np.full_like(y_true, train_mean)
    bench_r2 = r2_score(y_true, y_bench)
    
    print(f"\n--- Statistical Performance (Requirement Checklist) ---")
    print(f"Model OOS R2:     {oos_r2:.5f}")
    print(f"Benchmark OOS R2: {bench_r2:.5f}")
    
    if oos_r2 > bench_r2:
        print("RESULT: Random Forest BEATS the Benchmark!")
    else:
        print("RESULT: Random Forest UNDERPERFORMS the Benchmark (Common in daily crypto).")

def get_feature_importance(model, feature_names):
    """Extracts RF feature importance."""
    if hasattr(model, 'named_steps'):
        regressor = model.named_steps['model']
    else:
        regressor = model

    if hasattr(regressor, 'feature_importances_'):
        importances = regressor.feature_importances_
    else:
        print("Model does not expose feature importance.")
        return

    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feat_imp = feat_imp.sort_values('importance', ascending=False).head(20)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feat_imp, palette='viridis')
    plt.title('Top 20 Features (Random Forest)')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "importance_rf.png")
    plt.close()

def run_training():
    print(f"\n--- Running Pipeline: RANDOM FOREST ---")
    
    # 1. Load & Engineer
    train_data, oos_features = prepare_data()
    print(f"Training Data Shape: {train_data.shape}")
    
    # 2. X/y Split
    drop_cols = ['time', 'label']
    X_train = train_data.drop(columns=drop_cols)
    y_train = train_data['label']
    
    # FIX: Drop label from OOS to avoid shape mismatch
    X_oos = oos_features.drop(columns=drop_cols, errors='ignore')
    
    feature_names = X_train.columns
    
    # 3. Pipeline & Grid Search
    tscv = TimeSeriesSplit(n_splits=5)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    
    # --- ADJUSTMENT: FINANCIAL HYPERPARAMETERS ---
    # We force the trees to be "shallower" and "wider" to avoid overfitting noise.
    param_grid = {
        'model__n_estimators': [200], 
        
        # 1. Limit Depth: Don't let the tree grow too complex
        'model__max_depth': [3, 5], 
        
        # 2. Big Leaves: REQUIRE at least 50 days of data to make a decision
        # This prevents "memorizing" specific past events.
        'model__min_samples_leaf': [50], 
        
        # 3. Max Features: Force it to consider more options to find the true signal
        'model__max_features': ['sqrt'] 
    }

    print("Tuning hyperparameters...")
    grid = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    print(f"Best Params: {grid.best_params_}")
    
    # 4. Predict
    pred_is = best_model.predict(X_train)
    pred_oos = best_model.predict(X_oos)
    
    # 5. Save Predictions
    pd.DataFrame({'time': train_data['time'], 'signal': pred_is}).to_csv(OUTPUT_DIR / "pred_is.csv", index=False)
    pd.DataFrame({'time': oos_features['time'], 'signal': pred_oos}).to_csv(OUTPUT_DIR / "pred_oos.csv", index=False)
    
    # FIX: Overwrite Stale Labels for Reporting
    train_data[['time', 'label']].to_csv(LABEL_IS, index=False)
    oos_features[['time', 'label']].to_csv(LABEL_OOS, index=False)
    
    print(f"Success! Predictions saved.")
    
    # 6. Metrics & Importance
    get_feature_importance(best_model, feature_names)
    
    # Calculate OOS R2 vs Benchmark
    train_mean = y_train.mean()
    if 'label' in oos_features.columns:
        calculate_oos_metrics(oos_features['label'], pred_oos, train_mean)

# --- PLOTTING FUNCTION (Adapted for RF) ---
def plot_forecast_vs_actual():
    # 1. Load Data
    PRED_OOS_FILE = OUTPUT_DIR / "pred_oos.csv"
    LABEL_OOS_FILE = OUTPUT_DIR / "label_oos.csv"
    PLOT_FILE = OUTPUT_DIR / "plots" / "rf_forecast_vs_actual.png"

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
    PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_FILE, dpi=300)
    print(f"Plot saved to {PLOT_FILE}")
    # plt.show() # Uncomment if running locally

if __name__ == "__main__":
    run_training()
    plot_forecast_vs_actual()