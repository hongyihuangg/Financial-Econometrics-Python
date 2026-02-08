"""
src/train_advanced.py
Fixed Pipeline: Handles NaN values and Date Mismatches preventing empty data errors.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# --- CONFIGURATION ---
OUTPUT_DIR = Path("outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_IS = OUTPUT_DIR / "feature_is.csv"
LABEL_IS = OUTPUT_DIR / "label_is.csv"
FEATURE_OOS = OUTPUT_DIR / "feature_oos.csv"

def keep_close_only(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop Open/High/Low to reduce noise
    drop_cols = [c for c in df.columns if c.endswith(("_open", "_high", "_low"))]
    return df.drop(columns=drop_cols, errors="ignore")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("--- Engineering Features ---")
    df = df.copy()
    
    # 1. Forward Fill Macro Data (CRITICAL FIX)
    # Macro data often comes weekly/monthly, leaving NaNs in daily data.
    # We fill forward so 'today' uses the most recent known value.
    df = df.ffill()
    
    # 2. Setup Column Names
    col_btc_close = "BTCUSD__close"
    col_nasdaq = "IG_NASDAQ_1D__close"
    col_vix = "TVC_VIX_1D__close"
    col_us10y = "TVC_US10Y_1D__close"
    col_us02y = "TVC_US02Y_1D__close"
    col_us03m = "TVC_US03MY_1D__close"
    
    # 3. Calculate Returns (Stationarity)
    # Use 'fill_method=None' to avoid warnings in newer pandas
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
        
        # Trend Slope (Short MA - Long MA)
        if len(present_mas) >= 2:
            short_ma = present_mas[0]
            long_ma = present_mas[-1]
            df["TREND_SLOPE"] = (df[short_ma] - df[long_ma]) / df[col_btc_close]

    # 6. Drop Raw Levels (Non-Stationary)
    drop_cols = [col_btc_close, col_nasdaq, col_vix, col_us10y, col_us02y, col_us03m] + present_mas
    drop_cols += ["ECONOMICS_USCBBS_1D__close", "FRED_M2SL_1D__close"]
    
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return df

def prepare_data():
    """Loads, Merges, Engineers Features."""
    print("Loading data...")
    df_is = pd.read_csv(FEATURE_IS)
    df_oos = pd.read_csv(FEATURE_OOS)
    
    # Format Dates
    df_is['time'] = pd.to_datetime(df_is['time'])
    df_oos['time'] = pd.to_datetime(df_oos['time'])

    # Combine
    df_is['split'] = 'train'
    df_oos['split'] = 'test'
    full_df = pd.concat([df_is, df_oos], axis=0).sort_values('time').reset_index(drop=True)
    
    # 1. Clean Data
    full_df = keep_close_only(full_df)

    # --- CRITICAL FIX: CALCULATE LABEL FIRST ---
    # We must do this BEFORE build_features() because that function deletes the Price column!
    # This predicts the return from Close(T) to Close(T+1)
    full_df['label'] = full_df['BTCUSD__close'].pct_change().shift(-1)

    # 2. Engineer Features (Now safe to drop columns)
    full_df = build_features(full_df)
    
    # 3. Clean NaNs
    # Drop the LAST row (no tomorrow to predict)
    full_df = full_df.dropna(subset=['label'])
    
    # Fill feature NaNs
    meta_cols = ['time', 'split', 'label', 'BTCUSD__close']
    feature_cols = [c for c in full_df.columns if c not in meta_cols]
    full_df[feature_cols] = full_df[feature_cols].ffill().fillna(0)
    
    # 4. Split back to Train/Test
    train_df = full_df[full_df['split'] == 'train'].drop(columns=['split']).copy()
    test_df = full_df[full_df['split'] == 'test'].drop(columns=['split']).copy()

    if len(train_df) == 0:
        raise ValueError("Training Data is empty!")

    return train_df, test_df

def get_feature_importance(model, feature_names, model_name):
    """Extracts and plots feature importance from a Pipeline."""
    
    # Access the actual model step from the pipeline
    if hasattr(model, 'named_steps'):
        regressor = model.named_steps['model']
    else:
        regressor = model

    # Extract importance
    if hasattr(regressor, 'coef_'): # Ridge
        importances = regressor.coef_
    elif hasattr(regressor, 'feature_importances_'): # Random Forest
        importances = regressor.feature_importances_
    else:
        print("Model does not expose feature importance.")
        return

    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feat_imp['abs_importance'] = feat_imp['importance'].abs()
    feat_imp = feat_imp.sort_values('abs_importance', ascending=False).head(20)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feat_imp, palette='viridis')
    plt.title(f'Top 20 Features ({model_name})')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"importance_{model_name}.png")
    plt.close()

from sklearn.pipeline import Pipeline

def run_training(model_type="ridge"):
    print(f"\n--- Running Pipeline: {model_type.upper()} ---")
    
    # 1. Load & Engineer
    train_data, oos_features = prepare_data()
    print(f"Training Data Shape: {train_data.shape}")
    
    # 2. X/y Split (Keep X Raw!)
    drop_cols = ['time', 'label']
    
    X_train = train_data.drop(columns=drop_cols)
    y_train = train_data['label']
    
    # FIX 1: Drop 'label' from OOS data too so columns match exactly
    X_oos = oos_features.drop(columns=drop_cols, errors='ignore')
    
    # FIX 2: Define feature_names explicitly so the plotter can use it later
    feature_names = X_train.columns
    
    # 3. Setup Pipeline (Scale INSIDE the Cross-Validation)
    tscv = TimeSeriesSplit(n_splits=5)
    
    if model_type == "ridge":
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge())
        ])
        # Note the double underscore: 'step_name__parameter'
        param_grid = {'model__alpha': [0.01, 0.1, 1.0, 10.0]} 

    elif model_type == "rf":
        pipeline = Pipeline([
            ('scaler', StandardScaler()), # Optional for RF, but good practice
            ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
        ])
        param_grid = {
            'model__n_estimators': [200], 
            'model__max_depth': [5, 10]
        }

    print("Tuning hyperparameters...")
    # 4. Train
    grid = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train) # Pass RAW X_train
    
    best_model = grid.best_estimator_
    print(f"Best Params: {grid.best_params_}")
    
    # 5. Predict (Pass RAW data - the Pipeline handles scaling automatically)
    pred_is = best_model.predict(X_train)
    pred_oos = best_model.predict(X_oos)
    
    # Save Outputs
    pd.DataFrame({'time': train_data['time'], 'signal': pred_is}).to_csv(OUTPUT_DIR / "pred_is.csv", index=False)
    pd.DataFrame({'time': oos_features['time'], 'signal': pred_oos}).to_csv(OUTPUT_DIR / "pred_oos.csv", index=False)
    
    # --- INSERT THESE 2 LINES HERE ---
    # Overwrite the old label files with the correct 'shifted' target
    # This aligns the report plots with what the model actually learned
    train_data[['time', 'label']].to_csv(LABEL_IS, index=False)
    oos_features[['time', 'label']].to_csv(OUTPUT_DIR / "label_oos.csv", index=False)
    # ---------------------------------

    print(f"Success! {len(pred_oos)} predictions saved.")
    
    # FIX 3: Pass the defined feature_names to the plotter
    get_feature_importance(best_model, feature_names, model_type)

if __name__ == "__main__":
    # Change to "rf" to run Random Forest next
    run_training("ridge")

    import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- CONFIG ---
OUTPUT_DIR = Path("outputs")
PRED_OOS_FILE = OUTPUT_DIR / "pred_oos.csv"
LABEL_OOS_FILE = OUTPUT_DIR / "label_oos.csv"
PLOT_FILE = OUTPUT_DIR / "plots" / "ridge_forecast_vs_actual.png"

def plot_forecast_vs_actual():
    # 1. Load Data
    print("Loading OOS data...")
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
    
    # Plot Ridge Forecast (Orange, smooth)
    plt.plot(df['time'], df['signal'], 
             label='Ridge Forecast', 
             color='#ff7f0e', 
             marker='.', 
             markersize=4, 
             linestyle='-', 
             linewidth=2)

    # Style
    plt.title("Ridge OOS Returns: Forecast vs Actual", fontsize=14)
    plt.ylabel("Return", fontsize=12)
    plt.xlabel("Time", fontsize=12)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Optional: Zoom in to the "Crash" area if you know the date
    # plt.xlim(pd.Timestamp('2025-01-01'), pd.Timestamp('2025-06-01'))

    plt.tight_layout()
    
    # Ensure plots dir exists
    PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_FILE, dpi=300)
    print(f"Plot saved to {PLOT_FILE}")
    plt.show()

if __name__ == "__main__":
    plot_forecast_vs_actual()

