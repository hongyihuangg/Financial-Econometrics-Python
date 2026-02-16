"""
src/explain_model.py
Model Interpretation using SHAP:
1. Re-trains the best Restricted RF model (to ensure we have the exact object).
2. Calculates SHAP values.
3. Generates Summary and Dependence plots.
"""
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path

# --- CONFIGURATION ---
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_IS = DATA_DIR / "feature_is.csv"
FEATURE_OOS = DATA_DIR / "feature_oos.csv"
LABEL_IS = DATA_DIR / "label_is.csv"
LABEL_OOS = DATA_DIR / "label_oos.csv"

# --- HELPER FUNCTIONS (Same as before) ---
def keep_close_only(df):
    drop_cols = [c for c in df.columns if c.endswith(("_open", "_high", "_low"))]
    return df.drop(columns=drop_cols, errors="ignore")

def build_features(df):
    df = df.copy().ffill()
    # (Simplified feature build for brevity - ensure this matches your training script)
    tradfi_cols = ["IG_NASDAQ_1D__close", "TVC_VIX_1D__close", "TVC_US10Y_1D__close", "TVC_US02Y_1D__close"]
    df[[c for c in tradfi_cols if c in df.columns]] = df[[c for c in tradfi_cols if c in df.columns]].ffill()
    
    col_btc = "BTCUSD__close"
    if "IG_NASDAQ_1D__close" in df.columns: df["NASDAQ_ret"] = df["IG_NASDAQ_1D__close"].pct_change()
    if "TVC_VIX_1D__close" in df.columns: df["VIX_ret"] = df["TVC_VIX_1D__close"].pct_change()
    
    vol_col = next((c for c in ["BTCUSD__volume", "BTCUSD_volume", "volume"] if c in df.columns), None)
    if vol_col: df["VOL_REL_MA"] = (df[vol_col] / df[vol_col].rolling(20).mean()) - 1.0
    
    if "TVC_US10Y_1D__close" in df.columns and "TVC_US02Y_1D__close" in df.columns:
        df["TERM_10Y_2Y"] = df["TVC_US10Y_1D__close"] - df["TVC_US02Y_1D__close"]
        
    ma_cols = ["BTCUSD__MA", "BTCUSD__MA.1", "BTCUSD__MA.2"]
    present_mas = [c for c in ma_cols if c in df.columns]
    if col_btc in df.columns and present_mas:
        for ma in present_mas: df[f"DIST_{ma}"] = (df[col_btc] / df[ma]) - 1.0
    
    if col_btc in df.columns:
        delta = df[col_btc].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 0.001)
        df['RSI_14'] = (100 - (100 / (1 + rs))) / 100.0 - 0.5 

    drop_cols = [col_btc] + tradfi_cols + present_mas + ["ECONOMICS_USCBBS_1D__close", "FRED_M2SL_1D__close"]
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

def prepare_data():
    print("Loading Data...")
    df_is = pd.read_csv(FEATURE_IS)
    lbl_is = pd.read_csv(LABEL_IS)
    
    for d in [df_is, lbl_is]: d['time'] = pd.to_datetime(d['time'])
    full_df = pd.merge(df_is, lbl_is, on='time', how='inner')
    full_df = keep_close_only(full_df)
    full_df['label'] = full_df['label'].shift(-1) # Fix Leakage
    full_df = build_features(full_df).dropna()
    return full_df

def run_shap_analysis():
    # 1. Prepare Training Data
    df = prepare_data()
    X = df.drop(columns=['time', 'label'])
    y = df['label']
    
    # 2. Train the "Winner" Model (Updated with your specific params)
    best_params = {
        'n_estimators': 200,            # Keep fixed
        'max_depth': 3,                 # Your Result
        'min_samples_leaf': 112,        # Your Result
        'max_features': 0.13005340686974226 # Your Result
    }
    
    print(f"Training Model with params: {best_params}...")
    model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # 3. Calculate SHAP Values
    print("Calculating SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # 4. Plot 1: Summary Plot (Global Importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Feature Importance (Global)", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary.png")
    print(f"Saved Summary Plot to {PLOTS_DIR / 'shap_summary.png'}")
    plt.close()
    
    # 5. Plot 2: Dependence Plot (Local Logic) for Top Feature
    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    top_feature = feature_importance.iloc[0]['col_name']
    
    print(f"Generating Dependence Plot for Top Feature: {top_feature}")
    
    # Create dependence plot manually to save it
    shap.dependence_plot(top_feature, shap_values, X, show=False)
    plt.title(f"SHAP Dependence: {top_feature}", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"shap_dependence_{top_feature}.png")
    print(f"Saved Dependence Plot to {PLOTS_DIR}")

if __name__ == "__main__":
    run_shap_analysis()