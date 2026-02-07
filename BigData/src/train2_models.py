"""
src/train_advanced.py
Fixed Version: Uses fillna(0) instead of dropna() to prevent empty output.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# --- CONFIGURATION ---
OUTPUT_DIR = Path("outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Paths
FEATURE_IS = OUTPUT_DIR / "feature_is.csv"
LABEL_IS = OUTPUT_DIR / "label_is.csv"
FEATURE_OOS = OUTPUT_DIR / "feature_oos.csv"
# Note: We do NOT load LABEL_OOS. The model must never know the OOS answers.

def prepare_data(feature_path, label_path=None, is_training=True):
    """
    Loads features and applies 1-day lag to fix Look-Ahead Bias.
    """
    df_feat = pd.read_csv(feature_path)
    
    # --- CRITICAL FIX: LAG FEATURES ---
    # We shift features down by 1 row so row T has data from T-1.
    time_col = df_feat['time']
    df_feat_lagged = df_feat.drop(columns=['time']).shift(1)
    df_feat_lagged['time'] = time_col
    
    # --- BUG FIX: FILL MISSING VALUES ---
    # Instead of dropping rows with missing data (which wipes out the file),
    # we fill them with 0. 
    # We only drop the very first row (index 0) because the shift made it pure NaN.
    df_feat_lagged = df_feat_lagged.iloc[1:] # Drop first row only
    df_feat_lagged = df_feat_lagged.fillna(0) # Fill gaps with 0

    # If this is for prediction (OOS), return features
    if not is_training:
        return df_feat_lagged

    # If this is for training (IS), merge with labels
    if label_path:
        df_label = pd.read_csv(label_path)
        # Inner merge aligns the lagged features with the target
        df_merged = pd.merge(df_feat_lagged, df_label, on='time')
        return df_merged
    
    return None

def get_feature_importance(model, feature_names, model_name):
    """Extracts and plots feature importance."""
    if hasattr(model, 'coef_'): # Ridge/Lasso
        importances = model.coef_
    elif hasattr(model, 'feature_importances_'): # Random Forest
        importances = model.feature_importances_
    else:
        return
        
    # Create DataFrame
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feat_imp['abs_importance'] = feat_imp['importance'].abs()
    feat_imp = feat_imp.sort_values('abs_importance', ascending=False).head(20)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feat_imp, palette='viridis')
    plt.title(f'Top 20 Features - {model_name}')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"importance_{model_name}.png")
    plt.close()
    print(f"Saved importance plot to {PLOTS_DIR}/importance_{model_name}.png")

def run_training(model_type="ridge"):
    print(f"\n--- Running Pipeline: {model_type.upper()} ---")
    
    # 1. Load In-Sample Data ONLY
    print("Loading In-Sample (Training) data...")
    train_data = prepare_data(FEATURE_IS, LABEL_IS, is_training=True)
    
    # 2. Separate X and y
    drop_cols = ['time', 'label']
    X_train = train_data.drop(columns=drop_cols)
    y_train = train_data['label']
    feature_names = X_train.columns
    
    # 3. Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 4. Hyperparameter Tuning
    tscv = TimeSeriesSplit(n_splits=5)
    
    if model_type == "ridge":
        model = Ridge()
        param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0, 500.0]}
    elif model_type == "rf":
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [50, 100], 
            'max_depth': [3, 5],
            'min_samples_leaf': [2, 10]
        }

    print(f"Tuning hyperparameters using TimeSeriesSplit...")
    grid = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    
    best_model = grid.best_estimator_
    print(f"Best Parameters: {grid.best_params_}")
    
    # 5. Generate Predictions
    # A) In-Sample
    pred_is = best_model.predict(X_train_scaled)
    
    # B) Out-of-Sample
    print("Loading OOS features for prediction...")
    oos_features = prepare_data(FEATURE_OOS, is_training=False)
    X_oos = oos_features.drop(columns=['time'])
    
    # Apply Scaler
    X_oos_scaled = scaler.transform(X_oos)
    pred_oos = best_model.predict(X_oos_scaled)
    
    # 6. Save Predictions
    pd.DataFrame({'time': train_data['time'], 'signal': pred_is}).to_csv(OUTPUT_DIR / "pred_is.csv", index=False)
    
    # Ensure OOS length matches
    print(f"Generated {len(pred_oos)} predictions for OOS.")
    pd.DataFrame({'time': oos_features['time'], 'signal': pred_oos}).to_csv(OUTPUT_DIR / "pred_oos.csv", index=False)
    
    print("Saved predictions.")
    
    # 7. Analyze Features
    get_feature_importance(best_model, feature_names, model_type)

if __name__ == "__main__":
    # --- CHANGE THIS TO "rf" TO RUN RANDOM FOREST ---
    run_training("rf")