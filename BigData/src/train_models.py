"""
src/train_models.py
Train Ridge and Random Forest models and export predictions.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Paths
OUTPUT_DIR = Path("outputs")
FEATURE_IS = OUTPUT_DIR / "feature_is.csv"
LABEL_IS = OUTPUT_DIR / "label_is.csv"
FEATURE_OOS = OUTPUT_DIR / "feature_oos.csv"
LABEL_OOS = OUTPUT_DIR / "label_oos.csv"

def load_and_merge(feature_path, label_path):
    """Load features and labels, merge on time to ensure alignment."""
    df_X = pd.read_csv(feature_path)
    df_y = pd.read_csv(label_path)
    
    # Merge on time so we don't mix up rows
    data = pd.merge(df_X, df_y, on='time')
    return data

def train_and_predict(model_type="ridge"):
    print(f"--- Training {model_type.upper()} Model ---")
    
    # 1. Load Data
    train_data = load_and_merge(FEATURE_IS, LABEL_IS)
    test_data = load_and_merge(FEATURE_OOS, LABEL_OOS) # We need this just for the features
    
    # 2. Prepare X (Features) and y (Target)
    # Drop 'time' and 'label' to get just features
    drop_cols = ['time', 'label']
    X_train = train_data.drop(columns=drop_cols)
    y_train = train_data['label']
    
    X_test = test_data.drop(columns=drop_cols)
    
    # 3. Preprocessing (Scaling is crucial for Ridge)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fill any NaNs that might crash the model
    X_train_scaled = np.nan_to_num(X_train_scaled)
    X_test_scaled = np.nan_to_num(X_test_scaled)

    # 4. Train Model
    if model_type == "ridge":
        model = Ridge(alpha=1.0)
    elif model_type == "rf":
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    
    model.fit(X_train_scaled, y_train)
    print(f"Model trained. R2 Score on Train: {model.score(X_train_scaled, y_train):.4f}")

    # 5. Generate Predictions
    pred_is = model.predict(X_train_scaled)
    pred_oos = model.predict(X_test_scaled)
    
    # 6. Save in format required by backtest_pipeline
    # Needs columns: 'time', 'signal'
    
    # Save In-Sample
    df_pred_is = pd.DataFrame({'time': train_data['time'], 'signal': pred_is})
    df_pred_is.to_csv(OUTPUT_DIR / "pred_is.csv", index=False)
    
    # Save Out-of-Sample
    df_pred_oos = pd.DataFrame({'time': test_data['time'], 'signal': pred_oos})
    df_pred_oos.to_csv(OUTPUT_DIR / "pred_oos.csv", index=False)
    
    print(f"Saved predictions to {OUTPUT_DIR}/pred_is.csv and pred_oos.csv")

if __name__ == "__main__":
    # CHANGE THIS to "ridge" or "rf" depending on what you want to test
    MODEL_TO_RUN = "rf" 
    
    train_and_predict(MODEL_TO_RUN)