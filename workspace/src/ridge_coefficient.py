"""
src/explain_ridge_coeffs.py
Generates the Coefficient Importance Plot for Ridge.
(The theoretically 'correct' way to interpret Linear Models)
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# --- CONFIGURATION ---
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Update this with your specific Alpha from the Bayesian search
BEST_ALPHA = 981.790675  # <--- UPDATE THIS MANUALLY

def get_data():
    # Load and prep (Simplified for brevity)
    df_is = pd.read_csv(DATA_DIR / "feature_is.csv")
    lbl_is = pd.read_csv(DATA_DIR / "label_is.csv")
    # Merge and simple fill
    df = pd.merge(df_is, lbl_is, on='time').ffill().dropna()
    # Keep numeric only (drop time/label)
    X = df.drop(columns=['time', 'label', 'BTCUSD__close'], errors='ignore')
    # Drop OHLC columns
    X = X[[c for c in X.columns if not c.endswith(('_open','_high','_low','_close'))]]
    y = df['label'].shift(-1).dropna()
    X = X.iloc[:-1] # Align with shifted label
    return X, y

def plot_coefficients():
    X, y = get_data()
    
    # Ridge REQUIRES scaling for coefficients to be comparable
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Train
    model = Ridge(alpha=BEST_ALPHA)
    model.fit(X_scaled, y)
    
    # Extract Coefficients
    coefs = pd.DataFrame({
        'Feature': X_scaled.columns,
        'Coefficient': model.coef_,
        'Abs_Coefficient': abs(model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False).head(15)
    
    # Plot
    plt.figure(figsize=(10, 6))
    # Color bar by positive/negative sign
    colors = ['red' if c < 0 else 'blue' for c in coefs['Coefficient']]
    sns.barplot(x='Coefficient', y='Feature', data=coefs, palette=colors)
    plt.title(f"Ridge Coefficients (Alpha={BEST_ALPHA})\nBlue = Buy Signal, Red = Sell Signal", fontsize=14)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    
    save_path = PLOTS_DIR / "ridge_coefficients.png"
    plt.savefig(save_path)
    print(f"Saved Coefficient Plot to {save_path}")

if __name__ == "__main__":
    plot_coefficients()