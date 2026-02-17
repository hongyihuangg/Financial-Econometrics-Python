"""
src/market_benchmark.py
Calculates and plots the performance of Benchmarks (NASDAQ & Bitcoin).
Input: 
 - data/feature_oos.csv (IG_NASDAQ_1D__close)
 - data/label_oos.csv (Bitcoin Returns)
Output: Metrics printout & Cumulative Return Plot
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURATION ---
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MARKET_FILE = DATA_DIR / "feature_oos.csv"
LABEL_FILE = DATA_DIR / "label_oos.csv"

def calculate_metrics(series, name="Asset"):
    """Calculates annualized risk-adjusted metrics."""
    # 252 for stocks, 365 for crypto. We'll use 252 to be conservative/standard.
    ann_factor = 252 
    
    total_ret = (1 + series).prod() - 1
    mean_ret = series.mean() * ann_factor
    volatility = series.std() * np.sqrt(ann_factor)
    sharpe = mean_ret / volatility if volatility != 0 else 0
    
    # Max Drawdown
    cum_ret = (1 + series).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()
    
    return {
        "Metric": name,
        "Total Return": f"{total_ret:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Volatility": f"{volatility:.2%}"
    }

def run_market_analysis():
    print("Loading Benchmark Data...")
    
    # 1. Load NASDAQ (Feature File)
    mkt_df = pd.read_csv(MARKET_FILE)
    mkt_df['time'] = pd.to_datetime(mkt_df['time'])
    
    # 2. Load Bitcoin Returns (Label File)
    btc_df = pd.read_csv(LABEL_FILE)
    btc_df['time'] = pd.to_datetime(btc_df['time'])
    
    # 3. Merge them
    df = pd.merge(mkt_df[['time', 'IG_NASDAQ_1D__close']], btc_df, on='time', how='inner')
    
    # 4. Calculate Returns
    # NASDAQ: Percentage Change
    df['NASDAQ_Ret'] = df['IG_NASDAQ_1D__close'].pct_change().fillna(0)
    
    # Bitcoin: Already in 'label' column (Check if it needs shifting based on your generation logic)
    # Usually 'label' is the target return. We treat it as the Buy & Hold return here.
    df['BTC_Ret'] = df['label']
    
    # 5. Generate Metrics
    stats = []
    stats.append(calculate_metrics(df['NASDAQ_Ret'], "NASDAQ 100 (Market)"))
    stats.append(calculate_metrics(df['BTC_Ret'], "Bitcoin (Buy & Hold)"))
    
    stats_df = pd.DataFrame(stats)
    print("\n--- Benchmark Performance (OOS) ---")
    print(stats_df.to_string(index=False))
    
    # Save Metrics
    stats_df.to_csv(OUTPUT_DIR / "benchmark_metrics.csv", index=False)
    
    # 6. Plot Cumulative Returns
    df['Cum_NASDAQ'] = (1 + df['NASDAQ_Ret']).cumprod()
    df['Cum_BTC'] = (1 + df['BTC_Ret']).cumprod()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df['Cum_NASDAQ'], label='NASDAQ 100', color='#1f77b4', linewidth=2)
    plt.plot(df['time'], df['Cum_BTC'], label='Bitcoin (Buy & Hold)', color='gray', linewidth=1.5, alpha=0.7)
    
    plt.title("Benchmarks: Market vs. Asset (OOS Period)", fontsize=14)
    plt.ylabel("Cumulative Growth ($1 Invested)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = PLOTS_DIR / "market_benchmark_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved to: {save_path}")

if __name__ == "__main__":
    run_market_analysis()