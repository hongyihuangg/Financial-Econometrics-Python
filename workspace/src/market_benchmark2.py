"""
src/market_benchmark.py
Calculates and plots the performance of Benchmarks:
1. NASDAQ 100 (Market)
2. Bitcoin (Buy & Hold - The "Lazy" Crypto approach)
3. Bitcoin (Timed Exit - The "Perfect" Crypto approach, selling at the absolute top)
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
    
    # 1. Load Data
    mkt_df = pd.read_csv(MARKET_FILE)
    mkt_df['time'] = pd.to_datetime(mkt_df['time'])
    
    btc_df = pd.read_csv(LABEL_FILE)
    btc_df['time'] = pd.to_datetime(btc_df['time'])
    
    df = pd.merge(mkt_df[['time', 'IG_NASDAQ_1D__close']], btc_df, on='time', how='inner')
    
    # 2. Base Returns
    df['NASDAQ_Ret'] = df['IG_NASDAQ_1D__close'].pct_change().fillna(0)
    df['BTC_Ret'] = df['label'] # Buy and Hold
    
    # 3. CREATE "TIMED EXIT" STRATEGY
    # Find the cumulative peak
    df['Cum_BTC'] = (1 + df['BTC_Ret']).cumprod()
    peak_date = df.loc[df['Cum_BTC'].idxmax(), 'time']
    peak_val = df['Cum_BTC'].max()
    
    print(f"\nDetected Market Top: {peak_date.date()} (Return: {(peak_val-1):.2%})")
    print("Creating 'Timed Exit' strategy (Sell to Cash on this date)...")
    
    # Create the strategy: Equal to BTC before peak, 0.0 (Cash) after peak
    df['BTC_Timed_Exit'] = np.where(df['time'] <= peak_date, df['BTC_Ret'], 0.0)
    
    # 4. Generate Metrics
    stats = []
    stats.append(calculate_metrics(df['NASDAQ_Ret'], "NASDAQ 100 (Passive)"))
    stats.append(calculate_metrics(df['BTC_Ret'], "Bitcoin (Hold to End)"))
    stats.append(calculate_metrics(df['BTC_Timed_Exit'], f"Bitcoin (Sold on {peak_date.date()})"))
    
    stats_df = pd.DataFrame(stats)
    print("\n--- Benchmark Performance (OOS) ---")
    print(stats_df.to_string(index=False))
    
    stats_df.to_csv(OUTPUT_DIR / "benchmark_metrics.csv", index=False)
    
    # 5. Plot
    df['Cum_NASDAQ'] = (1 + df['NASDAQ_Ret']).cumprod()
    df['Cum_BTC'] = (1 + df['BTC_Ret']).cumprod()
    df['Cum_Timed'] = (1 + df['BTC_Timed_Exit']).cumprod()
    
    plt.figure(figsize=(12, 7))
    
    # Plot Lines
    plt.plot(df['time'], df['Cum_Timed'], label=f'Bitcoin (Timed Exit)', color='#d62728', linewidth=2.5, linestyle='-') # Red
    plt.plot(df['time'], df['Cum_NASDAQ'], label='NASDAQ 100', color='#1f77b4', linewidth=2) # Blue
    plt.plot(df['time'], df['Cum_BTC'], label='Bitcoin (Hold Forever)', color='gray', linewidth=1, alpha=0.5, linestyle='--') # Gray Faded
    
    # Annotate the "Sell" point
    plt.scatter([peak_date], [peak_val], color='black', zorder=5)
    plt.annotate(f'Exit Point\n{peak_date.date()}', xy=(peak_date, peak_val), xytext=(peak_date, peak_val+0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05), ha='center')

    plt.title("Comparison: Market vs. Asset vs. Perfect Timing", fontsize=14)
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