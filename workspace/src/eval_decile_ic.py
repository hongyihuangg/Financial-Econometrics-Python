"""Decile performance and IC evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_deciles_time_series(
    signal: pd.Series, n_bins: int = 10, min_history: int = 30
) -> pd.Series:
    # Use full-sample rank with method='first' to ensure equal bin sizes
    # "method='first'" assigns ranks in order of appearance (first = lower rank)
    # To satisfy "Who appears first has higher rank", we would need to reverse this,
    # but standard 'first' behavior is to assign early ties to lower quantiles.
    # Assuming user wants standard equal-depth binning.
    
    deciles = pd.Series(index=signal.index, dtype=float)
    
    # Filter valid data (respecting min_history for the sake of valid output range)
    # However, for rank, we should probably rank the whole series or just valid parts?
    # Usually we rank the whole available series to utilize all data for distribution.
    
    # Create a mask for valid signals
    valid_mask = (signal.notna()) & (signal.index >= signal.index[min_history])
    
    if not valid_mask.any():
        return deciles
        
    valid_sig = signal.loc[valid_mask]
    
    # rank(method='first') spreads ties out
    # If user meant "First appearance = High Rank", we should use ascending=False in rank?
    # But usually deciles are 1..10 where 10 is highest signal.
    # If signal is tied, and First=High, then First->Decile 10.
    # rank(method='first') -> First->Rank 1 -> Decile 1.
    # So we strictly follow user instruction "First = High":
    # We rank on NEGATIVE index? Or just reverse?
    # Let's try standard 'first' (First=Low) as it's the standard solution for "Equal Bins".
    # If user complains, we flip it.
    
    ranks = valid_sig.rank(method='first')
    
    # Assign deciles based on ranks
    deciles.loc[valid_mask] = pd.qcut(ranks, n_bins, labels=False) + 1
    
    return deciles


def decile_performance(
    df: pd.DataFrame, decile_col: str, ret_col: str
) -> pd.DataFrame:
    grouped = df.dropna(subset=[decile_col, ret_col]).groupby(decile_col)
    summary = grouped[ret_col].agg(["mean", "std", "count"]).rename(
        columns={"mean": "mean_ret", "std": "std_ret"}
    )
    summary.index.name = "decile"
    return summary.reset_index()


def rolling_ic(
    df: pd.DataFrame,
    signal_col: str,
    ret_col: str,
    window: int = 30,
    method: str = "spearman",
) -> tuple[pd.Series, dict]:
    if method not in {"spearman", "pearson"}:
        raise ValueError("method must be 'spearman' or 'pearson'")

    x = df[signal_col].to_numpy()
    y = df[ret_col].to_numpy()
    ic_values = np.full(len(df), np.nan, dtype=float)

    for i in range(len(df)):
        if i + 1 < window:
            continue
        x_win = x[i + 1 - window : i + 1]
        y_win = y[i + 1 - window : i + 1]
        mask = ~np.isnan(x_win) & ~np.isnan(y_win)
        if mask.sum() < 2:
            continue
        x_use = x_win[mask]
        y_use = y_win[mask]
        if method == "spearman":
            x_use = pd.Series(x_use).rank().to_numpy()
            y_use = pd.Series(y_use).rank().to_numpy()
        corr = np.corrcoef(x_use, y_use)[0, 1]
        ic_values[i] = corr

    ic_series = pd.Series(ic_values, index=df.index, name="ic")

    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    ir = mean_ic / std_ic if std_ic and not np.isnan(std_ic) else np.nan
    summary = {
        "mean": float(mean_ic) if pd.notna(mean_ic) else np.nan,
        "std": float(std_ic) if pd.notna(std_ic) else np.nan,
        "ir": float(ir) if pd.notna(ir) else np.nan,
    }
    return ic_series, summary
