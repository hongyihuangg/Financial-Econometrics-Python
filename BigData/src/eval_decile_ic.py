"""Decile performance and IC evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_deciles_time_series(
    signal: pd.Series, n_bins: int = 10, min_history: int = 30
) -> pd.Series:
    deciles = pd.Series(index=signal.index, dtype=float)
    values = signal.to_numpy()
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(len(signal)):
        if i < min_history:
            deciles.iloc[i] = np.nan
            continue
        hist = values[:i]
        hist = hist[~np.isnan(hist)]
        if len(hist) == 0:
            deciles.iloc[i] = np.nan
            continue
        cutpoints = np.quantile(hist, qs)
        val = values[i]
        if np.isnan(val):
            deciles.iloc[i] = np.nan
            continue
        bin_idx = np.searchsorted(cutpoints, val, side="right") - 1
        bin_idx = min(max(bin_idx, 0), n_bins - 1)
        deciles.iloc[i] = bin_idx + 1
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
