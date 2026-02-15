"""Preprocessing utilities for merging sources and labeling returns."""

from __future__ import annotations

from typing import Dict

import pandas as pd


def merge_sources(
    dfs: Dict[str, pd.DataFrame],
    base_source: str = "BTCUSD",
    time_col: str = "time",
) -> pd.DataFrame:
    if base_source not in dfs:
        raise ValueError(f"Base source '{base_source}' not found in data sources")
    base_df = dfs[base_source].copy()
    base_df = base_df.sort_values(time_col)
    for src, df in dfs.items():
        if src == base_source:
            continue
        base_df = base_df.merge(df, on=time_col, how="left")
    return base_df


def add_label_open_to_open(
    df: pd.DataFrame, open_col: str, close_col: str, label_col: str = "label"
) -> pd.DataFrame:
    """
    Compute label as Next Day's Intraday Return:
    Label[t] = Close[t+1] / Open[t+1] - 1.0
    """
    out = df.copy()
    # Shift(-1) gets t+1 data aligned to t
    next_open = out[open_col].shift(-1)
    next_close = out[close_col].shift(-1)
    
    out[label_col] = next_close / next_open - 1.0
    return out


def basic_sanity_checks(
    df: pd.DataFrame, open_col: str, label_col: str
) -> dict:
    meta = {
        "row_count": int(len(df)),
        "start": df["time"].min(),
        "end": df["time"].max(),
    }
    meta["missing_rates"] = (
        df.isna().mean().sort_values(ascending=False).to_dict()
    )
    open_series = df[open_col]
    meta["open_le_zero_count"] = int((open_series <= 0).sum())
    meta["label_nan_count"] = int(df[label_col].isna().sum())
    return meta


def winsorize_series(
    series: pd.Series, lower_quantile: float = 0.01, upper_quantile: float = 0.99
) -> pd.Series:
    """Winsorize a series by clipping values at the specified quantiles."""
    lower_limit = series.quantile(lower_quantile)
    upper_limit = series.quantile(upper_quantile)
    return series.clip(lower=lower_limit, upper=upper_limit)
