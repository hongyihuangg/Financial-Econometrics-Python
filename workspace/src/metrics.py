"""Metrics for trade-level and daily-level performance."""

from __future__ import annotations

import numpy as np
import pandas as pd


def trade_level_metrics(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {
            "avg_trade_return": np.nan,
            "total_trade_return": np.nan,
            "long_avg": np.nan,
            "long_total": np.nan,
            "short_avg": np.nan,
            "short_total": np.nan,
            "avg_holding_days": np.nan,
            "win_rate": np.nan,
            "payoff_ratio": np.nan,
            "compounded": True,
        }

    returns = trades_df["trade_return"]
    total_compounded = float((1.0 + returns).prod() - 1.0)

    long_trades = trades_df[trades_df["side"] == 1]
    short_trades = trades_df[trades_df["side"] == -1]

    avg_win = returns[returns > 0].mean()
    avg_loss = returns[returns < 0].mean()
    payoff_ratio = (
        float(avg_win / abs(avg_loss)) if avg_loss is not None and avg_loss < 0 else np.nan
    )

    metrics = {
        "avg_trade_return": float(returns.mean()),
        "total_trade_return": total_compounded,
        "long_avg": float(long_trades["trade_return"].mean()) if not long_trades.empty else np.nan,
        "long_total": float((1.0 + long_trades["trade_return"]).prod() - 1.0)
        if not long_trades.empty
        else np.nan,
        "short_avg": float(short_trades["trade_return"].mean()) if not short_trades.empty else np.nan,
        "short_total": float((1.0 + short_trades["trade_return"]).prod() - 1.0)
        if not short_trades.empty
        else np.nan,
        "avg_holding_days": float(trades_df["holding_days"].mean()),
        "win_rate": float((returns > 0).mean()),
        "payoff_ratio": payoff_ratio,
        "compounded": True,
    }
    return metrics


def _max_drawdown(cum_returns: pd.Series) -> float:
    if cum_returns.empty:
        return np.nan
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1.0
    return float(drawdown.min())


def daily_level_metrics(daily_ret_series: pd.Series, ann_factor: int = 365) -> dict:
    if daily_ret_series.empty:
        return {
            "mean_daily": np.nan,
            "std_daily": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }

    mean_daily = daily_ret_series.mean()
    std_daily = daily_ret_series.std()
    sharpe = (
        float(mean_daily / std_daily * np.sqrt(ann_factor)) if std_daily else np.nan
    )

    cum_returns = (1.0 + daily_ret_series).cumprod()
    return {
        "mean_daily": float(mean_daily),
        "std_daily": float(std_daily),
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(cum_returns),
    }


def merge_metrics(a: dict, b: dict) -> dict:
    out = dict(a)
    out.update(b)
    return out
