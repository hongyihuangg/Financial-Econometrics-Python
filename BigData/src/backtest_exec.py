"""Trading simulation using one-day holding after quantile signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Trade:
    side: int
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    holding_days: int
    exit_reason: str
    trade_return: float


def build_position_series(
    df: pd.DataFrame,
    time_col: str,
    open_col: str,
    signal_col: str,
    q_df: pd.DataFrame,
    min_history: int = 30,
) -> Tuple[pd.Series, List[Trade]]:
    times = df[time_col].to_numpy()
    opens = df[open_col].to_numpy()
    signal = df[signal_col].to_numpy()
    q10 = q_df["q10"].to_numpy()
    q90 = q_df["q90"].to_numpy()

    n = len(df)
    pos = np.zeros(n, dtype=float)
    trades: List[Trade] = []

    for i in range(n - 2):
        if i < min_history or np.isnan(q10[i]) or np.isnan(q90[i]):
            continue

        sig = signal[i]
        if np.isnan(sig):
            continue

        entry_idx = i + 1
        exit_idx = i + 2
        entry_price = opens[entry_idx]
        exit_price = opens[exit_idx]

        if sig >= q90[i]:
            pos[entry_idx] = 1
            trades.append(
                Trade(
                    side=1,
                    entry_time=times[entry_idx],
                    entry_price=float(entry_price),
                    exit_time=times[exit_idx],
                    exit_price=float(exit_price),
                    holding_days=1,
                    exit_reason="hold_one_day",
                    trade_return=float(exit_price / entry_price - 1.0),
                )
            )
        elif sig <= q10[i]:
            pos[entry_idx] = -1
            trades.append(
                Trade(
                    side=-1,
                    entry_time=times[entry_idx],
                    entry_price=float(entry_price),
                    exit_time=times[exit_idx],
                    exit_price=float(exit_price),
                    holding_days=1,
                    exit_reason="hold_one_day",
                    trade_return=float(entry_price / exit_price - 1.0),
                )
            )
    pos_series = pd.Series(pos, index=df.index, name="position")
    return pos_series, trades


def simulate_strategy(
    df: pd.DataFrame,
    time_col: str,
    open_col: str,
    signal_col: str,
    q_df: pd.DataFrame,
    min_history: int = 30,
    fee_bps: float = 0.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    pos, trades = build_position_series(
        df,
        time_col=time_col,
        open_col=open_col,
        signal_col=signal_col,
        q_df=q_df,
        min_history=min_history,
    )

    open_prices = df[open_col]
    daily_ret = pos * (open_prices.shift(-1) / open_prices - 1.0)
    daily_ret = daily_ret.dropna()

    if fee_bps:
        fee = fee_bps / 10000.0
        fee_adj = pd.Series(0.0, index=daily_ret.index)
        for t in trades:
            entry_idx = df.index[df[time_col] == t.entry_time]
            exit_idx = df.index[df[time_col] == t.exit_time]
            if len(entry_idx) > 0:
                fee_adj.loc[entry_idx[0]] -= fee
            if len(exit_idx) > 0:
                fee_adj.loc[exit_idx[0]] -= fee
        daily_ret = daily_ret.add(fee_adj, fill_value=0.0)

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    return trades_df, daily_ret
