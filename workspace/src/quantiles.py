"""Expanding quantile computation with no leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_expanding_quantiles(
    signal: pd.Series,
    qs: tuple[float, float, float] = (0.1, 0.5, 0.9),
    min_history: int = 30,
) -> pd.DataFrame:
    q_labels = [f"q{int(q * 100)}" for q in qs]
    out = pd.DataFrame(index=signal.index, columns=q_labels, dtype=float)

    values = signal.to_numpy()
    for i in range(len(signal)):
        if i < min_history:
            out.iloc[i] = np.nan
            continue
        hist = values[:i]
        hist = hist[~np.isnan(hist)]
        if len(hist) == 0:
            out.iloc[i] = np.nan
            continue
        quantiles = np.quantile(hist, qs)
        out.iloc[i] = quantiles
    return out
