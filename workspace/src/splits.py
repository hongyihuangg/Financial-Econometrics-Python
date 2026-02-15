"""Dataset split utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .config import (
    DEV_END,
    IS_END,
    IS_TEST_START,
    OOS_END,
    OOS_START,
    START_DATE,
)


def make_splits() -> Dict[str, Tuple[str, str | None]]:
    return {
        "is": (START_DATE, IS_END),
        "oos": (OOS_START, OOS_END),
        "dev": (START_DATE, DEV_END),
        "is_test": (IS_TEST_START, IS_END),
    }


def slice_by_date(
    df: pd.DataFrame,
    start: str,
    end: str | None,
    time_col: str = "time",
) -> pd.DataFrame:
    mask = df[time_col] >= pd.to_datetime(start)
    if end is not None:
        mask &= df[time_col] <= pd.to_datetime(end)
    return df.loc[mask].copy()


def slice_by_split(
    df: pd.DataFrame,
    splits: Dict[str, Tuple[str, str | None]],
    key: str,
    time_col: str = "time",
) -> pd.DataFrame:
    if key not in splits:
        raise KeyError(f"Split '{key}' not found")
    start, end = splits[key]
    return slice_by_date(df, start, end, time_col=time_col)


def save_splits(splits: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)
