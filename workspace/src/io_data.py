"""I/O utilities for loading and saving datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, cast

import pandas as pd


def list_csv_files(data_dir: str | Path) -> List[Path]:
    data_path = Path(data_dir)
    return sorted([p for p in data_path.rglob("*.csv") if p.is_file()])


def infer_source_name(filename: str) -> str:
    name = filename.upper()
    if "BTCUSD" in name:
        return "BTCUSD"
    if "ETHBTC" in name:
        return "ETHBTC"
    if "BTC.D" in name or "BTC.D," in name:
        return "BTCDOM"
    stem = Path(filename).stem
    cleaned = (
        stem.replace(" ", "")
        .replace(",", "_")
        .replace("-", "_")
        .replace(".", "_")
    )
    return cleaned.upper()


def load_one_csv(path: str | Path, source: str, time_col: str = "time") -> pd.DataFrame:
    df = cast(pd.DataFrame, pd.read_csv(path))
    if time_col not in df.columns:
        raise ValueError(f"Missing required column '{time_col}' in {path}")
    df[time_col] = pd.to_datetime(df[time_col], utc=False).dt.normalize()
    df = df.sort_values(time_col)
    df = df.drop_duplicates(subset=[time_col], keep="last")
    rename_cols = {
        c: f"{source}__{c}" for c in df.columns if c != time_col
    }
    df = df.rename(columns=rename_cols)
    return df


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
