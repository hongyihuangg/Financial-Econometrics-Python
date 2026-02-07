"""Shared utilities for pipeline entrypoints."""

from __future__ import annotations

from pathlib import Path
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import BASE_SOURCE, LABEL_COL, OPEN_COL, SIGNAL_COL, TIME_COL
from .io_data import infer_source_name, list_csv_files, load_one_csv
from .preprocess import add_label_open_to_open, merge_sources

logger = logging.getLogger(__name__)


def load_master(data_dir: str | Path) -> pd.DataFrame:
    csv_paths = list_csv_files(data_dir)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    logger.info("Found %d CSV files in %s", len(csv_paths), data_dir)

    dfs: dict[str, pd.DataFrame] = {}
    for p in csv_paths:
        source = infer_source_name(p.name)
        logger.info("Loading %s as source %s", p.name, source)
        dfs[source] = load_one_csv(p, source, time_col=TIME_COL)

    master = merge_sources(dfs, base_source=BASE_SOURCE, time_col=TIME_COL)
    master = add_label_open_to_open(master, open_col=OPEN_COL, label_col=LABEL_COL)
    logger.info("Master dataset rows: %d", len(master))
    return master


def load_or_generate_predictions(
    df_seg: pd.DataFrame,
    outputs_dir: Path,
    segment: str,
    seed: int = 42,
) -> pd.Series:
    pred_path = outputs_dir / f"pred_{segment}.csv"
    if pred_path.exists():
        logger.info("Loading predictions from %s", pred_path)
        pred_df = pd.read_csv(pred_path)
        if TIME_COL not in pred_df.columns or SIGNAL_COL not in pred_df.columns:
            raise ValueError("Prediction file must contain 'time' and 'signal' columns")
        pred_df[TIME_COL] = pd.to_datetime(pred_df[TIME_COL]).dt.normalize()
        merged = df_seg[[TIME_COL]].merge(
            pred_df[[TIME_COL, SIGNAL_COL]], on=TIME_COL, how="left"
        )
        return pd.Series(merged[SIGNAL_COL].to_numpy(), index=df_seg.index)

    logger.info("Prediction file missing. Generating random signals for %s", segment)
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(len(df_seg))
    gen_df = pd.DataFrame({TIME_COL: df_seg[TIME_COL].values, SIGNAL_COL: signal})
    gen_df.to_csv(pred_path, index=False)
    logger.info("Saved generated predictions to %s", pred_path)
    return pd.Series(signal, index=df_seg.index)


def plot_backtest_report(
    segment: str,
    ic_series: pd.Series,
    decile_summary: pd.DataFrame,
    daily_ret: pd.Series,
    metrics: dict,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(ic_series.index, ic_series.values)
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set_title(f"{segment.upper()} IC (rolling)")

    if not decile_summary.empty:
        axes[0, 1].bar(decile_summary["decile"], decile_summary["mean_ret"])
    axes[0, 1].set_title(f"{segment.upper()} Decile Mean Returns")
    axes[0, 1].set_xlabel("Decile")
    axes[0, 1].set_ylabel("Mean Return")

    if not daily_ret.empty:
        cum = (1.0 + daily_ret).cumprod()
        axes[1, 0].plot(cum.index, cum.values)
    axes[1, 0].set_title(f"{segment.upper()} Cumulative Return")

    axes[1, 1].axis("off")
    metrics_text = "\n".join(
        [f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))]
    )
    axes[1, 1].text(0.02, 0.98, metrics_text, va="top", ha="left", fontsize=9)
    axes[1, 1].set_title(f"{segment.upper()} Metrics")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)  # type: ignore
    plt.close(fig)
