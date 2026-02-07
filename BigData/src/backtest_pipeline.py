"""Backtest entrypoint using prediction files or generated signals."""

from __future__ import annotations

from pathlib import Path
import logging

from .config import LABEL_COL, MIN_HISTORY, OPEN_COL, QS, SIGNAL_COL, TIME_COL
from .eval_decile_ic import assign_deciles_time_series, decile_performance, rolling_ic
from .io_data import save_json
from .metrics import daily_level_metrics, merge_metrics, trade_level_metrics
from .pipeline_utils import load_master, load_or_generate_predictions, logger, plot_backtest_report
from .quantiles import compute_expanding_quantiles
from .splits import make_splits, save_splits, slice_by_split
from .backtest_exec import simulate_strategy


def run_backtest_with_predictions(
    data_dir: str | Path = "data",
    outputs_dir: str | Path = "outputs",
) -> None:
    outputs_dir = Path(outputs_dir)
    backtest_dir = outputs_dir / "backtest"
    backtest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting backtest with predictions")
    master = load_master(data_dir)

    splits = make_splits()
    save_splits(splits, outputs_dir / "splits.json")

    for segment in ["is", "oos"]:
        df_seg = slice_by_split(master, splits, segment, time_col=TIME_COL)
        df_seg = df_seg.dropna(subset=[OPEN_COL, LABEL_COL])

        df_seg[SIGNAL_COL] = load_or_generate_predictions(
            df_seg, outputs_dir, segment
        )

        q_df = compute_expanding_quantiles(
            df_seg[SIGNAL_COL], qs=QS, min_history=MIN_HISTORY
        )

        deciles = assign_deciles_time_series(
            df_seg[SIGNAL_COL], n_bins=10, min_history=MIN_HISTORY
        )
        decile_summary = decile_performance(
            df_seg.assign(decile=deciles), "decile", LABEL_COL
        )
        decile_summary.to_csv(
            backtest_dir / f"{segment}_decile_summary.csv", index=False
        )

        ic_series, ic_summary = rolling_ic(
            df_seg, SIGNAL_COL, LABEL_COL, window=MIN_HISTORY, method="spearman"
        )
        ic_series.to_csv(backtest_dir / f"{segment}_ic_series.csv")
        save_json(ic_summary, backtest_dir / f"{segment}_ic_summary.json")

        trades_df, daily_ret = simulate_strategy(
            df_seg,
            time_col=TIME_COL,
            open_col=OPEN_COL,
            signal_col=SIGNAL_COL,
            q_df=q_df,
            min_history=MIN_HISTORY,
            fee_bps=0.0,
        )

        if segment == "oos" and trades_df.empty:
            raise ValueError("OOS has no trades. Check signal thresholds or data coverage.")

        trades_df.to_csv(backtest_dir / f"{segment}_trades.csv", index=False)
        daily_ret.to_csv(backtest_dir / f"{segment}_daily_returns.csv")

        metrics = merge_metrics(
            trade_level_metrics(trades_df), daily_level_metrics(daily_ret)
        )
        save_json(metrics, backtest_dir / f"{segment}_metrics.json")

        report_path = backtest_dir / f"{segment}_report.png"
        plot_backtest_report(
            segment, ic_series, decile_summary, daily_ret, metrics, report_path
        )
        logger.info("Saved report image: %s", report_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_backtest_with_predictions()


if __name__ == "__main__":
    main()
