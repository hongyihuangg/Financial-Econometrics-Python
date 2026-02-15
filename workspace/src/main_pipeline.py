"""Deprecated entrypoint. Use data_pipeline.py or backtest_pipeline.py."""

from __future__ import annotations
import logging
from .data_pipeline import run_feature_label_export
from .backtest_pipeline import run_backtest_with_predictions


__all__ = ["run_feature_label_export", "run_backtest_with_predictions"]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_backtest_with_predictions()
