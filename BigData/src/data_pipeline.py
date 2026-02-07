"""Data generation/export entrypoint."""

from __future__ import annotations

from pathlib import Path
import logging

from .config import LABEL_COL, OPEN_COL, TIME_COL
from .splits import save_splits
from .pipeline_utils import load_master, logger
from .splits import make_splits, slice_by_split


def run_feature_label_export(
    data_dir: str | Path = "data",
    outputs_dir: str | Path = "outputs",
) -> None:
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting feature/label export")
    logger.info("Data dir: %s", data_dir)
    logger.info("Outputs dir: %s", outputs_dir)

    master = load_master(data_dir)

    splits = make_splits()
    save_splits(splits, outputs_dir / "splits.json")
    logger.info("Saved splits.json")

    df_is = slice_by_split(master, splits, "is", time_col=TIME_COL)
    df_oos = slice_by_split(master, splits, "oos", time_col=TIME_COL)

    df_is = df_is.dropna(subset=[OPEN_COL, LABEL_COL])
    df_oos = df_oos.dropna(subset=[OPEN_COL, LABEL_COL])
    logger.info("IS rows: %d | OOS rows: %d", len(df_is), len(df_oos))

    feature_cols = [c for c in master.columns if c not in {LABEL_COL}]

    df_is[feature_cols].to_csv(outputs_dir / "feature_is.csv", index=False)
    df_oos[feature_cols].to_csv(outputs_dir / "feature_oos.csv", index=False)

    df_is[[TIME_COL, LABEL_COL]].to_csv(outputs_dir / "label_is.csv", index=False)
    df_oos[[TIME_COL, LABEL_COL]].to_csv(outputs_dir / "label_oos.csv", index=False)
    logger.info("Exported feature/label CSVs")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_feature_label_export()


if __name__ == "__main__":
    main()
