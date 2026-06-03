from __future__ import annotations

import argparse
import os
from typing import Dict, Any

import numpy as np
import pandas as pd

from m5_pipeline.utils import get_logger, write_json
from m5_pipeline.validation import data_quality_report, segment_wmape


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)) + 1e-9)
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _add_segments(data_dir: str, df: pd.DataFrame) -> pd.DataFrame:
    """Add event-day and price-drop flags using calendar + sell_prices (best-effort)."""
    cal_path = os.path.join(data_dir, "calendar.csv")
    sales_path = os.path.join(data_dir, "sales_train_validation.csv")
    prices_path = os.path.join(data_dir, "sell_prices.csv")

    if not (
        os.path.exists(cal_path) and os.path.exists(sales_path) and os.path.exists(prices_path)
    ):
        return df

    calendar = pd.read_csv(cal_path)
    calendar["date"] = pd.to_datetime(calendar["date"]).dt.date
    calendar = calendar[["date", "wm_yr_wk", "event_name_1", "event_name_2"]].copy()

    meta = pd.read_csv(sales_path, usecols=["id", "item_id", "store_id"])
    prices = pd.read_csv(prices_path)

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date

    out = out.merge(calendar, on="date", how="left", validate="many_to_one")
    out = out.merge(meta, on="id", how="left", validate="many_to_one")
    out = out.merge(
        prices, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="many_to_one"
    )

    out = out.sort_values(["id", "date"])
    out["sell_price"] = pd.to_numeric(out["sell_price"], errors="coerce")
    out["price_pct_change_1"] = out.groupby("id")["sell_price"].pct_change()

    out["is_event_day"] = (out["event_name_1"].notna() | out["event_name_2"].notna()).astype(int)
    out["is_price_drop_day"] = (out["price_pct_change_1"] < 0).astype(int)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Daily drift checks for forecasting.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--actuals", default="actuals.csv", help="Must contain columns: id,date,y")
    p.add_argument("--preds", default="predictions.csv", help="Must contain columns: id,date,yhat")
    p.add_argument(
        "--warn-wmape", type=float, default=0.25, help="Warn if WMAPE above this threshold"
    )
    p.add_argument("--out-report", default="reports/drift_report.json")
    args = p.parse_args()

    logger = get_logger("drift_check")

    a_path = os.path.join(args.data_dir, args.actuals)
    p_path = os.path.join(args.data_dir, args.preds)
    if not os.path.exists(a_path) or not os.path.exists(p_path):
        logger.info("Drift check skipped: missing %s or %s", a_path, p_path)
        return

    actuals = pd.read_csv(a_path)
    preds = pd.read_csv(p_path)

    df = actuals.merge(preds, on=["id", "date"], how="inner")
    if df.empty:
        logger.info("Drift check: no overlapping rows between actuals and predictions.")
        return

    df = _add_segments(args.data_dir, df)

    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
    yhat = pd.to_numeric(df["yhat"], errors="coerce").to_numpy(dtype=float)

    wm = wmape(y, yhat)
    m = mae(y, yhat)

    seg: Dict[str, Any] = {}
    if "is_event_day" in df.columns:
        seg["wmape_event_days"] = segment_wmape(
            df, y_col="y", yhat_col="yhat", seg_mask=df["is_event_day"] == 1
        )
        seg["wmape_non_event_days"] = segment_wmape(
            df, y_col="y", yhat_col="yhat", seg_mask=df["is_event_day"] == 0
        )
    if "is_price_drop_day" in df.columns:
        seg["wmape_price_drop_days"] = segment_wmape(
            df, y_col="y", yhat_col="yhat", seg_mask=df["is_price_drop_day"] == 1
        )
        seg["wmape_non_price_drop_days"] = segment_wmape(
            df, y_col="y", yhat_col="yhat", seg_mask=df["is_price_drop_day"] == 0
        )

    logger.info("Daily drift: WMAPE=%.4f, MAE=%.4f on %d rows", wm, m, len(df))
    if wm > float(args.warn_wmape):
        logger.warning("WMAPE drift WARNING: %.4f > %.4f", wm, float(args.warn_wmape))

    # Data quality report: actuals and predictions distributions
    dq = {
        "actuals": data_quality_report(df, cols=["y"], name="actuals"),
        "predictions": data_quality_report(df, cols=["yhat"], name="predictions"),
    }

    rep = {"wmape": float(wm), "mae": float(m), "n": int(len(df)), **seg, "data_quality": dq}

    out_path = os.path.join(args.data_dir, args.out_report)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_json(out_path, rep)
    logger.info("Wrote drift report: %s", out_path)


if __name__ == "__main__":
    main()
