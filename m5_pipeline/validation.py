from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .utils import get_logger, write_json

# -----------------------------
# Public config / entrypoints
# -----------------------------


@dataclass
class ValidationConfig:
    """Validation policy for M5-style CSV inputs.

    Keep this strict in production: failing early beats silent bad outputs.
    """

    data_dir: str

    # Missingness thresholds
    max_missing_frac_per_col: float = 0.40
    max_missing_frac_sell_price: float = 0.25  # sell_prices gaps are common but still bounded

    # Range / sanity thresholds
    max_abs_sell_price: float = 10_000.0  # "obviously wrong" outlier guardrail
    max_abs_sales: float = 1_000_000.0

    # Write human-readable report artefacts
    write_reports: bool = True
    reports_subdir: str = "reports"

    # If True, raise on any issue. If False, log warnings and continue.
    strict: bool = True


REQUIRED_COLS: Dict[str, List[str]] = {
    "sales_train_validation.csv": ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    "calendar.csv": [
        "d",
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ],
    "sell_prices.csv": ["store_id", "item_id", "wm_yr_wk", "sell_price"],
    # produced by forecast; validated when present
    "submission.csv": ["id"],
    "sample_submission.csv": ["id"],
}

# Columns that are allowed to be sparse without failing missingness checks (common in M5)
MISSINGNESS_EXEMPT: Dict[str, List[str]] = {
    "calendar.csv": ["event_name_1", "event_type_1", "event_name_2", "event_type_2"],
}


# -----------------------------
# Core checks
# -----------------------------


def _fail_or_warn(logger, strict: bool, msg: str) -> None:
    if strict:
        raise ValueError(msg)
    logger.warning(msg)


def _required_columns(
    df: pd.DataFrame, required: Sequence[str], *, name: str, strict: bool
) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        _fail_or_warn(
            get_logger("validation"), strict, f"{name}: missing required columns: {missing}"
        )


def _key_uniqueness(
    df: pd.DataFrame,
    keys: Sequence[str],
    *,
    name: str,
    strict: bool,
    allow_null: bool = False,
) -> None:
    logger = get_logger("validation")
    for k in keys:
        if k not in df.columns:
            _fail_or_warn(logger, strict, f"{name}: key column missing: {k}")
            return

    tmp = df[list(keys)].copy()
    if not allow_null and tmp.isna().any().any():
        _fail_or_warn(logger, strict, f"{name}: nulls found in key columns: {list(keys)}")

    dup = tmp.duplicated(keep=False)
    if dup.any():
        ex = df.loc[dup, list(keys)].head(10).to_dict(orient="records")
        _fail_or_warn(logger, strict, f"{name}: duplicate keys on {list(keys)}. Example rows: {ex}")


def _value_range(
    s: pd.Series,
    *,
    name: str,
    ge: Optional[float] = None,
    gt: Optional[float] = None,
    le: Optional[float] = None,
    lt: Optional[float] = None,
    strict: bool,
) -> None:
    logger = get_logger("validation")
    x = pd.to_numeric(s, errors="coerce")
    bad = pd.Series(False, index=x.index)

    if ge is not None:
        bad |= x < float(ge)
    if gt is not None:
        bad |= x <= float(gt)
    if le is not None:
        bad |= x > float(le)
    if lt is not None:
        bad |= x >= float(lt)

    if bad.any():
        ex = x.loc[bad].head(10).tolist()
        _fail_or_warn(logger, strict, f"{name}: range violation. Example bad values: {ex}")


def _missingness(
    df: pd.DataFrame,
    *,
    name: str,
    per_col_thresh: float,
    strict: bool,
    exempt_cols: Sequence[str] | None = None,
) -> Dict[str, float]:
    """Check per-column missingness.

    Notes:
        M5 calendar event columns are naturally sparse. Use `exempt_cols` to avoid
        failing on expected sparsity.
    """
    miss = (df.isna().mean()).to_dict()
    if exempt_cols:
        for c in exempt_cols:
            miss.pop(c, None)

    bad = {k: float(v) for k, v in miss.items() if float(v) > float(per_col_thresh)}
    if bad:
        _fail_or_warn(
            get_logger("validation"),
            strict,
            f"{name}: missingness above threshold {per_col_thresh:.2f} on columns: {bad}",
        )
    return {k: float(v) for k, v in miss.items()}


# -----------------------------
# Data quality report (monitoring)
# -----------------------------


def _nonzero_skew(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    x = x[x > 0]
    if len(x) < 5:
        return float("nan")
    return float(pd.Series(x).skew())


def data_quality_report(
    df: pd.DataFrame,
    *,
    cols: Sequence[str],
    name: str,
) -> Dict[str, Any]:
    """Lightweight stats you can compare day-to-day to catch shifts."""
    out: Dict[str, Any] = {"name": name, "n": int(len(df)), "columns": {}}
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        nz = s.dropna()
        out["columns"][c] = {
            "missing_rate": float(s.isna().mean()),
            "zero_rate": float((nz == 0).mean()) if len(nz) else float("nan"),
            "nonzero_skew": float(_nonzero_skew(s)),
            "p99": float(nz.quantile(0.99)) if len(nz) else float("nan"),
            "p999": float(nz.quantile(0.999)) if len(nz) else float("nan"),
            "max": float(nz.max()) if len(nz) else float("nan"),
        }
    return out


# -----------------------------
# Segment metrics helpers
# -----------------------------


def segment_wmape(
    df: pd.DataFrame,
    *,
    y_col: str,
    yhat_col: str,
    seg_mask: pd.Series,
) -> float:
    y = df.loc[seg_mask, y_col].to_numpy(dtype=np.float64)
    yhat = df.loc[seg_mask, yhat_col].to_numpy(dtype=np.float64)
    denom = float(np.sum(np.abs(y)) + 1e-9)
    return float(np.sum(np.abs(y - yhat)) / denom) if len(y) else float("nan")


def add_segment_metrics(
    valid_df: pd.DataFrame,
    *,
    y_col: str,
    yhat_col: str,
    event_cols: Tuple[str, str] = ("event_name_1", "event_name_2"),
    price_change_col: str = "price_pct_change_1",
) -> Dict[str, Any]:
    """Compute WMAPE on important subsets (event/non-event, price-drop/non-price-drop)."""
    out: Dict[str, Any] = {}

    if all(c in valid_df.columns for c in event_cols):
        is_event = valid_df[event_cols[0]].notna() | valid_df[event_cols[1]].notna()
        out["wmape_event_days"] = segment_wmape(
            valid_df, y_col=y_col, yhat_col=yhat_col, seg_mask=is_event
        )
        out["wmape_non_event_days"] = segment_wmape(
            valid_df, y_col=y_col, yhat_col=yhat_col, seg_mask=~is_event
        )

    if price_change_col in valid_df.columns:
        is_price_drop = pd.to_numeric(valid_df[price_change_col], errors="coerce") < 0
        out["wmape_price_drop_days"] = segment_wmape(
            valid_df, y_col=y_col, yhat_col=yhat_col, seg_mask=is_price_drop
        )
        out["wmape_non_price_drop_days"] = segment_wmape(
            valid_df, y_col=y_col, yhat_col=yhat_col, seg_mask=~is_price_drop
        )

    return out


# -----------------------------
# Main validation entrypoint
# -----------------------------


def validate_m5_inputs(cfg: ValidationConfig) -> Dict[str, Any]:
    """Validate the raw M5 CSVs (and optionally submission.csv if present)."""
    logger = get_logger("validation")
    dd = cfg.data_dir

    def _p(name: str) -> str:
        return os.path.join(dd, name)

    report: Dict[str, Any] = {
        "data_dir": dd,
        "strict": bool(cfg.strict),
        "checks": {},
    }

    # ---- calendar.csv ----
    cal_path = _p("calendar.csv")
    calendar = pd.read_csv(cal_path)
    _required_columns(
        calendar, REQUIRED_COLS["calendar.csv"], name="calendar.csv", strict=cfg.strict
    )
    _key_uniqueness(calendar, ["d"], name="calendar.csv", strict=cfg.strict)
    _key_uniqueness(calendar, ["date"], name="calendar.csv", strict=cfg.strict)
    _missingness(
        calendar,
        name="calendar.csv",
        per_col_thresh=cfg.max_missing_frac_per_col,
        strict=cfg.strict,
        exempt_cols=MISSINGNESS_EXEMPT.get("calendar.csv"),
    )
    report["checks"]["calendar"] = {"rows": int(len(calendar))}

    # ---- sell_prices.csv ----
    prices_path = _p("sell_prices.csv")
    prices = pd.read_csv(prices_path)
    _required_columns(
        prices, REQUIRED_COLS["sell_prices.csv"], name="sell_prices.csv", strict=cfg.strict
    )
    _key_uniqueness(
        prices, ["store_id", "item_id", "wm_yr_wk"], name="sell_prices.csv", strict=cfg.strict
    )
    _value_range(
        prices["sell_price"],
        name="sell_prices.csv.sell_price",
        gt=0,
        le=cfg.max_abs_sell_price,
        strict=cfg.strict,
    )
    miss_prices = _missingness(
        prices,
        name="sell_prices.csv",
        per_col_thresh=cfg.max_missing_frac_sell_price,
        strict=cfg.strict,
        exempt_cols=None,
    )
    report["checks"]["sell_prices"] = {"rows": int(len(prices)), "missingness": miss_prices}

    # ---- sales_train_validation.csv ----
    sales_path = _p("sales_train_validation.csv")
    sales = pd.read_csv(sales_path)
    _required_columns(
        sales,
        REQUIRED_COLS["sales_train_validation.csv"],
        name="sales_train_validation.csv",
        strict=cfg.strict,
    )

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    if not day_cols:
        _fail_or_warn(logger, cfg.strict, "sales_train_validation.csv: no d_ columns found")
    else:
        # Sales must be >= 0, and not absurdly large
        # (vectorised check on a small sample first is cheaper; then full if needed)
        vals = sales[day_cols]
        _value_range(
            pd.Series(vals.to_numpy().ravel()),
            name="sales_train_validation.csv.sales",
            ge=0,
            le=cfg.max_abs_sales,
            strict=cfg.strict,
        )

    # ID uniqueness (id should be unique in wide format)
    _key_uniqueness(sales, ["id"], name="sales_train_validation.csv", strict=cfg.strict)

    miss_sales = _missingness(
        sales,
        name="sales_train_validation.csv",
        per_col_thresh=cfg.max_missing_frac_per_col,
        strict=cfg.strict,
        exempt_cols=None,
    )
    report["checks"]["sales"] = {
        "rows": int(len(sales)),
        "day_cols": int(len(day_cols)),
        "missingness": miss_sales,
    }

    # ---- submission.csv (optional) ----
    sub_path = _p("submission.csv")
    if os.path.exists(sub_path):
        sub = pd.read_csv(sub_path)
        _required_columns(
            sub, REQUIRED_COLS["submission.csv"], name="submission.csv", strict=cfg.strict
        )
        if "id" in sub.columns:
            _key_uniqueness(sub, ["id"], name="submission.csv", strict=cfg.strict)
        report["checks"]["submission"] = {"rows": int(len(sub))}

    # ---- lightweight data quality reports for monitoring ----
    if cfg.write_reports:
        rep_dir = os.path.join(dd, cfg.reports_subdir)
        os.makedirs(rep_dir, exist_ok=True)

        # These are "cheap but useful" columns to track.
        dq = {
            "sell_prices": data_quality_report(prices, cols=["sell_price"], name="sell_prices"),
        }
        # Sales quality: compute on a sample of series to keep runtime sane.
        if day_cols:
            sample = (
                sales[["id"] + day_cols]
                .head(200)
                .melt(id_vars=["id"], var_name="d", value_name="sales")
            )
            dq["sales"] = data_quality_report(sample, cols=["sales"], name="sales_sample_long")

        out_path = os.path.join(rep_dir, "data_quality_inputs.json")
        write_json(out_path, dq)
        logger.info("Wrote data quality report: %s", out_path)
        report["reports"] = {"data_quality_inputs": out_path}

    logger.info("Input validation complete.")
    return report
