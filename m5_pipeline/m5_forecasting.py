from __future__ import annotations

import os
import sys
import platform
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal, Sequence

import numpy as np
import joblib
import pandas as pd

from .utils import get_logger, require_files, write_json, select_representative_series_subset
from .validation import (
    ValidationConfig,
    validate_m5_inputs,
    add_segment_metrics,
    data_quality_report,
)

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


def _stable_hash(obj: Any) -> str:
    import json
    import hashlib
    from .utils import _json_default  # type: ignore

    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=_json_default)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _now_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class BacktestSplit:
    train_end_d: str
    valid_start_d: str
    valid_end_d: str
    inner_train_end_d: Optional[str] = None
    inner_valid_start_d: Optional[str] = None
    inner_valid_end_d: Optional[str] = None


@dataclass(frozen=True)
class PromotionPolicy:
    promotion_min_backtests: int = 2
    promotion_min_aggregate_improvement_pct: float = 2.0
    promotion_min_win_rate: float = 0.67
    promotion_max_single_split_regression_pct: float = 5.0
    promotion_max_critical_segment_regression_pct: float = 10.0
    promotion_min_segment_actual_sum: float = 25.0
    allow_single_split_promotion: bool = False


BASELINE_COLS = (
    "pred_baseline_mean_28",
    "pred_baseline_seasonal_7",
    "pred_baseline_seasonal_364",
)

BASELINE_NAME_TO_COL = {
    "mean_28": "pred_baseline_mean_28",
    "seasonal_7": "pred_baseline_seasonal_7",
    "seasonal_364": "pred_baseline_seasonal_364",
}


@dataclass
class ForecastConfig:
    data_dir: str
    max_series: Optional[int] = 0  # 0 -> all series
    start_d: str = "d_1500"  # speed control
    horizon: int = 28
    last_train_d: str = "d_1913"
    out_submission: str = "submission.csv"

    # --- Production / reproducibility ---
    split_strategy: Literal["last_window", "rolling_origin"] = "rolling_origin"
    n_backtests: int = 3
    backtest_stride: int = 28  # days between cutoffs for rolling origin
    promotion_min_backtests: int = 2
    promotion_min_aggregate_improvement_pct: float = 2.0
    promotion_min_win_rate: float = 0.67
    promotion_max_single_split_regression_pct: float = 5.0
    promotion_max_critical_segment_regression_pct: float = 10.0
    promotion_min_segment_actual_sum: float = 25.0
    allow_single_split_promotion: bool = False
    prediction_interval_coverage: float = 0.80

    save_artifacts: bool = True
    artifacts_dir: str = "artifacts/forecast"
    run_name: Optional[str] = None

    # Validation (safe default: validate)
    validate_inputs: bool = True
    validation_strict: bool = True

    # Model
    objective: str = "tweedie"  # "tweedie" tends to work well for sparse retail
    tweedie_variance_power: float = 1.1  # 1.1–1.3 common
    n_estimators: int = 4000
    learning_rate: float = 0.03
    num_leaves: int = 128
    min_data_in_leaf: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    random_state: int = 42
    two_stage: bool = True  # P(y>0) × E[y|y>0] for intermittent demand (recommended for M5)

    early_stopping_rounds: int = 200

    # Features (past-only)
    lags: Tuple[int, ...] = (1, 7, 28, 56, 364)
    rolls: Tuple[int, ...] = (7, 28, 56, 364)

    # Intermittency features
    nonzero_rolls: Tuple[int, ...] = (7, 28, 56)


def _d_to_int(d: str) -> int:
    return int(d.split("_")[1])


def _int_to_d(day: int) -> str:
    return f"d_{int(day)}"


def make_backtest_splits(
    *,
    split_strategy: Literal["last_window", "rolling_origin"],
    start_day: int,
    last_train_day: int,
    horizon: int,
    n_backtests: int,
    stride: int,
    max_required_history: int,
) -> List[BacktestSplit]:
    """Create deployment-style outer backtest windows.

    The returned list is most-recent first. The outer validation block is always
    exactly ``horizon`` days and starts after the outer training end.
    """
    if split_strategy not in {"last_window", "rolling_origin"}:
        raise ValueError("split_strategy must be 'last_window' or 'rolling_origin'")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if n_backtests <= 0:
        raise ValueError("n_backtests must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if last_train_day < start_day:
        raise ValueError("last_train_day must be >= start_day")

    min_train_end = int(start_day) + int(max_required_history)

    def build(valid_end: int) -> BacktestSplit:
        valid_start = int(valid_end) - int(horizon) + 1
        train_end = valid_start - 1
        if valid_start < start_day or valid_end > last_train_day:
            raise ValueError("validation window extends outside requested day range")
        if train_end < min_train_end:
            raise ValueError(
                "not enough history before validation for requested horizon and lag requirements"
            )

        inner_valid_end = train_end
        inner_valid_start = inner_valid_end - int(horizon) + 1
        inner_train_end = inner_valid_start - 1
        if inner_train_end < min_train_end:
            return BacktestSplit(
                train_end_d=_int_to_d(train_end),
                valid_start_d=_int_to_d(valid_start),
                valid_end_d=_int_to_d(valid_end),
            )
        return BacktestSplit(
            train_end_d=_int_to_d(train_end),
            valid_start_d=_int_to_d(valid_start),
            valid_end_d=_int_to_d(valid_end),
            inner_train_end_d=_int_to_d(inner_train_end),
            inner_valid_start_d=_int_to_d(inner_valid_start),
            inner_valid_end_d=_int_to_d(inner_valid_end),
        )

    if split_strategy == "last_window":
        return [build(last_train_day)]

    splits: List[BacktestSplit] = []
    for i in range(int(n_backtests)):
        valid_end = int(last_train_day) - i * int(stride)
        try:
            splits.append(build(valid_end))
        except ValueError:
            if i == 0:
                raise
            break
    return splits


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def _wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted MAPE: sum(|e|) / sum(|y|).

    Stable and business-friendly for intermittent retail demand.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = float(np.sum(np.abs(y_true)) + 1e-9)
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-9
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def _smape_nonzero(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sMAPE over non-zero actuals only (y_true > 0) to avoid inflation from zeros."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    m = y_true > 0
    if not np.any(m):
        return float("nan")
    yt = y_true[m]
    yp = y_pred[m]
    denom = np.abs(yt) + np.abs(yp) + 1e-9
    return float(np.mean(2.0 * np.abs(yp - yt) / denom))


def _binary_logloss(y_true_bin: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y_true_bin, dtype=np.int32)
    p = np.asarray(p, dtype=np.float64)
    if y.size == 0:
        return float("nan")
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1.0 - p))))


def _roc_auc_tie_correct(y_true_bin: np.ndarray, p: np.ndarray) -> float:
    """Tie-correct AUC using the rank-based (Mann–Whitney U) formulation."""
    y = np.asarray(y_true_bin, dtype=np.int32)
    p = np.asarray(p, dtype=np.float64)
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")

    order = np.argsort(p)
    p_sorted = p[order]
    y_sorted = y[order]

    n = len(p_sorted)
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    r = 1.0
    while i < n:
        j = i
        while j + 1 < n and p_sorted[j + 1] == p_sorted[i]:
            j += 1
        avg = (r + (r + (j - i))) / 2.0
        ranks[i : j + 1] = avg
        r += (j - i) + 1.0
        i = j + 1

    n_pos = float(np.sum(y_sorted == 1))
    n_neg = float(np.sum(y_sorted == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    sum_ranks_pos = float(np.sum(ranks[y_sorted == 1]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _make_snap(df: pd.DataFrame) -> pd.Series:
    snap = np.zeros(len(df), dtype=np.int8)
    st = df["state_id"].values
    if "snap_CA" in df.columns:
        snap[st == "CA"] = df.loc[st == "CA", "snap_CA"].astype(np.int8).values
    if "snap_TX" in df.columns:
        snap[st == "TX"] = df.loc[st == "TX", "snap_TX"].astype(np.int8).values
    if "snap_WI" in df.columns:
        snap[st == "WI"] = df.loc[st == "WI", "snap_WI"].astype(np.int8).values
    return pd.Series(snap, index=df.index, name="snap")


def _fill_sell_price_train(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill sell_price without time leakage:
      - ffill within each id (uses only past)
      - leave leading NaNs missing so models can handle unknown historical prices
    Also creates a price_was_missing flag BEFORE filling.
    """
    df = df.sort_values(["id", "date"]).copy()
    df["price_was_missing"] = df["sell_price"].isna().astype(np.int8)

    observed = df["sell_price"].notna()
    ffilled = df.groupby("id")["sell_price"].ffill()
    df["sell_price"] = ffilled.astype(np.float32)
    df["price_imputation_source"] = np.where(
        observed,
        "observed",
        np.where(df["sell_price"].notna(), "historical_forward_fill", "missing_leading"),
    )

    return df


def _fill_sell_price_future(fut: pd.DataFrame, train_filled: pd.DataFrame) -> pd.DataFrame:
    """
    Future prices:
      - keep a price_was_missing flag
      - ffill within future (week-level prices may have gaps)
      - if still missing, carry forward last known TRAIN price per id
      - final fallback: global median of train prices
    """
    fut = fut.sort_values(["id", "date"]).copy()
    fut["price_was_missing"] = fut["sell_price"].isna().astype(np.int8)
    scheduled = fut["sell_price"].notna()

    fut_ff = fut.groupby("id")["sell_price"].ffill()

    last_train_price = train_filled.groupby("id")["sell_price"].last()
    fut["sell_price"] = fut_ff

    # carry last train price
    from_train_history = fut["sell_price"].isna() & fut["id"].map(last_train_price).notna()
    fut["sell_price"] = fut["sell_price"].fillna(fut["id"].map(last_train_price))

    # final fallback
    global_med = float(train_filled["sell_price"].median())
    from_train_stat = fut["sell_price"].isna()
    fut["sell_price"] = fut["sell_price"].fillna(global_med).astype(np.float32)
    fut["price_imputation_source"] = np.where(
        scheduled,
        "scheduled_price",
        np.where(
            from_train_history,
            "last_training_price",
            np.where(from_train_stat, "training_median_fallback", "future_forward_fill"),
        ),
    )

    return fut


def _add_lag_roll_and_intermittency(
    df: pd.DataFrame,
    lags: Tuple[int, ...],
    rolls: Tuple[int, ...],
    nonzero_rolls: Tuple[int, ...],
) -> pd.DataFrame:
    df = df.sort_values(["id", "date"]).copy()
    g = df.groupby("id")["sales"]

    # Lags (safe)
    for L in lags:
        df[f"lag_{L}"] = g.shift(L).astype(np.float32)

    # Past-only history (safe)
    hist = g.shift(1)

    # Past-only rolling stats (group-wise rolling to avoid cross-id leakage)
    for W in rolls:
        roll_mean = (
            hist.groupby(df["id"]).rolling(W, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        roll_std = (
            hist.groupby(df["id"])
            .rolling(W, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )
        df[f"roll_mean_{W}"] = roll_mean.astype(np.float32)
        df[f"roll_std_{W}"] = roll_std.astype(np.float32)

        roll_sum = (
            hist.groupby(df["id"]).rolling(W, min_periods=1).sum().reset_index(level=0, drop=True)
        )
        df[f"roll_sum_{W}"] = roll_sum.astype(np.float32)

    # Intermittency: nonzero rates (group-wise rolling)
    nz = (hist > 0).astype(np.float32)
    for W in nonzero_rolls:
        nz_rate = (
            nz.groupby(df["id"]).rolling(W, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df[f"nonzero_rate_{W}"] = nz_rate.astype(np.float32)

    # Baselines (past-only)
    baseline28 = (
        hist.groupby(df["id"]).rolling(28, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["baseline_mean_28"] = baseline28.astype(np.float32)
    df["baseline_seasonal_7"] = g.shift(7).astype(np.float32)
    df["baseline_seasonal_364"] = g.shift(364).astype(np.float32)

    # Days since last positive sale (past-only)
    last_sale_date = df["date"].where(hist > 0).groupby(df["id"]).ffill()
    days_since = (df["date"] - last_sale_date).dt.days
    df["days_since_last_sale"] = days_since.fillna(9999).astype(np.float32)

    return df


def _add_price_feats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["id", "date"]).copy()
    gp = df.groupby("id")["sell_price"]
    df["price_lag_1"] = gp.shift(1).astype(np.float32)

    # Past-only rolling mean (group-wise rolling to avoid cross-id leakage)
    ph = gp.shift(1)
    price_roll = (
        ph.groupby(df["id"]).rolling(28, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["price_roll_mean_28"] = price_roll.astype(np.float32)

    rel_ok = df["sell_price"].notna() & df["price_roll_mean_28"].notna()
    df["price_rel_28"] = np.where(
        rel_ok,
        df["sell_price"] / (df["price_roll_mean_28"] + 1e-6) - 1.0,
        np.nan,
    ).astype(np.float32)

    # Price momentum
    pct_ok = df["sell_price"].notna() & df["price_lag_1"].notna()
    df["price_pct_change_1"] = np.where(
        pct_ok,
        df["sell_price"] / (df["price_lag_1"] + 1e-6) - 1.0,
        np.nan,
    ).astype(np.float32)

    # Price change features
    price_changed = (
        df["sell_price"].notna()
        & df["price_lag_1"].notna()
        & df["sell_price"].ne(df["price_lag_1"])
    )
    df["price_changed_today"] = price_changed.astype(np.float32)

    # Count of price changes in the last 28 days (includes today)
    change_cnt_28 = (
        df["price_changed_today"]
        .groupby(df["id"])
        .rolling(28, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    df["price_change_count_28"] = change_cnt_28.astype(np.float32)

    # Weeks since last price change (0 if changed today)
    last_change_date = df["date"].where(price_changed).groupby(df["id"]).ffill()
    days_since_change = (df["date"] - last_change_date).dt.days
    df["weeks_since_price_change"] = (days_since_change.fillna(9999) / 7.0).astype(np.float32)

    return df


def _prepare_categoricals(
    train: pd.DataFrame, future: pd.DataFrame, cat_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    for c in cat_cols:
        train[c] = train[c].astype("object").fillna("none").astype(str)
        future[c] = future[c].astype("object").fillna("none").astype(str)
        cats = pd.Index(pd.concat([train[c], future[c]], ignore_index=True).unique())
        if "none" not in cats:
            cats = cats.insert(0, "none")
        train[c] = pd.Categorical(train[c], categories=cats)
        future[c] = pd.Categorical(future[c], categories=cats)
    return train, future


def _baseline_prediction_from_history(history: Sequence[float], family: str) -> float:
    values = [float(v) for v in history if np.isfinite(float(v))]
    if not values:
        return 0.0
    mean28 = float(np.mean(values[-28:]))
    if family == "mean_28":
        return max(0.0, mean28)
    if family == "seasonal_7":
        pred = values[-7] if len(values) >= 7 else mean28
        return max(0.0, float(pred))
    if family == "seasonal_364":
        if len(values) >= 364:
            pred = values[-364]
        elif len(values) >= 7:
            pred = values[-7]
        else:
            pred = mean28
        return max(0.0, float(pred))
    raise ValueError(f"Unknown baseline family: {family}")


def _history_by_id(frame: pd.DataFrame, value_col: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for key, group in frame.sort_values(["id", "date"]).groupby("id", sort=False):
        out[str(key)] = pd.to_numeric(group[value_col], errors="coerce").astype(float).tolist()
    return out


def _recursive_feature_rows(
    day_rows: pd.DataFrame,
    *,
    hist_sales: Dict[str, List[float]],
    hist_price: Dict[str, List[float]],
    cfg: ForecastConfig,
    origin_date: pd.Timestamp,
    cat_cols: List[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in day_rows.itertuples(index=False):
        _id = str(r.id)
        sh = hist_sales.setdefault(_id, [])
        ph = hist_price.setdefault(_id, [])

        def lag(periods: int) -> float:
            return float(sh[-periods]) if len(sh) >= periods else float("nan")

        def roll_values(window: int) -> List[float]:
            return sh[-window:] if len(sh) >= window else sh

        def roll_mean(window: int) -> float:
            x = roll_values(window)
            return float(np.mean(x)) if x else float("nan")

        def roll_std(window: int) -> float:
            x = roll_values(window)
            return float(np.std(x, ddof=0)) if len(x) > 1 else 0.0

        def roll_sum(window: int) -> float:
            x = roll_values(window)
            return float(np.sum(x)) if x else float("nan")

        def nonzero_rate(window: int) -> float:
            x = roll_values(window)
            return float(np.mean([1.0 if v > 0 else 0.0 for v in x])) if x else 0.0

        def days_since_last_sale() -> float:
            for i in range(1, len(sh) + 1):
                if sh[-i] > 0:
                    return float(i)
            return 9999.0

        raw_price = getattr(r, "sell_price")
        price = float(raw_price) if pd.notna(raw_price) else (float(ph[-1]) if ph else float("nan"))
        price_lag_1 = float(ph[-1]) if ph else float("nan")
        observed_prices = [float(v) for v in ph[-28:] if np.isfinite(float(v))]
        price_roll_mean_28 = float(np.mean(observed_prices)) if observed_prices else float("nan")
        price_ok = pd.notna(price) and pd.notna(price_lag_1)
        rel_ok = pd.notna(price) and pd.notna(price_roll_mean_28)
        price_rel_28 = float(price / (price_roll_mean_28 + 1e-6) - 1.0) if rel_ok else float("nan")
        price_pct_change_1 = float(price / (price_lag_1 + 1e-6) - 1.0) if price_ok else float("nan")
        price_changed_today = bool(price_ok and not np.isclose(price, price_lag_1))

        all_prices = [v for v in ph[-27:] if np.isfinite(float(v))]
        if np.isfinite(price):
            all_prices.append(price)
        price_change_count_28 = 0.0
        if len(all_prices) >= 2:
            price_change_count_28 = float(
                sum(
                    1
                    for j in range(1, len(all_prices))
                    if not np.isclose(all_prices[j], all_prices[j - 1])
                )
            )

        days_since_change = 9999.0
        price_path = [v for v in ph if np.isfinite(float(v))]
        if np.isfinite(price):
            price_path.append(price)
        for j in range(len(price_path) - 1, 0, -1):
            if not np.isclose(price_path[j], price_path[j - 1]):
                days_since_change = float((len(price_path) - 1) - j)
                break

        day = pd.Timestamp(getattr(r, "date"))
        row = {
            "id": _id,
            "sell_price": price,
            "price_lag_1": price_lag_1,
            "price_roll_mean_28": price_roll_mean_28,
            "price_rel_28": price_rel_28,
            "price_pct_change_1": price_pct_change_1,
            "price_changed_today": float(price_changed_today),
            "price_change_count_28": price_change_count_28,
            "weeks_since_price_change": float(days_since_change / 7.0),
            "price_was_missing": int(getattr(r, "price_was_missing", int(pd.isna(raw_price)))),
            "t": int((day - origin_date).days),
            "days_since_last_sale": days_since_last_sale(),
            "snap": int(getattr(r, "snap")),
            "wday": int(getattr(r, "wday")),
            "month": int(getattr(r, "month")),
            "year": int(getattr(r, "year")),
        }
        for periods in cfg.lags:
            row[f"lag_{periods}"] = lag(periods)
        for window in cfg.rolls:
            row[f"roll_mean_{window}"] = roll_mean(window)
            row[f"roll_std_{window}"] = roll_std(window)
            row[f"roll_sum_{window}"] = roll_sum(window)
        for window in cfg.nonzero_rolls:
            row[f"nonzero_rate_{window}"] = nonzero_rate(window)
        for col in cat_cols:
            row[col] = getattr(r, col)
        rows.append(row)
    return rows


def _predict_model(
    model_obj: Any, X_day: pd.DataFrame, feature_cols: List[str], two_stage: bool
) -> np.ndarray:
    if model_obj is None:
        return np.zeros(len(X_day), dtype=np.float64)
    if two_stage:
        clf, reg = model_obj
        p = clf.predict_proba(X_day[feature_cols])[:, 1].astype(np.float64)
        q = np.clip(reg.predict(X_day[feature_cols]).astype(np.float64), 0, None)
        return np.clip(p * q, 0, None)
    return np.clip(model_obj.predict(X_day[feature_cols]).astype(np.float64), 0, None)


def recursive_evaluate_split(
    frame: pd.DataFrame,
    *,
    split: BacktestSplit,
    model_obj: Any,
    cfg: ForecastConfig,
    feature_cols: List[str],
    cat_cols: List[str],
    category_reference: pd.DataFrame,
    origin_date: pd.Timestamp,
) -> pd.DataFrame:
    """Generate row-level recursive predictions for one untouched outer split."""
    train_hist = frame.loc[frame["d"].map(_d_to_int) <= _d_to_int(split.train_end_d)].copy()
    valid = frame.loc[
        (frame["d"].map(_d_to_int) >= _d_to_int(split.valid_start_d))
        & (frame["d"].map(_d_to_int) <= _d_to_int(split.valid_end_d))
    ].copy()
    if train_hist.empty or valid.empty:
        return pd.DataFrame()

    hist_model = _history_by_id(train_hist, "sales")
    hist_price = _history_by_id(train_hist, "sell_price")
    hist_baselines = {
        name: {key: list(vals) for key, vals in hist_model.items()} for name in BASELINE_NAME_TO_COL
    }

    rows_out: List[Dict[str, Any]] = []
    dates = list(pd.Series(valid["date"].unique()).sort_values())
    date_to_step = {pd.Timestamp(day): i + 1 for i, day in enumerate(dates)}

    for day in dates:
        day_ts = pd.Timestamp(day)
        day_rows = valid.loc[valid["date"] == day].sort_values("id").copy()
        feature_rows = _recursive_feature_rows(
            day_rows,
            hist_sales=hist_model,
            hist_price=hist_price,
            cfg=cfg,
            origin_date=origin_date,
            cat_cols=cat_cols,
        )
        X_day = pd.DataFrame(feature_rows)
        for col in cat_cols:
            X_day[col] = pd.Categorical(
                X_day[col].astype(str), categories=category_reference[col].cat.categories
            )

        pred_lgbm = _predict_model(model_obj, X_day, feature_cols, cfg.two_stage)

        baseline_preds: Dict[str, List[float]] = {col: [] for col in BASELINE_COLS}
        for _id in X_day["id"].astype(str).tolist():
            for name, col in BASELINE_NAME_TO_COL.items():
                pred = _baseline_prediction_from_history(
                    hist_baselines[name].setdefault(_id, []), name
                )
                baseline_preds[col].append(pred)

        actuals = pd.to_numeric(day_rows["sales"], errors="coerce").astype(float).to_numpy()
        for i, _id in enumerate(X_day["id"].astype(str).tolist()):
            rows_out.append(
                {
                    "id": _id,
                    "date": day_ts,
                    "d": str(day_rows.iloc[i]["d"]),
                    "horizon_step": int(date_to_step[day_ts]),
                    "actual": float(actuals[i]),
                    "pred_lgbm": float(pred_lgbm[i]),
                    "pred_baseline_mean_28": float(baseline_preds["pred_baseline_mean_28"][i]),
                    "pred_baseline_seasonal_7": float(
                        baseline_preds["pred_baseline_seasonal_7"][i]
                    ),
                    "pred_baseline_seasonal_364": float(
                        baseline_preds["pred_baseline_seasonal_364"][i]
                    ),
                    "event_flag": bool(
                        pd.notna(day_rows.iloc[i].get("event_name_1"))
                        or pd.notna(day_rows.iloc[i].get("event_name_2"))
                    ),
                    "price_drop_flag": bool(
                        pd.notna(X_day.iloc[i].get("price_pct_change_1"))
                        and float(X_day.iloc[i].get("price_pct_change_1")) < 0
                    ),
                }
            )

        for i, _id in enumerate(X_day["id"].astype(str).tolist()):
            hist_model.setdefault(_id, []).append(float(pred_lgbm[i]))
            price = X_day.iloc[i]["sell_price"]
            hist_price.setdefault(_id, []).append(float(price) if pd.notna(price) else float("nan"))
            for name, col in BASELINE_NAME_TO_COL.items():
                hist_baselines[name].setdefault(_id, []).append(float(baseline_preds[col][i]))

    return pd.DataFrame(rows_out)


def _wmape_from_sums(actual_sum: float, error_sum: float) -> float:
    return float(error_sum / (actual_sum + 1e-9))


def split_diagnostics_from_evaluation(
    eval_df: pd.DataFrame, split: BacktestSplit
) -> Dict[str, Any]:
    actual = eval_df["actual"].abs().astype(float)
    out: Dict[str, Any] = {
        "split": {
            "train_end_d": split.train_end_d,
            "valid_start_d": split.valid_start_d,
            "valid_end_d": split.valid_end_d,
            "inner_train_end_d": split.inner_train_end_d,
            "inner_valid_start_d": split.inner_valid_start_d,
            "inner_valid_end_d": split.inner_valid_end_d,
        },
        "row_count": int(len(eval_df)),
        "actual_abs_sum": float(actual.sum()),
    }
    prediction_cols = ["pred_lgbm", *BASELINE_COLS]
    for col in prediction_cols:
        err = (eval_df["actual"].astype(float) - eval_df[col].astype(float)).abs()
        key = col.replace("pred_", "")
        out[f"{key}_abs_error_sum"] = float(err.sum())
        out[f"wmape_{key}"] = _wmape_from_sums(out["actual_abs_sum"], float(err.sum()))

    segment_defs = {
        "event": eval_df.get("event_flag", pd.Series(False, index=eval_df.index)).astype(bool),
        "non_event": ~eval_df.get("event_flag", pd.Series(False, index=eval_df.index)).astype(bool),
        "price_drop": eval_df.get("price_drop_flag", pd.Series(False, index=eval_df.index)).astype(
            bool
        ),
        "non_price_drop": ~eval_df.get(
            "price_drop_flag", pd.Series(False, index=eval_df.index)
        ).astype(bool),
    }
    segments: Dict[str, Any] = {}
    for name, mask in segment_defs.items():
        seg_actual = eval_df.loc[mask, "actual"].abs().astype(float)
        seg: Dict[str, Any] = {
            "actual_abs_sum": float(seg_actual.sum()),
            "row_count": int(mask.sum()),
        }
        for col in prediction_cols:
            err = (
                eval_df.loc[mask, "actual"].astype(float) - eval_df.loc[mask, col].astype(float)
            ).abs()
            key = col.replace("pred_", "")
            seg[f"{key}_abs_error_sum"] = float(err.sum())
            seg[f"wmape_{key}"] = _wmape_from_sums(float(seg_actual.sum()), float(err.sum()))
        segments[name] = seg
    out["segments"] = segments
    return out


def evaluate_promotion_policy(
    split_diagnostics: List[Dict[str, Any]], policy: PromotionPolicy
) -> Dict[str, Any]:
    if not split_diagnostics:
        raise ValueError("split_diagnostics must not be empty")

    baseline_error_sums = {
        name: sum(float(s[f"baseline_{name}_abs_error_sum"]) for s in split_diagnostics)
        for name in BASELINE_NAME_TO_COL
    }
    actual_sum = sum(float(s["actual_abs_sum"]) for s in split_diagnostics)
    selected_baseline = min(baseline_error_sums, key=baseline_error_sums.get)
    selected_baseline_wmape = _wmape_from_sums(actual_sum, baseline_error_sums[selected_baseline])
    ml_error_sum = sum(float(s["lgbm_abs_error_sum"]) for s in split_diagnostics)
    ml_wmape = _wmape_from_sums(actual_sum, ml_error_sum)
    improvement_pct = (
        (selected_baseline_wmape - ml_wmape) / (selected_baseline_wmape + 1e-9) * 100.0
    )

    wins = 0
    regressions: List[float] = []
    split_rows = []
    for split in split_diagnostics:
        base_wmape = float(split[f"wmape_baseline_{selected_baseline}"])
        ml_split_wmape = float(split["wmape_lgbm"])
        if ml_split_wmape <= base_wmape:
            wins += 1
        regression = max(0.0, (ml_split_wmape - base_wmape) / (base_wmape + 1e-9) * 100.0)
        regressions.append(regression)
        split_rows.append(
            {
                "split": split["split"],
                "actual_abs_sum": split["actual_abs_sum"],
                "row_count": split["row_count"],
                "wmape_lgbm": ml_split_wmape,
                f"wmape_selected_baseline_{selected_baseline}": base_wmape,
                "regression_pct": regression,
            }
        )

    split_win_rate = wins / max(len(split_diagnostics), 1)
    worst_split_regression_pct = max(regressions) if regressions else 0.0
    critical_regressions: List[Dict[str, Any]] = []
    for seg_name in ("event", "non_event", "price_drop", "non_price_drop"):
        seg_actual = sum(
            float(s["segments"][seg_name]["actual_abs_sum"]) for s in split_diagnostics
        )
        if seg_actual < float(policy.promotion_min_segment_actual_sum):
            continue
        seg_ml_error = sum(
            float(s["segments"][seg_name]["lgbm_abs_error_sum"]) for s in split_diagnostics
        )
        seg_base_error = sum(
            float(s["segments"][seg_name][f"baseline_{selected_baseline}_abs_error_sum"])
            for s in split_diagnostics
        )
        seg_ml_wmape = _wmape_from_sums(seg_actual, seg_ml_error)
        seg_base_wmape = _wmape_from_sums(seg_actual, seg_base_error)
        seg_regression = max(0.0, (seg_ml_wmape - seg_base_wmape) / (seg_base_wmape + 1e-9) * 100.0)
        if seg_regression > float(policy.promotion_max_critical_segment_regression_pct):
            critical_regressions.append(
                {
                    "segment": seg_name,
                    "actual_abs_sum": seg_actual,
                    "regression_pct": seg_regression,
                }
            )

    reasons: List[str] = []
    if len(split_diagnostics) < int(policy.promotion_min_backtests):
        if not (policy.allow_single_split_promotion and len(split_diagnostics) == 1):
            reasons.append("insufficient_backtests")
    if improvement_pct < float(policy.promotion_min_aggregate_improvement_pct):
        reasons.append("insufficient_aggregate_improvement")
    if split_win_rate < float(policy.promotion_min_win_rate):
        reasons.append("low_split_win_rate")
    if worst_split_regression_pct > float(policy.promotion_max_single_split_regression_pct):
        reasons.append("unstable_worst_split")
    if critical_regressions:
        reasons.append("critical_segment_regression")

    winner = "lgbm" if not reasons else "baseline"
    return {
        "winner": winner,
        "selected_baseline": selected_baseline,
        "promotion_status": "accepted" if winner == "lgbm" else "rejected",
        "promotion_reasons": reasons,
        "policy": policy.__dict__.copy(),
        "aggregate_metrics": {
            "actual_abs_sum": actual_sum,
            "aggregate_lgbm_wmape": ml_wmape,
            "aggregate_selected_baseline_wmape": selected_baseline_wmape,
            "aggregate_improvement_pct": improvement_pct,
            "split_win_rate": split_win_rate,
            "worst_split_regression_pct": worst_split_regression_pct,
            "critical_segment_regressions": critical_regressions,
            "deployable_baseline_error_sums": baseline_error_sums,
        },
        "split_diagnostics": split_rows,
    }


def finite_sample_conformal_quantile(abs_residuals: Sequence[float], alpha: float) -> float:
    vals = np.sort(
        np.asarray([x for x in abs_residuals if np.isfinite(float(x))], dtype=np.float64)
    )
    if vals.size == 0:
        raise ValueError("Cannot compute conformal quantile without residuals")
    rank = int(np.ceil((vals.size + 1) * (1.0 - float(alpha))))
    index = min(max(rank, 1), vals.size) - 1
    return float(vals[index])


def build_conformal_interval_summary(
    evaluation_rows: pd.DataFrame,
    *,
    prediction_col: str,
    target_coverage: float = 0.80,
) -> Dict[str, Any]:
    if evaluation_rows.empty:
        return {"status": "no_backtest_rows", "target_coverage": float(target_coverage)}
    alpha = 1.0 - float(target_coverage)
    df = evaluation_rows.copy()
    df["abs_residual"] = (df["actual"].astype(float) - df[prediction_col].astype(float)).abs()
    split_count = int(df["split_index"].nunique()) if "split_index" in df.columns else 1
    if split_count > 1:
        cal = df[df["split_index"] != 0].copy()
        heldout = df[df["split_index"] == 0].copy()
        status = "coverage_evaluated_on_latest_fold"
    else:
        cal = df
        heldout = pd.DataFrame()
        status = "insufficient_backtests_for_coverage_evaluation"
    pooled = cal["abs_residual"].dropna().astype(float).to_numpy()
    if pooled.size == 0:
        return {
            "status": "insufficient_calibration_residuals",
            "target_coverage": float(target_coverage),
        }
    pooled_q = finite_sample_conformal_quantile(pooled, alpha)
    quantiles: Dict[int, Dict[str, Any]] = {}
    fallback_usage = {"pooled_horizon_residuals": 0}
    for h in sorted(df["horizon_step"].astype(int).unique()):
        vals = cal.loc[cal["horizon_step"].astype(int) == h, "abs_residual"].dropna().astype(float)
        if len(vals) >= 2:
            quantiles[int(h)] = {
                "q": finite_sample_conformal_quantile(vals.to_numpy(), alpha),
                "source": "horizon_specific",
                "n": int(len(vals)),
            }
        else:
            quantiles[int(h)] = {
                "q": pooled_q,
                "source": "pooled_horizon_residuals",
                "n": int(len(vals)),
            }
            fallback_usage["pooled_horizon_residuals"] += 1
    coverage = None
    mean_width = None
    median_width = None
    if not heldout.empty:
        q = heldout["horizon_step"].astype(int).map(lambda h: quantiles[int(h)]["q"]).astype(float)
        lower = np.maximum(0.0, heldout[prediction_col].astype(float) - q)
        upper = heldout[prediction_col].astype(float) + q
        coverage = float(
            (
                (heldout["actual"].astype(float) >= lower)
                & (heldout["actual"].astype(float) <= upper)
            ).mean()
        )
        widths = upper - lower
        mean_width = float(widths.mean())
        median_width = float(widths.median())
    return {
        "status": status,
        "target_coverage": float(target_coverage),
        "alpha": alpha,
        "prediction_col": prediction_col,
        "empirical_coverage_latest_fold": coverage,
        "mean_interval_width_latest_fold": mean_width,
        "median_interval_width_latest_fold": median_width,
        "calibration_sample_counts_by_horizon": {
            int(h): int((cal["horizon_step"].astype(int) == int(h)).sum()) for h in quantiles
        },
        "quantiles_by_horizon": quantiles,
        "fallback_usage": fallback_usage,
        "calibration_split_indexes": (
            sorted(cal["split_index"].unique().tolist()) if "split_index" in cal.columns else []
        ),
        "heldout_split_index": 0 if split_count > 1 else None,
    }


def apply_conformal_intervals(
    point_df: pd.DataFrame,
    interval_summary: Dict[str, Any],
    *,
    point_col: str = "pred",
) -> pd.DataFrame:
    out = point_df.copy()
    quantiles = interval_summary.get("quantiles_by_horizon", {})
    pooled = None
    if quantiles:
        pooled = max(float(v["q"]) for v in quantiles.values())

    def q_for_horizon(h: Any) -> float:
        key = int(h)
        if key in quantiles:
            return float(quantiles[key]["q"])
        if str(key) in quantiles:
            return float(quantiles[str(key)]["q"])
        return float(pooled or 0.0)

    q = out["horizon_step"].map(q_for_horizon).astype(float)
    out["lower"] = np.maximum(0.0, out[point_col].astype(float) - q)
    out["upper"] = out[point_col].astype(float) + q
    out["target_coverage"] = float(interval_summary.get("target_coverage", 0.80))
    return out


def _fit_lgbm_with_inner_split(
    train_df: pd.DataFrame,
    *,
    cfg: ForecastConfig,
    feature_cols: List[str],
    split: BacktestSplit,
) -> Tuple[Any, Dict[str, Any]]:
    """Tune on an inner chronological split, then refit on all outer training rows."""
    X_full = train_df[feature_cols]
    y_full = train_df["sales"].to_numpy(dtype=np.float64)
    metadata: Dict[str, Any] = {
        "inner_train_end_d": split.inner_train_end_d,
        "inner_valid_start_d": split.inner_valid_start_d,
        "inner_valid_end_d": split.inner_valid_end_d,
        "used_inner_early_stopping": False,
        "chosen_classifier_iterations": None,
        "chosen_regressor_iterations": None,
    }

    inner_available = (
        split.inner_train_end_d is not None
        and split.inner_valid_start_d is not None
        and split.inner_valid_end_d is not None
    )
    inner_train = pd.DataFrame()
    inner_valid = pd.DataFrame()
    if inner_available:
        inner_train = train_df.loc[train_df["d"] <= split.inner_train_end_d].copy()
        inner_valid = train_df.loc[
            (train_df["d"] >= split.inner_valid_start_d)
            & (train_df["d"] <= split.inner_valid_end_d)
        ].copy()
        inner_available = not inner_train.empty and not inner_valid.empty

    def clf_params(n_estimators: int) -> Dict[str, Any]:
        return dict(
            objective="binary",
            n_estimators=int(n_estimators),
            learning_rate=0.05,
            num_leaves=128,
            min_child_samples=cfg.min_data_in_leaf,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=cfg.random_state,
            n_jobs=-1,
        )

    def reg_params(n_estimators: int) -> Dict[str, Any]:
        params = dict(
            objective=cfg.objective,
            n_estimators=int(n_estimators),
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            min_child_samples=cfg.min_data_in_leaf,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_lambda=cfg.reg_lambda,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
        if cfg.objective == "tweedie":
            params["tweedie_variance_power"] = cfg.tweedie_variance_power
        return params

    if cfg.two_stage:
        clf_iters = 2000
        reg_iters = int(cfg.n_estimators)
        if inner_available:
            X_inner_train = inner_train[feature_cols]
            y_inner_train = inner_train["sales"].to_numpy(dtype=np.float64)
            X_inner_valid = inner_valid[feature_cols]
            y_inner_valid = inner_valid["sales"].to_numpy(dtype=np.float64)

            y_inner_train_bin = (y_inner_train > 0).astype(int)
            y_inner_valid_bin = (y_inner_valid > 0).astype(int)
            if len(np.unique(y_inner_train_bin)) >= 2 and len(np.unique(y_inner_valid_bin)) >= 2:
                clf_tuned = lgb.LGBMClassifier(**clf_params(clf_iters))
                clf_tuned.fit(
                    X_inner_train,
                    y_inner_train_bin,
                    eval_set=[(X_inner_valid, y_inner_valid_bin)],
                    eval_metric="binary_logloss",
                    callbacks=[lgb.early_stopping(100, verbose=False)],
                )
                clf_iters = int(getattr(clf_tuned, "best_iteration_", 0) or clf_iters)

            pos_inner_train = y_inner_train > 0
            pos_inner_valid = y_inner_valid > 0
            if np.any(pos_inner_train):
                reg_tuned = lgb.LGBMRegressor(**reg_params(reg_iters))
                eval_x = (
                    X_inner_valid.loc[pos_inner_valid] if np.any(pos_inner_valid) else X_inner_valid
                )
                eval_y = (
                    y_inner_valid[pos_inner_valid] if np.any(pos_inner_valid) else y_inner_valid
                )
                reg_tuned.fit(
                    X_inner_train.loc[pos_inner_train],
                    y_inner_train[pos_inner_train],
                    eval_set=[(eval_x, eval_y)],
                    eval_metric="l1",
                    callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
                )
                reg_iters = int(getattr(reg_tuned, "best_iteration_", 0) or reg_iters)

            metadata["used_inner_early_stopping"] = True
        metadata["chosen_classifier_iterations"] = int(clf_iters)
        metadata["chosen_regressor_iterations"] = int(reg_iters)

        clf_final = lgb.LGBMClassifier(**clf_params(clf_iters))
        clf_final.fit(X_full, (y_full > 0).astype(int))

        reg_final = lgb.LGBMRegressor(**reg_params(reg_iters))
        pos_full = y_full > 0
        if np.any(pos_full):
            reg_final.fit(X_full.loc[pos_full], y_full[pos_full])
        else:
            reg_final.fit(X_full, y_full)
        return (clf_final, reg_final), metadata

    reg_iters = int(cfg.n_estimators)
    if inner_available:
        reg_tuned = lgb.LGBMRegressor(**reg_params(reg_iters))
        reg_tuned.fit(
            inner_train[feature_cols],
            inner_train["sales"].to_numpy(dtype=np.float64),
            eval_set=[(inner_valid[feature_cols], inner_valid["sales"].to_numpy(dtype=np.float64))],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
        )
        reg_iters = int(getattr(reg_tuned, "best_iteration_", 0) or reg_iters)
        metadata["used_inner_early_stopping"] = True
    metadata["chosen_regressor_iterations"] = int(reg_iters)

    reg_final = lgb.LGBMRegressor(**reg_params(reg_iters))
    reg_final.fit(X_full, y_full)
    return reg_final, metadata


def run_forecast(cfg: ForecastConfig) -> Dict[str, object]:
    logger = get_logger("forecast")
    if lgb is None:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")

    # 0) Validate inputs early (fail fast)
    if cfg.validate_inputs:
        vrep = validate_m5_inputs(
            ValidationConfig(
                data_dir=cfg.data_dir,
                strict=bool(cfg.validation_strict),
                write_reports=True,
                reports_subdir="reports",
            )
        )
    else:
        vrep = {"skipped": True}

    require_files(
        cfg.data_dir,
        [
            "sales_train_validation.csv",
            "calendar.csv",
            "sell_prices.csv",
            "sample_submission.csv",
        ],
    )

    # -----------------------------
    # 1) Load
    # -----------------------------
    logger.info("Loading CSVs...")
    sales_wide = pd.read_csv(os.path.join(cfg.data_dir, "sales_train_validation.csv"))
    if cfg.max_series and cfg.max_series > 0:
        sales_wide = select_representative_series_subset(sales_wide, max_series=cfg.max_series)
        logger.info("Using subset series: %s", len(sales_wide))
    else:
        logger.info("Using ALL series: %s", len(sales_wide))

    calendar = pd.read_csv(os.path.join(cfg.data_dir, "calendar.csv"))
    calendar["date"] = pd.to_datetime(calendar["date"])
    origin_date = calendar["date"].min()

    prices = pd.read_csv(os.path.join(cfg.data_dir, "sell_prices.csv"))

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    day_cols_sorted = sorted(day_cols, key=_d_to_int)
    if cfg.start_d in day_cols_sorted:
        day_cols_sorted = day_cols_sorted[day_cols_sorted.index(cfg.start_d) :]
    if cfg.last_train_d in day_cols_sorted:
        day_cols_sorted = day_cols_sorted[: day_cols_sorted.index(cfg.last_train_d) + 1]

    # -----------------------------
    # 2) Long + join calendar/prices
    # -----------------------------
    logger.info("Melting to long format...")
    long = sales_wide[id_cols + day_cols_sorted].melt(
        id_vars=id_cols, var_name="d", value_name="sales"
    )
    long["sales"] = long["sales"].astype(np.float32)

    long = long.merge(calendar, on="d", how="left", validate="many_to_one")
    long["snap"] = _make_snap(long)
    long["t"] = (long["date"] - origin_date).dt.days.astype(np.int32)

    long = long.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )
    long = _fill_sell_price_train(long)

    # Future frame (metadata × future calendar)
    last_int = _d_to_int(cfg.last_train_d)
    future_ds = [f"d_{i}" for i in range(last_int + 1, last_int + 1 + cfg.horizon)]
    cal_future = calendar.loc[calendar["d"].isin(future_ds)].copy()

    meta = sales_wide[id_cols].copy()
    meta["_k"] = 1
    cal_future["_k"] = 1
    fut = meta.merge(cal_future, on="_k").drop(columns=["_k"])
    fut["snap"] = _make_snap(fut)
    fut["t"] = (fut["date"] - origin_date).dt.days.astype(np.int32)
    fut = fut.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )
    fut = fut.sort_values(["id", "date"])
    fut = _fill_sell_price_future(fut, long)
    fut["sales"] = np.nan

    cat_cols = [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "weekday",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
    ]
    long2, fut2 = _prepare_categoricals(long.copy(), fut.copy(), cat_cols)

    for df_ in (long2, fut2):
        df_["wday"] = df_["wday"].astype(np.int16)
        df_["month"] = df_["month"].astype(np.int16)
        df_["year"] = df_["year"].astype(np.int16)

    # -----------------------------
    # 3) Feature engineering (past-only)
    # -----------------------------
    logger.info("Feature engineering (lags/rolls/intermittency/prices)...")
    feats = _add_lag_roll_and_intermittency(long2, cfg.lags, cfg.rolls, cfg.nonzero_rolls)
    feats = _add_price_feats(feats)

    max_lag = int(max(cfg.lags))
    usable = feats.dropna(subset=[f"lag_{max_lag}"]).copy()

    feature_cols = (
        [f"lag_{lag}" for lag in cfg.lags]
        + [f"roll_mean_{w}" for w in cfg.rolls]
        + [f"roll_std_{w}" for w in cfg.rolls]
        + [f"roll_sum_{w}" for w in cfg.rolls]
        + [f"nonzero_rate_{w}" for w in cfg.nonzero_rolls]
        + [
            "sell_price",
            "price_lag_1",
            "price_roll_mean_28",
            "price_rel_28",
            "price_pct_change_1",
            "price_changed_today",
            "price_change_count_28",
            "weeks_since_price_change",
            "price_was_missing",
            "snap",
            "wday",
            "month",
            "year",
            "t",
            "days_since_last_sale",
        ]
        + cat_cols
    )

    # -----------------------------
    # 4) Split strategy
    # -----------------------------
    start_int = _d_to_int(cfg.start_d) if cfg.start_d.startswith("d_") else 1
    splits = make_backtest_splits(
        split_strategy=cfg.split_strategy,
        start_day=start_int,
        last_train_day=last_int,
        horizon=int(cfg.horizon),
        n_backtests=int(cfg.n_backtests),
        stride=int(cfg.backtest_stride),
        max_required_history=max_lag,
    )
    train_end_d, _valid_start_d, valid_end_d = (
        splits[0].train_end_d,
        splits[0].valid_start_d,
        splits[0].valid_end_d,
    )

    # -----------------------------
    # 5) Fit/eval helper (single split)
    # -----------------------------
    def _fit_and_eval(
        train_df: pd.DataFrame, valid_df: pd.DataFrame
    ) -> Tuple[Dict[str, Any], Any, np.ndarray]:
        # Baselines
        b1_s = valid_df["baseline_mean_28"]
        b2_s = valid_df["baseline_seasonal_7"].fillna(b1_s)
        b3_s = valid_df["baseline_seasonal_364"].fillna(b2_s)

        yv = valid_df["sales"].to_numpy(dtype=np.float64)
        b1 = b1_s.to_numpy(dtype=np.float64)
        b2 = b2_s.to_numpy(dtype=np.float64)
        b3 = b3_s.to_numpy(dtype=np.float64)

        mae_b1 = _mae(yv, b1)
        mae_b2 = _mae(yv, b2)
        mae_b3 = _mae(yv, b3)

        wmape_b1 = _wmape(yv, b1)
        wmape_b2 = _wmape(yv, b2)
        wmape_b3 = _wmape(yv, b3)

        smape_nz_b1 = _smape_nonzero(yv, b1)
        smape_nz_b2 = _smape_nonzero(yv, b2)
        smape_nz_b3 = _smape_nonzero(yv, b3)

        X_train = train_df[feature_cols]
        y_train = train_df["sales"].to_numpy(dtype=np.float64)
        X_valid = valid_df[feature_cols]
        y_valid = valid_df["sales"].to_numpy(dtype=np.float64)

        # Model metrics (two-stage adds classifier metrics)
        nonzero_auc: float = float("nan")
        nonzero_logloss: float = float("nan")

        if cfg.two_stage:
            # A) classifier
            y_train_bin = (y_train > 0).astype(int)
            y_valid_bin = (y_valid > 0).astype(int)

            clf_params = dict(
                objective="binary",
                n_estimators=2000,
                learning_rate=0.05,
                num_leaves=128,
                min_child_samples=cfg.min_data_in_leaf,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=cfg.random_state,
                n_jobs=-1,
            )
            clf = lgb.LGBMClassifier(**clf_params)
            clf.fit(
                X_train,
                y_train_bin,
                eval_set=[(X_valid, y_valid_bin)],
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )

            # B) regressor on positives
            pos_tr = y_train > 0
            pos_va = y_valid > 0

            X_train_pos = X_train.loc[pos_tr]
            y_train_pos = y_train[pos_tr]

            X_valid_pos = X_valid.loc[pos_va] if np.any(pos_va) else X_train_pos
            y_valid_pos = y_valid[pos_va] if np.any(pos_va) else y_train_pos

            reg_params = dict(
                objective=cfg.objective,
                n_estimators=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                num_leaves=cfg.num_leaves,
                min_child_samples=cfg.min_data_in_leaf,
                subsample=cfg.subsample,
                colsample_bytree=cfg.colsample_bytree,
                reg_lambda=cfg.reg_lambda,
                random_state=cfg.random_state,
                n_jobs=-1,
            )
            if cfg.objective == "tweedie":
                reg_params["tweedie_variance_power"] = cfg.tweedie_variance_power

            reg = lgb.LGBMRegressor(**reg_params)
            reg.fit(
                X_train_pos,
                y_train_pos,
                eval_set=[(X_valid_pos, y_valid_pos)],
                eval_metric="l1",
                callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
            )

            p_valid = clf.predict_proba(X_valid)[:, 1].astype(np.float64)
            q_valid = np.clip(reg.predict(X_valid).astype(np.float64), 0, None)
            pred_valid = np.clip(p_valid * q_valid, 0, None)

            nonzero_auc = _roc_auc_tie_correct(y_valid_bin, p_valid)
            nonzero_logloss = _binary_logloss(y_valid_bin, p_valid)

            model_obj: Any = (clf, reg)

        else:
            params = dict(
                objective=cfg.objective,
                n_estimators=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                num_leaves=cfg.num_leaves,
                min_child_samples=cfg.min_data_in_leaf,
                subsample=cfg.subsample,
                colsample_bytree=cfg.colsample_bytree,
                reg_lambda=cfg.reg_lambda,
                random_state=cfg.random_state,
                n_jobs=-1,
            )
            if cfg.objective == "tweedie":
                params["tweedie_variance_power"] = cfg.tweedie_variance_power

            reg = lgb.LGBMRegressor(**params)
            reg.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="l1",
                callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
            )
            pred_valid = np.clip(reg.predict(X_valid).astype(np.float64), 0, None)
            model_obj = reg

        mae_lgbm = _mae(yv, pred_valid)
        wmape_lgbm = _wmape(yv, pred_valid)
        smape_lgbm = _smape(yv, pred_valid)
        smape_nonzero_lgbm = _smape_nonzero(yv, pred_valid)

        metrics = {
            # Baselines
            "mae_baseline_mean_28": float(mae_b1),
            "mae_baseline_seas_7": float(mae_b2),
            "mae_baseline_seas_364": float(mae_b3),
            "wmape_baseline_mean_28": float(wmape_b1),
            "wmape_baseline_seas_7": float(wmape_b2),
            "wmape_baseline_seas_364": float(wmape_b3),
            "smape_nonzero_baseline_mean_28": (
                float(smape_nz_b1) if np.isfinite(smape_nz_b1) else float("nan")
            ),
            "smape_nonzero_baseline_seas_7": (
                float(smape_nz_b2) if np.isfinite(smape_nz_b2) else float("nan")
            ),
            "smape_nonzero_baseline_seas_364": (
                float(smape_nz_b3) if np.isfinite(smape_nz_b3) else float("nan")
            ),
            # Model
            "mae_lgbm": float(mae_lgbm),
            "wmape_lgbm": float(wmape_lgbm),
            "smape_lgbm": float(smape_lgbm),
            "smape_nonzero_lgbm": (
                float(smape_nonzero_lgbm) if np.isfinite(smape_nonzero_lgbm) else float("nan")
            ),
            "nonzero_auc": float(nonzero_auc) if np.isfinite(nonzero_auc) else float("nan"),
            "nonzero_logloss": (
                float(nonzero_logloss) if np.isfinite(nonzero_logloss) else float("nan")
            ),
        }
        return metrics, model_obj, pred_valid

    # -----------------------------
    # 6) Rolling-origin backtests (multiple cutoffs)
    # -----------------------------
    backtests: List[Dict[str, Any]] = []
    model_latest: Any = None
    pred_latest: Optional[np.ndarray] = None
    valid_latest: Optional[pd.DataFrame] = None
    metrics_latest: Optional[Dict[str, Any]] = None
    evaluation_frames: List[pd.DataFrame] = []

    for j, split in enumerate(splits):
        train_df = usable.loc[usable["d"].map(_d_to_int) <= _d_to_int(split.train_end_d)].copy()
        if train_df.empty:
            continue
        model_obj, fit_meta = _fit_lgbm_with_inner_split(
            train_df, cfg=cfg, feature_cols=feature_cols, split=split
        )
        eval_rows = recursive_evaluate_split(
            long2,
            split=split,
            model_obj=model_obj,
            cfg=cfg,
            feature_cols=feature_cols,
            cat_cols=cat_cols,
            category_reference=long2,
            origin_date=origin_date,
        )
        if eval_rows.empty:
            continue
        eval_rows.insert(0, "split_index", j)
        bt = split_diagnostics_from_evaluation(eval_rows, split)
        bt["fit_metadata"] = fit_meta
        bt["train_rows"] = int(len(train_df))
        bt["valid_rows"] = int(len(eval_rows))
        backtests.append(bt)
        evaluation_frames.append(eval_rows)
        if j == 0:
            model_latest = model_obj
            valid_latest = eval_rows.copy()
            pred_latest = eval_rows["pred_lgbm"].to_numpy(dtype=np.float64)
            metrics_latest = {
                "wmape_lgbm": float(bt["wmape_lgbm"]),
                "wmape_baseline_mean_28": float(bt["wmape_baseline_mean_28"]),
                "wmape_baseline_seas_7": float(bt["wmape_baseline_seasonal_7"]),
                "wmape_baseline_seas_364": float(bt["wmape_baseline_seasonal_364"]),
            }

    for j, (tr_end, va_start, va_end) in enumerate([]):
        train_df = usable.loc[usable["d"] <= tr_end].copy()
        valid_df = usable.loc[(usable["d"] >= va_start) & (usable["d"] <= va_end)].copy()
        if train_df.empty or valid_df.empty:
            continue

        m_split, model_obj, pred_valid = _fit_and_eval(train_df, valid_df)

        # segment metrics (events, price drops) on this split
        valid_df2 = valid_df.copy()
        valid_df2["_pred"] = pred_valid
        seg = add_segment_metrics(valid_df2, y_col="sales", yhat_col="_pred")

        # Confidence interval calibration (residual quantiles) only for the latest split
        resid_q = {}
        if j == 0:
            resid = (valid_df2["sales"].to_numpy(dtype=np.float64) - pred_valid).astype(np.float64)
            resid_q = {
                "residual_q10": float(np.quantile(resid, 0.10)),
                "residual_q50": float(np.quantile(resid, 0.50)),
                "residual_q90": float(np.quantile(resid, 0.90)),
            }
            # Segment residual quantiles (more honest than one global band)
            if ("event_name_1" in valid_df2.columns) and ("event_name_2" in valid_df2.columns):
                is_event = valid_df2["event_name_1"].notna() | valid_df2["event_name_2"].notna()
                for nm, mask in [("event", is_event), ("non_event", ~is_event)]:
                    rr = resid[mask.to_numpy()] if mask.any() else np.array([], dtype=float)
                    if rr.size:
                        resid_q[f"residual_q10_{nm}"] = float(np.quantile(rr, 0.10))
                        resid_q[f"residual_q50_{nm}"] = float(np.quantile(rr, 0.50))
                        resid_q[f"residual_q90_{nm}"] = float(np.quantile(rr, 0.90))

        bt = {
            "split": {"train_end_d": tr_end, "valid_start_d": va_start, "valid_end_d": va_end},
            **m_split,
            **seg,
            **resid_q,
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
        }
        backtests.append(bt)

        if j == 0:
            model_latest = model_obj
            pred_latest = pred_valid
            valid_latest = valid_df2
            metrics_latest = m_split

    if not backtests or metrics_latest is None or valid_latest is None or pred_latest is None:
        raise RuntimeError(
            "Backtesting produced no valid splits; check start_d/last_train_d and horizon."
        )

    # -----------------------------
    # 7) Choose winner using latest split WMAPE (business-friendly)
    # -----------------------------
    best_baseline_wmape = min(
        metrics_latest["wmape_baseline_mean_28"],
        metrics_latest["wmape_baseline_seas_7"],
        metrics_latest["wmape_baseline_seas_364"],
    )
    wmape_lgbm = float(metrics_latest["wmape_lgbm"])
    winner = "lgbm" if wmape_lgbm <= best_baseline_wmape else "baseline"

    logger.info(
        "Latest split (%s→%s): best_baseline_wmape=%.4f, lgbm_wmape=%.4f => winner=%s",
        train_end_d,
        valid_end_d,
        float(best_baseline_wmape),
        float(wmape_lgbm),
        winner,
    )

    promotion = evaluate_promotion_policy(
        backtests,
        PromotionPolicy(
            promotion_min_backtests=cfg.promotion_min_backtests,
            promotion_min_aggregate_improvement_pct=cfg.promotion_min_aggregate_improvement_pct,
            promotion_min_win_rate=cfg.promotion_min_win_rate,
            promotion_max_single_split_regression_pct=cfg.promotion_max_single_split_regression_pct,
            promotion_max_critical_segment_regression_pct=(
                cfg.promotion_max_critical_segment_regression_pct
            ),
            promotion_min_segment_actual_sum=cfg.promotion_min_segment_actual_sum,
            allow_single_split_promotion=cfg.allow_single_split_promotion,
        ),
    )
    winner = str(promotion["winner"])
    selected_baseline = str(promotion["selected_baseline"])
    served_prediction_col = (
        "pred_lgbm" if winner == "lgbm" else BASELINE_NAME_TO_COL[selected_baseline]
    )
    evaluation_all = pd.concat(evaluation_frames, ignore_index=True)
    prediction_interval_summary = build_conformal_interval_summary(
        evaluation_all,
        prediction_col=served_prediction_col,
        target_coverage=float(cfg.prediction_interval_coverage),
    )
    logger.info(
        "Promotion policy: status=%s selected_baseline=%s reasons=%s",
        promotion["promotion_status"],
        selected_baseline,
        promotion["promotion_reasons"],
    )

    # -----------------------------
    # 8) Fit final model on FULL history (for serving) using inner-stopped iterations
    # -----------------------------
    model_for_serving: Any = None
    if winner == "lgbm":
        full_df = usable.loc[usable["d"] <= cfg.last_train_d].copy()
        X_full = full_df[feature_cols]
        y_full = full_df["sales"].to_numpy(dtype=np.float64)

        if cfg.two_stage:
            clf0, reg0 = model_latest  # type: ignore[misc]
            best_clf = int(getattr(clf0, "best_iteration_", 0) or 0)
            best_reg = int(getattr(reg0, "best_iteration_", 0) or 0)

            clf_params = clf0.get_params()
            if best_clf > 0:
                clf_params["n_estimators"] = best_clf
            clf_f = lgb.LGBMClassifier(**clf_params)
            clf_f.fit(X_full, (y_full > 0).astype(int))

            reg_params = reg0.get_params()
            if best_reg > 0:
                reg_params["n_estimators"] = best_reg
            reg_f = lgb.LGBMRegressor(**reg_params)
            pos = y_full > 0
            reg_f.fit(X_full.loc[pos], y_full[pos])

            model_for_serving = (clf_f, reg_f)
        else:
            reg0 = model_latest
            best_reg = int(getattr(reg0, "best_iteration_", 0) or 0)
            params = reg0.get_params()
            if best_reg > 0:
                params["n_estimators"] = best_reg
            reg_f = lgb.LGBMRegressor(**params)
            reg_f.fit(X_full, y_full)
            model_for_serving = reg_f
    else:
        model_for_serving = model_latest  # unused, but keep for artefacts

    # -----------------------------
    # 9) Forecast recursively into the future
    # -----------------------------
    logger.info("Forecasting %s days ahead (recursive)...", cfg.horizon)

    hist_sales = {
        k: g["sales"].astype(float).tolist()
        for k, g in long.sort_values(["id", "date"]).groupby("id")
    }
    hist_price = {
        k: g["sell_price"].astype(float).tolist()
        for k, g in long.sort_values(["id", "date"]).groupby("id")
    }

    fut2 = fut2.sort_values(["date", "id"]).copy()
    dates = sorted(fut2["date"].unique())
    date_to_h = {d: i + 1 for i, d in enumerate(dates)}

    preds = []
    for day in dates:
        day_rows = fut2.loc[fut2["date"] == day].copy()
        rows = []

        for r in day_rows.itertuples(index=False):
            _id = r.id
            sh = hist_sales[_id]
            ph = hist_price[_id]

            def lag(L: int):
                return sh[-L] if len(sh) >= L else np.nan

            def roll_mean(W: int):
                x = sh[-W:] if len(sh) >= W else sh
                return float(np.mean(x)) if len(x) else np.nan

            def roll_std(W: int):
                x = sh[-W:] if len(sh) >= W else sh
                return float(np.std(x, ddof=0)) if len(x) > 1 else 0.0

            def roll_sum(W: int):
                x = sh[-W:] if len(sh) >= W else sh
                return float(np.sum(x)) if len(x) else np.nan

            def days_since_last_sale():
                for i2 in range(1, len(sh) + 1):
                    if sh[-i2] > 0:
                        return float(i2)
                return 9999.0

            base_pred = _baseline_prediction_from_history(sh, selected_baseline)

            # price
            price = float(r.sell_price) if pd.notna(r.sell_price) else (ph[-1] if ph else np.nan)
            price_lag_1 = ph[-1] if ph else np.nan
            observed_prices = [float(v) for v in ph[-28:] if np.isfinite(float(v))]
            price_roll_mean_28 = float(np.mean(observed_prices)) if observed_prices else np.nan
            price_rel_28 = (
                (price / (price_roll_mean_28 + 1e-6) - 1.0)
                if pd.notna(price) and pd.notna(price_roll_mean_28)
                else np.nan
            )
            price_pct_change_1 = (
                (price / (price_lag_1 + 1e-6) - 1.0)
                if pd.notna(price) and pd.notna(price_lag_1)
                else np.nan
            )
            price_changed_today = (
                pd.notna(price) and pd.notna(price_lag_1) and not np.isclose(price, price_lag_1)
            )
            price_was_missing = 0 if pd.notna(r.sell_price) else 1

            # count price changes over last 28 days (includes today)
            all_prices = [v for v in (ph[-27:] if len(ph) >= 27 else ph) if np.isfinite(float(v))]
            if np.isfinite(price):
                all_prices.append(price)
            price_change_count_28 = 0.0
            if len(all_prices) >= 2:
                price_change_count_28 = float(
                    sum(
                        1
                        for j in range(1, len(all_prices))
                        if not np.isclose(all_prices[j], all_prices[j - 1])
                    )
                )

            # weeks since last price change
            days_since_change = 9999.0
            if len(ph) >= 1:
                hp = [v for v in ph if np.isfinite(float(v))]
                if np.isfinite(price):
                    hp.append(price)
                last_change_idx = None
                for j in range(len(hp) - 1, 0, -1):
                    if not np.isclose(hp[j], hp[j - 1]):
                        last_change_idx = j
                        break
                if last_change_idx is not None:
                    days_since_change = float((len(hp) - 1) - last_change_idx)
            weeks_since_price_change = float(days_since_change / 7.0)

            # intermittency
            def nonzero_rate(W: int):
                x = sh[-W:] if len(sh) >= W else sh
                return float(np.mean([1.0 if v > 0 else 0.0 for v in x])) if len(x) else 0.0

            row = {
                "id": _id,
                "sell_price": price,
                "price_lag_1": price_lag_1,
                "price_roll_mean_28": price_roll_mean_28,
                "price_rel_28": price_rel_28,
                "price_pct_change_1": price_pct_change_1,
                "price_changed_today": float(price_changed_today),
                "price_change_count_28": price_change_count_28,
                "weeks_since_price_change": weeks_since_price_change,
                "price_was_missing": int(price_was_missing),
                "t": int((day - origin_date).days),
                "days_since_last_sale": days_since_last_sale(),
                "snap": int(r.snap),
                "wday": int(r.wday),
                "month": int(r.month),
                "year": int(r.year),
                "_baseline": base_pred,
                "_baseline_family": selected_baseline,
            }
            for L in cfg.lags:
                row[f"lag_{L}"] = lag(L)
            for W in cfg.rolls:
                row[f"roll_mean_{W}"] = roll_mean(W)
                row[f"roll_std_{W}"] = roll_std(W)
                row[f"roll_sum_{W}"] = roll_sum(W)
            for W in cfg.nonzero_rolls:
                row[f"nonzero_rate_{W}"] = nonzero_rate(W)
            for c in cat_cols:
                row[c] = getattr(r, c)
            rows.append(row)

        X_day = pd.DataFrame(rows)
        for c in cat_cols:
            X_day[c] = pd.Categorical(X_day[c].astype(str), categories=long2[c].cat.categories)

        if winner == "lgbm":
            if cfg.two_stage:
                clf_f, reg_f = model_for_serving  # type: ignore[misc]
                p = clf_f.predict_proba(X_day[feature_cols])[:, 1].astype(np.float64)
                q = np.clip(reg_f.predict(X_day[feature_cols]).astype(np.float64), 0, None)
                yhat = np.clip(p * q, 0, None)
            else:
                yhat = np.clip(model_for_serving.predict(X_day[feature_cols]).astype(np.float64), 0, None)  # type: ignore[union-attr]
        else:
            yhat = np.asarray(X_day["_baseline"].values, dtype=np.float64)
            yhat = np.clip(yhat, 0, None)

        for i2, _id in enumerate(X_day["id"].values):
            hist_sales[_id].append(float(yhat[i2]) if np.isfinite(yhat[i2]) else 0.0)
            sp = X_day.loc[i2, "sell_price"]
            hist_price[_id].append(
                float(sp) if pd.notna(sp) else (hist_price[_id][-1] if hist_price[_id] else np.nan)
            )

        preds.append(
            pd.DataFrame(
                {"id": X_day["id"].values, "h": date_to_h[day], "pred": yhat.astype(np.float32)}
            )
        )

    pred_df = pd.concat(preds, ignore_index=True)
    interval_points = pred_df.rename(columns={"h": "horizon_step"}).copy()
    interval_df = apply_conformal_intervals(
        interval_points, prediction_interval_summary, point_col="pred"
    )
    wide = pred_df.pivot(index="id", columns="h", values="pred").reset_index()
    wide.columns = ["id"] + [f"F{i}" for i in range(1, cfg.horizon + 1)]

    sample_sub = pd.read_csv(os.path.join(cfg.data_dir, "sample_submission.csv"))[["id"]]
    submit = sample_sub.merge(wide, on="id", how="left")
    for i2 in range(1, cfg.horizon + 1):
        submit[f"F{i2}"] = submit[f"F{i2}"].fillna(0.0)

    out_path = os.path.join(cfg.data_dir, cfg.out_submission)
    submit.to_csv(out_path, index=False)
    logger.info("Wrote: %s", out_path)

    # -----------------------------
    # 10) Artefacts (model + metadata)
    # -----------------------------
    # Prepare a stable run id (timestamp + config hash)
    cfg_dict = {k: v for k, v in cfg.__dict__.items() if k not in {"data_dir"}}
    cfg_hash = _stable_hash({"config": cfg_dict, "feature_cols": feature_cols})
    run_id = (cfg.run_name or _now_utc_compact()) + "_" + cfg_hash[:10]

    artefacts = {}
    if cfg.save_artifacts:
        run_dir = _ensure_dir(os.path.join(cfg.data_dir, cfg.artifacts_dir, run_id))
        artefacts = {
            "run_id": run_id,
            "run_dir": run_dir,
            "config_hash": cfg_hash,
        }

        # Write metadata
        write_json(os.path.join(run_dir, "config.json"), cfg_dict)
        write_json(os.path.join(run_dir, "feature_cols.json"), feature_cols)
        write_json(os.path.join(run_dir, "backtests.json"), backtests)
        write_json(os.path.join(run_dir, "promotion.json"), promotion)
        write_json(
            os.path.join(run_dir, "prediction_intervals_summary.json"), prediction_interval_summary
        )
        evaluation_all.to_csv(os.path.join(run_dir, "backtest_recursive_rows.csv"), index=False)
        interval_df.to_csv(os.path.join(run_dir, "prediction_intervals.csv"), index=False)
        write_json(
            os.path.join(run_dir, "system.json"),
            {
                "python": sys.version,
                "platform": platform.platform(),
                "lightgbm": getattr(lgb, "__version__", "unknown"),
            },
        )

        # Data quality report for the latest validation frame (monitoring baseline)
        dq = data_quality_report(valid_latest, cols=["actual", served_prediction_col], name="valid_frame")  # type: ignore[arg-type]
        write_json(os.path.join(run_dir, "data_quality_valid.json"), dq)

        # Save model
        joblib.dump(
            {"winner": winner, "two_stage": bool(cfg.two_stage), "model": model_for_serving},
            os.path.join(run_dir, "model.joblib"),
        )

    latest = backtests[0]
    latest_eval = evaluation_all.loc[evaluation_all["split_index"] == 0].copy()
    latest_resid = (
        latest_eval["actual"].astype(float) - latest_eval[served_prediction_col].astype(float)
    ).to_numpy(dtype=np.float64)
    resid_payload = {
        "residual_q10": float(np.quantile(latest_resid, 0.10)),
        "residual_q50": float(np.quantile(latest_resid, 0.50)),
        "residual_q90": float(np.quantile(latest_resid, 0.90)),
        "deprecated_residual_quantiles": True,
    }

    return {
        "submission_path": out_path,
        "winner": winner,
        "selected_baseline": selected_baseline,
        "promotion": promotion,
        "prediction_intervals": prediction_interval_summary,
        "n_series": int(len(sales_wide)),
        "backtests": backtests,
        "latest_split": latest.get("split", {}),
        **metrics_latest,
        **{k: v for k, v in latest.items() if k.startswith("wmape_") and k not in metrics_latest},
        **resid_payload,
        "validation": vrep,
        "artifacts": artefacts,
    }
