from __future__ import annotations

import os
import sys
import datetime
import platform
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal

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
except Exception:
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
    return datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


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
      - for leading NaNs, fill with the first observed price within the same id (from observed history)
      - if an id has no observed prices at all, fall back to global median
    Also creates a price_was_missing flag BEFORE filling.
    """
    df = df.sort_values(["id", "date"]).copy()
    df["price_was_missing"] = df["sell_price"].isna().astype(np.int8)

    # past-only fill
    ffilled = df.groupby("id")["sell_price"].ffill()

    # first observed per id (does not look forward in time beyond observed values)
    first_obs = df.groupby("id")["sell_price"].transform(
        lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan
    )

    filled = ffilled.fillna(first_obs)

    # global fallback (rare)
    global_med = df["sell_price"].median()
    df["sell_price"] = filled.fillna(global_med).astype(np.float32)

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

    fut_ff = fut.groupby("id")["sell_price"].ffill()

    last_train_price = train_filled.groupby("id")["sell_price"].last()
    fut["sell_price"] = fut_ff

    # carry last train price
    fut["sell_price"] = fut["sell_price"].fillna(fut["id"].map(last_train_price))

    # final fallback
    global_med = float(train_filled["sell_price"].median())
    fut["sell_price"] = fut["sell_price"].fillna(global_med).astype(np.float32)

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

    df["price_rel_28"] = (df["sell_price"] / (df["price_roll_mean_28"] + 1e-6) - 1.0).astype(
        np.float32
    )

    # Price momentum
    df["price_pct_change_1"] = (df["sell_price"] / (df["price_lag_1"] + 1e-6) - 1.0).astype(
        np.float32
    )

    # Price change features
    price_changed = (df["sell_price"] != df["price_lag_1"]) & df["price_lag_1"].notna()
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
    usable = feats.dropna(subset=[f"lag_{max_lag}", "sell_price"]).copy()

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
    # 4) Split strategy (rolling origin)
    # -----------------------------
    def _mk_splits() -> List[Tuple[str, str, str]]:
        start_int = _d_to_int(cfg.start_d) if cfg.start_d.startswith("d_") else _d_to_int("d_1")
        splits: List[Tuple[str, str, str]] = []
        stride = int(cfg.backtest_stride)
        for i in range(int(cfg.n_backtests)):
            valid_end = last_int - i * stride
            valid_start = valid_end - int(cfg.horizon) + 1
            train_end = valid_start - 1
            if train_end <= start_int + max_lag:
                break
            splits.append((f"d_{train_end}", f"d_{valid_start}", f"d_{valid_end}"))
        if not splits:
            # fallback: last window
            valid_start = last_int - int(cfg.horizon) + 1
            splits = [(f"d_{valid_start - 1}", f"d_{valid_start}", f"d_{last_int}")]
        return splits

    splits = _mk_splits()
    # Most recent split = first
    train_end_d, valid_start_d, valid_end_d = splits[0]

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

    for j, (tr_end, va_start, va_end) in enumerate(splits):
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

    # -----------------------------
    # 8) Fit final model on FULL history (for serving) using early-stopped iterations
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

            # baselines
            mean28 = roll_mean(28)
            seas7 = sh[-7] if len(sh) >= 7 else np.nan
            seas364 = sh[-364] if len(sh) >= 364 else np.nan
            base_pred = (
                seas364 if np.isfinite(seas364) else (seas7 if np.isfinite(seas7) else mean28)
            )
            base_pred = float(max(0.0, base_pred if np.isfinite(base_pred) else 0.0))

            # price
            price = float(r.sell_price) if pd.notna(r.sell_price) else (ph[-1] if ph else np.nan)
            price_lag_1 = ph[-1] if ph else np.nan
            price_roll_mean_28 = float(np.mean(ph[-28:])) if len(ph) else np.nan
            price_rel_28 = (
                (price / (price_roll_mean_28 + 1e-6) - 1.0)
                if pd.notna(price) and pd.notna(price_roll_mean_28)
                else 0.0
            )
            price_pct_change_1 = (
                (price / (price_lag_1 + 1e-6) - 1.0)
                if pd.notna(price) and pd.notna(price_lag_1)
                else 0.0
            )
            price_was_missing = 0 if pd.notna(r.sell_price) else 1

            # count price changes over last 28 days (includes today)
            all_prices = (ph[-28:] if len(ph) >= 28 else ph) + [price]
            price_change_count_28 = 0.0
            if len(all_prices) >= 2:
                price_change_count_28 = float(
                    sum(1 for j in range(1, len(all_prices)) if all_prices[j] != all_prices[j - 1])
                )

            # weeks since last price change
            days_since_change = 9999.0
            if len(ph) >= 1:
                hp = ph + [price]
                last_change_idx = None
                for j in range(len(hp) - 1, 0, -1):
                    if hp[j] != hp[j - 1]:
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
        write_json(
            os.path.join(run_dir, "system.json"),
            {
                "python": sys.version,
                "platform": platform.platform(),
                "lightgbm": getattr(lgb, "__version__", "unknown"),
            },
        )

        # Data quality report for the latest validation frame (monitoring baseline)
        dq = data_quality_report(valid_latest, cols=["sales", "sell_price"], name="valid_frame")  # type: ignore[arg-type]
        write_json(os.path.join(run_dir, "data_quality_valid.json"), dq)

        # Save model
        joblib.dump(
            {"winner": winner, "two_stage": bool(cfg.two_stage), "model": model_for_serving},
            os.path.join(run_dir, "model.joblib"),
        )

    # Pull residual quantiles from latest backtest (j==0)
    latest = backtests[0]
    resid_payload = {k: latest[k] for k in latest.keys() if k.startswith("residual_q")}

    return {
        "submission_path": out_path,
        "winner": winner,
        "n_series": int(len(sales_wide)),
        "backtests": backtests,
        "latest_split": latest.get("split", {}),
        **metrics_latest,
        **{k: v for k, v in latest.items() if k.startswith("wmape_") and k not in metrics_latest},
        **resid_payload,
        "validation": vrep,
        "artifacts": artefacts,
    }
