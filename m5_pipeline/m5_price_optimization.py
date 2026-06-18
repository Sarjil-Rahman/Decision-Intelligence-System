from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from .utils import get_logger, require_files, write_json, select_representative_series_subset


@dataclass
class PriceOptConfig:
    data_dir: str
    submission_path: str = "submission.csv"
    out_path: str = "price_optimization_results.csv"

    # Which training day represents "current" price in sell_prices
    last_train_d: str = "d_1913"

    # Unit economics proxy (fallback if you don't provide item-level margins/costs)
    margin: float = 0.30

    # Optional: item/category/store specific unit economics (more realistic)
    # Supported schemas:
    #  A) costs.csv  columns: store_id?, item_id?, cat_id?, cost
    #  B) margins.csv columns: store_id?, item_id?, cat_id?, margin  (0..1)
    unit_econ_path: Optional[str] = None  # relative to data_dir

    # Speed control
    max_series: Optional[int] = 0
    lookback_days: int = 365
    elasticity_end_d: Optional[str] = (
        None  # if set, estimate elasticities using history up to this day (for backtests)
    )

    # --- Guardrails (industry-standard) ---
    # Elasticity must be negative. Clip to a realistic range (upper bound is still negative).
    elasticity_clip: Tuple[float, float] = (-5.0, -0.1)

    # Cap absolute price moves vs current price (e.g. 0.20 => +/-20%)
    max_abs_price_change_pct: float = 0.20

    # Candidate deltas (will be filtered by max_abs_price_change_pct)
    price_grid: Tuple[float, ...] = (-0.20, -0.10, 0.0, 0.10, 0.20)

    # Prevent absurd outcomes (helps on noisy real-world elasticities)
    min_price: float = 0.01
    max_demand_mult: float = 3.0  # q <= base * max_demand_mult (sanity cap)

    # "Suspicious uplift" guardrails (helps credibility)
    suspicious_profit_gain_pct: float = 400.0  # flag if profit_gain/base_profit > 4x
    suspicious_demand_gain_pct: float = 500.0  # flag if demand_gain/base_demand > 5x

    # Write extra reports
    write_reports: bool = True
    reports_subdir: str = "reports"


def _estimate_elasticity_weekly(
    sales_wide: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    lookback_days: int,
    elasticity_clip: Tuple[float, float],
    *,
    end_d: Optional[str] = None,
) -> pd.DataFrame:
    """Estimate per-series price elasticity using a simple weekly log-log regression.

    Model: log(1 + Q_week) ~ beta * log(P_week)

    Guardrails:
      - If beta >= 0 (nonsense), return NaN and later fall back to category/global.
      - If beta < 0, clip to elasticity_clip (e.g. [-5, -0.1]).

    end_d: if provided (e.g. 'd_1800'), only use history up to that day.
           Useful for *backtesting*.
    """
    clip_lo, clip_hi = float(elasticity_clip[0]), float(elasticity_clip[1])
    if not (clip_lo < clip_hi < 0):
        raise ValueError("elasticity_clip must be (lo, hi) with lo < hi < 0")

    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    day_cols_sorted = sorted(day_cols, key=lambda x: int(x.split("_")[1]))

    if end_d is not None and end_d in day_cols_sorted:
        day_cols_sorted = day_cols_sorted[: day_cols_sorted.index(end_d) + 1]

    day_cols_sorted = day_cols_sorted[-int(lookback_days) :]

    id_cols = ["id", "item_id", "cat_id", "store_id", "state_id"]
    long = sales_wide[id_cols + day_cols_sorted].melt(
        id_vars=id_cols, var_name="d", value_name="sales"
    )
    long = long.merge(calendar[["d", "wm_yr_wk"]], on="d", how="left", validate="many_to_one")

    wk_sales = long.groupby(["id", "store_id", "item_id", "cat_id", "wm_yr_wk"], as_index=False)[
        "sales"
    ].sum()

    wk = (
        wk_sales.merge(
            prices, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="many_to_one"
        )
        .dropna(subset=["sell_price"])
        .copy()
    )

    wk["y"] = np.log1p(wk["sales"].astype(float))
    wk["x"] = np.log(wk["sell_price"].astype(float) + 1e-9)

    out: List[Tuple[str, float]] = []
    for _id, g in wk.groupby("id", sort=False):
        # Require price movement; otherwise the regression is meaningless.
        if g["x"].nunique() < 4 or len(g) < 10:
            out.append((_id, np.nan))
            continue

        x = g["x"].to_numpy(dtype=np.float64)
        y = g["y"].to_numpy(dtype=np.float64)

        x0 = x - x.mean()
        var = float(np.mean(x0 * x0))
        if var < 1e-10:
            out.append((_id, np.nan))
            continue

        beta = float(np.mean(x0 * (y - y.mean())) / var)

        # Guardrail 1: elasticity must be negative
        if not np.isfinite(beta) or beta >= 0:
            out.append((_id, np.nan))
            continue

        # Guardrail 2: clip to realistic negative range
        out.append((_id, float(np.clip(beta, clip_lo, clip_hi))))

    return pd.DataFrame(out, columns=["id", "elasticity"])


def _estimate_elasticity_weekly_segment(
    sales_wide: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    lookback_days: int,
    elasticity_clip: Tuple[float, float],
    *,
    end_d: Optional[str] = None,
    segment: str = "all",  # "all" | "event" | "non_event"
) -> pd.DataFrame:
    """Like _estimate_elasticity_weekly but restricted to event-weeks or non-event-weeks.

    We define an "event week" as any wm_yr_wk that contains >=1 event day
    (event_name_1 OR event_name_2 is not null in M5 calendar).
    """
    if segment not in {"all", "event", "non_event"}:
        raise ValueError("segment must be one of: all, event, non_event")

    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    day_cols_sorted = sorted(day_cols, key=lambda x: int(x.split("_")[1]))

    if end_d is not None and end_d in day_cols_sorted:
        day_cols_sorted = day_cols_sorted[: day_cols_sorted.index(end_d) + 1]

    day_cols_sorted = day_cols_sorted[-int(lookback_days) :]

    id_cols = ["id", "item_id", "cat_id", "store_id", "state_id"]
    long = sales_wide[id_cols + day_cols_sorted].melt(
        id_vars=id_cols, var_name="d", value_name="sales"
    )

    # Event-day flag (calendar is global)
    has_event_cols = all(c in calendar.columns for c in ["event_name_1", "event_name_2"])
    cal = calendar[["d", "wm_yr_wk"]].copy()
    if has_event_cols:
        cal["_is_event_day"] = calendar["event_name_1"].notna() | calendar["event_name_2"].notna()
    else:
        cal["_is_event_day"] = False

    # Event-week flag
    wk_flag = (
        cal.groupby("wm_yr_wk", as_index=False)["_is_event_day"]
        .any()
        .rename(columns={"_is_event_day": "_is_event_week"})
    )
    cal = cal.merge(wk_flag, on="wm_yr_wk", how="left")

    long = long.merge(
        cal[["d", "wm_yr_wk", "_is_event_week"]], on="d", how="left", validate="many_to_one"
    )

    wk_sales = long.groupby(
        ["id", "store_id", "item_id", "cat_id", "wm_yr_wk", "_is_event_week"], as_index=False
    )["sales"].sum()

    wk = (
        wk_sales.merge(
            prices, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="many_to_one"
        )
        .dropna(subset=["sell_price"])
        .copy()
    )

    if segment == "event":
        wk = wk[wk["_is_event_week"]]
    elif segment == "non_event":
        wk = wk[~wk["_is_event_week"]]

    wk["y"] = np.log1p(wk["sales"].astype(float))
    wk["x"] = np.log(wk["sell_price"].astype(float) + 1e-9)

    clip_lo, clip_hi = float(elasticity_clip[0]), float(elasticity_clip[1])
    out: List[Tuple[str, float]] = []
    for _id, g in wk.groupby("id", sort=False):
        if g["x"].nunique() < 4 or len(g) < 10:
            out.append((_id, np.nan))
            continue
        x = g["x"].to_numpy(dtype=np.float64)
        y = g["y"].to_numpy(dtype=np.float64)

        x0 = x - x.mean()
        var = float(np.mean(x0 * x0))
        if var < 1e-10:
            out.append((_id, np.nan))
            continue

        beta = float(np.mean(x0 * (y - y.mean())) / var)
        if not np.isfinite(beta) or beta >= 0:
            out.append((_id, np.nan))
            continue
        out.append((_id, float(np.clip(beta, clip_lo, clip_hi))))

    return pd.DataFrame(out, columns=["id", "elasticity"])


def _q(series: pd.Series, qs=(0.10, 0.50, 0.90)) -> Dict[str, float]:
    """Safe quantiles for reporting (handles empty/inf/nan)."""
    vals = series.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(vals) == 0:
        return {f"p{int(q * 100)}": float("nan") for q in qs}
    return {f"p{int(q * 100)}": float(np.quantile(vals.to_numpy(), q)) for q in qs}


def _build_price_opt_summary(
    out: pd.DataFrame,
    *,
    el_event: Optional[pd.DataFrame] = None,
    el_nonevent: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Stakeholder-grade summary: £ uplift, distributions, contributions, risk bands, behaviour insights."""
    df = out.copy()
    df["uplift_gbp"] = df["best_profit"] - df["base_profit"]

    base_profit_gbp = float(df["base_profit"].sum())
    optimised_profit_gbp = float(df["best_profit"].sum())
    uplift_gbp = float(optimised_profit_gbp - base_profit_gbp)
    uplift_pct = float((uplift_gbp / (abs(base_profit_gbp) + 1e-9)) * 100.0)

    uplift_dist_gbp = _q(df["uplift_gbp"])
    uplift_dist_pct = _q((df["uplift_gbp"] / (df["base_profit"].abs() + 1e-9)) * 100.0)

    by_store = df.groupby("store_id", as_index=False).agg(
        base_profit_gbp=("base_profit", "sum"),
        optimised_profit_gbp=("best_profit", "sum"),
        uplift_gbp=("uplift_gbp", "sum"),
    )
    by_store["uplift_pct"] = (
        by_store["uplift_gbp"] / (by_store["base_profit_gbp"].abs() + 1e-9)
    ) * 100.0

    by_cat = df.groupby("cat_id", as_index=False).agg(
        base_profit_gbp=("base_profit", "sum"),
        optimised_profit_gbp=("best_profit", "sum"),
        uplift_gbp=("uplift_gbp", "sum"),
    )
    by_cat["uplift_pct"] = (by_cat["uplift_gbp"] / (by_cat["base_profit_gbp"].abs() + 1e-9)) * 100.0

    # Top/bottom contributors (by £ uplift)
    keep_cols = [
        "id",
        "store_id",
        "item_id",
        "cat_id",
        "uplift_gbp",
        "profit_gain_pct",
        "demand_gain_pct",
        "suspicious_uplift",
    ]
    contrib = df[keep_cols].copy()
    top10 = contrib.sort_values("uplift_gbp", ascending=False).head(10)
    bottom10 = contrib.sort_values("uplift_gbp", ascending=True).head(10)

    # Risk bands
    def _band(row) -> str:
        if int(row.get("suspicious_uplift", 0)) == 1:
            return "red"
        if (
            float(row.get("profit_gain_pct", 0.0)) > 150.0
            or float(row.get("demand_gain_pct", 0.0)) > 200.0
        ):
            return "amber"
        return "green"

    df["risk_band"] = df.apply(_band, axis=1)
    risk_bands = df.groupby("risk_band", as_index=False).agg(
        n=("id", "count"),
        base_profit_gbp=("base_profit", "sum"),
        optimised_profit_gbp=("best_profit", "sum"),
        uplift_gbp=("uplift_gbp", "sum"),
    )
    risk_bands["uplift_pct"] = (
        risk_bands["uplift_gbp"] / (risk_bands["base_profit_gbp"].abs() + 1e-9)
    ) * 100.0

    # Behaviour insights: most elastic categories/items (more negative = more sensitive)
    top_elastic_cats = (
        df.groupby("cat_id", as_index=False)["elasticity"]
        .median()
        .rename(columns={"elasticity": "median_elasticity"})
        .sort_values("median_elasticity", ascending=True)
        .head(10)
    )
    top_elastic_items = (
        df.groupby("item_id", as_index=False)["elasticity"]
        .median()
        .rename(columns={"elasticity": "median_elasticity"})
        .sort_values("median_elasticity", ascending=True)
        .head(10)
    )

    # Event vs non-event responsiveness: compare elasticity estimates and summarise by category
    evt = []
    if el_event is not None and el_nonevent is not None and len(el_event) and len(el_nonevent):
        e = el_event[["id", "elasticity"]].rename(columns={"elasticity": "elasticity_event"})
        n = el_nonevent[["id", "elasticity"]].rename(columns={"elasticity": "elasticity_non_event"})
        merged = (
            df[["id", "cat_id"]]
            .drop_duplicates()
            .merge(e, on="id", how="left")
            .merge(n, on="id", how="left")
        )
        merged = merged.dropna(subset=["elasticity_event", "elasticity_non_event"])
        if len(merged):
            per_cat = merged.groupby("cat_id", as_index=False).agg(
                n_ids=("id", "count"),
                elasticity_event=("elasticity_event", "median"),
                elasticity_non_event=("elasticity_non_event", "median"),
            )
            per_cat["delta_event_minus_non_event"] = (
                per_cat["elasticity_event"] - per_cat["elasticity_non_event"]
            )
            # most negative deltas first: "more elastic on event weeks" (more negative)
            evt = (
                per_cat.sort_values("delta_event_minus_non_event", ascending=True)
                .head(30)
                .to_dict(orient="records")
            )

    suspicious_rows = int(df["suspicious_uplift"].sum()) if "suspicious_uplift" in df.columns else 0

    elasticity_source_mix = []
    if "elasticity_source" in df.columns:
        src = (
            df.groupby("elasticity_source", as_index=False)
            .agg(n=("id", "count"), uplift_gbp=("uplift_gbp", "sum"))
            .sort_values("n", ascending=False)
        )
        src["share_rows"] = src["n"] / max(len(df), 1)
        elasticity_source_mix = src.to_dict(orient="records")

    return {
        "evidence_type": "observational_scenario_model",
        "causal_validated": False,
        "requires_randomized_validation": True,
        "base_profit_gbp": base_profit_gbp,
        "optimised_profit_gbp": optimised_profit_gbp,
        "uplift_gbp": uplift_gbp,
        "uplift_pct": uplift_pct,
        "uplift_opt_pct": uplift_pct,
        "n_actions": int(len(df)),
        "suspicious_uplift_rows": suspicious_rows,
        "guardrail_hits": suspicious_rows,
        "uplift_distribution_gbp": uplift_dist_gbp,
        "uplift_distribution_pct": uplift_dist_pct,
        "uplift_by_store_top10": by_store.sort_values("uplift_gbp", ascending=False)
        .head(10)
        .to_dict(orient="records"),
        "uplift_by_store_bottom10": by_store.sort_values("uplift_gbp", ascending=True)
        .head(10)
        .to_dict(orient="records"),
        "uplift_by_category_top10": by_cat.sort_values("uplift_gbp", ascending=False)
        .head(10)
        .to_dict(orient="records"),
        "uplift_by_category_bottom10": by_cat.sort_values("uplift_gbp", ascending=True)
        .head(10)
        .to_dict(orient="records"),
        "top10_contributors": top10.to_dict(orient="records"),
        "bottom10_contributors": bottom10.to_dict(orient="records"),
        "risk_bands": risk_bands.sort_values("risk_band").to_dict(orient="records"),
        "customer_behaviour": {
            "top_elastic_categories": top_elastic_cats.to_dict(orient="records"),
            "top_elastic_items": top_elastic_items.to_dict(orient="records"),
            "event_day_vs_non_event_responsiveness_by_cat": evt,
        },
        "elasticity_source_mix": elasticity_source_mix,
    }


def _load_unit_economics(data_dir: str, path: str) -> pd.DataFrame:
    """Load unit economics (cost OR margin) and normalise to a usable schema."""
    p = os.path.join(data_dir, path)
    df = pd.read_csv(p)

    # Identify payload column
    if "cost" in df.columns:
        mode = "cost"
        val_col = "cost"
    elif "margin" in df.columns:
        mode = "margin"
        val_col = "margin"
    else:
        raise ValueError(f"{path}: must contain either a 'cost' column or a 'margin' column")

    # Allow flexible keys: store_id/item_id/cat_id (any subset, but at least one)
    key_cols = [c for c in ["store_id", "item_id", "cat_id"] if c in df.columns]
    if not key_cols:
        raise ValueError(f"{path}: must contain at least one key column: store_id, item_id, cat_id")

    df = df[key_cols + [val_col]].copy()
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

    if mode == "margin":
        # keep margin in [0, 0.99] for sanity
        df[val_col] = df[val_col].clip(lower=0.0, upper=0.99)

    df = df.dropna(subset=[val_col]).copy()
    df["_mode"] = mode
    df = df.drop_duplicates(subset=key_cols, keep="last")
    return df


def _apply_unit_economics(
    base: pd.DataFrame,
    *,
    econ: Optional[pd.DataFrame],
    default_margin: float,
) -> Tuple[pd.Series, pd.Series]:
    """Return (cost, margin_used) aligned to base rows."""
    if econ is None or econ.empty:
        margin = pd.Series(float(default_margin), index=base.index, dtype=float)
        cost = base["price"].astype(float) * (1.0 - margin)
        return cost, margin

    # Priority: store+item > item > cat
    cost = pd.Series(np.nan, index=base.index, dtype=float)
    margin = pd.Series(np.nan, index=base.index, dtype=float)

    def merge_on(keys: List[str]) -> None:
        nonlocal cost, margin
        if not keys:
            return
        tmp = econ.loc[[c in econ.columns for c in keys].count(False) == 0]  # type: ignore[truthy-bool]
        # ^ above is defensive; we still filter below
        tmp = econ.copy()
        if not all(k in tmp.columns for k in keys):
            return
        m = base.merge(tmp, on=keys, how="left")
        if "cost" in m.columns:
            cost.fillna(m["cost"], inplace=True)
        if "margin" in m.columns:
            margin.fillna(m["margin"], inplace=True)
        # if econ table is mixed cost/margin, keep mode column and interpret:
        if "_mode" in m.columns and "cost" not in tmp.columns and "margin" not in tmp.columns:
            pass

    # Split econ into separate tables for simplicity
    econ_cost = econ[econ["_mode"] == "cost"].copy()
    econ_margin = econ[econ["_mode"] == "margin"].copy()

    if not econ_cost.empty:
        for keys in (["store_id", "item_id"], ["item_id"], ["cat_id"]):
            if all(k in econ_cost.columns for k in keys):
                m = base.merge(econ_cost[keys + ["cost"]], on=keys, how="left")
                cost = cost.fillna(m["cost"])

    if not econ_margin.empty:
        for keys in (["store_id", "item_id"], ["item_id"], ["cat_id"]):
            if all(k in econ_margin.columns for k in keys):
                m = base.merge(econ_margin[keys + ["margin"]], on=keys, how="left")
                margin = margin.fillna(m["margin"])

    # Fallback
    margin = margin.fillna(float(default_margin))
    # Convert margin -> cost when needed
    cost = cost.fillna(base["price"].astype(float) * (1.0 - margin))

    # Sanity: cost must be in (0, price]
    cost = pd.to_numeric(cost, errors="coerce")
    cost = cost.clip(lower=0.0)
    cost = np.minimum(cost, base["price"].astype(float))

    return cost.astype(float), margin.astype(float)


def run_price_optimization(cfg: PriceOptConfig) -> Dict[str, object]:
    logger = get_logger("price_opt")

    require_files(
        cfg.data_dir,
        ["sales_train_validation.csv", "calendar.csv", "sell_prices.csv", cfg.submission_path],
    )

    sub = pd.read_csv(os.path.join(cfg.data_dir, cfg.submission_path))
    fcols = [c for c in sub.columns if c.startswith("F")]
    if not fcols:
        raise ValueError("submission_path must contain F1..Fh columns")

    sub["base_demand_28d"] = sub[fcols].sum(axis=1).astype(np.float64)
    demand = sub[["id", "base_demand_28d"]]

    sales = pd.read_csv(os.path.join(cfg.data_dir, "sales_train_validation.csv"))
    if cfg.max_series and cfg.max_series > 0:
        sales = select_representative_series_subset(sales, max_series=cfg.max_series)

    cal = pd.read_csv(os.path.join(cfg.data_dir, "calendar.csv"))
    prices = pd.read_csv(os.path.join(cfg.data_dir, "sell_prices.csv"))

    # Base price = last observed train week price (proxy for "current")
    if cfg.last_train_d not in set(cal["d"].astype(str)):
        raise ValueError(f"calendar.csv missing cfg.last_train_d={cfg.last_train_d}")

    last_wk = cal.loc[cal["d"] == cfg.last_train_d, "wm_yr_wk"].iloc[0]

    base_price = (
        sales[["id", "store_id", "item_id", "cat_id"]]
        .merge(
            prices.loc[prices["wm_yr_wk"] == last_wk, ["store_id", "item_id", "sell_price"]],
            on=["store_id", "item_id"],
            how="left",
        )
        .rename(columns={"sell_price": "price"})
    )
    base_price["price"] = pd.to_numeric(base_price["price"], errors="coerce").astype(np.float64)

    # Drop rows with missing / invalid prices (avoid garbage economics)
    price_df = base_price.dropna(subset=["price"]).copy()
    price_df = price_df.loc[price_df["price"] > 0].copy()
    price_df["price"] = np.maximum(price_df["price"], float(cfg.min_price))

    # Elasticities (optionally bounded by end_d for backtest)
    logger.info("Estimating elasticities (weekly regression)...")
    el = _estimate_elasticity_weekly(
        sales_wide=sales,
        calendar=cal,
        prices=prices,
        lookback_days=cfg.lookback_days,
        elasticity_clip=cfg.elasticity_clip,
        end_d=cfg.elasticity_end_d,
    ).merge(price_df[["id", "cat_id"]], on="id", how="left")
    # Optional: elasticity split (event weeks vs non-event weeks) for responsiveness insights
    el_event = None
    el_nonevent = None
    try:
        el_event = _estimate_elasticity_weekly_segment(
            sales,
            cal,
            prices,
            lookback_days=int(cfg.lookback_days),
            elasticity_clip=tuple(cfg.elasticity_clip),
            end_d=cfg.elasticity_end_d,
            segment="event",
        )
        el_nonevent = _estimate_elasticity_weekly_segment(
            sales,
            cal,
            prices,
            lookback_days=int(cfg.lookback_days),
            elasticity_clip=tuple(cfg.elasticity_clip),
            end_d=cfg.elasticity_end_d,
            segment="non_event",
        )
    except (ValueError, KeyError, TypeError):
        # Never fail the optimiser because of a diagnostic split
        el_event, el_nonevent = None, None

    # Fallbacks: category median; then global default.
    # Keep the source so business users can see when recommendations rest on weaker assumptions.
    el["elasticity_raw"] = el["elasticity"]
    cat_med = el.groupby("cat_id")["elasticity"].median()
    global_default = -1.2
    el["elasticity_category_fallback"] = el["cat_id"].map(cat_med)
    el["elasticity_source"] = np.where(
        el["elasticity_raw"].notna(),
        "estimated",
        np.where(
            el["elasticity_category_fallback"].notna(), "category_fallback", "global_fallback"
        ),
    )
    el["elasticity"] = (
        el["elasticity_raw"]
        .fillna(el["elasticity_category_fallback"])
        .fillna(global_default)
        .astype(np.float64)
    )

    df = (
        demand.merge(
            price_df[["id", "store_id", "item_id", "cat_id", "price"]], on="id", how="inner"
        )
        .merge(el[["id", "elasticity", "elasticity_source"]], on="id", how="left")
        .dropna(subset=["base_demand_28d", "price", "elasticity"])
        .copy()
    )
    df = df.loc[df["base_demand_28d"] > 0].copy()

    # Unit economics (cost / margin)
    econ = None
    if cfg.unit_econ_path:
        econ = _load_unit_economics(cfg.data_dir, cfg.unit_econ_path)

    df["cost"], df["margin_used"] = _apply_unit_economics(
        df, econ=econ, default_margin=float(cfg.margin)
    )

    # Baseline economics
    df["base_revenue"] = df["price"] * df["base_demand_28d"]
    df["base_profit"] = (df["price"] - df["cost"]) * df["base_demand_28d"]

    # Guardrail: avoid negative margin items (common when cost is bad)
    df["bad_unit_econ"] = (df["price"] <= df["cost"]).astype(int)

    # Cap price moves (use only deltas within max_abs_price_change_pct)
    cap = float(cfg.max_abs_price_change_pct)
    grid = [float(d) for d in cfg.price_grid if abs(float(d)) <= cap + 1e-12]
    if 0.0 not in grid:
        grid.append(0.0)
    grid = sorted(set(grid))

    # Choose best per-item price from grid
    df["best_profit"] = -np.inf
    df["best_price"] = df["price"]
    df["best_demand"] = df["base_demand_28d"]
    df["best_delta"] = 0.0
    df["hit_demand_cap"] = 0

    base_p = df["price"].to_numpy(dtype=np.float64)
    base_q = df["base_demand_28d"].to_numpy(dtype=np.float64)
    e = df["elasticity"].to_numpy(dtype=np.float64)
    cost = df["cost"].to_numpy(dtype=np.float64)

    for delta in grid:
        p = np.maximum(base_p * (1.0 + delta), float(cfg.min_price))

        # Iso-elastic demand curve (elasticity is negative by construction)
        q = base_q * (p / (base_p + 1e-9)) ** e

        # Demand cap (real-world sanity)
        q_cap = base_q * float(cfg.max_demand_mult)
        hit = q > q_cap
        q = np.minimum(q, q_cap)
        q = np.maximum(q, 0.0)

        prof = (p - cost) * q

        m = prof > df["best_profit"].to_numpy(dtype=np.float64)
        if np.any(m):
            df.loc[m, "best_profit"] = prof[m]
            df.loc[m, "best_price"] = p[m]
            df.loc[m, "best_demand"] = q[m]
            df.loc[m, "best_delta"] = float(delta)
            df.loc[m, "hit_demand_cap"] = hit[m].astype(int)

    out = df[
        [
            "id",
            "store_id",
            "item_id",
            "cat_id",
            "price",
            "cost",
            "margin_used",
            "bad_unit_econ",
            "elasticity",
            "elasticity_source",
            "base_demand_28d",
            "base_revenue",
            "base_profit",
            "best_price",
            "best_demand",
            "best_profit",
            "best_delta",
            "hit_demand_cap",
        ]
    ].copy()

    # Credibility flags (suspicious uplift)
    out["profit_gain"] = out["best_profit"] - out["base_profit"]
    out["demand_gain"] = out["best_demand"] - out["base_demand_28d"]
    out["profit_gain_pct"] = (out["profit_gain"] / (out["base_profit"].abs() + 1e-9)) * 100.0
    out["demand_gain_pct"] = (out["demand_gain"] / (out["base_demand_28d"].abs() + 1e-9)) * 100.0

    suspicious = []
    suspicious.append(out["profit_gain_pct"] > float(cfg.suspicious_profit_gain_pct))
    suspicious.append(out["demand_gain_pct"] > float(cfg.suspicious_demand_gain_pct))
    suspicious.append(out["bad_unit_econ"] == 1)
    suspicious.append(out["hit_demand_cap"] == 1)
    out["suspicious_uplift"] = np.logical_or.reduce(suspicious).astype(int)

    # Helpful "why" string (human readable)
    reasons = np.where(out["bad_unit_econ"] == 1, "bad_unit_econ;", "")
    reasons = np.where(out["hit_demand_cap"] == 1, reasons + "hit_demand_cap;", reasons)
    reasons = np.where(
        out["profit_gain_pct"] > float(cfg.suspicious_profit_gain_pct),
        reasons + "huge_profit_gain;",
        reasons,
    )
    reasons = np.where(
        out["demand_gain_pct"] > float(cfg.suspicious_demand_gain_pct),
        reasons + "huge_demand_gain;",
        reasons,
    )
    out["suspicious_reason"] = reasons

    out_path = os.path.join(cfg.data_dir, cfg.out_path)
    out.to_csv(out_path, index=False)
    logger.info("Wrote: %s", out_path)

    # Optional: store a report summary for monitoring / stakeholder trust
    reports = {}
    if cfg.write_reports:
        rep_dir = os.path.join(cfg.data_dir, cfg.reports_subdir)
        os.makedirs(rep_dir, exist_ok=True)
        rep = {
            "n": int(len(out)),
            "n_suspicious": int(out["suspicious_uplift"].sum()),
            "share_suspicious": float(out["suspicious_uplift"].mean()),
            "share_hit_demand_cap": float(out["hit_demand_cap"].mean()),
            "share_bad_unit_econ": float(out["bad_unit_econ"].mean()),
        }
        rep_path = os.path.join(rep_dir, "price_opt_report.json")
        write_json(rep_path, rep)
        reports["price_opt_report"] = rep_path
        # Stakeholder summary (profits/uplift distribution/contributors/risk/behaviour)
        summary = _build_price_opt_summary(out, el_event=el_event, el_nonevent=el_nonevent)
        sum_path = os.path.join(rep_dir, "price_opt_summary.json")
        write_json(sum_path, summary)
        reports["price_opt_summary"] = sum_path
        # Backward/BI-friendly aliases
        alias_sum_path = os.path.join(rep_dir, "price_optimization_summary.json")
        write_json(alias_sum_path, summary)
        reports["price_optimization_summary"] = alias_sum_path

        # Convenience CSVs
        try:
            # full by store/category (not only top10) for BI work
            df_store = (
                out.assign(uplift_gbp=out["best_profit"] - out["base_profit"])
                .groupby("store_id", as_index=False)[["base_profit", "best_profit", "uplift_gbp"]]
                .sum()
            )
            df_store.to_csv(os.path.join(rep_dir, "uplift_by_store.csv"), index=False)
            df_cat = (
                out.assign(uplift_gbp=out["best_profit"] - out["base_profit"])
                .groupby("cat_id", as_index=False)[["base_profit", "best_profit", "uplift_gbp"]]
                .sum()
            )
            df_cat.to_csv(os.path.join(rep_dir, "uplift_by_category.csv"), index=False)
            reports["uplift_by_store_csv"] = os.path.join(rep_dir, "uplift_by_store.csv")
            reports["uplift_by_category_csv"] = os.path.join(rep_dir, "uplift_by_category.csv")
            # Alias filename expected by the dashboard / BI loader
            out.to_csv(os.path.join(rep_dir, "price_actions.csv"), index=False)
            reports["price_actions_csv"] = os.path.join(rep_dir, "price_actions.csv")
            # Minimal KPI pack so the dashboard has headline metrics without the full CLI pipeline
            kpis = {
                "base_profit": summary.get("base_profit_gbp"),
                "optimised_profit": summary.get("optimised_profit_gbp"),
                "uplift_opt_pct": summary.get("uplift_pct"),
                "n_actions": summary.get("n_actions"),
                "suspicious_uplift_rows": summary.get("suspicious_uplift_rows"),
            }
            write_json(os.path.join(rep_dir, "kpis.json"), kpis)
            reports["kpis_json"] = os.path.join(rep_dir, "kpis.json")
        except (OSError, ValueError, KeyError):
            pass

    # Always compute summary for API/pipeline consumption (even if you don't write reports)
    summary = _build_price_opt_summary(out, el_event=el_event, el_nonevent=el_nonevent)

    return {
        "opt_path": out_path,
        "n": int(len(out)),
        "elasticity_clip": tuple(cfg.elasticity_clip),
        "max_abs_price_change_pct": float(cfg.max_abs_price_change_pct),
        "max_demand_mult": float(cfg.max_demand_mult),
        "reports": reports,
        "summary": summary,
        "limitations": [
            "No cross-item cannibalisation modeled.",
            "No explicit stock / inventory constraints modeled.",
            "Uplift is observational and scenario-implied from elasticities + costs/margins; it is not causal incremental lift.",
            "Validate price actions with randomized experiments before claiming causal impact.",
        ],
    }


def backtest_price_uplift(
    *,
    data_dir: str,
    cutoffs: List[str],
    horizon: int = 28,
    max_series: int = 0,
    margin: float = 0.30,
    unit_econ_path: Optional[str] = None,
    lookback_days: int = 365,
    elasticity_clip: Tuple[float, float] = (-5.0, -0.1),
    max_abs_price_change_pct: float = 0.20,
    max_demand_mult: float = 3.0,
    price_grid: Tuple[float, ...] = (-0.20, -0.10, 0.0, 0.10, 0.20),
) -> Dict[str, Any]:
    """Approximate observational scenario backtest on multiple historical cutoffs.

    Method (approx):
      - For each cutoff d_k:
          base_p = price at wm_yr_wk of d_k
          base_q = sum of *actual* sales over next horizon days (if available)
          elasticity = estimated using history up to d_k
          run same isoelastic optimiser to compute best_profit vs base_profit
      - Aggregate model-implied uplift per store_id and cat_id so business can see where it comes from.

    This is not causal validation. It gives a directional credibility layer when
    costs are proxy / elasticity is noisy.
    """
    logger = get_logger("price_uplift_backtest")

    require_files(data_dir, ["sales_train_validation.csv", "calendar.csv", "sell_prices.csv"])

    sales = pd.read_csv(os.path.join(data_dir, "sales_train_validation.csv"))
    if max_series and max_series > 0:
        sales = select_representative_series_subset(sales, max_series=max_series)

    cal = pd.read_csv(os.path.join(data_dir, "calendar.csv"))
    prices = pd.read_csv(os.path.join(data_dir, "sell_prices.csv"))

    # Unit economics
    econ = _load_unit_economics(data_dir, unit_econ_path) if unit_econ_path else None

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    day_cols_sorted = sorted(day_cols, key=lambda x: int(x.split("_")[1]))

    def d_int(d: str) -> int:
        return int(d.split("_")[1])

    results = []
    per_store = []
    per_cat = []

    def _stability_from_results(rows: List[Dict[str, Any]]) -> Dict[str, float]:
        uplift_pct_series = pd.Series([r.get("uplift_pct") for r in rows], dtype="float64")
        uplift_gbp_series = pd.Series([r.get("uplift_gbp") for r in rows], dtype="float64")
        return {
            "mean_uplift_pct": (
                float(uplift_pct_series.mean()) if len(uplift_pct_series) else float("nan")
            ),
            "std_uplift_pct": (
                float(uplift_pct_series.std(ddof=0)) if len(uplift_pct_series) else float("nan")
            ),
            "uplift_pct_p10": (
                float(uplift_pct_series.quantile(0.10)) if len(uplift_pct_series) else float("nan")
            ),
            "uplift_pct_p50": (
                float(uplift_pct_series.quantile(0.50)) if len(uplift_pct_series) else float("nan")
            ),
            "uplift_pct_p90": (
                float(uplift_pct_series.quantile(0.90)) if len(uplift_pct_series) else float("nan")
            ),
            "mean_uplift_gbp": (
                float(uplift_gbp_series.mean()) if len(uplift_gbp_series) else float("nan")
            ),
            "std_uplift_gbp": (
                float(uplift_gbp_series.std(ddof=0)) if len(uplift_gbp_series) else float("nan")
            ),
            "uplift_gbp_p10": (
                float(uplift_gbp_series.quantile(0.10)) if len(uplift_gbp_series) else float("nan")
            ),
            "uplift_gbp_p50": (
                float(uplift_gbp_series.quantile(0.50)) if len(uplift_gbp_series) else float("nan")
            ),
            "uplift_gbp_p90": (
                float(uplift_gbp_series.quantile(0.90)) if len(uplift_gbp_series) else float("nan")
            ),
        }

    for d0 in cutoffs:
        if d0 not in set(cal["d"].astype(str)):
            logger.warning("Skipping cutoff %s (not in calendar)", d0)
            continue

        # base demand = actual next horizon sum
        i0 = d_int(d0)
        future_days = [
            f"d_{i}" for i in range(i0 + 1, i0 + 1 + int(horizon)) if f"d_{i}" in day_cols_sorted
        ]
        if not future_days:
            logger.warning("Skipping cutoff %s (no future actuals available)", d0)
            continue

        base_q = sales[future_days].sum(axis=1).astype(np.float64)
        demand = pd.DataFrame({"id": sales["id"], "base_demand_28d": base_q})

        # base price at that week
        wk = cal.loc[cal["d"] == d0, "wm_yr_wk"].iloc[0]
        base_price = (
            sales[["id", "store_id", "item_id", "cat_id"]]
            .merge(
                prices.loc[prices["wm_yr_wk"] == wk, ["store_id", "item_id", "sell_price"]],
                on=["store_id", "item_id"],
                how="left",
            )
            .rename(columns={"sell_price": "price"})
        )
        base_price["price"] = pd.to_numeric(base_price["price"], errors="coerce").astype(np.float64)
        price_df = base_price.dropna(subset=["price"]).copy()
        price_df = price_df.loc[price_df["price"] > 0].copy()

        # elasticity up to cutoff
        el = _estimate_elasticity_weekly(
            sales_wide=sales,
            calendar=cal,
            prices=prices,
            lookback_days=lookback_days,
            elasticity_clip=elasticity_clip,
            end_d=d0,
        ).merge(price_df[["id", "cat_id"]], on="id", how="left")

        cat_med = el.groupby("cat_id")["elasticity"].median()
        el["elasticity"] = (
            el["elasticity"].fillna(el["cat_id"].map(cat_med)).fillna(-1.2).astype(np.float64)
        )

        df = (
            demand.merge(
                price_df[["id", "store_id", "item_id", "cat_id", "price"]], on="id", how="inner"
            )
            .merge(el[["id", "elasticity"]], on="id", how="left")
            .dropna(subset=["base_demand_28d", "price", "elasticity"])
            .copy()
        )
        df = df.loc[df["base_demand_28d"] > 0].copy()
        df["cost"], df["margin_used"] = _apply_unit_economics(
            df, econ=econ, default_margin=float(margin)
        )
        df["base_profit"] = (df["price"] - df["cost"]) * df["base_demand_28d"]

        # grid and optimisation
        cap = float(max_abs_price_change_pct)
        grid = [float(d) for d in price_grid if abs(float(d)) <= cap + 1e-12]
        if 0.0 not in grid:
            grid.append(0.0)
        grid = sorted(set(grid))

        base_p = df["price"].to_numpy(dtype=np.float64)
        base_q = df["base_demand_28d"].to_numpy(dtype=np.float64)
        e = df["elasticity"].to_numpy(dtype=np.float64)
        cost = df["cost"].to_numpy(dtype=np.float64)

        best_profit = np.full(len(df), -np.inf, dtype=np.float64)
        best_p = base_p.copy()

        for delta in grid:
            p = np.maximum(base_p * (1.0 + delta), 0.01)
            q = base_q * (p / (base_p + 1e-9)) ** e
            q = np.minimum(q, base_q * float(max_demand_mult))
            q = np.maximum(q, 0.0)
            prof = (p - cost) * q

            m = prof > best_profit
            best_profit[m] = prof[m]
            best_p[m] = p[m]

        base_profit_gbp = float(df["base_profit"].sum())
        optimised_profit_gbp = float(best_profit.sum())
        uplift_gbp = float(optimised_profit_gbp - base_profit_gbp)
        uplift = float(optimised_profit_gbp / (base_profit_gbp + 1e-9) - 1.0) * 100.0
        results.append(
            {
                "cutoff_d": d0,
                "n": int(len(df)),
                "base_profit_gbp": base_profit_gbp,
                "optimised_profit_gbp": optimised_profit_gbp,
                "uplift_gbp": uplift_gbp,
                "uplift_pct": uplift,
            }
        )

        # store/category contributions
        tmp = df[["store_id", "cat_id", "base_profit"]].copy()
        tmp["best_profit"] = best_profit
        tmp["profit_uplift"] = tmp["best_profit"] - tmp["base_profit"]

        per_store.append(
            tmp.groupby("store_id", as_index=False)[["base_profit", "best_profit", "profit_uplift"]]
            .sum()
            .assign(cutoff_d=d0)
        )
        per_cat.append(
            tmp.groupby("cat_id", as_index=False)[["base_profit", "best_profit", "profit_uplift"]]
            .sum()
            .assign(cutoff_d=d0)
        )

    stability = _stability_from_results(results)

    out = {
        "cutoff_results": results,
        "stability": stability,
        "uplift_by_store": (
            pd.concat(per_store, ignore_index=True).to_dict(orient="records") if per_store else []
        ),
        "uplift_by_cat": (
            pd.concat(per_cat, ignore_index=True).to_dict(orient="records") if per_cat else []
        ),
    }

    # Write report artefact
    rep_dir = os.path.join(data_dir, "reports")
    os.makedirs(rep_dir, exist_ok=True)
    rep_path = os.path.join(rep_dir, "uplift_backtest.json")
    write_json(rep_path, out)
    out["report_path"] = rep_path
    logger.info("Wrote uplift backtest report: %s", rep_path)
    return out
